import numpy as np
import torch
import cupy as cp

weighted_histogram_kernel = cp.RawKernel(r'''
extern "C" __global__
void weighted_histogram_kernel(
    float* residuals, 
    float* weights, 
    float* hist,
    float minval,
    float maxval,
    float bins,
    int npair,
    int npts
    ) {
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    int hist_bin_id;
    
    int binsint = (int) bins;
    float interval = (maxval - minval) / bins;

    for (; threadid < npair * npts; threadid += blockDim.x * gridDim.x) {

        hist_bin_id = (residuals[threadid] - minval) / interval;
        if (hist_bin_id < binsint){
            atomicAdd(&hist[hist_bin_id], weights[threadid]);
        }

    }
}
''', 'weighted_histogram_kernel')

forward_backward_cdf_kernel = cp.RawKernel(r'''
extern "C" __global__
void forward_backward_cdf_kernel(
    float* residuals, 
    float* weights, 
    float* cdf,
    float* grad_cdf,
    float* residuals_cdf,
    float* residuals_cdf_grad,
    float minval,
    float maxval,
    float bins,
    int npair,
    int npts
    ) {
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    int hist_bin_id;

    int binsint = (int) bins;
    float interval = (maxval - minval) / bins;

    for (; threadid < npair * npts; threadid += blockDim.x * gridDim.x) {

        hist_bin_id = (residuals[threadid] - minval) / interval + 0.5;
        if ((hist_bin_id < binsint) && (weights[threadid] > 0.0)){
            residuals_cdf[threadid] = cdf[hist_bin_id];
            residuals_cdf_grad[threadid] = grad_cdf[hist_bin_id] * weights[threadid];
        }
        else{
            residuals_cdf[threadid] = 2.0;
            residuals_cdf_grad[threadid] = 0.0;
        }

    }
}
''', 'forward_backward_cdf_kernel')

def tensor2cparray(input_tensor):
    return cp.asarray(input_tensor.detach().contiguous())

class CDFLossTorchWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residuals, residuals_cdf, residuals_cdf_grad):
        ctx.save_for_backward(residuals_cdf_grad)
        return residuals_cdf

    @staticmethod
    def backward(ctx, grad_out):
        residuals_cdf_grad, = ctx.saved_tensors
        return residuals_cdf_grad * grad_out, None, None

class CDFLossCupy(torch.nn.Module):
    def __init__(
            self,
            min,
            max,
            bins,
            gradient_smooth=1
    ):
        """
        min: min value in histogram
        max: max value in histogram
        gradient_smooth: A smooth bandwidth to compute gradient
        """
        super().__init__()
        self.min, self.max, self.bins = min, max, bins
        self.gradient_smooth, self.bins = gradient_smooth, bins

        self.grad_conv = torch.nn.Conv1d(1, 1, 2*gradient_smooth+1, padding=gradient_smooth, bias=False, padding_mode="replicate")
        for i in range(2*gradient_smooth+1):
            if i < gradient_smooth:
                self.grad_conv.weight.data[0, 0, i] = -1.0
            elif i > gradient_smooth:
                self.grad_conv.weight.data[0, 0, i] = 1.0
            else:
                self.grad_conv.weight.data[0, 0, i] = 0.0
        delta = (max - min) / bins * gradient_smooth * 2
        self.grad_conv.weight.data = self.grad_conv.weight.data / delta
        self.grad_conv.weight.requires_grad = False

        self.bin_edges = torch.linspace(self.min, self.max, self.bins + 1)
    
    def forward(self, residuals, weights):
        """
        inputs of non-zero weights and in-range are assigned with corresponding cdf value
        inputs reset is assigned a negative one
        inputs of non-zero weights and in-range are assigned with weights * gra, otherwise 0
        """
        residuals, weights = residuals.contiguous(), weights.contiguous()
        pdf, cdf, grad_cdf = self.compute_weighted_pdf_cdf(
            residuals, weights
        )
        residuals_cdf, residuals_cdf_grad = self.compute_weighted_cdf_forward_backward(
            residuals, weights, cdf, grad_cdf
        )
        misc = {
            "prb": ((self.bin_edges[:-1] + self.bin_edges[1:]) / 2).cpu().numpy(),
            "cdf": cdf.cpu().numpy(),
            "cdf grad": grad_cdf.cpu().numpy()
        }

        residuals_cdf_out = CDFLossTorchWrapper.apply(residuals, residuals_cdf, residuals_cdf_grad)
        # residuals_cdf_out_mean = residuals_cdf_out[residuals_cdf_out <= 1.0].mean()
        return residuals_cdf_out, misc

    @torch.no_grad()
    def compute_weighted_cdf_forward_backward(self, residuals, weights, cdf, grad_cdf):
        device = residuals.device
        residuals_cp, weights_cp = tensor2cparray(residuals), tensor2cparray(weights)
        cdf_cp, grad_cdf_cp = tensor2cparray(cdf), tensor2cparray(grad_cdf)
        residuals_cdf, residuals_cdf_grad = tensor2cparray(torch.zeros_like(residuals)), tensor2cparray(torch.zeros_like(residuals))

        npair, npts = residuals.shape
        forward_backward_cdf_kernel(
            (8,), (1024,),
            (
                residuals_cp,
                weights_cp,
                cdf_cp,
                grad_cdf_cp,
                residuals_cdf,
                residuals_cdf_grad,
                cp.float32(self.min),
                cp.float32(self.max),
                cp.float32(self.bins),
                cp.int32(npair),
                cp.int32(npts)
            )
        )
        residuals_cdf, residuals_cdf_grad = torch.as_tensor(residuals_cdf, device=device), torch.as_tensor(residuals_cdf_grad, device=device)

        """
        # unit test
        interval = (self.max - self.min) / self.bins
        rnd1, rnd2 = np.random.randint(npair), np.random.randint(npts)
        rnd_residual = residuals[rnd1, rnd2]
        rnd_cdf_grad_id = int(np.round((rnd_residual - self.min).item() / interval))
        if rnd_cdf_grad_id < self.bins and weights[rnd1, rnd2] > 0.0:
            grad_rnd = grad_cdf[rnd_cdf_grad_id]
            grad_rnd_ref = residuals_cdf_grad[rnd1, rnd2]
            
            cdf_rnd = cdf[rnd_cdf_grad_id]
            cdf_rnd_ref = residuals_cdf[rnd1, rnd2]
        """
        return residuals_cdf, residuals_cdf_grad

    @torch.no_grad()
    def compute_weighted_pdf_cdf(self, residuals, weights):
        device = residuals.device
        residuals_cp, weights_cp = tensor2cparray(residuals), tensor2cparray(weights)
        hist = tensor2cparray(torch.zeros(self.bins, device=device))

        npair, npts = residuals.shape
        weighted_histogram_kernel(
            (8,), (1024,),
            (
                residuals_cp,
                weights_cp,
                hist,
                cp.float32(self.min),
                cp.float32(self.max),
                cp.float32(self.bins),
                cp.int32(npair),
                cp.int32(npts)
            )
        )
        hist = torch.as_tensor(hist, device=device)
        pdf = hist / torch.sum(hist)
        cdf = torch.cumsum(pdf, dim=0)
        grad_cdf = self.grad_conv(cdf.view(1, 1, self.bins)).view(self.bins)

        """
        # unit test
        interval = (self.max - self.min) / self.bins
        bin_edges = self.get_bin_edges()
        rndid = np.random.randint(self.bins)
        minv, maxv = self.min + interval * rndid, self.min + interval * (rndid + 1)
        minv_bin_edge, maxv_bin_edge = bin_edges[rndid], bin_edges[rndid+1]
        hist_rnd = torch.sum((residuals > minv) * (residuals < maxv))
        hist_rnd_ref = hist[rndid]
        grad = (cdf[rndid + self.gradient_smooth] - cdf[rndid - self.gradient_smooth]) / self.gradient_smooth / 2 / interval
        grad_ref = grad_cdf[rndid]
        """
        return pdf, cdf, grad_cdf