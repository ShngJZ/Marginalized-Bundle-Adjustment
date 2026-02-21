import numpy as np
import torch
import cupy as cp

weighted_histogram_kernel = cp.RawKernel(r'''
extern "C" __global__
void weighted_histogram_kernel(
    float* residuals, 
    float* weights, 
    float* hist,
    int* src_idx1,
    int* dst_idx2,
    float minval,
    float maxval,
    float bins,
    int npair,
    int npts
    ) {
    int threadid = blockDim.x * blockIdx.x + threadIdx.x;
    int hist_bin_id;
    
    int npairid;
    int nptsid;
    int hist_id;

    int binsint = (int) bins;
    float interval = (maxval - minval) / bins;

    for (; threadid < npair * npts; threadid += blockDim.x * gridDim.x) {

        npairid = threadid / npts;
        nptsid = threadid - npairid * npts;
        hist_bin_id = (residuals[threadid] - minval) / interval;
        
        if (hist_bin_id < binsint){
            hist_id = bins * src_idx1[npairid] + hist_bin_id;
            atomicAdd(&hist[hist_id], weights[threadid]);
            
            hist_id = bins * dst_idx2[npairid] + hist_bin_id;
            atomicAdd(&hist[hist_id], weights[threadid]);
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
    int* src_or_dst_index,
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
    
    int npairid;
    int nptsid;
    int cdf_id;

    int binsint = (int) bins;
    float interval = (maxval - minval) / bins;

    for (; threadid < npair * npts; threadid += blockDim.x * gridDim.x) {

        npairid = threadid / npts;
        nptsid = threadid - npairid * npts;

        hist_bin_id = (residuals[threadid] - minval) / interval + 0.5;
        if ((hist_bin_id < binsint) && (weights[threadid] > 0.0)){
            cdf_id = binsint * src_or_dst_index[npairid] + hist_bin_id;
            residuals_cdf[threadid] = cdf[cdf_id];
            residuals_cdf_grad[threadid] = grad_cdf[cdf_id] * weights[threadid];
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
        # residuals_cdf_grad[torch.isnan(residuals_cdf_grad)] = 0.0
        # grad_out[torch.isnan(grad_out)] = 0.0
        return residuals_cdf_grad * grad_out, None, None

class CDFLossIndexCupy(torch.nn.Module):
    def __init__(
            self,
            min,
            max,
            bins,
            src_idx1,
            dst_idx2,
            gradient_smooth=1,
            nnodes=None,
    ):
        """
        The function computes the CDF loss for per subgraph defined by one frame and all its connected neighbouring frames
        min: min value in histogram
        max: max value in histogram
        gradient_smooth: A smooth bandwidth to compute gradient
        """
        super().__init__()
        self.min, self.max, self.bins = min, max, bins
        self.gradient_smooth, self.bins = gradient_smooth, bins

        self.grad_conv = torch.nn.Conv1d(1, 1, 2 * gradient_smooth + 1, padding=gradient_smooth, bias=False, padding_mode="replicate")
        for i in range(2 * gradient_smooth + 1):
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

        self.src_idx1, self.dst_idx2 = src_idx1, dst_idx2

        if nnodes is None:
            all_nodes = list()
            for idx in self.src_idx1:
                all_nodes.append(int(idx.item()))
            self.nnodes = len(set(all_nodes))
        else:
            self.nnodes = nnodes

    def forward(self, residuals, weights):
        """
        inputs of non-zero weights and in-range are assigned with corresponding cdf value
        inputs reset is assigned a negative one
        inputs of non-zero weights and in-range are assigned with weights * grad, otherwise 0
        """
        pdf, cdf, grad_cdf = self.compute_weighted_pdf_cdf(
            residuals, weights
        )
        residuals_cdf_srcidx1, residuals_cdf_grad_srcidx1, residuals_cdf_dstidx2, residuals_cdf_grad_dstidx2 = \
            self.compute_weighted_cdf_forward_backward(
                residuals, weights, cdf, grad_cdf
            )
        residuals_cdf_out_srcidx1 = CDFLossTorchWrapper.apply(residuals, residuals_cdf_srcidx1, residuals_cdf_grad_srcidx1)
        residuals_cdf_out_dstidx2 = CDFLossTorchWrapper.apply(residuals, residuals_cdf_dstidx2, residuals_cdf_grad_dstidx2)
        return residuals_cdf_out_srcidx1, residuals_cdf_out_dstidx2

    @torch.no_grad()
    def compute_weighted_cdf_forward_backward(self, residuals, weights, cdf, grad_cdf):
        """
        :param residuals: npair x npts
        :param weights: npair x npts
        :param cdf: nfrm x nbins
        :param grad_cdf: nfrm x nbins
        :return:
        """

        """
        # unit test
        interval = (self.max - self.min) / self.bins
        rnd1, rnd2 = np.random.randint(npair), np.random.randint(npts)
        rnd_residual = residuals[rnd1, rnd2]
        rnd_flag = np.random.randint(2)
        if rnd_flag == 0:
            idx = self.src_idx1[rnd1].item()
            residuals_cdf = residuals_cdf_srcidx1
            residuals_cdf_grad = residuals_cdf_grad_srcidx1
        else:
            idx = self.dst_idx2[rnd1].item()
            residuals_cdf = residuals_cdf_dstidx2
            residuals_cdf_grad = residuals_cdf_grad_dstidx2

        rnd_cdf_grad_id = int(np.round((rnd_residual - self.min).item() / interval))
        if rnd_cdf_grad_id < self.bins and weights[rnd1, rnd2] > 0.0:
            grad_rnd = grad_cdf[idx, rnd_cdf_grad_id]
            grad_rnd_ref = residuals_cdf_grad[rnd1, rnd2]

            cdf_rnd = cdf[idx, rnd_cdf_grad_id]
            cdf_rnd_ref = residuals_cdf[rnd1, rnd2]
        """
        npair, npts = residuals.shape
        device = residuals.device
        residuals_cp, weights_cp = tensor2cparray(residuals), tensor2cparray(weights)
        cdf_cp, grad_cdf_cp = tensor2cparray(cdf), tensor2cparray(grad_cdf)

        # compute the cdf value and gradient on the source indices
        src_or_dst_index_srcidx1 = tensor2cparray(self.src_idx1)
        residuals_cdf_srcidx1, residuals_cdf_grad_srcidx1 = tensor2cparray(torch.zeros_like(residuals)), tensor2cparray(torch.zeros_like(residuals))
        forward_backward_cdf_kernel(
            (8,), (1024,),
            (
                residuals_cp,
                weights_cp,
                cdf_cp,
                grad_cdf_cp,
                src_or_dst_index_srcidx1,
                residuals_cdf_srcidx1,
                residuals_cdf_grad_srcidx1,
                cp.float32(self.min),
                cp.float32(self.max),
                cp.float32(self.bins),
                cp.int32(npair),
                cp.int32(npts)
            )
        )
        residuals_cdf_srcidx1, residuals_cdf_grad_srcidx1 = torch.as_tensor(residuals_cdf_srcidx1, device=device), torch.as_tensor(residuals_cdf_grad_srcidx1, device=device)

        # compute the cdf value and gradient on the target indices
        src_or_dst_index_dstidx2 = tensor2cparray(self.dst_idx2)
        residuals_cdf_dstidx2, residuals_cdf_grad_dstidx2 = tensor2cparray(torch.zeros_like(residuals)), tensor2cparray(torch.zeros_like(residuals))
        forward_backward_cdf_kernel(
            (8,), (1024,),
            (
                residuals_cp,
                weights_cp,
                cdf_cp,
                grad_cdf_cp,
                src_or_dst_index_dstidx2,
                residuals_cdf_dstidx2,
                residuals_cdf_grad_dstidx2,
                cp.float32(self.min),
                cp.float32(self.max),
                cp.float32(self.bins),
                cp.int32(npair),
                cp.int32(npts)
            )
        )
        residuals_cdf_dstidx2, residuals_cdf_grad_dstidx2 = torch.as_tensor(residuals_cdf_dstidx2, device=device), torch.as_tensor(residuals_cdf_grad_dstidx2, device=device)
        return residuals_cdf_srcidx1, residuals_cdf_grad_srcidx1, residuals_cdf_dstidx2, residuals_cdf_grad_dstidx2

    @torch.no_grad()
    def compute_weighted_pdf_cdf(self, residuals, weights):
        nnodes, device = self.nnodes, residuals.device
        residuals_cp, weights_cp = tensor2cparray(residuals), tensor2cparray(weights)
        hist = tensor2cparray(torch.zeros([nnodes, self.bins], device=device))

        src_idx1, dst_idx2 = tensor2cparray(self.src_idx1.int()), tensor2cparray(self.dst_idx2.int())

        npair, npts = residuals.shape
        weighted_histogram_kernel(
            (8,), (1024,),
            (
                residuals_cp,
                weights_cp,
                hist,
                src_idx1,
                dst_idx2,
                cp.float32(self.min),
                cp.float32(self.max),
                cp.float32(self.bins),
                cp.int32(npair),
                cp.int32(npts)
            )
        )
        hist = torch.as_tensor(hist, device=device)
        pdf = hist / (torch.sum(hist, dim=1, keepdim=True) + 1e-10)
        cdf = torch.cumsum(pdf, dim=1)
        grad_cdf = self.grad_conv(cdf.view(nnodes, 1, self.bins)).view(nnodes, self.bins)

        """
        # unit test
        interval = (self.max - self.min) / self.bins
        rndindex = np.random.randint(0, len(self.all_nodes))
        rndid = np.random.randint(self.bins)
        minv, maxv = self.min + interval * rndid, self.min + interval * (rndid + 1)
        selector = ((self.src_idx1 == rndindex) + (self.dst_idx2 == rndindex)) > 0
        selector = selector.view(npair, 1) * (weights == 1)
        hist_rnd = torch.sum((residuals[selector] > minv) * (residuals[selector] < maxv))
        hist_rnd_ref = hist[rndindex, rndid]
        grad = (cdf[rndindex, rndid+1 : rndid+self.gradient_smooth+1].sum() - cdf[rndindex, rndid - self.gradient_smooth : rndid].sum()) / self.gradient_smooth / 2 / interval
        grad_ref = grad_cdf[rndindex, rndid]
        """
        return pdf, cdf, grad_cdf


class CDFLossBatchCupy(torch.nn.Module):
    def __init__(
            self,
            min,
            max,
            bins,
            gradient_smooth=1
    ):
        """
        The function computes the CDF loss for per subgraph defined by one frame and all its connected neighbouring frames
        min: min value in histogram
        max: max value in histogram
        gradient_smooth: A smooth bandwidth to compute gradient
        """
        super().__init__()
        self.min, self.max, self.bins = min, max, bins
        self.gradient_smooth, self.bins = gradient_smooth, bins

        self.grad_conv = torch.nn.Conv1d(1, 1, 2 * gradient_smooth + 1, padding=gradient_smooth, bias=False, padding_mode="replicate")
        for i in range(2 * gradient_smooth + 1):
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
        inputs of non-zero weights and in-range are assigned with weights * grad, otherwise 0
        """
        residuals, weights = residuals.contiguous(), weights.contiguous()
        pdf, cdf, grad_cdf = self.compute_weighted_pdf_cdf(
            residuals, weights
        )
        residuals_cdf, residuals_cdf_grad = \
            self.compute_weighted_cdf_forward_backward(
                residuals, weights, cdf, grad_cdf
            )
        residuals_cdf_out = CDFLossTorchWrapper.apply(residuals, residuals_cdf, residuals_cdf_grad)
        return residuals_cdf_out

    @torch.no_grad()
    def compute_weighted_cdf_forward_backward(self, residuals, weights, cdf, grad_cdf):
        """
        :param residuals: npair x npts
        :param weights: npair x npts
        :param cdf: nfrm x nbins
        :param grad_cdf: nfrm x nbins
        :return:
        """
        nnodes = len(residuals)
        npair, npts = residuals.shape
        device = residuals.device
        residuals_cp, weights_cp = tensor2cparray(residuals), tensor2cparray(weights)
        cdf_cp, grad_cdf_cp = tensor2cparray(cdf), tensor2cparray(grad_cdf)

        # compute the cdf value and gradient on the source indices
        src_or_dst_index = torch.arange(0, nnodes).to(device).int()
        src_or_dst_index = tensor2cparray(src_or_dst_index)
        residuals_cdf, residuals_cdf_grad = tensor2cparray(torch.zeros_like(residuals)), tensor2cparray(torch.zeros_like(residuals))
        forward_backward_cdf_kernel(
            (8,), (1024,),
            (
                residuals_cp,
                weights_cp,
                cdf_cp,
                grad_cdf_cp,
                src_or_dst_index,
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
        return residuals_cdf, residuals_cdf_grad

    @torch.no_grad()
    def compute_weighted_pdf_cdf(self, residuals, weights):
        nnodes, device = len(residuals), residuals.device
        residuals_cp, weights_cp = tensor2cparray(residuals), tensor2cparray(weights)
        hist = tensor2cparray(torch.zeros([nnodes, self.bins], device=device))

        src_idx1, dst_idx2 = torch.arange(0, nnodes).to(device).int(), torch.arange(0, nnodes).to(device).int()
        src_idx1, dst_idx2 = tensor2cparray(src_idx1), tensor2cparray(dst_idx2)

        npair, npts = residuals.shape
        weighted_histogram_kernel(
            (8,), (1024,),
            (
                residuals_cp,
                weights_cp,
                hist,
                src_idx1,
                dst_idx2,
                cp.float32(self.min),
                cp.float32(self.max),
                cp.float32(self.bins),
                cp.int32(npair),
                cp.int32(npts)
            )
        )
        hist = torch.as_tensor(hist, device=device)
        pdf = hist / (torch.sum(hist, dim=1, keepdim=True) + 1e-10)
        cdf = torch.cumsum(pdf, dim=1)
        grad_cdf = self.grad_conv(cdf.view(nnodes, 1, self.bins)).view(nnodes, self.bins)
        return pdf, cdf, grad_cdf
