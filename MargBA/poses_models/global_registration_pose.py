import numpy as np
import torch
from MargBA.geometry import to_homogeneous, from_homogeneous, pad_poses
from .utils import wrap_to_parameter, r6d2mat, pose_to_r6_t

class GradientBlocker(torch.autograd.Function):
    """
    Custom autograd Function to selectively block gradients during backpropagation.
    The forward pass simply clones the input tensor.
    The backward pass applies a mask to the gradients.
    """
    @staticmethod
    def forward(ctx, x, mask):
        # Save mask for backward pass and return input clone
        ctx.save_for_backward(mask)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # Apply saved mask to gradients during backward pass
        mask, = ctx.saved_tensors
        return grad_output * mask, None  # Return masked gradients (None for mask gradient)

def gradient_block(x, mask):
    """
    Applies gradient blocking to input tensor using the provided mask.

    Args:
        x (torch.Tensor): Input tensor to apply gradient blocking
        mask (torch.Tensor): Binary mask (0=block gradient, 1=allow gradient)
                            Will be broadcast to match input tensor dimensions

    Returns:
        torch.Tensor: Output tensor with gradient control applied
    """
    # Expand mask dimensions to match input tensor
    for i in range(x.dim() - mask.dim()):
        mask = mask.unsqueeze(-1)
    return GradientBlocker.apply(x, mask)

class GlobalOptimizationPoseParameters(torch.nn.Module):
    """
    A module for managing global pose optimization parameters including:
    - Camera poses (translation and rotation)
    - Depth adjustments
    - Intrinsic camera parameters
    """
    def __init__(
            self,
            nfrm,  # Number of frames
            optimizefocal,  # Whether to optimize focal length
            sharefocal,  # Whether to share focal length across frames
            intrinsic=None,  # Initial intrinsic parameters
            gradient_mask=None,  # Mask for gradient computation
            gradient_mask_exemplar=None  # Parameters to exclude from gradient masking
    ):
        super().__init__()

        # Camera parameter optimization flags
        self.optimizefocal = optimizefocal
        self.sharefocal = sharefocal

        # Initialize gradient mask (default: all parameters enabled)
        self.gradient_mask = wrap_to_parameter(
            torch.ones(nfrm),
            requires_grad=False
        ) if gradient_mask is None else wrap_to_parameter(gradient_mask, requires_grad=False)

        # Pose parameters (translation and rotation)
        self.t = wrap_to_parameter(torch.ones([nfrm, 3, 1]), requires_grad=True)  # Translation vectors
        self.r6 = wrap_to_parameter(torch.ones([nfrm, 6]), requires_grad=True)  # Rotation in 6D representation

        # Depth adjustment parameters
        self.depth_adj = wrap_to_parameter(torch.ones(nfrm), requires_grad=True)  # Depth scaling factor
        self.depth_bias = wrap_to_parameter(torch.zeros(nfrm), requires_grad=True)  # Depth bias term

        # Focal length adjustment parameters
        self.depth_focalx = wrap_to_parameter(torch.zeros(nfrm), requires_grad=True)  # Focal length x adjustment
        self.depth_focaly = wrap_to_parameter(torch.zeros(nfrm), requires_grad=True)  # Focal length y adjustment

        self.nfrm = nfrm  # Store number of frames

        # Initialize all poses to identity
        for i in range(len(self.t)):
            self.update_pose_adjustment(
                nodeid=i,
                newpose=torch.eye(4),
                newadjustment=1.0,
                newbias=0.0
            )

        # Initialize intrinsic parameters
        self.init_intrinsic(intrinsic)

        # Validate gradient mask exemplar parameters
        if gradient_mask_exemplar is not None:
            valid_params = ["f", "t", "r6", "depth_adj", "depth_bias", "depth_focalx", "depth_focaly"]
            for x in gradient_mask_exemplar:
                assert x in valid_params, f"Invalid parameter {x} in gradient_mask_exemplar"
        else:
            gradient_mask_exemplar = []

        self.gradient_mask_exemplar = gradient_mask_exemplar

    def register_idx(self, src_idx1, dst_idx2):
        # src and dst idx
        self.src_idx1, self.dst_idx2 = src_idx1, dst_idx2

    def init_intrinsic(self, intrinsic):
        if self.sharefocal:
            intrinsic = np.zeros((4, 4)) if intrinsic is None else intrinsic
            assert len(intrinsic.shape) == 2
            intrinsic = intrinsic.astype(np.float32)
            fx, fy, bx, by = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
            self.f = wrap_to_parameter(torch.Tensor([(fx + fy) / 2]), requires_grad=self.optimizefocal)
            self.bx = wrap_to_parameter(torch.Tensor([bx]), requires_grad=False)
            self.by = wrap_to_parameter(torch.Tensor([by]), requires_grad=False)
        else:
            intrinsic = np.zeros((self.nfrm, 4, 4)) if intrinsic is None else intrinsic
            assert len(intrinsic.shape) == 3
            intrinsic = intrinsic.astype(np.float32)
            fx, fy, bx, by = intrinsic[:, 0, 0], intrinsic[:, 1, 1], intrinsic[:, 0, 2], intrinsic[:, 1, 2]
            self.f = wrap_to_parameter(torch.Tensor((fx + fy) / 2), requires_grad=self.optimizefocal)
            self.bx = wrap_to_parameter(torch.Tensor(bx), requires_grad=False)
            self.by = wrap_to_parameter(torch.Tensor(by), requires_grad=False)

    def get_intrinsic(self):
        """
        Constructs and returns the intrinsic camera matrix for all frames.

        Returns:
            torch.Tensor: A tensor of shape (n_frames, 3, 3) containing the intrinsic matrices.
                         The gradient flow is controlled by gradient_mask_exemplar settings.
        """
        device = self.f.data.device  # Get device from focal length parameter
        intrinsic = torch.zeros(len(self.t), 3, 3).to(device)  # Initialize empty intrinsic matrices

        # Set matrix components:
        intrinsic[:, 0, 0] = self.f  # Focal length x
        intrinsic[:, 1, 1] = self.f  # Focal length y
        intrinsic[:, 0, 2] = self.bx  # Principal point x
        intrinsic[:, 1, 2] = self.by  # Principal point y
        intrinsic[:, 2, 2] = 1.0     # Homogeneous coordinate scaling

        # Apply gradient blocking if 'f' is not in exemplar list
        return intrinsic if "f" in self.gradient_mask_exemplar else gradient_block(intrinsic, self.gradient_mask)

    def get_intrinsic_pairwise(self):
        intrinsic = self.get_intrinsic()
        intrinsics_pairwise_src = intrinsic[self.src_idx1, :, :].contiguous()
        intrinsics_pairwise_dst = intrinsic[self.dst_idx2, :, :].contiguous()
        return intrinsics_pairwise_src, intrinsics_pairwise_dst

    def get_params(self, name):
        if name == "extrinsic":
            return [self.t, self.r6, self.depth_adj, self.depth_bias, self.depth_focalx, self.depth_focaly]
        elif name == "intrinsic":
            return [self.f]
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def update_pose_adjustment(self, nodeid, newpose, newadjustment, newbias=None):
        r6, t = pose_to_r6_t(newpose.view([1, 4, 4]))
        self.depth_adj.data[nodeid] = torch.Tensor([newadjustment])
        self.r6.data[nodeid], self.t.data[nodeid] = r6.squeeze(0), t.squeeze(0)
        if newbias is not None:
            self.depth_bias.data[nodeid] = newbias

    def get_pose_w2c(self):
        """
        Compute world-to-camera pose matrices from rotation and translation parameters.

        Returns:
            torch.Tensor: Pose matrices of shape (n_frames, 4, 4) where each matrix transforms
                         world coordinates to camera coordinates.
                         Gradient flow is controlled by gradient_mask_exemplar settings.
        """
        # Convert 6D rotation representation to rotation matrices
        R = r6d2mat(self.r6)
        # Get translation vectors
        t = self.t

        # Combine rotation and translation into pose matrices
        pose = pad_poses(torch.concat([R, t], dim=-1))

        # Apply gradient blocking if neither 'r6' nor 't' are in exemplar list
        if "r6" in self.gradient_mask_exemplar or "t" in self.gradient_mask_exemplar:
            pose = pose  # No gradient blocking needed
        else:
            pose = gradient_block(pose, self.gradient_mask)

        return pose

    def get_depth_adjustment(self):
        """
        Get depth adjustment parameters with gradient control based on gradient_mask_exemplar.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                depth_adj, depth_bias, depth_focalx, depth_focaly
        """
        depth_adj, depth_bias = self.depth_adj, self.depth_bias
        depth_focalx, depth_focaly = self.depth_focalx, self.depth_focaly

        # Apply gradient blocking if none of the parameters are in exemplar list
        if not any(
                param in self.gradient_mask_exemplar
                for param in ["depth_adj", "depth_bias", "depth_focalx", "depth_focaly"]
        ):
            depth_adj = gradient_block(depth_adj, self.gradient_mask)
            depth_bias = gradient_block(depth_bias, self.gradient_mask)
            depth_focalx = gradient_block(depth_focalx, self.gradient_mask)
            depth_focaly = gradient_block(depth_focaly, self.gradient_mask)

        return depth_adj, depth_bias, depth_focalx, depth_focaly

    def get_pose_pairwise(self):
        odometry = self.get_pose_w2c()
        pose_w2c_srcidx1 = odometry[self.src_idx1, :, :]
        pose_w2c_dstidx2 = odometry[self.dst_idx2, :, :]
        return pose_w2c_dstidx2 @ torch.inverse(pose_w2c_srcidx1)

    def adjust_intrinsic(self, intrinsic_src, intrinsic_dst):
        _, _, depth_focalx, depth_focaly = self.get_depth_adjustment()
        fx_src, fy_src = intrinsic_src[:, 0, 0], intrinsic_src[:, 1, 1]
        fx_dst, fy_dst = intrinsic_dst[:, 0, 0], intrinsic_dst[:, 1, 1]
        
        delta_fx_src, delta_fx_dst = depth_focalx[self.src_idx1], depth_focalx[self.dst_idx2]
        delta_fy_src, delta_fy_dst = depth_focaly[self.src_idx1], depth_focaly[self.dst_idx2]

        fx_src_adj, fx_dst_adj = torch.exp(torch.log(fx_src) + delta_fx_src), torch.exp(torch.log(fx_dst) + delta_fx_dst)
        fy_src_adj, fy_dst_adj = torch.exp(torch.log(fy_src) + delta_fy_src), torch.exp(torch.log(fy_dst) + delta_fy_dst)

        intrinsic_src_new, intrinsic_dst_new = torch.clone(intrinsic_src), torch.clone(intrinsic_dst)
        intrinsic_src_new[:, 0, 0] = fx_src_adj
        intrinsic_src_new[:, 1, 1] = fy_src_adj
        intrinsic_dst_new[:, 0, 0] = fx_dst_adj
        intrinsic_dst_new[:, 1, 1] = fy_dst_adj
        return intrinsic_src_new, intrinsic_dst_new
        
    def depth2pts(self, pts_src, depth_src, intrinsic_src, pts_dst, depth_dst, intrinsic_dst):
        depth_adj, depth_bias, _, _ = self.get_depth_adjustment()
        npair, npts, _ = pts_src.shape
        adjustments_src, adjustments_dst = depth_adj[self.src_idx1].view([npair, 1, 1]), depth_adj[self.dst_idx2].view([npair, 1, 1])
        bias_src, bias_dst = depth_bias[self.src_idx1].view([npair, 1, 1]), depth_bias[self.dst_idx2].view([npair, 1, 1])

        intrinsic_src, intrinsic_dst = self.adjust_intrinsic(intrinsic_src, intrinsic_dst)
        pts_3d_src = pts_src * (depth_src.view([npair, npts, 1]) * adjustments_src + bias_src)
        pts_3d_src = pts_3d_src @ torch.transpose(torch.inverse(intrinsic_src), dim0=1, dim1=2)
        pts_3d_dst = pts_dst * (depth_dst.view([npair, npts, 1]) * adjustments_dst + bias_dst)
        pts_3d_dst = pts_3d_dst @ torch.transpose(torch.inverse(intrinsic_dst), dim0=1, dim1=2)
        return pts_3d_src, pts_3d_dst

    def forward(self, pts_src, pts_dst, depth_src, depth_dst):
        residual2D, weights_2D = self.compute_residual(
            pts_src,
            pts_dst,
            depth_src,
            depth_dst
        )
        return residual2D, weights_2D

    def compute_residual(self, pts_src, pts_dst, depth_src, depth_dst):
        # acquire pose
        pose = self.get_pose_pairwise()
        # map depth to 3D points
        intrinsic_src, intrinsic_dst = self.get_intrinsic_pairwise()
        pts_3d_src, pts_3d_dst_ref = self.depth2pts(
            pts_src,
            depth_src,
            intrinsic_src,
            pts_dst,
            depth_dst,
            intrinsic_dst,
        )

        # re-projection error
        pts_3d_src = to_homogeneous(pts_3d_src)
        pts_3d_dst = pts_3d_src @ torch.transpose(pose, dim0=1, dim1=2)
        pts_3d_dst = pts_3d_dst[:, :, 0:3]
        pts_3d_dst_projected = pts_3d_dst @ torch.transpose(intrinsic_dst, dim0=1, dim1=2)
        pts_2d_dst = from_homogeneous(pts_3d_dst_projected)
        residual2D = torch.sqrt(torch.sum((pts_2d_dst - pts_dst[:, :, 0:2]) ** 2, dim=-1) + 1e-10)
        weights_2D = (pts_3d_dst_projected[:, :, 2] > 0).float()
        return residual2D, weights_2D