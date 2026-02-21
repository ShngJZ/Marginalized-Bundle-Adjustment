import torch
from typing import Optional
from MargBA.geometry import to_homogeneous, from_homogeneous

class LocalOptimizationPoseParameters(torch.nn.Module):
    def __init__(
            self
    ):
        # World to Camera Pose
        super().__init__()

        # Initialize empty lists for storing pose and point data
        # For new2old direction
        self.old_w2c_poses_new2old = []
        self.pts_src_new2old = []
        self.pts_dst_new2old = []
        self.depth_src_new2old = []
        self.intrinsic_new2old = []

        # For old2new direction
        self.old_w2c_poses_old2new = []
        self.pts_src_old2new = []
        self.pts_dst_old2new = []
        self.depth_src_old2new = []
        self.intrinsic_old2new = []

    def register_old2new(
            self,
            old_pose_w2c,
            pts_src,
            pts_dst,
            depth_src,
            intrinsic
    ):
        self.old_w2c_poses_old2new.append(old_pose_w2c)
        self.pts_src_old2new.append(pts_src)
        self.pts_dst_old2new.append(pts_dst)
        self.depth_src_old2new.append(depth_src)
        self.intrinsic_old2new.append(intrinsic)

    def register_new2old(
            self,
            old_pose_w2c,
            pts_src,
            pts_dst,
            depth_src,
            intrinsic
    ):
        self.old_w2c_poses_new2old.append(old_pose_w2c)
        self.pts_src_new2old.append(pts_src)
        self.pts_dst_new2old.append(pts_dst)
        self.depth_src_new2old.append(depth_src)
        self.intrinsic_new2old.append(intrinsic)

    def organize(self):
        """
        Organizes and stacks all registered pose and point data into tensors.
        Also computes 3D points in camera coordinates for both new2old and old2new directions.

        Returns:
            bool: True if data was successfully organized, False if no data was registered
        """
        if not self.old_w2c_poses_new2old or not self.old_w2c_poses_old2new:
            print("Warning: Empty pose list detected in organize() - no data was registered")
            return False

        # Stack all pose data into tensors
        self.old_w2c_poses_new2old = torch.stack(self.old_w2c_poses_new2old, dim=0)
        self.old_w2c_poses_old2new = torch.stack(self.old_w2c_poses_old2new, dim=0)

        # Stack all point data for new2old direction
        self.pts_src_new2old = torch.stack(self.pts_src_new2old, dim=0)
        self.pts_dst_new2old = torch.stack(self.pts_dst_new2old, dim=0)
        self.depth_src_new2old = torch.stack(self.depth_src_new2old, dim=0)
        self.intrinsic_new2old = torch.stack(self.intrinsic_new2old, dim=0)

        # Stack all point data for old2new direction
        self.pts_src_old2new = torch.stack(self.pts_src_old2new, dim=0)
        self.pts_dst_old2new = torch.stack(self.pts_dst_old2new, dim=0)
        self.depth_src_old2new = torch.stack(self.depth_src_old2new, dim=0)
        self.intrinsic_old2new = torch.stack(self.intrinsic_old2new, dim=0)

        # Compute 3D points in camera coordinates
        self.pts_3d_src_camera_new2old = self.map23D(
            self.pts_src_new2old,
            self.depth_src_new2old,
            self.intrinsic_new2old
        )
        self.pts_3d_src_camera_old2new = self.map23D(
            self.pts_src_old2new,
            self.depth_src_old2new,
            self.intrinsic_old2new
        )

        return True

    def map23D(self, pts_src, depth_src, intrinsic):
        """
        Convert 2D image points with depth to 3D camera coordinates

        Args:
            pts_src: Source 2D points in image coordinates [N, M, 2]
            depth_src: Corresponding depth values [N, M]
            intrinsic: Camera intrinsic matrix [N, 3, 3]

        Returns:
            pts_3d_src_camera: 3D points in camera coordinates [N, M, 3]
        """
        num_pairs, num_points, _ = pts_src.shape
        # Scale 2D points by depth
        pts_3d_src_camera = pts_src * depth_src.view([num_pairs, num_points, 1])
        # Transform to camera coordinates using inverse intrinsic matrix
        pts_3d_src_camera = pts_3d_src_camera @ torch.transpose(torch.inverse(intrinsic), dim0=1, dim1=2)
        # Convert to homogeneous coordinates and back to get clean 3D points
        pts_3d_src_camera = to_homogeneous(pts_3d_src_camera)
        pts_3d_src_camera = from_homogeneous(pts_3d_src_camera)
        return pts_3d_src_camera

    def optimize(self, old_pose_w2c, newoldpose, adjustment=None, mode="new2old"):
        """
        Optimize the pose by computing residuals between projected and observed points

        Args:
            old_pose_w2c: Original world-to-camera pose [4,4]
            newoldpose: Relative pose transformation between new and old frames [4,4]
            adjustment: Optional scaling factor for 3D points
            mode: Direction of optimization ("new2old" or "old2new")

        Returns:
            tuple: (optimized pose [4,4], residuals [N,M])
        """
        if mode == "new2old":
            new_pose_w2c = newoldpose.inverse() @ old_pose_w2c
        elif mode == "old2new":
            new_pose_w2c = newoldpose @ old_pose_w2c
        else:
            raise ValueError(f"Invalid optimization mode: {mode}")

        residuals = self.compute_residual(new_pose_w2c, adjustment, mode)
        return new_pose_w2c.detach(), residuals

    def compute_residual(
            self,
            new_pose_w2c: torch.Tensor,
            adjustment: Optional[torch.Tensor] = None,
            mode: str = "new2old"
    ) -> torch.Tensor:
        """
        Compute 2D projection residuals between projected source points and observed destination points

        Args:
            new_pose_w2c: New world-to-camera pose [4,4]
            adjustment: Optional scaling factor for 3D points
            mode: Direction of computation ("new2old" or "old2new")

        Returns:
            residual2D: 2D projection residuals [N,M]
        """
        if mode == "new2old":
            src2dst_pose = self.old_w2c_poses_new2old @ torch.inverse(new_pose_w2c).unsqueeze(0)
            pts_3d_src = self.pts_3d_src_camera_new2old * adjustment
            intrinsic = self.intrinsic_new2old
            pts_2d_dst = self.pts_dst_new2old
        elif mode == "old2new":
            src2dst_pose = new_pose_w2c.unsqueeze(0) @ torch.inverse(self.old_w2c_poses_old2new)
            pts_3d_src = self.pts_3d_src_camera_old2new
            intrinsic = self.intrinsic_old2new
            pts_2d_dst = self.pts_dst_old2new
        else:
            raise ValueError(f"Invalid computation mode: {mode}")

        # Project 3D source points to destination image plane
        pts_3d_src = to_homogeneous(pts_3d_src)
        pts_2d_src_prj = pts_3d_src @ torch.transpose(src2dst_pose, dim0=1, dim1=2)
        pts_2d_src_prj = pts_2d_src_prj[:, :, 0:3] @ torch.transpose(intrinsic, dim0=1, dim1=2)
        pts_2d_src_prj = from_homogeneous(pts_2d_src_prj)

        # Compute Euclidean distance between projected and observed points
        residual2D = torch.sqrt(torch.sum((pts_2d_src_prj - pts_2d_dst[:, :, 0:2]) ** 2, dim=-1) + 1e-10)
        return residual2D
