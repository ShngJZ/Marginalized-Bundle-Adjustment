import os
import io
import h5py
import numpy as np
import torch
import natsort

from PIL import Image
from typing import Any, Dict, List, Tuple, Optional
from MargBA.datasets.data_utils import numpy_image_to_torch, resize
from MargBA.datasets import cvt_png_to_monodepth

class ScanNet(torch.utils.data.Dataset):
    """
    ScanNet dataset for monocular depth estimation or pair-wise correspondence estimation.

    This class handles loading and preprocessing of ScanNet data from HDF5 files.
    It supports two modes: 'monodepth' for single image depth estimation and
    'correspondence' for pair-wise correspondence estimation.
    """

    def __init__(self, data_root: str, mode: str, rank: int, world_size: int):
        """
        Initialize the ScanNet dataset.

        Args:
            data_root: Path to the dataset root directory
            mode: Dataset mode, either 'monodepth' or 'correspondence'
            rank: Process rank for distributed processing
            world_size: Total number of processes for distributed processing
        """
        self.height = 480
        self.width = 640
        self.mode = mode
        self.frame_skip = 6  # Use every 6th frame (30/6 = 5 FPS)

        # Validate mode
        if mode not in ['monodepth', 'correspondence']:
            raise ValueError(f"Mode '{mode}' not supported. Use 'monodepth' or 'correspondence'")

        # Set up file paths
        self.h5py_path = os.path.join(data_root, f"{os.path.basename(data_root)}.hdf5")

        # Load dataset metadata
        self.image_indices, self.image_names, self.poses_c2w, self.intrinsics = self.read_images_poses_intrinsics()

        # Initialize pair-wise indices for correspondence mode
        self.init_image_indices_pairwise()

        # Distribute data across processes
        self.image_indices = self.image_indices[rank::world_size]
        self.image_indices_pair = self.image_indices_pair[rank::world_size]

    def read_images_poses_intrinsics(self) -> Tuple[List[int], List[str], List[np.ndarray], List[np.ndarray]]:
        """
        Read image names, camera poses, and intrinsic matrices from the HDF5 file.

        Returns:
            Tuple containing:
            - List of image indices
            - List of image names
            - List of camera-to-world pose matrices
            - List of camera intrinsic matrices
        """
        with h5py.File(self.h5py_path, 'r') as h5f:
            image_names = h5f['color'].keys()
            image_names = natsort.natsorted(list(image_names))
            image_names = image_names[::self.frame_skip]  # Sample at reduced frame rate

            poses_c2w = []
            intrinsics = []

            for image_name in image_names:
                stem_name = image_name.split('.')[0]

                # Read camera pose
                pose_bytes = np.array(h5f['pose'][f'{stem_name}.txt'])
                pose_text = io.BytesIO(pose_bytes).read().decode('UTF-8')
                pose_c2w = self.read_scannet_pose(io.StringIO(pose_text))

                # Read camera intrinsics (same for all frames)
                intrinsic_bytes = np.array(h5f['intrinsic']['intrinsic_color.txt'])
                intrinsic_text = io.BytesIO(intrinsic_bytes).read().decode('UTF-8')
                intrinsic = self.read_scannet_intrinsic(io.StringIO(intrinsic_text))

                poses_c2w.append(pose_c2w)
                intrinsics.append(intrinsic)

        image_indices = list(range(len(image_names)))
        return image_indices, image_names, poses_c2w, intrinsics

    def init_image_indices_pairwise(self) -> None:
        """
        Initialize pair-wise image indices for correspondence estimation.

        Reads pairs from a text file and stores them for later use.
        """
        scene_name = os.path.basename(self.h5py_path).split('.')[0]
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(self.h5py_path)))
        pair_txt_path = os.path.join(base_dir, 'scans_test_pairs', f'{scene_name}.txt')

        with open(pair_txt_path, 'r') as f:
            pairs = f.readlines()

        # Clean line endings
        pairs = [x.strip() for x in pairs]

        self.image_indices_pair = []
        for pair in pairs:
            i, j = map(int, pair.split(' '))
            if i < j:  # Only keep pairs where first index is smaller
                self.image_indices_pair.append([i, j])

    def read_scannet_pose(self, path: io.StringIO) -> np.ndarray:
        """
        Read ScanNet's Camera2World pose matrix.

        Args:
            path: StringIO object containing the pose data

        Returns:
            Camera-to-world transformation matrix as numpy array
        """
        cam2world = np.loadtxt(path, delimiter=' ')
        return cam2world

    def read_scannet_intrinsic(self, path: io.StringIO) -> np.ndarray:
        """
        Read ScanNet's intrinsic camera matrix.

        Args:
            path: StringIO object containing the intrinsic data

        Returns:
            3x3 camera intrinsic matrix as numpy array
        """
        intrinsic = np.loadtxt(path, delimiter=' ')
        return intrinsic[:-1, :-1]  # Return only the 3x3 part

    def load_depth(self, depth_ref: io.BytesIO) -> np.ndarray:
        """
        Load and preprocess depth image.

        Args:
            depth_ref: BytesIO object containing the depth image

        Returns:
            Processed depth image as numpy array
        """
        depth = Image.open(depth_ref)
        depth = np.array(depth).astype(np.float32)
        depth = depth / 1000  # Convert millimeters to meters

        # Resize to target dimensions
        depth, _ = resize(depth, (self.height, self.width))

        # Create validity mask and filter invalid depths
        depth_valid = (depth > 0).astype(np.float32)
        depth_valid, _ = resize(depth_valid, (self.height, self.width))
        depth[depth_valid < 0.9] = 0

        return depth

    def load_rgb_intrinsic(self, rgb: io.BytesIO, K: np.ndarray) -> Tuple[torch.Tensor, Image.Image, np.ndarray]:
        """
        Load and preprocess RGB image and adjust intrinsic matrix accordingly.

        Args:
            rgb: BytesIO object containing the RGB image
            K: Original camera intrinsic matrix

        Returns:
            Tuple containing:
            - Processed RGB image as torch tensor
            - Original RGB image resized to 640x480
            - Adjusted camera intrinsic matrix
        """
        rgb_image = Image.open(rgb)
        rgb_wo_resize = rgb_image  # Keep original for later resizing

        # Get original dimensions
        width_orig, height_orig = rgb_image.size

        # Resize to target dimensions
        rgb_resized, _ = resize(rgb_image, (self.height, self.width))
        rgb_array = np.array(rgb_resized).astype(np.float32)
        rgb_tensor = numpy_image_to_torch(rgb_array)

        # Scale intrinsic matrix to match new dimensions
        K_scaled = self.scale_intrinsic(
            K,
            wtarget=self.width,
            htarget=self.height,
            worg=width_orig,
            horg=height_orig
        )

        return rgb_tensor, rgb_wo_resize.resize((640, 480)), K_scaled

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            Number of samples based on the current mode
        """
        if self.mode == 'monodepth':
            return len(self.image_indices)
        elif self.mode == 'correspondence':
            return len(self.image_indices_pair)
        else:
            raise NotImplementedError(f"Mode '{self.mode}' not implemented")

    def scale_intrinsic(self, K: np.ndarray, wtarget: int, htarget: int,
                        worg: int, horg: int) -> np.ndarray:
        """
        Scale the intrinsic matrix to match new image dimensions.

        Args:
            K: Original intrinsic matrix
            wtarget: Target width
            htarget: Target height
            worg: Original width
            horg: Original height

        Returns:
            Scaled intrinsic matrix
        """
        sx, sy = wtarget / worg, htarget / horg
        scaling_matrix = np.eye(3)
        scaling_matrix[0, 0], scaling_matrix[1, 1] = sx, sy
        return scaling_matrix @ K

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing the sample data
        """
        if self.mode == 'monodepth':
            return self.monodepth_getitem(idx)
        elif self.mode == 'correspondence':
            return self.corres_getitem(idx)
        else:
            raise NotImplementedError(f"Mode '{self.mode}' not implemented")

    def read_processed_elements_from_h5(self, image_idx: int) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, Image.Image, np.ndarray]:
        """
        Read and process image, pose, and intrinsic data from the HDF5 file.

        Args:
            image_idx: Index of the image to read

        Returns:
            Tuple containing:
            - Camera intrinsic matrix
            - Camera-to-world pose matrix
            - RGB image as torch tensor
            - RGB image without resizing
            - Ground truth depth image
        """
        rgb_file = self.image_names[image_idx]
        intrinsic = self.intrinsics[image_idx]
        pose_c2w = self.poses_c2w[image_idx]

        with h5py.File(self.h5py_path, 'r') as h5f:
            stem_name = rgb_file.split('.')[0]

            # Load RGB image and adjust intrinsics
            rgb_bytes = np.array(h5f['color'][f'{stem_name}.jpg'])
            rgb, rgb_wo_resize, intrinsic = self.load_rgb_intrinsic(
                io.BytesIO(rgb_bytes), K=intrinsic
            )

            # Load depth image
            depth_bytes = np.array(h5f['depth'][f'{stem_name}.png'])
            depth_gt = self.load_depth(io.BytesIO(depth_bytes))

        # Ensure dimensions match
        assert depth_gt.shape[:2] == rgb.shape[1:], "Depth and RGB dimensions don't match"

        # Ensure correct data types
        intrinsic = intrinsic[:3, :3].astype(np.float32)

        return intrinsic, pose_c2w, rgb, rgb_wo_resize, depth_gt

    def monodepth_getitem(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample for monocular depth estimation.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing the sample data for monocular depth estimation
        """
        image_idx = self.image_indices[idx]
        rgb_file = self.image_names[image_idx]
        intrinsic, pose_c2w, rgb, rgb_wo_resize, depth_gt = self.read_processed_elements_from_h5(image_idx)

        return {
            'image_idx': image_idx,
            "rgb_file": rgb_file,
            'depth_gt': depth_gt,  # (H, W)
            'image': rgb,  # torch tensor (3, H, W)
            'intr': intrinsic,
            'pose_c2w': pose_c2w
        }

    def read_predicted_depth(self, depth_pred_path: str) -> np.ndarray:
        """
        Read and convert predicted depth from a PNG file.

        Args:
            depth_pred_path: Path to the predicted depth PNG file

        Returns:
            Converted monocular depth as numpy array
        """
        depth_image = Image.open(depth_pred_path)
        monodepth = cvt_png_to_monodepth(depth_image)
        return monodepth

    def corres_getitem(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample for correspondence estimation.

        Args:
            idx: Index of the sample pair to retrieve

        Returns:
            Dictionary containing the sample data for correspondence estimation
        """
        # Get source and destination image indices
        image_idx_src, image_idx_dst = self.image_indices_pair[idx]

        # Load source image data
        intrinsic_src, pose_c2w_src, rgb_src, rgb_src_wo_resize, depth_gt_src = self.read_processed_elements_from_h5(image_idx_src)

        # Load destination image data
        intrinsic_dst, pose_c2w_dst, rgb_dst, rgb_dst_wo_resize, depth_gt_dst = self.read_processed_elements_from_h5(image_idx_dst)

        # Calculate relative pose from source to destination
        pose_src2dst = np.linalg.inv(pose_c2w_dst) @ pose_c2w_src

        return {
            'intrinsic_src': intrinsic_src,
            'intrinsic_dst': intrinsic_dst,
            "rgb_src": rgb_src,
            "rgb_dst": rgb_dst,
            "rgb_src_wo_resize": np.array(rgb_src_wo_resize),
            "rgb_dst_wo_resize": np.array(rgb_dst_wo_resize),
            'pose_src2dst': pose_src2dst[:3].astype(np.float32),
            'image_idx_pair': [image_idx_src, image_idx_dst],
        }
