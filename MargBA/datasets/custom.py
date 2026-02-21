import os, natsort, glob
import numpy as np
import torch

from PIL import Image
from typing import Any, Dict
from MargBA.datasets.data_utils import numpy_image_to_torch, resize

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, mode, rank, world_size):
        """
        scene: ScanNet Test Scene Name
        mode: Dataset for monocular depth estimation or pair-wise correspondence estimation
        """
        self.mode, self.data_root = mode, data_root
        self.image_indices, self.image_names = self.index_images()
        assert mode in ['monodepth', 'correspondence']

        # Init pair-wise indices
        self.init_image_indices_pairwise()

        # Down sample
        self.image_indices = self.image_indices[rank::world_size]
        self.image_indices_pair = self.image_indices_pair[rank::world_size]

    def index_images(self):
        image_names = []
        for ext in ['*.png', '*.jpg', '*.JPG']:
            image_names.extend(glob.glob(os.path.join(self.data_root, ext)))
        image_names = natsort.natsorted(list(image_names))
        image_indices = list(range(len(image_names)))
        return image_indices, image_names

    def init_image_indices_pairwise(self):
        self.image_indices_pair = list()
        for i in self.image_indices:
            for j in self.image_indices:
                if i < j:
                    self.image_indices_pair.append([i, j])

    def __len__(self):
        if self.mode == 'monodepth':
            return len(self.image_indices)
        elif self.mode == 'correspondence':
            return len(self.image_indices_pair)
        else:
            raise NotImplementedError()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == 'monodepth':
            ret = self.monodepth_getitem(idx)
        elif self.mode == 'correspondence':
            ret = self.corres_getitem(idx)
        else:
            raise NotImplementedError()
        return ret

    def read_rgb(self, image_idx):
        rgb_file = self.image_names[image_idx]
        rgb = Image.open(rgb_file)
        rgb = np.array(rgb).astype(np.float32)
        rgb = numpy_image_to_torch(rgb)
        return rgb

    def monodepth_getitem(self, idx: int) -> Dict[str, Any]:
        """
        Args:
            idx (int)

        Returns:
            a dictionary for each image index containing the following elements:
                * image_idx: the global index of the image
                * rgb_file: raw rgb image name
                * depth_gt: ground-truth depth map, numpy array of shape [H, W]
                * image: the corresponding image, a torch Tensor of shape [3, H, W]. The RGB values are
                            normalized to [0, 1] (not [0, 255]).
                * intr: intrinsics parameters, numpy array of shape [3, 3]
        """
        image_idx = self.image_indices[idx]
        rgb_file = self.image_names[image_idx]
        rgb = self.read_rgb(image_idx)

        ret = {
            'image_idx': image_idx,
            "rgb_file": rgb_file,
            'image': rgb,  # torch tensor (3, self.H, self.W)
        }
        return ret

    def corres_getitem(self, idx: int) -> Dict[str, Any]:
        image_idx_src, image_idx_dst = self.image_indices_pair[idx]
        rgb_src = self.read_rgb(image_idx_src)
        rgb_dst = self.read_rgb(image_idx_dst)

        ret = {
            "rgb_src": rgb_src,
            "rgb_dst": rgb_dst,
            'image_idx_pair': [image_idx_src, image_idx_dst],
        }
        return ret