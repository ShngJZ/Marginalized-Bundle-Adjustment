import os, copy, glob

import natsort
import torch
import numpy as np
import PIL.Image

from collections import OrderedDict
from MargBA.datasets.data_utils import numpy_image_to_torch

class IMC2021(torch.utils.data.Dataset):
    def __init__(self, root, output_location, mode):
        super().__init__()
        self.root = root
        self.output_location = output_location
        self.mode = mode

        # acquire map and query-seq images
        self.acquire_image_indices()
        # acquire pairs
        self.acquire_image_pairs()

    def acquire_image_indices(self):
        name2index_mapper = OrderedDict()
        rgbs = glob.glob(os.path.join(self.root, "images", "*.jpg"))
        rgbs = [os.path.basename(x) for x in rgbs]
        rgbs = natsort.natsorted(rgbs)
        for i in range(len(rgbs)):
            name2index_mapper[rgbs[i]] = i

        self.image_indices, self.image_names = list(), list()
        for name in name2index_mapper:
            self.image_indices.append(name2index_mapper[name])
            self.image_names.append(name)
        self.name2index_mapper = name2index_mapper

    def acquire_image_pairs(self):
        pairs = list()
        for idx1 in self.image_indices:
            for idx2 in self.image_indices:
                if int(idx1) < int(idx2):
                    pairs.append([idx1, idx2])
        self.image_indices_pair = pairs

    def name2index(self):
        self.name2index_mapper = dict()
        for idx, x in enumerate(self.query_data['searchindex']):
            self.name2index_mapper[x] = idx
        for idx, x in enumerate(self.map_data['searchindex']):
            self.name2index_mapper[x] = idx + len(self.query_data['searchindex'])

    def read_image_by_name(self, img_name):
        rgb = PIL.Image.open(os.path.join(self.root, "images", f"{img_name}.jpg")).convert('RGB')
        intrinsic = np.loadtxt(os.path.join(self.root, "intrins", f"{img_name}.txt"))
        world_to_cam = np.loadtxt(os.path.join(self.root, "poses", f"{img_name}.txt"))
        cam_to_world = np.linalg.inv(world_to_cam)
        wd, ht = rgb.size
        depthmap = np.zeros([ht, wd])
        return np.array(rgb), intrinsic, cam_to_world, depthmap

    def __len__(self):
        if self.mode == 'monodepth':
            return len(self.image_indices)
        elif self.mode == 'correspondence':
            return len(self.image_indices_pair)
        else:
            raise NotImplementedError()

    def corres_getitem(self, idx):
        image_idx_src, image_idx_dst = self.image_indices_pair[idx]
        name_src , name_dst = self.image_names[image_idx_src], self.image_names[image_idx_dst]
        rgb_src, intrinsic_src, cam_to_world_src, _ = self.read_image_by_name(img_name=name_src.rstrip('.jpg'))
        rgb_dst, intrinsic_dst, cam_to_world_dst, _ = self.read_image_by_name(img_name=name_dst.rstrip('.jpg'))
        rgb_src, rgb_dst = numpy_image_to_torch(rgb_src.astype(np.float32)), numpy_image_to_torch(rgb_dst.astype(np.float32))
        pose_src2dst = np.linalg.inv(cam_to_world_dst) @ cam_to_world_src
        ret = {
            'intrinsic_src': intrinsic_src,
            'intrinsic_dst': intrinsic_dst,
            "rgb_src": rgb_src,
            "rgb_dst": rgb_dst,
            "rgb_src_wo_resize": rgb_src,
            "rgb_dst_wo_resize": rgb_dst,
            'pose_src2dst': pose_src2dst[:3].astype(np.float32),
            'image_idx_pair': [image_idx_src, image_idx_dst],
        }
        return ret

    def monodepth_getitem(self, idx):
        image_idx = self.image_indices[idx]
        rgb_file = self.image_names[image_idx]
        rgb, intrinsic, pose_c2w, depth_gt = self.read_image_by_name(
            img_name=rgb_file.rstrip('.jpg')
        )
        rgb = numpy_image_to_torch(rgb.astype(np.float32))
        ret = {
            'image_idx': image_idx,
            "rgb_file": rgb_file,
            'depth_gt': depth_gt,  # (H, W)
            'image': rgb,  # torch tensor (3, self.H, self.W)
            'intr': intrinsic,
            'pose_c2w': pose_c2w
        }
        return ret

    def __getitem__(self, idx: int):
        """
        Returns:
            image_idx (int): Index of the image in the dataset.
            rgb_file (str): Filename of the RGB image.
            depth_gt (np.ndarray): Ground-truth depth map with shape (H, W)
            image (torch.Tensor): Normalized RGB image tensor of shape (3, H, W), with values in [0.0, 1.0].
            intr (np.ndarray): Camera intrinsic matrix of shape (3, 3).
            pose_c2w (np.ndarray): Camera-to-world transformation matrix of shape (4, 4).
        """
        if self.mode == 'monodepth':
            ret = self.monodepth_getitem(idx)
        elif self.mode == 'correspondence':
            ret = self.corres_getitem(idx)
        else:
            raise NotImplementedError()
        return ret