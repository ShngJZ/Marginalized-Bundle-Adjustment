import os, copy, glob
import time
import natsort
import torch
import numpy as np
import PIL.Image

from collections import OrderedDict
from MargBA.datasets.data_utils import numpy_image_to_torch

class Wayspots(torch.utils.data.Dataset):
    def __init__(self, root, output_location, scene, mode, rank, world_size):
        super().__init__()
        self.root = root
        self.scene = scene
        self.output_location = output_location
        self.mode = mode

        # acquire map and query-seq images
        self.acquire_image_indices()
        # acquire pairs
        self.acquire_image_pairs()

        # Down sample
        self.image_indices = self.image_indices[rank::world_size]
        self.image_indices_pair = self.image_indices_pair[rank::world_size]

    def acquire_image_indices(self):
        name2index_mapper = OrderedDict()
        for split in ["test", "train"]:
            st = len(name2index_mapper)
            rgbs = glob.glob(os.path.join(self.root, split, "rgb", "*.jpg"))
            rgbs = [os.path.basename(x) for x in rgbs]
            rgbs = natsort.natsorted(rgbs)
            rgbs = [os.path.join(split, "rgb", x) for x in rgbs]
            for i in range(len(rgbs)):
                name2index_mapper[rgbs[i]] = st + i

        if self.mode == 'monodepth':
            time.sleep(np.random.randint(15))
            lines = [f"{str(name2index_mapper[x]).zfill(6)} {x}\n" for x in name2index_mapper]
            name2index_mapper_txt = os.path.join(self.output_location, 'mapper.txt')
            open(name2index_mapper_txt, "w").writelines(lines)

        pair_path = os.path.join(self.output_location, 'mapper.txt')
        entries = open(pair_path, 'r').readlines()
        entries = [x.rstrip('\n') for x in entries]

        self.image_indices, self.image_names = list(), list()
        for entry in entries:
            ind, name = entry.split(' ')
            self.image_indices.append(int(ind))
            self.image_names.append(name)
        self.name2index_mapper = name2index_mapper

    def acquire_image_pairs(self):
        """Generate all possible image pairs for correspondence mode"""
        pairs = []
        # Create all unique ordered pairs where idx1 < idx2
        for idx1 in self.image_indices:
            for idx2 in self.image_indices:
                if int(idx1) < int(idx2):
                    pairs.append([idx1, idx2])
        self.image_indices_pair = pairs

    def name2index(self):
        """Create mapping from image names to indices using query and map data"""
        self.name2index_mapper = dict()
        for idx, x in enumerate(self.query_data['searchindex']):
            self.name2index_mapper[x] = idx
        for idx, x in enumerate(self.map_data['searchindex']):
            self.name2index_mapper[x] = idx + len(self.query_data['searchindex'])

    def read_image_by_name(self, img_name):
        """Read image and its metadata (intrinsics, pose, etc.) from given image name

        Args:
            img_name: Path to the image file

        Returns:
            tuple: (rgb_image, intrinsic_matrix, camera_to_world_matrix, depth_map)
        """
        # Load RGB image
        rgb = PIL.Image.open(os.path.join(self.root, img_name)).convert('RGB')
        width, height = rgb.size

        # Parse image path to get split and index
        split, wayspots_indx = img_name.split('/')[0], img_name.split('/')[-1].split('.')[0].split('_')[1]

        # Load camera intrinsics
        focal = np.loadtxt(os.path.join(self.root, split, "calibration", f"calibration_{wayspots_indx}.txt"))
        intrinsic = np.eye(3)
        intrinsic[0, 0] = focal
        intrinsic[1, 1] = focal
        intrinsic[0, 2] = width / 2  # Principal point x
        intrinsic[1, 2] = height / 2  # Principal point y

        # Load camera pose
        cam_to_world = np.loadtxt(os.path.join(self.root, split, "poses", f"pose_{wayspots_indx}.txt"))

        # Initialize empty depth map
        depthmap = np.zeros([height, width])
        rgb_np = np.array(rgb)

        return rgb_np, intrinsic, cam_to_world, depthmap

    def __len__(self):
        """Get dataset length based on current mode"""
        if self.mode == 'monodepth':
            return len(self.image_indices)
        elif self.mode == 'correspondence':
            return len(self.image_indices_pair)
        else:
            raise NotImplementedError()

    def corres_getitem(self, idx):
        """Get correspondence data item at given index

        Args:
            idx: Index of the item to retrieve

        Returns:
            dict: Dictionary containing correspondence data
        """
        image_idx_src, image_idx_dst = self.image_indices_pair[idx]
        name_src, name_dst = self.image_names[image_idx_src], self.image_names[image_idx_dst]

        # Read source and destination images
        rgb_src, intrinsic_src, cam_to_world_src, _ = self.read_image_by_name(img_name=name_src.rstrip('.JPG'))
        rgb_dst, intrinsic_dst, cam_to_world_dst, _ = self.read_image_by_name(img_name=name_dst.rstrip('.JPG'))

        # Convert to torch tensors
        rgb_src = numpy_image_to_torch(rgb_src.astype(np.float32))
        rgb_dst = numpy_image_to_torch(rgb_dst.astype(np.float32))

        # Compute relative pose
        pose_src2dst = np.linalg.inv(cam_to_world_dst) @ cam_to_world_src

        return {
            'intrinsic_src': intrinsic_src,
            'intrinsic_dst': intrinsic_dst,
            "rgb_src": rgb_src,
            "rgb_dst": rgb_dst,
            "rgb_src_wo_resize": rgb_src,
            "rgb_dst_wo_resize": rgb_dst,
            'pose_src2dst': pose_src2dst[:3].astype(np.float32),
            'image_idx_pair': [image_idx_src, image_idx_dst],
        }

    def monodepth_getitem(self, idx):
        """Get monodepth data item at given index

        Args:
            idx: Index of the item to retrieve

        Returns:
            dict: Dictionary containing monodepth data
        """
        image_idx = self.image_indices[idx]
        rgb_file = self.image_names[image_idx]

        # Read image and metadata
        rgb, intrinsic, pose_c2w, depth_gt = self.read_image_by_name(
            img_name=rgb_file.rstrip('.JPG')
        )
        rgb = numpy_image_to_torch(rgb.astype(np.float32))

        return {
            'image_idx': image_idx,
            "rgb_file": rgb_file,
            'depth_gt': depth_gt,
            'image': rgb,
            'intr': intrinsic,
            'pose_c2w': pose_c2w
        }
        return ret

    def __getitem__(self, idx: int):
        """Get item from dataset based on mode
        Args:
            idx (int): Index of the item to retrieve
        Returns:
            dict: Dictionary containing data based on mode
        Raises:
            NotImplementedError: If mode is not supported
        """
        if self.mode == 'monodepth':
            ret = self.monodepth_getitem(idx)  # Get monodepth data
        elif self.mode == 'correspondence':
            ret = self.corres_getitem(idx)  # Get correspondence data
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")
        return ret