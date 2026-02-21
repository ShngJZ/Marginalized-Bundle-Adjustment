import os, sys, copy, time
import numpy as np
import PIL.Image

import kapture
from collections import OrderedDict
from kapture.io.csv import kapture_from_dir

prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dust3r_root = os.path.join(prj_root, 'third_party', 'dust3r')
sys.path.insert(0, dust3r_root)

from dust3r_visloc.datasets.base_dataset import BaseVislocDataset
from MargBA.datasets.data_utils import numpy_image_to_torch, cam_to_world_from_kapture
from kapture.io.records import depth_map_from_file

class SevenScenes(BaseVislocDataset):
    def __init__(self, root, output_location, subscene, seq, mode, rank, world_size):
        super().__init__()
        self.root = root
        self.subscene = subscene
        self.output_location = output_location
        self.seq = seq
        self.mode = mode

        query_path = os.path.join(self.root, subscene, 'query')
        kdata_query = kapture_from_dir(query_path)
        assert kdata_query.records_camera is not None and kdata_query.trajectories is not None and kdata_query.rigs is not None
        kapture.rigs_remove_inplace(kdata_query.trajectories, kdata_query.rigs)
        kdata_query_searchindex = {kdata_query.records_camera[(timestamp, sensor_id)]: (timestamp, sensor_id) for timestamp, sensor_id in kdata_query.records_camera.key_pairs()}
        kdata_query = {
            'sensors': copy.deepcopy(kdata_query.sensors),
            'trajectories': copy.deepcopy(kdata_query.trajectories),
        }
        self.query_data = {'path': query_path, 'kdata': kdata_query, 'searchindex': kdata_query_searchindex}

        map_path = os.path.join(self.root, subscene, 'mapping')
        kdata_map = kapture_from_dir(map_path)
        assert kdata_map.records_camera is not None and kdata_map.trajectories is not None and kdata_map.rigs is not None
        kapture.rigs_remove_inplace(kdata_map.trajectories, kdata_map.rigs)
        kdata_map_searchindex = {kdata_map.records_camera[(timestamp, sensor_id)]: (timestamp, sensor_id) for timestamp, sensor_id in kdata_map.records_camera.key_pairs()}
        kdata_map = {
            'sensors': copy.deepcopy(kdata_map.sensors),
            'trajectories': copy.deepcopy(kdata_map.trajectories),
        }
        self.map_data = {'path': map_path, 'kdata': kdata_map, 'searchindex': kdata_map_searchindex}

        # acquire map and query-seq images
        self.acquire_image_indices()
        # acquire pairs
        self.acquire_image_pairs()

        # Down sample
        self.image_indices = self.image_indices[rank::world_size]
        self.image_indices_pair = self.image_indices_pair[rank::world_size]

    def acquire_image_indices(self):
        global_idx = 0
        name2index_mapper = OrderedDict()
        for idx, x in enumerate(self.query_data['searchindex']):
            if os.path.dirname(x) == self.seq:
                name2index_mapper[x] = global_idx
                global_idx += 1
        for idx, x in enumerate(self.map_data['searchindex']):
            name2index_mapper[x] = global_idx
            global_idx += 1

        if self.mode == 'monodepth':
            lines = [f"{str(name2index_mapper[x]).zfill(6)} {x}\n" for x in name2index_mapper]
            name2index_mapper_txt = os.path.join(self.output_location, 'mapper.txt')
            time.sleep(np.random.randint(15))
            if not os.path.exists(name2index_mapper_txt):
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
        pair_path = os.path.join(self.root, self.subscene, 'pairfiles', f'{self.subscene}_{self.seq}.txt')
        entries = open(pair_path, 'r').readlines()[2::]
        entries = [x.rstrip('\n') for x in entries]
        pairs = list()
        for x in entries:
            img1, img2, _ = x.split(', ')
            idx1, idx2 = self.name2index_mapper[img1], self.name2index_mapper[img2]
            if int(idx1) < int(idx2):
                pairs.append(f"{idx1} {idx2}")
            else:
                pairs.append(f"{idx2} {idx1}")
        pairs = list(set(pairs))
        pairs = [[int(x.split(' ')[0]), int(x.split(' ')[1])] for x in pairs]
        self.image_indices_pair = pairs

    def name2index(self):
        self.name2index_mapper = dict()
        for idx, x in enumerate(self.query_data['searchindex']):
            self.name2index_mapper[x] = idx
        for idx, x in enumerate(self.map_data['searchindex']):
            self.name2index_mapper[x] = idx + len(self.query_data['searchindex'])

    def read_image_by_name(self, img_name):
        if img_name in self.map_data['searchindex']:
            searchindex, kdata, imgpath = self.map_data['searchindex'], self.map_data['kdata'], self.map_data['path']
        elif img_name in self.query_data['searchindex']:
            searchindex, kdata, imgpath = self.query_data['searchindex'], self.query_data['kdata'], self.query_data['path']
        else:
            raise ValueError(f"Image {img_name} not found in map or query data")

        timestamp, camera_id = searchindex[img_name]

        # for 7scenes, SIMPLE_PINHOLE
        camera_params = kdata['sensors'][camera_id].camera_params
        W, H, f, cx, cy = camera_params
        intrinsics = np.float32([(f, 0, cx),
                                 (0, f, cy),
                                 (0, 0, 1)])

        # Load RGB image
        rgb_image = PIL.Image.open(os.path.join(imgpath, 'sensors/records_data', img_name)).convert('RGB')
        rgb_image.load()

        # Load Pose
        cam_to_world = cam_to_world_from_kapture(kdata, timestamp, camera_id)

        # Load Depth
        depthmap_filename = os.path.join(imgpath, 'sensors/records_data', img_name.replace('color.png', 'depth.reg'))
        depthmap = depth_map_from_file(depthmap_filename, (int(W), int(H))).astype(np.float32)
        return np.array(rgb_image), intrinsics, cam_to_world, depthmap

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
        rgb_src, intrinsic_src, cam_to_world_src, _ = self.read_image_by_name(img_name=name_src)
        rgb_dst, intrinsic_dst, cam_to_world_dst, _ = self.read_image_by_name(img_name=name_dst)
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
        rgb, intrinsic, pose_c2w, depth_gt = self.read_image_by_name(img_name=rgb_file)
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
        if self.mode == 'monodepth':
            ret = self.monodepth_getitem(idx)
        elif self.mode == 'correspondence':
            ret = self.corres_getitem(idx)
        else:
            raise NotImplementedError()
        return ret
