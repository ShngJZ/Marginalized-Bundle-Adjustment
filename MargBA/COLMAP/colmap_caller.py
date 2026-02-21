"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

"""Functions to get the poses using COLMAP with different matchers. """

import imageio
import os
import torch
import sys
import PIL.Image as Image
import pycolmap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict, Any, Optional

from third_party.colmap_read_write_model import read_images_binary_to_poses, read_images_binary, read_points3D_binary
from .reconstruction_know_intrinsics_for_hloc import reconstruction_w_known_intrinsics

sys.path.append(str(Path(__file__).parent / '../../third_party/Hierarchical-Localization'))
from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction, logger

def define_pycolmap_camera(K, height, width):
    """To use in HLOC. """
    focal_length = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    camera = pycolmap.Camera(
        model='SIMPLE_PINHOLE', width=width, height=height, params=[focal_length, cx, cy]
    )
    return camera

def get_poses_and_idx(sfm_dir: str, image_list: List[str]):
    """Reads from the files outputted by COLMAP and outputs the initial poses,
    valid indices.

    Args:
        sfm_dir (str):  Path to directory where the outputted file of COLMAP are, i.e
                        images.bin and points3D.bin
        image_list (list): list of image names, to make sure the outputted poses are
                           in the same order than the list of images.
        B (int): number of images
        H (int), W (int): Dimension of the images (to use in the depth maps)

    Returns:
        initial_poses_w2c: The world-to-camera poses obtained by COLMAP (opencv/colmap format)
                            Shape is (B, 4, 4)
        valid_poses_idx: List of bools, indicating is a pose was found
        excluded_poses_idx: List of bools, indicating indices for which no pose was found
    """
    poses_w2c, valid_index, invalid_index = dict(), list(), list()
    dict_images_to_pose = read_images_binary_to_poses(os.path.join(sfm_dir, 'images.bin'))

    for i, image_name in enumerate(image_list):
        if image_name in dict_images_to_pose:
            poses_w2c[image_name] = dict_images_to_pose[image_name]
            valid_index.append(i)
        else:
            invalid_index.append(i)
    return poses_w2c, valid_index, invalid_index


def compute_sfm_inloc_wt_intrinsic(
        save_dir,
        rgb_paths,
        intrinsics,
        H,
        W,
        matcher_model_type='outdoor'
):
    """COLMAP with SuperPoint-SuperGlue. That is the default hloc. """
    mapper_options = {
        'ba_refine_focal_length': False,  # arg (=1)
        'ba_refine_principal_point': False,  # arg (=0)
        'ba_refine_extra_params': False,  # (=1)
        'min_num_matches': 5,  # arg (=15)
        'ba_local_max_num_iterations': 25,  # arg (=25)
        'ba_global_max_num_iterations': 50,  # arg (=50)
    }

    outputs = Path(save_dir)
    sfm_pairs = outputs / 'pairs-exhaustive.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue'

    feature_conf = extract_features.confs['superpoint_max']
    matcher_conf = match_features.confs['superglue']
    matcher_conf['model']['weights'] = matcher_model_type

    # get image list
    intrinsics = intrinsics  # (3, 3)
    image_list = [os.path.basename(path) for path in rgb_paths]

    # create a fake image dir and save the images there
    image_dir = os.path.join(outputs, 'images')
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    for image_id, image_name in enumerate(image_list):
        imageio.imwrite(os.path.join(image_dir, image_name), Image.open(rgb_paths[image_id]))

    # list of all iamge pairs
    pairs_from_exhaustive.main(output=sfm_pairs, image_list=image_list)

    # Extract and match local features
    feature_path = extract_features.main(
        feature_conf, Path(image_dir), outputs
    )
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf['output'], outputs
    )

    camera_known_intrinsics = define_pycolmap_camera(intrinsics, height=H, width=W)
    model = reconstruction_w_known_intrinsics(
        sfm_dir,
        Path(image_dir),
        sfm_pairs,
        feature_path,
        match_path,
        cam=camera_known_intrinsics,
        image_list=image_list,
        mapper_options=mapper_options
    )


def compute_sfm_inloc_wo_intrinsic(
        save_dir,
        rgb_paths,
        matcher_model_type='outdoor'
):
    """COLMAP with SuperPoint-SuperGlue. That is the default hloc. """
    mapper_options = {
        'ba_refine_focal_length': True,  # arg (=1)
        'ba_refine_principal_point': False,  # arg (=0)
        'ba_refine_extra_params': True,  # (=1)
        'min_num_matches': 5,  # arg (=15)
        'ba_local_max_num_iterations': 25,  # arg (=25)
        'ba_global_max_num_iterations': 50,  # arg (=50)
    }

    outputs = Path(save_dir)
    sfm_pairs = outputs / 'pairs-exhaustive.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue'

    feature_conf = extract_features.confs['superpoint_max']
    matcher_conf = match_features.confs['superglue']
    matcher_conf['model']['weights'] = matcher_model_type

    # get image list
    image_list = [os.path.basename(path) for path in rgb_paths]

    # create a fake image dir and save the images there
    image_dir = os.path.join(outputs, 'images')
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    for image_id, image_name in enumerate(image_list):
        imageio.imwrite(os.path.join(image_dir, image_name), Image.open(rgb_paths[image_id]))

    # list of all iamge pairs
    pairs_from_exhaustive.main(output=sfm_pairs, image_list=image_list)

    # Extract and match local features
    feature_path = extract_features.main(
        feature_conf, Path(image_dir), outputs
    )
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf['output'], outputs
    )

    model = reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=Path(image_dir),
        pairs=sfm_pairs,
        features=feature_path,
        matches=match_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        mapper_options=mapper_options,
    )