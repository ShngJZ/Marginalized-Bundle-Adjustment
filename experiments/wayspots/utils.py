import copy, os, tqdm, sys

import torch
from collections import OrderedDict

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, proj_root)
torch.multiprocessing.set_sharing_strategy('file_system')

import numpy as np
from MargBA.geometry import pad_poses
from MargBA.datasets import HDF5Reader
from MargBA.BA.transform_utils import RandomCorrespondenceDepthSampler
from MargBA.corres_estimator import two_view_pose_estimation
from MargBA.poses_models.twoview_sfm_initialization import ReadVisibility

def read_intrinsic_pose_w2c(h5path):
    """Reads camera intrinsics and world-to-camera poses from HDF5 file.

    Args:
        h5path: Path to HDF5 file containing camera data

    Returns:
        Tuple of (intrinsics_list, poses_list) where each contains data for all frames
    """
    reader = HDF5Reader(h5path)
    img_names = reader.get_all_images()
    intrinsics_gt = [reader.read_intrinsic_gt(image_name) for image_name in img_names]
    poses_w2c_gt = [reader.read_pose_w2c_gt(image_name) for image_name in img_names]
    return intrinsics_gt, poses_w2c_gt


def pose_initialization(preprocess_per_scene_location, min_corres_conf=0.0):
    """Initializes camera poses using visibility information.

    Args:
        preprocess_per_scene_location: Directory containing preprocessed scene data
        min_corres_conf: Minimum correspondence confidence threshold

    Returns:
        Processed visibility connections between frames
    """
    h5file_path = os.path.join(
        preprocess_per_scene_location,
        "{}.hdf5".format(os.path.basename(preprocess_per_scene_location))
    )

    # Load visibility data from HDF5 file
    dataset = ReadVisibility(h5file_path)

    # Create dataloader for parallel processing
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=32,
        drop_last=False
    )

    # Collect all visibility connections
    connections = []
    for (idx, data) in enumerate(tqdm.tqdm(dataloader)):
        connections.append([
            data[0].item(),  # source frame index
            data[1].item(),  # target frame index
            data[2].item()   # visibility score
        ])
    connections_np = np.stack(connections, axis=0)

    idx2fname_mapper_path = os.path.join(preprocess_per_scene_location, 'mapper.txt')
    idx2fname_mapper = open(idx2fname_mapper_path).readlines()
    nframe = len(idx2fname_mapper)
    assert nframe == max([int(x.split(' ')[0]) for x in idx2fname_mapper]) + 1

    # read gt pose world_to_camera
    intrinsics_gt, poses_w2c_gt = read_intrinsic_pose_w2c(
        h5file_path
    )

    # assign query and map indices
    marker_query_map = list()
    for x in idx2fname_mapper:
        x = x.rstrip('\n')
        if 'test' in x:
            marker_query_map.append('qry')
        elif 'train' in x:
            marker_query_map.append('map')
        else:
            raise ValueError(f"{x} not found in map and query")

    # acquire intrinsic
    intrinsics = torch.from_numpy(np.stack(intrinsics_gt, axis=0))

    # read map pose world_to_camera
    poses_w2c_pr = [None] * (nframe)
    for entry in idx2fname_mapper:
        idx, fname = entry[0:-1].split(' ')
        idx = int(idx)
        if marker_query_map[idx] == 'map':
            poses_w2c_pr[idx] = poses_w2c_gt[idx]
    # Process frames that have visible pairs with respect to the map
    nqry = 0  # Counter for query frames
    twoview_frame_pairs = OrderedDict()  # Store frame pairs with highest visibility

    for entry in tqdm.tqdm(idx2fname_mapper):
        idx, fname = entry[0:-1].split(' ')
        idx = int(idx)

        # Only process query frames
        if marker_query_map[idx] == 'qry':
            nqry += 1
            connections_to_idx = []

            # Find all map frames connected to this query frame
            idx_src_connections = connections_np[connections_np[:, 0] == idx]
            idx_src_connections = list(idx_src_connections)
            for src_idx1, dst_idx2, visibility in idx_src_connections:
                src_idx1, dst_idx2 = int(src_idx1), int(dst_idx2)
                assert idx == src_idx1
                if marker_query_map[dst_idx2] == 'map':
                    connections_to_idx.append([src_idx1, dst_idx2, visibility])

            # Select the connection with highest visibility
            if connections_to_idx:
                visibility_to_idx = [vis for _, _, vis in connections_to_idx]
                max_visibility_id = np.argmax(np.array(visibility_to_idx))
                best_connection = connections_to_idx[max_visibility_id]
                twoview_frame_pairs[best_connection[0]] = best_connection[1]

    # Process frames that have no visible pairs with respect to the map
    for entry in tqdm.tqdm(idx2fname_mapper):
        idx, fname = entry[0:-1].split(' ')
        idx = int(idx)

        # Skip map frames and already processed query frames
        if marker_query_map[idx] == 'map' or idx in twoview_frame_pairs:
            continue

        # Find all possible connections for this query frame
        connections_to_idx = []
        idx_src_connections = connections_np[connections_np[:, 0] == idx]
        idx_src_connections = list(idx_src_connections)
        for src_idx1, dst_idx2, visibility in idx_src_connections:
            # Consider connections to map frames or already processed query frames
            src_idx1, dst_idx2 = int(src_idx1), int(dst_idx2)
            assert idx == src_idx1
            if (marker_query_map[dst_idx2] == 'map') or (dst_idx2 in twoview_frame_pairs):
                connections_to_idx.append([src_idx1, dst_idx2, visibility])

        # Select the connection with highest visibility
        if connections_to_idx:  # Only proceed if connections exist
            visibility_to_idx = [vis for _, _, vis in connections_to_idx]
            max_visibility_id = np.argmax(np.array(visibility_to_idx))
            best_connection = connections_to_idx[max_visibility_id]
            twoview_frame_pairs[best_connection[0]] = best_connection[1]

    # Verify all query frames have been processed
    assert len(twoview_frame_pairs) == nqry, "Number of processed frames doesn't match query count"

    # Process each query frame to estimate its pose
    for src_idx1 in tqdm.tqdm(list(twoview_frame_pairs.keys())):
        assert marker_query_map[src_idx1] == 'qry', "Source frame must be a query frame"
        dst_idx2 = twoview_frame_pairs[src_idx1]  # Get the paired destination frame index

        # Initialize sampler to get correspondence points between source and destination frames
        sample_num = 5000
        dataset = RandomCorrespondenceDepthSampler(
            h5file_path=h5file_path,
            src_idx1=[src_idx1],
            dst_idx2=[dst_idx2],
            marker_query_map=marker_query_map,
            sample_num=sample_num,
            min_corres_conf=min_corres_conf
        )

        # Get correspondence data between source and destination frames
        data = dataset.__getitem__(0)
        # Combine source and destination points from both directions (1->2 and 2->1)
        pts_src = np.concatenate([data['sampled_src_np_idx1to2'], data['sampled_dst_np_idx2to1']])
        pts_dst = np.concatenate([data['sampled_dst_np_idx1to2'], data['sampled_src_np_idx2to1']])
        depth_src = np.concatenate([data['depth_src_f_np_idx1to2'], data['depth_dst_f_np_idx2to1']])

        # Get frame indices and ground truth pose
        src_idx1 = data['idx1_idx1to2']
        dst_idx2 = data['idx2_idx1to2']
        gt_pose = data['gtposes_src_dst_np_idx1to2']
        gt_pose = pad_poses(gt_pose)  # Ensure pose is in 4x4 format

        # Normalize translation vector of ground truth pose
        gt_pose_norm = copy.deepcopy(gt_pose)
        gt_pose_norm[0:3, 3] = gt_pose_norm[0:3, 3] / np.sqrt(np.sum(gt_pose_norm[0:3, 3] ** 2))

        # Convert numpy arrays to Torch tensors
        pts_src = torch.from_numpy(pts_src)
        pts_dst = torch.from_numpy(pts_dst)
        depth_src = torch.from_numpy(depth_src)

        # Estimate relative pose between source and destination frames
        et_pose_src2dst = two_view_pose_estimation(
            intrinsics[src_idx1],
            intrinsics[dst_idx2],
            pts_src,
            pts_dst,
            depth_src,
            th=0.01,  # RANSAC threshold
        )

        # Normalize translation vector of estimated pose
        et_pose_src2dst_norm = copy.deepcopy(et_pose_src2dst)
        et_pose_src2dst_norm[0:3, 3] = et_pose_src2dst_norm[0:3, 3] / np.sqrt(np.sum(et_pose_src2dst_norm[0:3, 3] ** 2))

        # Compute world-to-camera pose for source frame using estimated relative pose
        poses_w2c_pr_idx = np.linalg.inv(et_pose_src2dst) @ poses_w2c_pr[dst_idx2]
        poses_w2c_pr[src_idx1] = poses_w2c_pr_idx

    # Convert results to numpy arrays and return
    poses_w2c_gt, poses_w2c_pr = np.stack(poses_w2c_gt), np.stack(poses_w2c_pr)
    intrinsics = intrinsics.cpu().numpy()

    return intrinsics, poses_w2c_gt, poses_w2c_pr