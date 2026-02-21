import os, loguru, tqdm, glob, random, json

import kornia
import natsort
import torch
import numpy as np
import networkx as nx
from MargBA.geometry import to_homogeneous, from_homogeneous, CDFLossCupy, CDFLossIndexCupy, pad_poses
from MargBA.poses_models import LocalOptimizationPoseParameters, GlobalOptimizationPoseParameters
from MargBA.datasets import HDF5Reader
from MargBA.BA.transform_utils import input2cuda, random_sample_correspondence_depth, read_h5_by_index
from MargBA.corres_estimator import two_view_pose_estimation, two_view_adjustment_estimation
from MargBA.datasets import initialize_dataset, to_cuda


def find_root_node_from_correspondences(src_indices1, dst_indices2, visibility):
    """
    Find the root node (frame with highest weighted degree) from correspondence data.
    
    Args:
        src_indices1: Source frame indices from correspondences
        dst_indices2: Destination frame indices from correspondences  
        visibility: Visibility weights for each correspondence
        largescene: Whether to use large scene mode (different weighting strategy)
        
    Returns:
        rootid: Index of the frame with highest weighted degree
    """
    # For regular scenes, use weighted degree based on visibility scores
    frame_weights = {}

    # Accumulate weighted connections for each frame
    for i in range(len(src_indices1)):
        src_idx = src_indices1[i].item()
        dst_idx = dst_indices2[i].item()
        weight = visibility[i].item()

        # Add weight to source frame
        if src_idx not in frame_weights:
            frame_weights[src_idx] = 0.0
        frame_weights[src_idx] += weight

        # Add weight to destination frame
        if dst_idx not in frame_weights:
            frame_weights[dst_idx] = 0.0
        frame_weights[dst_idx] += weight

    # Find frame with maximum weighted degree
    rootid = max(frame_weights.items(), key=lambda x: x[1])[0]
    
    return rootid


class ReadVisibility(torch.utils.data.Dataset):
    """Dataset class for reading visibility connections between frames from HDF5 file."""

    def __init__(self, h5file_path):
        """
        Args:
            h5pyreader: HDF5Reader instance containing visibility data
        """
        self.h5pyreader = HDF5Reader(h5file_path)
        self.connections = []

        # Read all visibility connections from HDF5 file
        for connection_str in self.h5pyreader.get_all_image_pairs():
            idx1, idx2 = connection_str.split('_')
            self.connections.append([
                int(idx1),
                int(idx2),
                connection_str
            ])

    def __len__(self):
        """Returns total number of visibility connections."""
        return len(self.connections)

    def __getitem__(self, idx):
        """Returns visibility data for a single connection.

        Args:
            idx: Index of the connection to retrieve

        Returns:
            List containing [source_idx, target_idx, visibility_score]
        """
        idx1, idx2, connection_str = self.connections[idx]
        visibility = np.array(self.h5pyreader.read_visibility(connection_str)).item()
        return [idx1, idx2, visibility]

def init_pose_graph(h5file_path):
    dataset = ReadVisibility(h5file_path)

    # Create dataloader for parallel processing
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=32,
        drop_last=False
    )

    pose_graph = nx.Graph()
    for (idx, data) in enumerate(tqdm.tqdm(dataloader, desc="Building pose graph from visibility data")):
        src_idx, dst_idx = data[0].item(), data[1].item()
        src_idx, dst_idx = int(src_idx), int(dst_idx)
        pose_graph.add_edge(src_idx, dst_idx, weight=data[2].item())

    return pose_graph

@torch.no_grad()
def twoview_sfm_initialization(h5file_path, params, hasgt=True, largescene=False):
    hdf5reader = HDF5Reader(h5file_path)
    img_names = hdf5reader.get_all_images()
    nfrm = params["nfrm"]

    # Get all image pairs from HDF5 file and parse into list of [src_idx, dst_idx] pairs
    connections = hdf5reader.get_all_image_pairs()
    connections = [[int(x.split('_')[0]), int(x.split('_')[1])] for x in connections]

    if largescene:
        # Log original connections info
        total_connections_before = len(connections)
        loguru.logger.info(f"Large scene mode: Original connections count: {total_connections_before}")

        # Count connections per frame
        frame_connections = {}
        for i, [src, dst] in enumerate(connections):
            if src not in frame_connections:
                frame_connections[src] = []
            if dst not in frame_connections:
                frame_connections[dst] = []
            
            frame_connections[src].append(i)  # Store connection index instead of the connection itself
            frame_connections[dst].append(i)
        
        # Limit connections per frame to max_connect (50)
        max_connect = 25
        connection_indices_to_keep = set()

        for frame, conn_indices in frame_connections.items():
            if len(conn_indices) > max_connect:
                # Randomly select max_connect connections
                selected_indices = random.sample(conn_indices, max_connect)
                connection_indices_to_keep.update(selected_indices)
            else:
                connection_indices_to_keep.update(conn_indices)
        
        # Create new connections list with only the selected connections
        filtered_connections = [connections[i] for i in sorted(connection_indices_to_keep)]
        
        # Update connections with filtered list
        connections = filtered_connections
        
        # Log results
        loguru.logger.info(f"Large scene mode: Filtered connections count: {len(connections)}")

    # Initialize device as CPU for tensor operations
    device = torch.device("cuda:0")

    pts_src, pts_dst, depth_src, depth_dst, src_indices1, dst_indices2, visibility = random_sample_correspondence_depth(
        h5file_path,
        ['qry'] * nfrm,
        connections,
        params["sample_num"],
        min_corres_conf=params["min_corres_conf"],
        device=device
    )

    if largescene:
        rootid = find_root_node_from_correspondences(src_indices1, dst_indices2, visibility)
    else:
        pose_graph = init_pose_graph(h5file_path)
        nodes_l2s = sorted(pose_graph.degree(weight='weight'), key=lambda x: x[1], reverse=True)
        rootid, _ = nodes_l2s[0]

    # Read ground truth and predicted camera parameters for all images
    intrinsics_gt = np.array([hdf5reader.read_intrinsic_gt(img_name) for img_name in img_names])
    poses_w2c_gt = np.array([hdf5reader.read_pose_w2c_gt(img_name) for img_name in img_names])
    intrinsics_pr = np.array([hdf5reader.read_intrinsic_et(img_name) for img_name in img_names])

    if params['calibrated']:
        intrinsics_pr = intrinsics_gt

    if params['sharefocal']:
        # Calculate median focal lengths and principal points from predicted intrinsics
        focal_x = np.median(intrinsics_pr[:, 0, 0])
        focal_y = np.median(intrinsics_pr[:, 1, 1])
        avg_focal = (focal_x + focal_y) / 2
        principal_x = np.median(intrinsics_pr[:, 0, 2])
        principal_y = np.median(intrinsics_pr[:, 1, 2])

        # Create shared camera intrinsics matrix for all frames
        intrinsics_pr = np.array([np.eye(3)] * nfrm)
        intrinsics_pr[:, 0, 0] = avg_focal  # Set x focal length
        intrinsics_pr[:, 1, 1] = avg_focal  # Set y focal length
        intrinsics_pr[:, 0, 2] = principal_x  # Set x principal point
        intrinsics_pr[:, 1, 2] = principal_y  # Set y principal point
    else:
        intrinsics_pr = intrinsics_pr

    # initialize groundtruth model
    gt_pose_model = None
    if hasgt:
        gt_pose_model = GlobalOptimizationPoseParameters(
            nfrm=nfrm,
            optimizefocal=False,
            sharefocal=params['sharefocal'],
            intrinsic=intrinsics_gt[0] if params['sharefocal'] else intrinsics_gt
        )
        for i in range(nfrm):
            gt_pose_model.update_pose_adjustment(
                nodeid=i,
                newpose=torch.from_numpy(poses_w2c_gt[i]).float(),
                newadjustment=1.0,
                newbias=0.0
            )
            
    # initialize predicted model
    pr_pose_model = GlobalOptimizationPoseParameters(
        nfrm=nfrm,
        optimizefocal=(not params['calibrated']),
        sharefocal=params['sharefocal'],
        intrinsic=intrinsics_pr[0] if params['sharefocal'] else intrinsics_pr
    )
    pr_pose_model = pr_pose_model.to(device)

    nodes_in = [rootid]
    nodes_al = set([int(x) for x in img_names])

    pairid2index = dict()
    for idx in range(len(src_indices1)):
        src_idx1, dst_idx2 = src_indices1[idx].item(), dst_indices2[idx].item()
        pairid2index[src_idx1, dst_idx2] = idx

    intrinsics_pr = input2cuda(intrinsics_pr, device)
    # Register new frames sequentially until all frames are processed
    for frame_idx in tqdm.tqdm(range(1, nfrm), desc="Registering frames"):
        register_new_frame_maximize_degree(
            nodes_in=nodes_in,
            nodes_al=nodes_al,
            visibility=visibility,
            intrinsics_pr=intrinsics_pr,
            src_idx1=src_indices1,
            dst_idx2=dst_indices2,
            pts_src=pts_src,
            pts_dst=pts_dst,
            depth_src=depth_src,
            depth_dst=depth_dst,
            pairid2index=pairid2index,
            pr_pose_model=pr_pose_model,
            twoview_ransac_th=params['twoview_ransac_th'],
            device=device
        )
    return pr_pose_model, gt_pose_model

def register_new_frame_maximize_degree(
        nodes_in,
        nodes_al,
        visibility,
        intrinsics_pr,
        src_idx1,
        dst_idx2,
        pts_src,
        pts_dst,
        depth_src,
        depth_dst,
        pairid2index,
        pr_pose_model,
        twoview_ransac_th,
        device
):
    # Get unregistered nodes (nodes not yet in the graph)
    unregistered_nodes = list(nodes_al.difference(set(nodes_in)))
    intrinsic_src = intrinsics_pr[src_idx1]
    intrinsic_dst = intrinsics_pr[dst_idx2]

    # Create masks for source and target nodes that are already registered
    src_registered_mask = torch.zeros_like(src_idx1)
    dst_registered_mask = torch.zeros_like(dst_idx2)
    for registered_node in nodes_in:
        src_registered_mask[src_idx1 == registered_node] = 1
        dst_registered_mask[dst_idx2 == registered_node] = 1

    # Calculate base degree (connections between already registered nodes)
    max_degree = 0
    best_new_node = None

    # Find the unregistered node that maximizes connections to registered nodes
    for candidate_node in unregistered_nodes:
        # Temporarily add candidate node to registered masks
        src_registered_mask[src_idx1 == candidate_node] = 1
        dst_registered_mask[dst_idx2 == candidate_node] = 1

        # Calculate degree with candidate node included
        current_degree = torch.sum(visibility * src_registered_mask * dst_registered_mask)

        # Remove candidate node from masks
        src_registered_mask[src_idx1 == candidate_node] = 0
        dst_registered_mask[dst_idx2 == candidate_node] = 0

        # Track node with maximum degree increase
        if current_degree > max_degree:
            max_degree = current_degree
            best_new_node = candidate_node

    # Initialize local graph optimizer for new node registration
    local_graph_optimizer = LocalOptimizationPoseParameters()

    # register local graph
    # Get current poses and depth adjustments from pose model
    poses_w2c = pr_pose_model.get_pose_w2c()
    depth_adjustments, _, _, _ = pr_pose_model.get_depth_adjustment()

    # Register new-to-old and old-to-new connections in local optimizer
    for registered_id in nodes_in:
        # Register new-to-old connection if exists
        if (best_new_node, registered_id) in pairid2index:
            pair_idx = pairid2index[(best_new_node, registered_id)]
            registered_pose = poses_w2c[registered_id]
            local_graph_optimizer.register_new2old(
                registered_pose,
                pts_src[pair_idx],
                pts_dst[pair_idx],
                depth_src[pair_idx],
                intrinsic_src[pair_idx]
            )

        # Register old-to-new connection if exists
        if (registered_id, best_new_node) in pairid2index:
            pair_idx = pairid2index[(registered_id, best_new_node)]
            registered_pose = poses_w2c[registered_id]
            local_graph_optimizer.register_old2new(
                registered_pose,
                pts_src[pair_idx],
                pts_dst[pair_idx],
                depth_src[pair_idx] * depth_adjustments[registered_id],
                intrinsic_dst[pair_idx]
            )

    # it will pop up error if it is empty
    succeed = local_graph_optimizer.organize()
    optimal_new_pose_w2c, optimal_adjustment = torch.eye(4), 1.0 # default
    minmr = 1e10
    nodes_in.append(best_new_node)

    if not succeed:
        return

    for old_node_id in nodes_in:
        if (best_new_node, old_node_id) in pairid2index:
            pair_idx = pairid2index[(best_new_node, old_node_id)]

            try:
                # Estimate relative pose from new node to old node
                new_to_old_pose = two_view_pose_estimation(
                    intrinsic1=intrinsic_src[pair_idx],
                    intrinsic2=intrinsic_dst[pair_idx],
                    pts1=from_homogeneous(pts_src[pair_idx]),
                    pts2=from_homogeneous(pts_dst[pair_idx]),
                    depth1=depth_src[pair_idx],
                    th=twoview_ransac_th
                )
                new_to_old_pose = input2cuda(new_to_old_pose, device)

                # Estimate depth adjustment from old node to new node
                pair_idx = pairid2index[(old_node_id, best_new_node)]
                old_to_new_pose = torch.inverse(new_to_old_pose)
                adjustment = two_view_adjustment_estimation(
                    intrinsic1=intrinsic_src[pair_idx],
                    intrinsic2=intrinsic_dst[pair_idx],
                    pts1=from_homogeneous(pts_src[pair_idx]),
                    pts2=from_homogeneous(pts_dst[pair_idx]),
                    depth1=depth_src[pair_idx] * depth_adjustments[old_node_id],
                    pose=old_to_new_pose
                )

                # Scale translation components by adjustment factor
                new_to_old_pose[0:3, 3:4] = new_to_old_pose[0:3, 3:4] * adjustment
                old_to_new_pose[0:3, 3:4] = old_to_new_pose[0:3, 3:4] * adjustment
                assert (torch.isnan(new_to_old_pose).sum() == 0) and (torch.isnan(old_to_new_pose).sum() == 0)
            except:
                new_to_old_pose = torch.eye(4).to(device)
                old_to_new_pose = torch.inverse(new_to_old_pose)
                adjustment = 1.0
                print("Camera pose initialization failed, using identity pose with unit translation")

            # Optimize pose and compute residuals
            old_pose_w2c = poses_w2c[old_node_id]
            new_pose_w2c, residuals_new_to_old = local_graph_optimizer.optimize(
                old_pose_w2c, new_to_old_pose, adjustment, mode="new2old"
            )
            new_pose_w2c_check, residuals_old_to_new = local_graph_optimizer.optimize(
                old_pose_w2c, old_to_new_pose, mode="old2new"
            )

            # Calculate median residuals
            median_new_to_old, _ = torch.median(residuals_new_to_old, dim=1)
            median_old_to_new, _ = torch.median(residuals_old_to_new, dim=1)
            mean_residual = torch.cat([median_new_to_old.flatten(), median_old_to_new.flatten()]).mean()

            # Update optimal pose if current residual is smaller
            if mean_residual < minmr:
                minmr = mean_residual.item()
                optimal_adjustment = adjustment
                optimal_new_pose_w2c = new_pose_w2c

    pr_pose_model.update_pose_adjustment(
        nodeid=best_new_node,
        newpose=optimal_new_pose_w2c,
        newadjustment=optimal_adjustment,
        newbias=0.0
    )
