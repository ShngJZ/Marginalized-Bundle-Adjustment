import torch
import numpy as np
import networkx as nx
from typing import Union

def to_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError

def from_homogeneous(points: Union[torch.Tensor, np.ndarray], eps=1e-6):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + eps)

def pad_poses_(p: torch.Tensor) -> torch.Tensor:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = torch.broadcast_to(torch.tensor([0, 0, 0, 1.0], device=p.device,
                              dtype=p.dtype),
                              p[..., :1, :4].shape)
    return torch.cat((p[..., :3, :4], bottom), dim=-2)

def pad_poses(p):
    if isinstance(p, torch.Tensor):
      return pad_poses_(p)
    elif isinstance(p, np.ndarray):
      return pad_poses_(torch.from_numpy(p)).numpy()
    else:
      raise NotImplementedError()

def pad_intrinsic44(intrinsic):
    bz, device = len(intrinsic), intrinsic.device
    intrinsic44 = torch.eye(4, device=device)
    intrinsic44 = intrinsic44.repeat([bz, 1, 1])
    intrinsic44[:, 0:3, 0:3] = intrinsic
    return intrinsic44

def unpad_poses_(p: torch.Tensor) -> torch.Tensor:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def unpad_poses(p):
    if isinstance(p, torch.Tensor):
      return unpad_poses_(p)
    elif isinstance(p, np.ndarray):
      return unpad_poses_(torch.from_numpy(p)).numpy()
    else:
      raise NotImplementedError()

def compute_odometry_l2_error(gt_poses, est_poses):
    error = gt_poses[:, 0:3, 3] - est_poses[:, 0:3, 3]
    error = np.sum(error ** 2, axis=1)
    error = np.mean(np.sqrt(error))
    return error

def angle_error_mat(R1, R2):
    R1T_dot_R2 = np.transpose(R1, [0, 2, 1]) @ R2
    trace = R1T_dot_R2[:, 0, 0] + R1T_dot_R2[:, 1, 1] + R1T_dot_R2[:, 2, 2]
    cos = (trace - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    return np.rad2deg(np.abs(np.arccos(cos)))

def angle_error_vec(v1, v2):
    nv1 = np.sqrt((v1 ** 2).sum(-1))
    nv2 = np.sqrt((v2 ** 2).sum(-1))
    n = np.squeeze(nv1 * nv2)
    return np.rad2deg(np.arccos(np.clip(np.sum(v1 * v2, axis=1) / n, -1.0, 1.0)))

def compute_odometry_angle_error(T_0to1_gt, T_0to1_est):
    # Compute Relative Pose Error
    R_gt, t_gt = T_0to1_gt[:, 0:3, 0:3], T_0to1_gt[:, 0:3, 3]
    R_est, t_est = T_0to1_est[:, 0:3, 0:3], T_0to1_est[:, 0:3, 3]
    error_t = angle_error_vec(t_est + 1e-6, t_gt + 1e-6)
    error_t = np.minimum(error_t, 180 - error_t)
    error_R = angle_error_mat(R_est, R_gt)
    return np.mean(error_t + error_R)

def spantree2odometry_oracle(spantree, connections_npose, connections_gtpose):
    frmnum = max(spantree.get_nodes()) + 1
    et_poses, gt_poses = np.zeros([frmnum, 4, 4]), np.zeros([frmnum, 4, 4])
    et_poses[spantree.root_idx] = np.eye(4)
    gt_poses[spantree.root_idx] = np.eye(4)

    for src_id, dst_id in nx.bfs_edges(spantree.nxgraph, spantree.root_idx):
        edge_idx = spantree.get_edge_idx(src_id, dst_id)
        gt_rel_pose = pad_poses(connections_gtpose["{}_{}".format(str(src_id).zfill(6), str(dst_id).zfill(6))])

        et_rel_pose = pad_poses(connections_npose["{}_{}".format(str(src_id).zfill(6), str(dst_id).zfill(6))][edge_idx])
        et_rel_pose[0:3, 3] = gt_rel_pose[0:3, 3]

        gt_poses[dst_id] = gt_rel_pose @ gt_poses[src_id]
        et_poses[dst_id] = et_rel_pose @ et_poses[src_id]

    gt_poses, et_poses = np.linalg.inv(gt_poses), np.linalg.inv(et_poses)
    return gt_poses, et_poses

def oracle_evaluate_spantree(
        spantree,
        connections_npose,
        connections_gtpose,
        id1s,
        id2s,
        measurement
):
    gt_poses, et_poses = spantree2odometry_oracle(spantree, connections_npose, connections_gtpose)
    rel_gt = np.linalg.inv(gt_poses[id2s]) @ gt_poses[id1s]
    rel_et = np.linalg.inv(et_poses[id2s]) @ et_poses[id1s]
    if measurement == 'angle':
        error = compute_odometry_angle_error(rel_gt, rel_et)
    elif measurement == 'l2':
        error = compute_odometry_angle_error(rel_gt, rel_et)
    else:
        raise NotImplementedError()
    return error

def nt2angular(nt):
    # Convert normalized translation vector to angular
    ntx, nty, ntz = torch.split(nt, [1, 1, 1], dim=1)
    ntx, nty, ntz = ntx.squeeze(), nty.squeeze(), ntz.squeeze()
    theta, phi = torch.arctan2(nty, ntx), torch.arctan2(ntz, torch.sqrt(ntx ** 2 + nty ** 2))
    return theta, phi

def angular2nt(angular):
    if angular.ndim == 2:
        return angular2nt_ndim2(angular)
    elif angular.ndim == 3:
        return angular2nt_ndim3(angular)

def angular2nt_ndim2(angular):
    assert angular.ndim == 2
    nbz = len(angular)
    theta, phi = torch.split(angular, [1, 1], dim=1)
    theta, phi = theta.view([nbz]), phi.view([nbz])
    ntz, ntxy = torch.sin(phi), torch.cos(phi)
    ntx, nty = ntxy * torch.cos(theta), ntxy * torch.sin(theta)
    return torch.stack([ntx, nty, ntz], dim=1).view([nbz, 3, 1])

def angular2nt_ndim3(angular):
    # Keep Batch Shape without view()
    assert angular.ndim == 3
    theta, phi = torch.split(angular, [1, 1], dim=2)
    ntz, ntxy = torch.sin(phi), torch.cos(phi)
    ntx, nty = ntxy * torch.cos(theta), ntxy * torch.sin(theta)
    return torch.concat([ntx, nty, ntz], dim=2)