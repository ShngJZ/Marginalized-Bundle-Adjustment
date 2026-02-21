import struct, kornia, cv2, torch, tqdm, copy
import networkx as nx
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from numba import njit
from torch.utils.data.dataloader import default_collate

from MargBA.datasets import HDF5Reader
from MargBA.corres_estimator import torchncoords2coordinates
from MargBA.geometry import to_homogeneous, from_homogeneous


def custom_collate_fn(batch):
    """
    Custom collate function that uses default_collate but skips None values.

    Args:
        batch: List of dictionaries containing the batch data

    Returns:
        Collated batch with None values handled
    """
    # Filter out None values
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    # Process each key separately to handle None values
    if not isinstance(batch[0], dict):
        return default_collate(batch)

    keys = batch[0].keys()
    result = {}

    for key in keys:
        # Get values for this key, skipping None
        values = [b[key] for b in batch if key in b and b[key] is not None]
        if values:  # Only process if we have non-None values
            result[key] = default_collate(values)

    return result


def input2cuda(np_or_tensor_input, device, dtype=torch.float32):
    # Case 1: Single numpy array
    if isinstance(np_or_tensor_input, np.ndarray):
        tensor = torch.from_numpy(np_or_tensor_input).to(dtype=dtype).to(device)
        return tensor

    # Case 2: Single torch tensor
    if isinstance(np_or_tensor_input, torch.Tensor):
        return np_or_tensor_input.to(dtype=dtype).to(device)

    # Case 3: List of numpy arrays
    if isinstance(np_or_tensor_input, list) and isinstance(np_or_tensor_input[0], np.ndarray):
        stacked = np.stack(np_or_tensor_input, axis=0)
        tensor = torch.from_numpy(stacked).to(dtype=dtype).to(device)
        return tensor

    # Case 4: List of torch tensors
    if isinstance(np_or_tensor_input, list) and isinstance(np_or_tensor_input[0], torch.Tensor):
        tensor = torch.cat([x.to(dtype=dtype) for x in np_or_tensor_input], dim=0).to(device)
        return tensor

    raise TypeError("Input must be a numpy array, torch tensor, or a list of them.")

def prj2vls(rgb_src, rgb_dst, pts1, pts2, pts2_prj, r, sv_path=None):
    rndcolor = np.random.rand(len(pts1), 3)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.scatter(pts1[:, 0], pts1[:, 1], 3, rndcolor)
    ax2.scatter(pts2[:, 0], pts2[:, 1], 3, rndcolor)
    ax2.scatter(pts2_prj[:, 0], pts2_prj[:, 1], 3, c='blue')
    for i in range(len(pts2)):
        ax2.plot([pts2[i, 0], pts2_prj[i, 0]], [pts2[i, 1], pts2_prj[i, 1]], linewidth=0.3, c='blue')
    ax1.imshow(rgb_src)
    ax2.imshow(rgb_dst)
    ax1.axis("off")
    ax2.axis("off")
    plt.savefig(sv_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    Image.open(sv_path).resize((1488, 500)).save(sv_path)

def read_connection_pair(h5path):
    h5pyreader = HDF5Reader(
        h5path
    )
    connection_strs = h5pyreader.get_all_image_pairs()
    connections = list()
    for connection_str in connection_strs:
        idx1, idx2 = connection_str.split('_')
        idx1, idx2 = int(idx1), int(idx2)
        connections.append([idx1, idx2])
    return connections

def read_h5_by_index(h5file_path, src_idx1, dst_idx2, sample_num=5000, use_pointcloud=False, min_corres_conf=0.0):
    h5pyreader = HDF5Reader(
        h5file_path
    )

    assert src_idx1 < dst_idx2

    src_idx1_name, dst_idx2_name = str(src_idx1).zfill(6), str(dst_idx2).zfill(6)
    pair_src2dst = "{}_{}".format(str(src_idx1).zfill(6), str(dst_idx2).zfill(6))
    pair_dst2src = "{}_{}".format(str(dst_idx2).zfill(6), str(src_idx1).zfill(6))

    intrinsic_src, intrinsic_dst = h5pyreader.read_intrinsic_gt(src_idx1_name), h5pyreader.read_intrinsic_gt(dst_idx2_name)
    gtposes_src_dst, gtposes_dst_src = h5pyreader.read_pose_i2j_gt(pair_src2dst), h5pyreader.read_pose_i2j_gt(pair_dst2src)

    coords_src_src2dst, coords_dst_src2dst, certainty_src2dst = h5pyreader.read_corres(pair_src2dst)
    coords_dst_dst2src, coords_src_dst2src, certainty_dst2src = h5pyreader.read_corres(pair_dst2src)

    coords_src = np.concatenate([coords_src_src2dst, coords_src_dst2src], axis=0)
    coords_dst = np.concatenate([coords_dst_src2dst, coords_dst_dst2src], axis=0)
    certainty = np.concatenate([certainty_src2dst, certainty_dst2src], axis=0)

    # Sample Correspondence
    selector = certainty > min_corres_conf
    coords_srcf, coords_dstf, probf = coords_src[selector, :], coords_dst[selector, :], certainty[selector]
    sampled_idx = np.random.choice(
        np.arange(len(probf)),
        size=sample_num,
        replace=True,
        p=probf / np.sum(probf),
    )
    coords_srcf, coords_dstf, probf = coords_srcf[sampled_idx, :], coords_dstf[sampled_idx, :], probf[sampled_idx]

    # Sample Depth
    depth_pr_src, depth_pr_dst = h5pyreader.read_depth_pr(src_idx1_name), h5pyreader.read_depth_pr(dst_idx2_name)
    h, w = depth_pr_src.shape
    depth_src_f = torch.nn.functional.grid_sample(
        torch.from_numpy(depth_pr_src).view([1, 1, h, w]).float().cuda(),
        torch.from_numpy(coords_srcf).view([1, 1, sample_num, 2]).float().cuda(),
        mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()
    h, w = depth_pr_dst.shape
    depth_dst_f = torch.nn.functional.grid_sample(
        torch.from_numpy(depth_pr_dst).view([1, 1, h, w]).float().cuda(),
        torch.from_numpy(coords_dstf).view([1, 1, sample_num, 2]).float().cuda(),
        mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()

    if use_pointcloud:
        incidence_pr_src, incidence_pr_dst = h5pyreader.read_incidence_pr(src_idx1_name), h5pyreader.read_incidence_pr(dst_idx2_name)
        incidence_pr_src_f = torch.nn.functional.grid_sample(
            torch.from_numpy(incidence_pr_src).view([1, 2, h, w]).float().cuda(),
            torch.from_numpy(coords_srcf).view([1, 1, sample_num, 2]).float().cuda(),
            mode="bilinear", align_corners=False
        ).squeeze().T.cpu().numpy()
        incidence_pr_dst_f = torch.nn.functional.grid_sample(
            torch.from_numpy(incidence_pr_dst).view([1, 2, h, w]).float().cuda(),
            torch.from_numpy(coords_dstf).view([1, 1, sample_num, 2]).float().cuda(),
            mode="bilinear", align_corners=False
        ).squeeze().T.cpu().numpy()

    h1, w1 = depth_pr_src.shape
    coords_srcf = torchncoords2coordinates(coords_srcf, h1, w1)
    h2, w2 = depth_pr_dst.shape
    coords_dstf = torchncoords2coordinates(coords_dstf, h2, w2)

    visibility = np.sum(certainty_src2dst > 0) + np.sum(certainty_dst2src > 0)
    visibility = visibility / (h1 * w1 + h2 * w2)

    halfnum = int(sample_num / 2)
    tuple_idx1to2 = {
        'visibility': visibility.item(),
        'sampled_src_np': coords_srcf[0:halfnum],
        'sampled_dst_np': coords_dstf[0:halfnum],
        'depth_src_f_np': depth_src_f[0:halfnum],
        'depth_dst_f_np': depth_dst_f[0:halfnum],
        'intrinsic_src_np': intrinsic_src,
        'intrinsic_dst_np': intrinsic_dst,
        'gtposes_src_dst_np': gtposes_src_dst,
        'gtposes_dst_src_np': gtposes_dst_src,
        'idx1': src_idx1,
        'idx2': dst_idx2
    }
    tuple_idx2to1 = {
        'visibility': visibility.item(),
        'sampled_src_np': coords_dstf[halfnum::],
        'sampled_dst_np': coords_srcf[halfnum::],
        'depth_src_f_np': depth_dst_f[halfnum::],
        'depth_dst_f_np': depth_src_f[halfnum::],
        'intrinsic_src_np': intrinsic_dst,
        'intrinsic_dst_np': intrinsic_src,
        'gtposes_src_dst_np': gtposes_dst_src,
        'gtposes_dst_src_np': gtposes_src_dst,
        'idx1': dst_idx2,
        'idx2': src_idx1
    }

    if use_pointcloud:
        tuple_idx1to2['incidence_src_f_np'] = incidence_pr_src_f[0:halfnum, :]
        tuple_idx1to2['incidence_dst_f_np'] = incidence_pr_dst_f[0:halfnum, :]

        tuple_idx2to1['incidence_src_f_np'] = incidence_pr_dst_f[halfnum::, :]
        tuple_idx2to1['incidence_dst_f_np'] = incidence_pr_src_f[halfnum::, :]

    return tuple_idx1to2, tuple_idx2to1


def register_pose_graph(h5path, connections):
    hdf5reader = HDF5Reader(
        h5path
    )
    pose_graph = nx.Graph()
    for src_idx, dst_idx in connections:
        weight = hdf5reader.read_visibility("{}_{}".format(str(src_idx).zfill(6), str(dst_idx).zfill(6)))
        pose_graph.add_edge(src_idx, dst_idx, weight=weight)
    return pose_graph


@torch.no_grad()
def triangulation_per_index(h5file_path, src_idx, cdf_bundle, cdf_function, connections, poses_model, topk=50000, mincovis=4, use_pointcloud=False):
    h5pyreader = HDF5Reader(
        h5file_path
    )

    src_idx_name = str(src_idx).zfill(6)
    intrinsic = poses_model.get_intrinsic()

    # Read pose
    odometry = poses_model.get_pose_w2c()

    # Read correspondence
    coords_srcs, coords_dsts, certaintys, poses_i2j = list(), list(), list(), list()
    for pair_idx, connection in enumerate(connections):
        src_idx_to_check, dst_idx = connection
        if src_idx_to_check == src_idx:
            pair_src2dst = "{}_{}".format(str(src_idx).zfill(6), str(dst_idx).zfill(6))
            coords_src, coords_dst, certainty = h5pyreader.read_corres(pair_src2dst)

            coords_srcs.append(coords_src)
            coords_dsts.append(coords_dst)
            certaintys.append(certainty)
            poses_i2j.append(odometry[dst_idx] @ odometry[src_idx].inverse())

    bz = len(coords_dsts)
    coords_srcs, coords_dsts, certaintys, poses_i2j = coords_srcs[0], np.stack(coords_dsts, axis=0), np.stack(certaintys, axis=0), torch.stack(poses_i2j, axis=0)
    coords_srcs, coords_dsts, certaintys = torch.from_numpy(coords_srcs).float().cuda().unsqueeze(0), torch.from_numpy(coords_dsts).float().cuda(), torch.from_numpy(certaintys).float().cuda()

    if use_pointcloud:
        incidence = h5pyreader.read_incidence_pr(src_idx_name)
        incidence = torch.from_numpy(incidence).float().cuda().unsqueeze(0)
        incidencef = torch.nn.functional.grid_sample(
            incidence,
            coords_srcs,
            mode="bilinear", align_corners=False
        ).view(1, 2, -1)
    else:
        incidencef = None

    # Acquire Depthmap
    depthmap = h5pyreader.read_depth_pr(src_idx_name)
    h, w = depthmap.shape
    depthmap = torch.from_numpy(depthmap).float().cuda().unsqueeze(0).unsqueeze(0)
    depthmapf = torch.nn.functional.grid_sample(
        depthmap,
        coords_srcs,
        mode="bilinear", align_corners=False
    )
    rgb = np.array(h5pyreader.read_jpg_image("{}.jpg".format(src_idx_name)))
    rgb = torch.from_numpy(rgb).float().cuda().permute([2, 0, 1]).unsqueeze(0)
    rgbf = torch.nn.functional.grid_sample(
        rgb,
        coords_srcs,
        mode="bilinear", align_corners=False
    )
    rgbf = rgbf.view(1, 3, -1)
    certaintys = certaintys.view(bz, -1)
    coords_srcs, coords_dsts = torchncoords2coordinates(coords_srcs, h, w, splitdim=3), torchncoords2coordinates(coords_dsts, h, w, splitdim=3)
    coords_dsts = coords_dsts.view(bz, -1, 2)
    pts3D = poses_model.depth2pts_triangulate(pts_src=coords_srcs.view([1, -1, 2]), depth_src=depthmapf.view(1, -1), intrinsic_src=intrinsic, src_idx1=src_idx, incidence_src=incidencef)

    pts3D_viewj = to_homogeneous(pts3D) @ poses_i2j.transpose(dim0=1, dim1=2)
    pts2D_viewj = from_homogeneous(pts3D_viewj) @ intrinsic.unsqueeze(0).transpose(dim0=1, dim1=2)
    positive_depth = pts2D_viewj[:, :, 2] > 0
    pts2D_viewj = from_homogeneous(pts2D_viewj)
    residual2D = torch.sqrt(torch.sum((pts2D_viewj - coords_dsts) ** 2, dim=-1) + 1e-10)
    weights = positive_depth * (certaintys.view(bz, -1) > 0)

    pdf, cdf, grad_cdf = cdf_bundle
    residuals_cdf, _ = cdf_function.compute_weighted_cdf_forward_backward(residual2D, weights.float(), cdf, grad_cdf)
    residuals_cdf[weights == False] = 1.0
    residuals_cdf = 1 - residuals_cdf
    residuals_cdf = torch.sum(residuals_cdf * weights.float(), dim=0) / (torch.sum(weights.float(), dim=0) + 1e-10)

    _, idx = torch.topk(residuals_cdf, topk)
    valid_from_topk = torch.zeros_like(residuals_cdf) == 1
    valid_from_topk[idx] = True
    valid_from_topk = valid_from_topk * (torch.sum(weights, dim=0) >= mincovis)
    if torch.sum(valid_from_topk) < 5:
        return None, None
    else:
        pts3D_valid = pts3D[0, valid_from_topk]
        pts3D_valid = to_homogeneous(pts3D_valid) @ odometry[src_idx].inverse().transpose(0, 1)
        pts3D_valid = from_homogeneous(pts3D_valid)

        rgbf_valid = rgbf[0, :, valid_from_topk].permute([1, 0])
        return pts3D_valid, rgbf_valid


def write_pointcloud(filename, xyz_points, rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()



@njit
def balanced_sample(viable_selections, reverse_residuals_cdf_bz_debug, large2small, sortedidx, sample_num, sparcity, h, w):
    sampled_idx, sampled_pdf = list(), list()
    cnt = 0
    while True:
        viable_selections_per_loop = np.copy(viable_selections)
        for seqid, cidx in enumerate(sortedidx):
            if cnt >= sample_num:
                break
            if large2small[seqid] < 1e-10:
                break
            if cidx == -1:
                continue
            idh = int(cidx / w)
            idw = int(cidx - idh * w)
            if viable_selections_per_loop[idh, idw]:
                cnt += 1
                sampled_idx.append(cidx)
                sampled_pdf.append(reverse_residuals_cdf_bz_debug[idh, idw])
                ymin, ymax = idh - sparcity, idh + sparcity
                ymin, ymax = max(ymin, 0), min(ymax, h-1)
                xmin, xmax = idw - sparcity, idw + sparcity
                xmin, xmax = max(xmin, 0), min(xmax, w-1)
                for xx in range(xmin, xmax):
                    for yy in range(ymin, ymax):
                        viable_selections_per_loop[yy, xx] = False
                sortedidx[seqid] = -1
        if cnt == sample_num:
            break
    return sampled_idx, sampled_pdf
            

def complete_sampling(sampled_idx, sample_num):
    if len(sampled_idx) < sample_num:
        repeat = [sampled_idx[0]] * (sample_num - len(sampled_idx))
        sampled_idx = sampled_idx + repeat
    return sampled_idx

@torch.no_grad()
def resample(h5file_path, src_idx, cdf_bundle, cdf_function, connections, poses_model, sample_num=5000, sparcity=2, use_pointcloud=False, min_corres_conf=0.0):
    """
    :param h5file_path:
    :param src_idx:
    :param cdf_bundle:
    :param cdf_function:
    :param connections:
    :param poses_model:
    :param sample_num:
    :param sparcity: within sparcity pixel square no other sampling
    :return:
    """
    h5pyreader = HDF5Reader(
        h5file_path
    )

    src_idx_name = str(src_idx).zfill(6)
    intrinsic = poses_model.get_intrinsic()

    # Read pose
    odometry = poses_model.get_pose_w2c()

    # Read correspondence
    coords_srcs, coords_dsts, certaintys, poses_i2j, src_idxs, dst_idxs, depth_dst = list(), list(), list(), list(), list(), list(), list()
    incidence_dst = list()
    for pair_idx, connection in enumerate(connections):
        src_idx_to_check, dst_idx = connection
        if src_idx_to_check == src_idx:
            pair_src2dst = "{}_{}".format(str(src_idx).zfill(6), str(dst_idx).zfill(6))
            coords_src, coords_dst, certainty = h5pyreader.read_corres(pair_src2dst)

            coords_srcs.append(coords_src)
            coords_dsts.append(coords_dst)
            certaintys.append(certainty)
            poses_i2j.append(odometry[dst_idx] @ odometry[src_idx].inverse())

            src_idxs.append(src_idx)
            dst_idxs.append(dst_idx)
            depth_dst.append(h5pyreader.read_depth_pr(str(dst_idx).zfill(6)))

            if use_pointcloud: incidence_dst.append(h5pyreader.read_incidence_pr(str(dst_idx).zfill(6)))

    bz, corresh, corresw = len(coords_dsts), coords_src.shape[0], coords_src.shape[1]
    coords_srcs, coords_dsts, certaintys, poses_i2j, depth_dst = coords_srcs[0], np.stack(coords_dsts, axis=0), np.stack(certaintys, axis=0), torch.stack(poses_i2j, axis=0), np.stack(depth_dst, axis=0)
    coords_srcs, coords_dsts, certaintys, depth_dst = torch.from_numpy(coords_srcs).float().cuda().unsqueeze(0), torch.from_numpy(coords_dsts).float().cuda(), torch.from_numpy(certaintys).float().cuda(), torch.from_numpy(depth_dst).unsqueeze(1).float().cuda()
    if use_pointcloud: incidence_dst = torch.from_numpy(np.stack(incidence_dst, axis=0)).float().cuda()

    # Acquire Depthmap
    depthmap = h5pyreader.read_depth_pr(src_idx_name)
    h, w = depthmap.shape
    depthmap = torch.from_numpy(depthmap).float().cuda().unsqueeze(0).unsqueeze(0)
    depthmapf = torch.nn.functional.grid_sample(
        depthmap,
        coords_srcs,
        mode="bilinear", align_corners=False
    ).view(1, -1)
    depth_dstf = torch.nn.functional.grid_sample(
        depth_dst,
        coords_dsts,
        mode="bilinear", align_corners=False
    ).view(bz, -1)

    if use_pointcloud:
        incidence = h5pyreader.read_incidence_pr(src_idx_name)
        incidence = torch.from_numpy(incidence).float().cuda().unsqueeze(0)
        incidencef = torch.nn.functional.grid_sample(
            incidence,
            coords_srcs,
            mode="bilinear", align_corners=False
        ).view(1, 2, -1)
        incidence_dstf = torch.nn.functional.grid_sample(
            incidence_dst,
            coords_dsts,
            mode="bilinear", align_corners=False
        ).view(bz, 2, -1)
    else:
        incidencef, incidence_dstf = None, None

    certaintys = certaintys.view(bz, -1)
    coords_srcs, coords_dsts = torchncoords2coordinates(coords_srcs, h, w, splitdim=3), torchncoords2coordinates(coords_dsts, h, w, splitdim=3)
    coords_srcs, coords_dsts = coords_srcs.view(1, -1, 2), coords_dsts.view(bz, -1, 2)
    pts3D = poses_model.depth2pts_triangulate(pts_src=coords_srcs, depth_src=depthmapf, intrinsic_src=intrinsic, src_idx1=src_idx, incidence_src=incidencef)
    pts3D_viewj = to_homogeneous(pts3D) @ poses_i2j.transpose(dim0=1, dim1=2)
    pts2D_viewj = from_homogeneous(pts3D_viewj) @ intrinsic.unsqueeze(0).transpose(dim0=1, dim1=2)
    prjdepth = pts2D_viewj[:, :, 2]
    pts2D_viewj = from_homogeneous(pts2D_viewj)

    residual2D = torch.sqrt(torch.sum((pts2D_viewj - coords_dsts) ** 2, dim=-1) + 1e-10)
    weights = (prjdepth > 0) * (certaintys.view(bz, -1) > min_corres_conf)

    pdf, cdf, grad_cdf = cdf_bundle
    residuals_cdf, _ = cdf_function.compute_weighted_cdf_forward_backward(residual2D, weights.float(), cdf, grad_cdf)

    reverse_residuals_cdf = torch.clone(residuals_cdf)
    reverse_residuals_cdf[reverse_residuals_cdf > 1.0] = 1.0
    reverse_residuals_cdf[weights == False] = 1.0
    reverse_residuals_cdf = 1 - reverse_residuals_cdf
    reverse_residuals_cdf = torch.sum(reverse_residuals_cdf * weights.float(), dim=0) / (torch.sum(weights.float(), dim=0) + 1e-10)

    pts_src_s, pts_dst_s, depth_src_s, depth_dst_s, src_idx_s, dst_idx_s = list(), list(), list(), list(), list(), list()
    incidence_src_s, incidence_dst_s = list(), list()
    for i in range(bz):
        visible_pair = (certaintys[i:i+1] > 0).float()
        reverse_residuals_cdf_bz = (reverse_residuals_cdf * visible_pair).squeeze(0)
        viable_selections = (reverse_residuals_cdf_bz.view([corresh, corresw]) > 0)
        if torch.sum(viable_selections) > sample_num + 10:
            large2small, sortedidx = torch.sort(reverse_residuals_cdf_bz, descending=True)

            viable_selections, large2small, sortedidx = viable_selections.cpu().numpy(), large2small.cpu().numpy(), sortedidx.cpu().numpy()
            reverse_residuals_cdf_bz_debug = reverse_residuals_cdf_bz.view([corresh, corresw]).cpu().numpy()
            sampled_idx, sampled_pdf = balanced_sample(viable_selections, reverse_residuals_cdf_bz_debug, large2small, sortedidx, sample_num, sparcity, corresh, corresw)
            pts_src_s.append(coords_srcs[0, sampled_idx])
            pts_dst_s.append(coords_dsts[i, sampled_idx])
            depth_src_s.append(depthmapf[0, sampled_idx])
            depth_dst_s.append(depth_dstf[i, sampled_idx])
            src_idx_s.append(src_idxs[i])
            dst_idx_s.append(dst_idxs[i])
            if use_pointcloud:
                incidence_src_s.append(incidencef[0, :, sampled_idx].T)
                incidence_dst_s.append(incidence_dstf[i, :, sampled_idx].T)
    return pts_src_s, pts_dst_s, depth_src_s, depth_dst_s, src_idx_s, dst_idx_s, incidence_src_s, incidence_dst_s

class RandomCorrespondenceDepthSampler(torch.utils.data.Dataset):
    def __init__(self, h5file_path, src_idx1, dst_idx2, marker_query_map, sample_num, min_corres_conf):
        self.h5pyreader = HDF5Reader(h5file_path)
        self.marker_query_map = marker_query_map
        self.sample_num = sample_num
        self.min_corres_conf = min_corres_conf
        self.src_idx1, self.dst_idx2 = src_idx1, dst_idx2

    def __len__(self):
        return len(self.src_idx1)

    def __getitem__(self, idx):
        src_idx1, dst_idx2 = self.src_idx1[idx], self.dst_idx2[idx]

        h5pyreader, sample_num = self.h5pyreader, self.sample_num

        src_idx1_name, dst_idx2_name = str(src_idx1).zfill(6), str(dst_idx2).zfill(6)
        pair_src2dst = "{}_{}".format(str(src_idx1).zfill(6), str(dst_idx2).zfill(6))
        pair_dst2src = "{}_{}".format(str(dst_idx2).zfill(6), str(src_idx1).zfill(6))

        intrinsic_src, intrinsic_dst = h5pyreader.read_intrinsic_gt(src_idx1_name), h5pyreader.read_intrinsic_gt(dst_idx2_name)
        gtposes_src_dst, gtposes_dst_src = h5pyreader.read_pose_i2j_gt(pair_src2dst), h5pyreader.read_pose_i2j_gt(pair_dst2src)

        coords_src_src2dst, coords_dst_src2dst, certainty_src2dst = h5pyreader.read_corres(pair_src2dst)
        coords_dst_dst2src, coords_src_dst2src, certainty_dst2src = h5pyreader.read_corres(pair_dst2src)

        if self.marker_query_map[src_idx1] == 'qry':
            depth_pr_src = h5pyreader.read_depth_pr(src_idx1_name)
        elif self.marker_query_map[src_idx1] == 'map':
            h, w = certainty_src2dst.shape
            depth_pr_src = h5pyreader.read_depth_gt(src_idx1_name)
            depth_pr_src_non_zero = cv2.resize((depth_pr_src > 0).astype(float), (w, h), interpolation=cv2.INTER_LINEAR)
            certainty_src2dst = (depth_pr_src_non_zero == 1.0).astype(float) * certainty_src2dst
        else:
            raise ValueError("Invalid marker query map for index {}.".format(src_idx1))

        if self.marker_query_map[dst_idx2] == 'qry':
            depth_pr_dst = h5pyreader.read_depth_pr(dst_idx2_name)
        elif self.marker_query_map[dst_idx2] == 'map':
            h, w = certainty_dst2src.shape
            depth_pr_dst = h5pyreader.read_depth_gt(dst_idx2_name)
            depth_pr_dst_non_zero = cv2.resize((depth_pr_dst > 0).astype(float), (w, h), interpolation=cv2.INTER_LINEAR)
            certainty_dst2src = (depth_pr_dst_non_zero == 1.0).astype(float) * certainty_dst2src
        else:
            raise ValueError("Invalid marker query map for index {}.".format(dst_idx2))

        coords_src = np.concatenate([coords_src_src2dst, coords_src_dst2src], axis=0)
        coords_dst = np.concatenate([coords_dst_src2dst, coords_dst_dst2src], axis=0)
        certainty = np.concatenate([certainty_src2dst, certainty_dst2src], axis=0)

        # Sample Correspondence
        selector = certainty > self.min_corres_conf
        coords_srcf, coords_dstf, probf = coords_src[selector, :], coords_dst[selector, :], certainty[selector]
        sampled_idx = np.random.choice(
            np.arange(len(probf)),
            size=sample_num,
            replace=True,
            p=probf / np.sum(probf),
        )
        coords_srcf, coords_dstf, probf = coords_srcf[sampled_idx, :], coords_dstf[sampled_idx, :], probf[sampled_idx]

        # Sample Depth
        h1, w1 = depth_pr_src.shape
        depth_src_f = torch.nn.functional.grid_sample(
            torch.from_numpy(depth_pr_src).view([1, 1, h1, w1]).float(),
            torch.from_numpy(coords_srcf).view([1, 1, sample_num, 2]).float(),
            mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        coords_srcf = torchncoords2coordinates(coords_srcf, h1, w1)
        
        h2, w2 = depth_pr_dst.shape
        depth_dst_f = torch.nn.functional.grid_sample(
            torch.from_numpy(depth_pr_dst).view([1, 1, h2, w2]).float(),
            torch.from_numpy(coords_dstf).view([1, 1, sample_num, 2]).float(),
            mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        coords_dstf = torchncoords2coordinates(coords_dstf, h2, w2)

        visibility = np.sum(certainty_src2dst > 0) + np.sum(certainty_dst2src > 0)
        visibility = visibility / (h1 * w1 + h2 * w2)

        halfnum = int(sample_num / 2)
        result = {
            'visibility_idx1to2': visibility.item(),
            'sampled_src_np_idx1to2': coords_srcf[0:halfnum],
            'sampled_dst_np_idx1to2': coords_dstf[0:halfnum],
            'depth_src_f_np_idx1to2': depth_src_f[0:halfnum],
            'depth_dst_f_np_idx1to2': depth_dst_f[0:halfnum],
            'intrinsic_src_np_idx1to2': intrinsic_src,
            'intrinsic_dst_np_idx1to2': intrinsic_dst,
            'gtposes_src_dst_np_idx1to2': gtposes_src_dst,
            'gtposes_dst_src_np_idx1to2': gtposes_dst_src,
            'idx1_idx1to2': src_idx1,
            'idx2_idx1to2': dst_idx2,
            'visibility_idx2to1': visibility.item(),
            'sampled_src_np_idx2to1': coords_dstf[halfnum::],
            'sampled_dst_np_idx2to1': coords_srcf[halfnum::],
            'depth_src_f_np_idx2to1': depth_dst_f[halfnum::],
            'depth_dst_f_np_idx2to1': depth_src_f[halfnum::],
            'intrinsic_src_np_idx2to1': intrinsic_dst,
            'intrinsic_dst_np_idx2to1': intrinsic_src,
            'gtposes_src_dst_np_idx2to1': gtposes_dst_src,
            'gtposes_dst_src_np_idx2to1': gtposes_src_dst,
            'idx1_idx2to1': dst_idx2,
            'idx2_idx2to1': src_idx1
        }
        return result

def random_sample_correspondence_depth(
        h5path,
        marker_query_map,
        connections,
        sample_num,
        min_corres_conf,
        device,
):
    dataset = RandomCorrespondenceDepthSampler(
        h5file_path=h5path,
        src_idx1=[x[0] for x in connections],
        dst_idx2=[x[1] for x in connections],
        marker_query_map=marker_query_map,
        sample_num=sample_num,
        min_corres_conf=min_corres_conf
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        drop_last=False,
        collate_fn=custom_collate_fn  # Add custom collate function
    )

    pts_src, pts_dst = list(), list()
    depth_src, depth_dst = list(), list()
    src_idx1, dst_idx2 = list(), list()
    visibility = list()
    for (idx, data) in enumerate(tqdm.tqdm(dataloader, desc="Downsample dense pair-wise data")):
        data_ = copy.deepcopy(data)
        for key in [
            'idx1to2',
            'idx2to1'
        ]:
            pts_src.append(data_[f'sampled_src_np_{key}'])
            pts_dst.append(data_[f'sampled_dst_np_{key}'])
            depth_src.append(data_[f'depth_src_f_np_{key}'])
            depth_dst.append(data_[f'depth_dst_f_np_{key}'])
            src_idx1.append(data_[f'idx1_{key}'])
            dst_idx2.append(data_[f'idx2_{key}'])
            visibility.append(data_[f'visibility_{key}'])

    pts_src = input2cuda(pts_src, device)
    pts_dst = input2cuda(pts_dst, device)
    depth_src = input2cuda(depth_src, device)
    depth_dst = input2cuda(depth_dst, device)
    pts_src, pts_dst = kornia.geometry.conversions.convert_points_to_homogeneous(pts_src), kornia.geometry.conversions.convert_points_to_homogeneous(pts_dst)
    src_idx1, dst_idx2 = input2cuda(src_idx1, device).int(), input2cuda(dst_idx2, device).int()
    visibility = input2cuda(visibility, device)
    return pts_src, pts_dst, depth_src, depth_dst, src_idx1, dst_idx2, visibility
