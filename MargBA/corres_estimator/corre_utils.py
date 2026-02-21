import copy, os, loguru, time, tqdm, pickle, sys, natsort, shutil, cv2
import glob

import torch
import torch.multiprocessing as mp
import numpy as np
import PIL.Image as Image
from torch.utils.data import DataLoader
from MargBA.geometry import pad_poses
from MargBA.datasets import monodepth2vls, image2vls, HDF5Reader, HDF5Writer

def forward_backward_check(corres_src_dst, corres_dst_src, max_cyclic_err=1e-2):
    pts_src_src2dst, pts_dst_src2dst = torch.split(corres_src_dst, [2, 2], dim=3)
    pts_dst_dst2src, pts_src_dst2src = torch.split(corres_dst_src, [2, 2], dim=3)
    pts_src_resampled = torch.nn.functional.grid_sample(pts_src_dst2src.permute([0, 3, 1, 2]), pts_dst_src2dst, mode="bilinear", align_corners=False)
    pts_src_resampled = pts_src_resampled.permute([0, 2, 3, 1])
    cyclic_err = torch.sqrt(torch.sum((pts_src_src2dst - pts_src_resampled) ** 2, dim=-1) + 1e-10)
    cyclic_err = cyclic_err / max_cyclic_err
    cyclic_err = 1 - cyclic_err
    cyclic_err = torch.clamp_min(cyclic_err, min=0.0)
    return cyclic_err.unsqueeze(1)

def compute_visibility(valid_src_dst, valid_dst_src):
    _, _, h, w = valid_src_dst.shape
    visibility = valid_src_dst.sum(dim=[1, 2, 3]) + valid_dst_src.sum(dim=[1, 2, 3])
    visibility = visibility / h / w / 2
    return visibility

def init_corres_model(corres_method_name, min_confidence=0.25, min_visibility=0.15):
    supported_methods = ['RoMa', 'MASt3R']
    assert corres_method_name in supported_methods
    if corres_method_name == 'RoMa':
        from MargBA.corres_estimator import RoMa
        coores_model = RoMa(min_confidence=min_confidence, min_visibility=min_visibility)
    elif corres_method_name == 'MASt3R':
        from MargBA.corres_estimator import MASt3R
        coores_model = MASt3R(min_confidence=min_confidence, min_visibility=min_visibility)
    return coores_model

def cvt2ncoordinates(torch_coords, intrinsic, h, w):
    # Convert from pytorch coordinates to normalized coordinates
    assert intrinsic.shape[0] == 3 and intrinsic.shape[1] == 3
    fx, fy, bx, by = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    x, y = torch.split(torch_coords, 1, dim=1)
    x = (x + 1) / 2 * w
    y = (y + 1) / 2 * h
    x = (x - bx) / fx
    y = (y - by) / fy
    return torch.cat([x, y], dim=1)

def torchncoords2coordinates(torchncoords, h, w, splitdim=1):
    if isinstance(torchncoords, torch.Tensor):
        x, y = torch.split(torchncoords, 1, dim=splitdim)
        x = (x + 1) / 2 * w
        y = (y + 1) / 2 * h
        return torch.cat([x, y], dim=splitdim)
    elif isinstance(torchncoords, np.ndarray):
        x, y = torchncoords[:, 0], torchncoords[:, 1]
        x = (x + 1) / 2 * w
        y = (y + 1) / 2 * h
        return np.stack([x, y], axis=1)
    else:
        raise NotImplementedError()

def coordinates2ncoordinates(coordinates, intrinsic):
    assert intrinsic.shape[0] == 3 and intrinsic.shape[1] == 3
    fx, fy, bx, by = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    x, y = torch.split(coordinates, 1, dim=1)
    x = (x - bx) / fx
    y = (y - by) / fy
    return torch.cat([x, y], dim=1)

def two_view_pose_estimation(intrinsic1, intrinsic2, pts1, pts2, depth1, th):
    norm_threshold1 = th / intrinsic1[:2, :2].abs().mean().item()
    norm_threshold2 = th / intrinsic2[:2, :2].abs().mean().item()
    norm_threshold = (norm_threshold1 + norm_threshold2) / 2
    npts1 = coordinates2ncoordinates(pts1, intrinsic1)
    npts2 = coordinates2ncoordinates(pts2, intrinsic2)

    npts1, npts2 = npts1.cpu().numpy(), npts2.cpu().numpy()
    E, inliers = cv2.findEssentialMat(npts1, npts2, method=cv2.USAC_MAGSAC, threshold=norm_threshold)
    cheirality_cnt, R, t, _ = cv2.recoverPose(E, npts1, npts2)
    npose, inliers = np.concatenate([R, t], axis=1), inliers[:, 0]

    pts1np, pts2np, mdn = pts1.cpu().numpy(), pts2.cpu().numpy(), depth1.cpu().numpy()
    pose = npose2pose(npts1, npts2, mdn, np.eye(3), npose, inliers)
    return pose

def two_view_adjustment_estimation(intrinsic1, intrinsic2, pts1, pts2, depth1, pose):
    npts1 = coordinates2ncoordinates(pts1, intrinsic1)
    npts2 = coordinates2ncoordinates(pts2, intrinsic2)

    npts1, npts2, mdn, pose = npts1.cpu().numpy(), npts2.cpu().numpy(), depth1.cpu().numpy(), pose.cpu().numpy()
    R, t = pose[0:3, 0:3], pose[0:3, 3:4]
    scale = np.sqrt(np.sum(t ** 2) + 1e-10)
    nt = t / scale
    npose = np.concatenate([R, nt], axis=1)
    inliers = mdn > 0
    newpose = npose2pose(npts1, npts2, mdn, np.eye(3), npose, inliers)
    newscale = np.sqrt(np.sum(newpose[0:3, 3:4] ** 2) + 1e-10)
    adjustment = newscale / scale
    return adjustment

def depth2scale(pts2d1, pts2d2, intrinsic, R, t, coorespondedDepth):
    intrinsic33 = intrinsic[0:3, 0:3]
    M = intrinsic33 @ R @ np.linalg.inv(intrinsic33)
    delta_t = (intrinsic33 @ t).squeeze()
    minval = 1e-6

    denom = (pts2d2[0, :] * (np.expand_dims(M[2, :], axis=0) @ pts2d1).squeeze() - (np.expand_dims(M[0, :], axis=0) @ pts2d1).squeeze()) ** 2 + \
            (pts2d2[1, :] * (np.expand_dims(M[2, :], axis=0) @ pts2d1).squeeze() - (np.expand_dims(M[1, :], axis=0) @ pts2d1).squeeze()) ** 2

    selector = (denom > minval)

    rel_d = np.sqrt(
        ((delta_t[0] - pts2d2[0, selector] * delta_t[2]) ** 2 +
         (delta_t[1] - pts2d2[1, selector] * delta_t[2]) ** 2) / denom[selector])
    coorespondedDepth = coorespondedDepth[selector]
    alpha = np.median(coorespondedDepth / rel_d)
    return alpha

def t2T(t):
    T = np.zeros([3, 3])
    T[0, 1] = -t[2]
    T[0, 2] = t[1]
    T[1, 0] = t[2]
    T[1, 2] = -t[0]
    T[2, 0] = -t[1]
    T[2, 1] = t[0]
    return T

def npose2pose(pts1, pts2, mdn, intrinsic, npose, inliers):
    R, t = npose[0:3, 0:3], npose[0:3, 3:4]

    # Estimate Scale
    inliers_mask = inliers == 1
    pts1_inliers = pts1[inliers_mask, :].T
    pts2_inliers = pts2[inliers_mask, :].T

    pts1_inliers = np.concatenate([pts1_inliers, np.ones([1, pts1_inliers.shape[1]])], axis=0)
    pts2_inliers = np.concatenate([pts2_inliers, np.ones([1, pts2_inliers.shape[1]])], axis=0)
    mdn_inlier = mdn[inliers_mask]

    intrinsic33 = intrinsic[0:3, 0:3]

    epppoint = intrinsic33 @ t
    epppoint[0, 0] = epppoint[0, 0] / epppoint[2, 0]
    epppoint[1, 0] = epppoint[1, 0] / epppoint[2, 0]
    epppoint[2, 0] = 1

    E_init = t2T(t) @ R
    F_init = np.linalg.inv(intrinsic33).T @ E_init @ np.linalg.inv(intrinsic33)
    eppline = F_init @ pts1_inliers

    vec1 = np.stack([-eppline[1, :], eppline[0, :]], axis=0)
    vec2 = pts2_inliers[0:2, :] - epppoint[0:2, :]
    vecp = np.sum(vec1 * vec2, axis=0) / np.sum(vec1[0:2, :] ** 2, axis=0) * vec1
    pts2_inliers_new = vecp + epppoint[0:2, :]

    # Initializtion of camera scale
    scale_md = depth2scale(
        pts1_inliers,
        pts2_inliers_new,
        intrinsic,
        R, t,
        mdn_inlier
    )

    pose = np.eye(4)
    pose[0:3, 0:3] = R
    pose[0:3, 3:4] = scale_md * t
    pose = pose.astype(np.float32)
    return pose

def vls_corres_confidence(rgb_src, rgb_dst, certainty, cyclic_err, min_confidence, viewind):
    from MargBA.datasets import image2vls, monodepth2vls
    rgb_src_vls = image2vls(rgb_src, viewind=viewind)
    rgb_dst_vls = image2vls(rgb_dst, viewind=viewind)
    w, h = rgb_src_vls.size
    certainty_vls = monodepth2vls(certainty, vmax=1.0, viewind=viewind).resize((w, h))
    cyclic_err_vls = monodepth2vls(cyclic_err, vmax=1.0, viewind=viewind).resize((w, h))
    certainty_valid_vls = monodepth2vls(certainty > min_confidence, vmax=1.0, viewind=viewind).resize((w, h))
    cyclic_err_valid_vls = monodepth2vls(cyclic_err > 0, vmax=1.0, viewind=viewind).resize((w, h))
    images_row1 = np.concatenate(
        [np.array(rgb_src_vls), np.array(rgb_dst_vls)], axis=1
    )
    images_row2 = np.concatenate(
        [np.array(certainty_vls), np.array(cyclic_err_vls)], axis=1
    )
    images_row3 = np.concatenate(
        [np.array(certainty_valid_vls), np.array(cyclic_err_valid_vls)], axis=1
    )
    images_vls = np.concatenate(
        [images_row1, images_row2, images_row3], axis=0
    )
    return Image.fromarray(images_vls)

def write_corres(correspng, pairname, foldername):
    folder_path = os.path.join(foldername, pairname)
    os.makedirs(folder_path, exist_ok=True)
    correspng[0].save(os.path.join(folder_path, "{}_x.png".format(pairname)))
    correspng[1].save(os.path.join(folder_path, "{}_y.png".format(pairname)))
    correspng[2].save(os.path.join(folder_path, "{}_conf.png".format(pairname)))

@torch.no_grad()
def inference_pairwise_per_gpu(
        rank,
        world_size,
        data_root,
        dataset,
        output_location,
        corres_method_name,
        min_confidence=0.25, min_visibility=0.15
):
    from MargBA.datasets import initialize_dataset, to_cuda
    torch.cuda.set_device(rank)
    dataset = initialize_dataset(
        data_root=data_root,
        dataset=dataset,
        output_location=output_location,
        mode='correspondence',
        rank=rank,
        world_size=world_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    corres_model = init_corres_model(corres_method_name, min_confidence=min_confidence, min_visibility=min_visibility)
    corres_model.to_cuda(rank)

    visibility_vls_root = os.path.join(output_location, 'vls', 'visibility_vls')
    os.makedirs(visibility_vls_root, exist_ok=True)

    tmp_folder = os.path.join(output_location, 'intermediate')
    pose_i2j_gt_folder = os.path.join(tmp_folder, "pose_i2j_gt")
    corres_i2j_folder = os.path.join(tmp_folder, "corres_i2j")
    visibility_i2j_folder = os.path.join(tmp_folder, "visibility_i2j")
    os.makedirs(corres_i2j_folder, exist_ok=True)
    os.makedirs(visibility_i2j_folder, exist_ok=True)

    loguru.logger.info("GPU-%d inference %d image-pairs with method %s" % (rank, len(dataset), corres_method_name))
    time.sleep(2)
    for (idx, data) in enumerate(tqdm.tqdm(dataloader)):
        # only visibile image pairs are read
        data = to_cuda(data)
        corres_src_dst, corres_dst_src, certainty_src_dst, certainty_dst_src, visibility, valid_src_dst, valid_dst_src = corres_model.inference(data)

        bz = len(corres_src_dst)
        for i in range(bz):
            if visibility[i] < corres_model.min_visibility:
                continue

            idx1, idx2 = data['image_idx_pair'][0][i].item(), data['image_idx_pair'][1][i].item()
            # skip if empty
            if torch.sum(valid_src_dst[i]) == 0 or torch.sum(valid_dst_src[i]) == 0: continue

            # convert to np file
            if True:
                corres_src_dst_png = corres2png(corres_src_dst[i], certainty_src_dst[i].squeeze(), valid_src_dst[i].squeeze())
                corres_dst_src_png = corres2png(corres_dst_src[i], certainty_dst_src[i].squeeze(), valid_dst_src[i].squeeze())
                write_corres(corres_src_dst_png, '%s_%s' % (str(idx1).zfill(6), str(idx2).zfill(6)), corres_i2j_folder)
                write_corres(corres_dst_src_png, '%s_%s' % (str(idx2).zfill(6), str(idx1).zfill(6)), corres_i2j_folder)

            # write gt i2j pose
            if 'pose_src2dst' in data:
                os.makedirs(pose_i2j_gt_folder, exist_ok=True)
                txt_path = os.path.join(pose_i2j_gt_folder, '%s_%s.txt' % (str(idx1).zfill(6), str(idx2).zfill(6)))
                np.savetxt(txt_path, data['pose_src2dst'][i].cpu().numpy())
                txt_path = os.path.join(pose_i2j_gt_folder, '%s_%s.txt' % (str(idx2).zfill(6), str(idx1).zfill(6)))
                np.savetxt(txt_path, np.linalg.inv(pad_poses(data['pose_src2dst'][i].cpu().numpy()))[0:3])

            # write visibility
            if True:
                visibility_path = os.path.join(visibility_i2j_folder, '%s_%s.txt' % (str(idx1).zfill(6), str(idx2).zfill(6)))
                np.savetxt(visibility_path, np.array([visibility[i].item()]))
                visibility_path = os.path.join(visibility_i2j_folder, '%s_%s.txt' % (str(idx2).zfill(6), str(idx1).zfill(6)))
                np.savetxt(visibility_path, np.array([visibility[i].item()]))

            # visualization skipped
            if np.random.randint(50) == -1:
                src_dst_vls = vls_corres_confidence(data['rgb_src'], data['rgb_dst'], certainty_src_dst, certainty_src_dst, corres_model.min_confidence, viewind=i)
                dst_src_vls = vls_corres_confidence(data['rgb_dst'], data['rgb_src'], certainty_dst_src, certainty_dst_src, corres_model.min_confidence, viewind=i)
                all_vls = np.concatenate([np.array(src_dst_vls), np.array(dst_src_vls)], axis=1)
                src_dst_vls_path = os.path.join(visibility_vls_root, '%s_%s_%.1f.jpg' % (str(idx1).zfill(6), str(idx2).zfill(6), visibility[i].item() * 100))
                Image.fromarray(all_vls).save(src_dst_vls_path)
    return

def write_to_hdf5(scene, output_location):
    h5pypath = os.path.join(output_location, f"{scene}.hdf5")
    intermediate_folder = os.path.join(output_location, "intermediate")
    keys_to_add = ["corres_i2j", "visibility_i2j"]
    if os.path.exists(os.path.join(intermediate_folder, "pose_i2j_gt")):
        keys_to_add.append("pose_i2j_gt")
    h5pywriter = HDF5Writer(
        h5pypath,
        keys_to_add=keys_to_add,
        mode="a"
    )

    if "pose_i2j_gt" in keys_to_add:
        pose_i2j_gt_paths = natsort.natsorted(glob.glob(os.path.join(intermediate_folder, "pose_i2j_gt", "*.txt")))
        for pose_i2j_gt_path in pose_i2j_gt_paths:
            h5pywriter.write_pose_i2j_gt(pose_i2j_gt_path, os.path.basename(pose_i2j_gt_path))

    corres_pair_paths = natsort.natsorted(glob.glob(os.path.join(intermediate_folder, "corres_i2j", "*")))
    for corres_pair_path in corres_pair_paths:
        h5pywriter.write_corres_pair(corres_pair_path, os.path.basename(corres_pair_path))

    visibility_pair_paths = natsort.natsorted(glob.glob(os.path.join(intermediate_folder, "visibility_i2j", "*")))
    for visibility_pair_path in visibility_pair_paths:
        h5pywriter.write_visibility(visibility_pair_path, os.path.basename(visibility_pair_path))

    shutil.rmtree(intermediate_folder)

def corres2vls(rgb_src, rgb_dst, coords_src, coords_dst, certaintynp):
    w, h = rgb_src.size
    def coords2cuda(coords):
        return torch.from_numpy(coords).permute([2, 0, 1]).unsqueeze(0).float().cuda()
    def png2cuda(rgb):
        return torch.from_numpy(np.array(rgb)).permute([2, 0, 1]).unsqueeze(0).float().cuda() / 255.0
    
    def certainty2cuda(certainty):
        return torch.from_numpy(certainty).unsqueeze(0).unsqueeze(0).float().cuda()

    def mapimg2white(rgb, certaintynp):
        invalid = certaintynp == 0
        rgb = np.array(rgb)
        rgb[:, :, 0][invalid] = 255
        rgb[:, :, 1][invalid] = 255
        rgb[:, :, 2][invalid] = 255
        return Image.fromarray(rgb)

    coords_src, coords_dst = coords2cuda(coords_src), coords2cuda(coords_dst)
    rgb_src, rgb_dst, certainty = png2cuda(rgb_src), png2cuda(rgb_dst), certainty2cuda(certaintynp)

    rgb_src_sampled = torch.nn.functional.grid_sample(rgb_src, coords_src.permute([0, 2, 3, 1]), mode="bilinear", align_corners=False)
    rgb_dst_sampled = torch.nn.functional.grid_sample(rgb_dst, coords_dst.permute([0, 2, 3, 1]), mode="bilinear", align_corners=False)

    rgb_src_sampled = image2vls(rgb_src_sampled, viewind=0)
    rgb_dst_sampled = image2vls(rgb_dst_sampled, viewind=0)
    rgb_dst_sampled = mapimg2white(rgb_dst_sampled, certaintynp)
    rgb_dst = image2vls(rgb_dst, viewind=0)
    rgb_src_sampled, rgb_dst_sampled, rgb_dst = rgb_src_sampled.resize((w, h)), rgb_dst_sampled.resize((w, h)), rgb_dst.resize((w, h))
    certainty = monodepth2vls(certainty, vmax=1.0, viewind=0).resize((w, h))
    vls_row1 = np.concatenate([
        np.array(rgb_src_sampled), np.array(rgb_dst)
    ], axis=1)
    vls_row2 = np.concatenate([
        np.array(rgb_dst_sampled), np.array(certainty)
    ], axis=1)
    vls = np.concatenate([
        np.array(vls_row1), np.array(vls_row2)
    ], axis=0)
    return Image.fromarray(vls)

def vls_wt_hdf5(scene, output_location):
    h5pypath = os.path.join(output_location, f"{scene}.hdf5")
    h5pyreader = HDF5Reader(
        h5pypath
    )
    all_pairs = h5pyreader.get_all_image_pairs()

    vls_folder = os.path.join(output_location, "vls")
    os.makedirs(vls_folder, exist_ok=True)

    corres_folder = os.path.join(vls_folder, "corres")
    os.makedirs(corres_folder, exist_ok=True)
    for idx, pair_name in enumerate(all_pairs):
        if np.random.randint(len(all_pairs)) < 100:
            # random visualize approximately 100 correspondences
            src_idx1, dst_idx2 = pair_name.split('_')
            corresvls_path = os.path.join(vls_folder, "corres", "{}.jpg".format(pair_name))
            rgb_src, rgb_dst = h5pyreader.read_jpg_image("{}.jpg".format(src_idx1)), h5pyreader.read_jpg_image("{}.jpg".format(dst_idx2))
            coords_src, coords_dst, certainty = h5pyreader.read_corres(pair_name)
            corresvls = corres2vls(rgb_src, rgb_dst, coords_src, coords_dst, certainty)
            corresvls.save(corresvls_path)

def inference_pairwise(
        data_root,
        dataset,
        output_location,
        corres_method_name,
        min_confidence=0.25, min_visibility=0.15
):
    world_size = torch.cuda.device_count()
    mp.spawn(
        inference_pairwise_per_gpu,
        args=(
            world_size,
            data_root,
            dataset,
            output_location,
            corres_method_name,
            min_confidence,
            min_visibility
        ),
        nprocs=world_size,
        join=True
    )
    scene = os.path.basename(data_root)
    write_to_hdf5(
        scene,
        output_location
    )
    vls_wt_hdf5(
        scene,
        output_location
    )


def corres2png(corres, certainty, valid):
    corrdsi, coordsj = torch.split(corres, [2, 2], dim=2)
    coordsjx, coordsjy = torch.split(coordsj, [1, 1], dim=2)
    coordsjx, coordsjy, certainty, valid = coordsjx.squeeze().cpu().numpy(), coordsjy.squeeze().cpu().numpy(), certainty.cpu().numpy(), valid.cpu().numpy()

    def coords2uint16(coords, valid):
        uint16max = 65535.0
        valid = valid * (coords > -1.0) * (coords < 1.0)
        coords = (coords + 1) / 2 * uint16max
        coords[valid == 0] = uint16max
        coords = coords.astype(np.uint16)
        return coords, valid

    def certainty2uint16(certainty, valid):
        certainty = certainty * valid
        certainty = (certainty * 1000).astype(np.uint16)
        return certainty

    coordsjx, validx = coords2uint16(coordsjx, valid)
    coordsjy, validj = coords2uint16(coordsjy, valid)
    certainty = certainty2uint16(certainty, valid * validx * validj)
    return [Image.fromarray(coordsjx), Image.fromarray(coordsjy), Image.fromarray(certainty)]


def png2corres(coordsjx, coordsjy, certainty):
    def png2certainty(certainty):
        return np.array(certainty).astype(np.float32) / 1000

    def png2coords(coords):
        uint16max = 65535.0
        return np.array(coords).astype(np.float32) / uint16max * 2 - 1

    return [png2coords(coordsjx), png2coords(coordsjy), png2certainty(certainty)]
