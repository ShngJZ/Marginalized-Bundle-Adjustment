import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from MargBA.geometry import to_homogeneous, from_homogeneous, pad_poses, unpad_poses

def monodepth2vls(monodepth, vmax=0.18, percentile=None, viewind=0):
    cm = plt.get_cmap('magma')
    if isinstance(monodepth, torch.Tensor):
        if monodepth.ndim == 3:
            monodepth = monodepth.unsqueeze(1)
        monodepth = monodepth[viewind, 0, :, :].detach().cpu().numpy()
    if percentile is not None:
        if np.sum(monodepth > 0) > 100:
            vmax = np.percentile(monodepth[monodepth > 0], 95)
        else:
            vmax = 1.0
    monodepth = monodepth / vmax
    monodepth = (cm(monodepth) * 255).astype(np.uint8)
    return Image.fromarray(monodepth[:, :, 0:3])

def image2vls(image, viewind=0):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().permute([0, 2, 3, 1]).contiguous()[viewind, :, :, :].numpy()
    if np.max(image) <= 1.0001:
        image = image * 255.0
    image = np.clip(image, a_min=0, a_max=255).astype(np.uint8)
    return Image.fromarray(image)

def corres2vls(rgb_src, rgb_dst, corres, certainty, viewind=0):
    # rgb_src_jpg = image2vls(rgb_src, viewind=viewind)
    # rgb_dst_jpg = image2vls(rgb_dst, viewind=viewind)
    _, _, h, w = rgb_src.shape
    corres_src, corres_dst = torch.split(corres, [2, 2], dim=-1)
    rgb_src_sampled = F.grid_sample(rgb_src, corres_src, mode="bilinear", align_corners=False)
    rgb_dst_ref = F.grid_sample(rgb_dst, corres_src, mode="bilinear", align_corners=False)
    rgb_dst_sampled = F.grid_sample(rgb_dst, corres_dst, mode="bilinear", align_corners=False)
    invalid = certainty < 1e-3
    invalid = invalid.expand([-1, 3, -1, -1])
    rgb_dst_sampled[invalid] = 1.0

    rgb_src_sampled = image2vls(rgb_src_sampled, viewind=viewind).resize((w, h))
    rgb_dst_ref = image2vls(rgb_dst_ref, viewind=viewind).resize((w, h))
    rgb_dst_sampled = image2vls(rgb_dst_sampled, viewind=viewind).resize((w, h))
    corresvls = np.concatenate([np.array(rgb_src_sampled), np.array(rgb_dst_sampled), np.array(rgb_dst_ref)], axis=1)
    corresvls = Image.fromarray(corresvls)
    return corresvls

def tuple2vls(rgb_src, rgb_dst, intrinsic_src, intrinsic_dst, pose, sampled_src, sampled_dst, depth_src_f, depth_dst_f, nlim=20, sv_path=None):
    if nlim is not None:
        # sampled_src = sampled_src[0:nlim, :]
        # sampled_dst = sampled_dst[0:nlim, :]
        # depth_src_f = depth_src_f[0:nlim]
        # depth_dst_f = depth_dst_f[0:nlim]
        sampled_src = sampled_src[4623:4625, :]
        sampled_dst = sampled_dst[4623:4625, :]
        depth_src_f = depth_src_f[4623:4625]
        depth_dst_f = depth_dst_f[4623:4625]
        nlim = 2
    else:
        nlim = len(depth_src_f)

    rndcolor = np.random.rand(nlim, 3)

    pts3d_dst = to_homogeneous(sampled_src) * depth_src_f.view([nlim, 1]) @ torch.inverse(intrinsic_src).T
    pts3d_dst = to_homogeneous(pts3d_dst) @ pose.T @ intrinsic_dst.T
    pts2d_dst = from_homogeneous(pts3d_dst)

    pts3d_src = to_homogeneous(sampled_dst) * depth_dst_f.view([nlim, 1]) @ torch.inverse(intrinsic_dst).T
    pts3d_src = to_homogeneous(pts3d_src) @ unpad_poses(pad_poses(pose).inverse()).T @ intrinsic_src.T
    pts2d_src = from_homogeneous(pts3d_src)

    fig, ax = plt.subplots(nrows=3, ncols=2)
    fig.tight_layout()
    ax[0, 0].scatter(sampled_src[:, 0].cpu().numpy(), sampled_src[:, 1].cpu().numpy(), 3, rndcolor)
    ax[0, 1].scatter(sampled_dst[:, 0].cpu().numpy(), sampled_dst[:, 1].cpu().numpy(), 3, rndcolor)
    ax[0, 0].imshow(rgb_src)
    ax[0, 1].imshow(rgb_dst)
    ax[0, 0].axis("off")
    ax[0, 1].axis("off")

    ax[1, 0].scatter(sampled_src[:, 0].cpu().numpy(), sampled_src[:, 1].cpu().numpy(), 3, rndcolor)
    ax[1, 1].scatter(pts2d_dst[:, 0].cpu().numpy(), pts2d_dst[:, 1].cpu().numpy(), 3, rndcolor)
    ax[1, 0].imshow(rgb_src)
    ax[1, 1].imshow(rgb_dst)
    ax[1, 0].axis("off")
    ax[1, 1].axis("off")

    ax[2, 0].scatter(sampled_dst[:, 0].cpu().numpy(), sampled_dst[:, 1].cpu().numpy(), 3, rndcolor)
    ax[2, 1].scatter(pts2d_src[:, 0].cpu().numpy(), pts2d_src[:, 1].cpu().numpy(), 3, rndcolor)
    ax[2, 0].imshow(rgb_dst)
    ax[2, 1].imshow(rgb_src)
    ax[2, 0].axis("off")
    ax[2, 1].axis("off")
    fig.tight_layout()
    if sv_path is not None:
        plt.savefig(sv_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()