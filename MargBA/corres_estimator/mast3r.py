import os, sys
import torch
import numpy as np
import PIL.Image as Image
import torchvision.transforms as tvf
import matplotlib.pyplot as plt

prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mast3r_root = os.path.join(prj_root, 'third_party', 'mast3r')
sys.path.insert(0, mast3r_root)

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.utils.coarse_to_fine import select_pairs_of_crops, crop_slice
from mast3r.utils.collate import cat_collate, cat_collate_fn_map
from mast3r.datasets.utils.cropping import crop_to_homography

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference, loss_of_one_batch
from dust3r.utils.geometry import geotrf, colmap_to_opencv_intrinsics, opencv_to_colmap_intrinsics
from dust3r.datasets.utils.transforms import ImgNorm
from dust3r_visloc.datasets.utils import get_HW_resolution, rescale_points3d, get_resize_function


@torch.no_grad()
def coarse_matching(query_view, map_view, model, device, pixel_tol, fast_nn_params):
    # prepare batch
    imgs = []
    for idx, img in enumerate([query_view['rgb_rescaled'], map_view['rgb_rescaled']]):
        imgs.append(dict(img=img.unsqueeze(0), true_shape=np.int32([img.shape[1:]]),
                         idx=idx, instance=str(idx)))
    output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    conf_list = [pred1['desc_conf'].squeeze(0).cpu().numpy(), pred2['desc_conf'].squeeze(0).cpu().numpy()]
    desc_list = [pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()]

    # find 2D-2D matches between the two images
    PQ, PM = desc_list[0], desc_list[1]
    if len(PQ) == 0 or len(PM) == 0:
        return [], [], []

    if pixel_tol == 0:
        matches_im_map, matches_im_query = fast_reciprocal_NNs(PM, PQ, subsample_or_initxy1=8, **fast_nn_params)
        HM, WM = map_view['rgb_rescaled'].shape[1:]
        HQ, WQ = query_view['rgb_rescaled'].shape[1:]
        # ignore small border around the edge
        valid_matches_map = (matches_im_map[:, 0] >= 3) & (matches_im_map[:, 0] < WM - 3) & (
            matches_im_map[:, 1] >= 3) & (matches_im_map[:, 1] < HM - 3)
        valid_matches_query = (matches_im_query[:, 0] >= 3) & (matches_im_query[:, 0] < WQ - 3) & (
            matches_im_query[:, 1] >= 3) & (matches_im_query[:, 1] < HQ - 3)
        valid_matches = valid_matches_map & valid_matches_query
        matches_im_map = matches_im_map[valid_matches]
        matches_im_query = matches_im_query[valid_matches]
        matches_confs = []
    else:
        yM, xM = torch.where(map_view['valid_rescaled'])
        matches_im_map, matches_im_query = fast_reciprocal_NNs(PM, PQ, (xM, yM), pixel_tol=pixel_tol, **fast_nn_params)
        matches_confs = np.minimum(
            conf_list[1][matches_im_map[:, 1], matches_im_map[:, 0]],
            conf_list[0][matches_im_query[:, 1], matches_im_query[:, 0]]
        )
    # from cv2 to colmap
    matches_im_query = matches_im_query.astype(np.float64)
    matches_im_map = matches_im_map.astype(np.float64)
    matches_im_query[:, 0] += 0.5
    matches_im_query[:, 1] += 0.5
    matches_im_map[:, 0] += 0.5
    matches_im_map[:, 1] += 0.5
    # rescale coordinates
    matches_im_query = geotrf(query_view['to_orig'], matches_im_query, norm=True)
    matches_im_map = geotrf(map_view['to_orig'], matches_im_map, norm=True)
    # from colmap back to cv2
    matches_im_query[:, 0] -= 0.5
    matches_im_query[:, 1] -= 0.5
    matches_im_map[:, 0] -= 0.5
    matches_im_map[:, 1] -= 0.5
    return matches_im_query, matches_im_map, matches_confs


@torch.no_grad()
def crops_inference(pairs, model, device, batch_size=48, verbose=True):
    assert len(pairs) == 2, "Error, data should be a tuple of dicts containing the batch of image pairs"
    # Forward a possibly big bunch of data, by blocks of batch_size
    B = pairs[0]['img'].shape[0]
    if B < batch_size:
        return loss_of_one_batch(pairs, model, None, device=device, symmetrize_batch=False)
    preds = []
    for ii in range(0, B, batch_size):
        sel = slice(ii, ii + min(B - ii, batch_size))
        temp_data = [{}, {}]
        for di in [0, 1]:
            temp_data[di] = {kk: pairs[di][kk][sel]
                             for kk in pairs[di].keys() if pairs[di][kk] is not None}  # copy chunk for forward
        preds.append(loss_of_one_batch(temp_data, model,
                                       None, device=device, symmetrize_batch=False))  # sequential forward
    # Merge all preds
    return cat_collate(preds, collate_fn_map=cat_collate_fn_map)


@torch.no_grad()
def fine_matching(query_views, map_views, model, device, max_batch_size, pixel_tol, fast_nn_params):
    assert pixel_tol > 0
    output = crops_inference([query_views, map_views],
                             model, device, batch_size=max_batch_size, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    descs1 = pred1['desc'].clone()
    descs2 = pred2['desc'].clone()
    confs1 = pred1['desc_conf'].clone()
    confs2 = pred2['desc_conf'].clone()

    # Compute matches
    matches_im_map, matches_im_query, matches_confs = [], [], []
    for ppi, (pp1, pp2, cc11, cc21) in enumerate(zip(descs1, descs2, confs1, confs2)):
        conf_list_ppi = [cc11.cpu().numpy(), cc21.cpu().numpy()]

        y_ppi, x_ppi = torch.where(torch.ones([384, 512]))
        matches_im_map_ppi, matches_im_query_ppi = fast_reciprocal_NNs(
            pp2, pp1, (x_ppi, y_ppi), pixel_tol=pixel_tol, **fast_nn_params
        )

        matches_confs_ppi = np.minimum(
            conf_list_ppi[1][matches_im_map_ppi[:, 1], matches_im_map_ppi[:, 0]],
            conf_list_ppi[0][matches_im_query_ppi[:, 1], matches_im_query_ppi[:, 0]]
        )
        # inverse operation where we uncrop pixel coordinates
        matches_im_map_ppi = geotrf(map_views['to_orig'][ppi].cpu().numpy(), matches_im_map_ppi.copy(), norm=True)
        matches_im_query_ppi = geotrf(query_views['to_orig'][ppi].cpu().numpy(), matches_im_query_ppi.copy(), norm=True)

        matches_im_map.append(matches_im_map_ppi)
        matches_im_query.append(matches_im_query_ppi)
        matches_confs.append(matches_confs_ppi)


    matches_im_map = np.concatenate(matches_im_map, axis=0)
    matches_im_query = np.concatenate(matches_im_query, axis=0)
    matches_confs = np.concatenate(matches_confs, axis=0)
    return matches_im_query, matches_im_map, matches_confs


def crop(img, mask, pts3d, crop, intrinsics=None):
    if mask is not None:
        out_cropped_mask = mask.clone()
    else:
        out_cropped_mask = None
    if pts3d is not None:
        out_cropped_pts3d = pts3d.clone()
    else:
        out_cropped_pts3d = None
    to_orig = torch.eye(3, device=img.device)

    # If intrinsics available, crop and apply rectifying homography. Otherwise, just crop
    if intrinsics is not None:
        K_old = intrinsics
        imsize, K_new, R, H = crop_to_homography(K_old, crop)
        # apply homography to image
        H /= H[2, 2]
        homo8 = H.ravel().tolist()[:8]
        # From float tensor to uint8 PIL Image
        pilim = Image.fromarray((255 * (img + 1.) / 2).to(torch.uint8).numpy())
        pilout_cropped_img = pilim.transform(imsize, Image.Transform.PERSPECTIVE,
                                             homo8, resample=Image.Resampling.BICUBIC)

        # From uint8 PIL Image to float tensor
        out_cropped_img = 2. * torch.tensor(np.array(pilout_cropped_img)).to(img) / 255. - 1.
        if out_cropped_mask is not None:
            pilmask = Image.fromarray((255 * out_cropped_mask).to(torch.uint8).numpy())
            pilout_cropped_mask = pilmask.transform(
                imsize, Image.Transform.PERSPECTIVE, homo8, resample=Image.Resampling.NEAREST)
            out_cropped_mask = torch.from_numpy(np.array(pilout_cropped_mask) > 0).to(out_cropped_mask.dtype)
        if out_cropped_pts3d is not None:
            out_cropped_pts3d = out_cropped_pts3d.numpy()
            out_cropped_X = np.array(Image.fromarray(out_cropped_pts3d[:, :, 0]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))
            out_cropped_Y = np.array(Image.fromarray(out_cropped_pts3d[:, :, 1]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))
            out_cropped_Z = np.array(Image.fromarray(out_cropped_pts3d[:, :, 2]).transform(imsize,
                                                                                           Image.Transform.PERSPECTIVE,
                                                                                           homo8,
                                                                                           resample=Image.Resampling.NEAREST))

            out_cropped_pts3d = torch.from_numpy(np.stack([out_cropped_X, out_cropped_Y, out_cropped_Z], axis=-1))

        to_orig = torch.tensor(H, device=img.device)
    else:
        out_cropped_img = img[crop_slice(crop)]
        if out_cropped_mask is not None:
            out_cropped_mask = out_cropped_mask[crop_slice(crop)]
        if out_cropped_pts3d is not None:
            out_cropped_pts3d = out_cropped_pts3d[crop_slice(crop)]
        to_orig[:2, -1] = torch.tensor(crop[:2])

    return out_cropped_img, out_cropped_mask, out_cropped_pts3d, to_orig

def resize_image_to_max(max_image_size, rgb):
    W, H = rgb.size
    if max_image_size and max(W, H) > max_image_size:
        islandscape = (W >= H)
        if islandscape:
            WMax = max_image_size
            HMax = int(H * (WMax / W))
        else:
            HMax = max_image_size
            WMax = int(W * (HMax / H))
        resize_op = tvf.Compose([ImgNorm, tvf.Resize(size=[HMax, WMax])])
        rgb_tensor = resize_op(rgb).permute(1, 2, 0)
        to_orig_max = np.array([[W / WMax, 0, 0],
                                [0, H / HMax, 0],
                                [0, 0, 1]])
        to_resize_max = np.array([[WMax / W, 0, 0],
                                  [0, HMax / H, 0],
                                  [0, 0, 1]])

    else:
        rgb_tensor = ImgNorm(rgb).permute(1, 2, 0)
        to_orig_max = np.eye(3)
        to_resize_max = np.eye(3)
        HMax, WMax = H, W
    return rgb_tensor, to_orig_max, to_resize_max, (HMax, WMax)

def vls_matches(viz_matches_im_query, viz_matches_im_map, viz_imgs, img_path="/home/ubuntu/tmp.png"):
    n_viz = 100
    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    plt.figure()
    plt.imshow(img)
    cmap = plt.get_cmap('jet')
    for i in range(n_viz):
        rndid  = np.random.randint(len(viz_matches_im_query))
        (x0, y0), (x1, y1) = viz_matches_im_query[rndid].T, viz_matches_im_map[rndid].T
        plt.scatter([x0, x1 + W0], [y0, y1], s=1.0, color=cmap(i / (n_viz - 1)))
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

class MASt3R:
    def __init__(self, min_confidence, min_visibility):
        prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        mast3r_root = os.path.join(prj_root, 'third_party', 'mast3r')
        sys.path.insert(0, mast3r_root)
        weights_path = "naver/" + "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        self.corres_estimator = AsymmetricMASt3R.from_pretrained(weights_path).eval()
        self.maxdim, self.patch_size = 512, (16, 16)
        self.TensorNorm = tvf.Compose([tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.conf_thr = 1e-2
        self.pixel_tol = 5

        self.min_confidence = min_confidence
        self.min_visibility = min_visibility

    def to_cuda(self, deviceid):
        device = torch.device("cuda:{}".format(str(deviceid)))
        self.fast_nn_params = dict(device=device, dist='dot', block_size=2 ** 13)
        self.corres_estimator = self.corres_estimator.to(device)
        self.device = device

    def match_source_target(self, query_view, map_view):
        maxdim = max(self.corres_estimator.patch_embed.img_size)
        query_rgb_tensor, query_to_orig_max, query_to_resize_max, (HQ, WQ) = resize_image_to_max(
            None,
            query_view['rgb']
        )

        # pairs of crops have the same resolution
        query_resolution = get_HW_resolution(HQ, WQ, maxdim=maxdim, patchsize=self.corres_estimator.patch_embed.patch_size)

        # use all points
        coarse_matches_im0, coarse_matches_im1, _ = coarse_matching(
            query_view, map_view, self.corres_estimator, self.device, 0, self.fast_nn_params
        )

        map_rgb_tensor, map_to_orig_max, map_to_resize_max, (HM, WM) = resize_image_to_max(None, map_view['rgb'])

        coarse_matches_im0 = geotrf(query_to_resize_max, coarse_matches_im0, norm=True)
        coarse_matches_im1 = geotrf(map_to_resize_max, coarse_matches_im1, norm=True)

        crops1, crops2 = [], []
        to_orig1, to_orig2 = [], []
        map_resolution = get_HW_resolution(HM, WM, maxdim=maxdim, patchsize=self.corres_estimator.patch_embed.patch_size)

        for crop_q, crop_b, pair_tag in select_pairs_of_crops(
                map_rgb_tensor,
                query_rgb_tensor,
                coarse_matches_im1,
                coarse_matches_im0,
                maxdim=maxdim,
                overlap=.5,
                forced_resolution=[map_resolution, query_resolution]
        ):
            c1, _, _, trf1 = crop(map_rgb_tensor, None, None, crop_q, None)
            c2, _, _, trf2 = crop(query_rgb_tensor, None, None, crop_b, None)
            crops1.append(c1)
            crops2.append(c2)
            to_orig1.append(trf1)
            to_orig2.append(trf2)

        if len(crops1) == 0 or len(crops2) == 0:
            matches_im_query, matches_im_map, matches_conf = [], [], []
        else:
            crops1, crops2 = torch.stack(crops1), torch.stack(crops2)
            if len(crops1.shape) == 3: crops1, crops2 = crops1[None], crops2[None]
            to_orig1, to_orig2 = torch.stack(to_orig1), torch.stack(to_orig2)
            map_crop_view = dict(
                img=crops1.permute(0, 3, 1, 2),
                instance=['1' for _ in range(crops1.shape[0])],
                to_orig=to_orig1
            )
            query_crop_view = dict(
                img=crops2.permute(0, 3, 1, 2),
                instance=['2' for _ in range(crops2.shape[0])],
                to_orig=to_orig2
            )

            # Inference and Matching
            matches_im_query, matches_im_map, matches_conf = fine_matching(
                query_crop_view,
                map_crop_view,
                self.corres_estimator, self.device, 48, self.pixel_tol, self.fast_nn_params
            )
            matches_im_query = geotrf(query_to_orig_max, matches_im_query, norm=True)
            matches_im_map = geotrf(map_to_orig_max, matches_im_map, norm=True)

        # apply conf
        if len(matches_conf) > 0:
            mask = matches_conf >= self.conf_thr
            matches_im_query = matches_im_query[mask]
            matches_im_map = matches_im_map[mask]
            matches_conf = matches_conf[mask]
            if np.sum(mask) < 500:
                matches_im_query, matches_im_map, matches_conf = [], [], []

        return matches_im_query, matches_im_map, matches_conf

    def test_transform(self, rgb : torch.Tensor, rgb_wo_resize: Image):
        W, H = rgb_wo_resize.size
        resize_func, to_resize, to_orig = get_resize_function(self.maxdim, self.patch_size, H, W)
        rgb_tensor = resize_func(self.TensorNorm(rgb))
        view = {
            'rgb_rescaled': rgb_tensor,
            'to_orig': to_orig,
            'rgb': rgb_wo_resize
        }
        return view

    @torch.no_grad()
    def inference(self, data):
        """
        Output correspondence is in pytorch grid_sample format, with 0.5 to h/w-0.5 normalized to -1 to 1, align_corner=False, mode='bilinear'
        """
        rgb_src, rgb_dst = data['rgb_src'], data['rgb_dst']
        corres_src_dst, certainty_src_dst = self.match(
            im_A=rgb_src,
            im_B=rgb_dst,
            im_A_wo_resize=data['rgb_src_wo_resize'],
            im_B_wo_resize=data['rgb_dst_wo_resize'],
        )
        corres_dst_src, certainty_dst_src = self.match(
            im_A=rgb_dst,
            im_B=rgb_src,
            im_A_wo_resize=data['rgb_dst_wo_resize'],
            im_B_wo_resize=data['rgb_src_wo_resize'],
        )
        _, _, h1, w1 = certainty_src_dst.shape
        _, _, h2, w2 = certainty_dst_src.shape
        assert h1 == h2 and w1 == w2
        valid_src_dst, valid_dst_src = certainty_src_dst > 0, certainty_dst_src > 0
        visibility = torch.sum(valid_src_dst, dim=[1, 2, 3]) / h1 / w1 + torch.sum(valid_dst_src, dim=[1, 2, 3]) / h2 / w2
        visibility = visibility / 2
        return corres_src_dst, corres_dst_src, certainty_src_dst, certainty_dst_src, visibility, valid_src_dst, valid_dst_src

    def format_match_confidences(self, matches_im_query, matches_im_map, matches_conf, im_query, im_map, warp, certainty):
        def normalize_corres(matches, h, w):
            matches = torch.from_numpy(matches) + 0.5
            matches[:, 0] = (matches[:, 0] / w - 0.5) * 2
            matches[:, 1] = (matches[:, 1] / h - 0.5) * 2
            return matches.float()

        certainty[0, matches_im_query[:, 1].astype(int), matches_im_query[:, 0].astype(int)] = torch.from_numpy(matches_conf).float().to(self.device)
        wq, hq = im_query.size
        warp[matches_im_query[:, 1].astype(int), matches_im_query[:, 0].astype(int), 0:2] = normalize_corres(matches_im_query, hq, wq).to(self.device)
        wm, hm = im_map.size
        warp[matches_im_query[:, 1].astype(int), matches_im_query[:, 0].astype(int), 2:4] = normalize_corres(matches_im_map, hm, wm).to(self.device)

    def match(
            self,
            im_A,
            im_B,
            im_A_wo_resize,
            im_B_wo_resize
    ):
        b, _, _, _ = im_A.shape
        _, hq, wq, _ = im_A_wo_resize.shape
        # bz x sz x sz x 4 and bz x 1 x sz x sz
        warp, certainty = torch.zeros([b, hq, wq, 4], device=self.device), torch.zeros([b, 1, hq, wq], device=self.device)
        for bzid in range(b):
            query_wo_resize, map_wo_resize = Image.fromarray(im_A_wo_resize[bzid].cpu().numpy()), Image.fromarray(im_B_wo_resize[bzid].cpu().numpy())
            query_view, map_view = self.test_transform(im_A[bzid], query_wo_resize), self.test_transform(im_B[bzid], map_wo_resize)
            matches_im_query, matches_im_map, matches_conf = self.match_source_target(query_view, map_view)
            # vls_matches(matches_im_query, matches_im_map, viz_imgs = [np.array(query_view['rgb']), np.array(map_view['rgb'])], img_path="/home/ubuntu/tmp.png")
            if len(matches_conf) == 0: continue
            self.format_match_confidences(matches_im_query, matches_im_map, matches_conf, im_query=query_view['rgb'], im_map=map_view['rgb'], warp=warp[bzid], certainty=certainty[bzid])

        return warp, certainty
