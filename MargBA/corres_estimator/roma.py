import os, sys
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from .corre_utils import forward_backward_check, compute_visibility

class RoMa:
    def __init__(self, min_confidence=0.25, min_visibility=0.15):
        prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        RoMa_root = os.path.join(prj_root, 'third_party', 'RoMa')
        sys.path.insert(0, RoMa_root)
        from romatch import roma_indoor
        torch.set_float32_matmul_precision('highest')
        self.corres_estimator = roma_indoor(device=torch.device("cpu"), use_custom_corr=False).eval()
        self.corres_estimator.train(False)
        self.mean, self.std = torch.from_numpy(np.array([0.485, 0.456, 0.406])), torch.from_numpy(np.array([0.229, 0.224, 0.225]))
        self.mean, self.std = self.mean.view([1, 3, 1, 1]), self.std.view([1, 3, 1, 1])
        self.resize = transforms.Resize((self.corres_estimator.h_resized, self.corres_estimator.w_resized), InterpolationMode.BICUBIC)
        self.min_confidence, self.min_visibility, self.sample_num, self.max_cyclic_err = min_confidence, min_visibility, 5000, 1e-2
        print(f"min_confidence: {min_confidence}, min_visibility: {min_visibility}")

    def to_cuda(self, deviceid):
        device = torch.device("cuda:{}".format(str(deviceid)))
        self.corres_estimator = self.corres_estimator.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

    def test_transform(self, rgb_src, rgb_dst):
        rgb_src = (rgb_src - self.mean) / self.std
        rgb_dst = (rgb_dst - self.mean) / self.std
        rgb_src, rgb_dst = self.resize(rgb_src).half(), self.resize(rgb_dst).half()
        return rgb_src, rgb_dst

    def inference(self, data):
        """
        Output correspondence is in pytorch grid_sample format, with 0.5 to h/w-0.5 normalized to -1 to 1, align_corner=False, mode='bilinear'
        """
        rgb_src, rgb_dst = data['rgb_src'], data['rgb_dst']
        _, _, h1, w1 = rgb_src.shape
        _, _, h2, w2 = rgb_dst.shape
        corres_src_dst, certainty_src_dst = self.match(im_A=rgb_src, im_B=rgb_dst)
        certainty_src_dst[certainty_src_dst < self.min_confidence] = 0.0
        corres_dst_src, certainty_dst_src = self.match(im_A=rgb_dst, im_B=rgb_src)
        certainty_dst_src[certainty_dst_src < self.min_confidence] = 0.0
        
        cyclic_err_src_dst = forward_backward_check(corres_src_dst, corres_dst_src, max_cyclic_err=self.max_cyclic_err)
        cyclic_err_dst_src = forward_backward_check(corres_dst_src, corres_src_dst, max_cyclic_err=self.max_cyclic_err)

        valid_src_dst = (cyclic_err_src_dst > 0) * (certainty_src_dst > self.min_confidence)
        valid_dst_src = (cyclic_err_dst_src > 0) * (certainty_dst_src > self.min_confidence)
        visibility = compute_visibility(valid_src_dst, valid_dst_src)

        return corres_src_dst, corres_dst_src, certainty_src_dst, certainty_dst_src, visibility, valid_src_dst, valid_dst_src

    def match(
            self,
            im_A,
            im_B,
    ):
        b, _, _, _ = im_A.shape
        hs, ws = self.corres_estimator.h_resized, self.corres_estimator.w_resized
        cert_clamp, factor, finest_scale, device = 0, 0.5, 1, im_A.device

        im_A, im_B = self.test_transform(im_A, im_B)
        corresps = self.corres_estimator.forward({"im_A": im_A, "im_B": im_B}, batched=True)

        low_res_certainty = F.interpolate(
            corresps[16]["certainty"], size=(hs, ws), align_corners=False, mode="bilinear"
        )
        low_res_certainty = factor * low_res_certainty * (low_res_certainty < cert_clamp)
        certainty = corresps[finest_scale]["certainty"] - (low_res_certainty if self.corres_estimator.attenuate_cert else 0)
        certainty = certainty.sigmoid()  # logits -> probs

        im_A_to_im_B = corresps[finest_scale]["flow"].permute(0, 2, 3, 1)
        # Create im_A meshgrid
        im_A_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / hs, 1 - 1 / hs, hs, device=device),
                torch.linspace(-1 + 1 / ws, 1 - 1 / ws, ws, device=device),
            )
        )
        im_A_coords = torch.stack((im_A_coords[1], im_A_coords[0]))[None].expand(b, 2, hs, ws)
        im_A_coords = im_A_coords.permute(0, 2, 3, 1)

        wrong = (im_A_to_im_B.abs() > 1).sum(dim=-1) > 0
        certainty[wrong[:, None]] = 0

        im_A_to_im_B = torch.clamp(im_A_to_im_B, -1, 1)
        warp = torch.cat((im_A_coords, im_A_to_im_B), dim=-1)
        return warp, certainty
    
    def sample_corres_depth(self, corres_src_dst, corres_dst_src, certainty_src_dst, certainty_dst_src, depth_prd_src, depth_prd_dst, h, w):
        selector = certainty_src_dst > 0
        corres_src_dst_f = corres_src_dst[selector, :]
        prob_src_dst_f = certainty_src_dst[selector]

        selector = certainty_dst_src > 0
        corres_dst_src_f = corres_dst_src[selector, :]
        pts_dst, pts_src = torch.split(corres_dst_src_f, [2, 2], dim=1)
        prob_dst_src_f = certainty_dst_src[selector]

        corres_f = torch.cat([corres_src_dst_f, torch.cat([pts_src, pts_dst], dim=1)], dim=0)
        prob_f = torch.cat([prob_src_dst_f, prob_dst_src_f], dim=0)
        assert len(prob_f) > self.sample_num

        prob_f = prob_f.cpu().numpy()
        sampled_idx = np.random.choice(
            np.arange(len(prob_f)),
            size=self.sample_num,
            replace=False,
            p=prob_f / np.sum(prob_f),
        )
        sampled_src, sampled_dst, prob_f = corres_f[sampled_idx, 0:2], corres_f[sampled_idx, 2:4], prob_f[sampled_idx]
        depth_src_f = torch.nn.functional.grid_sample(depth_prd_src.view([1, 1, h, w]), sampled_src.view([1, 1, self.sample_num, 2]), mode="bilinear", align_corners=False).squeeze()
        depth_dst_f = torch.nn.functional.grid_sample(depth_prd_dst.view([1, 1, h, w]), sampled_dst.view([1, 1, self.sample_num, 2]), mode="bilinear", align_corners=False).squeeze()
        return sampled_src, sampled_dst, depth_src_f, depth_dst_f, prob_f