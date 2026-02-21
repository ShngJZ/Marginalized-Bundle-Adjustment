import os, sys, torch
import numpy as np
import PIL.Image as Image
prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dust3r_root = os.path.join(prj_root, 'third_party', 'dust3r')
sys.path.insert(0, dust3r_root)

import PIL.Image as Image
from PIL import ImageOps
from dust3r.inference import inference
from dust3r.utils.image import load_images, _resize_pil_image, exif_transpose, ImgNorm
from dust3r.image_pairs import make_pairs

from MargBA.monodepth_estimator.calibrator import MonocularCalibrator

class ImageLoader:
    def __init__(self, images, size):
        gridres = 16
        imgs, imgsops = [], []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            img = img.convert('RGB')
            W1, H1 = img.size

            maxlen = max(H1, W1)
            resizer = size / maxlen
            W1rs, H1rs = int(resizer * W1), int(resizer * H1)
            img = img.resize((W1rs, H1rs), Image.BILINEAR)

            padW, padH = int(gridres * np.ceil(W1rs / gridres) - W1rs), int(gridres * np.ceil(H1rs / gridres) - H1rs)
            img = ImageOps.expand(img, border=(0, padH, padW, 0), fill=0)

            imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32([img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

            imgsops.append([W1rs, H1rs, padW, padH])

        self.imgs = imgs
        self.imgsops = imgsops

    def decrop(self, point3d):
        assert point3d.shape[3] == 3
        _, _, padW, padH = self.imgsops[0]
        _, H, W, _ = point3d.shape
        point3d = point3d[:, padH:H, 0:W-padW, :]
        return point3d

class WildDUSt3R:
    def __init__(self):
        prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dust3r_root = os.path.join(prj_root, 'third_party', 'dust3r')
        sys.path.insert(0, dust3r_root)
        from dust3r.model import AsymmetricCroCo3DStereo

        model_name = os.path.join(prj_root, "modelzoo", "dust3r", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        self.monopointcloud_model = AsymmetricCroCo3DStereo.from_pretrained(model_name)
        self.monopointcloud_model.eval()

        self.mc = MonocularCalibrator()

    def to_cuda(self, device):
        self.device = device
        self.monopointcloud_model = self.monopointcloud_model.to(device)

    def inference(self, jpg_path):
        w, h = Image.open(jpg_path).size
        imgloader = ImageLoader(
            [
                jpg_path,
                jpg_path
            ], size=512)
        images = imgloader.imgs
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=False)
        output = inference(pairs, self.monopointcloud_model, self.device, batch_size=1)
        pts3d = output['pred1']['pts3d']
        pts3d = imgloader.decrop(pts3d)

        x, y, z = torch.split(pts3d, 1, -1)
        incidence = pts3d / z
        incidence_1DoF = torch.nn.functional.interpolate(incidence.permute([0, 3, 1, 2]), (h, w), mode='bilinear', align_corners=True)
        intrinsic = self.mc.calibrate_camera_1DoF(incidence_1DoF[0:1], r=1.0)
        return intrinsic.cpu().numpy()