import os, sys, torch
import numpy as np
import logging
logger = logging.getLogger(__name__)

import PIL.Image as Image
prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dust3r_root = os.path.join(prj_root, 'third_party', 'dust3r')
sys.path.insert(0, dust3r_root)

import PIL.Image as Image
from PIL import ImageOps
from dust3r.inference import inference
from dust3r.utils.image import load_images, _resize_pil_image, exif_transpose, ImgNorm
from dust3r.image_pairs import make_pairs

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

class DUSt3R:
    def __init__(self):
        prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dust3r_root = os.path.join(prj_root, 'third_party', 'dust3r')
        sys.path.insert(0, dust3r_root)
        from dust3r.model import AsymmetricCroCo3DStereo

        model_name = os.path.join(prj_root, "modelzoo", "dust3r", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        self.monopointcloud_model = AsymmetricCroCo3DStereo.from_pretrained(model_name)
        self.monopointcloud_model.eval()

    def to_cuda(self, device):
        self.device = device
        self.monopointcloud_model = self.monopointcloud_model.to(device)

    def inference_single_image(self, image):
        image = image.permute([1, 2, 0]) * 255.0
        image = torch.round(image)
        image = torch.clip_(image, min=0, max=255)
        image = image.cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image)
        w, h = image.size

        imgloader = ImageLoader(
            images=
            [
                image,
                image
            ],
            size=512
        )

        images = imgloader.imgs
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=False)
        output = inference(pairs, self.monopointcloud_model, self.device, batch_size=1)
        pts3d = output['pred1']['pts3d']

        pts3d = imgloader.decrop(pts3d)
        pts3d = pts3d.permute([0, 3, 1, 2])
        pts3d = torch.nn.functional.interpolate(pts3d, (h, w), mode='bilinear', align_corners=False)
        _, _, monodepth = torch.split(pts3d, 1, 1)
        incidence = pts3d / monodepth
        incidence, monodepth = incidence.squeeze()[0:2], monodepth.squeeze()
        if incidence.max() > 1.0 or incidence.min() < -1.0:
            logger.warning("Warning: Incidence field value out of bound.")
        incidence = torch.clamp(incidence, min=-1.0, max=1.0)
        return monodepth, incidence

    def inference(self, image):
        bz, _, h, w = image.shape
        prs = [self.inference_single_image(image[i]) for i in range(bz)]
        monodepth, incidence = [pr[0] for pr in prs], [pr[1] for pr in prs]
        monodepth, incidence = torch.stack(monodepth, dim=0).unsqueeze(1), torch.stack(incidence, dim=0)
        return monodepth, incidence