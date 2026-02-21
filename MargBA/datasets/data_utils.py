import torch, cv2, os
import numpy as np
import quaternion
import PIL.Image as Image
import torch.nn.functional as F
from typing import List, Callable

class InputPadder:
    """ Pads images such that dimensions are divisible by padding """
    def __init__(self, dims, padding=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // padding) + 1) * padding - self.ht) % padding
        pad_wd = (((self.wd // padding) + 1) * padding - self.wd) % padding
        self._pad = [0, pad_wd, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def numpy_image_to_torch(image: np.ndarray):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f'Not an image: {image.shape}')
    return torch.from_numpy((image / 255.).astype(np.float32, copy=False))

def resize(image: np.ndarray, size: List[int], fn: Callable[[List], float] = None,
           interp: str = 'linear'):
    """Resize an image to a fixed size, or according to max or min edge."""
    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
    else:
        # pil
        w, h = image.size[:2]

    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
    else:
        raise ValueError(f'Incorrect new size: {size}')

    scale = (w_new / w, h_new / h)

    mode = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'nearest': cv2.INTER_NEAREST
    }[interp]

    if isinstance(image, np.ndarray):
        image = cv2.resize(image, (w_new, h_new), interpolation=mode)
    else:
        image = image.resize((w_new, h_new))
    return image, scale

def initialize_dataset(data_root, dataset, mode, rank, world_size, output_location=None):
    """
    Read RGB images and intrinsic. If GT poses and depths exist, read them for debug purpose.
    """
    if dataset == 'ScanNet':
        from .scannet import ScanNet
        scannet = ScanNet(
            data_root=data_root,
            mode=mode,
            rank=rank,
            world_size=world_size
        )
        return scannet
    elif dataset == '7scenes':
        root = os.path.dirname(os.path.dirname(data_root))
        seq = os.path.basename(data_root)
        subscene = os.path.basename(os.path.dirname(data_root))
        from .sevenscenes import SevenScenes
        sevenscenes = SevenScenes(
            root=root,
            output_location=output_location,
            seq=seq,
            subscene=subscene,
            rank=rank,
            mode=mode,
            world_size=world_size
        )
        return sevenscenes
    elif dataset == 'eth3d':
        from .eth3d import ETH3D
        eth3d = ETH3D(
            root=data_root,
            output_location=output_location,
            scene=os.path.basename(data_root),
            rank=rank,
            mode=mode,
            world_size=world_size
        )
        return eth3d
    elif dataset == 'wayspots':
        from .wayspots import Wayspots
        wayspots = Wayspots(
            root=data_root,
            output_location=output_location,
            scene=os.path.basename(data_root),
            mode=mode,
            rank=rank,
            world_size=world_size
        )
        return wayspots
    elif dataset == "imc2021":
        from .imc2021 import IMC2021
        imc2021 = IMC2021(
            root=data_root,
            output_location=output_location,
            mode=mode
        )
        return imc2021
    elif dataset == "custom":
        from .custom import CustomDataset
        custom = CustomDataset(
            data_root=data_root,
            mode=mode,
            rank=rank,
            world_size=world_size
        )
        return custom
    elif dataset == "fastmap_sfm":
        from .fastmap_sfm import FASTMAPSfM
        fastmap_sfm = FASTMAPSfM(
            root=data_root,
            cache_dir=output_location,
            mode=mode,
            rank=rank,
            world_size=world_size
        )
        return fastmap_sfm
    else:
        raise NotImplementedError()

def to_cuda(bundle):
    if isinstance(bundle, dict):
        for x in bundle.keys():
            if isinstance(bundle[x], torch.Tensor):
                bundle[x] = bundle[x].cuda()
    elif isinstance(bundle, list):
        for i in range(len(bundle)):
            if isinstance(bundle[i], torch.Tensor):
                bundle[i] = bundle[i].cuda()
    return bundle

def cvt_monodepth_to_png(monodepth):
    assert monodepth.ndim == 2
    if isinstance(monodepth, torch.Tensor):
        monodepth = monodepth.detach().cpu().numpy()
    monodepth[monodepth > 65.0] = 65.0
    monodepth_uint16 = monodepth * 1000
    assert np.max(monodepth_uint16) < 65535
    monodepth_uint16 = monodepth_uint16.astype(np.uint16)
    return Image.fromarray(monodepth_uint16)

def cvt_png_to_monodepth(monodepth_uint16):
    monodepth_uint16 = np.array(monodepth_uint16)
    monodepth_uint16 = monodepth_uint16.astype(np.float32)
    monodepth = monodepth_uint16 / 1000
    return monodepth

def cvt_incidence_to_png(incidence):
    assert incidence.ndim == 3
    incidence_uint16 = (incidence + 1) / 2 * 65535.0
    incidence_uint16 = torch.round(incidence_uint16)
    if isinstance(incidence_uint16, torch.Tensor):
        incidence_uint16 = incidence_uint16.detach().cpu().numpy()
    incidence_x_uint16 = incidence_uint16[0].astype(np.uint16)
    incidence_y_uint16 = incidence_uint16[1].astype(np.uint16)
    return Image.fromarray(incidence_x_uint16), Image.fromarray(incidence_y_uint16)

def cvt_pngs_to_incidence(incidence_x_uint16, incidence_y_uint16):
    incidence_x = np.array(incidence_x_uint16).astype(np.float32) / 65535.0 * 2 - 1
    incidence_y = np.array(incidence_y_uint16).astype(np.float32) / 65535.0 * 2 - 1
    return np.stack([incidence_x, incidence_y], axis=0)

def cam_to_world_from_kapture(kdata, timestamp, camera_id):
    camera_to_world = kdata['trajectories'][timestamp, camera_id].inverse()
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[:3, :3] = quaternion.as_rotation_matrix(camera_to_world.r)
    camera_pose[:3, 3] = camera_to_world.t_raw
    return camera_pose