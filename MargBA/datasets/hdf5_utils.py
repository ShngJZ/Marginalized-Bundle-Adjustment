import h5py, os, io
import natsort
import numpy as np
import PIL.Image as Image
from .data_utils import cvt_png_to_monodepth, cvt_pngs_to_incidence

class HDF5Writer():
    def __init__(self, h5pypath, keys_to_add, mode):
        hfw = h5py.File(h5pypath, mode)
        if 'depth_gt' in keys_to_add and 'depth_gt' not in hfw:
            hfw.create_group('depth_gt')
        if 'depth_pr' in keys_to_add and 'depth_pr' not in hfw:
            hfw.create_group('depth_pr')
        if 'rgb' in keys_to_add and 'rgb' not in hfw:
            hfw.create_group('rgb')
        if 'pose_w2c_gt' in keys_to_add and 'pose_w2c_gt' not in hfw:
            hfw.create_group('pose_w2c_gt')
        if 'intrinsic_gt' in keys_to_add and 'intrinsic_gt' not in hfw:
            hfw.create_group('intrinsic_gt')
        if 'intrinsic_et' in keys_to_add and 'intrinsic_et' not in hfw:
            hfw.create_group('intrinsic_et')
        if 'pose_i2j_gt' in keys_to_add and 'pose_i2j_gt' not in hfw:
            hfw.create_group('pose_i2j_gt')
        if 'corres_i2j' in keys_to_add and 'corres_i2j' not in hfw:
            hfw.create_group('corres_i2j')
        if 'visibility_i2j' in keys_to_add and 'visibility_i2j' not in hfw:
            hfw.create_group('visibility_i2j')
        if 'incidence_pr' in keys_to_add and 'incidence_pr' not in hfw:
            if 'incidence_pr_x' not in hfw: hfw.create_group('incidence_pr_x')
            if 'incidence_pr_y' not in hfw: hfw.create_group('incidence_pr_y')
        self.hfw = hfw

    def write_depth_gt(self, monodepth_gt_path, image_name):
        self.hfw['depth_gt'].create_dataset(image_name, data=self.image2binary(monodepth_gt_path))

    def write_depth_pr(self, monodepth_pr_path, image_name):
        self.hfw['depth_pr'].create_dataset(image_name, data=self.image2binary(monodepth_pr_path))

    def image2binary(self, image_path):
        with open(image_path, 'rb') as img_f:
            image_file = img_f.read()
        img_np_array = np.asarray(image_file)
        return img_np_array

    def write_jpg_image(self, image_path, image_name):
        self.hfw['rgb'].create_dataset(image_name, data=self.image2binary(image_path))

    def write_pose_w2c_gt(self, pose_w2c_gt_path, txt_name):
        self.hfw['pose_w2c_gt'].create_dataset(txt_name, data=np.loadtxt(pose_w2c_gt_path))

    def write_intrinsic_gt(self, intrinsic_gt_path, txt_name):
        self.hfw['intrinsic_gt'].create_dataset(txt_name, data=np.loadtxt(intrinsic_gt_path))

    def write_intrinsic_et(self, intrinsic_et_path, txt_name):
        self.hfw['intrinsic_et'].create_dataset(txt_name, data=np.loadtxt(intrinsic_et_path))

    def write_pose_i2j_gt(self, pose_i2j_path, txt_name):
        self.hfw['pose_i2j_gt'].create_dataset(txt_name, data=np.loadtxt(pose_i2j_path))

    def write_corres_pair(self, corres_pair_path, corres_pair_name):
        self.hfw['corres_i2j'].create_group(corres_pair_name)
        for end in ["{}_conf.png", "{}_x.png", "{}_y.png"]:
            image_name = end.format(corres_pair_name)
            self.hfw['corres_i2j'][corres_pair_name].create_dataset(image_name, data=self.image2binary(
                os.path.join(corres_pair_path, image_name)))

    def write_visibility(self, visibility_i2j_path, txt_name):
        self.hfw['visibility_i2j'].create_dataset(txt_name, data=np.loadtxt(visibility_i2j_path))

    def write_incidence(self, incidence_pr_x_path, incidence_pr_y_path, image_name):
        self.hfw['incidence_pr_x'].create_dataset(image_name, data=self.image2binary(incidence_pr_x_path))
        self.hfw['incidence_pr_y'].create_dataset(image_name, data=self.image2binary(incidence_pr_y_path))

class HDF5Reader:
    def __init__(self, h5pypath):
        self.h5pypath = h5pypath

    def get_all_images(self):
        hfw = h5py.File(self.h5pypath, "r")
        rgbs = natsort.natsorted(hfw['rgb'].keys())
        return [x.rstrip(".jpg") for x in rgbs]

    def get_all_image_pairs(self):
        hfw = h5py.File(self.h5pypath, "r")
        pairs = natsort.natsorted(hfw['corres_i2j'].keys())
        return [x.rstrip(".png") for x in pairs]

    def have_image_pairs(self, pair_name):
        hfw = h5py.File(self.h5pypath, "r")
        return pair_name in hfw['corres_i2j']

    def read_depth_gt(self, image_name):
        hfw = h5py.File(self.h5pypath, "r")
        image_name = "{}.png".format(image_name)
        if 'depth_gt' in hfw:
            return cvt_png_to_monodepth(np.array(Image.open(io.BytesIO(np.array(hfw['depth_gt'][image_name])))))
        else:
            return None

    def read_depth_pr(self, image_name):
        hfw = h5py.File(self.h5pypath, "r")
        image_name = "{}.png".format(image_name)
        if 'depth_pr' in hfw:
            return cvt_png_to_monodepth(np.array(Image.open(io.BytesIO(np.array(hfw['depth_pr'][image_name])))))
        else:
            return None

    def read_incidence_pr(self, image_name):
        hfw = h5py.File(self.h5pypath, "r")
        if 'incidence_pr_x' in hfw and 'incidence_pr_y' in hfw:
            incidence_pr_x = np.array(Image.open(io.BytesIO(np.array(hfw['incidence_pr_x'][image_name]))))
            incidence_pr_y = np.array(Image.open(io.BytesIO(np.array(hfw['incidence_pr_y'][image_name]))))
            return cvt_pngs_to_incidence(incidence_pr_x, incidence_pr_y)
        else:
            return None

    def read_jpg_image(self, image_name):
        hfw = h5py.File(self.h5pypath, "r")
        rgb = Image.open(io.BytesIO(np.array(hfw['rgb'][image_name])))
        return rgb

    def read_pose_w2c_gt(self, idx_name):
        hfw = h5py.File(self.h5pypath, "r")
        if 'pose_w2c_gt' in hfw:
            idx_name = "{}.txt".format(idx_name)
            return np.array(hfw['pose_w2c_gt'][idx_name])
        else:
            return None

    def read_pose_i2j_gt(self, pair_name):
        hfw = h5py.File(self.h5pypath, "r")
        if 'pose_i2j_gt' in hfw:
            pair_name = "{}.txt".format(pair_name)
            return np.array(hfw['pose_i2j_gt'][pair_name])
        else:
            return None

    def read_intrinsic_gt(self, idx_name):
        hfw = h5py.File(self.h5pypath, "r")
        if 'intrinsic_gt' in hfw:
            idx_name = "{}.txt".format(idx_name)
            return np.array(hfw['intrinsic_gt'][idx_name])
        else:
            return None

    def read_intrinsic_et(self, idx_name):
        hfw = h5py.File(self.h5pypath, "r")
        idx_name = "{}.txt".format(idx_name)
        return np.array(hfw['intrinsic_et'][idx_name])

    def get_coords_src(self, hs, ws):
        xx, yy = np.meshgrid(
            np.linspace(-1 + 1 / ws, 1 - 1 / ws, ws), np.linspace(-1 + 1 / hs, 1 - 1 / hs, hs), indexing="xy"
        )
        coords_src = np.stack([xx, yy], axis=-1)
        return coords_src

    def read_corres(self, pair_name):
        from MargBA.corres_estimator.corre_utils import png2corres
        hfw = h5py.File(self.h5pypath, "r")
        h5group = hfw['corres_i2j'][pair_name]
        coordsjx = np.array(Image.open(io.BytesIO(np.array(h5group["{}_x.png".format(pair_name)]))))
        coordsjy = np.array(Image.open(io.BytesIO(np.array(h5group["{}_y.png".format(pair_name)]))))
        certainty = np.array(Image.open(io.BytesIO(np.array(h5group["{}_conf.png".format(pair_name)]))))

        hs, ws = certainty.shape

        coordsjx, coordsjy, certainty = png2corres(coordsjx, coordsjy, certainty)
        coords_dst = np.stack([coordsjx, coordsjy], axis=-1)
        coords_src = self.get_coords_src(hs, ws)

        return coords_src, coords_dst, certainty

    def read_visibility(self, pair_name):
        hfw = h5py.File(self.h5pypath, "r")
        return np.array(hfw['visibility_i2j']["{}.txt".format(pair_name)]).item()