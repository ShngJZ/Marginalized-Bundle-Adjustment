import glob
import shutil
import natsort
import numpy as np
import tqdm, loguru, os, time, copy, torch
import PIL.Image as Image
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from MargBA.datasets import initialize_dataset, to_cuda, cvt_monodepth_to_png, cvt_incidence_to_png, monodepth2vls, image2vls, HDF5Reader, HDF5Writer

def init_monodepth_model(depth_method_name):
    supported_methods = ['ZoeDepth', 'UniDepth', 'DUSt3R']
    assert depth_method_name in supported_methods
    if depth_method_name == 'ZoeDepth':
        from MargBA.monodepth_estimator import ZoeDepth
        monodepth_model = ZoeDepth()
    elif depth_method_name == 'UniDepth':
        from MargBA.monodepth_estimator import UniDepth
        monodepth_model = UniDepth()
    elif depth_method_name == 'DUSt3R':
        from MargBA.monodepth_estimator import DUSt3R
        monodepth_model = DUSt3R()
    return monodepth_model

def init_monocalibrator_model():
    from MargBA.monodepth_estimator import WildDUSt3R
    monocalibraor = WildDUSt3R()
    return monocalibraor

def median_scale(monodepth, monodepth_gt):
    selector = monodepth_gt > 0
    gt_scale = torch.median(monodepth_gt[selector]) / torch.median(monodepth[selector])
    monodepth = monodepth * gt_scale
    return monodepth

def intrinsic_et_naive(image):
    _, _, h, w = image.shape
    f = np.sqrt(h ** 2 + w ** 2)
    intrinsic_et = np.eye(3)
    intrinsic_et[0, 0] = f
    intrinsic_et[1, 1] = f
    intrinsic_et[0, 2] = w / 2
    intrinsic_et[1, 2] = h / 2
    return intrinsic_et

@torch.no_grad()
def inference_unary_per_gpu(
        rank,
        world_size,
        data_root,
        dataset,
        output_location,
        depth_method_name
):
    torch.cuda.set_device(rank)
    dataset = initialize_dataset(
        data_root=data_root,
        dataset=dataset,
        mode='monodepth',
        rank=rank,
        world_size=world_size,
        output_location=output_location
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    monodepth_model = init_monodepth_model(depth_method_name)
    monodepth_model.to_cuda(torch.device(f"cuda:{rank}"))

    monocalibrator = init_monocalibrator_model()
    monocalibrator.to_cuda(torch.device(f"cuda:{rank}"))

    intermediate_folder = os.path.join(output_location, "intermediate")
    rgbs_root = os.path.join(intermediate_folder, 'jpgs')
    monodepth_pr_root = os.path.join(intermediate_folder, 'depth_pr')
    monodepth_gt_root = os.path.join(intermediate_folder, 'depth_gt')
    intrinsic_gt_root = os.path.join(intermediate_folder, 'intrinsic_gt')
    intrinsic_et_root = os.path.join(intermediate_folder, 'intrinsic_et')
    pose_w2c_gt_root = os.path.join(intermediate_folder, 'pose_w2c_gt')
    os.makedirs(rgbs_root, exist_ok=True)
    os.makedirs(monodepth_pr_root, exist_ok=True)
    os.makedirs(intrinsic_et_root, exist_ok=True)

    loguru.logger.info("GPU-%d inference %d images with method %s" % (rank, len(dataset), depth_method_name))
    time.sleep(2)
    for (idx, data) in enumerate(tqdm.tqdm(dataloader)):
        data = to_cuda(data)
        bz = len(data['rgb_file'])
        try:
            monodepths, incidence = monodepth_model.inference(data['image'])
        except:
            print("========================================")
            print(f"{data_root}, {data['rgb_file']}, Error!!!")
            print("========================================")
            raise ValueError()

        for i in range(bz):
            image_name = str(data['image_idx'][i].item()).zfill(6)

            if 'depth_gt' in data:
                os.makedirs(monodepth_gt_root, exist_ok=True)
                monodepth_gt = data['depth_gt'][i].squeeze()
                monodepth_gt_png = cvt_monodepth_to_png(monodepth_gt)
                sv_path_gt = os.path.join(monodepth_gt_root, '{}.png'.format(image_name))
                monodepth_gt_png.save(sv_path_gt)

            if True:
                monodepth_pr = monodepths[i].squeeze()
                monodepth_pr_png = cvt_monodepth_to_png(monodepth_pr)
                sv_path_pr = os.path.join(monodepth_pr_root, '{}.png'.format(image_name))
                monodepth_pr_png.save(sv_path_pr)

            if True:
                sv_path_jpg = os.path.join(rgbs_root, '{}.jpg'.format(image_name))
                image2vls(data['image'], viewind=i).save(sv_path_jpg)

            if 'intr' in data:
                os.makedirs(intrinsic_gt_root, exist_ok=True)
                intrinsic_gt = data['intr'][i].cpu().numpy()
                sv_path_intrinsicgt = os.path.join(intrinsic_gt_root, '{}.txt'.format(image_name))
                np.savetxt(sv_path_intrinsicgt, intrinsic_gt)

            if True:
                intrinsic_et = monocalibrator.inference(sv_path_jpg) # intrinsic_et_naive(data['image'])
                sv_path_intrinsicet = os.path.join(intrinsic_et_root, '{}.txt'.format(image_name))
                np.savetxt(sv_path_intrinsicet, intrinsic_et)

            if 'pose_c2w' in data:
                os.makedirs(pose_w2c_gt_root, exist_ok=True)
                pose_c2w_gt = data['pose_c2w'][i].cpu().numpy()
                if np.sum(np.isnan(pose_c2w_gt)) == 0: pose_w2c_gt = np.linalg.inv(pose_c2w_gt)
                sv_path_posegt = os.path.join(pose_w2c_gt_root, '{}.txt'.format(image_name))
                np.savetxt(sv_path_posegt, pose_w2c_gt)
    return

def write_to_hdf5(scene, output_location):
    h5pypath = os.path.join(output_location, f"{scene}.hdf5")
    intermediate_folder = os.path.join(output_location, "intermediate")
    keys_to_add = ["depth_pr", "rgb", "intrinsic_et"]
    for mode in ["depth_gt", "pose_w2c_gt", "intrinsic_gt"]:
        if os.path.exists(os.path.join(intermediate_folder, mode)):
            keys_to_add.append(mode)

    h5pywriter = HDF5Writer(
        h5pypath, 
        keys_to_add=keys_to_add,
        mode="w"
    )

    rgbs = natsort.natsorted(glob.glob(os.path.join(intermediate_folder, "jpgs", "*.jpg")))
    for rgb in rgbs:
        h5pywriter.write_jpg_image(rgb, os.path.basename(rgb))

    monodepth_prs = natsort.natsorted(glob.glob(os.path.join(intermediate_folder, "depth_pr", "*.png")))
    for monodepth_pr in monodepth_prs:
        h5pywriter.write_depth_pr(monodepth_pr, os.path.basename(monodepth_pr))

    if "depth_gt" in keys_to_add:
        monodepth_gts = natsort.natsorted(glob.glob(os.path.join(intermediate_folder, "depth_gt", "*.png")))
        for monodepth_gt in monodepth_gts:
            h5pywriter.write_depth_gt(monodepth_gt, os.path.basename(monodepth_gt))

    if "intrinsic_gt" in keys_to_add:
        intrinsic_gts = natsort.natsorted(glob.glob(os.path.join(intermediate_folder, "intrinsic_gt", "*.txt")))
        for intrinsic_gt in intrinsic_gts:
            h5pywriter.write_intrinsic_gt(intrinsic_gt, os.path.basename(intrinsic_gt))

    if True:
        intrinsic_ets = natsort.natsorted(glob.glob(os.path.join(intermediate_folder, "intrinsic_et", "*.txt")))
        for intrinsic_et in intrinsic_ets:
            h5pywriter.write_intrinsic_et(intrinsic_et, os.path.basename(intrinsic_et))

    if "pose_w2c_gt" in keys_to_add:
        pose_w2c_gts = natsort.natsorted(glob.glob(os.path.join(intermediate_folder, "pose_w2c_gt", "*.txt")))
        for pose_w2c_gt in pose_w2c_gts:
            h5pywriter.write_pose_w2c_gt(pose_w2c_gt, os.path.basename(pose_w2c_gt))

    shutil.rmtree(intermediate_folder)
    assert len(rgbs) == len(monodepth_prs)

def vls_wt_hdf5(scene, output_location):
    h5pypath = os.path.join(output_location, f"{scene}.hdf5")
    h5pyreader = HDF5Reader(
        h5pypath
    )
    all_imgs = h5pyreader.get_all_images()

    vls_folder = os.path.join(output_location, "vls")
    os.makedirs(vls_folder, exist_ok=True)
    
    rgb_folder = os.path.join(vls_folder, "rgb")
    os.makedirs(rgb_folder, exist_ok=True)
    depth_folder = os.path.join(vls_folder, "depth")
    os.makedirs(depth_folder, exist_ok=True)

    to_vls_idx = np.random.choice(
        np.arange(len(all_imgs)),
        size=10,
        replace=True,
    )
    for idx, img in enumerate(all_imgs):
        img = img.rstrip(".jpg")
        
        jpg_name = "{}.jpg".format(img)
        rgb = h5pyreader.read_jpg_image(jpg_name)
        rgb.save(os.path.join(rgb_folder, jpg_name))

        if idx in to_vls_idx:
            w, h = rgb.size
            
            depth_pr = h5pyreader.read_depth_pr(img)
            depth_pr = torch.from_numpy(depth_pr).view([1, 1, h, w])
            depth_pr = monodepth2vls(1 / depth_pr, percentile=90, viewind=0)
            vls_concat = [np.array(rgb), np.array(depth_pr)]

            depth_gt = h5pyreader.read_depth_gt(img)
            if depth_gt is not None:
                depth_gt[depth_gt == 0] = np.inf
                depth_gt = torch.from_numpy(depth_gt).view([1, 1, h, w])
                depth_gt = monodepth2vls(1 / depth_gt, vmax=1.0, viewind=0)
                vls_concat.append(np.array(depth_gt))

            vls_concat = np.concatenate(vls_concat, axis=1)
            Image.fromarray(vls_concat).save(os.path.join(depth_folder, jpg_name))

def inference_unary(
        data_root,
        dataset,
        output_location,
        depth_model
):
    world_size = torch.cuda.device_count()
    mp.spawn(
        inference_unary_per_gpu,
        args=(
            world_size,
            data_root,
            dataset,
            output_location,
            depth_model,
        ),
        nprocs=world_size,
        join=True
    )
    scene = os.path.basename(data_root)
    write_to_hdf5(scene, output_location)
    vls_wt_hdf5(scene, output_location)