import os, argparse, sys, copy, glob
import torch
import random
import numpy as np
import socket


def get_scene_params(scene_path):
    """
    Determine the sharefocal and calibrated parameters based on the scene path.

    Args:
        scene_path: Path to the scene database file

    Returns:
        tuple: (sharefocal, calibrated) parameters
    """
    scene_name = os.path.basename(scene_path)

    # For eth3d_dslr scenes
    if scene_name.startswith("eth3d_dslr_"):
        return False, True

    # For nerf-osr (nosr_), dploy_house4, urbanscene (urbn_), eyeful tower (eft_)
    if (scene_name.startswith("nosr_") or
            scene_name.startswith("dploy_house4") or
            scene_name.startswith("urbn_") or
            scene_name.startswith("eft_")):
        return False, False

    # For all other scenes
    return True, False


prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)
from MargBA.datasets import HDF5Reader
from MargBA.poses_models.twoview_sfm_initialization import twoview_sfm_initialization

def find_free_port(start_port=12362, end_port=13962, max_attempts=100):
    """
    Find a free port in the given range.

    Args:
        start_port: Lower bound of port range
        end_port: Upper bound of port range
        max_attempts: Maximum number of attempts to find a free port

    Returns:
        A free port number as string, or None if no free port is found
    """
    used_ports = set()
    attempt = 0

    while attempt < max_attempts:
        # Generate a random port in the specified range
        port = np.random.randint(start_port, end_port)

        # Skip if we've already tried this port
        if port in used_ports:
            continue

        used_ports.add(port)
        attempt += 1

        # Check if the port is available
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Try to bind to the port
                s.bind(('localhost', port))
                return str(port)
            except socket.error:
                # Port is in use, try another one
                continue

    raise RuntimeError(f"Could not find a free port after {max_attempts} attempts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')
    parser.add_argument(
        '--data-root', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/fastmap_sfm"
    )
    parser.add_argument(
        '--output-location', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm"
    )
    parser.add_argument(
        '--depth-model', type=str, choices=['ZoeDepth', 'UniDepth', 'DUSt3R'], default='DUSt3R'
    )
    parser.add_argument(
        '--corres-model', type=str, choices=['RoMa', 'MASt3R'], default='RoMa'
    )

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    estimate_constant = 3500000  # Constant for memory estimation

    scenes = [d for d in glob.glob(os.path.join(args.data_root, "databases/*.db"))]
    random.shuffle(scenes)
    preprocess_location = os.path.join(
        args.output_location, f"{args.depth_model}_{args.corres_model}"
    )
    
    params = {
        "data_root": args.data_root,
        "lr": 1e-3,
        "sample_num": 1000,
        "log_freq": 500,
        "max_pair_residual": 40,
        "min_corres_conf": 0.0,
        "wt_gt": False,
        "evaluate_function": None,
        "twoview_ransac_th": 0.1,
        "camera_model": "SIMPLE_PINHOLE",
        "sharefocal": True,
        "calibrated": False,
        "read_init_pose": False,
        "mapfree": False,
        "maxpair": np.inf,
        "per_gpu_ram": 32.0,
        "max_sample_num": 10000,  # Maximum number of samples
    }

    for scene_path in scenes:
        # Extract just the scene name (without extension) from the path
        scene = os.path.splitext(os.path.basename(scene_path))[0]
        dst_perscene = os.path.join(preprocess_location, scene)
        scene_db = f"{scene}.db"
        sfm_perscene = os.path.join(f"{preprocess_location}_sfm_init", scene)

        # Check if dst_perscene directory exists (preprocessing completed)
        if not os.path.exists(dst_perscene):
            print(f"Scene {scene} not preprocessed yet (dst_perscene not available). Skipping.")
            continue
        
        if os.path.exists(sfm_perscene):
            continue

        os.makedirs(sfm_perscene, exist_ok=True)

        # Update sharefocal and calibrated parameters based on scene path
        params['sharefocal'], params['calibrated'] = get_scene_params(scene_path)
        print(f"Processing scene: {scene}, sharefocal: {params['sharefocal']}, calibrated: {params['calibrated']}")
        print(f"Export to {sfm_perscene}")

        hdf5reader = HDF5Reader(
            os.path.join(dst_perscene, f"{scene_db}.hdf5")
        )
        nfrm = len(hdf5reader.get_all_images())

        params_perscene_perstage = copy.deepcopy(params)
        params_perscene_perstage.update(
            {
                "preprocess_location": dst_perscene,
                "sfm_location": sfm_perscene,
                "hdf5name": scene_db,
                "nfrm": nfrm
            }
        )

        # Initialize two-view structure from motion
        pr_pose_model, _ = twoview_sfm_initialization(
            os.path.join(dst_perscene, f"{scene_db}.hdf5"),
            params_perscene_perstage,
            hasgt=False,
            largescene=True
        )
        torch.save(
            pr_pose_model.cpu().state_dict(),
            os.path.join(sfm_perscene, "pr_pose_model_init.ckpt")
        )

