import os, argparse, sys, glob, copy
import random
import torch
import numpy as np
import torch.multiprocessing as mp
import socket
import time

prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)
from experiments.imc2021.evaluation import imc_evaluate
from MargBA.datasets import HDF5Reader
from MargBA.BA.bundle_adjustment import bundle_adjustment
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
        '--data-root', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/imc2021"
    )
    parser.add_argument(
        '--output-location', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/imc2021"
    )
    parser.add_argument(
        '--depth-model', type=str, choices=['ZoeDepth', 'UniDepth', 'DUSt3R'], default='DUSt3R'
    )
    parser.add_argument(
        '--corres-model', type=str, choices=['RoMa', 'MASt3R'], default='RoMa'
    )

    args = parser.parse_args()
    scenes = [
        'phototourism_british_museum',
        'phototourism_lincoln_memorial_statue',
        'phototourism_milan_cathedral',
        'phototourism_piazza_san_marco',
        'phototourism_st_pauls_cathedral',
        'phototourism_florence_cathedral_side',
        'phototourism_london_bridge',
        'phototourism_mount_rushmore',
        'phototourism_sagrada_familia'
    ]
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    world_size = torch.cuda.device_count()
    assert world_size == 1

    params = {
        "data_root": args.data_root,
        "lr": 1e-3,
        "sample_num": 10000,
        "log_freq": 500,
        "max_pair_residual": 40,
        "min_corres_conf": 0.0,
        "wt_gt": True,
        "evaluate_function": imc_evaluate,
        "twoview_ransac_th": 0.5,
        "camera_model": "SIMPLE_PINHOLE",
        "sharefocal": False,
        "calibrated": False,
        "read_init_pose": False,
        "mapfree": False,
    }

    preprocess_location = os.path.join(
        args.output_location, f"{args.depth_model}_{args.corres_model}"
    )
    all_scenes_bags = list()
    for scene in scenes:
        bags = glob.glob(os.path.join(args.data_root, scene, "*"))
        for bag in bags:
            all_scenes_bags.append(bag)
    random.shuffle(all_scenes_bags)

    for src_perscene in all_scenes_bags:
        time.sleep(random.uniform(0, 2))  # Random delay to avoid race conditions across GPUs

        scene, bag = os.path.basename(os.path.dirname(src_perscene)), os.path.basename(src_perscene)
        dst_perscene = os.path.join(preprocess_location, f"{scene}_{bag}")
        sfm_perscene = os.path.join(f"{preprocess_location}_sfm", f"{scene}_{bag}")
        params.update(
            {
                "preprocess_location": dst_perscene,
                "sfm_location": sfm_perscene,
                "hdf5name": bag
            }
        )
        if os.path.exists(sfm_perscene):
            continue
        os.makedirs(sfm_perscene, exist_ok=True)

        hdf5reader = HDF5Reader(
            os.path.join(dst_perscene, f"{bag}.hdf5")
        )
        nfrm = len(hdf5reader.get_all_images())

        params_perscene_perstage = copy.deepcopy(params)
        params_perscene_perstage.update(
            {
                "preprocess_location": dst_perscene,
                "sfm_location": sfm_perscene,
                "hdf5name": bag,
                "nfrm": nfrm
            }
        )

        # Initialize two-view structure from motion
        pr_pose_model, gt_pose_model = twoview_sfm_initialization(
            os.path.join(dst_perscene, f"{bag}.hdf5"),
            params_perscene_perstage
        )

        torch.save(
            gt_pose_model.cpu().state_dict(),
            os.path.join(sfm_perscene, "gt_pose_model.ckpt")
        )
        torch.save(
            pr_pose_model.cpu().state_dict(),
            os.path.join(sfm_perscene, "pr_pose_model_init.ckpt")
        )

        del pr_pose_model, gt_pose_model
        torch.cuda.empty_cache()

        device = torch.device("cuda")
        allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # in MB
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # in MB
        print(f"CUDA memory allocated: {allocated:.2f} MB")
        print(f"CUDA memory reserved : {reserved:.2f} MB")

        # coarse stage
        connections = hdf5reader.get_all_image_pairs()
        connections = [[int(x.split('_')[0]), int(x.split('_')[1])] for x in connections]

        marker_query_map = ['qry'] * nfrm
        gradient_mask = np.ones(nfrm)
        per_gpu_pair = [connections]
        per_gpu_weight_on_pair_dict = [{tuple([x[0], x[1]]): np.array([1.0, 1.0]) for x in connections}]

        # Find a free port for distributed training
        master_port = find_free_port()
        print(f"Using port {master_port} for coarse stage")

        # Create stage-specific parameters for bundle adjustment
        params_perscene_perstage.update(
            {
                "per_gpu_pair": per_gpu_pair,         # Connections assigned to each GPU
                "marker_query_map": marker_query_map, # Mapping of nodes to query/map status
                "per_gpu_weight_on_pair_dict": per_gpu_weight_on_pair_dict,  # Weight dictionary
                "stage": "coarse",                    # Current optimization stage
                "gradient_mask": gradient_mask,       # Mask for gradient computation,
                "lr_intrinsic_boost": 50,             # Set larger lr for intrinsic
                "losses": {
                    "cdf_log_subgraph": {             # Loss configuration
                        "maxpx": 15.0,
                        "bins": 250,
                        "gradient_smooth": 2,
                        "iterations": 50000
                    },
                },
                "master_port": master_port  # Use the found free port
            }
        )

        mp.spawn(
            bundle_adjustment,
            args=(world_size, params_perscene_perstage),
            nprocs=world_size,
            join=True
        )

        # Wait a moment to ensure port is released
        time.sleep(1)

        # Find another free port for the fine stage
        master_port = find_free_port()
        print(f"Using port {master_port} for fine stage")

        # fine stage
        per_gpu_weight_on_pair_dict = [{tuple([x[0], x[1]]): np.array([1.0]) for x in connections}]
        params_perscene_perstage.update(
            {
                "stage": "fine",  # Current optimization stage,
                "per_gpu_weight_on_pair_dict": per_gpu_weight_on_pair_dict,  # Weight dictionary,
                "lr_intrinsic_boost": 10,  # Set larger lr for intrinsic
                "losses": {
                    "cdf_log": {  # Log-space loss configuration
                        "maxpx": 15.0,
                        "bins": 250,
                        "gradient_smooth": 2,
                        "iterations": 10000,
                    },
                    "cdf_euclidean": {  # Euclidean-space loss configuration
                        "maxpx": 50.0,
                        "bins": 600,
                        "gradient_smooth": 3,
                        "iterations": 10000,
                    },
                },
                "master_port": master_port  # Use the found free port
            }
        )

        mp.spawn(
            bundle_adjustment,
            args=(world_size, params_perscene_perstage),
            nprocs=world_size,
            join=True
        )
