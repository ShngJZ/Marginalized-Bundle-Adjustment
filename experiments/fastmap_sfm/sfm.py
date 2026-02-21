import os, argparse, sys, copy, tqdm, glob
import torch
import random
import numpy as np
import torch.multiprocessing as mp
import socket
import shutil

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
from MargBA.BA.bundle_adjustment import bundle_adjustment

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
        sfm_perscene = os.path.join(f"{preprocess_location}_sfm", scene)

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

        # Copy pre-computed initialization from sfm_init location
        sfm_init_perscene = os.path.join(f"{preprocess_location}_sfm_init", scene)
        init_ckpt_src = os.path.join(sfm_init_perscene, "pr_pose_model_init.ckpt")
        init_ckpt_dst = os.path.join(sfm_perscene, "pr_pose_model_init.ckpt")
        
        assert os.path.exists(init_ckpt_src)
        shutil.copy2(init_ckpt_src, init_ckpt_dst)
        print(f"Copied initialization from {init_ckpt_src} to {init_ckpt_dst}")

    
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
    
        pair_qry_node = dict()
        for idx1, idx2 in tqdm.tqdm(connections):
            if idx1 not in pair_qry_node:
                pair_qry_node[idx1] = list()
            pair_qry_node[idx1].append([idx1, idx2])  # Store as [src_idx, dst_idx]
            if idx2 not in pair_qry_node:
                pair_qry_node[idx2] = list()
            pair_qry_node[idx2].append([idx1, idx2])
    
        # Initialize lists to store pairs and assigned nodes per GPU
        per_gpu_pair = [[] for x in range(world_size)]  # Connections assigned to each GPU
        per_gpu_assigned_node = [[] for x in range(world_size)]  # Query nodes assigned to each GPU
    
        # Distribute query nodes and their connections across GPUs for load balancing
        for node in pair_qry_node:
            # Find GPU with least assigned pairs
            gpu_assigned_pair = [len(x) for x in per_gpu_pair]
            to_assign_idx = np.argmin(np.array(gpu_assigned_pair))
            # Assign current node's connections to selected GPU
            per_gpu_pair[to_assign_idx].extend(pair_qry_node[node])
            per_gpu_assigned_node[to_assign_idx].extend([node])
    
        # Limit number of pairs per GPU to max allowed value
        per_gpu_pair = [x[0:min(params["maxpair"], len(x))] for x in per_gpu_pair]
        assert all(len(x) > 0 for x in per_gpu_pair), "each gpu allocated pairs should be larger than 0"
    
        # Create weight dictionary for each GPU pair
        per_gpu_weight_on_pair_dict = [None for x in range(world_size)]
        for gpuid in range(world_size):
            per_gpu_pair_num = len(per_gpu_pair[gpuid])
            weight_per_pair = dict()
            for i in range(per_gpu_pair_num):
                i1 = int(per_gpu_pair[gpuid][i][0])
                i2 = int(per_gpu_pair[gpuid][i][1])
                # Consider both directions of each connection
                for srcid, dstid in [[i1, i2], [i2, i1]]:
                    # Initialize weights (2-dim vector for source and target)
                    weight_coarse = np.ones(2)
                    weight_per_pair[tuple([srcid, dstid])] = weight_coarse
            per_gpu_weight_on_pair_dict[gpuid] = weight_per_pair
    
        # Calculate number of samples per GPU based on memory constraints
        nconnections = max([len(x) for x in per_gpu_pair])
        sample_num = int(min(params["per_gpu_ram"] / nconnections * estimate_constant, params["max_sample_num"]))
        sample_num = sample_num if sample_num % 2 == 0 else sample_num - 1  # Ensure even number
    
        # Find a free port for distributed training
        master_port = find_free_port(12362, 12962)
    
        # Create stage-specific parameters for bundle adjustment
        params_perscene_perstage.update(
            {
                "per_gpu_pair": per_gpu_pair,         # Connections assigned to each GPU
                "marker_query_map": marker_query_map, # Mapping of nodes to query/map status
                "per_gpu_weight_on_pair_dict": per_gpu_weight_on_pair_dict,  # Weight dictionary
                "stage": "coarse",                    # Current optimization stage
                "gradient_mask": gradient_mask,       # Mask for gradient computation,
                "lr_intrinsic_boost": 50,             # Set larger lr for intrinsic
                "sample_num": sample_num,             # Number of samples per GPU
                "losses": {
                    "cdf_log_subgraph": {             # Loss configuration
                        "maxpx": 15.0,
                        "bins": 250,
                        "gradient_smooth": 2,
                        "iterations": 50000
                    },
                },
                "master_port": find_free_port()
            }
        )
        mp.spawn(
            bundle_adjustment,
            args=(world_size, params_perscene_perstage),
            nprocs=world_size,
            join=True
        )
    
        # =============== Fine Stage Optimization ===============
        # Distribute connections evenly across GPUs
        per_gpu_pair = [connections[i::world_size] for i in range(world_size)]
        # Limit number of pairs per GPU to max allowed value
        per_gpu_pair = [x[0:min(params["maxpair"], len(x))] for x in per_gpu_pair]
    
        # Calculate number of samples per GPU based on memory constraints
        nconnections = max([len(x) for x in per_gpu_pair])
        sample_num = int(min(params["per_gpu_ram"] / nconnections * estimate_constant, params["max_sample_num"]))
        sample_num = sample_num if sample_num % 2 == 0 else sample_num - 1  # Ensure even number
    
        # Create weight dictionary for each GPU pair
        per_gpu_weight_on_pair_dict = [None for x in range(world_size)]
        for gpuid in range(world_size):
            per_gpu_pair_num = len(per_gpu_pair[gpuid])
            weight_per_pair = dict()
            for i in range(per_gpu_pair_num):
                i1 = int(per_gpu_pair[gpuid][i][0])
                i2 = int(per_gpu_pair[gpuid][i][1])
                # Consider both directions of each connection
                for srcid, dstid in [[i1, i2], [i2, i1]]:
                    weight_fine = np.ones(1)  # Initialize uniform weights
                    weight_per_pair[tuple([srcid, dstid])] = weight_fine
            per_gpu_weight_on_pair_dict[gpuid] = weight_per_pair
    
        master_port = find_free_port(13000, 13600)
    
        # Update parameters for fine stage optimization
        params_perscene_perstage.update(
            {
                "per_gpu_pair": per_gpu_pair,  # Pairs distributed across GPUs
                "marker_query_map": marker_query_map,  # Mapping of nodes to query/map status
                "per_gpu_weight_on_pair_dict": per_gpu_weight_on_pair_dict,  # Weight dictionary
                "sample_num": sample_num,  # Number of samples per GPU
                "stage": "fine",  # Current optimization stage
                "losses": {
                    "cdf_log": {  # Log-space loss configuration
                        "maxpx": 15.0,
                        "bins": 250,
                        "gradient_smooth": 2,
                        "iterations": 10000
                    },
                    "cdf_euclidean": {  # Euclidean-space loss configuration
                        "maxpx": 50.0,
                        "bins": 600,
                        "gradient_smooth": 3,
                        "iterations": 10000
                    },
                },
                "master_port": master_port  # Use the found free port
            }
        )
    
        # Run bundle adjustment in parallel across GPUs
        print(f"stage {params_perscene_perstage['stage']} sample {sample_num} points")
        mp.spawn(
            bundle_adjustment,
            args=(world_size, params_perscene_perstage),
            nprocs=world_size,
            join=True
        )


