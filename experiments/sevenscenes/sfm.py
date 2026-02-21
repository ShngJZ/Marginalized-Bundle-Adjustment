import os, argparse, sys, torch, random, tqdm, copy
import numpy as np
import torch.multiprocessing as mp
import socket
import time

# Set project root path and add to system path
prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)

# Import required modules
from MargBA.BA.transform_utils import read_connection_pair
from MargBA.datasets.hdf5_utils import HDF5Reader
from MargBA.poses_models import GlobalOptimizationPoseParameters
from MargBA.BA.bundle_adjustment import bundle_adjustment
from experiments.sevenscenes.utils import read_scene_query_seqs, txt2seqs, acquire_marker_query_map
from experiments.sevenscenes.evaluation import sevenscenes_evaluate

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
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')

    # Define command line arguments
    parser.add_argument(
        '--data-root',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/7scenes",
        help='Root directory containing 7scenes dataset'
    )
    parser.add_argument(
        '--output-location',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/7scenes",
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--initpose-location',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/7scenes/mast3r_pose_init",
        help='Directory containing initial pose estimates'
    )
    parser.add_argument(
        '--depth-model',
        type=str,
        choices=['ZoeDepth', 'UniDepth', 'DUSt3R'],
        default='DUSt3R',
        help='Depth estimation model to use'
    )
    parser.add_argument(
        '--corres-model',
        type=str,
        choices=['RoMa', 'MASt3R'],
        default='RoMa',
        help='Correspondence estimation model to use'
    )

    # Parse arguments
    args = parser.parse_args()

    # Read scene sequences and queries
    seqs_qrys = read_scene_query_seqs(args.data_root)

    # Set preprocessing output location
    preprocess_location = os.path.join(
        args.output_location, f"{args.depth_model}_{args.corres_model}"
    )

    # Configuration parameters
    params = {
        "data_root": args.data_root,               # Dataset root path
        "lr": 1e-3,                                # Learning rate
        "max_sample_num": 10000,                   # Maximum number of samples
        "log_freq": 500,                           # Logging frequency
        "max_pair_residual": 40,                   # Maximum residual for pairs
        "min_corres_conf": 0.0,                    # Minimum correspondence confidence
        "wt_gt": True,                             # Whether to use ground truth
        "evaluate_function": sevenscenes_evaluate,  # Evaluation function
        "twoview_ransac_th": 0.5,                  # RANSAC threshold
        "camera_model": "SIMPLE_PINHOLE",          # Camera model type
        "sharefocal": True,                        # Share focal length
        "calibrated": True,                        # Whether camera is calibrated
        "maxpair": np.inf,                       # Maximum number of pairs
        "per_gpu_ram": 32,                         # GPU memory in GB
        "weight2qry": 0.5,                         # Weight for query-to-query connections
        "mapfree": False,                          # Flag indicating whether a pre-built map is available for optimization,
        "lr_intrinsic_boost": 1,                   # Set larger lr for intrinsic
    }

    # Initialize GPU settings
    world_size = torch.cuda.device_count()
    estimate_constant = 3500000  # Constant for memory estimation

    random.shuffle(seqs_qrys)
    for scene, seq in seqs_qrys:
        time.sleep(random.uniform(0, 2))  # Random delay to avoid race conditions across GPUs

        # Set up paths for current scene and sequence
        dst_perscene = os.path.join(preprocess_location, f"{scene}_{seq}")  # Preprocessed data output path
        sfm_perscene = os.path.join(f"{preprocess_location}_sfm", f"{scene}_{seq}")  # SFM output path
        src_perscene = os.path.join(args.data_root, scene, seq)  # Source data path

        if os.path.exists(sfm_perscene):
            print(f"Skipping already processed scene: {scene}, sequence: {seq}")
            continue
        # Create SFM output directory if not exists
        os.makedirs(sfm_perscene, exist_ok=True)
        print(f"Processing scene: {scene}, sequence: {seq}")


        # Load connection pairs from HDF5 file
        h5path = os.path.join(dst_perscene, "{}.hdf5".format(seq))
        assert os.path.exists(h5path), "HDF5 file not found"
        connections = read_connection_pair(h5path)
        print(f"Total connections {len(connections)}")

        # Load test and train sequences from split files
        seqs_qry = txt2seqs(os.path.join(os.path.dirname(src_perscene), 'TestSplit.txt'))
        seqs_map = txt2seqs(os.path.join(os.path.dirname(src_perscene), 'TrainSplit.txt'))

        # Create mapping between image indices and their query/map status
        idx2fname_mapper_path = os.path.join(dst_perscene, 'mapper.txt')
        idx2fname_mapper = open(idx2fname_mapper_path).readlines()
        marker_query_map = acquire_marker_query_map(
            idx2fname_mapper,
            seqs_map,
            seqs_qry
        )

        # Initialize HDF5 reader and get all image names
        reader = HDF5Reader(h5path)
        img_names = reader.get_all_images()

        # Load ground truth intrinsics for all images
        intrinsics_gt = [reader.read_intrinsic_gt(image_name) for image_name in img_names]

        # Load ground truth world-to-camera poses
        poses_w2c_gt = [reader.read_pose_w2c_gt(image_name) for image_name in img_names]

        # Initialize predicted poses (use GT for map frames, load from file for query frames)
        poses_w2c_pr = [np.zeros_like(x) for x in poses_w2c_gt]
        for entry in idx2fname_mapper:
            idx, fname = entry[0:-1].split(' ')
            idx = int(idx)
            if marker_query_map[idx] == 'map':
                poses_w2c_pr[idx] = poses_w2c_gt[idx]  # Use GT pose for map frames
            elif marker_query_map[idx] == 'qry':
                poses_w2c_pr[idx] = np.loadtxt(
                    os.path.join(
                        args.initpose_location, scene, fname.replace('.color.png', '.txt')
                    )
                )  # Load initial pose for query frames
            else:
                raise ValueError("Invalid marker type")

        # Get total number of frames
        nfrm = len(img_names)

        # Create and save ground truth pose model
        gt_pose_model = GlobalOptimizationPoseParameters(
            nfrm=nfrm,
            optimizefocal=False,
            sharefocal=params['sharefocal'],
            intrinsic=intrinsics_gt[0]
        )
        for i in range(nfrm):
            gt_pose_model.update_pose_adjustment(
                nodeid=i,
                newpose=torch.from_numpy(poses_w2c_gt[i]).float(),
                newadjustment=1.0,
                newbias=0.0
            )
        torch.save(
            gt_pose_model.state_dict(),
            os.path.join(sfm_perscene, "gt_pose_model.ckpt")
        )

        # Create and save predicted pose model (initial state)
        gradient_mask = torch.Tensor([x == 'qry' for x in marker_query_map]).float()
        pr_pose_model = GlobalOptimizationPoseParameters(
            nfrm=nfrm,
            optimizefocal=(not params["calibrated"]),
            sharefocal=params['sharefocal'],
            intrinsic=intrinsics_gt[0],
            gradient_mask=gradient_mask  # Only optimize query frames
        )
        for i in range(nfrm):
            pr_pose_model.update_pose_adjustment(
                nodeid=i,
                newpose=torch.from_numpy(poses_w2c_pr[i]).float(),
                newadjustment=1.0,
                newbias=0.0
            )
        torch.save(
            pr_pose_model.state_dict(),
            os.path.join(sfm_perscene, "pr_pose_model_init.ckpt")
        )
        # =============== Coarse Stage ===============
        # Split connections into query-node pairs and distribute across GPUs
        # Create a dictionary mapping query nodes to their connections
        pair_qry_node = dict()
        for idx1, idx2 in tqdm.tqdm(connections):
            # Only consider connections involving query nodes
            if marker_query_map[idx1] == 'qry':
                if idx1 not in pair_qry_node:
                    pair_qry_node[idx1] = list()
                pair_qry_node[idx1].append([idx1, idx2])  # Store as [src_idx, dst_idx]
            if marker_query_map[idx2] == 'qry':
                if idx2 not in pair_qry_node:
                    pair_qry_node[idx2] = list()
                pair_qry_node[idx2].append([idx1, idx2])
            # Ensure at least one node in each connection is a query node
            assert marker_query_map[idx1] == 'qry' or marker_query_map[idx2] == 'qry'

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
                    weight_coarse = np.zeros(2)
                    # Only apply gradients to nodes assigned to this GPU
                    if srcid in per_gpu_assigned_node[gpuid]:
                        weight_coarse[0] = 1.0  # Source weight
                    if dstid in per_gpu_assigned_node[gpuid]:
                        weight_coarse[1] = 1.0  # Target weight
                    # Reduce weight for query-to-query connections
                    if (marker_query_map[srcid] == 'qry') and (marker_query_map[dstid] == 'qry'):
                        weight_coarse = weight_coarse * params["weight2qry"]
                    weight_per_pair[tuple([srcid, dstid])] = weight_coarse
            per_gpu_weight_on_pair_dict[gpuid] = weight_per_pair

        # Calculate number of samples per GPU based on memory constraints
        nconnections = max([len(x) for x in per_gpu_pair])
        sample_num = int(min(params["per_gpu_ram"] / nconnections * estimate_constant, params["max_sample_num"]))
        sample_num = sample_num if sample_num % 2 == 0 else sample_num - 1  # Ensure even number

        # Find a free port for distributed training
        master_port = find_free_port(12362, 12962)

        # Create stage-specific parameters for bundle adjustment
        params_perscene_perstage = copy.deepcopy(params)
        params_perscene_perstage.update(
            {
                "preprocess_location": dst_perscene,  # Path to preprocessed data
                "sfm_location": sfm_perscene,         # Output path for SFM results
                "hdf5name": seq,                      # Sequence name
                "per_gpu_pair": per_gpu_pair,         # Connections assigned to each GPU
                "marker_query_map": marker_query_map, # Mapping of nodes to query/map status
                "per_gpu_weight_on_pair_dict": per_gpu_weight_on_pair_dict,  # Weight dictionary
                "sample_num": sample_num,             # Number of samples per GPU
                "stage": "coarse",                    # Current optimization stage
                "nfrm": nfrm,                         # Total number of frames
                "gradient_mask": gradient_mask,       # Mask for gradient computation
                "losses": {
                    "cdf_log_subgraph": {             # Loss configuration
                        "maxpx": 15.0,
                        "bins": 250,
                        "gradient_smooth": 2,
                        "iterations": 10000
                    },
                },
                "master_port": master_port  # Use the found free port
            }
        )

        # Run bundle adjustment in parallel across GPUs
        print(f"stage {params_perscene_perstage['stage']} sample {sample_num} points")
        mp.spawn(bundle_adjustment, args=(
            world_size,
            params_perscene_perstage
        ), nprocs=world_size, join=True)

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
                    # Reduce weight for query-to-query connections
                    if (marker_query_map[srcid] == 'qry') and (marker_query_map[dstid] == 'qry'):
                        weight_fine = weight_fine * params["weight2qry"]
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