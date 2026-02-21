import os, argparse, sys, glob, torch, random, tqdm, copy, natsort, pickle, time
import numpy as np
import torch.multiprocessing as mp

# Set project root path and add to system path
prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)

# Import required modules
from MargBA.BA.transform_utils import read_connection_pair
from MargBA.datasets.hdf5_utils import HDF5Reader
from MargBA.poses_models import GlobalOptimizationPoseParameters
from MargBA.BA.bundle_adjustment import bundle_adjustment
from experiments.wayspots.utils import pose_initialization
from experiments.wayspots.evaluation import wayspots_evaluate


if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')

    # Define command line arguments
    parser.add_argument(
        '--data-root',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/wayspots",
        help='Root directory containing 7scenes dataset'
    )
    parser.add_argument(
        '--output-location',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/wayspots",
        help='Output directory for processed data'
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

    # Read scene sequences
    scenes = glob.glob(os.path.join(args.data_root, "*"))
    scenes = [x for x in scenes if os.path.isdir(x)]  # Filter only directories
    scenes = [os.path.basename(x) for x in scenes]    # Get base names
    scenes = natsort.natsorted(scenes)  # Natural sort

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
        "evaluate_function": wayspots_evaluate,    # Evaluation function
        "twoview_ransac_th": 0.5,                  # RANSAC threshold
        "camera_model": "SIMPLE_PINHOLE",          # Camera model type
        "sharefocal": False,                       # Share focal length
        "calibrated": True,                        # Whether camera is calibrated
        "read_init_pose": True,                    # Read initial pose
        "maxpair": np.inf,                       # Maximum number of pairs
        "per_gpu_ram": 32,                         # GPU memory in GB
        "weight2qry": 1.0,                         # Weight for query-to-query connections,
        "mapfree": True,                           # Flag indicating whether a pre-built map is available for optimization
        "lr_intrinsic_boost": 1,                   # Set larger lr for intrinsic
        "gradient_mask_exemplar": ["depth_adj", "depth_bias", "depth_focalx", "depth_focaly"],  # Network components that always receive gradients (exempt from gradient masking)
    }

    # Initialize GPU settings
    world_size = torch.cuda.device_count()
    estimate_constant = 3500000  # Constant for memory estimation
    # world_size = 1

    random.shuffle(scenes)
    for scene in scenes:
        time.sleep(random.uniform(0, 2))  # Random delay to avoid race conditions across GPUs

        # Set up paths for current scene and sequence
        dst_perscene = os.path.join(preprocess_location, scene)  # Preprocessed data output path
        sfm_perscene = os.path.join(f"{preprocess_location}_sfm", scene)  # SFM output path
        src_perscene = os.path.join(args.data_root, scene)  # Source data path

        # Create SFM output directory if not exists
        os.makedirs(sfm_perscene, exist_ok=True)

        # Load connection pairs from HDF5 file
        h5path = os.path.join(dst_perscene, "{}.hdf5".format(scene))
        assert os.path.exists(h5path), "HDF5 file not found"
        connections = read_connection_pair(h5path)
        print(f"Total connections {len(connections)}")
        # Create SFM output directory if not exists
        os.makedirs(sfm_perscene, exist_ok=True)

        # Check if initialization file exists, if not perform pose initialization
        init_file = os.path.join(sfm_perscene, 'init.pickle')
        if not os.path.exists(init_file):
            # Initialize poses and intrinsics with minimum correspondence confidence
            intrinsics, poses_w2c_gt, poses_w2c_pr = pose_initialization(
                dst_perscene,
                min_corres_conf=params['min_corres_conf']
            )
            # Save initialization data to pickle file
            pickle.dump(
                {
                    "intrinsics": intrinsics,
                    "poses_w2c_gt": poses_w2c_gt,
                    "poses_w2c_pr": poses_w2c_pr
                },
                open(init_file, 'wb')
            )

        # Verify initialization file was created successfully
        assert os.path.exists(init_file), "Initialization file not found"

        # Create mapping between image indices and their query/map status
        idx2fname_mapper_path = os.path.join(dst_perscene, 'mapper.txt')
        idx2fname_mapper = open(idx2fname_mapper_path).readlines()
        marker_query_map = list()
        for x in idx2fname_mapper:
            x = x.rstrip('\n')
            if 'test' in x:
                marker_query_map.append('qry')
            elif 'train' in x:
                marker_query_map.append('map')
            else:
                raise ValueError(f"{x} not found in map and query")

        # Load initialization data from pickle file
        with open(init_file, 'rb') as f:
            initialization = pickle.load(f)
        intrinsics = initialization['intrinsics']
        poses_w2c_gt = initialization['poses_w2c_gt']
        poses_w2c_pr = initialization['poses_w2c_pr']

        # Get total number of frames
        # Initialize HDF5 reader and get all image names
        reader = HDF5Reader(h5path)
        img_names = reader.get_all_images()
        nfrm = len(img_names)

        # Create and save ground truth pose model
        gt_pose_model = GlobalOptimizationPoseParameters(
            nfrm=nfrm,
            optimizefocal=False,
            sharefocal=params['sharefocal'],
            intrinsic=intrinsics
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
            intrinsic=intrinsics,
            gradient_mask=gradient_mask,  # Only optimize query frames,
            gradient_mask_exemplar=params["gradient_mask_exemplar"] if params["mapfree"] else None,
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

        # Create stage-specific parameters for bundle adjustment
        params_perscene_perstage = copy.deepcopy(params)
        params_perscene_perstage.update(
            {
                "preprocess_location": dst_perscene,  # Path to preprocessed data
                "sfm_location": sfm_perscene,         # Output path for SFM results
                "hdf5name": scene,                      # Sequence name
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
                        "iterations": 50000
                    },
                },
                "master_port": str(np.random.randint(12362, 12562))  # Random port for distributed training
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
                        "iterations": 10000,
                    },
                    "cdf_euclidean": {  # Euclidean-space loss configuration
                        "maxpx": 30.0,
                        "bins": 600,
                        "gradient_smooth": 3,
                        "iterations": 10000,
                    },
                },
                "master_port": str(np.random.randint(12362, 12562))  # Random port for distributed training
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