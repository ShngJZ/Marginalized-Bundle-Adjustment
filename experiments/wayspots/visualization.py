"""WaysPots visualization script with multiprocessing support.

This script generates visualizations for ScanNet scenes using multiple processes,
with each process handling a subset of scenes based on its rank.
"""

import os, glob, natsort
import argparse
import sys
import torch.multiprocessing as mp

prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)
from MargBA.geometry.export2ply import export2ply


def process_scene(scene: str, args: argparse.Namespace, rank: int) -> None:
    """Process a single scene.

    Args:
        scene: Scene name
        args: Command line arguments
        rank: Process rank for logging
    """
    preprocess_location = os.path.join(
        args.output_location,
        f"{args.depth_model}_{args.corres_model}"
    )

    # Set up paths for current scene and sequence
    dst_perscene = os.path.join(preprocess_location, scene)  # Preprocessed data output path
    sfm_perscene = os.path.join(f"{preprocess_location}_sfm", scene)  # SFM output path

    h5path = os.path.join(dst_perscene, f"{scene}.hdf5")
    pr_modelpath = os.path.join(sfm_perscene, "pr_pose_model_fine.ckpt")
    gt_modelpath = os.path.join(sfm_perscene, "gt_pose_model.ckpt")

    if not args.mute:
        print(f"[Process {rank}] Processing scene: {scene}")

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

    export2ply(
        h5path=h5path,
        scene=scene,
        framerate=45,
        pr_modelpath=pr_modelpath,
        gt_modelpath=gt_modelpath,
        sharefocal=False,
        mute=args.mute,
        marker_query_map=marker_query_map
    )


def worker(rank: int, world_size: int, scenes: list, args: argparse.Namespace) -> None:
    """Worker function for parallel scene processing.

    Args:
        rank: Process rank
        world_size: Total number of processes
        scenes: List of all scenes
        args: Command line arguments
    """
    # Get subset of scenes for this process
    process_scenes = scenes[rank::world_size]

    if not args.mute:
        print(f"Process {rank} handling {len(process_scenes)} scenes")

    # Process each assigned scene
    for scene in process_scenes:
        process_scene(scene, args, rank)


def main():
    parser = argparse.ArgumentParser(description='ScanNet Visualization with Multiprocessing')
    parser.add_argument(
        '--data-root',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/wayspots"
    )
    parser.add_argument(
        '--output-location',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/wayspots"
    )
    parser.add_argument(
        '--depth-model',
        type=str,
        choices=['ZoeDepth', 'UniDepth', 'DUSt3R'],
        default='DUSt3R'
    )
    parser.add_argument(
        '--corres-model',
        type=str,
        choices=['RoMa', 'MASt3R'],
        default='RoMa'
    )
    parser.add_argument(
        '--calibrated',
        action="store_true",
        help="Enable calibrated camera mode (default: False)"
    )

    parser.add_argument(
        '--num-processes',
        type=int,
        default=10,
        help='Number of parallel processes'
    )
    parser.add_argument(
        '--mute',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Read scene sequences
    scenes = glob.glob(os.path.join(args.data_root, "*"))
    scenes = [x for x in scenes if os.path.isdir(x)]  # Filter only directories
    scenes = [os.path.basename(x) for x in scenes]    # Get base names
    scenes = natsort.natsorted(scenes)  # Natural sort

    # Initialize multiprocessing
    world_size = min(args.num_processes, len(scenes))
    if not args.mute:
        print(f"Processing {len(scenes)} scenes using {world_size} processes")

    # Start processes
    mp.spawn(
        worker,
        args=(world_size, scenes, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    # Required for Windows support
    mp.set_start_method('spawn', force=True)
    main()
