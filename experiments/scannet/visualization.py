"""ScanNet visualization script with multiprocessing support.

This script generates visualizations for ScanNet scenes using multiple processes,
with each process handling a subset of scenes based on its rank.
"""

import os
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
    
    dst_perscene = os.path.join(preprocess_location, scene)
    if args.calibrated:
        sfm_perscene = os.path.join(f"{preprocess_location}_sfm_calibrated", scene)
    else:
        sfm_perscene = os.path.join(f"{preprocess_location}_sfm_uncalibrated", scene)

    h5path = os.path.join(dst_perscene, f"{scene}.hdf5")
    pr_modelpath = os.path.join(sfm_perscene, "pr_pose_model_fine.ckpt")
    gt_modelpath = os.path.join(sfm_perscene, "gt_pose_model.ckpt")

    if not args.mute:
        print(f"[Process {rank}] Processing scene: {scene}")

    export2ply(
        h5path=h5path,
        scene=scene,
        framerate=3,
        pr_modelpath=pr_modelpath,
        gt_modelpath=gt_modelpath,
        sharefocal=True,
        mute=args.mute
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
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet"
    )
    parser.add_argument(
        '--output-location',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet"
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

    # Read scene list
    with open(os.path.join(os.path.dirname(__file__), 'scannet.txt'), 'r') as f:
        scenes = [line.strip() for line in f.readlines() if line.strip()]

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
