"""IMC2021 visualization script with multiprocessing support.

This script generates visualizations for IMC2021 scenes using multiple processes,
with each process handling a subset of scenes based on its rank.
"""

import os
import argparse
import sys
import glob
import random
import torch.multiprocessing as mp

prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)
from MargBA.geometry.export2ply import export2ply


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

def process_scene(scene_path: str, args: argparse.Namespace, rank: int) -> None:
    """Process a single scene.

    Args:
        scene_path: Path to scene database file
        args: Command line arguments
        rank: Process rank for logging
    """
    scene = os.path.splitext(os.path.basename(scene_path))[0]
    
    preprocess_location = os.path.join(
        args.output_location,
        f"{args.depth_model}_{args.corres_model}"
    )
    
    dst_perscene = os.path.join(preprocess_location, scene)
    sfm_perscene = os.path.join(f"{preprocess_location}_sfm", scene)
    
    h5path = os.path.join(dst_perscene, f"{scene}.db.hdf5")
    pr_modelpath = os.path.join(sfm_perscene, "pr_pose_model_fine.ckpt")
    
    if not os.path.exists(pr_modelpath):
        if not args.mute:
            print(f"[Process {rank}] Skipping {scene}: pr_modelpath not found")
        return
    
    sharefocal, _ = get_scene_params(scene_path)
    
    if not args.mute:
        print(f"[Process {rank}] Processing scene: {scene}")
    
    export2ply(
        h5path=h5path,
        scene=scene,
        framerate=3,
        pr_modelpath=pr_modelpath,
        sharefocal=sharefocal,
        mute=args.mute
    )


def worker(rank: int, world_size: int, scenes: list, args: argparse.Namespace) -> None:
    """Worker function for parallel scene processing.

    Args:
        rank: Process rank
        world_size: Total number of processes
        scenes: List of all scene paths
        args: Command line arguments
    """
    # Get subset of scenes for this process
    process_scenes = scenes[rank::world_size]
    
    if not args.mute:
        print(f"Process {rank} handling {len(process_scenes)} scenes")
    
    # Process each assigned scene
    for scene_path in process_scenes:
        process_scene(scene_path, args, rank)


def main():
    parser = argparse.ArgumentParser(description='IMC2021 Visualization with Multiprocessing')
    parser.add_argument(
        '--data-root',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/fastmap_sfm"
    )
    parser.add_argument(
        '--output-location',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm"
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

    # Collect all scene database files
    scenes = [d for d in glob.glob(os.path.join(args.data_root, "databases/*.db"))]
    random.shuffle(scenes)
    
    if not scenes:
        print("No scene database files found!")
        return
    
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
