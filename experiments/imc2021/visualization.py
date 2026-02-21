"""IMC2021 visualization script with multiprocessing support.

This script generates visualizations for IMC2021 scenes using multiple processes,
with each process handling a subset of scenes based on its rank.
"""

import os
import argparse
import sys
import glob
import torch.multiprocessing as mp

prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)
from MargBA.geometry.export2ply import export2ply


def process_scene(scene_bag: str, args: argparse.Namespace, rank: int) -> None:
    """Process a single scene bag.

    Args:
        scene_bag: Path to scene bag
        args: Command line arguments
        rank: Process rank for logging
    """
    scene = os.path.basename(os.path.dirname(scene_bag))
    bag = os.path.basename(scene_bag)
    
    preprocess_location = os.path.join(
        args.output_location,
        f"{args.depth_model}_{args.corres_model}"
    )
    
    dst_perscene = os.path.join(preprocess_location, f"{scene}_{bag}")
    sfm_perscene = os.path.join(f"{preprocess_location}_sfm", f"{scene}_{bag}")

    h5path = os.path.join(dst_perscene, f"{bag}.hdf5")
    pr_modelpath = os.path.join(sfm_perscene, "pr_pose_model_fine.ckpt")
    gt_modelpath = os.path.join(sfm_perscene, "gt_pose_model.ckpt")

    if not args.mute:
        print(f"[Process {rank}] Processing scene: {scene}, bag: {bag}")

    export2ply(
        h5path=h5path,
        scene=f"{scene}_{bag}",
        framerate=3,
        pr_modelpath=pr_modelpath,
        gt_modelpath=gt_modelpath,
        sharefocal=False,
        mute=args.mute
    )


def worker(rank: int, world_size: int, scene_bags: list, args: argparse.Namespace) -> None:
    """Worker function for parallel scene processing.

    Args:
        rank: Process rank
        world_size: Total number of processes
        scene_bags: List of all scene bags
        args: Command line arguments
    """
    # Get subset of scenes for this process
    process_scene_bags = scene_bags[rank::world_size]
    
    if not args.mute:
        print(f"Process {rank} handling {len(process_scene_bags)} scene bags")
    
    # Process each assigned scene
    for scene_bag in process_scene_bags:
        process_scene(scene_bag, args, rank)


def main():
    parser = argparse.ArgumentParser(description='IMC2021 Visualization with Multiprocessing')
    parser.add_argument(
        '--data-root',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/imc2021"
    )
    parser.add_argument(
        '--output-location',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/imc2021"
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

    # Collect all scene bags
    all_scenes_bags = []
    for scene in scenes:
        bags = glob.glob(os.path.join(args.data_root, scene, "*"))
        all_scenes_bags.extend(bags)

    # Initialize multiprocessing
    world_size = min(args.num_processes, len(all_scenes_bags))
    if not args.mute:
        print(f"Processing {len(all_scenes_bags)} scene bags using {world_size} processes")

    # Start processes
    mp.spawn(
        worker,
        args=(world_size, all_scenes_bags, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    # Required for Windows support
    mp.set_start_method('spawn', force=True)
    main()
