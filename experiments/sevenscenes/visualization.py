"""7Scenes visualization script with multiprocessing support.

Generates visualizations for 7Scenes dataset using multiple processes,
with each process handling a subset of scenes based on its rank.
"""

from pathlib import Path
from typing import List, Tuple
import argparse
import torch.multiprocessing as mp
import natsort

from MargBA.geometry.export2ply import export2ply
from experiments.sevenscenes.utils import read_scene_query_seqs, txt2seqs, acquire_marker_query_map


def process_scene(scene_data: Tuple[str, str], args: argparse.Namespace, rank: int) -> None:
    """Process a single scene for visualization.

    Args:
        scene_data: Tuple of (scene_name, sequence_name)
        args: Command line arguments
        rank: Process rank for logging
    """
    scene, seq = scene_data
    model_suffix = f"{args.depth_model}_{args.corres_model}"
    
    # Setup paths
    output_root = Path(args.output_location)
    data_root = Path(args.data_root)
    
    preprocess_dir = output_root / model_suffix
    sfm_dir = output_root / f"{model_suffix}_sfm"
    source_dir = data_root / scene / seq
    
    scene_dir = preprocess_dir / f"{scene}_{seq}"
    sfm_scene_dir = sfm_dir / f"{scene}_{seq}"

    h5_path = scene_dir / f"{seq}.hdf5"
    pr_model_path = sfm_scene_dir / "pr_pose_model_fine.ckpt"
    gt_model_path = sfm_scene_dir / "gt_pose_model.ckpt"

    if not args.mute:
        print(f"[Process {rank}] Processing scene: {scene}-{seq}")

    # Load sequence splits
    seqs_qry = txt2seqs(source_dir.parent / 'TestSplit.txt')
    seqs_map = txt2seqs(source_dir.parent / 'TrainSplit.txt')

    # Create mapping between image indices and their query/map status
    idx2fname_path = scene_dir / 'mapper.txt'
    with open(idx2fname_path) as f:
        idx2fname_mapper = f.readlines()
        
    marker_query_map = acquire_marker_query_map(
        idx2fname_mapper,
        seqs_map,
        seqs_qry
    )

    export2ply(
        h5path=str(h5_path),
        scene=f"{scene}_{seq}",
        framerate=45,
        pr_modelpath=str(pr_model_path),
        gt_modelpath=str(gt_model_path),
        sharefocal=True,
        mute=args.mute,
        marker_query_map=marker_query_map
    )


def worker(rank: int, world_size: int, scene_data: List[Tuple[str, str]], args: argparse.Namespace) -> None:
    """Worker function for parallel scene processing.

    Args:
        rank: Process rank
        world_size: Total number of processes
        scene_data: List of (scene_name, sequence_name) tuples
        args: Command line arguments
    """
    process_scenes = scene_data[rank::world_size]

    if not args.mute:
        print(f"Process {rank} handling {len(process_scenes)} scenes")

    for scene in process_scenes:
        process_scene(scene, args, rank)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='7Scenes Visualization with Multiprocessing')
    
    parser.add_argument(
        '--data-root',
        type=Path,
        default=Path("/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/7scenes"),
        help='Path to 7scenes dataset'
    )
    parser.add_argument(
        '--output-location',
        type=Path,
        default=Path("/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/7scenes"),
        help='Path for output files'
    )
    parser.add_argument(
        '--depth-model',
        type=str,
        choices=['ZoeDepth', 'UniDepth', 'DUSt3R'],
        default='DUSt3R',
        help='Depth estimation model'
    )
    parser.add_argument(
        '--corres-model',
        type=str,
        choices=['RoMa', 'MASt3R'],
        default='RoMa',
        help='Correspondence model'
    )
    parser.add_argument(
        '--calibrated',
        action="store_true",
        help="Enable calibrated camera mode"
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

    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # Validate paths
    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    
    if not args.output_location.exists():
        args.output_location.mkdir(parents=True)

    # Get scene sequences
    scene_data = read_scene_query_seqs(args.data_root)
    scene_data = natsort.natsorted(scene_data)

    # Initialize multiprocessing
    world_size = min(args.num_processes, len(scene_data))
    if not args.mute:
        print(f"Processing {len(scene_data)} scenes using {world_size} processes")

    # Start processes
    mp.spawn(
        worker,
        args=(world_size, scene_data, args),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Required for Windows support
    main()
