#!/usr/bin/env python3
"""
COLMAP-based Structure from Motion (SfM) processing script for ScanNet dataset.

This script processes ScanNet scenes using COLMAP, supporting both calibrated and
uncalibrated camera scenarios. It converts dataset images to the required format
and runs SfM reconstruction.
"""

import os
import sys
import argparse
import random
import time
import tqdm
from typing import List

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from MargBA.COLMAP import (
    compute_sfm_inloc_wt_intrinsic,
    compute_sfm_inloc_wo_intrinsic
)
from MargBA.datasets import ScanNet
from MargBA.datasets.vls_utils import image2vls


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command line argument parser.

    Returns:
        ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Process ScanNet scenes using COLMAP for Structure from Motion reconstruction.'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet/scans_test",
        help='Root directory containing ScanNet test scenes'
    )
    parser.add_argument(
        '--output-location',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet",
        help='Directory where COLMAP results will be saved'
    )
    return parser


def read_scene_list(file_path: str) -> List[str]:
    """
    Read scene names from a text file.

    Args:
        file_path: Path to the text file containing scene names

    Returns:
        List of scene names
    """
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def process_scene(
    scene: str,
    dataset: ScanNet,
    output_folder: str,
    calibrated: bool
) -> None:
    """
    Process a single ScanNet scene using COLMAP.

    Args:
        scene: Scene name
        dataset: ScanNet dataset instance
        output_folder: Base output directory for COLMAP results
        calibrated: Whether to use calibrated or uncalibrated reconstruction
    """
    # Set up output directories
    scene_folder = os.path.join(output_folder, scene)
    rgb_folder = os.path.join(scene_folder, 'rgb')

    # Skip if already processed
    if os.path.exists(rgb_folder):
        print(f"Scene {scene} already processed, skipping...")
        return

    # Create output directory
    os.makedirs(rgb_folder, exist_ok=True, mode=0o777)

    # Get image dimensions and intrinsics from first sample
    initial_sample = dataset[0]
    _, height, width = initial_sample['image'].shape
    intrinsic = initial_sample['intr']

    # Process all images in the scene
    rgb_paths = []
    for idx in tqdm.tqdm(range(len(dataset)), desc=f"Processing {scene}"):
        sample = dataset[idx]

        # Generate output image path
        rgb_path = os.path.join(
            rgb_folder,
            f"{str(sample['image_idx']).zfill(6)}.png"
        )

        # Convert and save image
        image_vls = image2vls(
            sample['image'].view([1, 3, height, width]),
            viewind=0
        )
        image_vls.save(rgb_path)
        rgb_paths.append(rgb_path)

    # Run COLMAP reconstruction
    if calibrated:
        compute_sfm_inloc_wt_intrinsic(
            save_dir=rgb_folder,
            rgb_paths=rgb_paths,
            intrinsics=intrinsic,
            H=height,
            W=width,
            matcher_model_type="indoor"
        )
    else:
        compute_sfm_inloc_wo_intrinsic(
            save_dir=rgb_folder,
            rgb_paths=rgb_paths,
            matcher_model_type="indoor"
        )

    # Clean up PNG files
    for rgb_path in rgb_paths:
        if os.path.exists(rgb_path):
            os.remove(rgb_path)


def main():
    """
    Main function to process ScanNet scenes using COLMAP.
    """
    # Parse command line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Read and shuffle scene list
    scenes = read_scene_list(os.path.join(os.path.dirname(__file__), 'scannet.txt'))
    random.shuffle(scenes)

    # Process scenes for both calibrated and uncalibrated cases
    for calibrated in [True, False]:
        # Set up output directory
        colmap_folder = os.path.join(
            args.output_location,
            'colmap_calibrated' if calibrated else 'colmap_uncalibrated'
        )

        # Process each scene
        for scene in scenes:
            time.sleep(random.uniform(0, 2))  # Random delay to avoid race conditions across GPUs

            scene_folder = os.path.join(colmap_folder, scene)
            if os.path.exists(scene_folder):
                continue
            os.makedirs(scene_folder, exist_ok=True)

            # Initialize dataset
            dataset = ScanNet(
                data_root=os.path.join(args.data_root, scene),
                mode="monodepth",
                rank=0,
                world_size=1
            )

            # Process the scene
            process_scene(
                scene=scene,
                dataset=dataset,
                output_folder=colmap_folder,
                calibrated=calibrated
            )



if __name__ == "__main__":
    main()
