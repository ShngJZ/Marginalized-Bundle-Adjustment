# Import required libraries
import os
import argparse
import sys
import glob
import random
import time
import natsort

# Get project root path and add to system path
prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')

    # Define command line arguments
    parser.add_argument(
        '--data-root',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/wayspots",
        help="Root directory of input data"
    )
    parser.add_argument(
        '--output-location',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/wayspots",
        help="Output directory for processed data"
    )
    parser.add_argument(
        '--depth-model',
        type=str,
        choices=['ZoeDepth', 'UniDepth', 'DUSt3R'],
        default='DUSt3R',
        help="Depth estimation model to use"
    )
    parser.add_argument(
        '--corres-model',
        type=str,
        choices=['RoMa', 'MASt3R'],
        default='RoMa',
        help="Correspondence estimation model to use"
    )

    # Parse arguments
    args = parser.parse_args()

    # Get all scene directories and sort them naturally
    scenes = glob.glob(os.path.join(args.data_root, "*"))
    scenes = [x for x in scenes if os.path.isdir(x)]  # Filter only directories
    scenes = [os.path.basename(x) for x in scenes]    # Get base names
    scenes = natsort.natsorted(scenes)  # Natural sort
    # Ensure there are scenes to process
    assert len(scenes) > 0, "not find any scene"

    # Create output directory path
    preprocess_location = os.path.join(
        args.output_location,
        f"{args.depth_model}_{args.corres_model}"
    )

    # Shuffle and process each scene
    random.shuffle(scenes)
    for scene in scenes:
        time.sleep(random.uniform(0, 2))  # Random delay to avoid race conditions across GPUs

        dst_perscene = os.path.join(preprocess_location, scene)  # Output dir for scene
        src_perscene = os.path.join(args.data_root, scene)       # Input dir for scene

        if os.path.exists(dst_perscene):
            continue

        # Create output directory if not exists
        os.makedirs(dst_perscene, exist_ok=True)

        # Run depth estimation
        from MargBA.monodepth_estimator import inference_unary
        inference_unary(
            data_root=src_perscene,
            dataset='wayspots',
            output_location=dst_perscene,
            depth_model=args.depth_model,
        )

        # Run correspondence estimation
        from MargBA.corres_estimator import inference_pairwise
        inference_pairwise(
            data_root=src_perscene,
            dataset="wayspots",
            output_location=dst_perscene,
            corres_method_name="RoMa",
            min_confidence=0.2, min_visibility=0.1
        )