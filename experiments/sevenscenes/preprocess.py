import os
import argparse
import sys
import random
import time

# Get project root path and add to system path
prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)

from experiments.sevenscenes.utils import read_scene_query_seqs

if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')
    parser.add_argument(
        '--data-root',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/7scenes",
        help='Root directory of 7scenes dataset'
    )
    parser.add_argument(
        '--output-location',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/7scenes",
        help='Output directory for preprocessed data'
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

    args = parser.parse_args()

    # Read scene sequences from dataset
    seqs_qrys = read_scene_query_seqs(args.data_root)

    # Create output directory path based on model choices
    preprocess_location = os.path.join(
        args.output_location, f"{args.depth_model}_{args.corres_model}"
    )

    # Shuffle and process each scene sequence
    random.shuffle(seqs_qrys)
    for scene, seq in seqs_qrys:
        time.sleep(random.uniform(0, 2))  # Random delay to avoid race conditions across GPUs

        dst_perscene = os.path.join(preprocess_location, f"{scene}_{seq}")
        src_perscene = os.path.join(args.data_root, scene, seq)

        if os.path.exists(dst_perscene):
            continue

        # Create output directory if not exists
        os.makedirs(dst_perscene, exist_ok=True)

        # Run depth estimation
        from MargBA.monodepth_estimator import inference_unary
        inference_unary(
            data_root=src_perscene,
            dataset='7scenes',
            output_location=dst_perscene,
            depth_model=args.depth_model,
        )

        # Run correspondence estimation
        from MargBA.corres_estimator import inference_pairwise
        inference_pairwise(
            data_root=src_perscene,
            dataset="7scenes",
            output_location=dst_perscene,
            corres_method_name="RoMa"
        )