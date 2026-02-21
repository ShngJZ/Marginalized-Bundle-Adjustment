import os, argparse, sys, glob, random
import time

prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')
    parser.add_argument(
        '--data-root', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/fastmap_sfm"
    )
    parser.add_argument(
        '--output-location', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm"
    )
    parser.add_argument(
        '--depth-model', type=str, choices=['ZoeDepth', 'UniDepth', 'DUSt3R'], default='DUSt3R'
    )
    parser.add_argument(
        '--corres-model', type=str, choices=['RoMa', 'MASt3R'], default='RoMa'
    )

    args = parser.parse_args()
    scenes = [d for d in glob.glob(os.path.join(args.data_root, "databases/*.db"))]
    random.shuffle(scenes)
    preprocess_location = os.path.join(
        args.output_location, f"{args.depth_model}_{args.corres_model}"
    )

    for scene_path in scenes:
        time.sleep(random.uniform(0, 2))  # Random delay to avoid race conditions across GPUs

        # Extract just the scene name (without extension) from the path
        scene_name = os.path.splitext(os.path.basename(scene_path))[0]
        dst_perscene = os.path.join(preprocess_location, scene_name)

        if os.path.exists(dst_perscene):
            continue

        os.makedirs(dst_perscene, exist_ok=True)
        print(f"Processing scene {scene_name}")
        print(f"Export to {dst_perscene}")

        from MargBA.monodepth_estimator import inference_unary
        inference_unary(
            data_root=scene_path,
            dataset='fastmap_sfm',
            output_location=dst_perscene,
            depth_model=args.depth_model,
        )

        from MargBA.corres_estimator import inference_pairwise
        inference_pairwise(
            data_root=scene_path,
            dataset="fastmap_sfm",
            output_location=dst_perscene,
            corres_method_name=args.corres_model,
            min_confidence=0.2, min_visibility=0.1
        )