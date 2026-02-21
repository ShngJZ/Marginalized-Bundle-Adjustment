import os, argparse, sys, time, random
prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')
    parser.add_argument(
        '--data-root', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/eth3d"
    )
    parser.add_argument(
        '--output-location', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/eth3d"
    )
    parser.add_argument(
        '--depth-model', type=str, choices=['ZoeDepth', 'UniDepth', 'DUSt3R'], default='DUSt3R'
    )
    parser.add_argument(
        '--corres-model', type=str, choices=['RoMa', 'MASt3R'], default='RoMa'
    )

    args = parser.parse_args()
    scenes = [
        "courtyard",
        "delivery_area",
        "electro",
        "facade",
        "kicker",
        "meadow",
        "office",
        "pipes",
        "playground",
        "relief",
        "relief_2",
        "terrace",
        "terrains"
    ]
    preprocess_location = os.path.join(
        args.output_location, f"{args.depth_model}_{args.corres_model}"
    )

    random.shuffle(scenes)
    for scene in scenes:
        time.sleep(random.uniform(0, 2))  # Random delay to avoid race conditions across GPUs

        dst_perscene = os.path.join(preprocess_location, scene)
        src_perscene = os.path.join(args.data_root, scene)

        if os.path.exists(dst_perscene):
            continue

        os.makedirs(dst_perscene, exist_ok=True)
        from MargBA.monodepth_estimator import inference_unary
        inference_unary(
            data_root=src_perscene,
            dataset='eth3d',
            output_location=dst_perscene,
            depth_model=args.depth_model,
        )

        from MargBA.corres_estimator import inference_pairwise
        inference_pairwise(
            data_root=src_perscene,
            dataset="eth3d",
            output_location=dst_perscene,
            corres_method_name="RoMa",
            min_confidence=0.2, min_visibility=0.1
        )