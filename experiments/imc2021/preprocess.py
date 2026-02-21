import os, argparse, sys, glob, random, time
prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')
    parser.add_argument(
        '--data-root', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/imc2021"
    )
    parser.add_argument(
        '--output-location', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/imc2021"
    )
    parser.add_argument(
        '--depth-model', type=str, choices=['ZoeDepth', 'UniDepth', 'DUSt3R'], default='DUSt3R'
    )
    parser.add_argument(
        '--corres-model', type=str, choices=['RoMa', 'MASt3R'], default='RoMa'
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
    preprocess_location = os.path.join(
        args.output_location, f"{args.depth_model}_{args.corres_model}"
    )

    all_scenes_bags = list()
    for scene in scenes:
        bags = glob.glob(os.path.join(args.data_root, scene, "*"))
        for bag in bags:
            all_scenes_bags.append(bag)
    random.shuffle(all_scenes_bags)

    for src_perscene in all_scenes_bags:
        time.sleep(random.uniform(0, 2))  # Random delay to avoid race conditions across GPUs

        scene, bag = os.path.basename(os.path.dirname(src_perscene)), os.path.basename(src_perscene)
        dst_perscene = os.path.join(preprocess_location, f"{scene}_{bag}")

        if os.path.exists(dst_perscene):
            continue
        os.makedirs(dst_perscene, exist_ok=True)
        print(f"export to {dst_perscene}")

        from MargBA.monodepth_estimator import inference_unary
        inference_unary(
            data_root=src_perscene,
            dataset='imc2021',
            output_location=dst_perscene,
            depth_model=args.depth_model,
        )

        from MargBA.corres_estimator import inference_pairwise
        inference_pairwise(
            data_root=src_perscene,
            dataset="imc2021",
            output_location=dst_perscene,
            corres_method_name="RoMa",
            min_confidence=0.2, min_visibility=0.1
        )