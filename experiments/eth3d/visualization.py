import os, argparse, sys
prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)
from MargBA.geometry.export2ply import export2ply

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
    for scene in scenes:
        dst_perscene = os.path.join(preprocess_location, scene)
        sfm_perscene = os.path.join(f"{preprocess_location}_sfm", scene)

        h5path = os.path.join(dst_perscene, f"{scene}.hdf5")
        pr_modelpath = os.path.join(sfm_perscene, "pr_pose_model_fine.ckpt")
        gt_modelpath = os.path.join(sfm_perscene, "gt_pose_model.ckpt")

        export2ply(
            h5path=h5path,
            scene=scene,
            framerate=3,
            pr_modelpath=pr_modelpath,
            gt_modelpath=gt_modelpath,
            sharefocal=True,
        )