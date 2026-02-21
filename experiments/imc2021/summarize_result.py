import os
import argparse
import sys
import glob
from typing import List, Dict

import natsort
import torch
import tqdm
from tabulate import tabulate
from collections import OrderedDict

prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)
from MargBA.poses_models import GlobalOptimizationPoseParameters
from MargBA.datasets import HDF5Reader
from experiments.imc2021.evaluation import rotation_angle, translation_angle, pose_auc

# Configuration
IMC_SCENES = [
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

def format_output(aggregated_results: List[Dict]) -> List[Dict]:
    """Format results for tabulate display with consistent keys."""
    keys = ["scene", "auc-3", "auc-5", "auc-10"]
    return [OrderedDict((k, result[k]) for k in keys) for result in aggregated_results]

def evaluate_poses(predicted_pose_w2c: torch.Tensor, gt_pose_w2c: torch.Tensor) -> torch.Tensor:
    """Evaluate pose errors between predicted and ground truth poses."""
    poses_w2c_gt_i2j, poses_w2c_pred_i2j = [], []
    
    for i in range(len(gt_pose_w2c)):
        for j in range(len(gt_pose_w2c)):
            if i != j:
                gt_relative = torch.inverse(gt_pose_w2c[j] @ torch.inverse(gt_pose_w2c[i]))
                pred_relative = torch.inverse(predicted_pose_w2c[j] @ torch.inverse(predicted_pose_w2c[i]))
                poses_w2c_gt_i2j.append(gt_relative)
                poses_w2c_pred_i2j.append(pred_relative)

    poses_w2c_gt_i2j = torch.stack(poses_w2c_gt_i2j)
    poses_w2c_pred_i2j = torch.stack(poses_w2c_pred_i2j)
    
    err_rot = rotation_angle(
        rot_gt=poses_w2c_gt_i2j[:, 0:3, 0:3],
        rot_pred=poses_w2c_pred_i2j[:, 0:3, 0:3]
    )
    err_tls = translation_angle(
        tvec_gt=poses_w2c_gt_i2j[:, 0:3, 3],
        tvec_pred=poses_w2c_pred_i2j[:, 0:3, 3]
    )

    return torch.max(torch.stack([err_rot, err_tls], dim=1), dim=1)[0]

def process_scene(scene_path: str, preprocess_location: str) -> torch.Tensor:
    """Process a single scene and return pose errors."""
    scene = os.path.basename(os.path.dirname(scene_path))
    bag = os.path.basename(scene_path)
    
    sfm_perscene = os.path.join(f"{preprocess_location}_sfm", f"{scene}_{bag}")
    ckpt_fine = os.path.join(sfm_perscene, "pr_pose_model_fine.ckpt")
    ckpt_gt = os.path.join(sfm_perscene, "gt_pose_model.ckpt")
    
    if not all(os.path.exists(p) for p in [ckpt_fine, ckpt_gt]):
        print(f"Warning: Missing checkpoint files for {scene}_{bag}")
        return None

    dst_perscene = os.path.join(preprocess_location, f"{scene}_{bag}")
    hdf5_path = os.path.join(dst_perscene, f"{bag}.hdf5")
    hdf5reader = HDF5Reader(hdf5_path)
    num_frames = len(hdf5reader.get_all_images())

    predicted_pose_model = GlobalOptimizationPoseParameters(
        nfrm=num_frames,
        optimizefocal=False,
        sharefocal=False
    )
    predicted_pose_model.load_state_dict(torch.load(ckpt_fine), strict=True)

    gt_pose_model = GlobalOptimizationPoseParameters(
        nfrm=num_frames,
        optimizefocal=False,
        sharefocal=False
    )
    gt_pose_model.load_state_dict(torch.load(ckpt_gt), strict=True)

    return evaluate_poses(
        predicted_pose_model.get_pose_w2c(),
        gt_pose_model.get_pose_w2c()
    )

def main():
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')
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
    args = parser.parse_args()

    preprocess_location = os.path.join(
        args.output_location,
        f"{args.depth_model}_{args.corres_model}"
    )

    # Collect all scene paths
    all_scenes_bags = []
    for scene in IMC_SCENES:
        scene_bags = glob.glob(os.path.join(args.data_root, scene, "*"))
        all_scenes_bags.extend(scene_bags)
    all_scenes_bags = natsort.natsorted(all_scenes_bags)

    # Process all scenes
    err_pose_all = []
    for scene_path in tqdm.tqdm(all_scenes_bags):
        err_pose = process_scene(scene_path, preprocess_location)
        if err_pose is not None:
            err_pose_all.append(err_pose)

    # Calculate and display results
    err_pose_all = torch.cat(err_pose_all)
    aucs = pose_auc(err_pose_all.cpu().detach().numpy(), thresholds=[3, 5, 10])
    
    results = {
        'auc-3': f"{aucs[0]*100:.2f}%",
        'auc-5': f"{aucs[1]*100:.2f}%",
        'auc-10': f"{aucs[2]*100:.2f}%"
    }
    
    print(tabulate([results], headers="keys"))


if __name__ == "__main__":
    main()
