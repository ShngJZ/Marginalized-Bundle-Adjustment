import os, sys, argparse
import numpy as np
import torch
from tabulate import tabulate
from collections import OrderedDict

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, proj_root)
from experiments.sevenscenes.utils import read_split_txt
from experiments.sevenscenes.evaluation import sevenscenes_evaluate
from MargBA.poses_models.global_registration_pose import r6d2mat, pad_poses


def get_pose_w2c(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    t, r6 = ckpt['t'], ckpt['r6']
    R = r6d2mat(r6)
    pose = pad_poses(torch.concat([R, t], dim=-1))
    return pose

def format_output(aggregated_results):
    keys = ["scene", 'median_pos_error', 'median_angular_error', 'acc_01m_1deg', 'acc_025m_2deg', 'acc_05m_5deg', 'acc_5m_10deg']
    aggregated_results_sorted = list()
    for x in aggregated_results:
        x_ = OrderedDict()
        for k in keys:
            x_[k] = x[k]
        aggregated_results_sorted.append(x_)
    return aggregated_results_sorted

if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')

    # Define command line arguments
    parser.add_argument(
        '--data-root',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/7scenes",
        help='Root directory containing 7scenes dataset'
    )
    parser.add_argument(
        '--output-location',
        type=str,
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/7scenes",
        help='Output directory for processed data'
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

    # Parse arguments
    args = parser.parse_args()

    scenes = [
        'chess',
        'fire',
        'heads',
        'office',
        'pumpkin',
        'redkitchen',
        'stairs'
    ]

    preprocess_location = os.path.join(
        args.output_location, f"{args.depth_model}_{args.corres_model}"
    )

    aggregated_result = list()
    for scene in scenes:
        seqs = read_split_txt(os.path.join(args.data_root, scene, 'TestSplit.txt'))

        scene_done = True
        for seq in seqs:
            pr_pose_model_path = os.path.join(f"{preprocess_location}_sfm", f"{scene}_{seq}", "pr_pose_model_fine.ckpt")
            if not os.path.exists(pr_pose_model_path):
                scene_done = False

        if not scene_done:
            continue

        # start to evaluate on the new stuff
        poses_w2c_gts, poses_w2c_prs, marker_query_maps = list(), list(), list()
        for seq in seqs:
            pr_pose_model_path = os.path.join(f"{preprocess_location}_sfm", f"{scene}_{seq}", "pr_pose_model_fine.ckpt")
            gt_pose_model_path = os.path.join(f"{preprocess_location}_sfm", f"{scene}_{seq}", "gt_pose_model.ckpt")

            pr_pose = get_pose_w2c(pr_pose_model_path)
            gt_pose = get_pose_w2c(gt_pose_model_path)

            ckpt = torch.load(pr_pose_model_path, map_location='cpu')
            qry = ckpt['gradient_mask'] == 1

            pr_pose, gt_pose = pr_pose[qry], gt_pose[qry]

            poses_w2c_gts.append(pr_pose)
            poses_w2c_prs.append(gt_pose)

        sfm_perscene = os.path.join(f"{preprocess_location}_sfm")
        params_perscene_perstage ={
                "sfm_location": sfm_perscene,
            }
        poses_w2c_gts = torch.cat(poses_w2c_gts)
        poses_w2c_prs = torch.cat(poses_w2c_prs)
        result = sevenscenes_evaluate(
            poses_w2c_prs, poses_w2c_gts, ['qry'] * len(poses_w2c_gts), len(poses_w2c_gts), params_perscene_perstage
        )
        result['scene'] = scene
        aggregated_result.append(result)

    average = {'scene': 'average'}
    for key in ['median_pos_error', 'median_angular_error', 'acc_01m_1deg', 'acc_025m_2deg', 'acc_05m_5deg', 'acc_5m_10deg']:
        average[key] = np.mean(np.array([x[key] for x in aggregated_result]))
    aggregated_result.append(average)
    print(tabulate(format_output(aggregated_result), headers="keys"))