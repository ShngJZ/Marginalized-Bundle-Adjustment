import os
import sys
import glob
import argparse
import io
from typing import List, Tuple, Dict, Optional, Sequence
from dataclasses import dataclass

import h5py
import torch
import numpy as np
import natsort
import tabulate
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from MargBA.poses_models import GlobalOptimizationPoseParameters
from MargBA.COLMAP.colmap_caller import get_poses_and_idx

MIN_VALID_FRAMES = 10
ANGLE_THRESHOLDS = [3.0, 5.0, 10.0]

@dataclass
class EvaluationConfig:
    data_root: str
    output_location: str
    depth_model: str
    corres_model: str
    calibrated: bool

def find_nan_poses(h5py_path: str) -> np.ndarray:
    with h5py.File(h5py_path, 'r') as h5f:
        image_names = natsort.natsorted(list(h5f['color'].keys()))[::6]
        nan_flags = []
        
        for img_name in image_names:
            stem = img_name.split('.')[0]
            pose_bytes = np.array(h5f['pose'][f'{stem}.txt'])
            pose_text = io.BytesIO(pose_bytes).read().decode('UTF-8')
            pose_matrix = np.loadtxt(io.StringIO(pose_text), delimiter=' ')
            nan_flags.append(np.any(np.isnan(pose_matrix)))

    return np.array(nan_flags)

def angle_error_mat(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    R1T_dot_R2 = np.transpose(R1, [0, 2, 1]) @ R2
    trace = np.sum(np.diagonal(R1T_dot_R2, axis1=1, axis2=2))
    cos = (trace - 1) / 2
    return np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))

def angle_error_vec(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    norms = np.sqrt((v1 ** 2).sum(-1) * (v2 ** 2).sum(-1))
    dot_product = np.sum(v1 * v2, axis=1) / norms
    return np.rad2deg(np.arccos(np.clip(dot_product, -1.0, 1.0)))

def compute_pose_errors(T_gt: np.ndarray, T_est: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    R_gt, t_gt = T_gt[:, :3, :3], T_gt[:, :3, 3]
    R_est, t_est = T_est[:, :3, :3], T_est[:, :3, 3]
    
    error_t = angle_error_vec(t_est + 1e-6, t_gt + 1e-6)
    error_t = np.minimum(error_t, 180 - error_t)
    error_R = angle_error_mat(R_est, R_gt)
    error_max = np.maximum(error_t, error_R)
    
    return error_t, error_R, error_max

def compute_pair_accuracies(T_gt: np.ndarray, T_est: np.ndarray, pairs: Sequence[Sequence[int]]) -> List[float]:
    pairs = np.asarray(pairs)
    src_idx, dst_idx = pairs[:, 0], pairs[:, 1]
    
    rel_pose_gt = T_gt[dst_idx] @ np.linalg.inv(T_gt[src_idx])
    rel_pose_est = T_est[dst_idx] @ np.linalg.inv(T_est[src_idx])

    _, _, errors = compute_pose_errors(rel_pose_gt, rel_pose_est)
    return [np.mean(errors < th) for th in ANGLE_THRESHOLDS]

def get_colmap_poses(output_dir: str) -> Tuple[np.ndarray, np.ndarray, int]:
    image_paths = natsort.natsorted(glob.glob(os.path.join(output_dir, 'rgb/images', '*.png')))
    image_names = [os.path.basename(x) for x in image_paths]

    poses_dict, valid_idx, invalid_idx = get_poses_and_idx(
        os.path.join(output_dir, 'rgb', 'sfm_superpoint+superglue'),
        image_names
    )

    assert len(image_names) == len(valid_idx) + len(invalid_idx), "Frame count mismatch"

    valid_poses = [poses_dict[name] for name in image_names if name in poses_dict]
    return np.stack(valid_poses), np.array(valid_idx), len(image_names)

def check_results_exist(sfm_dir: str, colmap_dir: str) -> bool:
    return all(os.path.exists(p) for p in [
        os.path.join(sfm_dir, 'pr_pose_model_fine.ckpt'),
        os.path.join(colmap_dir, 'rgb', 'sfm_superpoint+superglue', 'images.bin')
    ])

def read_pairs(data_root: str, scene: str) -> List[List[int]]:
    pairs_path = os.path.join(data_root, "scans_test_pairs", f"{scene}.txt")
    with open(pairs_path) as f:
        return [list(map(int, line.strip().split())) for line in f]

def filter_valid_pairs(pairs: List[List[int]], valid_idx: List[int], nan_flags: np.ndarray) -> List[List[int]]:
    idx_map = {idx: pos for pos, idx in enumerate(valid_idx)}
    return [
        [idx_map[src], idx_map[dst]]
        for src, dst in pairs
        if src in idx_map and dst in idx_map and not nan_flags[src] and not nan_flags[dst]
    ]

def evaluate_scene(scene: str, config: EvaluationConfig, preprocess_dir: str) -> Optional[Dict[str, float]]:
    calibration_suffix = "calibrated" if config.calibrated else "uncalibrated"
    sfm_dir = os.path.join(f"{preprocess_dir}_sfm_{calibration_suffix}", scene)
    colmap_dir = os.path.join(config.output_location, f"colmap_{calibration_suffix}", scene)

    if not check_results_exist(sfm_dir, colmap_dir):
        return None

    pairs = read_pairs(config.data_root, scene)
    colmap_poses, valid_idx, num_frames = get_colmap_poses(colmap_dir)
    
    if len(valid_idx) < MIN_VALID_FRAMES:
        return None

    pose_params = {"nfrm": num_frames, "optimizefocal": False, "sharefocal": True}
    gt_model = GlobalOptimizationPoseParameters(**pose_params)
    gt_model.load_state_dict(torch.load(os.path.join(sfm_dir, 'gt_pose_model.ckpt')))

    pr_model = GlobalOptimizationPoseParameters(**pose_params)
    pr_model.load_state_dict(torch.load(os.path.join(sfm_dir, 'pr_pose_model_fine.ckpt')))

    poses_gt = gt_model.get_pose_w2c().detach().numpy()[valid_idx]
    poses_pr = pr_model.get_pose_w2c().detach().numpy()[valid_idx]

    nan_flags = find_nan_poses(os.path.join(config.data_root, 'scans_test', scene, f"{scene}.hdf5"))
    valid_pairs = filter_valid_pairs(pairs, valid_idx, nan_flags)

    return {
        "colmap": compute_pair_accuracies(poses_gt, colmap_poses, valid_pairs),
        "mfs": compute_pair_accuracies(poses_gt, poses_pr, valid_pairs),
        "scene": scene,
        "nfrm": num_frames
    }

def print_results(metrics: Dict[str, List]) -> None:
    method_results = []
    for method in ["colmap", "mfs"]:
        mean_acc = np.mean(np.stack(metrics[method]), axis=0)
        method_results.append([method] + [f"{x:.3f}" for x in mean_acc])

    print(f"\nEvaluated {len(metrics['scene'])} Sequences")
    print(tabulate.tabulate(method_results, headers=["Method", "ACC-03", "ACC-05", "ACC-10"]))

    scene_results = [
        [scene, metrics["nfrm"][i]] + [metrics[m][i][1] for m in ["colmap", "mfs"]]
        for i, scene in enumerate(metrics["scene"])
    ]
    print("\nPer-Scene Results:")
    print(tabulate.tabulate(scene_results, headers=["Scene", "Frames", "COLMAP-ACC-05", "MFS-ACC-05"]))

def main() -> None:
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment Evaluation')
    parser.add_argument('--data-root', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/scannet")
    parser.add_argument('--output-location', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/scannet")
    parser.add_argument('--depth-model', type=str, choices=['ZoeDepth', 'UniDepth', 'DUSt3R'], default='DUSt3R')
    parser.add_argument('--corres-model', type=str, choices=['RoMa', 'MASt3R'], default='RoMa')
    parser.add_argument('--calibrated', action="store_true")

    config = EvaluationConfig(**vars(parser.parse_args()))
    preprocess_dir = os.path.join(config.output_location, f"{config.depth_model}_{config.corres_model}")

    with open(os.path.join(os.path.dirname(__file__), 'scannet.txt')) as f:
        scenes = [line.strip() for line in f if line.strip()]

    metrics = {key: [] for key in ["colmap", "mfs", "scene", "nfrm"]}
    
    for scene in tqdm(scenes, desc="Processing scenes"):
        if result := evaluate_scene(scene, config, preprocess_dir):
            for key in metrics:
                metrics[key].append(result[key])

    print_results(metrics)

if __name__ == "__main__":
    main()
