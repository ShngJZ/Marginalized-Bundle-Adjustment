import argparse, sys
import json
import glob
import os
from pathlib import Path

import numpy as np
import torch
from scipy import spatial
import einops
import natsort

sys.path.append(str(Path(__file__).parent / '../../third_party/fastmap'))
from fastmap.io import read_model as read_colmap_output


def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)


def write_json(fname, payload, **kwargs):
    with open(fname, "w") as f:
        json.dump(payload, f, **kwargs)


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.round(cos, decimals=4)

    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def pose_auc(errors, thresholds, ret_dict=False):
    n = len(errors)
    assert n > 0
    assert (np.array(thresholds) > 0).all()

    errors = np.sort(errors)
    recall = (np.arange(n) + 1) / n

    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]

    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        # insert (t, recall[last_index - 1]) as the closing tick of the band [0, t]
        # last_index-1 cuz often all pose errors are < t; last_index becomes N; can only use N - 1.

        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
        # aucs.append(recall[last_index - 1])  RRA

    if ret_dict:
        return {f"auc@{t}": auc for t, auc in zip(thresholds, aucs)}
    else:
        return aucs


def compute_ate(gt, pd):
    '''compute average translation error. used by FlowMap.
    Args:
        gt (torch.Tensor): [N, 3] ground truth translation vectors
        pd (torch.Tensor): [N, 3] predicted translation vectors
    '''
    aligned_gt, aligned_pd, _ = spatial.procrustes(
        gt.detach().cpu().numpy(),
        pd.cpu().numpy(),
    )
    aligned_gt = torch.tensor(aligned_gt, dtype=torch.float32, device=gt.device)
    aligned_pd = torch.tensor(
        aligned_pd, dtype=torch.float32, device=pd.device
    )
    ate = ((aligned_gt - aligned_pd) ** 2).mean() ** 0.5
    return ate.item()


def Rt_to_T(R, t) -> torch.Tensor:
    T = torch.cat([R, t[:, :, None]], dim=-1)  # (B, 3, 4)
    last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=T.device, dtype=T.dtype).view(
        1, 1, 4
    )  # (1, 1, 4)
    T = torch.cat([T, last_row.expand(T.shape[0], 1, 4)], dim=1)  # (B, 4, 4)
    return T


def batch_inv_pose(poses):
    # poses: [B, 4, 4]
    Rs = poses[:, 0:3, 0:3]
    ts = poses[:, 0:3, 3]
    n = poses.shape[0]

    T = torch.eye(4, device=poses.device, dtype=poses.dtype).unsqueeze(0).repeat(n, 1, 1)
    T[:, 0:3, 0:3] = einops.rearrange(Rs,  "n a b -> n b a")
    T[:, 0:3, 3] = einops.einsum(Rs, -ts, "n a b, n a -> n b")  # note the R.T
    return T


def _cos_to_angle(cos_theta):
    cos_theta = torch.clip(cos_theta, -1., 1.)
    return torch.rad2deg(
        torch.acos(cos_theta).abs()
    )


def batch_rot_angle_error(R1s, R2s):
    R1s_T = einops.rearrange(R1s,  "n a b -> n b a")
    trace = einops.einsum(torch.bmm(R1s_T, R2s), '... i i -> ...')
    assert torch.allclose(trace, torch.vmap(torch.trace)(torch.bmm(R1s_T, R2s)))
    cos_theta = (trace - 1) / 2
    # WARN: this changes numbers quit a bit when cos_theta is close to 0.
    cos_theta = torch.round(cos_theta, decimals=4)
    return _cos_to_angle(cos_theta)


def batch_vec_angle_error(v1s, v2s):
    v1s = torch.nn.functional.normalize(v1s, dim=-1)
    v2s = torch.nn.functional.normalize(v2s, dim=-1)
    cos_theta = einops.einsum(v1s, v2s, "n d, n d -> n")
    return _cos_to_angle(cos_theta)


@torch.inference_mode()
def pose_pair_angle_error(gt_poses, pd_poses):
    assert gt_poses.shape == pd_poses.shape, f"{gt_poses.shape} vs {pd_poses.shape}"
    n = len(gt_poses)
    inds = torch.combinations(torch.arange(n), 2, with_replacement=False)
    inds = inds.to(gt_poses.device).T  # [2, n_pairs]
    gt_pose_pairs = gt_poses[inds, :, :]
    pd_pose_pairs = pd_poses[inds, :, :]
    del gt_poses, pd_poses

    gt_a2b = torch.bmm(batch_inv_pose(gt_pose_pairs[1]), gt_pose_pairs[0])
    pd_a2b = torch.bmm(batch_inv_pose(pd_pose_pairs[1]), pd_pose_pairs[0])

    r_err = batch_rot_angle_error(gt_a2b[:, :3, :3], pd_a2b[:, :3, :3])
    t_err = batch_vec_angle_error(gt_a2b[:, :3, -1], pd_a2b[:, :3, -1])

    return r_err, t_err


def compute_auc(errs):
    angle_thresholds = [1, 3, 5, 10]
    aucs = pose_auc(errs.cpu().numpy(), angle_thresholds, ret_dict=False)
    aucs = np.array(aucs) * 100
    aucs = aucs.tolist()
    return aucs


def mAA(errs, max_threshold=30):
    """pose diffusion's mAA
    """
    # WARN: rightmost bin edge is 31. Err above 31 degrees are ignored.
    # hence must normalize by total N; np.histogram(density=True) would be incorrect.
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(errs, bins=bins)
    # Normalize the histogram
    N = float(len(errs))
    normalized_histogram = histogram.astype(float) / N

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram)).item() * 100


@torch.inference_mode()
def pose_stats_suite(gt_poses, pd_poses):
    stats = {}

    if gt_poses.shape != pd_poses.shape:
        raise ValueError(f"{gt_poses.shape} vs {pd_poses.shape}")

    ate = compute_ate(gt_poses[:, 0:3, 3], pd_poses[:, 0:3, 3])
    stats['ate'] = ate
    del ate

    r_err, t_err = pose_pair_angle_error(gt_poses, pd_poses)
    p_err = torch.max(
        torch.stack([r_err, t_err], dim=0),
        dim=0
    ).values

    stats['auc_p'] = compute_auc(p_err)
    stats['auc_t'] = compute_auc(t_err)
    stats['auc_r'] = compute_auc(r_err)

    def _frac_below(seq, thresh):
        return ((seq < thresh).float().mean() * 100).item()

    stats['RRA'] = [_frac_below(r_err, t) for t in [1, 3, 5, 10, 15]]
    stats['RTA'] = [_frac_below(t_err, t) for t in [1, 3, 5, 10, 15]]
    stats['mAA@30'] = mAA(p_err.cpu().numpy())

    def _frac_above(seq, thresh):
        return ((seq > thresh).float().mean() * 100).item()

    # stats['GRA'] = [_frac_above(r_err, t) for t in [5, 10, 15, 35, 60]]
    # stats['GTA'] = [_frac_above(t_err, t) for t in [5, 10, 20, 30, 60, 90, 120, 150]]

    return stats


def load_colmap_db_cams(fname, ext, return_all=True, device="cuda"):
    colmap_output = read_colmap_output(fname, device=device, ext=ext)
    names = colmap_output.names
    Rs, ts = colmap_output.rotation, colmap_output.translation
    Rs, ts = Rs.to(device), ts.to(device)

    w2c_colmap = Rt_to_T(R=Rs, t=ts)  # [N, 4, 4]
    c2w_colmap = batch_inv_pose(w2c_colmap)

    # assert torch.allclose(c2w_colmap, w2c_colmap.inverse(), atol=1e-4, rtol=1e-2)

    c2w_colmap = c2w_colmap.to(torch.float64)
    
    # Keep original filenames from COLMAP (may include directory structure)
    # Check that no ground truth files share the same name
    assert len(set(names)) == len(names), "Duplicate ground truth filenames found"
    
    return names, c2w_colmap


def load_poses_from_checkpoint(checkpoint_path, name_mapping, sharefocal=True):
    """Load poses from checkpoint file and return camera names and c2w poses"""
    from MargBA.poses_models import GlobalOptimizationPoseParameters
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    
    # Get number of frames from checkpoint
    nfrm = checkpoint['t'].shape[0]
    
    # Initialize pose model
    pose_model = GlobalOptimizationPoseParameters(
        nfrm=nfrm,
        optimizefocal=False,
        sharefocal=sharefocal
    )
    pose_model.load_state_dict(checkpoint, strict=True)
    
    # Get w2c poses from model
    w2c_poses = pose_model.get_pose_w2c().cuda().detach()
    
    # Convert w2c to c2w
    c2w_poses = batch_inv_pose(w2c_poses)
    c2w_poses = c2w_poses.to(torch.float64)
    
    # Get image names from name mapping, sorted by index
    sorted_items = sorted(name_mapping.items(), key=lambda x: x[1]['index'])
    # Keep original filenames from name mapping (may include directory structure)
    image_names = [item[0] for item in sorted_items]
    
    return image_names, c2w_poses


def load_largest_cam_cluster(res_dir):
    """This function is kept for compatibility but now loads from checkpoint"""
    # Look for checkpoint file
    checkpoint_path = res_dir / "pr_pose_model_fine.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load name mapping to get sharefocal parameter
    scene_name = res_dir.name
    preprocess_dir = res_dir.parent.parent / res_dir.parent.name.replace("_sfm", "")
    name_mapping_file = preprocess_dir / scene_name / "name2index_mapper.json"
    
    if not name_mapping_file.exists():
        raise FileNotFoundError(f"Name mapping file not found: {name_mapping_file}")
    
    with open(name_mapping_file, 'r') as f:
        name_mapping = json.load(f)
    
    # Determine sharefocal based on scene name (same logic as in sfm.py)
    if scene_name.startswith("eth3d_dslr_"):
        sharefocal = False
    elif (scene_name.startswith("nosr_") or 
          scene_name.startswith("dploy_house4") or 
          scene_name.startswith("urbn_") or 
          scene_name.startswith("eft_")):
        sharefocal = False
    else:
        sharefocal = True
    
    # Load poses from checkpoint
    image_names, c2w_poses = load_poses_from_checkpoint(checkpoint_path, name_mapping, sharefocal)
    
    return image_names, c2w_poses


def read_json_gt_cams(fname):
    cams = load_json(fname)
    fnames = [e['fname'] for e in cams]
    c2w = [e['c2w'] for e in cams]
    c2w = torch.tensor(c2w, dtype=torch.float64, device="cuda")
    return fnames, c2w


def do_eval(pd: Path, gt: Path):
    assert pd.is_dir()
    # Load poses directly from checkpoint - no need for name conversion
    pd_fnames, pd_poses = load_largest_cam_cluster(Path(pd))

    assert gt.exists()
    if gt.is_dir():
        is_bin, is_txt = (gt / 'cameras.bin').is_file(), (gt / 'cameras.txt').is_file()
        assert is_bin or is_txt
        ext = '.bin' if is_bin else '.txt'
        gt_fnames, gt_poses = load_colmap_db_cams(gt, ext)
    elif gt.is_file():
        gt_fnames, gt_poses = read_json_gt_cams(gt)

    # no duplicate filenames
    assert len(set(gt_fnames)) == len(gt_fnames), "Duplicate ground truth filenames found"
    assert len(set(pd_fnames)) == len(pd_fnames), "Duplicate predicted filenames found"
    
    N = len(gt_fnames)

    # Try to match filenames - first exact match, then basename match
    missing_frames = []
    inds = []
    used_pd_indices = set()  # Track which prediction indices have been used
    
    # Create mapping for predicted filenames for faster lookup
    pd_fname_to_idx = {fname: idx for idx, fname in enumerate(pd_fnames)}
    pd_basename_to_idx = {os.path.basename(fname): idx for idx, fname in enumerate(pd_fnames)}
    
    for i in range(N):
        gt_fname = gt_fnames[i]
        matched_idx = None
        
        # Try exact match first
        if gt_fname in pd_fname_to_idx:
            matched_idx = pd_fname_to_idx[gt_fname]
        # Try basename match
        elif os.path.basename(gt_fname) in pd_basename_to_idx:
            matched_idx = pd_basename_to_idx[os.path.basename(gt_fname)]
        
        if matched_idx is not None:
            # Check if this prediction index has already been used
            if matched_idx in used_pd_indices:
                raise ValueError(
                    f"Duplicate mapping detected: prediction file '{pd_fnames[matched_idx]}' "
                    f"maps to multiple ground truth files. This indicates ambiguous filename matching."
                )
            used_pd_indices.add(matched_idx)
            inds.append(matched_idx)
        else:
            missing_frames.append(gt_fname)
    
    # Raise exception if any frames are missing
    if missing_frames:
        raise ValueError(
            f"Missing {len(missing_frames)} ground truth frames in predictions:\n"
            f"{missing_frames[:50]}{'...' if len(missing_frames) > 50 else ''}\n"
            f"Ground truth has {len(gt_fnames)} frames, predictions have {len(pd_fnames)} frames."
        )

    inds = torch.tensor(inds).cuda()
    resampled_pd_poses = pd_poses[inds]

    results = pose_stats_suite(gt_poses, resampled_pd_poses)

    out = {}
    out['res'] = results
    out['nums'] = (len(gt_poses), len(pd_poses), 0)  # No fill-in since we raise exception on mismatch

    return out


def main():
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

    # Get all scenes
    scenes = [d for d in glob.glob(os.path.join(args.data_root, "databases/*.db"))]
    scenes = natsort.natsorted(scenes)
    
    # Paths
    gt_root = os.path.join(args.data_root, "ground_truths")
    sfm_location = os.path.join(args.output_location, f"{args.depth_model}_{args.corres_model}_sfm")
    
    # Results storage
    all_results = {}
    successful_scenes = []
    skipped_scenes = []
    
    print(f"Evaluating {len(scenes)} scenes...")
    print(f"Ground truth root: {gt_root}")
    print(f"SfM results location: {sfm_location}")
    print("-" * 80)

    # scenes = ["/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/fastmap_sfm/databases/eth3d_dslr_pipes.db"]
    
    for scene_path in scenes:
        scene = os.path.splitext(os.path.basename(scene_path))[0]
        
        # Check if fine model exists
        sfm_perscene = os.path.join(sfm_location, scene)
        fine_model_path = os.path.join(sfm_perscene, "pr_pose_model_fine.ckpt")
        
        if not os.path.exists(fine_model_path):
            skipped_scenes.append(scene)
            continue

        # Paths for this scene
        pd_path = Path(sfm_perscene)

        # Try to find ground truth
        gt_json = Path(gt_root) / f"{scene}.json"
        gt_colmap = Path(gt_root) / scene
        
        if gt_json.exists():
            gt_path = gt_json
        elif gt_colmap.exists():
            gt_path = gt_colmap
        else:
            raise FileNotFoundError(
                f"No ground truth found for scene '{scene}'. "
                f"Checked paths:\n  - {gt_json}\n  - {gt_colmap}"
            )
        
        print(f"\nEvaluating {scene}...")
        print(f"  GT: {gt_path}")
        print(f"  PD: {pd_path}")
        
        result = do_eval(pd_path, gt_path)
        all_results[scene] = result
        successful_scenes.append(scene)
        
        # Print individual scene results
        print("  Results:")
        print(f"    ATE: {result['res']['ate']:.4f}")
        print(f"    AUC@5 (pose): {result['res']['auc_p'][2]:.2f}")
        print(f"    AUC@5 (rot): {result['res']['auc_r'][2]:.2f}")
        print(f"    AUC@5 (trans): {result['res']['auc_t'][2]:.2f}")
        print(f"    mAA@30: {result['res']['mAA@30']:.2f}")
        print(f"    GT/PD/Fill-in: {result['nums']}")
    
    # Summarize results
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total scenes: {len(scenes)}")
    print(f"Successfully evaluated: {len(successful_scenes)}")
    print(f"Skipped: {len(skipped_scenes)}")
    
    if successful_scenes:
        print("\nAggregated Results:")
        
        # Collect all metrics
        metrics = ['ate', 'mAA@30']
        auc_metrics = ['auc_p', 'auc_t', 'auc_r']
        rra_rta_metrics = ['RRA', 'RTA']
        
        # Simple metrics
        for metric in metrics:
            values = [all_results[scene]['res'][metric] for scene in successful_scenes]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # AUC metrics (list of values)
        for metric in auc_metrics:
            print(f"\n  {metric}:")
            for i, threshold in enumerate([1, 3, 5, 10]):
                values = [all_results[scene]['res'][metric][i] for scene in successful_scenes]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"    @{threshold}°: {mean_val:.2f} ± {std_val:.2f}")
        
        # RRA/RTA metrics
        for metric in rra_rta_metrics:
            print(f"\n  {metric}:")
            for i, threshold in enumerate([1, 3, 5, 10, 15]):
                values = [all_results[scene]['res'][metric][i] for scene in successful_scenes]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"    @{threshold}°: {mean_val:.2f} ± {std_val:.2f}")
        
        # Per-scene summary matching FASTMAP paper format
        print("\nPer-scene Summary (FASTMAP paper format):")
        print(f"{'Scene':<30} {'ATE↓':<12} {'RTA@3↑':<10} {'AUC-R&T@3↑':<12} {'RTA@1↑':<10} {'AUC-R&T@1↑':<12}")
        print("-" * 100)
        for scene in successful_scenes:
            res = all_results[scene]['res']
            # ATE in scientific notation
            ate_str = f"{res['ate']:.1e}"
            # RTA@3 and RTA@1
            rta3 = res['RTA'][1]  # Index 1 is @3°
            rta1 = res['RTA'][0]  # Index 0 is @1°
            # AUC-R&T is the average of rotation and translation AUC
            auc_rt3 = (res['auc_r'][1] + res['auc_t'][1]) / 2  # Index 1 is @3°
            auc_rt1 = (res['auc_r'][0] + res['auc_t'][0]) / 2  # Index 0 is @1°
            
            print(f"{scene:<30} {ate_str:<12} {rta3:<10.1f} {auc_rt3:<12.1f} {rta1:<10.1f} {auc_rt1:<12.1f}")
        
        # Overall summary in FASTMAP paper format
        print("\nOverall Summary (FASTMAP paper format):")
        print(f"{'Metric':<20} {'Mean ± Std':<20}")
        print("-" * 40)
        
        # ATE
        ate_values = [all_results[scene]['res']['ate'] for scene in successful_scenes]
        ate_mean = np.mean(ate_values)
        ate_std = np.std(ate_values)
        print(f"{'ATE↓':<20} {f'{ate_mean:.1e} ± {ate_std:.1e}':<20}")
        
        # RTA@3
        rta3_values = [all_results[scene]['res']['RTA'][1] for scene in successful_scenes]
        rta3_mean = np.mean(rta3_values)
        rta3_std = np.std(rta3_values)
        print(f"{'RTA@3↑':<20} {f'{rta3_mean:.1f} ± {rta3_std:.1f}':<20}")
        
        # AUC-R&T@3
        auc_rt3_values = [(all_results[scene]['res']['auc_r'][1] + all_results[scene]['res']['auc_t'][1]) / 2 for scene in successful_scenes]
        auc_rt3_mean = np.mean(auc_rt3_values)
        auc_rt3_std = np.std(auc_rt3_values)
        print(f"{'AUC-R&T@3↑':<20} {f'{auc_rt3_mean:.1f} ± {auc_rt3_std:.1f}':<20}")
        
        # RTA@1
        rta1_values = [all_results[scene]['res']['RTA'][0] for scene in successful_scenes]
        rta1_mean = np.mean(rta1_values)
        rta1_std = np.std(rta1_values)
        print(f"{'RTA@1↑':<20} {f'{rta1_mean:.1f} ± {rta1_std:.1f}':<20}")
        
        # AUC-R&T@1
        auc_rt1_values = [(all_results[scene]['res']['auc_r'][0] + all_results[scene]['res']['auc_t'][0]) / 2 for scene in successful_scenes]
        auc_rt1_mean = np.mean(auc_rt1_values)
        auc_rt1_std = np.std(auc_rt1_values)
        print(f"{'AUC-R&T@1↑':<20} {f'{auc_rt1_mean:.1f} ± {auc_rt1_std:.1f}':<20}")
    
    # Save all results
    output_file = os.path.join(args.output_location, f"evaluation_results_{args.depth_model}_{args.corres_model}.json")
    write_json(output_file, {
        'args': vars(args),
        'results': all_results,
        'successful_scenes': successful_scenes,
        'skipped_scenes': skipped_scenes
    }, indent=2)
    print(f"\nResults saved to: {output_file}")


def generate_dataset_summary_csv():
    """
    Generate a CSV summary of results grouped by dataset.
    Groups scenes by dataset prefixes and computes aggregate metrics.
    """
    import csv
    
    # Hardcoded path to the JSON file
    json_path = "/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm/evaluation_results_DUSt3R_RoMa.json"
    
    # Load results
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Define dataset groupings
    dataset_groups = {
        'mipnerf360': {'prefix': 'm360_', 'expected_count': 9},
        'tnt_advanced': {'prefix': 'tnt_advn_', 'expected_count': 6},
        'tnt_intermediate': {'prefix': 'tnt_intrmdt_', 'expected_count': 8},
        'tnt_training': {'prefix': 'tnt_trng_', 'expected_count': 7},
        'nerf_osr': {'prefix': 'nosr_', 'expected_count': 8},
        'drone_deploy': {'prefix': 'dploy_', 'expected_count': 9},
        'urban_scene': {'prefix': 'urbn_', 'expected_count': 3},
        'mill19_building': {'exact': 'mill19_building', 'expected_count': 1},
        'mill19_rubble': {'exact': 'mill19_rubble', 'expected_count': 1},
        'eyeful_apartment': {'exact': 'eft_apartment', 'expected_count': 1},
        'eyeful_kitchen': {'exact': 'eft_kitchen', 'expected_count': 1}
    }
    
    # Group scenes by dataset
    grouped_scenes = {dataset: [] for dataset in dataset_groups.keys()}
    
    for scene_name in results.keys():
        for dataset, config in dataset_groups.items():
            if 'prefix' in config and scene_name.startswith(config['prefix']):
                grouped_scenes[dataset].append(scene_name)
                break
            elif 'exact' in config and scene_name == config['exact']:
                grouped_scenes[dataset].append(scene_name)
                break
    
    # Print dataset grouping summary
    print("\nDataset Grouping Summary:")
    print("-" * 50)
    for dataset, scenes in grouped_scenes.items():
        expected = dataset_groups[dataset]['expected_count']
        actual = len(scenes)
        status = "✓" if actual == expected else f"⚠ (expected {expected})"
        print(f"{dataset:<20}: {actual:>2} scenes {status}")
        if actual != expected:
            print(f"  Scenes: {scenes}")
    
    # Compute aggregate metrics for each dataset
    dataset_metrics = {}
    
    for dataset, scenes in grouped_scenes.items():
        if not scenes:
            continue
            
        # Extract metrics for all scenes in this dataset
        ate_values = []
        rta3_values = []
        auc_rt3_values = []
        rta1_values = []
        auc_rt1_values = []
        
        for scene in scenes:
            if scene in results:
                res = results[scene]['res']
                
                # ATE
                ate_values.append(res['ate'])
                
                # RTA@3 (index 1)
                rta3_values.append(res['RTA'][1])
                
                # AUC-R&T@3 (average of rotation and translation AUC at index 1)
                auc_rt3 = (res['auc_r'][1] + res['auc_t'][1]) / 2
                auc_rt3_values.append(auc_rt3)
                
                # RTA@1 (index 0)
                rta1_values.append(res['RTA'][0])
                
                # AUC-R&T@1 (average of rotation and translation AUC at index 0)
                auc_rt1 = (res['auc_r'][0] + res['auc_t'][0]) / 2
                auc_rt1_values.append(auc_rt1)
        
        # Compute mean and std for each metric
        if ate_values:  # Only compute if we have data
            dataset_metrics[dataset] = {
                'ate_mean': np.mean(ate_values),
                'ate_std': np.std(ate_values),
                'rta3_mean': np.mean(rta3_values),
                'rta3_std': np.std(rta3_values),
                'auc_rt3_mean': np.mean(auc_rt3_values),
                'auc_rt3_std': np.std(auc_rt3_values),
                'rta1_mean': np.mean(rta1_values),
                'rta1_std': np.std(rta1_values),
                'auc_rt1_mean': np.mean(auc_rt1_values),
                'auc_rt1_std': np.std(auc_rt1_values),
                'scene_count': len(ate_values)
            }
    
    # Create CSV output
    csv_path = "/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm/dataset_summary_DUSt3R_RoMa.csv"
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Dataset', 'Scenes', 'ATE↓', 'RTA@3↑', 'AUC-R&T@3↑', 'RTA@1↑', 'AUC-R&T@1↑']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Write data for each dataset
        for dataset in dataset_groups.keys():
            if dataset in dataset_metrics:
                metrics = dataset_metrics[dataset]
                writer.writerow({
                    'Dataset': dataset,
                    'Scenes': metrics['scene_count'],
                    'ATE↓': f"{metrics['ate_mean']:.1e} ± {metrics['ate_std']:.1e}",
                    'RTA@3↑': f"{metrics['rta3_mean']:.1f} ± {metrics['rta3_std']:.1f}",
                    'AUC-R&T@3↑': f"{metrics['auc_rt3_mean']:.1f} ± {metrics['auc_rt3_std']:.1f}",
                    'RTA@1↑': f"{metrics['rta1_mean']:.1f} ± {metrics['rta1_std']:.1f}",
                    'AUC-R&T@1↑': f"{metrics['auc_rt1_mean']:.1f} ± {metrics['auc_rt1_std']:.1f}"
                })
            else:
                writer.writerow({
                    'Dataset': dataset,
                    'Scenes': 0,
                    'ATE↓': 'N/A',
                    'RTA@3↑': 'N/A',
                    'AUC-R&T@3↑': 'N/A',
                    'RTA@1↑': 'N/A',
                    'AUC-R&T@1↑': 'N/A'
                })
    
    # Print summary table
    print("\nDataset Summary Table:")
    print("=" * 120)
    print(f"{'Dataset':<20} {'Scenes':<7} {'ATE↓':<20} {'RTA@3↑':<15} {'AUC-R&T@3↑':<15} {'RTA@1↑':<15} {'AUC-R&T@1↑':<15}")
    print("-" * 120)
    
    for dataset in dataset_groups.keys():
        if dataset in dataset_metrics:
            metrics = dataset_metrics[dataset]
            ate_str = f"{metrics['ate_mean']:.1e}±{metrics['ate_std']:.1e}"
            rta3_str = f"{metrics['rta3_mean']:.1f}±{metrics['rta3_std']:.1f}"
            auc_rt3_str = f"{metrics['auc_rt3_mean']:.1f}±{metrics['auc_rt3_std']:.1f}"
            rta1_str = f"{metrics['rta1_mean']:.1f}±{metrics['rta1_std']:.1f}"
            auc_rt1_str = f"{metrics['auc_rt1_mean']:.1f}±{metrics['auc_rt1_std']:.1f}"
            
            print(f"{dataset:<20} {metrics['scene_count']:<7} "
                  f"{ate_str:<20} {rta3_str:<15} {auc_rt3_str:<15} {rta1_str:<15} {auc_rt1_str:<15}")
        else:
            print(f"{dataset:<20} {'0':<7} {'N/A':<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    print(f"\nCSV file saved to: {csv_path}")
    return csv_path


if __name__ == "__main__":
    generate_dataset_summary_csv()

    # main()
