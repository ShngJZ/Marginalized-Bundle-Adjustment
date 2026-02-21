import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from scipy import spatial
import einops

# Add the read_write_model module to path
sys.path.append(str(Path(__file__).parent / '../../MargBA/datasets'))
from read_write_model import read_model, qvec2rotmat

# Add MargBA to path for pose models
sys.path.append(str(Path(__file__).parent / '../../'))
# Import specific functions we need without importing the full evaluation.py
# to avoid the fastmap.io dependency
def load_json(fname):
    with open(fname, "r") as f:
        return json.load(f)


def write_json(fname, payload, **kwargs):
    with open(fname, "w") as f:
        json.dump(payload, f, **kwargs)


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
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)

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

    return stats


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
    """Load poses from checkpoint file"""
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
    
    name_mapping = load_json(name_mapping_file)
    
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


def load_tnt_sail_gt_poses(gt_path, device="cuda"):
    """Load ground truth poses from TNT SAIL dataset in COLMAP format"""
    # Detect format (.bin or .txt)
    is_bin = (gt_path / 'cameras.bin').is_file()
    is_txt = (gt_path / 'cameras.txt').is_file()
    
    if not (is_bin or is_txt):
        raise FileNotFoundError(f"No COLMAP model found at {gt_path}")
    
    ext = '.bin' if is_bin else '.txt'
    
    # Read COLMAP model
    cameras, images, points3D = read_model(str(gt_path), ext)
    
    # Extract poses from images
    image_names = []
    Rs = []
    ts = []
    
    # Sort images by ID to ensure consistent ordering
    sorted_images = sorted(images.items(), key=lambda x: x[0])
    
    for image_id, image in sorted_images:
        image_names.append(image.name)
        
        # Convert quaternion to rotation matrix
        R = qvec2rotmat(image.qvec)
        t = image.tvec
        
        Rs.append(R)
        ts.append(t)
    
    # Convert to tensors
    Rs = torch.tensor(np.stack(Rs), dtype=torch.float64, device=device)
    ts = torch.tensor(np.stack(ts), dtype=torch.float64, device=device)
    
    # Convert from world-to-camera to camera-to-world
    w2c_poses = Rt_to_T(R=Rs, t=ts)  # [N, 4, 4]
    c2w_poses = batch_inv_pose(w2c_poses)
    
    return image_names, c2w_poses


def do_eval_tnt_sail(pd_path: Path, gt_path: Path):
    """Evaluate TNT SAIL dataset with new pseudo ground truth"""
    assert pd_path.is_dir(), f"Prediction path does not exist: {pd_path}"
    assert gt_path.is_dir(), f"Ground truth path does not exist: {gt_path}"
    
    # Load predicted poses from checkpoint
    pd_fnames, pd_poses = load_largest_cam_cluster(pd_path)
    
    # Load ground truth poses from COLMAP format
    gt_fnames, gt_poses = load_tnt_sail_gt_poses(gt_path)
    
    # Ensure no duplicate filenames
    assert len(set(gt_fnames)) == len(gt_fnames), "Duplicate ground truth filenames found"
    assert len(set(pd_fnames)) == len(pd_fnames), "Duplicate predicted filenames found"
    
    N = len(gt_fnames)
    
    # Match filenames between ground truth and predictions
    missing_frames = []
    inds = []
    used_pd_indices = set()
    
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
    
    # Compute evaluation metrics
    results = pose_stats_suite(gt_poses, resampled_pd_poses)
    
    out = {}
    out['res'] = results
    out['nums'] = (len(gt_poses), len(pd_poses), 0)  # No fill-in since we raise exception on mismatch
    
    return out


def main():
    parser = argparse.ArgumentParser(description='Evaluate TNT SAIL Dataset with New Pseudo Ground Truth')
    parser.add_argument(
        '--gt-root', type=str, 
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/tnt_sail",
        help='Root directory containing TNT SAIL pseudo ground truth'
    )
    parser.add_argument(
        '--output-location', type=str, 
        default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/fastmap_sfm",
        help='Directory containing SfM results'
    )
    parser.add_argument(
        '--depth-model', type=str, choices=['ZoeDepth', 'UniDepth', 'DUSt3R'], 
        default='DUSt3R',
        help='Depth model used for SfM'
    )
    parser.add_argument(
        '--corres-model', type=str, choices=['RoMa', 'MASt3R'], 
        default='RoMa',
        help='Correspondence model used for SfM'
    )
    parser.add_argument(
        '--scenes', type=str, nargs='*',
        help='Specific scenes to evaluate (if not provided, evaluates all available scenes)'
    )

    args = parser.parse_args()

    # TNT SAIL scene name mapping: GT name -> SfM result name
    tnt_scene_mapping = {
        # Advanced scenes
        'advanced__Auditorium': 'tnt_advn_Auditorium',
        'advanced__Ballroom': 'tnt_advn_Ballroom', 
        'advanced__Courtroom': 'tnt_advn_Courtroom',
        'advanced__Museum': 'tnt_advn_Museum',
        'advanced__Palace': 'tnt_advn_Palace',
        'advanced__Temple': 'tnt_advn_Temple',
        # Intermediate scenes
        'intermediate__Family': 'tnt_intrmdt_Family',
        'intermediate__Francis': 'tnt_intrmdt_Francis',
        'intermediate__Horse': 'tnt_intrmdt_Horse',
        'intermediate__Lighthouse': 'tnt_intrmdt_Lighthouse',
        'intermediate__M60': 'tnt_intrmdt_M60',
        'intermediate__Panther': 'tnt_intrmdt_Panther',
        'intermediate__Playground': 'tnt_intrmdt_Playground',
        'intermediate__Train': 'tnt_intrmdt_Train',
        # Training scenes
        'training__Barn': 'tnt_trng_Barn',
        'training__Caterpillar': 'tnt_trng_Caterpillar',
        'training__Church': 'tnt_trng_Church',
        'training__Courthouse': 'tnt_trng_Courthouse',
        'training__Ignatius': 'tnt_trng_Ignatius',
        'training__Meetingroom': 'tnt_trng_Meetingroom',
        'training__Truck': 'tnt_trng_Truck'
    }
    
    # Get all TNT SAIL scenes
    tnt_sail_scenes = list(tnt_scene_mapping.keys())
    
    # Filter scenes if specified
    if args.scenes:
        tnt_sail_scenes = [scene for scene in tnt_sail_scenes if scene in args.scenes]
    
    # Paths
    gt_root = Path(args.gt_root)
    sfm_location = Path(args.output_location) / f"{args.depth_model}_{args.corres_model}_sfm"
    
    # Results storage
    all_results = {}
    successful_scenes = []
    skipped_scenes = []
    
    print(f"Evaluating TNT SAIL dataset with {len(tnt_sail_scenes)} scenes...")
    print(f"Ground truth root: {gt_root}")
    print(f"SfM results location: {sfm_location}")
    print("-" * 80)
    
    for gt_scene in tnt_sail_scenes:
        # Get the corresponding SfM scene name
        sfm_scene = tnt_scene_mapping[gt_scene]
        
        # Check if fine model exists
        sfm_perscene = sfm_location / sfm_scene
        fine_model_path = sfm_perscene / "pr_pose_model_fine.ckpt"
        
        if not fine_model_path.exists():
            print(f"Skipping {gt_scene}: No fine model found at {fine_model_path}")
            skipped_scenes.append(gt_scene)
            continue

        # Ground truth path (using subfolder "0" as mentioned in the task)
        gt_path = gt_root / gt_scene / "0"
        
        if not gt_path.exists():
            print(f"Skipping {gt_scene}: No ground truth found at {gt_path}")
            skipped_scenes.append(gt_scene)
            continue
        
        print(f"\nEvaluating {gt_scene} (SfM: {sfm_scene})...")
        print(f"  GT: {gt_path}")
        print(f"  PD: {sfm_perscene}")

        result = do_eval_tnt_sail(sfm_perscene, gt_path)
        all_results[gt_scene] = result
        successful_scenes.append(gt_scene)

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
    print("TNT SAIL EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total scenes: {len(tnt_sail_scenes)}")
    print(f"Successfully evaluated: {len(successful_scenes)}")
    print(f"Skipped: {len(skipped_scenes)}")
    
    if skipped_scenes:
        print(f"\nSkipped scenes: {skipped_scenes}")
    
    if successful_scenes:
        # Use all successful scenes for summary calculations
        summary_scenes = successful_scenes
        
        print("\nAggregated Results:")
        print(f"Using {len(summary_scenes)} scenes for summary")
        
        # Simple metrics
        metrics = ['ate', 'mAA@30']
        for metric in metrics:
            values = [all_results[scene]['res'][metric] for scene in summary_scenes]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        # AUC metrics (list of values)
        auc_metrics = ['auc_p', 'auc_t', 'auc_r']
        for metric in auc_metrics:
            print(f"\n  {metric}:")
            for i, threshold in enumerate([1, 3, 5, 10]):
                values = [all_results[scene]['res'][metric][i] for scene in summary_scenes]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"    @{threshold}°: {mean_val:.2f} ± {std_val:.2f}")
        
        # RRA/RTA metrics
        rra_rta_metrics = ['RRA', 'RTA']
        for metric in rra_rta_metrics:
            print(f"\n  {metric}:")
            for i, threshold in enumerate([1, 3, 5, 10, 15]):
                values = [all_results[scene]['res'][metric][i] for scene in summary_scenes]
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"    @{threshold}°: {mean_val:.2f} ± {std_val:.2f}")
        
        # Group results by TNT difficulty level
        difficulty_groups = {
            'training': [s for s in successful_scenes if s.startswith('training__')],
            'intermediate': [s for s in successful_scenes if s.startswith('intermediate__')],
            'advanced': [s for s in successful_scenes if s.startswith('advanced__')]
        }
        
        print("\nResults by Difficulty Level:")
        print("-" * 60)
        for difficulty, scenes in difficulty_groups.items():
            if not scenes:
                continue
                
            print(f"\n{difficulty.upper()} ({len(scenes)} scenes):")
            
            # ATE
            ate_values = [all_results[scene]['res']['ate'] for scene in scenes]
            ate_mean = np.mean(ate_values)
            ate_std = np.std(ate_values)
            print(f"  ATE: {ate_mean:.4f} ± {ate_std:.4f}")
            
            # AUC@5 for pose, rotation, translation
            for metric, name in [('auc_p', 'Pose'), ('auc_r', 'Rotation'), ('auc_t', 'Translation')]:
                values = [all_results[scene]['res'][metric][2] for scene in scenes]  # Index 2 is @5°
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  AUC@5 ({name}): {mean_val:.2f} ± {std_val:.2f}")
            
            # mAA@30
            maa_values = [all_results[scene]['res']['mAA@30'] for scene in scenes]
            maa_mean = np.mean(maa_values)
            maa_std = np.std(maa_values)
            print(f"  mAA@30: {maa_mean:.2f} ± {maa_std:.2f}")
        
        # Per-scene detailed results
        print("\nPer-scene Detailed Results:")
        print(f"{'Scene':<25} {'ATE↓':<12} {'RRA@5↑':<10} {'RTA@5↑':<10} {'AUC-P@5↑':<10} {'AUC-R@5↑':<10} {'AUC-T@5↑':<10} {'mAA@30↓':<10}")
        print("-" * 105)
        
        for scene in sorted(successful_scenes):
            res = all_results[scene]['res']
            ate_str = f"{res['ate']:.1e}"
            rra5 = res['RRA'][2]  # @5° (index 2)
            rta5 = res['RTA'][2]  # @5° (index 2)
            auc_p5 = res['auc_p'][2]  # @5°
            auc_r5 = res['auc_r'][2]  # @5°
            auc_t5 = res['auc_t'][2]  # @5°
            maa30 = res['mAA@30']
            
            print(f"{scene:<25} {ate_str:<12} {rra5:<10.1f} {rta5:<10.1f} {auc_p5:<10.1f} {auc_r5:<10.1f} {auc_t5:<10.1f} {maa30:<10.1f}")
        
        # Final summary with key metrics (excluding specified scenes)
        print("\n" + "=" * 60)
        print("FINAL SUMMARY - Key Metrics")
        print("=" * 60)
        
        # Calculate overall averages using filtered scenes
        ate_values = [all_results[scene]['res']['ate'] for scene in summary_scenes]
        rra5_values = [all_results[scene]['res']['RRA'][2] for scene in summary_scenes]  # @5°
        rta5_values = [all_results[scene]['res']['RTA'][2] for scene in summary_scenes]  # @5°
        
        ate_mean = np.mean(ate_values)
        ate_std = np.std(ate_values)
        rra5_mean = np.mean(rra5_values)
        rra5_std = np.std(rra5_values)
        rta5_mean = np.mean(rta5_values)
        rta5_std = np.std(rta5_values)
        
        print(f"Average over {len(summary_scenes)} scenes:")
        print(f"  RRA@5↑: {rra5_mean:.2f} ± {rra5_std:.2f}")
        print(f"  RTA@5↑: {rta5_mean:.2f} ± {rta5_std:.2f}")
        print(f"  ATE↓:   {ate_mean:.1e} ± {ate_std:.1e}")
    
    # Save results
    output_file = sfm_location.parent / f"tnt_sail_evaluation_results_{args.depth_model}_{args.corres_model}.json"
    write_json(str(output_file), {
        'args': vars(args),
        'results': all_results,
        'successful_scenes': successful_scenes,
        'skipped_scenes': skipped_scenes,
        'dataset': 'TNT_SAIL'
    }, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
