import os, sys, torch, re

# Get the root path of dust3r library
mast3r_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
    'third_party',
    'dust3r'
)
sys.path.insert(0, mast3r_root)
from dust3r_visloc.evaluation import get_pose_error, aggregate_stats

def sevenscenes_evaluate_sequence_results(
        predicted_poses_w2c: torch.Tensor,  # Predicted world-to-camera poses
        ground_truth_poses_w2c: torch.Tensor,  # Ground truth world-to-camera poses
        query_frame_mask: list,  # List indicating which frames are query frames ('qry' or other)
        total_frames: int,  # Total number of frames in sequence
        eval_params: dict  # Evaluation parameters dictionary
) -> tuple[list, list]:
    """Calculate position and angular errors for query frames in a sequence.

    Returns:
        tuple: (position_errors, angular_errors) lists for all query frames
    """
    position_errors = []
    angular_errors = []

    for frame_idx in range(total_frames):
        if query_frame_mask[frame_idx] == 'qry':
            # Convert poses to camera-to-world and compute errors
            pred_pose_c2w = torch.inverse(predicted_poses_w2c[frame_idx]).detach().cpu().numpy()
            gt_pose_c2w = torch.inverse(ground_truth_poses_w2c[frame_idx]).detach().cpu().numpy()

            trans_error, angle_error = get_pose_error(pred_pose_c2w, gt_pose_c2w)
            position_errors.append(trans_error)
            angular_errors.append(angle_error)

    return position_errors, angular_errors


@torch.no_grad()
def sevenscenes_evaluate(
        pr_pose_w2c,
        gt_pose_w2c,
        marker_query_map,
        nfrm,
        params
) -> dict:
    """Evaluate pose estimation performance on 7Scenes dataset.

    Returns:
        dict: Dictionary containing median errors and accuracy metrics
    """

    # Calculate errors for query frames
    pos_errors, ang_errors = sevenscenes_evaluate_sequence_results(
        pr_pose_w2c, gt_pose_w2c, marker_query_map, nfrm, params
    )

    # Aggregate statistics and parse results
    stats_str = aggregate_stats(
        f'{os.path.basename(params["sfm_location"])}',
        pos_errors,
        ang_errors
    )

    # Extract metrics using regex
    median_metrics = re.search(
        r'median_pos_error=([\d.]+), median_angular_error=([\d.]+)',
        stats_str
    )
    accuracy_metrics = re.findall(
        r'acc@[\d.m]+,?[\ddeg]+=(\d+\.\d+)',
        stats_str
    )

    assert median_metrics and len(accuracy_metrics) == 4

    # Package results into dictionary
    return {
        "median_pos_error": float(median_metrics.group(1)),
        "median_angular_error": float(median_metrics.group(2)),
        "acc_01m_1deg": float(accuracy_metrics[0]),
        "acc_025m_2deg": float(accuracy_metrics[1]),
        "acc_05m_5deg": float(accuracy_metrics[2]),
        "acc_5m_10deg": float(accuracy_metrics[3]),
    }