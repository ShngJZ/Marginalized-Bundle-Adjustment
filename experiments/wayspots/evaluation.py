import numpy as np
import cv2
import torch

@torch.no_grad()
def wayspots_evaluate(
        pr_pose_w2c,
        gt_pose_w2c,
        marker_query_map,  # Dictionary mapping frame indices to 'qry'/'ref'
        nfrm,  # Total number of frames
        params  # Additional parameters (unused in current implementation)
):
    """
    Evaluate pose estimation accuracy by comparing predicted poses against ground truth.
    Returns metrics including rotation/translation errors and accuracy percentages.
    """

    # Initialize error containers
    rErrs, tErrs = [], []  # Rotation and translation errors

    for i in range(nfrm):
        if marker_query_map[i] == 'qry':
            # Get predicted and ground truth poses for current frame
            out_pose, gt_pose_44 = pr_pose_w2c[i], gt_pose_w2c[i]

            # Calculate translation error (Euclidean distance)
            t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]).item())

            # Convert poses to numpy arrays for rotation error calculation
            out_pose, gt_pose_44 = out_pose.detach().cpu().numpy(), gt_pose_44.detach().cpu().numpy()

            # Calculate rotation error using Rodrigues' rotation formula
            gt_R = gt_pose_44[0:3, 0:3]
            out_R = out_pose[0:3, 0:3]
            r_err = np.matmul(out_R, np.transpose(gt_R))
            r_err = cv2.Rodrigues(r_err)[0]  # Get axis-angle representation
            r_err = np.linalg.norm(r_err) * 180 / np.pi  # Convert to degrees

            # Store errors
            rErrs.append(r_err)
            tErrs.append(t_err)

    # Convert to numpy arrays for vectorized operations
    rErrs = np.array(rErrs)
    tErrs = np.array(tErrs)

    # Package evaluation metrics
    result = {
        "10cm/5deg": ((rErrs < 5) * (tErrs < 0.1)).mean() * 100,  # Both conditions
        "50cm/5deg": ((rErrs < 5) * (tErrs < 0.5)).mean() * 100,  # Relaxed translation
        "5deg": (rErrs < 5).mean() * 100,  # Rotation only
        "5cm": (tErrs < 0.1).mean() * 100,  # Translation only
    }
    return result