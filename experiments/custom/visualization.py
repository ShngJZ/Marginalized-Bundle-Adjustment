import os, argparse, sys
import numpy as np
import socket


prj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, prj_root)
from MargBA.geometry.export2ply import export2ply

def find_free_port(start_port=12362, end_port=13962, max_attempts=100):
    """
    Find a free port in the given range.

    Args:
        start_port: Lower bound of port range
        end_port: Upper bound of port range
        max_attempts: Maximum number of attempts to find a free port

    Returns:
        A free port number as string, or None if no free port is found
    """
    used_ports = set()
    attempt = 0

    while attempt < max_attempts:
        # Generate a random port in the specified range
        port = np.random.randint(start_port, end_port)

        # Skip if we've already tried this port
        if port in used_ports:
            continue

        used_ports.add(port)
        attempt += 1

        # Check if the port is available
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Try to bind to the port
                s.bind(('localhost', port))
                return str(port)
            except socket.error:
                # Port is in use, try another one
                continue

    raise RuntimeError(f"Could not find a free port after {max_attempts} attempts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Marginalized Bundle Adjustment')
    parser.add_argument(
        '--data-root', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment-Datasets/custom"
    )
    parser.add_argument(
        '--output-location', type=str, default="/home/ubuntu/disk6/Marginalized-Bundle-Adjustment/release/custom"
    )
    parser.add_argument(
        '--depth-model', type=str, choices=['ZoeDepth', 'UniDepth', 'DUSt3R'], default='DUSt3R'
    )
    parser.add_argument(
        '--corres-model', type=str, choices=['RoMa', 'MASt3R'], default='RoMa'
    )

    args = parser.parse_args()

    preprocess_location = os.path.join(
        args.output_location, f"{args.depth_model}_{args.corres_model}"
    )

    scene = "custom"
    dst_perscene = os.path.join(preprocess_location)
    sfm_perscene = os.path.join(f"{preprocess_location}_sfm")

    h5path = os.path.join(dst_perscene, f"{scene}.hdf5")
    pr_modelpath = os.path.join(sfm_perscene, "pr_pose_model_fine.ckpt")
    gt_modelpath = os.path.join(sfm_perscene, "gt_pose_model.ckpt")

    export2ply(
        h5path=h5path,
        scene=scene,
        framerate=20,
        pr_modelpath=pr_modelpath,
        gt_modelpath=None,
        sharefocal=True,
    )