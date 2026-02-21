"""3D Point Cloud Transformation and Visualization Module

Provides functionality for:
- Point cloud generation from depth and RGB images
- Camera pose visualization
- Point cloud export to PLY format
- Video creation from point cloud visualizations
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
import open3d as o3d
from PIL import Image
import natsort
import glob
import tqdm
import shutil
import os

from MargBA.datasets import HDF5Reader, monodepth2vls
from MargBA.geometry import to_homogeneous
from MargBA.poses_models import GlobalOptimizationPoseParameters

# Point Cloud Parameters
MAX_POINTS_PER_FRAME = 5000
TOTAL_POINT_CLOUD_SIZE = 500000
MIN_POINTS_THRESHOLD = 1000

# Visualization Parameters
CAMERA_SCALE = 0.01
FOV_DEGREES = 40.0
BACKGROUND_COLOR = [1, 1, 1, 1]
DEFAULT_RESOLUTION = 2048
DEFAULT_IMAGE_HEIGHT = 640

# Video Parameters
MIN_FRAME_RATE = 3
MIN_VIDEO_FPS = 30
PLY_VIDEO_DURATION = 3.0
PLY_ROTATION_FRAMES = 90


def adjust_intrinsic(intrinsic: torch.Tensor, depth_focalx: torch.Tensor, depth_focaly: torch.Tensor) -> torch.Tensor:
    """Adjust camera intrinsic parameters by modifying focal lengths."""
    fx_src, fy_src = intrinsic[:, 0, 0], intrinsic[:, 1, 1]
    fx_src_adj = torch.exp(torch.log(fx_src) + depth_focalx)
    fy_src_adj = torch.exp(torch.log(fy_src) + depth_focaly)

    intrinsic_adjusted = torch.clone(intrinsic)
    intrinsic_adjusted[:, 0, 0] = fx_src_adj
    intrinsic_adjusted[:, 1, 1] = fy_src_adj
    return intrinsic_adjusted


def compute_camera_size(poses: np.ndarray) -> float:
    """Compute appropriate camera frame size based on camera trajectory."""
    positions = poses[:, :3, 3]
    scene_diagonal = np.linalg.norm(np.max(positions, axis=0) - np.min(positions, axis=0))
    return scene_diagonal * CAMERA_SCALE


def create_camera_frame(pose: np.ndarray, size: float) -> o3d.geometry.PointCloud:
    """Create a coordinate frame for camera visualization."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(pose)
    return frame.sample_points_uniformly(number_of_points=300)


def create_camera_marker(pose: np.ndarray, size: float, color: str) -> o3d.geometry.PointCloud:
    """Create a colored sphere marker for camera visualization."""
    colors = {"green": [0.0, 1.0, 0.0], "blue": [0.0, 0.0, 1.0]}
    
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=size)
    sphere.compute_vertex_normals()
    sphere.translate(pose[:3, 3])
    
    pcd = sphere.sample_points_uniformly(number_of_points=500)
    pcd.paint_uniform_color(colors[color])
    return pcd


def create_point_cloud(points: np.ndarray, colors: np.ndarray, poses: np.ndarray, camera_size: float) -> o3d.geometry.PointCloud:
    """Create a point cloud with camera positions marked as coordinate frames."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    camera_points = []
    camera_colors = []

    for pose in poses:
        cam_pcd = create_camera_frame(pose, size=camera_size)
        camera_points.append(np.asarray(cam_pcd.points))
        camera_colors.append(np.asarray(cam_pcd.colors))

    if camera_points:
        camera_points = np.concatenate(camera_points, axis=0)
        camera_colors = np.concatenate(camera_colors, axis=0)
        pcd.points = o3d.utility.Vector3dVector(np.concatenate([points, camera_points], axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate([colors, camera_colors], axis=0))

    return pcd


def create_point_cloud_relocalization(points: np.ndarray, colors: np.ndarray, poses: np.ndarray, poses_gt: np.ndarray, camera_size: float) -> o3d.geometry.PointCloud:
    """Create a point cloud with camera positions marked as colored spheres."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    camera_points = []
    camera_colors = []

    for pose in poses:
        cam_pcd = create_camera_marker(pose, size=camera_size/4, color="green")
        camera_points.append(np.asarray(cam_pcd.points))
        camera_colors.append(np.asarray(cam_pcd.colors))

    for pose in poses_gt:
        cam_pcd = create_camera_marker(pose, size=camera_size/4, color="blue")
        camera_points.append(np.asarray(cam_pcd.points))
        camera_colors.append(np.asarray(cam_pcd.colors))

    if camera_points:
        camera_points = np.concatenate(camera_points, axis=0)
        camera_colors = np.concatenate(camera_colors, axis=0)
        pcd.points = o3d.utility.Vector3dVector(np.concatenate([points, camera_points], axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate([colors, camera_colors], axis=0))

    return pcd


def estimate_scene_radius(points: np.ndarray) -> float:
    """Estimate scene radius for camera placement."""
    center = np.mean(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return np.percentile(distances, 95)


def render_point_cloud_frames(pcd: o3d.geometry.PointCloud, output_dir: Path, mute: bool = False) -> None:
    """Render point cloud into a sequence of images using orbital camera."""
    points = np.asarray(pcd.points)
    radius = estimate_scene_radius(points) * 3
    center = np.median(points, axis=0)

    renderer = o3d.visualization.rendering.OffscreenRenderer(DEFAULT_RESOLUTION, DEFAULT_RESOLUTION)
    scene = renderer.scene
    scene.set_background(BACKGROUND_COLOR)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    scene.add_geometry("pcd", pcd, mat)

    iterator = range(PLY_ROTATION_FRAMES)
    if not mute:
        iterator = tqdm.tqdm(iterator, desc="Creating rotation frames")

    for i in iterator:
        theta = 2 * np.pi * i / PLY_ROTATION_FRAMES
        eye = [
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            center[2]
        ]

        renderer.setup_camera(FOV_DEGREES, center, eye, [0, 0, 1])
        img = renderer.render_to_image()
        frame_path = output_dir / f"{i:06d}.png"
        o3d.io.write_image(str(frame_path), img)


def process_frame_point_cloud(h5reader: HDF5Reader, image_name: str, intrinsic: torch.Tensor,
                            pose: torch.Tensor, depth_adj: float, depth_bias: float, num_points: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Process a single frame to extract point cloud and colors."""
    rgb = h5reader.read_jpg_image(f"{image_name}.jpg")
    depth = h5reader.read_depth_pr(image_name)
    depth = depth * depth_adj + depth_bias
    depth = torch.from_numpy(depth).float()

    selector = (depth > 0).view(-1)
    if torch.sum(selector) < MIN_POINTS_THRESHOLD:
        return None, None

    h, w = depth.shape
    xx, yy = np.meshgrid(range(w), range(h), indexing="xy")
    pts3d = torch.stack([
        torch.from_numpy(xx).float() * depth,
        torch.from_numpy(yy).float() * depth,
        depth
    ], dim=2)

    pts3d = pts3d.view(-1, 3, 1)
    pts3d = torch.inverse(intrinsic).unsqueeze(0) @ pts3d
    pts3d = to_homogeneous(pts3d.squeeze())
    pts3d = torch.inverse(pose) @ pts3d.unsqueeze(-1)
    pts3d = pts3d.squeeze()[:, :3]

    colors = torch.from_numpy(np.array(rgb)).float().view(-1, 3)
    pts3d, colors = pts3d[selector], colors[selector]

    if len(pts3d) == 0:
        return None, None

    indices = np.random.choice(len(pts3d), num_points, replace=True)
    return pts3d[indices].numpy(), colors[indices].numpy() / 255.0


def ensure_even_dimensions(image: np.ndarray) -> np.ndarray:
    """Ensure image dimensions are even for video encoding."""
    h, w = image.shape[:2]
    return image[:(h - h % 2), :(w - w % 2)]


def create_synchronized_visualization(rgb_depth_path: Path, ply_path: Path, output_path: Path,
                                   frame_idx: int, render_fps: int, input_fps: int, num_input_frames: int) -> None:
    """Create a synchronized visualization frame combining RGB-D and point cloud views."""
    ply_time = frame_idx / render_fps * MIN_VIDEO_FPS
    ply_idx = int(np.mod(int(ply_time), PLY_ROTATION_FRAMES))
    
    rgbd_time = frame_idx / render_fps
    rgbd_idx = int(min(np.floor(rgbd_time * input_fps), num_input_frames - 1))

    rgb_depth_paths = natsort.natsorted(glob.glob(str(rgb_depth_path / "*.jpg")))
    ply_paths = natsort.natsorted(glob.glob(str(ply_path / "*.png")))

    rgb_depth = Image.open(rgb_depth_paths[rgbd_idx])
    ply_frame = Image.open(ply_paths[ply_idx])
    
    w = int(ply_frame.width * rgb_depth.height / ply_frame.height)
    ply_frame = ply_frame.resize((w, rgb_depth.height))
    
    combined = np.concatenate([np.array(rgb_depth), np.array(ply_frame)], axis=1)
    combined = ensure_even_dimensions(combined)
    Image.fromarray(combined).save(output_path)


def create_visualization_video(frame_paths: Dict[str, Path], output_path: Path,
                             input_fps: int, num_input_frames: int, mute: bool = False) -> None:
    """Create visualization video from frame images."""
    if output_path.exists():
        output_path.unlink()

    render_fps = max(input_fps, MIN_VIDEO_FPS)
    duration = max(num_input_frames / input_fps, PLY_VIDEO_DURATION)
    total_frames = int(np.ceil(duration * render_fps))

    iterator = range(total_frames)
    if not mute:
        iterator = tqdm.tqdm(iterator, desc="Creating combined visualization")

    for idx in iterator:
        output_frame = frame_paths['combined'] / f"{idx:06d}.jpg"
        create_synchronized_visualization(
            frame_paths['video'],
            frame_paths['ply'],
            output_frame,
            idx,
            render_fps,
            input_fps,
            num_input_frames
        )
    
    os.system(
        f"ffmpeg -y -framerate {render_fps} -i {frame_paths['combined'] / '%06d.jpg'} "
        f"-c:v libx264 -pix_fmt yuv420p {output_path}"
    )


@torch.no_grad()
def export2ply(h5path: str, scene: str, framerate: int, pr_modelpath: str,
               gt_modelpath: Optional[str] = None, sharefocal: bool = True,
               mute: bool = False, marker_query_map: List[str] = None) -> Optional[Path]:
    """Export 3D reconstruction to PLY format and create visualization video."""
    if not Path(h5path).exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5path}")
    
    if framerate < MIN_FRAME_RATE:
        raise ValueError(f"Frame rate must be at least {MIN_FRAME_RATE}")

    h5reader = HDF5Reader(h5path)
    images = natsort.natsorted(h5reader.get_all_images())
    num_frames = len(images)

    pose_model = GlobalOptimizationPoseParameters(
        nfrm=num_frames,
        optimizefocal=False,
        sharefocal=sharefocal
    )
    pose_model.load_state_dict(torch.load(pr_modelpath), strict=True)

    output_dir = Path(pr_modelpath).parent
    temp_dirs = {
        'video': output_dir / "video",
        'ply': output_dir / "plys",
        'combined': output_dir / "combined"
    }
    
    for dir_path in temp_dirs.values():
        dir_path.mkdir(exist_ok=True)

    depth_adj, depth_bias, depth_focalx, depth_focaly = [
        x.cpu().detach().numpy() for x in pose_model.get_depth_adjustment()
    ]
    intrinsics = adjust_intrinsic(
        pose_model.get_intrinsic(),
        depth_focalx,
        depth_focaly
    ).cpu().detach()
    poses = pose_model.get_pose_w2c().cpu().detach()

    camera_size = compute_camera_size(poses.numpy())
    points_per_frame = min(TOTAL_POINT_CLOUD_SIZE // num_frames, MAX_POINTS_PER_FRAME)

    aggregated_points = []
    aggregated_colors = []

    iterator = enumerate(images)
    if not mute:
        iterator = enumerate(tqdm.tqdm(images, desc="Processing frames"))

    num_frames_processed = 0
    for idx, img_name in iterator:
        if marker_query_map is not None and marker_query_map[idx] == "map":
            continue

        points, colors = process_frame_point_cloud(
            h5reader, img_name, intrinsics[idx],
            poses[idx:idx + 1], depth_adj[idx].item(),
            depth_bias[idx].item(), points_per_frame
        )

        if points is not None:
            aggregated_points.append(points)
            aggregated_colors.append(colors)

        depth = h5reader.read_depth_pr(img_name)
        depth = monodepth2vls(
            1 / torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
            percentile=90,
            viewind=0
        )
        rgb = h5reader.read_jpg_image(f"{img_name}.jpg")
        
        rgb_depth = Image.fromarray(np.concatenate([np.array(rgb), depth], axis=1))
        w = int(DEFAULT_IMAGE_HEIGHT * rgb_depth.width / rgb_depth.height)
        rgb_depth = rgb_depth.resize((w, DEFAULT_IMAGE_HEIGHT))
        rgb_depth.save(temp_dirs['video'] / f"{img_name}.jpg")

        num_frames_processed += 1

    video_path = None
    if aggregated_points:
        points = np.concatenate(aggregated_points, axis=0)
        colors = np.concatenate(aggregated_colors, axis=0)

        if marker_query_map is None:
            pcd = create_point_cloud(points, colors, poses.numpy(), camera_size)
        else:
            pose_model_gt = GlobalOptimizationPoseParameters(
                nfrm=num_frames,
                optimizefocal=False,
                sharefocal=sharefocal
            )
            pose_model_gt.load_state_dict(torch.load(gt_modelpath), strict=True)
            poses_gt = pose_model_gt.get_pose_w2c().cpu().detach().numpy()
            pcd = create_point_cloud_relocalization(points, colors, poses.numpy(), poses_gt, camera_size)
        
        ply_path = output_dir / f"{scene}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)

        render_point_cloud_frames(pcd, temp_dirs['ply'], mute=mute)

        video_path = output_dir / f"{scene}.mp4"
        create_visualization_video(
            temp_dirs,
            video_path,
            framerate,
            num_frames_processed,
            mute=mute
        )

    for dir_path in temp_dirs.values():
        shutil.rmtree(dir_path)

    return video_path
