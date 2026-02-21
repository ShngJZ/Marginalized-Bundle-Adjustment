import os, copy, glob, sqlite3, json, sys
import natsort
import torch
import random
import numpy as np
import PIL.Image
import hashlib

from pathlib import Path
from collections import OrderedDict
from MargBA.datasets.data_utils import numpy_image_to_torch
from MargBA.datasets.read_write_model import read_cameras_text, read_cameras_binary, read_images_text, read_images_binary
sys.path.append(str(Path(__file__).parent / '../../third_party/fastmap'))
from eval import load_colmap_db_cams, read_json_gt_cams


class FASTMAPSfM(torch.utils.data.Dataset):
    def __init__(self, root, mode, cache_dir, rank, world_size):
        super().__init__()
        # Set a fixed seed for deterministic behavior
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        self.root = root
        self.mode = mode
        self.rank = rank
        self.world_size = world_size
        self.cache_dir = cache_dir

        # acquire map and query-seq images
        self.acquire_image_indices()
        # acquire pairs
        self.acquire_image_pairs()
        # load all ground truth poses
        self.load_all_ground_truth_poses()

        # Compute hashes before downsampling
        name2index_hash = self.compute_hash(self.name2index_mapper)
        image_indices_pair_hash_before = self.compute_hash(self.image_indices_pair)

        print(f"name2index_mapper/image_indices_pair hash: {name2index_hash[:6]}/{image_indices_pair_hash_before[:6]}")
        
        # Create empty files named after the hash values (using only first 6 digits)
        # If all ranks have the same hash, they will try to create the same file
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create empty file for name2index_hash
        open(os.path.join(self.cache_dir, f"name2index_hash_{name2index_hash[:6]}.txt"), 'w').close()
        
        # Create empty file for image_indices_pair_hash
        open(os.path.join(self.cache_dir, f"image_indices_pair_hash_{image_indices_pair_hash_before[:6]}.txt"), 'w').close()
        
        # Only save the full mapper for rank 0 to avoid duplication
        if rank == 0:
            # Convert OrderedDict to regular dict for JSON serialization
            name2index_dict = {k: dict(v) for k, v in self.name2index_mapper.items()}
            # Save the mapper to a JSON file
            cache_file = os.path.join(self.cache_dir, "name2index_mapper.json")
            with open(cache_file, 'w') as f:
                json.dump(name2index_dict, f, indent=2)
        
        # Down sample
        self.image_indices = self.image_indices[rank::world_size]
        self.image_indices_pair = self.image_indices_pair[rank::world_size]

    def acquire_image_indices(self):
        name2index_mapper = OrderedDict()

        # Root is a database file path
        db_path = self.root
        
        # Extract scene name from the database filename
        scene_name = os.path.splitext(os.path.basename(db_path))[0]
        
        # Get base directory (parent of the databases directory)
        base_dir = os.path.dirname(os.path.dirname(db_path))
        
        # Verify database file exists
        assert os.path.exists(db_path), f"Database file not found: {db_path}"
        
        # Read image names from the COLMAP database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT image_id, name FROM images ORDER BY image_id;")
        images = cursor.fetchall()
        conn.close()
        
        # Sort images by image_id for guaranteed deterministic ordering
        images = sorted(images, key=lambda x: x[0])
        
        # Create mapping from image name to index and store path mapping
        for i, (colmap_image_id, name) in enumerate(images):
            # Try different possible image paths
            possible_paths = [
                # 1. First try: images are in the "images" directory at the base level
                os.path.join(base_dir, "images", name),
                
                # 2. Second try: images are directly in the scene directory
                os.path.join(self.root, name),
                
                # 3. Third try: images are in the parent directory of the scene
                os.path.join(os.path.dirname(self.root), name),
                
                # 4. Fourth try: images are in a directory with the same name as the scene
                os.path.join(base_dir, scene_name, name),
                
                # 5. Fifth try: images are in an "images" directory with the scene name
                os.path.join(base_dir, "images", scene_name, name),
                
                # 6. Sixth try: images are in a directory with the scene name under "images"
                os.path.join(base_dir, "images", scene_name, name),
                
                # 7. Seventh try: images are in a directory with the scene name at the same level as databases
                os.path.join(os.path.dirname(self.root), scene_name, name)
            ]
            
            # Try each path
            image_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    image_path = path
                    break
            
            # Verify that the image path exists
            assert image_path is not None, f"Image file not found: {name}. Tried paths: {', '.join(possible_paths)}"
            
            name2index_mapper[name] = {
                'index': i,
                'path': image_path,
                'colmap_image_id': colmap_image_id
            }
        
        # Create lists of indices and names, and sort them by index for deterministic ordering
        self.image_indices, self.image_names = list(), list()
        for name in name2index_mapper:
            self.image_indices.append(name2index_mapper[name]['index'])
            self.image_names.append(name)
        
        # Check for duplicate image names using set comparison
        assert len(set(self.image_names)) == len(self.image_names), f"Duplicate image names found in dataset"
        
        # Sort both lists based on image indices to ensure deterministic ordering
        sorted_pairs = sorted(zip(self.image_indices, self.image_names))
        self.image_indices = [pair[0] for pair in sorted_pairs]
        self.image_names = [pair[1] for pair in sorted_pairs]
        
        # Check for duplicates again after sorting
        assert len(set(self.image_names)) == len(self.image_names), f"Duplicate image names found after sorting"
        
        self.name2index_mapper = name2index_mapper
    
    def compute_hash(self, data):
        """
        Compute a deterministic hash of a data structure.
        
        Args:
            data: The data structure to hash (dict, list, etc.)
            
        Returns:
            str: A hexadecimal hash string
        """
        # Convert data to a canonical string representation
        if isinstance(data, dict) or isinstance(data, OrderedDict):
            # Sort dictionary items by key for deterministic ordering
            items = sorted(data.items(), key=lambda x: str(x[0]))
            data_str = json.dumps(items, sort_keys=True)
        elif isinstance(data, list):
            # Convert list to a string representation
            data_str = json.dumps(data, sort_keys=True)
        else:
            # For other types, convert to string
            data_str = str(data)
        
        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(data_str.encode())
        return hash_obj.hexdigest()

    def acquire_image_pairs(self):
        # Dictionary to store pairs with their match counts
        # Format: {(idx1, idx2): match_count}
        pairs_with_counts = {}
        
        # Dictionary to track connections per frame
        # Format: {frame_idx: [(connected_idx, match_count), ...]}
        frame_connections = {}
        
        # Root is already a database file path
        db_path = self.root
        
        # Create a reverse mapping from COLMAP image_id to our dataset index
        colmap_id_to_index = {}
        for name, info in self.name2index_mapper.items():
            colmap_id_to_index[info['colmap_image_id']] = info['index']
        
        # Read matches from the COLMAP database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query the matches table
        cursor.execute("SELECT pair_id, data FROM matches;")
        matches_data = cursor.fetchall()
        
        # First pass: collect all pairs with their match counts
        for pair_id, matches_blob in matches_data:
            # Skip if matches_blob is None
            if matches_blob is None:
                continue
                
            # Decode pair_id to image_id1 and image_id2
            image_id2 = pair_id % 2147483647
            image_id1 = (pair_id - image_id2) // 2147483647
            
            # Count matches
            matches_array = np.frombuffer(matches_blob, dtype=np.uint32)
            match_count = len(matches_array) // 2
            
            # Skip pairs with no matches
            if match_count < 20:
                continue

            # Get the corresponding indices from our mapping
            if image_id1 in colmap_id_to_index and image_id2 in colmap_id_to_index:
                idx1 = colmap_id_to_index[image_id1]
                idx2 = colmap_id_to_index[image_id2]
                
                # Ensure idx1 < idx2 to avoid processing pairs twice
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                
                # Store the pair with its match count
                pairs_with_counts[(idx1, idx2)] = match_count
                
                # Track connections for each frame
                if idx1 not in frame_connections:
                    frame_connections[idx1] = []
                if idx2 not in frame_connections:
                    frame_connections[idx2] = []
                
                frame_connections[idx1].append((idx2, match_count))
                frame_connections[idx2].append((idx1, match_count))
        
        conn.close()
        
        # Second pass: limit each frame to have at most 200 connections
        MAX_CONNECTIONS_PER_FRAME = 250
        filtered_pairs = set()
        
        for frame_idx, connections in frame_connections.items():
            # Sort connections by match count in descending order
            # Use the frame index as a secondary sort key to ensure deterministic ordering when match counts are equal
            sorted_connections = sorted(connections, key=lambda x: (x[1], -x[0]), reverse=True)
            
            # Keep only the top MAX_CONNECTIONS_PER_FRAME connections
            top_connections = sorted_connections[:MAX_CONNECTIONS_PER_FRAME]
            
            # Add the filtered pairs to our set
            for connected_idx, _ in top_connections:
                # Ensure idx1 < idx2
                if frame_idx < connected_idx:
                    filtered_pairs.add((frame_idx, connected_idx))
                else:
                    filtered_pairs.add((connected_idx, frame_idx))
        
        # Convert the set of filtered pairs to a sorted list for deterministic behavior
        pairs = sorted([list(pair) for pair in filtered_pairs])

        # Verify that all frames have at least one connection
        connected_frames = set()
        for idx1, idx2 in pairs:
            connected_frames.add(idx1)
            connected_frames.add(idx2)

        # Assert that all frames are connected
        all_frames = set(frame_connections.keys())
        assert all_frames == connected_frames, f"Not all frames are connected! Missing: {all_frames - connected_frames}"

        self.image_indices_pair = pairs

    def _read_json_ground_truth_all(self, json_path):
        """Read all ground truth poses from JSON file"""
        with open(json_path, 'r') as f:
            gt_data = json.load(f)
        
        poses = {}
        for entry in gt_data:
            img_name = entry['fname']
            # Extract c2w matrix and convert to w2c
            c2w = np.array(entry['c2w'])
            world_to_cam = np.linalg.inv(c2w)
            poses[img_name] = world_to_cam
        
        return poses
    
    def _read_colmap_ground_truth_all(self, colmap_path):
        """Read all ground truth poses from COLMAP format"""
        poses = {}
        
        # Read COLMAP images.txt to find the image
        images_path = os.path.join(colmap_path, "images.txt")
        if not os.path.exists(images_path):
            return poses
            
        with open(images_path, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines
        lines = [line for line in lines if not line.startswith("#")]
        
        # Process image entries (every two lines is one entry)
        for i in range(0, len(lines), 2):
            if i+1 < len(lines):
                # Parse image entry
                image_line = lines[i].strip().split()
                name_line = lines[i+1].strip()
                
                # Extract image name
                img_name = name_line.split()[-1]
                
                # Extract quaternion and translation
                qw, qx, qy, qz = map(float, image_line[1:5])
                tx, ty, tz = map(float, image_line[5:8])
                
                # Convert quaternion to rotation matrix
                R = self._quaternion_to_rotation_matrix(qw, qx, qy, qz)
                
                # Construct world_to_cam matrix
                world_to_cam = np.eye(4)
                world_to_cam[:3, :3] = R
                world_to_cam[:3, 3] = [tx, ty, tz]
                
                poses[img_name] = world_to_cam
        
        return poses
    
    def _quaternion_to_rotation_matrix(self, qw, qx, qy, qz):
        """Convert quaternion to rotation matrix"""
        R = np.zeros((3, 3))
        
        # Normalize quaternion
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        qw /= norm
        qx /= norm
        qy /= norm
        qz /= norm
        
        # Convert to rotation matrix
        R[0, 0] = 1 - 2*qy*qy - 2*qz*qz
        R[0, 1] = 2*qx*qy - 2*qz*qw
        R[0, 2] = 2*qx*qz + 2*qy*qw
        
        R[1, 0] = 2*qx*qy + 2*qz*qw
        R[1, 1] = 1 - 2*qx*qx - 2*qz*qz
        R[1, 2] = 2*qy*qz - 2*qx*qw
        
        R[2, 0] = 2*qx*qz - 2*qy*qw
        R[2, 1] = 2*qy*qz + 2*qx*qw
        R[2, 2] = 1 - 2*qx*qx - 2*qy*qy
        
        return R
    
    def _match_gt_name_to_image_name(self, gt_fname, image_names):
        """
        Match a ground truth filename to an image name in our dataset.
        
        Args:
            gt_fname (str): Ground truth filename
            image_names (list): List of image names in our dataset
            
        Returns:
            str or None: Matched image name if found, None otherwise
        """
        # Try direct match first
        if gt_fname in image_names:
            return gt_fname
        
        # Try matching just the basename
        gt_basename = os.path.basename(gt_fname)
        if gt_basename in image_names:
            return gt_basename
        
        # No match found
        return None
    
    def _create_intrinsic_matrix_from_camera(self, camera):
        """
        Create an intrinsic matrix based on camera model.
        
        Args:
            camera: Camera object from COLMAP with model, params, width, height
            
        Returns:
            np.ndarray: 3x3 intrinsic matrix
            
        Raises:
            ValueError: If the camera model is not supported
        """
        if camera.model == "SIMPLE_PINHOLE":
            # [f, cx, cy]
            intrinsic = np.array([
                [camera.params[0], 0, camera.params[1]],
                [0, camera.params[0], camera.params[2]],
                [0, 0, 1]
            ])
        elif camera.model == "PINHOLE":
            # [fx, fy, cx, cy]
            intrinsic = np.array([
                [camera.params[0], 0, camera.params[2]],
                [0, camera.params[1], camera.params[3]],
                [0, 0, 1]
            ])
        elif camera.model == "SIMPLE_RADIAL":
            # [f, cx, cy, k1] - ignore k1 (radial distortion parameter)
            intrinsic = np.array([
                [camera.params[0], 0, camera.params[1]],
                [0, camera.params[0], camera.params[2]],
                [0, 0, 1]
            ])
        else:
            # Raise an exception for unsupported camera models
            raise ValueError(f"Unsupported camera model: {camera.model}. Only SIMPLE_PINHOLE, PINHOLE, and SIMPLE_RADIAL models are supported.")
        
        return intrinsic
    
    def load_all_ground_truth_poses(self):
        """Load all ground truth poses during initialization using fastmap's functions"""
        # Root is a database file path
        scene_name = os.path.splitext(os.path.basename(self.root))[0]
        base_dir = os.path.dirname(os.path.dirname(self.root))

        # Path to ground truths
        gt_path = os.path.join(base_dir, "ground_truths")
        
        # Initialize poses and intrinsics dictionaries
        self.world_to_cam_poses = {}
        self.cam_to_world_poses = {}
        self.intrinsics = {}  # Initialize intrinsics dictionary

        # Check if JSON file exists for this scene
        json_path = os.path.join(gt_path, f"{scene_name}.json")
        if os.path.exists(json_path):
            # Use fastmap's read_json_gt_cams function
            gt_fnames, c2w_poses = read_json_gt_cams(json_path)
            # Convert torch tensors to numpy arrays
            c2w_poses = c2w_poses.cpu().numpy()
            
            # Create dictionary mapping from image name to pose using our matching function
            for i, gt_fname in enumerate(gt_fnames):
                matched_name = self._match_gt_name_to_image_name(gt_fname, self.image_names)
                # Assert that matched_name is not None
                assert matched_name is not None, f"Could not match ground truth filename '{gt_fname}' to any image name in the dataset. Please check your ground truth data."
                self.cam_to_world_poses[matched_name] = c2w_poses[i]
                self.world_to_cam_poses[matched_name] = np.linalg.inv(c2w_poses[i])
        
        # If JSON didn't work, try COLMAP format
        if not self.cam_to_world_poses:
            colmap_path = os.path.join(gt_path, scene_name)
            if os.path.exists(colmap_path):
                # Determine if we have .bin or .txt files
                is_bin = os.path.exists(os.path.join(colmap_path, "cameras.bin"))
                ext = ".bin" if is_bin else ".txt"
                
                # Use fastmap's load_colmap_db_cams function
                gt_fnames, c2w_poses = load_colmap_db_cams(colmap_path, ext, return_all=True, device="cpu")
                # Convert torch tensors to numpy arrays
                c2w_poses = c2w_poses.cpu().numpy()
                
                # Load camera intrinsics from COLMAP using read_write_model functions
                cameras_file = os.path.join(colmap_path, f"cameras{ext}")
                images_file = os.path.join(colmap_path, f"images{ext}")
                
                # Read cameras and images based on file extension
                if ext == ".bin":
                    cameras = read_cameras_binary(cameras_file)
                    images = read_images_binary(images_file)
                else:
                    cameras = read_cameras_text(cameras_file)
                    images = read_images_text(images_file)
                
                # Create a mapping from image name to camera intrinsics
                image_to_camera = {image.name: cameras[image.camera_id] for _, image in images.items()}

                print(f"Loaded {len(cameras)} cameras and {len(images)} images from COLMAP")
                
                # Track matched names to check for duplicates
                matched_names = []
                
                # Create dictionary mapping from image name to pose using our matching function
                for i, gt_fname in enumerate(gt_fnames):
                    matched_name = self._match_gt_name_to_image_name(gt_fname, self.image_names)
                    # Assert that matched_name is not None
                    assert matched_name is not None, f"Could not match ground truth filename '{gt_fname}' to any image name in the dataset. Please check your ground truth data."
                    self.cam_to_world_poses[matched_name] = c2w_poses[i]
                    self.world_to_cam_poses[matched_name] = np.linalg.inv(c2w_poses[i])
                    
                    # Add to matched names list for duplicate checking
                    matched_names.append(matched_name)
                    
                    # Store intrinsics if available for this image
                    if gt_fname in image_to_camera:
                        camera = image_to_camera[gt_fname]
                        # Assert that the camera model is valid before creating the intrinsic matrix
                        assert camera.model in ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL"], f"Unsupported camera model: {camera.model}. Only SIMPLE_PINHOLE, PINHOLE, and SIMPLE_RADIAL models are supported."
                        self.intrinsics[matched_name] = self._create_intrinsic_matrix_from_camera(camera)
                
                # Check for duplicate matches
                assert len(set(matched_names)) == len(matched_names), f"Duplicate matches found when loading ground truth poses. Some image names were matched to multiple ground truth filenames."
        
        # Track which images have ground truth poses
        self.has_gt_pose = {img_name: img_name in self.cam_to_world_poses for img_name in self.image_names}

        # Report statistics
        num_with_gt = sum(self.has_gt_pose.values())
        print(f"Loaded {len(self.cam_to_world_poses)} ground truth poses for scene {scene_name}")
        print(f"Found ground truth poses for {num_with_gt}/{len(self.image_names)} images ({num_with_gt/len(self.image_names)*100:.1f}%)")
        
    def read_image_by_name(self, img_name):
        # Use the path stored in the name2index_mapper
        rgb_path = self.name2index_mapper[img_name]['path']

        # Load RGB image
        rgb = PIL.Image.open(rgb_path).convert('RGB')
        wd, ht = rgb.size
        
        # Get intrinsic matrix if available, otherwise use identity
        if img_name in self.intrinsics:
            intrinsic = self.intrinsics[img_name]
        else:
            # Initialize intrinsic as identity matrix
            intrinsic = np.eye(3)
        
        # Get the pre-loaded camera poses if available
        if self.has_gt_pose[img_name]:
            cam_to_world = self.cam_to_world_poses[img_name]
        else:
            # If no ground truth pose is available, use identity matrix
            cam_to_world = np.eye(4)
        
        # Create empty depth map
        depthmap = np.zeros([ht, wd])
        
        return np.array(rgb), intrinsic, cam_to_world, depthmap

    def __len__(self):
        """
        Returns the number of items in the dataset based on the current mode.
        
        Returns:
            int: Number of items in the dataset.
                - For 'monodepth' mode: Number of individual images.
                - For 'correspondence' mode: Number of image pairs.
        
        Raises:
            ValueError: If the mode is not supported.
        """

        if self.mode == 'monodepth':
            return len(self.image_indices)
        elif self.mode == 'correspondence':
            return len(self.image_indices_pair)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Expected 'monodepth' or 'correspondence'.")

    def corres_getitem(self, idx):
        image_idx_src, image_idx_dst = self.image_indices_pair[idx]
        name_src, name_dst = self.image_names[image_idx_src], self.image_names[image_idx_dst]
        
        # Check if both images have ground truth poses
        has_valid_pose = self.has_gt_pose[name_src] and self.has_gt_pose[name_dst]
        
        # Read images and poses
        rgb_src, intrinsic_src, cam_to_world_src, _ = self.read_image_by_name(img_name=name_src)
        rgb_dst, intrinsic_dst, cam_to_world_dst, _ = self.read_image_by_name(img_name=name_dst)
        
        # Convert images to torch tensors
        rgb_src = numpy_image_to_torch(rgb_src.astype(np.float32))
        rgb_dst = numpy_image_to_torch(rgb_dst.astype(np.float32))
        
        # Compute relative pose
        pose_src2dst = np.linalg.inv(cam_to_world_dst) @ cam_to_world_src

        ret = {
            'intrinsic_src': intrinsic_src,
            'intrinsic_dst': intrinsic_dst,
            "rgb_src": rgb_src,
            "rgb_dst": rgb_dst,
            "rgb_src_wo_resize": rgb_src,
            "rgb_dst_wo_resize": rgb_dst,
            'pose_src2dst': pose_src2dst[:3].astype(np.float32),
            'image_idx_pair': [image_idx_src, image_idx_dst],
            'has_valid_pose': has_valid_pose,  # Flag indicating if both images have ground truth poses
        }
        return ret

    def monodepth_getitem(self, idx):
        image_idx = self.image_indices[idx]
        rgb_file = self.image_names[image_idx]
        
        # Check if image has ground truth pose
        has_gt_pose = self.has_gt_pose[rgb_file]
        
        # Read image and pose
        rgb, intrinsic, pose_c2w, depth_gt = self.read_image_by_name(img_name=rgb_file)
        
        # Convert image to torch tensor
        rgb = numpy_image_to_torch(rgb.astype(np.float32))

        ret = {
            'image_idx': image_idx,
            "rgb_file": rgb_file,
            'depth_gt': depth_gt,
            'image': rgb,
            'intr': intrinsic,
            'pose_c2w': pose_c2w,
            'has_gt_pose': has_gt_pose,  # Flag indicating if this image has ground truth pose
        }
        return ret

    def __getitem__(self, idx: int):
        """
        Retrieves dataset items based on the current mode.
        
        Args:
            idx (int): Index of the item to retrieve.
        
        Returns:
            dict: A dictionary containing the following keys:
                For 'monodepth' mode:
                    - image_idx (int): Index of the image in the dataset.
                    - rgb_file (str): Filename of the RGB image.
                    - depth_gt (np.ndarray): Ground-truth depth map with shape (H, W).
                    - image (torch.Tensor): Normalized RGB image tensor of shape (3, H, W), with values in [0.0, 1.0].
                    - intr (np.ndarray): Camera intrinsic matrix of shape (3, 3).
                    - pose_c2w (np.ndarray): Camera-to-world transformation matrix of shape (4, 4).
                    - has_gt_pose (bool): Flag indicating if this image has a ground truth pose.
                
                For 'correspondence' mode:
                    - intrinsic_src, intrinsic_dst (np.ndarray): Camera intrinsic matrices of shape (3, 3).
                    - rgb_src, rgb_dst (torch.Tensor): Normalized RGB image tensors of shape (3, H, W).
                    - rgb_src_wo_resize, rgb_dst_wo_resize (torch.Tensor): Same as above, included for compatibility.
                    - pose_src2dst (np.ndarray): Relative pose from source to destination camera of shape (3, 4).
                    - image_idx_pair (list): Indices of the source and destination images [src_idx, dst_idx].
                    - has_valid_pose (bool): Flag indicating if both images have ground truth poses.
        
        Raises:
            ValueError: If the mode is not supported.
            IndexError: If the index is out of bounds.
        """
        # Check if index is valid
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        
        # Dispatch to the appropriate getter based on mode
        if self.mode == 'monodepth':
            ret = self.monodepth_getitem(idx)
        elif self.mode == 'correspondence':
            ret = self.corres_getitem(idx)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Expected 'monodepth' or 'correspondence'.")
        
        return ret
