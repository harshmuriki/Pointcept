"""
Peach Tree Dataset for Binary Segmentation (Keep vs Prune)

Author: Custom implementation for peach tree pruning
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class PeachTreeDataset(Dataset):
    """
    Custom dataset for peach tree binary segmentation (keep vs prune)
    Uses .npy files in ScanNet-like format but with segment2.npy for binary labels
    """

    def __init__(
        self,
        split="train",
        data_root="data/peachtree",
        transform=None,
        ignore_index=-1,
        test_mode=False,
        test_cfg=None,
        cache=False,
        loop=1,
    ):
        super(PeachTreeDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        self.ignore_index = ignore_index
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        """Get list of scene directories"""
        if isinstance(self.split, str):
            split_path = os.path.join(self.data_root, self.split)
            if os.path.exists(split_path):
                data_list = [
                    os.path.join(split_path, d) 
                    for d in os.listdir(split_path) 
                    if os.path.isdir(os.path.join(split_path, d))
                ]
            else:
                data_list = []
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                split_path = os.path.join(self.data_root, split)
                if os.path.exists(split_path):
                    data_list += [
                        os.path.join(split_path, d) 
                        for d in os.listdir(split_path) 
                        if os.path.isdir(os.path.join(split_path, d))
                    ]
        else:
            raise NotImplementedError
        
        return sorted(data_list)

    def get_data_name(self, idx):
        """Get data name from index"""
        data_path = self.data_list[idx % len(self.data_list)]
        return os.path.basename(data_path)

    def get_data(self, idx):
        """Load data from .npy files"""
        data_path = self.data_list[idx % len(self.data_list)]
        
        # Load individual .npy files
        coord_path = os.path.join(data_path, "coord.npy")
        color_path = os.path.join(data_path, "color.npy")
        normal_path = os.path.join(data_path, "normal.npy")
        segment_path = os.path.join(data_path, "segment2.npy")  # Use segment2 for binary classification
        tree_type_path = os.path.join(data_path, "tree_type.npy")  # Tree type classification
        
        # Check if files exist
        required_files = [coord_path, color_path, segment_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load data
        coord = np.load(coord_path).astype(np.float32)
        color = np.load(color_path).astype(np.float32)
        segment = np.load(segment_path).astype(np.int64)
        
        # Load tree type if exists
        if os.path.exists(tree_type_path):
            tree_type = np.load(tree_type_path).astype(np.int64)
        else:
            # Default to type 0 if tree_type.npy doesn't exist
            tree_type = np.zeros(len(coord), dtype=np.int64)
        
        # Load normal if exists, otherwise compute normals from geometry
        if os.path.exists(normal_path):
            normal = np.load(normal_path).astype(np.float32)
        else:
            # Compute normals from point cloud geometry using local neighborhoods
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coord)
            
            # Estimate normals with reasonable parameters for tree data
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.01,  # 1cm radius for normal estimation
                    max_nn=30     # maximum 30 neighbors
                )
            )
            
            # Ensure consistent normal orientation (important for trees)
            pcd.orient_normals_consistent_tangent_plane(k=30)
            
            normal = np.asarray(pcd.normals).astype(np.float32)
            
            # Validate computed normals
            if normal.shape[0] != coord.shape[0]:
                logger = get_root_logger()
                logger.warning(f"Normal computation failed for {data_path}, using dummy normals")
                normal = np.zeros_like(coord)
                normal[:, 2] = 1.0  # Point up in Z direction
        
        # Ensure all arrays have the same number of points
        n_points = coord.shape[0]
        assert color.shape[0] == n_points, f"Color shape mismatch: {color.shape[0]} vs {n_points}"
        assert normal.shape[0] == n_points, f"Normal shape mismatch: {normal.shape[0]} vs {n_points}"
        assert segment.shape[0] == n_points, f"Segment shape mismatch: {segment.shape[0]} vs {n_points}"
        assert tree_type.shape[0] == n_points, f"Tree type shape mismatch: {tree_type.shape[0]} vs {n_points}"
        
        # Validate data ranges
        assert coord.shape[1] == 3, f"Coordinate should be 3D: {coord.shape}"
        assert color.shape[1] == 3, f"Color should be 3D: {color.shape}"
        assert normal.shape[1] == 3, f"Normal should be 3D: {normal.shape}"
        assert len(segment.shape) == 1, f"Segment should be 1D: {segment.shape}"
        assert len(tree_type.shape) == 1, f"Tree type should be 1D: {tree_type.shape}"
        
        # Ensure colors are in [0, 1] range
        if color.max() > 1.0:
            color = color / 255.0
        color = np.clip(color, 0.0, 1.0)
        
        # Ensure labels are 0 or 1, then map to 36-class format for PPT compatibility
        unique_labels = np.unique(segment)
        if not all(label in [0, 1] for label in unique_labels):
            logger = get_root_logger()
            logger.warning(f"Invalid labels in {data_path}: {unique_labels}. Clipping to [0, 1]")
            segment = np.clip(segment, 0, 1)
        
        # Map binary labels to 36-class format for PPT model compatibility
        # Keep classes 0 and 1 as they are, since they map directly to our binary segmentation
        # The pretrained model expects 36 classes, but we only use 2 of them
        # Classes 2-35 will never be predicted for our PeachTree data
        
        # Ensure tree types are 0 or 1
        unique_tree_types = np.unique(tree_type)
        if not all(tree_type_val in [0, 1] for tree_type_val in unique_tree_types):
            logger = get_root_logger()
            logger.warning(f"Invalid tree types in {data_path}: {unique_tree_types}. Clipping to [0, 1]")
            tree_type = np.clip(tree_type, 0, 1)
        
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            tree_type=tree_type,
            condition="ScanNet",  # Use ScanNet condition for PPT model compatibility
        )

        
        for key in ["coord", "color", "normal"]:
            if key in data_dict:
                arr = data_dict[key]
                if np.isnan(arr).any() or np.isinf(arr).any():
                    logger = get_root_logger()
                    logger.warning(f"{key} contains NaN or Inf in {data_path}, replacing with zeros.")
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    data_dict[key] = arr
        
        return data_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
    
    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        data_dict = dict(
            fragment_list=input_dict_list, segment=segment, name=self.get_data_name(idx)
        )
        # print("keys of data_dict:", data_dict.keys())
        return data_dict


@DATASETS.register_module()
class PeachTree2Dataset(PeachTreeDataset):
    """
    Variant of PeachTreeDataset for multi-class segmentation or additional label support.
    Assumes .npy files with possible extra keys (e.g., instance, scene_id).
    """

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]

        # Load individual .npy files
        coord_path = os.path.join(data_path, "coord.npy")
        color_path = os.path.join(data_path, "color.npy")
        normal_path = os.path.join(data_path, "normal.npy")
        segment_path = os.path.join(data_path, "segment2.npy")
        tree_type_path = os.path.join(data_path, "tree_type.npy")

        # Check if files exist
        required_files = [coord_path, color_path, segment_path]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Load data
        coord = np.load(coord_path).astype(np.float32)
        color = np.load(color_path).astype(np.float32)
        segment = np.load(segment_path).astype(np.int64)
        
        # Load tree type if exists
        if os.path.exists(tree_type_path):
            tree_type = np.load(tree_type_path).astype(np.int64)
        else:
            # Default to type 0 if tree_type.npy doesn't exist
            tree_type = np.zeros(len(coord), dtype=np.int64)

        # Load normal if exists, otherwise generate dummy normals
        if os.path.exists(normal_path):
            normal = np.load(normal_path).astype(np.float32)
        else:
            normal = np.zeros_like(coord)
            normal[:, 2] = 1.0

        # Ensure all arrays have the same number of points
        n_points = coord.shape[0]
        assert color.shape[0] == n_points, f"Color shape mismatch: {color.shape[0]} vs {n_points}"
        assert normal.shape[0] == n_points, f"Normal shape mismatch: {normal.shape[0]} vs {n_points}"
        assert segment.shape[0] == n_points, f"Segment shape mismatch: {segment.shape[0]} vs {n_points}"
        assert tree_type.shape[0] == n_points, f"Tree type shape mismatch: {tree_type.shape[0]} vs {n_points}"

        # Ensure colors are in [0, 1] range
        if color.max() > 1.0:
            color = color / 255.0
        color = np.clip(color, 0.0, 1.0)

        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            tree_type=tree_type,
            condition="ScanNet",  # Use ScanNet condition for PPT model compatibility
        )

        return data_dict
