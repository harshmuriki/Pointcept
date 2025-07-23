#!/usr/bin/env python3
"""
Convert peach tree data to ScanNet format for PointTransformer V3 training.

This script processes pairs of full tree and pruned tree PLY files to create
binary segmentation labels: 0 (keep) and 1 (prune).

Additionally, it assigns tree type labels (0 or 1) based on whether the tree
is in a specified list of type 1 trees.

Output files per scene:
- coord.npy: Normalized 3D coordinates
- color.npy: RGB colors
- normal.npy: Surface normals
- segment2.npy: Binary pruning labels (0=keep, 1=prune)
- tree_type.npy: Tree type labels (0 or 1 for all points in the scene)
- metadata.json: Scene metadata including tree type
"""

import numpy as np
import open3d as o3d
import os
import argparse
from pathlib import Path
import glob
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import json
import re

# [G, M, M, G, M, M]
OUTLIERS = [23, 28, 77, 48, 42, 71]

def load_ply_file(file_path):
    """Load PLY file and return coordinates and colors"""
    print(f"Loading {file_path}...")
    pcd = o3d.io.read_point_cloud(str(file_path))
    
    coords = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    print(f"  Points: {len(coords)}")
    print(f"  Has colors: {colors is not None}")
    
    return coords, colors

def estimate_normals(coords, k=20):
    """Estimate surface normals using KNN"""
    print(f"Estimating normals with k={k}...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    
    normals = np.asarray(pcd.normals)
    return normals

def create_pruning_labels(full_coords, pruned_coords):
    """
    Create binary labels for pruning segmentation.

    Args:
        full_coords: Full tree coordinates (N, 3)
        pruned_coords: Pruned tree coordinates (M, 3), a subset of full_coords

    Returns:
        labels: Binary labels (N,) where 1 = prune, 0 = keep
    """
    print("Creating pruning labels (pruned points are a subset of full points)...")

    # Build a set of tuples for fast lookup
    pruned_set = set(map(tuple, np.round(pruned_coords, 6)))
    labels = np.array([1 if tuple(np.round(pt, 6)) in pruned_set else 0 for pt in full_coords], dtype=np.int32)

    print(f"  Points to keep: {np.sum(labels == 0)} ({100 * np.sum(labels == 0) / len(labels):.1f}%)")
    print(f"  Points to prune: {np.sum(labels == 1)} ({100 * np.sum(labels == 1) / len(labels):.1f}%)")
    print(f"  Total points: {len(labels)}")

    return labels

def normalize_coordinates(coords):
    """Normalize coordinates to a reasonable range"""
    # Center the point cloud
    center = coords.mean(axis=0)
    coords_centered = coords - center
    
    # Scale to unit cube approximately
    scale = np.max(np.abs(coords_centered))
    coords_normalized = coords_centered / scale
    
    return coords_normalized, center, scale

def process_tree_pair(full_tree_path, pruned_tree_path, output_dir, scene_name, type_0_list=None):
    """Process a pair of full and pruned tree PLY files"""
    
    # Load data
    full_coords, full_colors = load_ply_file(full_tree_path)
    pruned_coords, pruned_colors = load_ply_file(pruned_tree_path)
    
    # Normalize coordinates
    full_coords_norm, center, scale = normalize_coordinates(full_coords)
    
    # Create labels
    labels = create_pruning_labels(full_coords, pruned_coords)
    
    # Determine tree type
    tree_type = determine_tree_type(full_tree_path.name, type_0_list)
    
    # Estimate normals
    normals = estimate_normals(full_coords_norm)
    
    # Ensure colors are in [0, 1] range
    if full_colors is not None:
        if full_colors.max() > 1.0:
            full_colors = full_colors / 255.0
    else:
        # Create default colors if none exist
        full_colors = np.ones_like(full_coords_norm) * 0.5
    
    # Create output directory
    scene_dir = Path(output_dir) / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in ScanNet format
    print(f"Saving to {scene_dir}...")
    
    # Coordinates (normalized)
    np.save(scene_dir / 'coord.npy', full_coords_norm.astype(np.float32))
    
    # Colors
    np.save(scene_dir / 'color.npy', full_colors.astype(np.float32))
    
    # Normals
    np.save(scene_dir / 'normal.npy', normals.astype(np.float32))
    
    # Binary pruning labels
    np.save(scene_dir / 'segment2.npy', labels.astype(np.int32))
    
    # Tree type classification
    tree_type_array = np.full(len(full_coords_norm), tree_type, dtype=np.int32)
    np.save(scene_dir / 'tree_type.npy', tree_type_array)
    
    # Save metadata
    metadata = {
        'scene_name': scene_name,
        'num_points': int(len(full_coords_norm)),
        'num_prune_points': int(np.sum(labels == 1)),
        'num_keep_points': int(np.sum(labels == 0)),
        'tree_type': int(tree_type),
        'original_bounds': {
            'min': full_coords.min(axis=0).tolist(),
            'max': full_coords.max(axis=0).tolist()
        },
        'normalization': {
            'center': center.tolist(),
            'scale': float(scale)
        },
        'files': {
            'full_tree': str(full_tree_path),
            'pruned_tree': str(pruned_tree_path)
        }
    }
    
    with open(scene_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Successfully processed {scene_name}")
    print(f"  Output: {scene_dir}")
    print(f"  Points: {len(full_coords_norm)}")
    print(f"  Keep: {np.sum(labels == 0)}, Prune: {np.sum(labels == 1)}")
    print(f"  Tree type: {tree_type}")
    
    return scene_dir

def find_tree_pairs(data_dir):
    """Find pairs of full and pruned tree files"""
    data_path = Path(data_dir)
    
    # Look for the specific directory structure
    full_dir = data_path / 'prepruned_data_skeleton'
    pruned_dir = data_path / 'pruned_branches_filled_skeleton'
    
    if not full_dir.exists():
        print(f"Full tree directory not found: {full_dir}")
        return []
    
    if not pruned_dir.exists():
        print(f"Pruned tree directory not found: {pruned_dir}")
        return []
    
    # Get all PLY files from both directories
    full_files = sorted(list(full_dir.glob("*.ply")), key=natural_sort_key)
    pruned_files = sorted(list(pruned_dir.glob("*.ply")), key=natural_sort_key)

    print(f"Found {len(full_files)} full tree files")
    print(f"Found {len(pruned_files)} pruned tree files")
    
    # Match files by name
    pairs = []

    for i, full_file in enumerate(full_files):
        pruned_branches = pruned_files[i] if i < len(pruned_files) else None
        # Skip outliers
        tree_id = int(full_file.name.split('_')[1].split('.')[0])
        if tree_id in OUTLIERS:
            print(f"Skipping outlier index {i} tree_id {tree_id}: {full_file.name}")
            continue
        if pruned_branches and pruned_branches.exists():
            pairs.append((full_file, pruned_branches))
            print(f"  Matched: {full_file.name} <-> {pruned_branches.name}")
        else:
            print(f"  No match for: {full_file.name}")

    return pairs


def natural_sort_key(s):
    """Key function for natural sorting that handles numbers in strings correctly"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', os.path.basename(s))]


def create_train_test_val_split(scenes, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1, i=None):
    """Split scenes into train/test/val sets"""
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    np.random.seed(i)  # For reproducible splits

    n_scenes = len(scenes)
    # Ensure n_train is a multiple of 5
    n_train = int(n_scenes * train_ratio)
    n_train = (n_train // 5) * 5  # Round down to nearest multiple of 5
    n_test = int(n_scenes * test_ratio)
    n_val = n_scenes - n_train - n_test  # Remaining scenes go to val
    
    shuffled_indices = np.random.permutation(n_scenes)
    train_indices = shuffled_indices[:n_train]
    test_indices = shuffled_indices[n_train:n_train + n_test]
    val_indices = shuffled_indices[n_train + n_test:]
    
    train_scenes = [scenes[i] for i in train_indices]
    test_scenes = [scenes[i] for i in test_indices]
    val_scenes = [scenes[i] for i in val_indices]
    
    return train_scenes, test_scenes, val_scenes

def create_train_val_test_split_predetermined(scenes, split_json_path):
    """
    Split scenes into train/test/val sets based on a predetermined split JSON file.

    Args:
        scenes: List of all scene names (e.g., ['tree_0001', 'tree_0002', ...])
        split_json_path: Path to a JSON file in the dataset summary format, with 'scene_names' key.

    Returns:
        train_scenes, test_scenes, val_scenes: Lists of scene names for each split.
    """
    with open(split_json_path, 'r') as f:
        split = json.load(f)
    scene_names = split.get('scene_names', {})
    train_scenes = [scene for scene in scene_names.get('train', []) if scene in scenes]
    test_scenes = [scene for scene in scene_names.get('test', []) if scene in scenes]
    val_scenes = [scene for scene in scene_names.get('val', []) if scene in scenes]
    return train_scenes, test_scenes, val_scenes

def determine_tree_type(tree_name, type_0_list):
    """
    Determine tree type based on whether the tree name is in a specified list.
    
    Args:
        tree_name: Name of the tree (e.g., 'tree_1.ply')
        type_0_list: List of tree names that should be classified as type 0.
                     If None, all trees are classified as type 1.
        # type 0 is Guardian - the ones in the file (type0_trees.txt)
        # type 1 is MP29
    
    Returns:
        tree_type: 0 or 1 based on whether the tree is in the type_0_list
    """
    if type_0_list is None:
        return 0
    
    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(tree_name))[0]
    
    # Check if this tree is in the type 1 list
    tree_type = 0 if base_name in type_0_list else 1

    print(f"Tree {base_name} classified as type {tree_type}")
    
    return tree_type


def visualize_tree(tree_folder_path):
    coordinates = np.load(tree_folder_path / 'coord.npy')
    colors = np.load(tree_folder_path / 'color.npy')
    normals = np.load(tree_folder_path / 'normal.npy')
    labels = np.load(tree_folder_path / 'segment2.npy')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinates)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Default color
    # Color points: red for pruned (label==1), green for keep (label==0)
    color_map = np.zeros((len(labels), 3))
    color_map[labels == 1] = [1.0, 0.0, 0.0]   # Red for pruned
    color_map[labels == 0] = [0.0, 1.0, 0.0]   # Green for keep
    pcd.colors = o3d.utility.Vector3dVector(color_map)
    o3d.visualization.draw_geometries([pcd])


def main(i=0):
    parser = argparse.ArgumentParser(description='Convert peach tree data to ScanNet format')
    parser.add_argument('--data_dir', type=str, default='/home/vmuriki3/Documents/transformer/peachtree-pruning-transformers/Final_data',
                       help='Directory containing PLY files')
    parser.add_argument('--output_dir', type=str, default=f'peachtree_skeleton_{i}',
                       help='Output directory for ScanNet format data')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Ratio of data to use for testing')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of data to use for validation')
    parser.add_argument('--type_0_file', type=str, default="/home/vmuriki3/Documents/transformer/peachtree-pruning-transformers/type0_trees.txt",
                       help='File containing list of tree names (one per line) that should be classified as type 0')
    parser.add_argument('--split_json_path', type=str, default='/home/vmuriki3/Documents/transformer/peachtree-pruning-transformers/Final_data/peachtree_data/peachtreev3_5/dataset_summary.json',
                       help='Path to a JSON file containing a predetermined train/val/test split. If provided, overrides train/test/val ratios.')
    args = parser.parse_args()
    
    # Process type 0 list
    type_0_list = None
    with open(args.type_0_file, 'r') as f:
        type_0_list = [line.strip() for line in f if line.strip()]
        print("All type 0 trees: ", type_0_list)
    print(f"Loaded {len(type_0_list)} tree names from {args.type_0_file}")
    
    # Validate ratios
    if abs(args.train_ratio + args.test_ratio + args.val_ratio - 1.0) > 1e-6:
        print(f"Error: Ratios must sum to 1.0, got {args.train_ratio + args.test_ratio + args.val_ratio}")
        return
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find tree pairs
    tree_pairs = find_tree_pairs(args.data_dir)
    
    if not tree_pairs:
        print("No tree pairs found!")
        print("Please ensure you have files with 'full' and 'pruned' in their names")
        return
    
    print(f"Found {len(tree_pairs)} tree pairs to process")
    
    # Process each pair
    processed_scenes = []
    
    for _, (full_file, pruned_file) in enumerate(tqdm(tree_pairs, desc="Processing trees")):
        # Extract tree_id from the full_file name (assumes format like 'tree_23.ply')
        tree_id = int(full_file.name.split('_')[1].split('.')[0])
        scene_name = f"tree_{tree_id:04d}"
        print(f"Processing tree {tree_id} as {scene_name}, {full_file.name} and {pruned_file.name}")

        try:
            scene_dir = process_tree_pair(
                full_file, 
                pruned_file, 
                args.output_dir, 
                scene_name,
                type_0_list
            )
            processed_scenes.append(scene_name)
            
        except Exception as e:
            print(f"Error processing {full_file} and {pruned_file}: {e}")
            continue
    
    if not processed_scenes:
        print("No scenes were successfully processed!")
        return
    
    if args.split_json_path:
        print("%" * 60)
        print(f"Using predetermined split from {args.split_json_path}")
        train_scenes, test_scenes, val_scenes = create_train_val_test_split_predetermined(
            processed_scenes,
            split_json_path=args.split_json_path
        )
    else:
        print(f"Creating train/test/val split with ratios: {args.train_ratio}, {args.test_ratio}, {args.val_ratio}")
        # Create train/test/val split
        train_scenes, test_scenes, val_scenes = create_train_test_val_split(
            processed_scenes, 
            args.train_ratio, 
            args.test_ratio, 
            args.val_ratio,
            i=i
        )

    print(f"\nCreating train/test/val split:")
    print(f"  Train scenes: {len(train_scenes)}")
    print(f"  Test scenes: {len(test_scenes)}")
    print(f"  Val scenes: {len(val_scenes)}")
    
    # Create split directories and move scenes
    train_dir = output_path / 'train'
    test_dir = output_path / 'test'
    val_dir = output_path / 'val'
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Move scenes to appropriate splits
    for scene in train_scenes:
        src = output_path / scene
        dst = train_dir / scene
        if src.exists():
            src.rename(dst)
    
    for scene in test_scenes:
        src = output_path / scene
        dst = test_dir / scene
        if src.exists():
            src.rename(dst)
    
    for scene in val_scenes:
        src = output_path / scene
        dst = val_dir / scene
        if src.exists():
            src.rename(dst)
    
    print(f"\nConversion complete!")
    print(f"Output directory: {output_path}")
    print(f"Train scenes: {len(train_scenes)}")
    print(f"Test scenes: {len(test_scenes)}")
    print(f"Val scenes: {len(val_scenes)}")
    
    # Create a summary file
    # Count tree types
    type_0_count = 0
    type_1_count = 0
    
    for scene in processed_scenes:
        # Try to read metadata from each split
        metadata_path = None
        for split_dir in [train_dir, test_dir, val_dir]:
            potential_path = split_dir / scene / 'metadata.json'
            if potential_path.exists():
                metadata_path = potential_path
                break
        
        if metadata_path:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    tree_type = metadata.get('tree_type', 0)
                    if tree_type == 1:
                        type_1_count += 1
                    else:
                        type_0_count += 1
            except:
                type_0_count += 1  # Default to type 0 if can't read metadata

    summary = {
        'total_scenes': len(processed_scenes),
        'train_scenes': len(train_scenes),
        'test_scenes': len(test_scenes),
        'val_scenes': len(val_scenes),
        'train_ratio': args.train_ratio,
        'test_ratio': args.test_ratio,
        'val_ratio': args.val_ratio,
        'tree_types': {
            'type_0_count': type_0_count,
            'type_1_count': type_1_count,
            'type_0_list': type_0_list if type_0_list else []
        },
        'scene_names': {
            'train': train_scenes,
            'test': test_scenes,
            'val': val_scenes
        }
    }
    
    with open(output_path / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDataset summary saved to {output_path / 'dataset_summary.json'}")
    print(f"Tree type distribution:")
    print(f"  Type 0: {type_0_count} trees")
    print(f"  Type 1: {type_1_count} trees")

if __name__ == "__main__":
    main()
    # visualize_tree(tree_folder_path=Path("/home/vmuriki3/Documents/transformer/peachtree-pruning-transformers/Pointcept/tools/peachtree_skeleton_0/tree_0001"))