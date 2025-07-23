import os
import glob
import re
import shutil
from pathlib import Path

def extract_tree_id(filename):
    """
    Extract tree ID from various filename formats
    
    Args:
        filename (str): The filename to extract ID from
        
    Returns:
        int or None: Extracted tree ID, or None if not found
    """
    # Common patterns for tree ID extraction
    patterns = [
        r'tree[_\-\s]*(\d+)',           # tree_123, tree-123, tree 123, tree123
        r't(\d+)',                      # t123
        r'(\d+)\.ply$',                 # 123.ply
        r'(\d+)$',                      # Just numbers at the end
        r'tree(\d+)',                   # tree123
        r'Tree[_\-\s]*(\d+)',          # Tree_123, Tree-123, Tree 123
    ]
    
    filename_lower = filename.lower()
    
    for pattern in patterns:
        match = re.search(pattern, filename_lower)
        if match:
            return int(match.group(1))
    
    return None

def rename_files_to_tree_id(source_dir, target_dir=None, file_extension='.ply', dry_run=False):
    """
    Rename files in a directory to tree_id.ply format
    
    Args:
        source_dir (str): Directory containing files to rename
        target_dir (str, optional): Target directory. If None, renames in place
        file_extension (str): File extension to look for (default: '.ply')
        dry_run (bool): If True, only print what would be done without actually renaming
    """
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist!")
        return
    
    # If target_dir is provided, create it if it doesn't exist
    if target_dir:
        target_path = Path(target_dir)
        if not dry_run:
            target_path.mkdir(parents=True, exist_ok=True)
    else:
        target_path = source_path
    
    # Find all files with the specified extension
    pattern = f"*{file_extension}"
    files = sorted(list(source_path.glob(pattern)))

    if not files:
        print(f"No files with extension '{file_extension}' found in '{source_dir}'")
        return
    
    print(f"Found {len(files)} files to process...")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_path}")
    print(f"Dry run: {dry_run}")
    print("-" * 50)
    
    renamed_count = 0
    failed_count = 0
    
    for file_path in sorted(files):
        filename = file_path.name
        tree_id = extract_tree_id(filename)
        print(f"Processing file: {filename} (tree_id: {tree_id})")
        
        if tree_id is not None:
            new_filename = f"tree_{tree_id}{file_extension}"
            new_file_path = target_path / new_filename
            
            # Check if target file already exists
            if new_file_path.exists() and new_file_path != file_path:
                print(f"⚠️  SKIP: {filename} -> {new_filename} (target already exists)")
                failed_count += 1
                continue
            
            if dry_run:
                print(f"✓  WOULD RENAME: {filename} -> {new_filename}")
            else:
                try:
                    if target_dir and target_path != source_path:
                        # Copy to new directory with new name
                        shutil.copy2(file_path, new_file_path)
                        print(f"✓  COPIED: {filename} -> {new_filename}")
                    else:
                        # Rename in place
                        file_path.rename(new_file_path)
                        print(f"✓  RENAMED: {filename} -> {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"❌ ERROR: Failed to rename {filename}: {str(e)}")
                    failed_count += 1
        else:
            print(f"❌ SKIP: Could not extract tree ID from '{filename}'")
            failed_count += 1
    
    print("-" * 50)
    print(f"Summary:")
    print(f"  Successfully processed: {renamed_count}")
    print(f"  Failed/Skipped: {failed_count}")
    print(f"  Total files: {len(files)}")

def batch_rename_directories(parent_dir, subdirs=None, file_extension='.ply', dry_run=False):
    """
    Rename files in multiple subdirectories
    
    Args:
        parent_dir (str): Parent directory containing subdirectories
        subdirs (list, optional): List of subdirectory names. If None, processes all subdirs
        file_extension (str): File extension to look for
        dry_run (bool): If True, only print what would be done
    """
    parent_path = Path(parent_dir)
    
    if not parent_path.exists():
        print(f"Error: Parent directory '{parent_dir}' does not exist!")
        return
    
    if subdirs is None:
        # Get all subdirectories
        subdirs = [d.name for d in parent_path.iterdir() if d.is_dir()]
    
    print(f"Processing {len(subdirs)} directories...")
    print("=" * 60)
    
    for subdir in subdirs:
        subdir_path = parent_path / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            print(f"\nProcessing directory: {subdir}")
            rename_files_to_tree_id(str(subdir_path), file_extension=file_extension, dry_run=dry_run)
        else:
            print(f"⚠️  Directory not found: {subdir}")

def main():
    """
    Main function with example usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Rename files to tree_id.ply format')
    parser.add_argument('source_dir', help='Source directory containing files to rename')
    parser.add_argument('--target-dir', help='Target directory (optional, renames in place if not provided)')
    parser.add_argument('--extension', default='.ply', help='File extension to process (default: .ply)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually renaming')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_rename_directories(
            args.source_dir, 
            file_extension=args.extension, 
            dry_run=args.dry_run
        )
    else:
        rename_files_to_tree_id(
            args.source_dir, 
            target_dir=args.target_dir,
            file_extension=args.extension, 
            dry_run=args.dry_run
        )

    
if __name__ == "__main__":
    rename_files_to_tree_id(
        source_dir="/home/vmuriki3/Documents/transformer/peachtree-pruning-transformers/Final_data/pruned_branches_filled_skeleton",
        dry_run=False
    )
