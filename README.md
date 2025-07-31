# Peachtree Pruning Transformers

<p align="center">
    <img alt="pointcept" src="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/logo.png" width="400">
</p>

[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/pointcept/pointcept/blob/main/LICENSE)

**Peachtree Pruning Transformers** is a specialized point cloud semantic segmentation system for identifying optimal pruning locations in peach tree point clouds. Built on the **Pointcept** framework, this implementation focuses on agricultural automation applications using state-of-the-art Point Transformer architectures.

## Key Features

- **Peachtree Dataset Support**: Custom dataset loader for peach tree point cloud data with pruning point annotations
- **Volume Analysis**: Calculate and compare volumes of ground truth vs predicted pruning regions using multiple methods (voxel, cylinder, hull)
- **Statistical Metrics**: Comprehensive evaluation including MAE, RMSE, correlation analysis, and summary statistics
- **3D Visualization**: Point cloud visualization with offscreen rendering support for headless environments
- **Batch Processing**: Automated processing and analysis of multiple tree samples with tree ID tracking
- **Data Export**: CSV export functionality for volume comparison data and statistical analysis

## Supported Models

**Backbones:**
- [Point Transformer V3](https://arxiv.org/abs/2312.10035) (CVPR 2024) - Primary model for peachtree segmentation
- [Point Transformer V2](https://arxiv.org/abs/2210.05666) (NeurIPS 2022) 
- [Point Transformer V1](https://arxiv.org/abs/2012.09164) (ICCV 2021)
- [SparseUNet](https://arxiv.org/abs/1904.08755) - Baseline sparse convolution model

**Datasets:**
- **Peachtree Dataset**: Custom dataset for pruning point identification with volume annotations
- [ScanNet](http://www.scan-net.org/): Used for pre-training and transfer learning

## Peachtree Dataset Structure

The peachtree dataset should be organized as follows:
```
Final_data/
├── peachtree_data/           # Input point clouds
│   ├── train/
│   │   ├── tree_001.ply
│   │   ├── tree_002.ply
│   │   └── ...
│   ├── val/
│   └── test/
├── peachtree_data_filled/    # Ground truth with pruning annotations  
│   ├── train/
│   ├── val/
│   └── test/
```

Each `.ply` file should contain:
- **xyz coordinates**: 3D point positions (required)
- **labels**: Semantic labels (0: trunk/branch, 1: pruning point)
- **tree_id**: Unique identifier for tracking individual trees

## Overview

- [Installation](#installation)
- [Peachtree Dataset Preparation](#peachtree-dataset-preparation) 
- [Training](#training)
- [Testing and Evaluation](#testing-and-evaluation)
- [Volume Analysis](#volume-analysis)
- [Visualization](#visualization)
- [Model Configurations](#model-configurations)

## Installation

### Requirements
- Ubuntu: 18.04 and above
- CUDA: 11.3 and above
- PyTorch: 1.10.0 and above

### Environment Setup
Create a conda environment with the required dependencies:

```bash
conda create -n pointcept python=3.10 -y
conda activate pointcept

# Install PyTorch with CUDA support
conda install pytorch==2.5.0 torchvision==0.13.1 torchaudio==0.20.0 pytorch-cuda=12.4 -c pytorch -y

# Install core dependencies
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# Install spconv for SparseUNet (optional)
pip install spconv-cu124

# Install Open3D for visualization
pip install open3d

# Build pointops for Point Transformer models
cd Pointcept/libs/pointops
python setup.py install
cd ../../..
```

## Peachtree Dataset Preparation

### Dataset Structure
Place your peachtree dataset in the following structure:
```
Final_data/
├── peachtree_data/           # Input point clouds
│   ├── train/
│   │   ├── tree_001.ply
│   │   ├── tree_002.ply
│   │   └── ...
│   ├── val/
│   └── test/
├── peachtree_data_filled/    # Ground truth with pruning annotations  
│   ├── train/
│   ├── val/
│   └── test/
```

### Link Dataset to Codebase
```bash
mkdir Pointcept/data
ln -s /path/to/Final_data/peachtree_data Pointcept/data/peachtree
ln -s /path/to/Final_data/peachtree_data_filled Pointcept/data/peachtree_filled
```

### Data Format
Each `.ply` file should contain:
- **xyz coordinates**: 3D point positions (required)
- **labels**: Semantic labels (0: trunk/branch, 1: pruning point)
- **tree_id**: Unique identifier for tracking individual trees

## Training

### Train Point Transformer V3 on Peachtree Dataset
```bash
cd Pointcept
export PYTHONPATH=./

# Point Transformer V3 (recommended)
sh scripts/train.sh -g 2 -d peachtree -c semseg-pt-v3m1-0-peachtree -n semseg-pt-v3m1-0-peachtree

# Point Transformer V2
sh scripts/train.sh -g 2 -d peachtree -c semseg-pt-v2m2-0-peachtree -n semseg-pt-v2m2-0-peachtree

# SparseUNet baseline
sh scripts/train.sh -g 2 -d peachtree -c semseg-spunet-v1m1-0-peachtree -n semseg-spunet-v1m1-0-peachtree
```

### Transfer Learning from ScanNet
```bash
# Pre-train on ScanNet, fine-tune on peachtree
sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-scannet-pretrain
sh scripts/train.sh -g 2 -d peachtree -w exp/scannet/semseg-pt-v3m1-0-scannet-pretrain/model/model_best.pth -c semseg-pt-v3m1-0-peachtree-ft -n semseg-pt-v3m1-0-peachtree-ft
```

## Testing and Evaluation

### Basic Testing
```bash
# Test trained model
cd Pointcept
sh scripts/test.sh -d peachtree -n semseg-pt-v3m1-0-peachtree -w model_best
```

### Volume Analysis
Calculate volume metrics for ground truth vs predicted pruning regions:
```bash
cd /path/to/workspace
python calculate_volume.py
```

This will generate:
- Volume comparison plots with embedded statistics (MAE, RMSE, correlation)
- CSV files with detailed volume data for each tree
- Statistical summaries and correlation analysis

### Generate Visualizations
Create labeled comparison images:
```bash
python save_screenshots.py
```

Generate 3D visualizations with tree IDs:
```bash
python visualize_cylinder_tree.py
```

### Metrics and Evaluation
Calculate precision, recall, and F1 scores:
```bash
python metrics.py
```

## Volume Analysis

The volume analysis system provides comprehensive comparison between ground truth and predicted pruning regions:

### Volume Calculation Methods
- **Voxel Volume**: Grid-based volume calculation using voxelization
- **Cylinder Volume**: Cylinder-based approximation for tree structure analysis
- **Hull Volume**: Convex hull volume calculation

### Statistical Analysis
- **MAE (Mean Absolute Error)**: Average absolute difference between volumes
- **RMSE (Root Mean Square Error)**: Root mean square error between volumes
- **Correlation**: Pearson correlation coefficient between GT and predicted volumes
- **Summary Statistics**: Mean, standard deviation, min/max values

### CSV Export
All volume data is automatically exported to CSV files for further analysis:
- Individual tree volume comparisons
- Statistical summaries
- Correlation matrices

## Model Configurations

### Available Configs
- `semseg-pt-v3m1-0-peachtree.py`: Point Transformer V3 for peachtree dataset
- `semseg-pt-v2m2-0-peachtree.py`: Point Transformer V2 for peachtree dataset  
- `semseg-spunet-v1m1-0-peachtree.py`: SparseUNet baseline for peachtree dataset

### Key Configuration Options
```python
# Example config for peachtree dataset
data = dict(
    num_classes=2,                    # trunk/branch vs pruning point
    ignore_index=-1,
    names=["trunk_branch", "pruning_point"]
)

model = dict(
    in_channels=3,                    # xyz coordinates only (no colors)
    num_classes=2,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
    )
)
```

### No-Color Training
Models are configured to use only xyz coordinates (no color information):
```python
# Dataset configuration removes color loading
# Model in_channels=3 for xyz coordinates only
```

## Visualization

### Screenshots and Comparisons
The visualization system generates:
- Three-view comparisons (ground truth, predicted, overlay)
- Labeled images with tree IDs
- Volume statistics embedded in plots
- Matplotlib-based fallback for headless environments

### 3D Visualization
- Open3D-based point cloud rendering
- Offscreen rendering support for SSH/headless environments
- Color-coded segmentation results
- Interactive visualization when display is available

## File Structure

```
peachtree-pruning-transformers/
├── Pointcept/                     # Main framework directory
│   ├── configs/                   # Model configurations
│   ├── pointcept/                 # Core framework code
│   ├── tools/                     # Training and testing scripts
│   └── scripts/                   # Shell scripts for training/testing
├── Final_data/                    # Dataset directory
│   ├── peachtree_data/           # Input point clouds
│   └── peachtree_data_filled/    # Ground truth annotations
├── calculate_volume.py            # Volume analysis and statistics
├── save_screenshots.py           # Visualization and screenshot generation
├── metrics.py                     # Evaluation metrics calculation
├── visualize_cylinder_tree.py    # 3D visualization
├── clean_dataset.py              # Dataset preprocessing utilities
└── README_peachtree.md           # This documentation
```

## Troubleshooting

### Common Issues

1. **PyTorch/torch-scatter compatibility**:
   ```bash
   pip uninstall torch-scatter
   pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
   ```

2. **Open3D segmentation faults**:
   - Use offscreen rendering mode
   - Set `DISPLAY` environment variable appropriately

3. **Matplotlib display errors**:
   - Configure Agg backend for headless environments
   - Use `export MPLBACKEND=Agg` before running scripts

## Acknowledgement

This peachtree pruning implementation is built on the **Pointcept** framework. We acknowledge the original Pointcept team and the following key contributions:

- **Pointcept Framework**: The base codebase for point cloud perception research
- **Point Transformer V3**: The primary backbone architecture used for peachtree segmentation  
- **Original Pointcept Contributors**: [Xiaoyang Wu](https://xywu.me/), [Yixing Lao](https://github.com/yxlao), [Hengshuang Zhao](https://hszhao.github.io/)

### Citation

If you use this peachtree implementation in your research, please cite the original Pointcept work:

```bibtex
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished = {\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

For Point Transformer V3 (the recommended backbone):

```bibtex
@article{wu2024point,
    title={Point Transformer V3: Simpler, Faster, Stronger},
    author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
    journal={arXiv preprint arXiv:2312.10035},
    year={2024}
}
```
