#!/bin/bash
# Bash script to set up a virtual environment and install PointNet++ dependencies with CUDA 12.6 support

# 1. Create and activate a Python virtual environment
# python -m venv pointnet2
# source pointnet2/bin/activate

# 2. Install PyTorch 2.6.0, torchvision 0.21.0, torchaudio 2.6.0 for CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# 3. Install torch_geometric (PyTorch Geometric)
pip install torch_geometric

# 4. Install torch-cluster for PyTorch 2.6.0 + CUDA 12.6
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

# 5. Install Open3D (for point cloud processing)
pip install open3d

# 6. Install torchmetrics (for evaluation metrics)
pip install torchmetrics

pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
pip install spconv-cu120

cd Pointcept/libs/pointops
python setup.py install

echo "Setup complete. To activate your environment again later, run: source pointnet2/bin/activate"
