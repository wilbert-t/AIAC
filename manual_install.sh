#!/bin/bash

# Optimized Installation Script for 13GB Disk Space
echo "Starting installation with 13GB available space..."

# Step 1: Clean existing cache
echo "Cleaning conda and pip cache..."
conda clean --all -y
rm -rf ~/.cache/pip/*

# Step 2: Create base environment
echo "Creating base environment..."
conda create -n cuda_ml_env python=3.11 -y

# Step 3: Activate environment
conda activate cuda_ml_env

# Step 4: Install conda packages first (more efficient)
echo "Installing conda packages..."
conda install -c nvidia -c conda-forge cuda-toolkit=11.8 cudnn numpy pandas scipy \
    scikit-learn matplotlib seaborn networkx numba tqdm joblib ase jupyter -y

# Step 5: Install PyTorch with CUDA 11.8
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Step 6: Install PyTorch extensions
echo "Installing PyTorch extensions..."
pip install torch-geometric==2.4.0 pytorch-lightning==2.1.2

# Step 7: Install ML packages
echo "Installing ML packages..."
pip install xgboost==2.0.3 lightgbm==4.1.0 catboost==1.2.2

# Step 8: Install molecular processing packages
echo "Installing molecular processing packages..."
pip install SALib==1.4.7 rdkit==2024.3.1 dscribe==2.1.0 mordred==1.2.0 openbabel-wheel==3.1.1.22

# Step 9: Install visualization and analysis packages
echo "Installing visualization packages..."
pip install plotly==5.17.0 py3Dmol==2.0.4 statsmodels==0.14.1 shap==0.44.1

# Step 10: Install optimization packages
echo "Installing optimization packages..."
pip install optuna==3.4.0 hyperopt==0.2.7

# Step 11: Final cleanup
conda clean --all -y
rm -rf ~/.cache/pip/*

echo ""
echo "Installation complete! Testing setup..."
python -c "
import torch
import torch_geometric
import rdkit
import numpy as np

print('Environment Test Results:')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'PyTorch Geometric: {torch_geometric.__version__}')
print(f'RDKit: {rdkit.__version__}')
print(f'NumPy: {np.__version__}')
print('All packages loaded successfully!')
"

echo ""
echo "Disk space after installation:"
df -h