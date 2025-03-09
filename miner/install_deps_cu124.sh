#!/bin/bash

# Enhanced installation script for GPU support with CUDA 12.4
# Optimized for H200 SXM 141GB VRAM

echo "Setting up GPU environment with CUDA 12.4 support for H200 SXM..."

# install uv:
wget -qO- https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv && source .venv/bin/activate \
    && uv pip install -r requirements/requirements_cu124.txt \
    && uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install torch-geometric==2.6.1 \
    && uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html \
    && uv pip install nvidia-cusparse-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-nccl-cu12 \
    && uv pip install triton numba

# Install additional CUDA optimization libraries
uv pip install cupy-cuda12x
uv pip install nvtx

# Set environment variables for optimal GPU performance
echo 'export CUDA_HOME=/usr/local/cuda' >> $HOME/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> $HOME/.bashrc
echo 'export TORCH_CUDA_ARCH_LIST="8.9"' >> $HOME/.bashrc  # For H200 architecture
echo 'export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128' >> $HOME/.bashrc

# Make the current shell use these settings right away
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export TORCH_CUDA_ARCH_LIST="8.9"  # For H200 architecture
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Check for NVIDIA driver and proper setup
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers are installed. Checking GPU..."
    nvidia-smi
else
    echo "WARNING: NVIDIA drivers not found. Make sure they're installed for GPU support."
fi

echo "GPU environment setup complete!"