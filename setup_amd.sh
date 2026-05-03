#!/bin/bash
set -e

# AMD MI300X droplet setup — PyTorch 2.11.0 / ROCm 7.2 (Ubuntu 24.04)

# Mount NVMe scratch disk
mkdir -p /scratch
if ! mountpoint -q /scratch; then
    mkfs.ext4 -F /dev/vdc
    mount /dev/vdc /scratch
    echo "/dev/vdc /scratch ext4 defaults,nofail 0 2" >> /etc/fstab
fi
echo "[setup] scratch disk mounted at /scratch"

# Python stack
pip install --upgrade pip setuptools wheel

# Purge any CUDA/NVIDIA torch wheels that may have snuck in
pip uninstall -y torch torchvision torchaudio \
    nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 2>/dev/null || true

# PyTorch ROCm 7.2
pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/rocm7.2 \
    torch==2.11.0 torchvision torchaudio

# ML deps (no-deps on anything that might drag in a CUDA torch)
pip install \
    transformers>=4.51.0 \
    accelerate \
    trl \
    datasets \
    sentencepiece \
    protobuf \
    hf_transfer \
    scipy \
    numpy \
    tqdm \
    wandb \
    peft

# Deep Chaos Scheduler — install without deps so pip can't clobber the ROCm torch
pip install --no-deps --no-cache-dir \
    git+https://github.com/JuiceB0xC0de/deep-chaos-scheduler.git

# ROCm env vars — set here so they're available for the rest of the session
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1

echo "[setup] all dependencies installed"
echo "[setup] ROCm env vars set (TORCH_BLAS_PREFER_HIPBLASLT=1, HIP_FORCE_DEV_KERNARG=1)"
echo "[setup] verify PyTorch ROCm:"
python -c "import torch; print(f'  torch={torch.__version__}  hip={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
