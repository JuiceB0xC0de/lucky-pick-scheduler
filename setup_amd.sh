#!/bin/bash
set -e

# AMD MI300X droplet setup — PyTorch 2.11.0 / ROCm 7.2 (Ubuntu 24.04)

VENV_DIR="/root/training-env"

# Mount NVMe scratch disk
mkdir -p /scratch
if ! mountpoint -q /scratch; then
    mkfs.ext4 -F /dev/vdc
    mount /dev/vdc /scratch
    echo "/dev/vdc /scratch ext4 defaults,nofail 0 2" >> /etc/fstab
fi
echo "[setup] scratch disk mounted at /scratch"

# Create venv
apt-get install -y python3-venv python3-full 2>/dev/null || true
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
echo "[setup] venv created at $VENV_DIR"

pip install --upgrade pip setuptools wheel

# Purge any CUDA/NVIDIA torch wheels that may have snuck in
pip uninstall -y torch torchvision torchaudio \
    nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 2>/dev/null || true

# PyTorch ROCm 7.2
pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/rocm7.2 \
    torch==2.11.0 torchvision torchaudio

# ML deps
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

# Deep Chaos Scheduler — no-deps so pip can't clobber the ROCm torch
pip install --no-deps --no-cache-dir \
    git+https://github.com/JuiceB0xC0de/deep-chaos-scheduler.git

# ROCm env vars
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1

# Persist env vars and auto-activate venv on login
cat >> /root/.bashrc << 'BASHRC'
source /root/training-env/bin/activate
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HIP_FORCE_DEV_KERNARG=1
BASHRC

echo "[setup] all dependencies installed"
echo "[setup] venv auto-activates on login (added to .bashrc)"
echo "[setup] verify PyTorch ROCm:"
python -c "import torch; print(f'  torch={torch.__version__}  hip={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
