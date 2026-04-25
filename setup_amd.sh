#!/bin/bash
set -e

# AMD MI300X droplet setup — gpu-amd-base (Ubuntu 24.04, ROCm 6.4.0)

# Mount NVMe scratch disk
mkdir -p /scratch
if ! mountpoint -q /scratch; then
    mkfs.ext4 -F /dev/nvme1n1
    mount /dev/nvme1n1 /scratch
    echo "/dev/nvme1n1 /scratch ext4 defaults,nofail 0 2" >> /etc/fstab
fi
echo "[setup] scratch disk mounted at /scratch"

# Python stack
pip install --upgrade pip setuptools wheel

# PyTorch ROCm 6.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.4

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
    tqdm

# Unsloth AMD (supports MI300X as of April 2026)
pip install unsloth

# Lucky Pick Scheduler from GitHub
pip install git+https://github.com/JuiceB0xC0de/lucky-pick-scheduler.git

echo "[setup] all dependencies installed"
echo "[setup] verify PyTorch ROCm:"
python -c "import torch; print(f'  torch={torch.__version__}  cuda/hip={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
