#!/bin/bash
# Build the lerobot venv on the H100 cluster (run on the login node, from the repo root).
#
# The cluster's NVIDIA driver is 570 (CUDA <= 12.8), but uv.lock pins torch 2.15 from PyPI, a
# CUDA 13 build (driver >= 580). torch 2.15 has no CUDA 12 build, so after syncing the lock this
# replaces torch/torchvision with 2.14.1/0.29.1 from PyTorch's cu126 index. uv.lock is untouched;
# jobs run `uv run --no-sync` so the override survives. Re-run this after every `uv sync`.
#
# All caches and the Python install live under /mnt/data/dungnt232_1, not $HOME.

set -euo pipefail

DATA_HOME=/mnt/data/dungnt232_1
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$DATA_HOME/temp/uv_cache"
export UV_PYTHON_INSTALL_DIR="$DATA_HOME/tools/uv_python"
export XDG_CACHE_HOME="$DATA_HOME/cache"

uv sync --locked --extra fineart_vla --extra training
uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.1" "torchvision==0.29.1"

uv run --no-sync python - <<'EOF'
import torch, torchvision, torchcodec
print("torch", torch.__version__, "cuda", torch.version.cuda, "| torchvision", torchvision.__version__,
      "| torchcodec", torchcodec.__version__)
EOF
