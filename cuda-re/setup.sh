#!/usr/bin/env bash
# Installs NVIDIA's CUDA compiler/disassembler tools from PyPI into ./.venv.
# No GPU and no system CUDA install are needed.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q \
  nvidia-cuda-nvcc-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cuda-nvdisasm \
  nvidia-cuda-cuobjdump
./.venv/bin/python -m sassre.toolchain
