#!/usr/bin/env bash
# Baseline: llama.cpp (the engine under LM Studio / Ollama) running the
# UNQUANTIZED F16 GGUF of Qwen2.5-7B-Instruct with only some layers on the GPU.
# Sweeps -ngl and records VRAM so the run matching our 3.55 GB budget can be
# picked. Single sequence decode speed (tg) and prompt speed (pp) via llama-bench.
#
#   bash llamacpp_bench.sh            (from /kaggle/working)
set -euo pipefail
cd /kaggle/working

if [ ! -x llama.cpp/build/bin/llama-bench ]; then
  [ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
  rm -rf llama.cpp/build
  # Kaggle ships the driver only as libcuda.so.1 (or a stub), which CMake's
  # FindCUDAToolkit does not find on its own -> point it at the library.
  LIBCUDA=$(ls /usr/local/cuda/lib64/stubs/libcuda.so \
               /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so \
               /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
               /usr/local/nvidia/lib64/libcuda.so.1 2>/dev/null | head -1 || true)
  echo "libcuda: $LIBCUDA"
  cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
        -DCUDA_cuda_driver_LIBRARY="$LIBCUDA" -DGGML_CCACHE=OFF \
        -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > build.log 2>&1 \
    || { tail -30 build.log; exit 1; }
  cmake --build llama.cpp/build -j4 --target llama-bench llama-cli >> build.log 2>&1
fi
llama.cpp/build/bin/llama-bench --version 2>&1 | head -2 || true

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B-Instruct-GGUF", allow_patterns=["*fp16*"], local_dir="gguf")
EOF
M=$(ls gguf/*fp16*00001-of-*.gguf 2>/dev/null || ls gguf/*fp16*.gguf | head -1)
echo "model: $M"

for NGL in 0 2 4 6 8; do
  ( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0; done ) > vram_$NGL.txt &
  MON=$!
  CUDA_VISIBLE_DEVICES=0 llama.cpp/build/bin/llama-bench -m "$M" -ngl $NGL -p 16 -n 64 -t 4 -r 2 -o md 2>/dev/null | grep -E "qwen|model" || true
  kill $MON
  echo "ngl=$NGL peak VRAM: $(sort -n vram_$NGL.txt | tail -1) MiB"
done
