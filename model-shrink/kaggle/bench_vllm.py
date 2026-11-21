"""Baseline: vLLM with --cpu-offload-gb (weights that do not fit stay in CPU
RAM and are streamed each forward pass). GPU memory limited to the same
3.55 GB budget via gpu_memory_utilization. FP16, greedy, same prompt/length.

    python bench_vllm.py Qwen/Qwen2.5-7B-Instruct 1 8
"""
import sys, time

import torch
from vllm import LLM, SamplingParams

MODEL = sys.argv[1]
BATCHES = [int(b) for b in sys.argv[2:]] or [1, 8]
GEN, PROMPT = 32, "The history of the transistor begins"
BUDGET_GB = 3.55
total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
WEIGHTS_GB = 14.19

t = time.time()
llm = LLM(model=MODEL, dtype="float16", enforce_eager=True, max_model_len=512,
          gpu_memory_utilization=BUDGET_GB / total_gb,
          cpu_offload_gb=WEIGHTS_GB - 1.5,  # keep ~1.5 GB of weights on the GPU
          max_num_seqs=max(BATCHES))
print(f"vllm build {time.time() - t:.0f}s", flush=True)
sp = SamplingParams(temperature=0, max_tokens=GEN, min_tokens=GEN)
llm.generate([PROMPT], sp)  # warmup
for b in BATCHES:
    t = time.time()
    outs = llm.generate([PROMPT] * b, sp)
    dt = time.time() - t
    print(f"vllm batch={b:3d}: {GEN} steps in {dt:6.1f}s -> {dt / GEN:6.3f} s/step, "
          f"{b * GEN / dt:7.2f} tokens/s", flush=True)
print("sample:", repr(outs[0].outputs[0].text[:160]))
