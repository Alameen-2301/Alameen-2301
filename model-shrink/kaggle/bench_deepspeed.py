"""Baseline: DeepSpeed ZeRO-Inference (ZeRO stage 3 with parameters offloaded
to pinned CPU RAM and prefetched layer by layer). Same model, prompt, length
and batch sizes as offload_bench.py.

    python bench_deepspeed.py Qwen/Qwen2.5-7B-Instruct 1 8
"""
import os, sys, time

import torch
import deepspeed
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import HfDeepSpeedConfig

MODEL = sys.argv[1]
BATCHES = [int(b) for b in sys.argv[2:]] or [1, 8]
GEN, PROMPT = 32, "The history of the transistor begins"
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29511")
torch.cuda.set_device(0)

cfg = AutoConfig.from_pretrained(MODEL)
ds_config = {
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_param": {"device": "cpu", "pin_memory": True},
        "overlap_comm": True,
        "stage3_prefetch_bucket_size": 2 * cfg.hidden_size * cfg.hidden_size,
        "stage3_param_persistence_threshold": 10 * cfg.hidden_size,
    },
    "train_micro_batch_size_per_gpu": 1,
}
dschf = HfDeepSpeedConfig(ds_config)  # must exist before from_pretrained (zero.Init)
tok = AutoTokenizer.from_pretrained(MODEL)
t = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
engine.module.eval()
print(f"deepspeed build {time.time() - t:.0f}s, GPU alloc {torch.cuda.memory_allocated() / 2**30:.2f} GB",
      flush=True)


@torch.no_grad()
def run(b):
    ids = tok([PROMPT] * b, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    t = time.time()
    out = engine.module.generate(**ids, max_new_tokens=GEN, min_new_tokens=GEN, do_sample=False,
                                 synced_gpus=False)
    torch.cuda.synchronize()
    return time.time() - t, out


run(1)
for b in BATCHES:
    torch.cuda.reset_peak_memory_stats()
    dt, out = run(b)
    print(f"deepspeed batch={b:3d}: {GEN} steps in {dt:6.1f}s -> {dt / GEN:6.3f} s/step, "
          f"{b * GEN / dt:7.2f} tokens/s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB",
          flush=True)
print("sample:", repr(tok.decode(out[0][-GEN:])[:160]))
