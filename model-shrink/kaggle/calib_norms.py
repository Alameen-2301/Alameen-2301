"""Collect Wanda calibration statistics once and save them.

For every linear layer inside the decoder blocks: ||x_j||_2 of each input
feature over 16 x 512 WikiText-2 train tokens. Uses both GPUs (the full model
must run once). offload_bench.py sparseNN then prunes with |w| * ||x||.

    python calib_norms.py Qwen/Qwen2.5-7B   -> norms_Qwen2.5-7B.pt
"""
import sys

import torch
from transformers import AutoTokenizer

import prune_eval as pe

if __name__ == "__main__":
    pe.MODEL = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(pe.MODEL)
    calib = tok(pe.wt("train"), return_tensors="pt").input_ids
    model = pe.load()
    norms = pe.act_norms(model, calib)
    out = f"norms_{pe.MODEL.split('/')[-1]}.pt"
    torch.save(norms, out)
    print(f"saved {len(norms)} layer norms -> {out}")
