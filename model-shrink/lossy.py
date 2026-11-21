"""Experiment 2: shrinking without quantization, at what quality cost?

Every method keeps weights in BF16-precision floats (no quantization) and
removes or restructures them instead:

  magnitude  zero the smallest |w| in each layer (unstructured)
  wanda      zero the smallest |w| * ||x|| (weight x input activation norm,
             from a small calibration set; Sun et al. 2023), no retraining
  2:4        in every group of 4 weights keep 2 -- the pattern NVIDIA tensor
             cores (Ampere+) run natively ("sparse tensor cores")
  lowrank    replace W (m x n) with A (m x r) @ B (r x n) via truncated SVD

Applied to every linear layer inside the transformer blocks (attention + MLP),
which hold ~73% of this model's parameters. Embeddings are left untouched.

Quality = perplexity on WikiText-2 test (lower is better).

    python lossy.py
"""
from __future__ import annotations

import copy
import time

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "models/qwen2.5-0.5b"
WT = "models/wikitext/wikitext-2-raw-v1/{}-00000-of-00001.parquet"
SEQ, EVAL_WINDOWS, CALIB = 512, 40, 16
torch.set_num_threads(4)


def text(split):
    return "\n\n".join(pq.read_table(WT.format(split)).column("text").to_pylist())


@torch.no_grad()
def perplexity(model, ids):
    nll, n = 0.0, 0
    for k in range(EVAL_WINDOWS):
        x = ids[:, k * SEQ:(k + 1) * SEQ]
        out = model(x, labels=x)
        nll += out.loss.item() * (x.shape[1] - 1)
        n += x.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / n)))


def block_linears(model):
    for name, m in model.model.layers.named_modules():
        if isinstance(m, torch.nn.Linear):
            yield name, m


@torch.no_grad()
def act_norms(model, ids):
    """||x_j||_2 over calibration tokens, per input feature of each linear."""
    norms, hooks = {}, []
    for name, m in block_linears(model):
        norms[name] = torch.zeros(m.in_features)

        def hook(mod, inp, out, name=name):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            norms[name] += (x * x).sum(0)
        hooks.append(m.register_forward_hook(hook))
    for k in range(CALIB):
        model(ids[:, k * SEQ:(k + 1) * SEQ])
    for h in hooks:
        h.remove()
    return {k: v.sqrt() for k, v in norms.items()}


@torch.no_grad()
def prune(model, frac, norms=None, n_m=None):
    for name, m in block_linears(model):
        W = m.weight
        score = W.abs().float()
        if norms is not None:
            score = score * norms[name][None, :]
        if n_m:  # keep n of every m along the input dimension
            n, mm = n_m
            s = score.reshape(W.shape[0], -1, mm)
            drop = s.argsort(-1)[..., : mm - n]
            mask = torch.ones_like(s, dtype=torch.bool).scatter_(-1, drop, False)
            mask = mask.reshape(W.shape)
        else:  # per output row, like Wanda
            k = int(W.shape[1] * frac)
            drop = score.argsort(1)[:, :k]
            mask = torch.ones_like(W, dtype=torch.bool).scatter_(1, drop, False)
        W.mul_(mask)


@torch.no_grad()
def lowrank(model, keep):
    """replace each block linear with a rank-r product holding `keep` of its params"""
    for name, m in list(block_linears(model)):
        W = m.weight.float()
        o, i = W.shape
        r = max(1, int(keep * o * i / (o + i)))
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        m.weight.copy_(((U[:, :r] * S[:r]) @ Vh[:r]).to(m.weight.dtype))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    test = tok(text("test"), return_tensors="pt").input_ids
    calib = tok(text("train"), return_tensors="pt").input_ids
    total = sum(p.numel() for p in base.parameters())
    inblk = sum(m.weight.numel() for _, m in block_linears(base))
    print(f"{total / 1e6:.0f}M params, {inblk / total * 100:.0f}% in block linear layers; "
          f"eval {EVAL_WINDOWS * SEQ} tokens\n")

    t = time.time()
    norms = act_norms(base, calib)
    print(f"calibration: {time.time() - t:.0f}s")

    runs = [
        ("baseline (BF16 weights)", lambda m: None, 0.0),
        ("magnitude 25%", lambda m: prune(m, 0.25), 0.25),
        ("wanda 25%", lambda m: prune(m, 0.25, norms), 0.25),
        ("magnitude 50%", lambda m: prune(m, 0.50), 0.50),
        ("wanda 50%", lambda m: prune(m, 0.50, norms), 0.50),
        ("wanda 2:4 (tensor-core sparse)", lambda m: prune(m, 0.5, norms, (2, 4)), 0.50),
        ("low-rank SVD, 75% params", lambda m: lowrank(m, 0.75), 0.25),
        ("low-rank SVD, 50% params", lambda m: lowrank(m, 0.50), 0.50),
    ]
    base_ppl = None
    print(f"{'method':34s} {'removed':>8s} {'perplexity':>11s} {'vs base':>8s}")
    for label, fn, removed in runs:
        m = copy.deepcopy(base)
        fn(m)
        ppl = perplexity(m, test)
        base_ppl = base_ppl or ppl
        print(f"{label:34s} {removed * 100:7.0f}% {ppl:11.2f} {ppl / base_ppl:7.2f}x", flush=True)
        del m


if __name__ == "__main__":
    main()
