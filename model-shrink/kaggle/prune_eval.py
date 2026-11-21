"""Pruning quality vs model size (GPU version of model-shrink/lossy.py).

    python prune_eval.py Qwen/Qwen2.5-1.5B
Reloads the model for each method (no deepcopy, so 7B fits).
"""
import sys, time, gc, torch
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1]
SEQ, EVAL_WINDOWS, CALIB = 512, 40, 16
DT = torch.float16  # T4 (sm_75) has fp16 tensor cores, no native bf16


def wt(split):
    p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                        repo_type="dataset")
    return "\n\n".join(pq.read_table(p).column("text").to_pylist())


def load():
    return AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=DT, device_map="auto",
        max_memory={0: "10GiB", 1: "10GiB", "cpu": "24GiB"}).eval()


def linears(model):
    for name, m in model.model.layers.named_modules():
        if isinstance(m, torch.nn.Linear):
            yield name, m


@torch.no_grad()
def ppl(model, ids):
    nll, n = 0.0, 0
    dev = model.get_input_embeddings().weight.device
    for k in range(EVAL_WINDOWS):
        x = ids[:, k * SEQ:(k + 1) * SEQ].to(dev)
        nll += model(x, labels=x).loss.float().item() * (x.shape[1] - 1)
        n += x.shape[1] - 1
    return float(torch.exp(torch.tensor(nll / n)))


@torch.no_grad()
def act_norms(model, ids):
    norms, hooks = {}, []
    dev = model.get_input_embeddings().weight.device
    for name, m in linears(model):
        norms[name] = torch.zeros(m.in_features, device=m.weight.device)

        def hook(mod, inp, out, name=name):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            norms[name] += (x * x).sum(0)
        hooks.append(m.register_forward_hook(hook))
    for k in range(CALIB):
        model(ids[:, k * SEQ:(k + 1) * SEQ].to(dev))
    for h in hooks:
        h.remove()
    return {k: v.sqrt().cpu() for k, v in norms.items()}


@torch.no_grad()
def prune(model, frac, norms=None, n_m=None):
    for name, m in linears(model):
        W = m.weight
        score = W.abs().float()
        if norms is not None:
            score = score * norms[name].to(W.device)[None, :]
        if n_m:
            n, mm = n_m
            s = score.reshape(W.shape[0], -1, mm)
            drop = s.argsort(-1)[..., : mm - n]
            mask = torch.ones_like(s, dtype=torch.bool).scatter_(-1, drop, False).reshape(W.shape)
        else:  # sort in row chunks to bound temporary GPU memory
            k = int(W.shape[1] * frac)
            mask = torch.ones_like(W, dtype=torch.bool)
            for r in range(0, W.shape[0], 1024):
                drop = score[r:r + 1024].argsort(1)[:, :k]
                mask[r:r + 1024].scatter_(1, drop, False)
                del drop
        W.mul_(mask)
        del score, mask


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    test = tok(wt("test"), return_tensors="pt").input_ids
    calib = tok(wt("train"), return_tensors="pt").input_ids
    m = load()
    total = sum(p.numel() for p in m.parameters())
    print(f"{MODEL}: {total / 1e9:.2f}B params, eval {EVAL_WINDOWS * SEQ} tokens", flush=True)
    base = ppl(m, test)
    norms = act_norms(m, calib)
    del m
    gc.collect(); torch.cuda.empty_cache()
    print(f"{'method':28s} {'removed':>8s} {'ppl':>9s} {'vs base':>8s}")
    print(f"{'baseline':28s} {0:7d}% {base:9.2f} {1:7.2f}x", flush=True)
    runs = [("magnitude 25%", 0.25, None, None), ("wanda 25%", 0.25, norms, None),
            ("magnitude 50%", 0.50, None, None), ("wanda 50%", 0.50, norms, None),
            ("wanda 2:4", 0.50, norms, (2, 4))]
    for label, frac, nrm, nm in runs:
        m = load()
        prune(m, frac, nrm, nm)
        p = ppl(m, test)
        print(f"{label:28s} {int(frac * 100):7d}% {p:9.2f} {p / base:7.2f}x", flush=True)
        del m
        gc.collect(); torch.cuda.empty_cache()
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
