"""Run a model whose weights do not fit in the GPU budget by streaming
decoder layers from pinned CPU RAM, with prefetch overlap and optional
lossless compression (decoded on the GPU).

    python offload_bench.py MODEL BUDGET_FRAC MODE [BATCH ...]
      MODE: raw   stream fp16 weights (2 bytes/weight over PCIe)
            comp  stream lossless 12-bit format (exponent code + sign/mantissa)
            accel Hugging Face accelerate offload (device_map + max_memory)
"""
import sys, time, gc, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL, FRAC, MODE = sys.argv[1], float(sys.argv[2]), sys.argv[3]
BATCHES = [int(b) for b in sys.argv[4:]] or [1, 16]
DEV = torch.device("cuda:0")
GEN, PROMPT = 32, "The history of the transistor begins"
GB = 1 << 30


# ---------------------------------------------------------------- lossless codec
CHUNK = 8 << 20


def compress(t: torch.Tensor):
    """bf16 tensor -> (sm uint8, packed 4-bit codes, lut, esc idx, esc val).
    Exact: decompress(compress(t)) has identical bits."""
    u = t.contiguous().view(torch.int16).flatten()
    exp = ((u >> 7) & 0xFF).to(torch.uint8)
    sm = (((u >> 8) & 0x80) | (u & 0x7F)).to(torch.uint8)
    cnt = torch.bincount(exp.long(), minlength=256)
    top = cnt.argsort(descending=True)[:15]
    code_of = torch.full((256,), 15, dtype=torch.uint8)
    code_of[top] = torch.arange(15, dtype=torch.uint8)
    codes = code_of[exp.long()]
    esc = (codes == 15).nonzero().flatten()
    if codes.numel() % 2:
        codes = torch.cat([codes, codes.new_zeros(1)])
    packed = codes[0::2] | (codes[1::2] << 4)
    lut = torch.zeros(16, dtype=torch.int16)
    lut[:15] = top.to(torch.int16)
    return dict(sm=sm, packed=packed, lut=lut, esc_i=esc.to(torch.int32),
                esc_v=exp[esc].to(torch.int16), n=u.numel(), shape=tuple(t.shape))


import triton
import triton.language as tl


@triton.jit
def _decode_kernel(packed, sm, lut, out, n, BLOCK: tl.constexpr):
    """One pass: 4-bit exponent code + sign/mantissa byte -> fp16 weight."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    pk = tl.load(packed + offs // 2, mask=m, other=0).to(tl.int32)
    code = tl.where(offs % 2 == 0, pk & 15, pk >> 4)
    exp = tl.load(lut + code, mask=m, other=0).to(tl.int32)
    s = tl.load(sm + offs, mask=m, other=0).to(tl.int32)
    bits = ((s & 0x80) << 8) | (exp << 7) | (s & 0x7F)          # bf16 bit pattern
    f32 = (bits << 16).to(tl.float32, bitcast=True)              # bf16 -> fp32 is a shift
    tl.store(out + offs, f32.to(tl.float16), mask=m)


def decompress_into(c, out: torch.Tensor):
    n = c["n"]
    flat = out.view(-1)
    BLOCK = 2048
    _decode_kernel[(triton.cdiv(n, BLOCK),)](c["packed"], c["sm"], c["lut"], flat, n, BLOCK=BLOCK)
    if c["esc_i"].numel():  # rare exponents: patch exactly
        i = c["esc_i"].long()
        smv = c["sm"][i].to(torch.int16)
        bits = (c["esc_v"] << 7) | (smv & 0x7F)
        bits = torch.where(smv >= 128, bits | -32768, bits)
        flat[i] = bits.view(torch.bfloat16).to(torch.float16)


def decompress_into_torch(c, out: torch.Tensor):
    """GPU decode into fp16 `out` (chunked to bound temporary memory)."""
    n = c["n"]
    flat = out.view(-1)
    lut = c["lut"]
    for s in range(0, n, CHUNK):
        e_ = min(n, s + CHUNK)
        pk = c["packed"][s // 2:(e_ + 1) // 2]
        codes = torch.stack([pk & 15, pk >> 4], 1).flatten()[: e_ - s]
        exp = lut[codes.long()]
        sm = c["sm"][s:e_].to(torch.int16)
        bits = (exp << 7) | (sm & 0x7F)
        bits = torch.where(sm >= 128, bits | -32768, bits)
        flat[s:e_] = bits.view(torch.bfloat16).to(torch.float16)
    if c["esc_i"].numel():  # rare exponents: patch exactly
        i = c["esc_i"].long()
        sm = c["sm"][i].to(torch.int16)
        bits = (c["esc_v"] << 7) | (sm & 0x7F)
        bits = torch.where(sm >= 128, bits | -32768, bits)
        flat[i] = bits.view(torch.bfloat16).to(torch.float16)


def to_dev(c, stream):
    return {k: (v.to(DEV, non_blocking=True) if torch.is_tensor(v) else v) for k, v in c.items()}


# ---------------------------------------------------------------- streaming engine
class CpuEmbedding(torch.nn.Module):
    """Embedding table stays in CPU RAM; only the looked-up rows go to the GPU."""

    def __init__(self, w):
        super().__init__()
        self.w = w.to(torch.float16)

    def forward(self, ids):
        return self.w[ids.cpu()].to(DEV, non_blocking=True)


def _align(x, a=16):
    return (x + a - 1) // a * a


class Streamer:
    """Streams decoder layers through two GPU slots. Each layer is ONE
    contiguous pinned host buffer -> one H2D copy (+ one decode kernel in
    comp mode) per layer, overlapped with compute of the previous layer."""

    def __init__(self, layers, streamed, mode):
        self.layers, self.ids, self.mode = layers, streamed, mode
        self.copy = torch.cuda.Stream(DEV)
        L0 = layers[streamed[0]]
        self.names = [n for n, _ in L0.named_parameters()]
        self.shapes = [tuple(p.shape) for _, p in L0.named_parameters()]
        self.offs, o = [], 0
        for sh in self.shapes:
            self.offs.append(o)
            o += int(torch.tensor(sh).prod()) if sh else 1
        self.n = o
        self.host, self.meta, self.luts = {}, {}, {}
        for li in streamed:
            flat = torch.cat([p.data.flatten() for _, p in layers[li].named_parameters()])
            if mode == "comp":
                c = compress(flat)
                n_sm, n_pk = c["n"], c["packed"].numel()
                n_esc = c["esc_i"].numel()
                o_pk = _align(n_sm)
                o_ei = _align(o_pk + n_pk)
                o_ev = _align(o_ei + 4 * n_esc)
                size = _align(o_ev + 2 * n_esc)
                blob = torch.zeros(size, dtype=torch.uint8).pin_memory()
                blob[:n_sm] = c["sm"]
                blob[o_pk:o_pk + n_pk] = c["packed"]
                blob[o_ei:o_ei + 4 * n_esc] = c["esc_i"].view(torch.uint8)
                blob[o_ev:o_ev + 2 * n_esc] = c["esc_v"].view(torch.uint8)
                self.host[li] = blob
                self.meta[li] = (o_pk, n_pk, o_ei, o_ev, n_esc)
                self.luts[li] = c["lut"].to(DEV)
            else:
                self.host[li] = flat.to(torch.float16).pin_memory()
            for _, p in layers[li].named_parameters():
                p.data = torch.empty(0, dtype=torch.float16)
            del flat
            gc.collect()
        self.flat = [torch.empty(self.n, dtype=torch.float16, device=DEV) for _ in range(2)]
        if mode == "comp":
            mx = max(b.numel() for b in self.host.values())
            self.stage = [torch.empty(mx, dtype=torch.uint8, device=DEV) for _ in range(2)]
        self.views = [[f[o:o + (int(torch.tensor(sh).prod()) if sh else 1)].view(sh)
                       for o, sh in zip(self.offs, self.shapes)] for f in self.flat]
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.slot_of, self.pos = {}, {li: k for k, li in enumerate(streamed)}
        self.bytes_per_layer = sum(h.numel() * h.element_size() for h in self.host.values()) / len(streamed)
        for li in streamed:
            layers[li].register_forward_pre_hook(self.pre)
            layers[li].register_forward_hook(self.post)
            layers[li]._sidx = li
        self.prefetch(streamed[0], 0)
        if len(streamed) > 1:
            self.prefetch(streamed[1], 1)

    def prefetch(self, li, slot):
        self.copy.wait_stream(torch.cuda.current_stream(DEV))  # slot is free
        with torch.cuda.stream(self.copy):
            if self.mode == "comp":
                blob = self.host[li]
                st = self.stage[slot][:blob.numel()]
                st.copy_(blob, non_blocking=True)
                o_pk, n_pk, o_ei, o_ev, n_esc = self.meta[li]
                out = self.flat[slot]
                BLOCK = 2048
                _decode_kernel[(triton.cdiv(self.n, BLOCK),)](
                    st[o_pk:o_pk + n_pk], st[:self.n], self.luts[li], out, self.n, BLOCK=BLOCK)
                if n_esc:
                    i = st[o_ei:o_ei + 4 * n_esc].view(torch.int32).long()
                    ev = st[o_ev:o_ev + 2 * n_esc].view(torch.int16)
                    smv = st[:self.n][i].to(torch.int16)
                    bits = (ev << 7) | (smv & 0x7F)
                    bits = torch.where(smv >= 128, bits | -32768, bits)
                    out[i] = bits.view(torch.bfloat16).to(torch.float16)
            else:
                self.flat[slot].copy_(self.host[li], non_blocking=True)
            self.ready[slot].record(self.copy)
        self.slot_of[li] = slot

    def pre(self, mod, args):
        slot = self.slot_of[mod._sidx]
        torch.cuda.current_stream(DEV).wait_event(self.ready[slot])
        for (n, p), v in zip(mod.named_parameters(), self.views[slot]):
            p.data = v

    def post(self, mod, args, out):
        li = mod._sidx
        slot = self.slot_of.pop(li)
        for _, p in mod.named_parameters():
            p.data = torch.empty(0, dtype=torch.float16)
        nxt = self.ids[(self.pos[li] + 2) % len(self.ids)]
        self.prefetch(nxt, slot)
        return out


def build(mode, frac):
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True).eval()
    total_bytes = sum(p.numel() for p in model.parameters()) * 2
    budget = frac * total_bytes
    layers = model.model.layers
    layer_bytes = sum(p.numel() for p in layers[0].parameters()) * 2
    head_bytes = model.lm_head.weight.numel() * 2
    tied = model.config.tie_word_embeddings
    model.model.embed_tokens = CpuEmbedding(model.model.embed_tokens.weight.data)
    if tied:
        model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.w.clone())
    n_res = int((budget - head_bytes - 2 * layer_bytes - 0.4 * GB) // layer_bytes)
    n_res = max(0, min(n_res, len(layers)))
    resident = list(range(n_res))
    streamed = list(range(n_res, len(layers)))
    # everything except streamed layers + embedding goes to the GPU in fp16
    for i in resident:
        layers[i].to(DEV, torch.float16)
    model.model.norm.to(DEV, torch.float16)
    model.model.rotary_emb.to(DEV)
    model.lm_head.to(DEV, torch.float16)
    model.config.torch_dtype = torch.float16
    st = Streamer(layers, streamed, mode) if streamed else None
    # generate() infers the device from the first parameter, which may be a
    # streamed (released) layer; pin the answer to the GPU.
    model.__class__ = type(model.__class__.__name__, (model.__class__,),
                           {"device": property(lambda self: DEV)})
    info = dict(total_gb=total_bytes / GB, budget_gb=budget / GB, resident=n_res,
                streamed=len(streamed), layer_gb=layer_bytes / GB)
    if st:
        info["pcie_gb_per_token"] = st.bytes_per_layer * len(streamed) / GB
    return model, info


def build_accel(frac):
    total = None
    cfg_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                     low_cpu_mem_usage=True)
    total = sum(p.numel() for p in cfg_model.parameters()) * 2
    del cfg_model
    gc.collect()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto",
        max_memory={0: int(frac * total), "cpu": "28GiB"}).eval()
    return model, dict(total_gb=total / GB, budget_gb=frac * total / GB)


@torch.no_grad()
def bench(model, tok, batch):
    ids = tok([PROMPT] * batch, return_tensors="pt").to(DEV)
    torch.cuda.synchronize()
    t = time.time()
    out = model.generate(**ids, max_new_tokens=GEN, min_new_tokens=GEN, do_sample=False)
    torch.cuda.synchronize()
    dt = time.time() - t
    return dt, out


def pcie_bw():
    x = torch.empty(512 << 20, dtype=torch.uint8).pin_memory()
    y = torch.empty_like(x, device=DEV)
    y.copy_(x); torch.cuda.synchronize()
    t = time.time()
    for _ in range(5):
        y.copy_(x, non_blocking=True)
    torch.cuda.synchronize()
    return 5 * x.numel() / (time.time() - t) / GB


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"PCIe host->GPU: {pcie_bw():.1f} GB/s", flush=True)
    t = time.time()
    model, info = build_accel(FRAC) if MODE == "accel" else build(MODE, FRAC)
    torch.cuda.synchronize()
    info["build_s"] = round(time.time() - t)
    info["gpu_alloc_gb"] = torch.cuda.memory_allocated(DEV) / GB
    print(MODE, json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in info.items()}),
          flush=True)
    bench(model, tok, 1)  # warmup
    for b in BATCHES:
        torch.cuda.reset_peak_memory_stats(DEV)
        dt, out = bench(model, tok, b)
        peak = torch.cuda.max_memory_allocated(DEV) / GB
        print(f"{MODE} batch={b:3d}: {GEN} steps in {dt:6.1f}s -> {dt / GEN:6.3f} s/step, "
              f"{b * GEN / dt:7.2f} tokens/s, peak GPU {peak:.2f} GB", flush=True)
    text = tok.decode(out[0][-GEN:])
    torch.save(out[0].cpu(), f"tokens_{MODE}.pt")
    print("sample:", repr(text[:160]))


if __name__ == "__main__":
    main()
