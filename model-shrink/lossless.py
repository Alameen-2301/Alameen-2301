"""Experiment 1: how small can BF16 weights get *losslessly*?

A BF16 value = 1 sign bit | 8 exponent bits | 7 mantissa bits.
We measure the Shannon entropy of each field across a real model, then
actually compress (Huffman-code the exponent, keep sign+mantissa raw) and
verify the round trip is bit-exact.

    python lossless.py models/qwen2.5-0.5b/model.safetensors
"""
from __future__ import annotations

import heapq
import sys
from collections import Counter

import numpy as np
import zstandard
from safetensors import safe_open


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def huffman_lengths(counts: dict[int, int]) -> dict[int, int]:
    """Huffman code length per symbol (canonical codes need only lengths)."""
    if len(counts) == 1:
        return {next(iter(counts)): 1}
    heap = [(c, k, [s]) for k, (s, c) in enumerate(counts.items())]
    heapq.heapify(heap)
    length = Counter()
    k = len(heap)
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            length[s] += 1
        heapq.heappush(heap, (c1 + c2, k, s1 + s2))
        k += 1
    return dict(length)


def encode_exponents(exp: np.ndarray, lengths: dict[int, int]) -> bytes:
    """Canonical Huffman encode -> packed bit stream (numpy, vectorized)."""
    syms = sorted(lengths, key=lambda s: (lengths[s], s))
    code, prev_len, codes = 0, 0, {}
    for s in syms:
        code <<= lengths[s] - prev_len
        codes[s] = (code, lengths[s])
        prev_len = lengths[s]
        code += 1
    lut_code = np.zeros(257, np.uint64)
    lut_len = np.zeros(257, np.int64)
    for s, (c, l) in codes.items():
        lut_code[s], lut_len[s] = c, l
    L = lut_len[exp]
    C = lut_code[exp]
    maxl = int(L.max())
    # expand every code into maxl bits, MSB first, then drop the padding
    shifts = np.arange(maxl - 1, -1, -1, dtype=np.uint64)
    bits = ((C[:, None] >> shifts) & 1).astype(np.uint8)
    keep = np.arange(maxl)[None, :] >= (maxl - L)[:, None]
    return np.packbits(bits[keep]).tobytes(), codes


def decode_exponents(stream: bytes, codes: dict, n: int) -> np.ndarray:
    """Table-driven canonical Huffman decode (what a GPU kernel would do)."""
    maxl = max(l for _, l in codes.values())
    table_sym = np.zeros(1 << maxl, np.int64)
    table_len = np.zeros(1 << maxl, np.uint8)
    for s, (c, l) in codes.items():
        lo = c << (maxl - l)
        table_sym[lo:lo + (1 << (maxl - l))] = s
        table_len[lo:lo + (1 << (maxl - l))] = l
    bits = np.unpackbits(np.frombuffer(stream, np.uint8))
    bits = np.concatenate([bits, np.zeros(maxl, np.uint8)])
    out = np.empty(n, np.int64)
    pos = 0
    weights = (1 << np.arange(maxl - 1, -1, -1)).astype(np.int64)
    bits = bits.astype(np.int64)
    for k in range(n):  # sequential reference decoder
        w = int(bits[pos:pos + maxl] @ weights)
        out[k] = table_sym[w]
        pos += int(table_len[w])
    return out


def main(path: str, verify_tensor_elems: int = 300_000):
    f = safe_open(path, "np" if False else "pt")
    import torch

    tot_n = 0
    cs, ce, cm = np.zeros(2, np.int64), np.zeros(256, np.int64), np.zeros(128, np.int64)
    raw_chunks = []
    for name in f.keys():
        t = f.get_tensor(name)
        assert t.dtype == torch.bfloat16, (name, t.dtype)
        u = t.contiguous().view(torch.int16).numpy().view(np.uint16).ravel()
        tot_n += u.size
        cs += np.bincount(u >> 15, minlength=2)
        ce += np.bincount((u >> 7) & 0xFF, minlength=256)
        cm += np.bincount(u & 0x7F, minlength=128)
        raw_chunks.append(u)
    u = np.concatenate(raw_chunks)

    Hs, He, Hm = entropy(cs), entropy(ce), entropy(cm)
    print(f"parameters: {tot_n / 1e6:.1f} M  ({tot_n * 2 / 1e6:.0f} MB as BF16)\n")
    print("field      bits stored   entropy (bits of real information)")
    print(f"sign            1          {Hs:.3f}")
    print(f"exponent        8          {He:.3f}   <- {256 - (ce > 0).sum()} of 256 values never used")
    print(f"mantissa        7          {Hm:.3f}")
    print(f"total          16         {Hs + He + Hm:.3f}  -> best possible "
          f"{(Hs + He + Hm) / 16 * 100:.1f}% of original size\n")

    top = np.argsort(ce)[::-1][:8]
    print("most common exponents (unbiased):",
          ", ".join(f"2^{int(e) - 127}:{ce[e] / tot_n * 100:.1f}%" for e in top))

    # practical codecs
    zc = zstandard.ZstdCompressor(level=19, threads=-1)
    sample = u[: 20_000_000]
    z_raw = len(zc.compress(sample.tobytes())) / sample.nbytes
    hi = (sample >> 8).astype(np.uint8)
    lo = (sample & 0xFF).astype(np.uint8)
    z_split = (len(zc.compress(hi.tobytes())) + len(zc.compress(lo.tobytes()))) / sample.nbytes
    lengths = huffman_lengths({int(s): int(c) for s, c in enumerate(ce) if c})
    huff_bits = sum(lengths[s] * int(ce[s]) for s in lengths)
    huff_ratio = (tot_n * 8 + huff_bits) / (tot_n * 16)  # sign+mantissa raw (8b) + coded exp
    print(f"\nzstd -19 on raw bytes:                   {z_raw * 100:.1f}% of original")
    print(f"zstd -19 on split hi/lo byte planes:     {z_split * 100:.1f}%")
    print(f"Huffman(exponent) + raw sign/mantissa:   {huff_ratio * 100:.1f}%  "
          f"(avg exponent code {huff_bits / tot_n:.2f} bits; max {max(lengths.values())})")

    # Practical, GPU-friendly variant: the 31 most common exponents get
    # Huffman codes, everything else is ESC + raw 8 bits. This keeps codes
    # short so decoding is a single table lookup.
    order = np.argsort(ce)[::-1]
    top = [int(e) for e in order[:31] if ce[e]]
    ESC = 256
    while True:  # shrink the codebook until every code is <= 16 bits
        cnt = {e: int(ce[e]) for e in top}
        cnt[ESC] = int(tot_n - sum(cnt.values())) or 1
        lens = huffman_lengths(cnt)
        if max(lens.values()) <= 16:
            break
        top = top[:-1]
    esc_n = cnt[ESC] if cnt[ESC] > 1 else 0
    bits = sum(lens[e] * cnt[e] for e in top) + esc_n * (lens[ESC] + 8)
    print(f"capped codebook ({len(top)} symbols + escape):   {(tot_n * 8 + bits) / (tot_n * 16) * 100:.1f}%  "
          f"(max code {max(lens.values())} bits, {esc_n / tot_n * 100:.4f}% escaped)")

    # round-trip proof: encode + decode real weights, compare bits
    n = verify_tensor_elems
    s = u[:n]
    exp = ((s >> 7) & 0xFF).astype(np.int64)
    sym = np.where(np.isin(exp, top), exp, ESC)
    stream, codes = encode_exponents(sym, lens)
    raw_esc = exp[sym == ESC].astype(np.uint8)
    got = decode_exponents(stream, codes, n)
    got = got.astype(np.int64)
    got[got == ESC] = raw_esc
    sm_hi = (s & 0x8000)
    rebuilt = sm_hi | (got.astype(np.uint16) << 7) | (s & 0x7F)
    ok = np.array_equal(rebuilt.astype(np.uint16), s)
    print(f"\nround trip on {n:,} real weights: {'BIT-EXACT' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models/qwen2.5-0.5b/model.safetensors")
