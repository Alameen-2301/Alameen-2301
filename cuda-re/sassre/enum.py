"""Enumerate every value of a bit field on a real instruction and print what
nvdisasm makes of it. Used to extract modifier tables.

    python -m sassre.enum "ISETP.GE.AND P0, PT, R0, 0x1, PT" 72 78
    python -m sassre.enum 0x80c 72 78            # by form
"""
from __future__ import annotations

import sys

from .corpus import load
from .fuzz import FORM_MASK, seeds, split_ops
from .toolchain import disasm_raw


def setbits(lo: int, hi: int, b0: int, b1: int, v: int) -> tuple[int, int]:
    x = (hi << 64) | lo
    n = b1 - b0 + 1
    x = (x & ~(((1 << n) - 1) << b0)) | (v << b0)
    return x & ((1 << 64) - 1), x >> 64


def enum(lo: int, hi: int, b0: int, b1: int, arch="sm_80") -> list[tuple[int, str | None]]:
    vals = range(1 << (b1 - b0 + 1))
    out = disasm_raw([setbits(lo, hi, b0, b1, v) for v in vals], arch)
    return list(zip(vals, out))


def pick(key: str, arch="sm_80") -> tuple[int, int, str]:
    if key.startswith("0x"):
        return seeds(arch)[int(key, 16)]
    for lo, hi, t in load(arch):
        if t == key or t.startswith(key):
            return lo, hi, t
    raise SystemExit(f"no corpus instruction matches {key!r}")


if __name__ == "__main__":
    key, b0, b1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    lo, hi, t = pick(key)
    print(f"seed: {t}   form 0x{lo & FORM_MASK:03x}")
    for v, s in enum(lo, hi, b0, b1):
        print(f"  {v:#6x}  {s if s is not None else '<illegal>'}")
