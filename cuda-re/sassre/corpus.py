"""Build a ground-truth corpus: compile kernels/*.cu for one or more
architectures and record every (encoding, nvdisasm text) pair.

    python -m sassre.corpus [sm_80 sm_86 ...]

Writes data/corpus_<arch>.tsv with columns: hi lo text (hex, no 0x).
"""
from __future__ import annotations

import sys
from collections import Counter

from .toolchain import ROOT, compile_cuda, disasm_cubin


def build(arch: str) -> list[tuple[int, int, str]]:
    seen: dict[tuple[int, int], str] = {}
    for cu in sorted((ROOT / "kernels").glob("*.cu")):
        for opt in ((), ("-G",), ("--use_fast_math",)):
            try:
                _, cubin = compile_cuda(cu.read_text(), arch, cu.name, opt)
            except Exception as e:  # noqa: BLE001
                print(f"  {cu.name} {opt}: {str(e).splitlines()[0]}", file=sys.stderr)
                continue
            for _, lo, hi, txt in disasm_cubin(cubin):
                seen.setdefault((lo, hi), txt)
    return [(lo, hi, t) for (lo, hi), t in seen.items()]


def save(arch: str, rows) -> None:
    out = ROOT / "data" / f"corpus_{arch}.tsv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for lo, hi, t in sorted(rows, key=lambda r: (r[2].split()[0], r[0])):
            f.write(f"{hi:016x}\t{lo:016x}\t{t}\n")
    print(f"{arch}: {len(rows)} unique instructions -> {out.relative_to(ROOT)}")


def load(arch: str = "sm_80") -> list[tuple[int, int, str]]:
    rows = []
    for line in (ROOT / "data" / f"corpus_{arch}.tsv").read_text().splitlines():
        hi, lo, t = line.split("\t", 2)
        rows.append((int(lo, 16), int(hi, 16), t))
    return rows


def mnemonic(text: str) -> str:
    t = text.split()
    if t and t[0].startswith("@"):
        t = t[1:]
    return t[0].split(".")[0] if t else ""


if __name__ == "__main__":
    for arch in sys.argv[1:] or ["sm_80"]:
        rows = build(arch)
        save(arch, rows)
        c = Counter(mnemonic(t) for _, _, t in rows)
        print(f"  {len(c)} distinct opcodes:", " ".join(f"{k}:{v}" for k, v in c.most_common()))
