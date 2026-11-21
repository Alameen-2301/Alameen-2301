"""Check sassre.disasm against nvdisasm.

Two tests:
  corpus   every instruction the compiler actually produced
  random   for each supported corpus instruction, randomize the operand
           fields (registers, predicates, immediates, constant offsets,
           guard, modifier bits) and keep the encodings nvdisasm accepts.
           This exercises combinations the compiler never emitted.

    python -m sassre.validate [--random N] [--show K]
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from .corpus import load, mnemonic
from .disasm import decode, supported
from .toolchain import disasm_raw

# bit ranges that are safe to randomize (operands, not opcode/form)
RANDOM_FIELDS = [(12, 15), (16, 23), (24, 31), (32, 39), (32, 63), (40, 53),
                 (64, 71), (62, 63), (72, 80), (81, 90), (122, 125)]


def norm(t: str) -> str:
    return " ".join(t.replace(" ,", ",").split())


def compare(rows, show: int, label: str):
    tot, ok = Counter(), Counter()
    bad = defaultdict(list)
    for lo, hi, ref in rows:
        m = mnemonic(ref)
        if not supported(lo) or "@lo(" in ref or "@hi(" in ref:
            continue  # relocated operands live in the ELF, not the instruction
        tot[m] += 1
        got = decode(lo, hi)
        # branch targets are labels in nvdisasm; compare everything else
        if m in ("BRA", "BSSY", "CALL") or norm(got) == norm(ref):
            ok[m] += 1
        else:
            bad[m].append((ref, got, lo, hi))
    T, O = sum(tot.values()), sum(ok.values())
    print(f"\n== {label}: {O}/{T} exact ({100 * O / max(T, 1):.1f}%)")
    for m in sorted(tot, key=lambda k: tot[k] - ok[k], reverse=True):
        if tot[m] != ok[m]:
            print(f"  {m:8s} {ok[m]}/{tot[m]}")
            for ref, got, lo, hi in bad[m][:show]:
                print(f"      nvdisasm: {ref}\n      ours:     {got}   ({hi:016x}_{lo:016x})")
    return O, T


def random_rows(n_per: int, seed: int = 1):
    rng = random.Random(seed)
    rows = [r for r in load() if supported(r[0])]
    words = []
    for lo, hi, _ in rows:
        for _ in range(n_per):
            x = (hi << 64) | lo
            for _ in range(rng.randint(1, 3)):
                a, b = rng.choice(RANDOM_FIELDS)
                n = b - a + 1
                x = (x & ~(((1 << n) - 1) << a)) | (rng.getrandbits(n) << a)
            x |= 1 << 109  # make .reuse visible
            x = (x & ~(0xF << 105)) | (1 << 105)  # stall 1: legal together with yield
            words.append((x & ((1 << 64) - 1), x >> 64))
    texts = []
    for k in range(0, len(words), 4096):
        texts += disasm_raw(words[k:k + 4096])
    return [(lo, hi, t) for (lo, hi), t in zip(words, texts) if t is not None]


def branches(show: int):
    """nvdisasm prints branch targets as labels (.L_x_N). Resolve each label
    to its address and compare with the target our decoder computes."""
    import re
    import subprocess
    import tempfile

    from .toolchain import ROOT, compile_cuda, find

    _, cub = compile_cuda((ROOT / "kernels" / "corpus.cu").read_text(), "sm_80", "c.cu")
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cub)
        f.flush()
        out = subprocess.run([str(find("nvdisasm")), "-hex", f.name],
                             capture_output=True, text=True).stdout
    labels, pending, brs, sec = {}, [], [], None
    for line in out.splitlines():
        if line.startswith("//-----"):
            sec = line
        m = re.match(r"^(\.L_x_\d+):", line.strip())
        if m:
            pending.append(m.group(1))
            continue
        m = re.search(r"/\*([0-9a-f]{4})\*/\s+(.*?)\s*/\* 0x([0-9a-f]{16}) \*/", line)
        if m:
            a = int(m.group(1), 16)
            for p in pending:
                labels[p] = a
            pending = []
            lab = re.search(r"`\((\.L_x_\d+)\)", m.group(2))
            brs.append([a, m.group(2), int(m.group(3), 16), None, lab and lab.group(1)])
            continue
        m = re.match(r"^\s*/\* 0x([0-9a-f]{16}) \*/\s*$", line)
        if m and brs and brs[-1][3] is None:
            brs[-1][3] = int(m.group(1), 16)
    ok = bad = 0
    for a, txt, lo, hi, lab in brs:
        if not lab or lab not in labels:
            continue
        got = re.search(r"\.L_(0x[0-9a-f]+)", decode(lo, hi, a))
        if got and int(got.group(1), 16) == labels[lab]:
            ok += 1
        else:
            bad += 1
            if bad <= show:
                print(f"      {a:#06x} nvdisasm: {txt} (= {labels[lab]:#x})  ours: {decode(lo, hi, a)}")
    print(f"\n== branch targets: {ok}/{ok + bad} resolve to nvdisasm's label address")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=4, help="mutations per instruction")
    ap.add_argument("--show", type=int, default=2)
    a = ap.parse_args()
    rows = load()
    sup = sum(supported(lo) for lo, _, _ in rows)
    print(f"corpus: {len(rows)} instructions, {sup} ({100 * sup / len(rows):.1f}%) "
          f"have an opcode the decoder implements")
    compare(rows, a.show, "corpus")
    if a.random:
        compare(random_rows(a.random), a.show, f"random mutations (x{a.random})")
    branches(a.show)


if __name__ == "__main__":
    main()
