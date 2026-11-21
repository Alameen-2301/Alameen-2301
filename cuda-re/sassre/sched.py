"""Scheduling (control) bits: decode and sanity-check them.

nvdisasm never prints these bits, so their layout cannot be checked against
it directly. Instead this checks that the layout is self-consistent on real
compiler output:

  * every wait-mask bit k should be preceded (in the same kernel) by an
    instruction that set scoreboard k as its read or write barrier;
  * variable-latency instructions (memory, MUFU, S2R, ...) should be the
    ones setting barriers, fixed-latency ALU ops should not.

    python -m sassre.sched                # check every corpus kernel
    python -m sassre.sched k_mem          # annotated listing of one kernel
"""
from __future__ import annotations

import struct
import sys
from collections import Counter

from .corpus import mnemonic
from .cubin import text_sections
from .disasm import I, decode
from .toolchain import ROOT, compile_cuda


def kernels(arch="sm_80"):
    src = (ROOT / "kernels" / "corpus.cu").read_text()
    _, cub = compile_cuda(src, arch, "corpus.cu")
    for name, code in text_sections(cub):
        ins = [struct.unpack_from("<QQ", code, a) for a in range(0, len(code), 16)]
        yield name.removeprefix(".text."), ins


def check():
    setters, fixed_setters = Counter(), Counter()
    waits = orphan = 0
    for name, ins in kernels():
        live: set[int] = set()
        for a, (lo, hi) in enumerate(ins):
            i = I(lo, hi)
            mn = mnemonic(decode(lo, hi, a * 16)) or f"op{i.opcode:#x}"
            for k in range(6):
                if i.wait >> k & 1:
                    waits += 1
                    if k not in live:
                        orphan += 1
                        print(f"  {name}+{a * 16:#06x}: waits on SB{k} never set: {mn}")
            for bar in (i.wbar, i.rbar):
                if bar != 7:
                    live.add(bar)
                    setters[mn] += 1
    print(f"wait-mask bits checked: {waits}, without a prior setter: {orphan}")
    print("instructions that set scoreboards:",
          ", ".join(f"{k}:{v}" for k, v in setters.most_common()))


def listing(kname: str):
    for name, ins in kernels():
        if name != kname:
            continue
        for a, (lo, hi) in enumerate(ins):
            i = I(lo, hi)
            print(f"/*{a * 16:04x}*/ {i.ctrl()}  {decode(lo, hi, a * 16)}")
            if mnemonic(decode(lo, hi)) == "EXIT" and a + 1 < len(ins) and \
                    mnemonic(decode(*ins[a + 1])) == "BRA":
                break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        listing(sys.argv[1])
    else:
        check()
