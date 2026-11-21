"""Extract lookup tables from nvdisasm by exhaustive enumeration, so the
decoder does not need them typed in by hand.

    python -m sassre.tables  -> data/tables_sm_80.json
"""
from __future__ import annotations

import json

from .enum import enum, pick
from .toolchain import ROOT


def sr_names() -> dict[str, str]:
    lo, hi, _ = pick("0x919")  # S2R Rd, SR_xxx
    out = {}
    for v, t in enum(lo, hi, 72, 79):
        if t:
            out[str(v)] = t.split(", ", 1)[1]
    return out


if __name__ == "__main__":
    t = {"sr": sr_names()}
    p = ROOT / "data" / "tables_sm_80.json"
    p.write_text(json.dumps(t, indent=1))
    print(f"{len(t['sr'])} special registers -> {p.relative_to(ROOT)}")
