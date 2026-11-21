"""Minimal parser for CUDA .cubin files (ELF64 with NVIDIA-specific
sections). No external dependencies.

    python -m sassre.cubin file.cubin
"""
from __future__ import annotations

import struct
import sys

# .nv.info attribute ids (EIATTR_*), recovered from nvdisasm/cuobjdump output
# and the public descriptions in open-source CUDA tooling.
EIATTR = {
    0x04: "CTAIDZ_USED", 0x0a: "PARAM_CBANK", 0x0d: "SYNC_STACK",
    0x0f: "EXTERNS", 0x10: "REQNTID", 0x11: "FRAME_SIZE", 0x12: "MIN_STACK_SIZE",
    0x17: "KPARAM_INFO", 0x19: "CBANK_PARAM_SIZE", 0x1b: "MAXREG_COUNT",
    0x1c: "EXIT_INSTR_OFFSETS", 0x1d: "S2RCTAID_INSTR_OFFSETS", 0x1e: "CRS_STACK_SIZE",
    0x23: "MAX_STACK_SIZE", 0x25: "LD_CACHEMOD_INSTR_OFFSETS",
    0x28: "COOP_GROUP_INSTR_OFFSETS", 0x2a: "SW_WAR", 0x2f: "REGCOUNT",
    0x31: "INT_WARP_WIDE_INSTR_OFFSETS", 0x36: "SW2861232_WAR",
    0x37: "CUDA_API_VERSION",
    0x50: "KERNEL_CALLS_STATS_MASK", 0x54: "SPARSE_MMA_MASK",
}
FMT = {1: "NVAL", 2: "BVAL", 3: "HVAL", 4: "SVAL"}


def sections(data: bytes) -> list[dict]:
    if data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = []
    for k in range(shnum):
        name, typ, flags, addr, off, size, link, info, align, ent = \
            struct.unpack_from("<IIQQQQIIQQ", data, shoff + k * shentsize)
        secs.append(dict(name_off=name, type=typ, flags=flags, offset=off, size=size,
                         link=link, info=info))
    strtab = secs[shstrndx]
    for s in secs:
        o = strtab["offset"] + s["name_off"]
        s["name"] = data[o:data.index(b"\0", o)].decode()
        s["data"] = data[s["offset"]:s["offset"] + s["size"]] if s["type"] != 8 else b""
    return secs


def text_sections(data: bytes):
    for s in sections(data):
        if s["name"].startswith(".text."):
            yield s["name"], s["data"]


def arch(data: bytes) -> str:
    flags, = struct.unpack_from("<I", data, 0x30)
    return f"sm_{flags & 0xff}"


def nv_info(blob: bytes):
    """Yield (attr_name, value) from a .nv.info / .nv.info.<fn> section.
    Records are: u8 format, u8 attr, then either u16 value (HVAL) or
    u16 size + payload (SVAL)."""
    o = 0
    while o + 4 <= len(blob):
        fmt, attr = blob[o], blob[o + 1]
        name = EIATTR.get(attr, f"ATTR_{attr:#04x}")
        if fmt == 4:
            size, = struct.unpack_from("<H", blob, o + 2)
            payload = blob[o + 4:o + 4 + size]
            o += 4 + size
            if size % 4 == 0 and size:
                val = list(struct.unpack(f"<{size // 4}I", payload))
            else:
                val = payload.hex()
        else:
            val, = struct.unpack_from("<H", blob, o + 2)
            o += 4
        yield name, val


def describe(data: bytes) -> None:
    print(f"arch {arch(data)}")
    for s in sections(data):
        print(f"  {s['name']:32s} type={s['type']:#x} size={s['size']}")
        if s["name"].startswith(".nv.info"):
            for k, v in nv_info(s["data"]):
                print(f"      {k:28s} {v}")


if __name__ == "__main__":
    describe(open(sys.argv[1], "rb").read())
