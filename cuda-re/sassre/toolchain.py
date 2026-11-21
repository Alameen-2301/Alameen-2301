"""Locate and drive NVIDIA's offline tools: NVRTC (CUDA C++ -> PTX/CUBIN),
ptxas (PTX -> CUBIN) and nvdisasm (CUBIN / raw SASS -> text).

nvdisasm is used only as an *oracle*: it gives ground truth that our own
decoder (sassre.disasm) is checked against.
"""
from __future__ import annotations

import ctypes
import glob
import os
import re
import struct
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _site_packages() -> list[Path]:
    out = []
    for p in sys.path:
        if p.endswith("site-packages") and os.path.isdir(p):
            out.append(Path(p))
    venv = ROOT / ".venv"
    out += [Path(p) for p in glob.glob(str(venv / "lib" / "python3*" / "site-packages"))]
    return out


@lru_cache(None)
def find(name: str) -> Path:
    """Find a tool binary or library inside the installed nvidia-* wheels."""
    for sp in _site_packages():
        hits = sorted(glob.glob(str(sp / "nvidia" / "**" / name), recursive=True))
        if hits:
            p = Path(hits[0])
            if p.suffix != ".so" and ".so" not in p.name:
                p.chmod(p.stat().st_mode | 0o111)
            return p
    raise FileNotFoundError(f"{name} not found; run ./setup.sh first")


# ---------------------------------------------------------------- NVRTC

class NvrtcError(RuntimeError):
    pass


@lru_cache(None)
def _nvrtc():
    lib = ctypes.CDLL(str(find("libnvrtc.so.12")))
    lib.nvrtcGetErrorString.restype = ctypes.c_char_p
    return lib


def _chk(rc: int):
    if rc:
        raise NvrtcError(_nvrtc().nvrtcGetErrorString(rc).decode())


def compile_cuda(src: str, arch: str = "sm_80", name: str = "k.cu",
                 extra: tuple[str, ...] = ()) -> tuple[str, bytes]:
    """Compile CUDA C++ with NVRTC. Returns (ptx_text, cubin_bytes)."""
    lib = _nvrtc()
    prog = ctypes.c_void_p()
    _chk(lib.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), name.encode(),
                                0, None, None))
    try:
        inc = find("cuda_fp16.h").parent
        opts = [f"--gpu-architecture={arch}", "-default-device", "-std=c++17",
                f"-I{inc}", *extra]
        arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
        rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
        if rc:
            n = ctypes.c_size_t()
            lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
            log = ctypes.create_string_buffer(n.value)
            lib.nvrtcGetProgramLog(prog, log)
            raise NvrtcError(log.value.decode(errors="replace"))
        n = ctypes.c_size_t()
        _chk(lib.nvrtcGetPTXSize(prog, ctypes.byref(n)))
        ptx = ctypes.create_string_buffer(n.value)
        _chk(lib.nvrtcGetPTX(prog, ptx))
        _chk(lib.nvrtcGetCUBINSize(prog, ctypes.byref(n)))
        cub = ctypes.create_string_buffer(n.value)
        _chk(lib.nvrtcGetCUBIN(prog, cub))
        return ptx.value.decode(), cub.raw
    finally:
        lib.nvrtcDestroyProgram(ctypes.byref(prog))


# ---------------------------------------------------------------- ptxas

def ptxas(ptx: str, arch: str = "sm_80") -> bytes:
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d, "k.ptx"), Path(d, "k.cubin")
        src.write_text(ptx)
        r = subprocess.run([str(find("ptxas")), f"-arch={arch}", str(src), "-o", str(out)],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        return out.read_bytes()


# ---------------------------------------------------------------- nvdisasm

_LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;?\s*/\* 0x([0-9a-f]{16}) \*/")
_HI = re.compile(r"^\s*/\* 0x([0-9a-f]{16}) \*/\s*$")
_ERR = re.compile(r"at address 0x([0-9a-f]+)")


def _parse_listing(text: str) -> list[tuple[int, int, int, str]]:
    """-> [(addr, lo, hi, asm_text)] from `nvdisasm -hex` output."""
    out, pend = [], None
    for line in text.splitlines():
        m = _LINE.search(line)
        if m:
            asm = m.group(2).strip().rstrip(";").strip()
            pend = (int(m.group(1), 16), int(m.group(3), 16), asm)
            continue
        m = _HI.match(line)
        if m and pend:
            out.append((pend[0], pend[1], int(m.group(1), 16), pend[2]))
            pend = None
    return out


def disasm_cubin(cubin: bytes) -> list[tuple[int, int, int, str]]:
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cubin)
        f.flush()
        r = subprocess.run([str(find("nvdisasm")), "-hex", f.name],
                           capture_output=True, text=True)
    return _parse_listing(r.stdout)


def _sm(arch: str) -> str:
    return "SM" + arch.split("_")[1]


def disasm_raw(words: list[tuple[int, int]], arch: str = "sm_80") -> list[str | None]:
    """Disassemble raw 128-bit instructions (lo, hi). Returns text per
    instruction, or None where nvdisasm calls the encoding illegal.

    nvdisasm prints nothing at all if any instruction in the batch is
    illegal, so illegal ones are reported by address, replaced with NOP and
    the batch retried.
    """
    nop = (0x0000000000007918, 0x000FC00000000000)
    words = list(words)
    bad: set[int] = set()
    while True:
        blob = b"".join(struct.pack("<QQ", *(nop if i in bad else w))
                        for i, w in enumerate(words))
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            f.write(blob)
            f.flush()
            r = subprocess.run([str(find("nvdisasm")), "-b", _sm(arch), "-hex", f.name],
                               capture_output=True, text=True)
        new = {int(a, 16) // 16 for a in _ERR.findall(r.stderr)} - bad
        if not new:
            if r.returncode and not r.stdout:
                raise RuntimeError(r.stderr)
            got = {a // 16: t for a, _, _, t in _parse_listing(r.stdout)}
            return [None if i in bad else got.get(i) for i in range(len(words))]
        bad |= new


if __name__ == "__main__":
    for t in ("ptxas", "nvdisasm", "cuobjdump", "libnvrtc.so.12"):
        print(f"{t:16s} {find(t)}")
    ptx, cub = compile_cuda("__global__ void k(float*a){a[threadIdx.x]+=1.f;}")
    print("NVRTC ok:", len(ptx), "bytes PTX,", len(cub), "bytes CUBIN")
    for a, lo, hi, s in disasm_cubin(cub)[:4]:
        print(f"  {a:04x}  {hi:016x}{lo:016x}  {s}")
