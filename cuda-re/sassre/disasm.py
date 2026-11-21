"""An independent SASS disassembler for Ampere (sm_80), written from the
field maps that sassre.fuzz / sassre.enum recovered. It does not call
nvdisasm; sassre.validate compares its output with nvdisasm.

    python -m sassre.disasm file.cubin        # disassemble every .text section
    python -m sassre.disasm 0x000fe40000000f00_00000a0000017a02   # one word (hi_lo)

Encoding summary (sm_80, 128-bit instructions, little-endian lo|hi words):

    bits   0..8    opcode
    bits   9..11   operand form (where B/C come from, see _src)
    bits  12..15   guard predicate (12..14 index, 15 negate; 7 = PT)
    bits  16..23   Rd           24..31 Ra
    bits  32..39   Rb  | 32..63 imm32 | 40..53 const offset/4, 54..58 bank
    bits  64..71   Rc
    bits 105..108  stall cycles      bit 109 yield
    bits 110..112  write barrier     113..115 read barrier
    bits 116..121  wait mask         122..125 operand reuse (a,b,c)
"""
from __future__ import annotations

import json
import math
import struct
import sys
from functools import lru_cache

from .toolchain import ROOT

# ------------------------------------------------------------------ helpers


class I:
    """One 128-bit instruction."""

    __slots__ = ("lo", "hi", "x")

    def __init__(self, lo: int, hi: int):
        self.lo, self.hi, self.x = lo, hi, (hi << 64) | lo

    def f(self, a: int, b: int) -> int:  # inclusive bit range
        return (self.x >> a) & ((1 << (b - a + 1)) - 1)

    def b(self, n: int) -> int:
        return (self.x >> n) & 1

    # --- control / scheduling info (not printed by nvdisasm except .reuse)
    @property
    def opcode(self): return self.f(0, 8)
    @property
    def form(self): return self.f(9, 11)
    @property
    def stall(self): return self.f(105, 108)
    @property
    def yld(self): return self.b(109)
    @property
    def wbar(self): return self.f(110, 112)
    @property
    def rbar(self): return self.f(113, 115)
    @property
    def wait(self): return self.f(116, 121)
    @property
    def reuse(self): return self.f(122, 125)

    def ctrl(self) -> str:
        """CuAssembler-style control annotation: [wait:rbar:wbar:yield:stall]"""
        w = "".join(str(i) if self.wait >> i & 1 else "-" for i in range(6))
        r = "-" if self.rbar == 7 else str(self.rbar)
        wb = "-" if self.wbar == 7 else str(self.wbar)
        return f"[B{w}:R{r}:W{wb}:{'Y' if self.yld else '-'}:S{self.stall:02d}]"


def R(n: int) -> str: return "RZ" if n == 255 else f"R{n}"
def UR(n: int) -> str: return "URZ" if n == 63 else f"UR{n}"
def P(n: int, neg: int = 0) -> str: return ("!" if neg else "") + ("PT" if n == 7 else f"P{n}")
def UP(n: int, neg: int = 0) -> str: return ("!" if neg else "") + ("UPT" if n == 7 else f"UP{n}")


def hx(v: int) -> str:
    return f"-{-v:#x}" if v < 0 else f"{v:#x}"


def s32(v: int) -> int: return v - (1 << 32) if v & 0x80000000 else v
def s16(v: int) -> int: return v - (1 << 16) if v & 0x8000 else v
def s24(v: int) -> int: return v - (1 << 24) if v & 0x800000 else v


def fmt_float(v: float) -> str:
    """nvdisasm's float style, recovered from 1500 random samples:
    %.20g below 1e9 (trailing zeros stripped), %.20e from 1e9 up."""
    if math.isinf(v):
        return "+INF" if v > 0 else "-INF"
    if v == 0:
        return "-0.0" if math.copysign(1, v) < 0 else "0"
    if abs(v) >= 1e9:
        return f"{v:.20e}"
    return f"{v:.20g}"


def f32imm(bits: int) -> str:
    v = struct.unpack("<f", struct.pack("<I", bits))[0]
    if math.isnan(v):
        return ("-" if bits >> 31 else "+") + ("QNAN" if bits >> 22 & 1 else "SNAN")
    return fmt_float(v)


def f64imm(bits32: int) -> str:
    v = struct.unpack("<d", struct.pack("<Q", bits32 << 32))[0]
    if math.isnan(v):
        return ("-" if bits32 >> 31 else "+") + ("QNAN" if bits32 >> 19 & 1 else "SNAN")
    return fmt_float(v)


def mod(s: str, neg: int = 0, ab: int = 0) -> str:
    if ab:
        s = f"|{s}|"
    return ("-" if neg else "") + s


@lru_cache(None)
def tables() -> dict:
    p = ROOT / "data" / "tables_sm_80.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ------------------------------------------------------------------ operands


class Ctx:
    """Collects the pieces of one instruction while decoding."""

    def __init__(self, i: I):
        self.i = i
        self.show_reuse = bool(i.yld)
        self.port: dict[int, int | None] = {}

    def ru(self, s: str, slot: int) -> str:
        """append .reuse for operand slot a=0,b=1,c=2"""
        slot = self.port.get(slot, slot)
        if self.show_reuse and slot is not None and (self.i.reuse >> slot) & 1:
            return s + ".reuse"
        return s

    def ra(self, neg_bit=None, abs_bit=None) -> str:
        i = self.i
        s = mod(R(i.f(24, 31)), i.b(neg_bit) if neg_bit else 0, i.b(abs_bit) if abs_bit else 0)
        return self.ru(s, 0)

    def const(self) -> str:
        i = self.i
        off = i.f(40, 53) << 2
        return f"c[{i.f(54, 58):#x}][{hx(off - 0x10000 if off & 0x8000 else off)}]"

    def ureg32(self) -> str:
        return UR(self.i.f(32, 37))


def srcs(c: Ctx, kind: str, nb=(63, 62), nc=(75, 74), three=True):
    """Decode the B and C source operands according to the operand form.

    kind: 'i' integer immediate, 'f' f32 immediate, 'd' f64 immediate,
          'h' packed half immediate.
    nb/nc: (neg_bit, abs_bit) for the B and C register slots (None = none).
    Returns (B, C); C is None for two-source instructions.
    """
    i = c.i
    fm = i.form

    def imm():
        v = i.f(32, 63)
        if kind == "f":
            return f32imm(v)
        if kind == "d":
            return f64imm(v)
        if kind == "h":
            return f"{fp16(v & 0xffff)}, {fp16(v >> 16)}"
        if kind == "u":
            return hx(v)
        return hx(s32(v))

    def m(s, bits):
        neg = i.b(bits[0]) if bits and bits[0] is not None else 0
        ab = i.b(bits[1]) if bits and len(bits) > 1 and bits[1] is not None else 0
        return mod(s, neg, ab)

    rb = lambda: c.ru(m(R(i.f(32, 39)), nb), 1)
    rc_as_b = lambda: c.ru(m(R(i.f(64, 71)), nc), 1)
    rc = lambda: c.ru(m(R(i.f(64, 71)), nc), 2)
    cb = lambda: m(c.const(), nb)
    ub = lambda: m(c.ureg32(), nb)

    if not three:
        return {1: rb, 2: imm, 4: imm, 3: cb, 5: cb, 6: ub, 7: ub}.get(fm, lambda: "?")(), None
    if fm == 1:
        return rb(), rc()
    if fm == 2:
        return rc_as_b(), m(imm(), None) if kind not in "i" else imm()
    if fm == 3:
        return rc_as_b(), cb()
    if fm == 4:
        return imm(), rc()
    if fm == 5:
        return cb(), rc()
    if fm == 6:
        return ub(), rc()
    if fm == 7:
        return rc_as_b(), ub()
    return "?", "?"


def fp16(h: int) -> str:
    v = struct.unpack("<e", struct.pack("<H", h))[0]
    if math.isinf(v):
        return "+INF" if v > 0 else "-INF"
    if math.isnan(v):
        return ("-" if h >> 15 else "+") + ("QNAN" if h >> 9 & 1 else "SNAN")
    if v == 0 and h >> 15:
        return "-0.0"
    return f"{v:.20g}"


# ------------------------------------------------------------------ opcodes

OPS = {}


def op(*codes):
    def deco(fn):
        for c in codes:
            OPS[c] = fn
        return fn
    return deco


RND = ["", ".RM", ".RP", ".RZ"]
FMZ = ["", ".FTZ", ".FMZ", ".INVALID3"]
FCMP = ["F", "LT", "EQ", "LE", "GT", "NE", "GE", "NUM", "NAN", "LTU", "EQU", "LEU",
        "GTU", "NEU", "GEU", "T"]
ICMP = ["F", "LT", "EQ", "LE", "GT", "NE", "GE", "T"]
BOOL = ["AND", "OR", "XOR", "INVALID3"]


@op(0x021)
def _fadd(c: Ctx):
    i = c.i
    b, _ = srcs(c, "f", three=False)
    m = ".FTZ" * i.b(80) + RND[i.f(78, 79)] + ".SAT" * i.b(77)
    return "FADD" + m, [R(i.f(16, 23)), c.ra(72, 73), b]


@op(0x020)
def _fmul(c: Ctx):
    i = c.i
    b, _ = srcs(c, "f", three=False)
    scale = [".INVALID0", ".D8", ".D4", ".D2", "", ".M2", ".M4", ".M8"][i.f(84, 86)]
    m = FMZ[i.b(80) | i.b(76) << 1] + scale + RND[i.f(78, 79)] + ".SAT" * i.b(77)
    return "FMUL" + m, [R(i.f(16, 23)), c.ra(72, 73), b]


@op(0x023)
def _ffma(c: Ctx):
    i = c.i
    b, cc = srcs(c, "f", nb=(63, 62), nc=(75, 74))
    m = FMZ[i.b(80) | i.b(76) << 1] + RND[i.f(78, 79)] + ".SAT" * i.b(77)
    return "FFMA" + m, [R(i.f(16, 23)), c.ra(72, 73), b, cc]


@op(0x00b)
def _fsetp(c: Ctx):
    i = c.i
    b, _ = srcs(c, "f", three=False)
    m = f".{FCMP[i.f(76, 79)]}" + ".FTZ" * i.b(80) + f".{BOOL[i.f(74, 75)]}"
    return "FSETP" + m, [P(i.f(81, 83)), P(i.f(84, 86)), c.ra(72, 73), b,
                         P(i.f(87, 89), i.b(90))]


@op(0x009)
def _fmnmx(c: Ctx):
    i = c.i
    b, _ = srcs(c, "f", three=False)
    m = ".FTZ" * i.b(80) + ".NAN" * i.b(81)
    return "FMNMX" + m, [R(i.f(16, 23)), c.ra(72, 73), b, P(i.f(87, 89), i.b(90))]


@op(0x008)
def _fsel(c: Ctx):
    i = c.i
    b, _ = srcs(c, "f", three=False)
    return "FSEL" + ".FTZ" * i.b(80), [R(i.f(16, 23)), c.ra(72, 73), b,
                                       P(i.f(87, 89), i.b(90))]


@op(0x007)
def _sel(c: Ctx):
    i = c.i
    b, _ = srcs(c, "u", nb=None, three=False)
    return "SEL", [R(i.f(16, 23)), c.ra(), b, P(i.f(87, 89), i.b(90))]


@op(0x002)
def _mov(c: Ctx):
    i = c.i
    b, _ = srcs(c, "u", nb=None, three=False)
    ops = [R(i.f(16, 23)), b]
    if i.f(72, 75) != 0xF:
        ops.append(hx(i.f(72, 75)))
    return "MOV", ops


@op(0x010)
def _iadd3(c: Ctx):
    i = c.i
    b, cc = srcs(c, "i", nb=(63,), nc=(75,))
    ops = [R(i.f(16, 23))]
    for p in (i.f(81, 83), i.f(84, 86)):
        if p != 7:
            ops.append(P(p))
    ops += [c.ra(72), b, cc]
    mn = "IADD3"
    if i.b(74):
        mn += ".X"
        ops = [_tilde(o) for o in ops]
        ops += [P(i.f(87, 89), i.b(90)), P(i.f(77, 79), i.b(80))]
    return mn, ops


@op(0x012)
def _lop3(c: Ctx):
    i = c.i
    b, cc = srcs(c, "u", nb=None, nc=None)
    ops = [R(i.f(16, 23))]
    if i.f(81, 83) != 7:
        ops.insert(0, P(i.f(81, 83)))
    ops += [c.ra(), b, cc, hx(i.f(72, 79)), P(i.f(87, 89), i.b(90))]
    return "LOP3.LUT" + ".PAND" * i.b(80), ops


@op(0x00c)
def _isetp(c: Ctx):
    i = c.i
    b, _ = srcs(c, "i", nb=None, three=False)
    m = f".{ICMP[i.f(76, 78)]}" + ".U32" * (1 - i.b(73)) + f".{BOOL[i.f(74, 75)]}" + ".EX" * i.b(72)
    ops = [P(i.f(81, 83)), P(i.f(84, 86)), c.ra(), b, P(i.f(87, 89), i.b(90))]
    if i.b(72):
        ops.append(P(i.f(68, 70), i.b(71)))
    return "ISETP" + m, ops


def _tilde(o: str) -> str:
    """in carry (.X) mode a negated *register* is printed as bitwise NOT"""
    return "~" + o[1:] if o.startswith("-") and not o.startswith("-0x") else o


def _is_pow2(s: str) -> bool:
    try:
        v = int(s, 16)
    except ValueError:
        return False
    return v > 0 and v & (v - 1) == 0


@op(0x024)
def _imad(c: Ctx):
    i = c.i
    fm = i.form
    # B is never negatable; C's negate bit sits in whichever slot holds C
    b, cc = srcs(c, "i", nb=(63,) if fm in (3, 7) else None,
                 nc=(75,) if fm in (1, 4, 5, 6) else None)
    a = c.ra()  # IMAD has no negate on A
    u32 = ".U32" * (1 - i.b(73))
    ops = [R(i.f(16, 23)), a, b, cc]
    mn = "IMAD"
    if i.b(74):
        mn += u32 + ".X"
        ops = [_tilde(o) for o in ops]
        ops.append(P(i.f(87, 89), i.b(90)))
    elif fm in (1, 2, 3) and "RZ" in (a.split(".")[0], b.split(".")[0]):
        mn += ".MOV" + u32  # a*b == 0: plain move of C
    elif fm == 4 and b == "0x1":
        mn += ".IADD" + u32
    elif fm == 4 and cc == "RZ" and _is_pow2(b):
        mn += ".SHL" + u32
    else:
        mn += u32
    return mn, ops


@op(0x025, 0x027)
def _imad_wide_hi(c: Ctx):
    i = c.i
    fm = i.form  # same negate rules as IMAD: only C can be negated
    b, cc = srcs(c, "i", nb=(63,) if fm in (3, 7) else None,
                 nc=(75,) if fm in (1, 4, 5, 6) else None)
    kind = ".WIDE" if i.opcode == 0x025 else ".HI"
    u32 = ".U32" * (1 - i.b(73))
    ops = [R(i.f(16, 23))]
    if i.f(81, 83) != 7:
        ops.append(P(i.f(81, 83)))
    ops += [c.ra(), b, cc]
    mn = "IMAD" + kind + u32
    if i.b(74):
        mn += ".X"
        ops.append(P(i.f(87, 89), i.b(90)))
    return mn, ops


@op(0x019)
def _shf(c: Ctx):
    i = c.i
    b, cc = srcs(c, "u", nb=None, nc=None)
    m = (".R" if i.b(76) else ".L") + ".W" * i.b(75) + \
        [".S64", ".U64", ".S32", ".U32"][i.f(73, 74)] + ".HI" * i.b(80)
    return "SHF" + m, [R(i.f(16, 23)), c.ra(), b, cc]


@op(0x016)
def _prmt(c: Ctx):
    i = c.i
    b, cc = srcs(c, "u", nb=None, nc=None)
    m = ["", ".F4E", ".B4E", ".RC8", ".ECL", ".ECR", ".RC16", ".INVALID7"][i.f(72, 74)]
    return "PRMT" + m, [R(i.f(16, 23)), c.ra(), b, cc]


@op(0x017)
def _imnmx(c: Ctx):
    i = c.i
    b, _ = srcs(c, "i", nb=None, three=False)
    return "IMNMX" + ".U32" * (1 - i.b(73)), [R(i.f(16, 23)), c.ra(), b,
                                             P(i.f(87, 89), i.b(90))]


@op(0x011)
def _lea(c: Ctx):
    i = c.i
    b, cc = srcs(c, "u", nb=(63,), nc=None)
    hi = i.b(80)
    ops = [R(i.f(16, 23))]
    if i.f(81, 83) != 7:
        ops.append(P(i.f(81, 83)))
    ops += [c.ra(72), b]
    if hi and not i.b(73):
        ops.append(cc)
    ops.append(hx(i.f(75, 79)))
    mn = "LEA" + ".HI" * hi + ".X" * i.b(74) + ".SX32" * i.b(73)
    if i.b(74):
        ops = [_tilde(o) for o in ops]
        ops.append(P(i.f(87, 89), i.b(90)))
    return mn, ops


@op(0x013)
def _iabs(c: Ctx):
    b, _ = srcs(c, "i", nb=None, three=False)
    return "IABS", [R(c.i.f(16, 23)), b]


@op(0x109)
def _popc(c: Ctx):
    i = c.i
    b, _ = srcs(c, "i", nb=(63,), three=False)
    return "POPC", [R(i.f(16, 23)), b.replace("-", "~", 1) if i.b(63) else b]


@op(0x101)
def _brev(c: Ctx):
    b, _ = srcs(c, "i", nb=None, three=False)
    return "BREV", [R(c.i.f(16, 23)), b]


@op(0x100)
def _flo(c: Ctx):
    i = c.i
    b, _ = srcs(c, "i", nb=(63,), three=False)
    if i.b(63):
        b = "~" + b[1:]
    ops = [R(i.f(16, 23))]
    if i.f(81, 83) != 7:
        ops.append(P(i.f(81, 83)))
    ops.append(b)
    return "FLO" + ".U32" * (1 - i.b(73)) + ".SH" * i.b(74), ops


MUFU = ["COS", "SIN", "EX2", "LG2", "RCP", "RSQ", "RCP64H", "RSQ64H", "SQRT", "TANH"]


@op(0x108)
def _mufu(c: Ctx):
    i = c.i
    c.show_reuse = False  # MUFU (XU pipe) has no operand reuse
    b, _ = srcs(c, "f", nb=(63, 62), three=False)
    k = i.f(74, 77)
    name = MUFU[k] if k < len(MUFU) else f"INVALID{k}"
    return "MUFU." + name + ".F16" * i.b(73), [R(i.f(16, 23)), b]


@op(0x119)
def _s2r(c: Ctx):
    i = c.i
    sr = tables().get("sr", {}).get(str(i.f(72, 79)), f"SR{i.f(72, 79)}")
    return "S2R", [R(i.f(16, 23)), sr]


@op(0x1c3)
def _s2ur(c: Ctx):
    i = c.i
    sr = tables().get("sr", {}).get(str(i.f(72, 79)), f"SR{i.f(72, 79)}")
    return "S2UR", [UR(i.f(16, 21)), sr]


@op(0x005)
def _cs2r(c: Ctx):
    i = c.i
    sr = tables().get("sr", {}).get(str(i.f(72, 79)), f"SR{i.f(72, 79)}")
    return "CS2R" + ".32" * (1 - i.b(80)), [R(i.f(16, 23)), sr]


@op(0x1c2)
def _r2ur(c: Ctx):
    i = c.i
    return "R2UR", [UR(i.f(16, 21)), R(i.f(24, 31))]


@op(0x082)
def _umov(c: Ctx):
    i = c.i
    if i.form == 6:
        b = UR(i.f(32, 37))
    else:
        b = hx(s32(i.f(32, 63)))
    return "UMOV", [UR(i.f(16, 21)), b]


@op(0x0b9)
def _uldc(c: Ctx):
    i = c.i
    sz = ["U8", "S8", "U16", "S16", "", "64", "128", "INVALID7"][i.f(73, 75)]
    return "ULDC" + (f".{sz}" if sz else ""), [UR(i.f(16, 21)),
                                             f"c[{i.f(54, 58):#x}][{hx(s16(i.f(38, 53)))}]"]


@op(0x090)
def _uiadd3(c: Ctx):
    i = c.i
    if i.form == 4:
        b = hx(s32(i.f(32, 63)))
    else:
        b = mod(UR(i.f(32, 37)), i.b(63))
    ops = [UR(i.f(16, 21))]
    for p in (i.f(81, 83), i.f(84, 86)):
        if p != 7:
            ops.append(UP(p))
    ops += [mod(UR(i.f(24, 29)), i.b(72)), b, mod(UR(i.f(64, 69)), i.b(75))]
    mn = "UIADD3"
    if i.b(74):
        mn += ".X"
        ops += [UP(i.f(87, 89), i.b(90)), UP(i.f(77, 79), i.b(80))]
    return mn, ops


# ---- fp64 / fp16


@op(0x029)
def _dadd(c: Ctx):
    i = c.i
    b, _ = srcs(c, "d", three=False)
    if i.form == 1:  # DADD reads its second register from bits 64..71
        b = c.ru(mod(R(i.f(64, 71)), i.b(75), i.b(74)), 2)
    return "DADD" + RND[i.f(78, 79)], [R(i.f(16, 23)), c.ra(72, 73), b]


@op(0x028)
def _dmul(c: Ctx):
    i = c.i
    b, _ = srcs(c, "d", three=False)
    return "DMUL" + RND[i.f(78, 79)], [R(i.f(16, 23)), c.ra(72, 73), b]


@op(0x02b)
def _dfma(c: Ctx):
    i = c.i
    b, cc = srcs(c, "d")
    return "DFMA" + RND[i.f(78, 79)], [R(i.f(16, 23)), c.ra(72, 73), b, cc]


@op(0x02a)
def _dsetp(c: Ctx):
    i = c.i
    b, _ = srcs(c, "d", three=False)
    m = f".{FCMP[i.f(76, 79)]}.{BOOL[i.f(74, 75)]}"
    return "DSETP" + m, [P(i.f(81, 83)), P(i.f(84, 86)), c.ra(72, 73), b,
                         P(i.f(87, 89), i.b(90))]


SWZ = ["", ".F32", ".H0_H0", ".H1_H1"]


def _h(c: Ctx, reg: str, slot: int, neg: int, ab: int, swz: int) -> str:
    return c.ru(mod(reg, neg, ab) + SWZ[swz], slot)


@op(0x031, 0x035)
def _hfma2(c: Ctx):
    """packed half FMA. Each operand has neg/abs bits and a swizzle that
    selects halves (.H0_H0, .H1_H1) or a .F32 broadcast."""
    i = c.i
    a = _h(c, R(i.f(24, 31)), 0, i.b(72), i.b(73), i.f(74, 75))
    if i.form == 2:  # C is a pair of f16 immediates: hi half printed first
        b = _h(c, R(i.f(64, 71)), 1, i.b(84), i.b(83), 0)
        cc = f"{fp16(i.f(48, 63))}, {fp16(i.f(32, 47))}"
    else:
        b = _h(c, R(i.f(32, 39)), 1, i.b(63), i.b(62), i.f(60, 61))
        cc = _h(c, R(i.f(64, 71)), 2, i.b(84), i.b(83), i.f(81, 82))
    m = ".MMA" * (i.opcode == 0x035) + ".F32" * i.b(78) + \
        ["", ".FMZ", ".FTZ", ".INVALID3"][i.b(76) | i.b(80) << 1] + ".SAT" * i.b(77) + ".RELU" * i.b(79)
    return "HFMA2" + m, [R(i.f(16, 23)), a, b, cc]


# ---- memory


SIZES = ["U8", "S8", "U16", "S16", "", "64", "128", "INVALID7"]
EVICT = [".EF", "", ".EL", ".LU", ".EU", ".NA", ".INVALID6", ".INVALID7"]
GSEM = ["", ".CONSTANT.PRIVATE", ".CONSTANT.CTA", ".CONSTANT.CTA.PRIVATE", ".CONSTANT",
        ".STRONG.SM", ".STRONG.GPU.PRIVATE", ".STRONG.GPU", ".MMIO.GPU", ".CONSTANT.SM",
        ".STRONG.SYS", ".CONSTANT.SM.PRIVATE", ".MMIO.SYS", ".CONSTANT.VC",
        ".CONSTANT.VC.PRIVATE", ".CONSTANT.GPU"]
LTC = ["", ".LTC64B", ".LTC128B", ".LTC256B"]
SSEM = GSEM[:4] + [".STRONG.SM.PRIVATE"] + GSEM[5:]  # stores: no .CONSTANT at 4


def _addr(c: Ctx, ur_bits=(32, 37)) -> str:
    """[Ra(.64) + URb + imm24] for generic/global memory.
    bit 90: 64-bit address register; bit 76 = 0: add uniform register
    bits 32..37; bit 101: print as desc[URb][...] (memory descriptor)."""
    i = c.i
    parts = []
    ra = R(i.f(24, 31))
    if ra != "RZ" or i.b(76):
        parts.append(ra + (".64" if i.b(90) else "" if i.b(72) else ".U32"))
    if not i.b(76):
        parts.append(UR(i.f(*ur_bits)))
    t = "+".join(parts)
    off = s24(i.f(40, 63))
    if off:
        t = f"{t}+{hx(off)}" if t else hx(off)
    s = f"[{t}]"
    if i.b(101):
        s = f"desc[{UR(i.f(*ur_bits))}]" + s
    return s


def _gsize(i: I) -> str:
    sz = SIZES[(i.f(73, 75))]
    return f".{sz}" if sz else ""


def _emod(i: I) -> str:
    """bit 72: extended (64-bit) addressing; bit 90: 64-bit address register"""
    return ".E" if i.b(72) else (".???0" if i.b(90) else "")


def _gmods(i: I, sem=None) -> str:
    return _emod(i) + EVICT[i.f(84, 86)] + LTC[i.f(68, 69)] + \
        _gsize(i) + (sem or GSEM)[i.f(77, 80)]


def _pz(i: I) -> str | None:
    """optional trailing predicate of LD/LDG: bits 64..66 hold 7-n, 67 negates"""
    v, neg = i.f(64, 66), i.b(67)
    if v == 0:
        return "!PT" if neg else None
    return ("!" if neg else "") + f"P{7 - v}"


@op(0x181, 0x180)
def _ldg(c: Ctx):
    i = c.i
    ops = []
    if i.opcode == 0x181 and i.f(81, 83) != 7:
        ops.append(P(i.f(81, 83)))
    ops += [R(i.f(16, 23)), _addr(c)]
    if _pz(i):
        ops.append(_pz(i))
    return ("LDG" if i.opcode == 0x181 else "LD") + _gmods(i), ops


@op(0x186, 0x185)
def _stg(c: Ctx):
    i = c.i
    name = "STG" if i.opcode == 0x186 else "ST"
    m = _emod(i) + EVICT[i.f(84, 86)] + _gsize(i) + SSEM[i.f(77, 80)]
    return name + m, [_addr(c, (64, 69)), R(i.f(32, 39))]


SCALE = ["", ".X4", ".X8", ".X16"]


def _saddr(c: Ctx, scaled: bool = True) -> str:
    i = c.i
    s = R(i.f(24, 31)) + (SCALE[i.f(78, 79)] if scaled else "")
    ur = i.b(91)
    off = s24(i.f(40, 63))
    parts = []
    if s != "RZ":
        parts.append(s)
    if ur:
        parts.append(UR(i.f(32, 37)))
    t = "+".join(parts)
    if off:
        t = f"{t}+{hx(off)}" if t else hx(off)
    return f"[{t or 'RZ'}]"


@op(0x184)
def _lds(c: Ctx):
    i = c.i
    sz = SIZES[i.f(73, 75)]
    return "LDS" + (f".{sz}" if sz else "") + ".U" * i.b(76), [R(i.f(16, 23)), _saddr(c)]


@op(0x188)
def _sts(c: Ctx):
    i = c.i
    sz = SIZES[i.f(73, 75)]
    return "STS" + (f".{sz}" if sz else ""), [_saddr(c), R(i.f(32, 39))]


@op(0x183)
def _ldl(c: Ctx):
    i = c.i
    sz = SIZES[i.f(73, 75)]
    return "LDL" + EVICT[i.f(84, 86)] + (f".{sz}" if sz else ""), [R(i.f(16, 23)), _saddr(c, False)]


@op(0x187)
def _stl(c: Ctx):
    i = c.i
    sz = SIZES[i.f(73, 75)]
    return "STL" + EVICT[i.f(84, 86)] + (f".{sz}" if sz else ""), [_saddr(c, False), R(i.f(32, 39))]


@op(0x182)
def _ldc(c: Ctx):
    i = c.i
    sz = SIZES[i.f(73, 75)]
    ra = R(i.f(24, 31))
    off = s16(i.f(38, 53))  # byte offset, signed 16-bit
    idx = f"{ra}+{hx(off)}" if ra != "RZ" else hx(off)
    return "LDC" + (f".{sz}" if sz else ""), [R(i.f(16, 23)), f"c[{i.f(54, 58):#x}][{idx}]"]


# ---- control flow


def _target(i: I, addr: int | None) -> str:
    """signed byte offset in bits 32..81, relative to the next instruction"""
    off = i.f(32, 81)
    if off >> 49:
        off -= 1 << 50
    if addr is None:
        return f"{off:+#x}"
    return f"`(.L_{addr + 16 + off:#x})"


@op(0x147)
def _bra(c: Ctx):
    i = c.i
    ops = []
    if i.f(87, 89) != 7 or i.b(90):
        ops.append(P(i.f(87, 89), i.b(90)))
    ops.append(_target(i, c.addr))
    m = ".U" * i.b(96) + ".DIV" * 0
    return "BRA" + m, ops


@op(0x145)
def _bssy(c: Ctx):
    """push a convergence barrier: Bn and the reconvergence address"""
    return "BSSY", [f"B{c.i.f(16, 19)}", _target(c.i, c.addr)]


@op(0x141)
def _bsync(c: Ctx):
    return "BSYNC", [f"B{c.i.f(16, 19)}"]


@op(0x14d)
def _exit(c: Ctx):
    i = c.i
    ops = []
    if i.f(87, 89) != 7 or i.b(90):
        ops.append(P(i.f(87, 89), i.b(90)))
    return "EXIT", ops


@op(0x118)
def _nop(c: Ctx):
    return "NOP", []


@op(0x11d)
def _bar(c: Ctx):
    i = c.i
    mode = {0: ".SYNC", 1: ".ARV", 2: ".RED"}.get(i.f(77, 78), ".?")
    m = mode + ".DEFER_BLOCKING" * (mode == ".SYNC")
    return "BAR" + m, [hx(i.f(54, 57))]


@op(0x146)
def _yield(c: Ctx):
    return "YIELD", []


# ------------------------------------------------------------------ driver

# uniform-datapath opcodes (guarded by uniform predicates UP0..UP6)
UNIFORM = {0x0b9, 0x082, 0x090, 0x099, 0x086, 0x08c, 0x091, 0x092, 0x093, 0x0a4}


# Ops that have an operand-reuse cache (measured: every reuse bit on any
# other opcode is ignored, and on uniform-datapath ops it is illegal).
REUSE_OPS = {0x023, 0x020, 0x021, 0x009, 0x008, 0x00b, 0x02b, 0x028, 0x029, 0x02a,
             0x030, 0x031, 0x035, 0x032, 0x03c, 0x010, 0x024, 0x025, 0x027, 0x017,
             0x00c, 0x011, 0x012, 0x002, 0x016, 0x007, 0x01a, 0x019, 0x013}
# "add" units built on FMA hardware read operand B through the C port
B_ON_C_PORT = {0x021, 0x029, 0x02a, 0x030}


def decode(lo: int, hi: int, addr: int | None = None) -> str:
    i = I(lo, hi)
    c = Ctx(i)
    c.addr = addr
    c.show_reuse = c.show_reuse and i.opcode in REUSE_OPS
    if i.opcode in B_ON_C_PORT:
        c.port = {1: 2}
    fn = OPS.get(i.opcode)
    if fn is None:
        return f"<op {i.opcode:#05x} form {i.form}>"
    mn, ops = fn(c)
    g = i.f(12, 14)
    guard = ""
    if g != 7 or i.b(15):
        guard = "@" + (UP if i.opcode in UNIFORM else P)(g, i.b(15)) + " "
    return guard + mn + ((" " + ", ".join(ops)) if ops else "")


def supported(lo: int) -> bool:
    return (lo & 0x1FF) in OPS


def disasm_cubin(path: str) -> None:
    from .cubin import text_sections
    data = open(path, "rb").read()
    for name, code in text_sections(data):
        print(f"//---- {name}")
        for a in range(0, len(code), 16):
            lo, hi = struct.unpack_from("<QQ", code, a)
            i = I(lo, hi)
            print(f"  /*{a:04x}*/ {i.ctrl()}  {decode(lo, hi, a)}")


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg.startswith("0x") and "_" in arg:
        h, l = arg[2:].split("_")
        print(decode(int(l, 16), int(h, 16)))
    else:
        disasm_cubin(arg)
