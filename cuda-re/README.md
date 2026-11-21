# cuda-re — reverse-engineering NVIDIA's GPU machine code without a GPU

NVIDIA documents **PTX** (the virtual ISA) but not **SASS**, the real machine
code its GPUs execute. This project recovers the SASS instruction encoding for
**Ampere (sm_80)** by treating NVIDIA's own offline tools as black boxes, and
uses what it learns to build an **independent disassembler**.

No GPU and no system CUDA install are needed. Everything comes from PyPI.

```
CUDA C++ ──NVRTC──▶ cubin ──▶ 128-bit instructions ──▶ our decoder
                                     │                       │
                                     └──▶ nvdisasm (oracle) ──┴──▶ compare
```

## Results

| check | result |
|---|---|
| every instruction the compiler emitted for `kernels/corpus.cu` | **2318 / 2318 exact** (100%) |
| random operand/modifier mutations nvdisasm accepts (never emitted by the compiler) | **21203 / 21375 exact** (99.2%) |
| branch / `BSSY` targets vs. nvdisasm label addresses | **72 / 72** |
| scheduling bits: wait-mask bits with a matching earlier scoreboard set | **179 / 179** |
| opcodes implemented | 52 of the 86 seen (91.8% of corpus instructions) |

"Exact" means character-for-character identical text to `nvdisasm`, after
normalizing whitespace.

## How it works

1. **Corpus** (`sassre/corpus.py`): compile `kernels/corpus.cu` with NVRTC
   (normal, `-G`, `--use_fast_math`), disassemble it with nvdisasm, and keep
   every unique `(encoding, text)` pair: 2528 instructions, 86 opcodes.
2. **Bit-flip fuzzing** (`sassre/fuzz.py`): for one encoding of every
   instruction form, flip each of the 128 bits on its own. Feed the
   mutants to `nvdisasm -b SM80` as a raw binary, then classify what changed:
   opcode, modifier, operand N, guard predicate, `.reuse`, nothing, or
   *illegal*. Output: [`docs/fields_sm_80.md`](docs/fields_sm_80.md).
3. **Table extraction** (`sassre/enum.py`, `sassre/tables.py`): enumerate every
   value of a modifier field. Example: `python -m sassre.enum 0x80c 76 78` prints
   the ISETP comparison table. The special-register names (`SR_TID.X`…) are
   extracted this way into `data/tables_sm_80.json`.
4. **Decoder** (`sassre/disasm.py`): hand-written from those maps. It never
   calls nvdisasm.
5. **Validation** (`sassre/validate.py`): checks the corpus, runs random
   mutation testing, and resolves branch targets.
6. **Scheduling bits** (`sassre/sched.py`): nvdisasm never prints these,
   so they are checked for internal consistency on real kernels instead.

## What we found (sm_80)

### Instruction layout

```
bits   0..8    opcode (unique per mnemonic family; e.g. 0x021 FADD, 0x024 IMAD)
bits   9..11   operand form: 1 R,R | 2 C=imm32 | 3 C=const | 4 B=imm32
                             5 B=const | 6 B=uniform reg | 7 C=uniform reg
bits  12..15   guard predicate (12..14 = P0..P6/PT, 15 = negate)
bits  16..23   Rd       24..31 Ra       32..39 Rb       64..71 Rc
bits  32..63   32-bit immediate (replaces Rb)
bits  40..53   constant-bank offset / 4, bits 54..58 bank  -> c[bank][off]
bits  62/63    |B| / -B      72/73 -A / |A|      74/75 |C| / -C   (float ops)
bits  81..90   predicate destinations / sources
bits 105..108  stall cycles          bit 109 yield
bits 110..112  write scoreboard      113..115 read scoreboard   (7 = none)
bits 116..121  wait mask (6 scoreboards)
bits 122..125  operand-reuse flags (ports a, b, c)
```

### Behaviors we discovered along the way

* **`.reuse` is only printed when bit 109 (yield) is set.** If the warp may
  yield, the operand-reuse cache can be lost, so the flag is meaningless.
  Setting bit 109 together with a stall count ≥ 13 is an illegal encoding.
* **Only the main ALU/FMA datapath has an operand-reuse cache.** Reuse bits are
  ignored on memory, special-function (MUFU), conversion and branch
  instructions. On uniform-datapath instructions (`U*`) they make the
  encoding illegal.
* **`FADD`, `DADD`, `DSETP` and `HADD2` read operand B through the *C* port**:
  their B reuse flag is bit 124 and bit 123 is illegal. This fits "add"
  being run as `a*1 + b` on FMA hardware.
* **`IMAD` cannot negate A or B**, only C. The compiler uses `IMAD` as a general
  move/shift/add, and nvdisasm prints aliases: `IMAD.MOV` (A or B is `RZ`),
  `IMAD.SHL` (B is a power-of-two immediate, C = `RZ`), `IMAD.IADD` (B = 1).
* **In carry (`.X`) mode** a negated register is printed as `~R` (bitwise NOT).
  Immediates keep the `-` sign.
* **Global/generic memory:** a 24-bit signed byte offset in bits 40..63.
  `UR4` (loaded by the `ULDC.64 UR4, c[0x0][0x118]` that starts every kernel)
  is the memory descriptor in bits 32..37. Bit 76 = 0 adds a uniform-register
  offset; stores keep that register in bits 64..69 because 32..39 holds the
  data register. Modifiers print in the order
  `.E` → eviction (`.EF/.EL/.LU/.EU/.NA`) → L2 prefetch (`.LTC64B…`) →
  size → memory semantics (`.CONSTANT`, `.STRONG.GPU`, …).
* **Branch offsets** are signed byte offsets in bits 32..81, relative to the
  *next* instruction.
* **Kernel parameters** start at `c[0x0][0x160]`. The cubin's `.nv.info.<kernel>`
  section records their offsets and sizes, plus `EXIT` locations
  (`python -m sassre.cubin file.cubin`).
* **Only variable-latency instructions set scoreboards.** In compiled code
  those are loads, `S2R`, `MUFU`, FP64 arithmetic, `POPC/FLO/BREV` and shared
  stores. Fixed-latency ALU ops (`IADD3`, `FFMA`, `IMAD` …) never do; they use
  the stall count instead.
* **nvdisasm's float format:** `%.20g` below 1e9, `%.20e` from 1e9 up
  (worked out from 1500 random samples), with `±INF`, `±QNAN`/`±SNAN` (by the
  quiet bit) and `-0.0`.

### Known gaps

Opcodes not implemented yet (they print as `<op 0x... form N>`): atomics,
`SHFL`, `VOTE`, `F2F/F2I/I2F/FRND`, texture/surface, `HMMA`, `LDSM`,
`LDGSTS`, `CALL/RET`, and a few others. Most of the remaining random-test
mismatches are rarely used encodings: `R2UR`'s predicate output, `BAR`'s
thread-count operand, `EXIT.KEEPREFCOUNT`, `HFMA2` immediate-form modifiers,
and `INVALIDn` placeholders. The scheduling-bit layout agrees with published
microbenchmark research and passes the consistency check, but nvdisasm
cannot confirm it directly. Timing needs real hardware.

## Usage

```bash
./setup.sh                                  # NVRTC, ptxas, nvdisasm from PyPI into .venv
.venv/bin/python -m sassre.corpus sm_80     # build the ground-truth corpus
.venv/bin/python -m sassre.fuzz sm_80       # bit-flip field maps -> docs/fields_sm_80.md
.venv/bin/python -m sassre.tables           # extract lookup tables
.venv/bin/python -m sassre.validate         # corpus + random + branch checks
.venv/bin/python -m sassre.sched            # scheduling-bit consistency
.venv/bin/python -m sassre.sched k_mem      # annotated listing with control bits
.venv/bin/python -m sassre.disasm x.cubin   # disassemble with our decoder
.venv/bin/python -m sassre.cubin  x.cubin   # sections + .nv.info attributes
.venv/bin/python -m sassre.enum "ISETP.GE.AND P0, PT, R0, 0x1, PT" 72 78
```

## Next steps

* Run on a real GPU (e.g. a Kaggle notebook) to measure latencies and confirm
  the stall/scoreboard semantics.
* Port the pipeline to sm_86 / sm_89 / sm_90 and diff the encodings.
* Add the missing opcodes, then an assembler: the inverse of `disasm.py`.

## Legal note

This works by observing the output of NVIDIA tools, which are installed from
PyPI under NVIDIA's license. No NVIDIA binaries are committed (`.venv/` is
ignored). The data files hold only encodings of kernels compiled from this
repo's own source and the text the tools printed for them.
