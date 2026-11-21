# Shrinking LLM weights without quantization: measured results

The question: can a big model (e.g. 20B parameters, 40 GB in BF16) be made small
enough for a ~10 GB GPU **without quantizing**? Measured on Qwen2.5-0.5B
(494M parameters, BF16), on CPU.

## 1. Lossless compression (`lossless.py`): output is bit-for-bit identical

| BF16 field | bits stored | real information (entropy) |
|---|---|---|
| sign | 1 | 1.00 |
| exponent | 8 | **2.64** (218 of 256 values never used) |
| mantissa | 7 | 6.96 |
| **total** | **16** | **10.60 → 66.2% is the lossless floor** |

Weights cluster tightly around 2^-5 … 2^-9, so almost all of the waste is in
the exponent. The mantissa looks like random noise and does not compress.

| codec | size vs. original |
|---|---|
| zstd -19 on raw bytes | 76.5% |
| zstd -19 on split high/low byte planes | 66.8% |
| Huffman on exponent, raw sign + mantissa | 66.7% |
| same, GPU-friendly: 24-symbol codebook + escape, codes ≤ 16 bits | **66.7%**, round trip verified **bit-exact** |

**Takeaway:** lossless compression gets BF16 to about **2/3 of its size and no
further**. 20B: 40 GB → ~27 GB.

## 2. Lossy, but not quantization (`lossy.py`)

Methods were applied to every linear layer in the transformer blocks (72% of
parameters). None were retrained. Quality is WikiText-2 perplexity over 20,480
tokens; lower is better.

| method | weights removed | perplexity | vs. original |
|---|---|---|---|
| original | 0% | 16.52 | 1.00× |
| magnitude pruning 25% | 25% | 22.09 | 1.34× |
| **Wanda 25%** | 25% | **17.17** | **1.04×** |
| magnitude pruning 50% | 50% | 521.14 | 31.5× |
| Wanda 50% | 50% | 27.76 | 1.68× |
| Wanda 2:4 (tensor-core sparse pattern) | 50% | 87.25 | 5.3× |
| low-rank SVD, 75% of params kept | 25% | 44,377 | destroyed |
| low-rank SVD, 50% of params kept | 50% | 166,220 | destroyed |

What the numbers show:

* **The choice of which weights to drop decides everything.** Dropping by
  |weight| alone wrecks the model at 50%. Dropping by |weight| × input-activation
  size (Wanda, 22 s of calibration) loses only 4% at 25%.
* **Trained weight matrices are not low-rank.** Plain SVD truncation destroys
  the model, so low-rank only works with retraining (as in LoRA-style
  fine-tunes).
* **The 2:4 pattern costs much more than free-form 50% pruning** on a small
  model. It is the only pattern current GPUs accelerate, though.
* This is a 0.5B model. Published results (SparseGPT, Wanda) show that larger
  models tolerate pruning much better, so 50% on a 20B+ model is typically
  far closer to the original than 1.68×. Treat these numbers as a
  pessimistic lower bound.

## 3. Putting it together for a 20B model (40 GB BF16)

| recipe | bits/weight | 20B size | quality |
|---|---|---|---|
| original BF16 | 16 | 40 GB | 1.00× |
| lossless only | 10.7 | ~27 GB | identical |
| Wanda 25% + lossless (+1-bit mask) | ~9.0 | ~22 GB | ~1.04× |
| Wanda 50% + lossless (+1-bit mask) | ~6.3 | ~16 GB | ≤1.68× (better at 20B) |

**Conclusion:** without quantization, a 20B model cannot honestly be made to
fit *entirely* in 10 GB. The best near-lossless result is about 22 GB. To run
it on 10 GB, keep about 8 GB of it on the GPU and **offload the rest** to system RAM.
Compression still pays off there: every byte saved is a byte not copied over
PCIe, so a model that is 45% smaller streams about 1.8× faster per token. For
Mixture-of-Experts models (gpt-oss-20b, Mixtral), only the active experts move
per token, which makes this practical at interactive speed.

## Next steps

1. Repeat on a larger model (1.5B / 7B) to confirm pruning gets cheaper with size.
2. Write a GPU kernel that decodes the Huffman exponents on the fly inside
   the matrix multiply, so compressed weights never decompress in VRAM.
3. Build the offload loop (prefetch layer N+1 while computing N) on top of
   the compressed format, and benchmark it on a real GPU (Kaggle T4 capped to 10 GB).

Reproduce: `python lossless.py` (≈1 min) and `python lossy.py` (≈10 min on 4 CPUs).
