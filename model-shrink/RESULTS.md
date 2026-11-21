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
* This is a 0.5B model. Larger models tolerate pruning much better, as
  measured below.

### 2b. Pruning vs. model size (Kaggle T4 GPU, `kaggle/prune_eval.py`)

Same method, same 20,480 WikiText-2 tokens, FP16 on GPU. Each cell is
perplexity relative to that model's own unpruned baseline.

| method | Qwen2.5-0.5B | Qwen2.5-1.5B | **Qwen2.5-7B** |
|---|---|---|---|
| baseline perplexity | 16.52 | 11.54 | 7.77 |
| magnitude 25% | 1.34× | 1.25× | 1.64× |
| **Wanda 25%** | 1.04× | 1.04× | **1.04×** |
| magnitude 50% | 31.8× | 8.0× | 86× |
| **Wanda 50%** | 1.68× | 1.52× | **1.31×** |
| **Wanda 2:4** | 5.37× | 3.91× | **2.24×** |

**The cost of 50% pruning drops steadily with model size** (1.68 → 1.52 → 1.31).
At 7B, half the weights can be removed with no retraining, and the pruned
model (perplexity 10.2) is still better than the *unpruned* 1.5B model (11.5).
Plain magnitude pruning, by contrast, gets *less* reliable at scale: large
models have a few huge activation outliers that the |w| × ‖x‖ score accounts for.

## 3. Running a model 4× bigger than the GPU budget (Kaggle T4, `kaggle/offload_bench.py`)

Qwen2.5-7B (14.2 GB in FP16) with the GPU limited to **25% of the model size
(3.55 GB)**. That is the same ratio as a 20B model (40 GB) on a 10 GB GPU.
Two of the 28 decoder layers stay on the GPU. The other 26 are streamed from
pinned system RAM through two GPU slots: layer N+1 is copied in while layer N
computes. The embedding table stays in RAM, and only the looked-up rows are sent.

Compressed mode keeps each streamed layer in RAM as one 12-bit lossless blob:
a 4-bit code for the 15 most common exponents plus the sign/mantissa byte,
with rare exponents stored exactly in a side list. A fused Triton kernel
decodes it on the GPU in a single pass.

| engine | PCIe per token | time per step | tokens/s, batch 1 | tokens/s, batch 8 | peak GPU |
|---|---|---|---|---|---|
| Hugging Face `accelerate` offload (standard) | 11.3 GB | 2.97 s | 0.34 | 2.74 | 3.37 GB |
| **ours: streaming + prefetch, FP16** | 11.29 GB | 1.055 s | 0.95 | 7.56 | 2.80 GB |
| **ours: streaming + prefetch, lossless 12-bit** | **8.47 GB** | **0.907 s** | **1.10** | **8.81** | 3.46 GB |

* **3.3× faster than the standard tool, at the same memory budget.**
* **Lossless is verified end to end:** compressed and FP16 streaming produce
  *identical* tokens, checked on both 0.5B and 7B.
* **The bottleneck is purely the PCIe link.** FP16 streaming hits
  11.29 GB ÷ 11.3 GB/s = 1.0 s per step almost exactly; prefetch hides all
  compute. Compression cuts the bytes by 25% and the time by 14%; the gap is
  decode overhead plus a slightly slower PCIe reading during that run.
* **Batching is nearly free**: one pass over the weights serves the whole
  batch, so batch 8 runs at the same step time with 8× the throughput.
* Getting here took three rounds. Plain PyTorch decode ops were 3× *slower*
  than no compression; a fused kernel recovered most of that; packing each
  layer into one buffer (one copy and one kernel per layer, instead of about 100
  small ops) made compression a net win.

### Projection: 20B model on a 10 GB GPU (no quantization)

About 32 GB has to stream per token (24 GB with the lossless format):

| PCIe link | FP16 streaming | lossless 12-bit | + Wanda 25% (~9 bits, 1.04× ppl) |
|---|---|---|---|
| gen3 x16 (~11 GB/s, like the T4) | ~2.9 s/token | ~2.2 s/token | ~1.6 s/token |
| gen4 x16 (~25 GB/s, RTX 30/40) | ~1.3 s/token | ~1.0 s/token | ~0.7 s/token |

These are per decoding step, so batch 8 gives about 8× the tokens/s. System RAM
must hold the streamed part: ~24 GB in the lossless format.

## 4. Putting it together for a 20B model (40 GB BF16)

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

1. **Stream pruned weights.** Store only the kept weights (bitmap + lossless
   values) so Wanda-pruned layers move ~45% fewer bytes. This is the biggest
   remaining speed-up for a dense model.
2. **Mixture-of-Experts.** Stream only the experts a token routes to
   (gpt-oss-20b moves ~3.6B of 21B parameters per token), so 10–20× less data.
3. Speculative decoding: one pass over the weights verifies several draft
   tokens, which multiplies tokens per step.
4. Move compression to the GPU at load time (it took 4 min on Kaggle's CPU for 7B).

Reproduce: `python lossless.py` (≈1 min) and `python lossy.py` (≈10 min on 4 CPUs)
locally; on a GPU machine `python kaggle/prune_eval.py Qwen/Qwen2.5-7B` and
`python kaggle/offload_bench.py Qwen/Qwen2.5-7B 0.25 comp 1 8`. Logs are in `kaggle/logs/`.
