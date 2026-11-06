# dcbench — can we beat (or simplify) H-Net's dynamic chunking?

A small, pure-PyTorch (CPU-runnable, no `mamba_ssm`) testbed for comparing
dynamic-chunking (DC) mechanisms inside a 1-stage H-Net-style byte LM, plus a new
mechanism, **Integrate-and-Fire chunking (IF-DC)**.

> **TL;DR (honest).** At this (tiny) scale, IF-DC with selection-forward /
> pooling-backward (`if_last`) has the best mean bits-per-byte of everything
> tried, it hits its compression target exactly without any H-Net machinery
> (no cosine router, no STE, no EMA de-chunking, no ratio loss), and it trains
> ~12% faster in wall-clock than H-Net DC. **But** all differences among
> `if_last`, H-Net DC and a fixed-stride baseline are ≈0.01 bpb, which is the
> same size as seed-to-seed noise; none is statistically significant with the
> seeds we could afford. Nothing here should be read as "beats H-Net" until
> repeated at a scale where H-Net's DC clearly beats fixed-stride (it does not
> here). The useful outcomes are (1) a much simpler DC mechanism that is at
> least on par, (2) a set of negative/ablation results, and (3) the benchmark.

## Mechanisms compared (`chunkers.py`)

| name | down-sampling | up-sampling | router gradient | compression control |
|---|---|---|---|---|
| `fixed` | every N-th byte | repeat | – | exact |
| `hnet` | bytes with `p_t=(1-cos(Wq h_{t-1}, Wk h_t))/2 ≥ .5` | EMA over chunks (rate p) + repeat, × STE(confidence) | STE + EMA | H-Net ratio loss |
| `hnet_noema` | same | repeat, × STE(confidence) | STE only | ratio loss |
| `if` | **IF:** chunk = a-weighted average of the bytes integrated | repeat | through pooling weights | `(mean(a)·N−1)²` |
| `if_cos` | IF with H-Net's cosine feature added to the router logit | repeat | pooling | same |
| **`if_last`** | **IF boundaries; forward = state of the firing byte; backward = pooling weights** | repeat | pooling (surrogate) | same |
| `if_last_frozen` | `if_last` with `a ≡ 1/N` (ablation: gradient trick w/o learned boundaries) | repeat | – | exact |

### Integrate-and-Fire dynamic chunking

Each byte emits a scalar `a_t = σ(w·LN(h_t)+b) ∈ (0,1)`. Weights are
integrated, `C_t = Σ_{s≤t} a_s`, and a chunk is emitted ("fires") whenever `C`
crosses an integer — continuous integrate-and-fire (Dong & Xu, 2020, used for
speech→token alignment), applied here to byte→chunk segmentation. Byte `t`
belongs to chunk `k` with weight `w_{t,k} = |[C_{t−1}, C_t] ∩ [k, k+1]|`
(a byte straddling a crossing is split between two chunks).

* **Exact length control.** #chunks = ⌊Σ a_t⌋, so the compression ratio is
  a direct, differentiable function of `mean(a)`. A one-line quadratic penalty
  keeps it at target (realised ratio 3.96–4.05 for target 4 across all seeds),
  whereas H-Net DC's ratio loss lets the realised ratio drift (3.4–3.8 here),
  i.e. it silently spends ~10–17% more main-network compute.
* **Gradient without STE/EMA.** The pooling weights are piecewise-linear in
  `a`, so the router receives gradient through the down-sampling path itself.
* **Causal by construction.** Chunk `k` is complete at its firing byte, and byte
  `t` only reads chunks that fired at `≤ t` (`test_causal.py` verifies zero
  leakage for every mechanism).
* `if_last` keeps the IF boundaries and the pooling gradient but uses the firing
  byte's encoder state as the chunk vector in the forward pass
  (`z = h_fire + (pool − sg(pool))`). Plain averaging (`if`) turned out to blur
  the most recent context and was clearly worse.

Implementation: ~40 lines, two `scatter_add`s, no custom kernels.

## Setup

* Data: Norvig's `big.txt` (6.5 MB English prose: Gutenberg books + misc.), bytes.
  Val = every 20th 64 KB block (interleaved; same distribution as train).
  Round 1 used the last 5% as val — this turned out to be distribution-shifted
  (end of *War and Peace*), so round-1 numbers are reported separately.
* Model: emb → encoder (2 Transformer layers, d=192, SWA 128) → chunker →
  main (6 Transformer layers, d=320) → de-chunk → + residual → decoder (2 layers)
  → head. ≈10M params; identical for all mechanisms up to the router (<0.1M).
* Training: 1500 steps × 16 × 512 bytes (≈12M bytes), AdamW 2e-3, cosine,
  1 CPU thread per run. Target compression N=4. `aux` weight 0.03 (H-Net
  default) for H-Net; IF uses 0.03×10 on its quadratic.
* Metric: val bits-per-byte on 40×16×512 bytes (fixed sample), realised
  compression L/M, and "word-boundary precision" = fraction of chunk starts that
  directly follow a space/punctuation byte (chance ≈ 0.23).

Reproduce: `python train.py --chunker if_last --data big.txt --seed 1`,
then `python summarize.py`.

## Results (interleaved val, step 1500)

RESULTS_TABLE

Seed-paired comparison (same data order per seed):

PAIRED_TABLE

Round 1 (shifted val split, seed 0, not comparable in absolute terms):
`if` 2.770, `hnet_noema` 2.832 (ratio 3.53), `hnet` 2.864, `fixed` 2.891.
This one-seed "IF beats H-Net by 0.09" did **not** replicate on the clean split
— a good reminder of how noisy single runs at this scale are.

## Findings

1. **No mechanism reliably beats fixed-stride at this scale.** H-Net DC (mean
   of 3 seeds) is within noise of fixed-stride, even with ~10% more main-network
   compute (realised ratio ≈3.6 vs 4). A fixed stride of 3.4 (compute-matched to
   H-Net) is no better than stride 4, so the compression difference is not what
   decides it either. H-Net's paper reports that DC's advantage appears with
   scale/training budget; our ~12M-byte budget is ~4 orders of magnitude smaller.
2. **H-Net DC does learn linguistic boundaries; IF-DC does not (yet).** H-Net's
   chunk starts follow a space/punct. 29–46% of the time (chance 23%); every IF
   variant stays at chance even though `if_last` has the best bpb. The cosine
   router is a strong inductive bias for "word-like" boundaries; adding it to IF
   (`if_cos`) did not help bpb.
3. **`if_last` is a candidate simplification.** It removes: the cosine
   router with identity-initialised Q/K, the STE, the EMA de-chunker (and with it
   the Mamba-2 kernel dependency of `DeChunkLayer`), and the probability/decision
   mismatch of the ratio loss — with equal-or-better bpb, exact compression and
   lower wall-clock (no realised-ratio drift).
4. **Ablations.**
   * Averaging chunk members (`if`) is worse than selecting the firing byte
     (`if_last`): ≈ +0.05 bpb.
   * The pooling-gradient trick with *frozen* uniform boundaries
     (`if_last_frozen`) is on par with plain fixed-stride (seed-paired), so
     whatever edge `if_last` has comes from its learned boundaries, not from the
     surrogate gradient as such.
   * Removing H-Net's EMA de-chunking (`hnet_noema`) costs ≈0.02 bpb (2 seeds)
     and destroys most of the word alignment (0.15–0.28) — the EMA does matter
     for H-Net's router to learn, consistent with the H-Net paper.

## Caveats / what would make this conclusive

* Scale: run at ≥100M params / ≥10B bytes (FineWeb-Edu), where H-Net reports
  DC > fixed; the question is whether `if_last` keeps parity there.
* More seeds (the per-seed SD is ≈0.01 bpb, so ~10 seeds per arm are needed to
  resolve 0.01 differences).
* 2-stage hierarchies, and inference-time behaviour (IF's state is a single
  scalar `frac(C)`, so incremental decoding is trivial).
* The IF router sees only `h_t`; giving it a short look-back (as H-Net's cosine
  does) without losing the gradient path is the obvious next step for getting
  semantic boundaries.

## Related work found

* H-Net++ (arXiv 2508.05628) — Transformer mixer + latent prior; Persian.
* Adaptive Targeted Dynamic Chunking (arXiv 2605.30080).
* ByteFlow (ICLR 2026, arXiv 2603.03583) — coding-rate boundaries, Top-K selection.
* Byte-level chunking for low-resource transfer (arXiv 2608.27658).
* DC-DiT (arXiv 2603.06351) — H-Net-style DC for image diffusion.
* Continuous Integrate-and-Fire (Dong & Xu, ICASSP 2020) — the segmentation
  primitive used here, from speech recognition.
