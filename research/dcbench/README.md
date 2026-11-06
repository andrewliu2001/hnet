# dcbench — can we beat (or simplify) H-Net's dynamic chunking?

A pure-PyTorch testbed (no `mamba_ssm`) for comparing dynamic-chunking (DC) mechanisms
inside a 1-stage H-Net-style byte LM, with several new mechanisms. Part 2 (GPU, 8×H100,
FineWeb-Edu) supersedes Part 1 (CPU, 6.5 MB of text).

> **Paper:** [`paper/main.pdf`](paper/main.pdf) — *Is the Hierarchy Doing Anything? Main-Network
> Collapse in Hierarchical Byte-Level Language Models* (80 valid GPU runs; final numbers there
> supersede the TL;DR below where they differ). Additional official-code results since the TL;DR:
> random (content-independent) boundaries collapse like fixed ones (3/3, `ctx_gain` ≤ 0.005);
> official H-Net over 9 seeds: 7 healthy (1.243–1.263 bpb), 1 partial (1.350), 1 fully collapsed
> (1.413, `ctx_gain` 0.005); collapse is decided by step 1,000 and never reverses; official
> H-Net + anticipatory: 0/6 collapsed but healthy runs slightly worse (1.258–1.300) at ~18 % more
> main-network compute — pooled collapse 0/26 vs 4/29, Fisher p = 0.11 (not significant).

> **TL;DR.**
> 1. **Main-network collapse decides everything here — and it happens in the official H-Net
>    code too.** In H-Net-style byte LMs the deep main network can collapse to a near-constant
>    output while the cheap encoder → decoder residual path does all the work. We measure this
>    with `ctx_gain` (bpb lost when the main network's output is replaced by its per-sequence
>    mean over chunks). Fixed-stride chunking (and every static / non-learned-router variant)
>    collapsed in **every** run — in our re-implementation and in **3/3 seeds of the official
>    implementation** (`ctx_gain` ≤ 0.0008, Mamba-2 encoder/decoder, official lr multipliers).
>    H-Net's learned router keeps the main network in use most of the time, which in this
>    setup *is* its advantage over fixed-stride: 0.11 bpb in the official code
>    (1.279 vs 1.390), ~0.04 bpb in ours. Collapse is seed-dependent even for H-Net: 3/20 of
>    our H-Net seeds collapsed (`ctx_gain` ≤ 0.05) and 1/3 official H-Net seeds was only
>    partially alive (`ctx_gain` 0.11 vs ≈0.5, and 0.1 bpb worse).
> 2. **New mechanism: anticipatory routing (`hnet_anticip`) — not a significant win.** H-Net's
>    router plus a feature that anticipates difficulty: the standardised predictive entropy of
>    the *next* byte from a detached next-byte probe on the encoder. With **20 seeds per arm**:
>    mean **1.322 vs 1.333** bpb, collapsed seeds **0/20 vs 3/20**, seeds worse than 1.34
>    bpb **3/20 vs 7/20** — all in its favour, but **none significant** (Welch p=0.15,
>    Mann–Whitney p=0.15, paired t p=0.20, Levene p=0.19, Fisher p=0.23). An interim analysis
>    at 6 and 14 seeds looked stronger (Levene p=0.046 at 14 vs 16) and did not survive more
>    seeds — `hnet_anticip` seed 3 is a 1.394-bpb run that is *not* a collapse
>    (`ctx_gain` 0.22). Best supported reading: at most a small (~0.01 bpb) reliability gain.
> 3. **Negative results** (each with ≥2 seeds): integrate-and-fire chunking, equal-information
>    (accumulated surprisal) chunking, pure anticipatory chunking without a learned router,
>    H-Net with *realised* surprisal, static boundaries + H-Net's EMA/STE de-chunking, and
>    three anti-collapse tricks (residual warmup, main-input LayerNorm, lower main lr) — none
>    beats H-Net; the anti-collapse tricks give fixed-stride ~0.02 bpb but do not revive its
>    main network.
> 4. **Caveat on validity.** The first three GPU waves were run with a configuration in
>    which the hierarchy was bypassed (full-attention encoder/decoder + too-high main lr);
>    they are kept in `results/gpu{,2,3}` for the record but excluded from all conclusions.

## Part 2 — GPU study (FineWeb-Edu bytes, H100/A100)

### Setup

* Data: FineWeb-Edu `sample-10BT` shards 0–1 as raw UTF-8 bytes (6.9 GB train, last 1 % of
  documents = 71 MB val), `prepare_fineweb.py`.
* Model (`train_gpu.py`): byte embedding → encoder (4 Transformer layers, d=512,
  **128-byte sliding window**) → chunker → main network (12 layers, d=768, full causal) →
  de-chunk → + residual → decoder (4 layers, d=512, 128-byte window) → head. ≈150M params.
* 9000 steps × 32 × 2048 bytes ≈ 590M bytes per run, AdamW lr 1e-3 (cosine, 1k warmup),
  **main-network lr ×0.3** (cf. H-Net's per-stage lr multipliers), bf16, one run per GPU
  (H100 80GB PCIe). Target compression N=4 for every arm. Same seed ⇒ same data order.
* Official-implementation check (`train_official.py`): the `hnet/` package as released
  (Mamba-2 ×4 encoder/decoder at d=512, Transformer ×12 main at d=768, ≈100M params),
  official `apply_lr_multiplier([2.0, 1.0])` on base lr 5e-4 and the official
  load-balancing loss (α=0.03, N=4); identical data / steps / eval. "fixed" forces the
  official routing module to a boundary every 4 bytes. Two runs shared each GPU
  (seeds 1–2 on H100, seed 3 on A100).
* Metrics: val bpb (100 × 32 × 2048 bytes), realised compression, fraction of chunk
  boundaries on space/punctuation bytes, **`ctx_gain`**, and an inference-time "dial" sweep.
* Runs were launched headless on Hyperstack (`gpu/vm_startup.sh`, `gpu/vm_official.sh`): the
  VM holds no credentials, clones this public repo, and returns results as a tarball via an
  unguessable ntfy.sh topic polled by `gpu/poll_ntfy.sh`.

### Results (valid runs; `python gpu_summary.py`)

| arm | seeds | val bpb (per seed) | mean | ctx_gain (per seed) | main alive | compression | boundaries on space/punct | wall s |
|---|---|---|---|---|---|---|---|---|
| `hnet_anticip` | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 | 1.321, 1.303, 1.394, 1.313, 1.313, 1.308, 1.311, 1.317, 1.324, 1.338, 1.318, 1.348, 1.316, 1.315, 1.307, 1.321, 1.319, 1.304, 1.304, 1.350 | **1.322** | 0.245, 0.199, 0.217, 0.301, 0.195, 0.188, 0.131, 0.130, 0.250, 0.080, 0.196, 0.175, 0.257, 0.148, 0.199, 0.158, 0.214, 0.186, 0.318, 0.125 | 20/20 | 3.56 | 0.15 | 3678 |
| `hnet` | 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 | 1.316, 1.312, 1.304, 1.391, 1.326, 1.310, 1.317, 1.352, 1.345, 1.321, 1.310, 1.358, 1.308, 1.361, 1.371, 1.318, 1.313, 1.364, 1.336, 1.330 | **1.333** | 0.194, 0.127, 0.198, 0.033, 0.118, 0.210, 0.246, 0.050, 0.100, 0.161, 0.163, 0.082, 0.199, 0.045, 0.155, 0.151, 0.155, 0.117, 0.144, 0.261 | 17/20 | 3.64 | 0.20 | 3587 |
| `fixed_main_in_norm1` | 1,2 | 1.355, 1.344 | **1.349** | 0.005, 0.007 | 0/2 | 4.00 | 0.19 | 2864 |
| `fixed_res_warmup3000` | 1,2 | 1.354, 1.347 | **1.350** | 0.018, 0.013 | 0/2 | 4.00 | 0.19 | 2876 |
| `fixed_ema` | 1,2,3 | 1.362, 1.364, 1.364 | **1.363** | 0.002, 0.002, 0.001 | 0/3 | 4.00 | 0.19 | 2942 |
| `fixed_N3.6` | 1,2 | 1.378, 1.355 | **1.366** | 0.003, 0.004 | 0/2 | 3.60 | 0.19 | 3041 |
| `fixed` | 1,2 | 1.381, 1.357 | **1.369** | 0.002, 0.003 | 0/2 | 4.00 | 0.19 | 2873 |
| `fixed_lr_main_mult0.1` | 1,2 | 1.382, 1.357 | **1.370** | 0.002, 0.003 | 0/2 | 4.00 | 0.19 | 2841 |
| `anticip_res_warmup3000` | 1 | 1.373 | **1.373** | 0.009 | 0/1 | 4.01 | 0.66 | 3744 |
| `anticip` | 1,2,3 | 1.382, 1.383, 1.362 | **1.376** | 0.007, 0.008, 0.014 | 0/3 | 3.99 | 0.66 | 3633 |
| `hnet_res_warmup3000` | 4 | 1.405 | **1.405** | 0.083 | 1/1 | 3.93 | 0.23 | 3528 |

* `hnet` mean over alive seeds only: 1.327 (17/20)

Official implementation (`train_official.py`, Mamba-2 encoder/decoder, ~100M params):

| arm | seeds | val bpb (per seed) | mean | ctx_gain (per seed) | compression |
|---|---|---|---|---|---|
| official `fixed` | 1,2,3 | 1.381, 1.389, 1.401 | **1.390** | 0.0006, 0.0008, 0.0007 | 4.00 |
| official `hnet` | 1,2,3 | 1.245, 1.350, 1.243 | **1.279** | 0.5023, 0.1077, 0.5369 | 3.49 |

"Main alive" = `ctx_gain` > 0.05 (an arbitrary threshold; e.g. `hnet_res_warmup3000` seed 4
has 0.083 but still scores poorly).

`hnet` vs `hnet_anticip`, 20 seeds each (seed-paired): mean difference −0.011 bpb
(sd 0.037), paired t p=0.20, Wilcoxon p=0.15; Welch p=0.15; Levene (median) p=0.19;
collapse (`ctx_gain` ≤ 0.05) 3/20 vs 0/20, Fisher p=0.23. Across both arms,
corr(bpb, `ctx_gain`) = −0.52.

### New mechanisms in Part 2 (`chunkers.py`)

| name | boundary rule | router gradient | notes |
|---|---|---|---|
| `eqinfo` | fire when accumulated surprisal `Σ s_t` of a detached next-byte probe reaches a running budget ≈ N·mean(s) | none | exact compression without a ratio loss; inference dial = budget scale (Part 2 wave 2, invalid config, ≈ fixed) |
| `anticip` | boundary at t when the probe's predictive entropy of x_{t+1} exceeds a rate-controlled threshold | none | 66 % of boundaries on space/punct; collapsed like fixed |
| `hnet_surp` | H-Net + γ·(standardised *realised* surprisal of x_t) | STE + EMA | ≈ H-Net (invalid config) |
| `hnet_anticip` | H-Net + γ·(standardised predictive entropy of x_{t+1}) | STE + EMA | 0/20 collapsed vs 3/20 for H-Net; difference not significant |
| `fixed_ema` | every N bytes, but H-Net's EMA de-chunking + STE gate with its learned p | STE + EMA | ≈ fixed: H-Net's gain is not its de-chunker |

All pass `test_causal.py` (an early version of `hnet_surp` standardised surprisal with batch
statistics, which leaks future bytes; the test caught it and running statistics are used).

### Interpretation

* Static or non-learned boundaries (`fixed`, `fixed_N3.6`, `fixed_ema`, `anticip`) all end
  at ≈1.36–1.38 with a dead main network: the 128-byte-window encoder/decoder alone is a
  ≈1.37-bpb byte LM, and the main network adds nothing. The official implementation behaves
  the same way (official fixed: 1.38–1.40 bpb, `ctx_gain` < 0.001), so this is not an
  artefact of our re-implementation.
* H-Net's learned, content-dependent router is what keeps the main network in use. Its
  boundaries sit mostly at word *starts* (only 13–24 % on space/punct), not at word ends.
  The official implementation uses its main network far more when it works (`ctx_gain`
  ≈0.5, 1.24 bpb) than our re-implementation does (`ctx_gain` 0.1–0.3, ≈1.31 bpb).
* bpb tracks main-network use but not perfectly: a few runs are poor for other reasons
  (e.g. `hnet_anticip` seed 3, 1.394 bpb with `ctx_gain` 0.22).
* Practical recommendation regardless of method: **always report `ctx_gain`** (or an
  equivalent hierarchy-bypass check) for hierarchical byte LMs, and run several seeds; a
  bypassed hierarchy looks like a normal, reasonably good LM in bpb alone, and single H-Net
  runs can land 0.1 bpb apart.

### Open questions

* Why does fixed-stride collapse while H-Net mostly does not, given identical capacity? The
  static chunkers never recovered with residual warmup, main-input normalisation or a lower
  main-network lr. Candidate explanations to test: the router's gradient path through the
  encoder; the content-aligned chunk inputs being easier for the main network to exploit early.
* A collapse-prevention method that works for static chunking would make the much simpler and
  faster fixed-stride hierarchy viable; none of ours did.
* Larger scale / longer training and 2-stage hierarchies.

## Part 1 — CPU study (superseded)

The CPU study (12.5M params, Norvig's `big.txt`, 1500 steps) below introduced
Integrate-and-Fire chunking. Its conclusion — all mechanisms within noise of fixed-stride —
is plausibly explained by Part 2: CPU diagnostics (not run on those exact models) showed the
main network barely used at that scale and learning rate.

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
  keeps it at target (`if_last`: realised ratio 3.96–4.05 for target 4 across all seeds),
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
  → head. ≈12.5M params; identical for all mechanisms up to the router (<0.1M).
* Training: 1500 steps × 16 × 512 bytes (≈12M bytes), AdamW 2e-3, cosine,
  1 CPU thread per run. Target compression N=4. `aux` weight 0.03 (H-Net
  default) for H-Net; IF uses 0.03×10 on its quadratic.
* Metric: val bits-per-byte on 40×16×512 bytes (fixed sample), realised
  compression L/M, and "word-boundary precision" = fraction of chunk starts that
  directly follow a space/punctuation byte (chance ≈ 0.23).

Reproduce: `python train.py --chunker if_last --data big.txt --seed 1`,
then `python summarize.py`.

## Results (interleaved val, step 1500)

| split | chunker | seeds | val bpb (each) | mean val bpb | compression L/M | word-boundary precision |
|---|---|---|---|---|---|---|
| interleaved | fixed | 1,2,3,4 | 1.977, 1.965, 1.955, 1.961 | 1.964 | 4.00 | 0.23 |
| interleaved | fixed_N3.4 | 1,2 | 1.973, 1.968 | 1.970 | 3.39 | 0.24 |
| interleaved | hnet | 1,2,3 | 1.974, 1.955, 1.960 | 1.963 | 3.61 | 0.37 |
| interleaved | hnet_noema | 1,2 | 1.993, 1.984 | 1.988 | 3.65 | 0.22 |
| interleaved | if | 1,2 | 2.016, 2.006 | 2.011 | 4.07 | 0.24 |
| interleaved | if_cos | 1 | 2.002 | 2.002 | 4.27 | 0.21 |
| interleaved | if_last | 1,2,3,4 | 1.960, 1.949, 1.961, 1.954 | 1.956 | 4.01 | 0.24 |
| interleaved | if_last_frozen | 1,2 | 1.985, 1.960 | 1.972 | 4.00 | 0.23 |

Wall-clock (1 CPU thread, 1500 steps, seeds 1–3): `fixed` 4771 s, `if_last` 4954 s,
`hnet` 5626 s → IF-DC is ~12% faster than H-Net DC (H-Net's realised ratio
drifts below target, so its main network processes more chunks) and ~4% slower
than fixed-stride.

Seed-paired comparison (same data order per seed):

| seed | fixed | fixed_N3.4 | hnet | hnet_noema | if | if_last | if_last_frozen |
|---|---|---|---|---|---|---|---|
| 1 | 1.977 | 1.973 | 1.974 | 1.993 | 2.016 | 1.960 | 1.985 |
| 2 | 1.965 | 1.968 | 1.955 | 1.984 | 2.006 | 1.949 | 1.960 |
| 3 | 1.955 | – | 1.960 | – | – | 1.961 | – |
| 4 | 1.961 | – | – | – | – | 1.954 | – |

* if_last − fixed over seeds [1, 2, 3, 4]: mean -0.0085, sd 0.0108, t = -1.57
* if_last − hnet  over seeds [1, 2, 3]: mean -0.0068, sd 0.0074, t = -1.58

(t is a paired t-statistic; |t| ≈ 1.6 with 3–4 seeds is **not** significant,
p ≈ 0.2.)

Round 1 (shifted val split, seed 0, not comparable in absolute terms):
`if` 2.770, `hnet_noema` 2.832 (ratio 3.53), `hnet` 2.864, `fixed` 2.891.
This one-seed "IF beats H-Net by 0.09" did **not** replicate on the clean split
— a good reminder of how noisy single runs at this scale are.

## Findings

1. **No mechanism reliably beats fixed-stride at this scale.** H-Net DC (1.963, 3 seeds) is within noise of fixed-stride (1.964, 4 seeds), even with ~10% more main-network
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
