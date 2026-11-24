# Run 27: the volume-controlled Top-K test at eight seeds per arm. Pre-registration

**Written 2026-10-06, committed on its own before any Run 27 kernel exists.**
- Nothing below may change after the first Run 27 output.
- A correction may be added only as a dated block at the end, and only before any metric is read.

## Why

Run 4b (`lrrnet-m3-k1000`, 2026-09-07) held the training corpus fixed at 12,025 pairs (11,025 LLVIP plus 1,000 MSRS
patches). It compared auxiliary patches **chosen at random and unaligned** (`k0`) with patches **selected and
gamma-aligned by Top-K** (`topk`), at three seeds per arm. It detected no stability effect:
- median σ ratio 0.89;
- 0 of 9 variance ratios significant;
- one mean of nine separating (MI, Top-K worse).

At three seeds a variance test can find only very large differences (F(2,2) needs a ratio near 39, two-sided). Claim
#20 therefore restated the result as **non-detection**. Run 27 adds five seeds per arm, so a variance ratio of about 5
is detectable, and so are mean differences of about 1.5–1.75 σ.

**Scope, unchanged and stated again:** this is LRRNet (tied, `num_block = 4`) with MSRS as the auxiliary corpus, not
CrossFuse with M3FD.

## Design

The Run 4b kernel, pulled from Kaggle unchanged except for:
1. the seed list (45–49);
2. one arm per kernel, so that each session stays well under Kaggle's 12-hour cap.

| arm | corpus | seeds | source |
|---|---|---|---|
| `k0` | 11,025 LLVIP + 1,000 random MSRS patches, unaligned | 42–44 | Run 4b rows (existing) |
| `k0` | same | **45–49** | new kernel `lrrnet-run27-k0` |
| `topk` | 11,025 LLVIP + the 1,000 Top-K MSRS patches, gamma-aligned | 42–44 | Run 4b rows (existing) |
| `topk` | same | **45–49** | new kernel `lrrnet-run27-topk` |

Everything else matches Run 4b:
- backbone: tied, `num_block = 4`, 20,014 parameters;
- training: 4 epochs, batch 8, 1,503 batches per epoch;
- scoring: TNO, the same evaluator;
- the same pinned container image.

## Kill criteria (before any metric is read)

1. **The corpora are Run 4b's.** The corpus builder's printed statistics must equal Run 4b's:
   - random-arm mean 59.595;
   - Top-K mean |diff| of kept patches 6.552;
   - gamma 1.0111;
   - auxiliary overlap 192 of 1,000.

   If any differs, the new seeds cannot be pooled with the old, and the run is reported as failed.
2. For each run:
   - `[CONFIG]` shows `tied=1` and the seed;
   - 1,503 batches per epoch and 4 epochs;
   - exit 0, one final checkpoint, 25 fused pairs.
3. No two of the 16 rows are identical.
4. **Minimum n:** if quota ends a kernel early, the reading is made at n ≥ 6 per arm, with the n stated. Below that,
   the run is reported as incomplete and not read.

## The reading (fixed now)

**Primary, the claim at stake (stability).**
- For each of the seven deciding metrics (EN, MI, SD, VIF, SCD, Qabf, SSIM): the variance ratio σ²(k0)/σ²(topk), with
  an F-test, two-sided 5 %, df (n−1, n−1).
- Holm across the seven.

| outcome | condition | what the paper and thesis will say |
|---|---|---|
| **S1: STABILITY EFFECT** | ≥ 1 survives with k0 more variable (Top-K stabilises), none the other way | "At fixed corpus volume, on LRRNet with MSRS, Top-K selection reduced run-to-run spread on [metrics]." The current "no effect detected" wording is withdrawn |
| **S2: NONE DETECTED** | nothing survives | "No stability effect of the mechanism was detected at eight seeds per arm (a variance ratio of about 5 is detectable)." |
| **S3: DESTABILISES** | ≥ 1 survives with Top-K more variable, none the other way | "At fixed volume, Top-K increased run-to-run spread on [metrics]." |
| **S4: MIXED** | survivors in both directions | the metrics and directions are listed, with no single-direction claim |

**Secondary (means), reported under every outcome:**
- Welch's t, two-sided 5 %, Holm across the seven. MI was worse under Top-K at n = 3.
- The median σ ratio across the nine metrics, with a bootstrap 95 % interval, descriptive only.

**What does not change under any outcome:** CrossFuse's own ablation is confounded by the 47 % change in corpus size.
That is a property of its design, not of this run.

## Cost

Ten training runs at about 69 minutes each (Run 4b measured 69.2–69.7 min), in two kernels of five, so about
11.5–12 GPU-hours.
