# Run 25: the positive control at twenty seeds per arm. Pre-registration

**Written 2026-10-06, committed on its own before any Run 25 kernel exists.**
- Nothing below may change after the first Run 25 output.
- A correction may be added only as a dated block at the end, and only before any metric is read.

## Why

Run 6 found that **SF and AG did not separate a randomly weighted LRRNet from a trained one at five seeds per arm**
(|t| = 1.43 and 1.52 against 2.776). The means were far apart (SF 10.91 against 9.02, AG 4.06 against 3.39), but
the untrained arm's spread was large. Claim #20 (2026-10-06) restated this as *non-detection at n = 5*.

Run 25 asks what it is: a metric that cannot resolve the difference at all, or one that needs more runs than an
ablation table uses. It needs **no training**:
- the trained arm is the published configuration at the twenty seeds already trained and scored (seeds 42–61);
- only the untrained arm is new.

## Design

| arm | what | seeds | source |
|---|---|---|---|
| **trained** | published LRRNet, `num_block = 4`, untied, 49,204 parameters, 4 epochs on LLVIP | 42–61 (n = 20) | existing TNO rows: `kernel_csv/lrrnet-stage1` (42–46), `results_run9_nb4` (47–49), `results_run10_nb4a` (50–55), `results_run10_nb4b` (56–61) |
| **untrained** | the same architecture, randomly initialised with `torch.manual_seed(s)`, **zero gradient steps** | 400–419 (n = 20), new and disjoint from every earlier seed | new: GPU kernel `lrrnet-run25-fuse` (construct + fuse TNO), CPU kernel `lrrnet-run25-score` |

Both arms use:
- the same test set: TNO, 25 pairs;
- the same fusion script (`fuse_lrrnet.py`);
- the same evaluator (`eval_fused.py`, CrossFuse's evaluator);
- the same patched repository (`patch_lrrnet.py`, `LRR_TIED=0`).

The untrained weights are each model's own initialisation, as in Run 6.

## Kill criteria (checked before any metric is read)

1. Every untrained model has **49,204 parameters** and `num_block = 4`.
2. **Determinism of the initialisation:**
   - for each seed, a second construction with the same seed is identical, tensor by tensor;
   - two different seeds differ.
3. Every untrained model fuses **25 of 25** TNO pairs.
4. There are 20 untrained and 20 trained rows, with **no two rows identical**.

If any criterion fails, Run 25 is reported as failed and no reading is made.

## The reading (fixed now)

**Primary family:** SF and AG, the two metrics that did not separate at n = 5.
- Untrained against trained: Welch's t, two-sided 5 %, exact degrees of freedom.
- Holm across the two metrics.

| outcome | condition | what the paper and thesis will say |
|---|---|---|
| **P1: SEPARATE AT TWENTY** | both survive Holm | "SF and AG separate an untrained LRRNet from a trained one at twenty seeds per arm but not at five: they resolve even this gross failure only with far more runs than an ablation table reports." |
| **P2: ONE SEPARATES** | exactly one survives | the same sentence for that metric, and the next row's sentence for the other |
| **P3: DO NOT SEPARATE AT TWENTY** | neither survives | "SF and AG did not separate an untrained LRRNet from a trained one even at twenty seeds per arm." |

**Under every outcome:**
- The effect size (pooled σ) and the variance ratio of each metric are reported.
- The **smallest n per arm** at which the observed effect would reach 90 % power is reported. It is computed from
  the measured means and spreads, and labelled exploratory.
- **Secondary, descriptive:** the other seven metrics (EN, MI, SD, VIF, SCD, Qabf, SSIM) under the same test. They
  are expected to separate, as they did at n = 5 (SD only at n = 5). Any that does not is reported, not reinterpreted.

**What does not change under any outcome:**
- the metric–architecture reading. SF and AG are the strongest detectors on CrossFuse, so that comparison is already
  established;
- Run 6's own numbers.

## Cost

- Fuse kernel (GPU, T4): construct 20 models and fuse 25 pairs each. Estimated well under 1 GPU-hour.
- Score kernel: CPU only, no GPU quota.
