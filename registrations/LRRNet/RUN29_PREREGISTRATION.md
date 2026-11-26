# Run 29 — pre-registration · do the headline results hold over test images as well as over runs?

**Written 2026-11-25, committed on its own before any checkpoint is re-scored per image and before any bootstrap is
computed.** Nothing below may change after the first per-image row exists.

## Why

Every test in the paper treats a run's mean over a fixed set of test pairs as the unit, so σ describes retraining,
not the sampling of test images. The limitations section says so and names the analysis that would bound both: a
nested resampling of runs and images. This run does it. The question is whether each headline result generalises
to new test images of the same kind, or holds only over retraining on these particular images.

It needs per-image scores. CrossFuse's were recorded; LRRNet's were not (the evaluator wrote one mean per run), so
the LRRNet checkpoints already trained are re-fused and re-scored with the per-image rows kept. **Nothing is
retrained.**

## What is run

**Re-scoring (Kaggle; fusion on GPU, scoring on CPU, the same scripts as Runs 16, 17b, 20 and 25).**
`eval_fused.py` gains a `--per-image` option that also writes one row per image; the mean row it already writes is
unchanged, and it is the integrity check below.

| family | comparison | checkpoints | test sets |
|---|---|---|---|
| **F1** (Run 17/17b) | tied d8 vs untied d8 | seeds 62–73 each (`lrrnet-run17-ckpts`, 24) | TNO, RoadScene |
| **F2** (Run 20) | tied d8 (seeds 74–85, `lrrnet-run20-ckpts`) vs the published d4 at seeds 42–49 (`lrrnet-run16-ckpts`) | 12 + 8 | TNO, RoadScene |
| **F3** (Table IV, Run 10/16) | `nb` = 2, 4, 6 at seeds 42–61 (`lrrnet-run16-ckpts`) | 60 | TNO, RoadScene |
| **F4** (Run 25) | trained `nb` = 4, seeds 42–61, vs untrained, `torch.manual_seed(s)` for s = 400–419, zero steps | 20 + 20 | TNO |

**CrossFuse (no new run).** **F5**: the positive control's per-image TNO tables already recorded (trained PC/A1
seeds and untrained U42–U46).

MSRS is not included: its 361 pairs are nine times RoadScene's, and the scoped MSRS claims are left as they stand.

## Integrity check (stop rule)

For every re-scored checkpoint, the mean of its new per-image rows must reproduce the run mean already on record to
six decimals on every metric (the same check Run 16 passed for 702 means). **If any checkpoint fails, nothing is
read**: the failure is reported and the run is investigated before any bootstrap is computed.

## The analysis, fixed now

Hierarchical (two-level) bootstrap, B = 10,000, random seed 20261125:

1. resample runs with replacement within each arm (preserving each arm's n);
2. resample test images with replacement, **the same resampled image set for both arms** (every run scores the same
   images, so the comparison stays paired over images);
3. statistic: difference of arm means (mean over the resampled images, then over the resampled runs), in the
   metric's own units, and in pooled run-level σ of the original data;
4. 95 % percentile interval.

Also reported, descriptively: the share of the variance of an arm's mean due to images versus runs (from the two
single-level bootstraps), per family and test set.

## The readings, fixed before the run

For every cell the paper reports as significant after Holm (F1: SF, AG, VIF, SCD on TNO and RoadScene; F2: TNO VIF,
Q^AB/F, SSIM and SD, RoadScene EN, VIF, Q^AB/F, SSIM; F3: RoadScene 2 vs 4 on EN and SCD, 6 vs 4 on VIF and SCD;
F4: the seven LRRNet metrics that pass; F5: all nine CrossFuse metrics, and the MI and VIF inversion):

- **HOLDS OVER IMAGES** — the 95 % hierarchical interval excludes zero in the reported direction. The paper may say
  the result holds over resampling of both runs and test images.
- **RUNS ONLY** — the interval includes zero. The paper keeps the result but scopes it: it holds over retraining on
  these test images, and is not shown to hold over test images. This is reported per cell, by name.

For the nulls (F3 on TNO; F4's SF and AG): an interval that includes zero changes nothing; one that excludes zero is
reported as exploratory and changes no verdict, because the registered tests were run-level.

**Kill criterion:** none on the outcome; any outcome is reported. The integrity check above is the only stop rule.

**Stated limitation, fixed now:** RoadScene and TNO pairs are resampled as independent images. Where several pairs
come from one video sequence they are correlated, so the image-level interval can understate the true width; any
cell read as HOLDS OVER IMAGES on RoadScene is reported with that caveat.

## Correction — 2026-11-25, after the integrity check and before any F1–F4 value is read

The stop rule fired, so it was investigated before anything else, as written above. **236 of 248 re-scored
checkpoints reproduce the record exactly** (zero difference on every metric: all of Runs 16, 17b and 25). The other
12 are Run 20's, on both test sets, and differ by at most 5.3 × 10⁻⁵ in absolute value (2.4 × 10⁻⁵ relative). The
cause is in Run 20's own registration (correction of 2026-09-27): its recorded rows were scored from images fused on
**CPU**, while the images re-scored here are the later GPU fuse kernel's, so they are not byte-identical. The size of
the difference is three orders of magnitude below any σ in the paper.

**Decision, fixed now:** F1, F3 and F4 are read exactly as registered (their sources pass the six-decimal check).
F2 is read with the GPU-fused images, labelled as such wherever it is reported, and the integrity check for Run 20
alone is relaxed to a relative tolerance of 10⁻⁴, which the 12 checkpoints meet. Nothing else in the registration
changes. If the relaxed check had failed, F2 would not have been read.
