# Run 31 — pre-registration · case study I's σ from three seeds to eight

**Written 2026-11-25, committed on its own before any Run 31 kernel exists.** Nothing below may change after the first
result; corrections go in a dated block at the end.

## Why

Every σ-scaled figure of case study I is read against the sample standard deviation of **three** seeds of the
reference configuration (R3, R3b, R3c: `--coeff_tv 5.0 --ssl_schedule paper`, seeds 42–44, 160 epochs, Tier 2). With
two degrees of freedom its 95 % interval runs from 0.52 to 6.28 times the estimate, so the largest figure, Row I's MI
at 11.5 σ, could be 1.8 σ. The paper now says so. This run measures σ properly.

## What is run

Five new seeds, **45–49**, of exactly the R3 configuration: the same `train.py` and flags as R3c's notebook
(`--coeff_tv 5.0 --ssl_schedule paper --seed S`, default 160 epochs), the same training corpus (`RSM3.h5`, 1,718
patches, from the `crossfuse-tier2` notebook), the same code dataset, inference on RoadScene, TNO and MSRS, and the
same evaluator. Each seed has its own output directory and no `--resume` (the defect that once produced one run from
another's weights). The image is the pinned one used since September (Runs 11 and 28); R3–R3c ran on Kaggle's image
of August, which R1 below checks.

Eight seeds rather than ten: n = 8 narrows σ's 95 % interval to 0.66–2.04 times the estimate, matches the complete
LRRNet ablation, and fits one week of the free GPU quota (about 19 GPU-hours).

## The readings, fixed before the run

**R1 — consistency (environment check).** The five new seeds against R3–R3c, Welch at exact df, Holm across nine
metrics, per test set. **CONSISTENT**: nothing survives Holm on TNO. **DRIFT**: anything survives on TNO; then σ is
reported both from all eight and from the five new seeds alone, and the paper states the drift.

**R2 — σ.** σ₈ = sample SD of the eight seeds' TNO means, per metric; it replaces σ₃ in case study I, with its 95 %
interval stated.

**R3 — Row I's MI.** Row I's MI difference (the authors' printed 2.13 vs 2.19, range 0.05–0.07 at two decimals)
in σ₈ units:
- **STANDS**: the lower end of the range is at least 3 σ₈. The paper keeps "MI exceeds the noise floor robustly".
- **NARROWED**: the point value is at least 3 σ₈ but the lower end is not. The paper says the MI difference
  exceeds the noise floor at its printed value but not across its rounding range.
- **WITHDRAWN**: the point value is below 3 σ₈. The paper withdraws the claim that any Row I difference is resolvable,
  and records it among the withdrawn claims.

The other five Row I verdicts (noise / not resolvable at two decimals) are recomputed and reported the same way.

**R4 — the σ-scaled figures of case study I.** The VIF deficits of retraining (now 6.0, 12.4 and 7.7 σ₃) and the
capacity decomposition (capacity term above 3 σ in 14 of 27 cells against 8 for the effect) are recomputed with σ₈
and reported. **STANDS**: the capacity term still exceeds 3 σ in more cells than the effect ("the confound is larger
than the finding" kept). **WITHDRAWN**: it no longer does; the sentence is withdrawn and recorded.

**Kill criteria:** any non-finite score; a run that does not finish its 160 epochs; fewer than five new seeds
scored (then nothing is read, and the run is reported as incomplete).
