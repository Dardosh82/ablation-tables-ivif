# Run 19 — pre-registration · confirming the arm the Run 18 screen selects

**Written 2026-09-26, committed on its own before any Run 18 screen result has been read** (Run 18 S0, the CPU
diagnostic, is read and recorded; none of the six training arms has finished). Nothing below may be edited after
the first Run 19 result; corrections go in a dated block at the end, only before data is read.

---

## What is confirmed

The single arm Run 18's selection rule advances (`RUN18_PREREGISTRATION.md`, "exactly one goes to confirmation").
**If no arm advances, Run 19 is not run**, and that is recorded as its outcome.

## Design — new seeds, and the baseline that already exists for them

- **Candidate:** the selected arm, exactly as configured in Run 18 (same template, same environment variables),
  at seeds **50–61** (12 runs). None of these seeds was used by the screen (42–44).
- **Baseline:** the published configuration (untied, `num_block` = 4) at the **same seeds 50–61**, which already
  exist: TNO rows from Run 10 (`results_run10_nb4a`, `results_run10_nb4b`) and RoadScene rows from Run 16
  (`results_run16/run16_RoadScene.csv`). Same code path, same torch 2.10.0+cu128, same evaluator. **No baseline run
  is repeated.** If the selected arm is the tied S3t, the baseline is still the published untied configuration.
- **RoadScene for the candidate:** its 12 checkpoints are fused on RoadScene's 40 pairs and scored by the same
  evaluator as Run 16, TNO re-scored in the same kernel as the consistency check Run 16 used (0.1 % relative).

≈ 14 GPU-h for training (12 × ~1.15 h), under 1 GPU-h for fusion, scoring on CPU.

## The reading, fixed now

Metrics: the seven that pass LRRNet's positive control — EN, MI, SD, VIF, SCD, Qabf, SSIM. Welch's *t*, two-sided
5 %, at each comparison's own df; **Holm across the seven, separately on TNO and on RoadScene**; TOST at ±1.75
pooled σ. Sample SD.

- **C1 — IMPROVEMENT CONFIRMED:** on **both** test sets at least one selection metric survives Holm in the
  candidate's favour, and on neither test set does any survive against it. Reported as an improvement on the named
  metrics, at 12 seeds, on two test sets.
- **C2 — ONE TEST SET ONLY:** C1's conditions hold on one test set and not the other. Reported as scoped to that
  test set; not called an improvement of the method.
- **C3 — NOT CONFIRMED:** nothing survives Holm in the candidate's favour on either test set. The screen's
  selection is reported as not replicated.
- **C4 — A COST:** any selection metric survives Holm **against** the candidate on either test set, whatever else
  holds. Reported as a measured cost, named, beside any gain.

Additionally, before any claim: the positive-control check (untrained network of the candidate's configuration,
five seeds, must be separated from the trained candidate on the metrics used) and the TNO pair 02 person check on
the candidate's seed-50 output. An improvement that fails either is not claimed.

## Kill criteria — before any statistic

1. `[CONFIG]` of every run shows the selected arm's settings; parameter count as the arm's template.
2. 4 epochs, exit 0, one final checkpoint, 25 TNO pairs scored, per run.
3. RoadScene: 40 pairs per checkpoint; every candidate TNO row re-produced within 0.1 % relative in the fusion kernel.
4. No two runs identical. Fewer than **10** candidate seeds surviving → Run 19 is reported as incomplete, not read.
