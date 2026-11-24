# Run 17b — pre-registration · Run 17's checkpoints on a second test set (RoadScene), no training

**Written 2026-09-26 ~16:45 UTC, committed on its own before any Run 17 checkpoint has been fused or scored on
RoadScene.** Run 17's TNO result (L1) is known and recorded, and so is an exploratory TNO comparison of the tied
depth-8 network against the published configuration (`kernel_csv/run17_exploratory_vs_published.txt`) — which is why
both questions below are read on **RoadScene only**, where no number for these checkpoints exists yet.

---

## What is run

The 24 Run 17 checkpoints — tied `num_block` = 8 (20,014 parameters) and untied `num_block` = 8 (88,124), seeds 62–73 —
fused on RoadScene's 40 pairs and scored by the same evaluator, with the same kernels as Run 16 (GPU fusion, CPU
scoring). The published configuration's RoadScene rows at seeds 50–61 already exist (Run 16) and are reused, not rerun.
As in Run 16, every checkpoint is also re-fused on TNO and must reproduce its stored TNO row within 0.1 % relative; more
than two failures discard the run.

## The readings, fixed now

Welch's *t*, two-sided 5 %, exact df; Holm within each family; TOST at ±1.75 pooled σ; sample SD.

**Q1 — does Run 17's result hold on RoadScene?** Tied d8 against untied d8, seeds 62–73, the four registered metrics
SF, AG, VIF, SCD, Holm across four.
- **Q1-HOLDS:** at least one survives Holm favouring tied, none favouring untied.
- **Q1-TNO-ONLY:** none survives Holm in either direction — Run 17's result is scoped to TNO.
- **Q1-REVERSED:** any survives favouring untied.

**Q2 — is the tied depth-8 network better than the published configuration?** Tied d8 (seeds 62–73) against the
published untied d4 (seeds 50–61, Run 16 rows), the seven metrics that pass LRRNet's positive control (EN, MI, SD, VIF,
SCD, Qabf, SSIM), Holm across seven.
- **Q2-BETTER:** at least one survives Holm favouring tied d8, **and none favouring the published model**.
- **Q2-MIXED:** survivors in both directions — reported metric by metric, named as a trade-off, never as an
  improvement. (The exploratory TNO reading was of this kind: VIF and SSIM for, SD against.)
- **Q2-NULL:** none survives in either direction — no difference found at 2.46× fewer parameters.
- **Q2-WORSE:** survivors only favouring the published model.

Q2 is a statement about the tied depth-8 network on RoadScene; a claim that it improves on the published model in
general needs both test sets and new seeds, which this run does not provide.

## Kill criteria

1. All 24 checkpoints load with the expected parameter count and fuse 40 RoadScene + 25 TNO pairs.
2. TNO re-score within 0.1 % relative of the stored row for all but at most two; otherwise the run is discarded.
3. No two checkpoints give identical rows.
