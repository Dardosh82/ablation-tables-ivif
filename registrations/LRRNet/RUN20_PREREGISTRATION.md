# Run 20 — pre-registration · a general test of the tied depth-8 network against the published configuration

**Written 2026-09-26 ~20:50 UTC, committed on its own before any Run 20 script or kernel exists.** Nothing below may
be edited after the first result; corrections go in a dated block at the end, only before data is read.

---

## Why this run exists

Runs 17 and 17b found that the weight-tied network at depth 8 (20,014 parameters) beats the untied one at depth 8 on two
test sets, and beats the **published** configuration (untied, depth 4, 49,204 parameters) on RoadScene on six of seven
control-passing metrics. But the published-model comparison was registered on RoadScene only, because its TNO version
had already been seen (and there it was a trade-off: VIF and SSIM up, SD down). A claim that the tied depth-8 network
**improves on the published model** needs new seeds of the candidate, a baseline that has not entered this comparison,
and both test sets read together under one registered rule. This run is that test. It can fail: the TNO trade-off may
be what is real.

## Design

- **Candidate:** tied LRRNet, `num_block` = 8, 4 epochs, the M1 depth-8 template (`lrrnet-m1-nb8`), **new seeds
  74–85** (12 runs), exactly as Run 17's tied arm otherwise. Fused and scored on TNO in the training kernel; the 12
  checkpoints then fused and scored on RoadScene by the Run 16/17b kernels (`build_run16_kernels.py`, a `--run 20`
  entry), with the same 0.1 % TNO re-score consistency check.
- **Baseline:** the published configuration at **seeds 42–49** (n = 8), whose TNO rows (stage 1, Run 9) and RoadScene
  rows (Run 16) already exist, from the same code path and torch 2.10.0+cu128. These seeds have **not entered any**
  comparison with the tied depth-8 network (Run 17b's Q2 used seeds 50–61; the exploratory TNO comparison used 50–61).
  No baseline run is repeated.

≈ 13 GPU-h (12 × ~1.07 h) + < 1 GPU-h fusion; scoring on CPU.

## The reading, fixed now

The seven metrics that pass LRRNet's positive control: EN, MI, SD, VIF, SCD, Qabf, SSIM. Welch's *t*, two-sided 5 %,
exact df; **Holm across the seven, separately on TNO and on RoadScene**; TOST at ±1.75 pooled σ; sample SD.

- **G1 — IMPROVES:** on **both** test sets at least one metric survives Holm in the candidate's favour and **none**
  survives against it. Reported as: *the tied depth-8 network, 2.46× smaller, improves on the published configuration on
  the named metrics, on two test sets, at new seeds.*
- **G2 — TRADE-OFF:** on either test set, survivors in both directions (the TNO pattern already seen). Reported metric
  by metric as a trade-off, never as an improvement.
- **G3 — ONE SET:** the G1 conditions hold on one test set and on the other nothing survives in either direction.
  Reported as scoped to that test set.
- **G4 — NOT CONFIRMED:** nothing survives in the candidate's favour on either set.

Any metric surviving against the candidate is named beside any gain, whatever the outcome. Before G1 is claimed: the
person check on TNO pair 02 (Figure 6's box; candidate seeds 74–76 against baseline seeds 42–44, contrast ratio and by
eye) must show the infrared target kept.

## Kill criteria — before any statistic

1. `[CONFIG]` of every run: `epochs=4 num_block=8 tied=1`; 20,014 parameters at load.
2. 4 epochs, exit 0, one checkpoint and 25 TNO pairs per run.
3. RoadScene: 40 pairs per checkpoint; every candidate TNO row re-produced within 0.1 % relative (more than two
   failures discard the RoadScene part).
4. No identical rows. Fewer than **10** candidate seeds → reported as incomplete, not read.

---

## Correction — 2026-09-27 ~05:40 UTC, before any Run 20 checkpoint has been fused or scored on RoadScene

**Execution only; no reading changes.** The weekly Kaggle GPU quota ran out after the 12 candidate runs, so the RoadScene
fusion kernel could not start. The 12 checkpoints are instead fused **locally on CPU** with `fuse_lrrnet.py` unchanged
except that its three `.cuda()` calls are removed (the `args.cuda` save path is kept, so images are written exactly as on
Kaggle), and scored locally with the same `eval_fused.py` and CrossFuse `Evaluator`. The registered kill criterion 3 is
what licenses this: every candidate must reproduce its stored TNO row within 0.1 % relative in the same local session,
and more than two failures discard the RoadScene part. The Kaggle score kernel pushed before the quota refusal was seen
has no fused input and is not used.

## Correction 2 — 2026-09-28, before any Run 20 RoadScene row has been read

**Execution only; no reading changes.** The local CPU session was interrupted twice (a session restart, then a stop
requested by the author) after writing 10 of the 24 row files; **none of them was read**. For a run that does not depend
on the author's machine staying awake, all 12 checkpoints are fused and scored on **Kaggle CPU kernels** (no GPU quota),
three kernels of four checkpoints, with the same patched LRRNet code, `fuse_lrrnet.py` with its three `.cuda()` calls
removed, and the same `eval_fused.py`. **The local partial rows are discarded unread.** Kill criterion 3 (TNO re-score
within 0.1 % of the stored rows) applies to the Kaggle CPU rows exactly as registered.
