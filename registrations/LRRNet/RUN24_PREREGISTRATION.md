# Run 24 — pre-registration · confirmation of Run 23's candidate: an improvement without a cost?

**Written 2026-09-28, committed on its own before Run 23 has produced any row.** The candidate is whichever arm Run
23's registered rule selects (`RUN23_PREREGISTRATION.md`); this document is written before anyone knows which, or
whether any. If Run 23 reads "no candidate", Run 24 is not run and this file stays as the record. Nothing below may be
edited after the first Run 24 result; corrections go in a dated block at the end, only before data is read.

---

## Question

Is the tied depth-8 network **with the selected change** better than the published LRRNet configuration on all three
test sets **with no metric against it**, at seeds that played no part in selecting it?

## Design

- **Candidate — C.** Tied, the selected arm's settings, 4 epochs, **new seeds 86–97** (12 runs). Three training
  kernels of four seeds, TNO scored in-kernel. RoadScene (40) and MSRS (361) are fused and scored afterwards from the
  12 checkpoints with the Run 16/21 kernels (`build_run16_kernels.py`, a `--run 24` entry covering both sets).
  - If the GPU quota is exhausted at that point, the fusion runs on Kaggle CPU kernels as in Run 20 (correction 2,
    `e344850`): `fuse_lrrnet.py` with exactly its three `.cuda()` calls removed. **That route is authorised here, in
    advance.**
- **Baseline — P.** The published configuration at **seeds 45–61** (n = 17). Seeds 42–44 are **excluded** because
  Run 23 selected on them. Rows already exist on all three sets, from the same code path:
  - TNO: stage 1 and Runs 9–10;
  - RoadScene: Run 16;
  - MSRS: Run 21.

  No baseline run is repeated.
- **Reference — T.** The unchanged tied depth-8 network, seeds 62–85 (Runs 17 and 20, n = 24). It is used only in
  the secondary test.

≈ 13 GPU-h training + < 1 GPU-h fusion; scoring on CPU.

## The reading, fixed now

**Primary.** C against P on each test set.
- Metrics: the seven that pass LRRNet's positive control — EN, MI, SD, VIF, SCD, Qabf and SSIM.
- Welch's *t*, two-sided 5 %, exact df.
- **Holm across the seven, separately on TNO, RoadScene and MSRS.**
- TOST at ±1.75 pooled σ; sample SD.

| outcome | condition | reported as |
|---|---|---|
| **H1 — IMPROVES, NO COST** | on **each** of the three sets, at least one metric survives Holm in C's favour and **none** survives against it | *the tied network with the change, 2.46× smaller than the published configuration, improves on it on the named metrics on three test sets, at new seeds, with no metric against it* |
| **H2 — TRADE-OFF** | on any set, survivors in both directions | metric by metric, never as an improvement |
| **H3 — PARTIAL** | nothing against C on any set, but on at least one set nothing in its favour | scoped to the sets where it holds |
| **H4 — NOT CONFIRMED** | nothing in C's favour on any set | — |

Whatever the outcome, any metric surviving against C is named beside any gain.

Before H1 is claimed, the person check must pass: TNO pair 02 (Figure 6's box), C seeds 86–88 against P seeds 45–47,
region contrast ≥ 0.9× the published output's, and by eye.

**Secondary (one test, not in the outcome):** on TNO, is C's SD above T's? One-sided Welch, 5 %. This records whether
the change is what moved SD.

## Kill criteria — before any statistic

1. `[CONFIG]` of every run shows the candidate's settings. The model loads with **20,014 parameters**.
2. 4 epochs, exit 0, one checkpoint, and 25 TNO pairs per run. Fewer than **10** candidate seeds → the run is reported
   as incomplete and not read.
3. RoadScene: 40 pairs and MSRS: 361 pairs per checkpoint. Every candidate's TNO row is re-produced within 0.1 %
   relative in the fusion session. More than two failures discard the RoadScene and MSRS parts.
4. No identical rows.

---

## Correction — 2026-09-28, before any Run 23, 23b or 24 row exists

Two changes, both made before any data of these runs exists. Neither loosens the outcome rule.

1. **Where the candidate comes from.** The candidate is the arm selected from the **single pool of Run 23 and Run 23b**
   (`RUN23B_PREREGISTRATION.md`, `048956e`). The selection rule is unchanged: pass rules 1–4, take the largest SD
   difference against P, subject to the person check. If neither screen yields a candidate, Run 24 is not run.
2. **The baseline is narrowed to seeds 50–61 (n = 12).**
   - Seeds 45–49 are removed. Their fused TNO images were read in diagnostic D4 (`3d2f2ee`), which motivated arm B1,
     so they must not also judge it.
   - Seeds 42–44 stay excluded (Run 23/23b select on them).
   - Rows for seeds 50–61 exist on all three sets: TNO (Run 10), RoadScene (Run 16) and MSRS (Run 21).
   - The person check uses published seeds **50–52** (in place of 45–47) against candidate seeds 86–88.

Everything else — new candidate seeds 86–97, Holm across seven per set, outcomes H1–H4, kill criteria — stands.
