# Run 16 — pre-registration · the LRRNet verdicts on a second test set (RoadScene), no training

**Written 2026-09-25, committed on its own before any Run 16 script, dataset or kernel exists.** Nothing below may
be edited after the first result; corrections go in a dated block at the end.

---

## Why this run exists

Every LRRNet verdict in the paper is measured on **25 TNO pairs** — the test set LRRNet's own paper uses. A
reviewer will ask whether the nulls are properties of the methods or of one small test set. The trained
checkpoints of every LRRNet run survive in the kernel outputs, so the question costs **no training**: fuse a
second test set with the same checkpoints and read the same comparisons.

**The second test set is RoadScene (40 pairs)**, the one CrossFuse's evaluation already uses in this project,
chosen before any LRRNet output on it exists. MSRS (361 pairs) is not used: at 361 pairs the evaluator's cost
per checkpoint is nine times RoadScene's, and nothing about the question needs it.

**This run can weaken the paper.** If differences appear on RoadScene that TNO did not show, the nulls are
test-set specific and the paper must say so in its scope sentences.

## What is run

The checkpoints already trained — **nothing is retrained** — fused on RoadScene's 40 pairs with the same fusion
script (`fuse_lrrnet.py`) and scored with the same evaluator (CrossFuse's `Evaluator`, `eval_fused.py`) as TNO.

| family | arms | checkpoints |
|---|---|---|
| **T — Table IV** | untied `nb` = 2, 4, 6 at 20 seeds (42–61); `nb` = 8 at 8 seeds (42–49) | 68 |
| **M — M1** | tied `nb` = 4 at 5 seeds (42–46) against untied `nb` = 4 at the same 5 seeds | 5 (+5 shared with T) |
| **D — free depth** | tied `nb` = 8 at 5 seeds (42–46) against tied `nb` = 4 | 5 |

78 checkpoints in all.

---

## The readings, fixed before the run

Welch's *t*, two-sided 5 %, at each comparison's own degrees of freedom; Holm within each family of nine;
TOST at ±1.75 pooled σ. The TNO verdicts are already in the record; RoadScene's are read against them.

**Family T, the paper's LRRNet headline.** The 18 comparisons of `nb` = 2 and 6 against 4 at twenty seeds, and
the 9 of `nb` = 8 against 4 at eight.
- **T1 — AGREES**: none of the 27 survives Holm, and the 18 twenty-seed comparisons are equivalent within ±1.75 σ
  in at least 15 of 18 (TNO: 18 of 18). Reported as: the Table IV null holds on a second test set.
- **T2 — DISAGREES**: at least one of the 27 survives Holm on RoadScene. Reported as: on RoadScene the block count
  *does* separate settings on the named metrics; the paper's LRRNet statements are scoped to TNO, and the
  direction of every surviving cell is reported, including whether it favours the published choice of 4.
- **T3 — WEAKER**: nothing survives Holm but fewer than 15 of 18 are equivalent. Reported as: no difference is
  found, but RoadScene resolves less, and the equivalence claim is scoped to TNO.

**Family M (M1).** **M-agrees**: no metric survives Holm (TNO: none). **M-differs**: at least one survives Holm;
reported by direction — against the tied model it is a measured cost of the reduction on RoadScene, named in §6.

**Family D (free depth).** **D-agrees**: no metric survives Holm (TNO: none, 7 of 9 equivalent). **D-differs**:
at least one survives Holm, reported by direction.

A cell that crosses uncorrected but not after Holm is never a finding in either direction.

## Kill criteria — before any statistic

1. **Every checkpoint loads strictly** into the network its tag names (`num_block`, tied or untied), with the
   parameter count the tag implies (29,744 / 49,204 / 68,664 / 88,124 untied; 20,014 tied).
2. **40 RoadScene pairs fused and scored per checkpoint**, the same list for all.
3. **No two checkpoints give identical outputs**, and every checkpoint's TNO row, re-scored in the same kernel on
   a sample of five pairs, matches its stored TNO row to within 1e-4 — so the pipeline that reads RoadScene is the
   pipeline that produced the TNO record.

## Cost

No training. Fusion of 40 pairs by 78 small networks, and 3,120 evaluator calls: a few CPU-hours, run where it
does not draw on the GPU quota if possible.

---

## Correction — 2026-09-25, before any Run 16 script, dataset or kernel exists (no data seen)

**Kill criterion 3 as written cannot be computed.** The stored TNO rows are **means over all 25 pairs**, so a
five-pair re-score has nothing to be compared with. And a tolerance of 1e-4 on metric means is tighter than a
re-fusion in a fresh GPU session can guarantee: inference may select different cuDNN kernels, a handful of
pixels can round the other way, and SD (≈ 43) or SF (≈ 11) then move by more than 1e-4 with nothing wrong.

**Replaced by, before any data exists:** every checkpoint is re-fused and re-scored on **all 25 TNO pairs** in the
same run, and each of its nine TNO means must match the stored row to within **0.1 % relative** (|Δ| ≤ 1e-3 ×
max(1, |stored value|)). A checkpoint that fails is excluded from every family and named; if more than two fail,
the whole run is discarded as a pipeline mismatch. Everything else in this document stands.

**Execution, also fixed now:** fusion needs a GPU (`fuse_lrrnet.py` moves the model to CUDA), the evaluator does
not. Fusion of TNO and RoadScene for all 78 checkpoints runs in one short GPU kernel; scoring runs in a CPU-only
kernel that reads the fused images, so the evaluator's hours do not draw on the GPU quota.
