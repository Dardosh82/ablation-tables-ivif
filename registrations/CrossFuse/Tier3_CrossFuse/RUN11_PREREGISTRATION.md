# Run 11 — pre-registration · case study I's two three-seed arms, taken to five

**Written 2026-09-24, committed on its own before any Run 11 kernel exists.** Nothing below may be
edited after the first result; corrections go in a dated block at the end.

---

## Why this run exists

Every CrossFuse conclusion now rests on five seeds **except two arms**, which rest on three:

| arm | what it supports | seeds |
|---|---|---|
| **A1b/A1c/A1d** — Top-K removed (k = 0), 299 epochs | the Top-K ablation read against seed noise, and the instability finding of §0.2 | 42, 43, 44 |
| **HB42/43/44** — half the training budget, 150 epochs | "2 of 9 metrics resolve half the budget on CrossFuse, where 0 of 9 did on LRRNet" | 42, 43, 44 |

Three seeds is the smallest sample in the thesis, and this project's own argument is that small
samples mislead — twice already in this case study (claims #1 and #2 were withdrawn by a second
seed; the MSRS VIF advantage was withdrawn by a third). The cheapest way to close that opening is to
run the two arms at five seeds, like everything else.

**This run can damage our own claims.** Two seeds can move a cell either way: a measurable
difference can dissolve, and a null can acquire one. Both readings are fixed below, per arm, before
the kernels exist.

## What is run

The Tier 3 configuration, unchanged: CrossFuse's released code with the recovered split, the same
evaluator, the same test sets, nine metrics. Only the seed changes; `--resume auto` is not used
(it is broken in this container), and each tag gets its own output directory.

| arm | new seeds | epochs / epoch_gap / optim_step | runs | h/run (measured) | GPU-h |
|---|---|---|---|---|---|
| k = 0 (Top-K removed) | **45, 46** | 299 / 75 / 38 | 2 | ~3.3 | ~6.6 |
| half budget | **45, 46** | 150 / 38 / 19 | 2 | ~1.65 | ~3.3 |

Total ≈ **10 GPU-h**. Seeds 45 and 46 are already used by the *released-configuration* arm (PC45,
PC46) and have never been used by either arm above.

---

## The readings, fixed before the run

σ is the **sample** standard deviation (ddof = 1) at n = 5 everywhere. The population SD that
inflated earlier σ-unit figures by 1.22 (withdrawn claim #13) is not used. Every test is **Welch's
*t*, two-sided 5 %, at each comparison's own degrees of freedom**, with **Holm** reported beside the
uncorrected count for each family of nine metrics.

### Arm A — Top-K removed (k = 0)

Four statements currently rest on this arm at n = 3. Each is re-tested at n = 5 and each has its two
outcomes written here.

| # | standing statement (n = 3) | holds if | withdrawn/revised if |
|---|---|---|---|
| A-1 | removing Top-K **improves VIF and MI** on RoadScene and TNO | the same four cells cross, same direction | any of them falls inside noise at n = 5 |
| A-2 | removing Top-K **costs SCD and SSIM** on TNO (and SF on RoadScene) | those cells cross, same direction | as above |
| A-3 | k = 0 training is **markedly less stable**: median σ(k=0)/σ(released) = 3.35 over 27 cells | the median ratio stays **> 2** at n = 5 | it falls to ≤ 2 — the instability claim is then stated as a smaller factor, with the new number |
| A-4 | on MSRS **almost nothing is distinguishable**, because k = 0 is too unstable there | ≤ 2 of 9 cells cross | more cross — MSRS is reported as measurable and §0.2's explanation is narrowed |

**The headline sentence of case study I does not depend on this arm** (it rests on the published
Table VI row read against σ, which costs no GPU time). If A-1 or A-2 changes, what changes is the
*supporting* ablation, and the change is reported in the same sentence as the original claim.

### Arm B — half the training budget

One statement rests on this arm: **2 of 9 metrics (MI, VIF) resolve half the budget on CrossFuse,
where 0 of 9 did on LRRNet** — and those two are the same two that rank an untrained network above a
trained one, which is what makes the point.

- **B1 — MI and VIF still cross, nothing else does.** The claim stands at n = 5, with the count
  unchanged: *the resolution floor is architecture-dependent, and the only two measures with
  resolution in this range are the two that fail the positive control.*
- **B2 — one of MI/VIF stops crossing.** Reported as **1 of 9**, with the metric named, and the
  sentence "2 of 9" corrected everywhere it appears (paper, both reports, EXPLAIN, deck, speaker
  report) — the count is load-bearing and `audit_numbers.py` will be given the old string.
- **B3 — a third metric crosses.** Reported as 3 of 9 with the new metric named, and the
  non-monotonicity claim re-examined: it is asserted only where **both** legs (untrained↔half and
  half↔full) clear their own critical value, as it was at n = 3.
- **B4 — neither crosses.** The strongest correction: CrossFuse would then resolve half the budget
  no better than LRRNet, and the "architecture-dependent resolution floor" claim would rest on the
  untrained arm alone. It would be rewritten that way, not dropped quietly.

### C. Equivalence

For every cell that does **not** cross in either arm, TOST at **±1.75 pooled σ** at n = 5 is
reported, so a null is stated as *equivalent* or as *inconclusive at this sample size* — never as
"no difference" by default.

### D. Power

At 5 against 5, the design resolves about **2.0 pooled σ** at 80 % power. Effects smaller than that
are not detectable here; a null is reported with that sentence attached.

---

## Kill criteria — checked before any statistic is computed

1. **Epochs completed** = 299 (arm A) and 150 (arm B) on every new run, read from the kernel log.
2. **Image counts** identical to the seeds already run, per test set (RoadScene, TNO, MSRS).
3. **No duplicate runs:** each new seed must differ from every existing seed on every metric.
4. **CV below 25 %** on every metric in every arm, as in Run 8.
5. **Seed fingerprint audit passes** for seeds 45 and 46 on this code path.
6. **Checkpoint hygiene:** each tag writes its own outdir; no `--resume auto`; the checkpoint file is
   cleared between runs. (`ckpt_ep<NNN>.pth` carries no tag in its name — the defect that made one
   run load another's weights.)

## What is *not* changed

- No configuration, no evaluator, no image list, no analysis script other than the seed lists.
- The n = 3 results stay in the record as written. The n = 5 results are reported **beside** them,
  with both numbers visible, exactly as Run 9 and Run 10 were.

---

## Cost

4 runs ≈ **10 GPU-h**, one weekly quota with margin.

---

## Correction — 2026-09-25, after arm A completed and before any arm A statistic was computed

**A factual statement above is wrong.** "Seeds 45 and 46 are already used by the *released-configuration*
arm (PC45, PC46)" — they are not. PC45 and PC46 (the positive-control kernel) were trained on the **k = 0
corpus, 299 / 75 / 37, seeds 45 and 46**: the same configuration as this run's arm A. They were scored on
**TNO only**. Consequences, stated before any arm A number is read:

1. **On TNO the k = 0 arm already stood at five seeds** (A1b, A1c, A1d, PC45, PC46). Arm A's A1e and
   A1f repeat PC45 and PC46 seed for seed, so on TNO they are **not new seeds** and are not counted as
   such.
2. **On RoadScene and MSRS the arm genuinely stood at three**, and A1e / A1f are its fourth and fifth
   seeds. The readings A-1 to A-4 are applied there as registered.
3. **The same result was available for about a quarter of an hour of GPU time**: the PC45 / PC46
   checkpoints exist (`crossfuse-poscontrol-ckpts`) and only needed scoring on RoadScene and MSRS. About
   **6.6 GPU-h** were spent retraining them. That is a planning error in this document, recorded here.
4. **What the retraining does buy** is two **same-seed replicate pairs** on TNO — A1e against PC45, A1f
   against PC46: same code, corpus, schedule and seed, different sessions. They are reported as a
   measurement of run-to-run nondeterminism beside the one already in the paper, and never as additional
   seeds.

The choice of rows, fixed now: **TNO uses A1b, A1c, A1d, PC45, PC46** (the arm as it existed); **RoadScene
and MSRS use A1b, A1c, A1d, A1e, A1f**. Every TNO reading is also reported with A1e / A1f substituted for
PC45 / PC46, and a conclusion that changes between the two is reported as unstable.
