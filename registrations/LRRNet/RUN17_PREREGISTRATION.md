# Run 17 — pre-registration · the one open lead: tied against untied at depth 8, on new seeds

**Written 2026-09-25, committed on its own before any Run 17 script or kernel exists.** Nothing below may be
edited after the first result; corrections go in a dated block at the end, and only before data is read.

---

## Why this run exists

Run 14's secondary 2×2 (tied arms at five seeds) left one contrast that the paper cannot leave as it is. **At
depth 8 the tied network (20,014 parameters) is nominally ahead of the untied one (88,124) on four metrics** —
SF, AG, VIF, SCD at *p* = 0.021, 0.007, 0.008, 0.017, uncorrected, 1.57 to 2.28 pooled σ — and none survives
Holm (0.007 > 0.05/9). All four favour our own model. The paper says this "would need its own registered test to
be claimed"; this is that test.

**The four metrics were chosen by looking at the data**, so the seeds that raised the question cannot answer it
(the same rule as Run 10). The confirmatory test uses **new seeds only**, and the effect sizes above are treated
as inflated by that selection.

**This run can remove a lead that favours us.** If nothing replicates, the paper says the lead did not survive
its own test, and the 2×2 paragraph is closed with that sentence.

## What is run

| arm | network | parameters | template (the kernel that produced this arm's existing rows) | seeds |
|---|---|---|---|---|
| **tied d8** | weight-tied LRRNet, `num_block` = 8 | 20,014 | `lrrnet-m1-nb8` | **62–73** (12) |
| **untied d8** | released LRRNet, `num_block` = 8 | 88,124 | `lrrnet-stage3` | **62–73** (12) |

Only the seed list changes from each template: 4 epochs, the same corpus, patch set, batch, evaluator and 25 TNO
pairs. No seed in 62–73 has been run at depth 8 on either network. Six kernels of four seeds, two at a time;
≈1.1 GPU-h per run measured on both arms, so **≈26 GPU-h** in all.

## The reading, fixed before the run

**Confirmatory family — the four metrics that raised the question: SF, AG, VIF, SCD.** New seeds only, 12
against 12. Welch's *t*, two-sided 5 %, at each comparison's own degrees of freedom; **Holm across the four**.
σ is the sample SD (ddof = 1).

- **L1 — REPLICATES.** At least one of the four survives Holm favouring the tied network, and none survives
  favouring the untied one. Reported as: *at depth 8, the tied network at 4.40× fewer parameters is better on
  the named metrics* — the first quality claim of the design half, scoped to depth 8, TNO and the published
  4-epoch budget, and stated beside Run 12's null at depth 4 (the effect is not a general property of tying).
- **L2 — NOT REPLICATED.** None of the four survives Holm. Reported as: *the depth-8 lead did not survive its own
  registered test*; each metric's effect is given in σ with TOST at ±1.75 σ, so a null is stated as equivalent
  or inconclusive by name, never as "no difference" by default.
- **L3 — REVERSED.** At least one survives Holm favouring the untied network, and none favours the tied one.
  Reported as a measured cost of tying at depth 8.
- **L4 — MIXED.** Survivors in both directions. Reported metric by metric, with no directional claim.

A cell that crosses uncorrected but not after Holm is never a finding in either direction.

**Secondary, descriptive only (no outcome depends on them):**
1. all nine metrics on the new seeds, Holm across nine;
2. the pooled arms (tied 17 seeds, 42–46 + 62–73; untied 20 seeds, 42–49 + 62–73), nine metrics, Holm across
   nine — labelled as including the seeds that raised the question;
3. seed σ of each arm, since the tied network's smaller spread (VIF σ 0.013 against 0.029 at depth 8) was part
   of the original contrast.

## Power, stated before the run

At 12 against 12 and the first Holm step (α = 0.0125 two-sided), a true effect of **1.6 pooled σ** — about the
smallest of the four observed — is detected with ≈87 % power; one of **1.3 σ** with ≈68 %, one of **1.0 σ** with ≈41 % (noncentral *t*, equal variances). Because the observed
sizes are inflated by selection, L2 means *not replicated at the size that raised the question*, not *zero*.

## Kill criteria — before any statistic

1. `[CONFIG]` of every run reads `epochs=4 num_block=8` and `tied=1` (tied arm) or `tied=0` (untied arm); the
   parameter count printed when the checkpoint is loaded for fusion is **20,014** (tied) or **88,124** (untied); the untied arm keeps its
   template's two-GPU `[GPUS] parallel= True` line, as every existing untied depth-8 row did.
2. Every run completes 4 epochs (601 logged losses) with exit 0 and one final checkpoint.
3. 25 TNO pairs scored per run, the same list for all.
4. No two runs give identical rows.

A run that fails is excluded and named; if either arm ends below **10** seeds, the confirmatory test is not
read and the run is reported as incomplete.

## What it costs, and what it does not touch

≈26 GPU-h. No existing row, reading or pre-registration is changed. The analysis script
(`scripts/analyse_run17.py`) is written before the last kernel lands and reads only the new seeds for the
confirmatory family.
