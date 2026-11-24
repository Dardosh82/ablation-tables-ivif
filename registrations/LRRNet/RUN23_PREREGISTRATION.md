# Run 23 — pre-registration · a screen for removing the tied depth-8 network's SD cost, exploratory by design

**Written 2026-09-28, committed on its own before any Run 23 kernel is built.** No Run 23 row exists. The CPU
diagnostic D4 (where the SD is lost on TNO) runs **after** this commit. It is exploratory and cannot change the arms or
the rule below. Nothing below may be edited after the first result; corrections go in a dated block at the end, only
before data is read.

---

## Why this run exists

Run 20 (registered outcome **G2**) found the tied depth-8 network, 2.46× smaller than the published configuration,
better than it on RoadScene with nothing against. On TNO, however, it is better on VIF, Qabf and SSIM but **worse on
SD** (−1.83 σ). On MSRS (Run 21) it is better with nothing against. An improvement *without a cost* needs a change
that restores global contrast on TNO and keeps the rest.

In the released loss, the infrared side enters only through two terms: the deep-feature target
`w_ir·φ(ir) + w_vi·φ(vi)` and the infrared Gram term weighted by `lam3_gram`. The visible side also has a pixel term and
a shallow-feature term weighted by `lam2_vi`. The authors' code carries their own search grid:
- `w_ir` ∈ {1, …, 6}, with `# w_ir = 4.0` left in the loss;
- `lam2_vi` ∈ {0.1, 0.5, 1.0, 1.5, 2.0, 2.5};
- `lam3_gram` ∈ {1500, 2000, 2500}.

Each arm below is **one step inside that grid**, toward the infrared side. This screen selects at most one of them. It
claims nothing.

## What is run

Base: **tied, `num_block = 8`, 4 epochs**, LLVIP, the M1 depth-8 template (as Runs 17 and 20). Seeds **42, 43, 44**.

| arm | change (everything else as the base) | switch |
|---|---|---|
| **A1** | `w_ir` 3.0 → **4.0** | `LRR_WIR=4.0` |
| **A2** | `lam3_gram` 2000 → **2500** | `LRR_GRAM=2500` |
| **A3** | `lam2_vi` 1.5 → **1.0** | `LRR_LAM2=1.0` |
| **A4** | `num_block` 8 → **6** (tied, no loss change) | `LRR_NUM_BLOCK=6` |

12 runs, ≈ 13 GPU-h. The switches are from `patch_lrrnet.py` (`ebd3e77`), whose defaults were verified to leave the
trainer identical.

**Existing rows, no re-run:**
- **P** — the published configuration at seeds 42–44 (`kernel_csv/lrrnet-stage1`);
- **A0** — the unchanged tied depth-8 network at seeds 42–44 (`kernel_csv/lrrnet-m1-nb8`).

## The reading, fixed now

Metrics: the seven that pass LRRNet's positive control — EN, MI, SD, VIF, SCD, Qabf and SSIM. Test set: TNO, 25 pairs,
the same evaluator. For each arm against P: the mean difference per metric in pooled σ.

**An arm ADVANCES if all five hold:**
1. it is ahead of P on at least **4 of the 7**. This is 4, not Run 18's 5, because this screen targets a cost; the base
   is already ahead on three metrics;
2. no metric is behind P by more than **1.0 pooled σ**;
3. **SD is not behind P** (difference ≥ 0);
4. **SD is above A0's** at the same seeds, so the change is what moved it;
5. the person check passes: on TNO pair 02 (Figure 6's box, seed 42) the infrared-only person is visible, with a
   region contrast of at least 0.9× the published output's, and by eye.

**Of the arms that advance, exactly one goes to Run 24**: the one with the largest SD difference against P. If none
advances, the reading is **"no candidate"**: no removal of the SD cost is claimed, and Run 24 is not run.

Known risk, stated in advance: Run 18's s2 raised SD and MI while SCD fell by 13.8 σ. Condition 2 exists to reject
exactly that.

No p-values are computed for selection: three seeds cannot carry them.

## Kill criteria — before any reading

1. The `[CONFIG]` line shows `tied=1`, the arm's `num_block`, `wir`, `gram` and `lam2`, and 4 epochs. The model loads
   with **20,014 parameters**.
2. 4 epochs, exit 0, one checkpoint per seed, 25 TNO pairs scored.
3. A run whose training loss is NaN or diverges (final epoch mean above the first epoch's) is excluded and named. An
   arm left with fewer than 3 seeds does **not advance**.
