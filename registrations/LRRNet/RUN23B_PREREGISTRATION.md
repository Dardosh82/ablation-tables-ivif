# Run 23b — pre-registration · one more screening arm, aimed at where diagnostic D4 found the SD cost

**Written 2026-09-28, committed on its own before any Run 23 or Run 23b kernel is built or pushed.** No Run 23 or 23b
row exists. Run 23 (`RUN23_PREREGISTRATION.md`, `60f6e37`) is **not changed**: its four arms, rule and baselines
stand exactly as written. Nothing below may be edited after the first result; corrections go in a dated block at the
end, only before data is read.

---

## Why this run exists

Diagnostic D4 (`LRRNET_RESULTS.md`, `3d2f2ee`) is exploratory and was run after Run 23 was registered. It used tied
depth 8 at Run 20 seeds 74–77 and the published configuration at seeds 45–49, on the 25 TNO pairs:
- **93 % of the variance the tied network loses on TNO is within the background**, not the infrared targets. Those
  stand further from the background than in the published output.
- The loss is the **dark end being lifted**: the 1st percentile is +4.4 grey levels, the 99th unchanged.

Run 23's arms A1–A3 strengthen the infrared side of the loss, so D4 suggests they aim at the wrong place. This run
adds **one arm aimed at the background**. It was chosen **after** seeing D4, and is labelled so everywhere.

In the released loss, the visible image's fine structure (edges, shadows, texture) enters through the shallow
VGG-feature term, weighted by `lam2_vi`. The authors' grid for it is {0.1, 0.5, 1.0, 1.5, 2.0, 2.5}, and the
published value is 1.5. The arm is the next step up in that grid.

## What is run

| arm | change (everything else as Run 23's base: tied, `num_block = 8`, 4 epochs, LLVIP) | switch |
|---|---|---|
| **B1** | `lam2_vi` 1.5 → **2.0** | `LRR_LAM2=2.0` |

- Seeds **42, 43, 44**. 3 runs, ≈ 3.3 GPU-h.
- Kernel `mostafanada82/lrrnet-run23b-b1`, from the same builder, template and scripts dataset as Run 23
  (`lrrnet-run23-scripts`).
- The same existing baselines as Run 23:
  - **P**: the published configuration, seeds 42–44;
  - **A0**: unchanged tied depth 8, seeds 42–44.

## The reading, fixed now

**Run 23's rules 1–5 apply to B1 unchanged**:
1. ahead of P on ≥ 4 of the 7 metrics that pass the positive control;
2. no metric behind P by more than 1.0 pooled σ;
3. SD not behind P;
4. SD above A0's;
5. the person check passes on TNO pair 02 (ratio ≥ 0.9 and by eye).

**Selection across both screens.** The arms of Run 23 and Run 23b that pass rules 1–4 form one pool. The one with the
largest SD difference against P is the candidate, subject to rule 5; if it fails rule 5, the next by SD is taken. If no
arm in either screen passes, the reading is **"no candidate"** and Run 24 is not run.

Known risk, stated in advance: pulling toward the visible image is the direction in which §7 shows MI rewarding a
fused image for copying one source. If B1 raises MI while SCD falls by more than 1 σ, rule 2 rejects it.

## Kill criteria — before any reading

As Run 23:
1. `[CONFIG]` shows `tied=1 num_block=8 wir=3.0 gram=2000 lam2=2.0` and 4 epochs; 20,014 parameters.
2. 4 epochs, exit 0, one checkpoint per seed, 25 TNO pairs.
3. A NaN or diverging run is excluded and named. Fewer than 3 seeds → B1 does not advance.
