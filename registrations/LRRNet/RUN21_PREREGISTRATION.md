# Run 21 — pre-registration · a third test set (MSRS, 361 pairs) for the depth-8 result, no training

**Written 2026-09-27, committed on its own before any checkpoint has been fused or scored on MSRS.** Nothing below may
be edited after the first result; corrections go in a dated block at the end, only before data is read.

---

## Why

The tied depth-8 result (Runs 17, 17b) holds on TNO and RoadScene. MSRS is the third test set this project already
holds (361 pairs, the set CrossFuse's evaluation uses), and no LRRNet checkpoint has ever been scored on it. Scoring the
existing checkpoints there costs no training and asks whether the result is a property of the method or of two test
sets. MSRS is also where CrossFuse's own comparisons resolved least (Run 11 A-4), so a null here is a live possibility and
would be reported as scoping the result.

## What is run

The same checkpoints, fused on MSRS's 361 pairs (the IR folder's names; the visible folder's eight "(1)" duplicates are
not pairs and are ignored, as by every script here) and scored by the same evaluator:
- the 24 Run 17 checkpoints — tied `num_block` = 8 and untied `num_block` = 8, seeds 62–73;
- the published configuration (untied `num_block` = 4) at seeds 42–61 (20 checkpoints; Run 9/10 training, the Run 16
  checkpoint set).

Every checkpoint is re-fused and re-scored on TNO in the same session and must reproduce its stored TNO row within 0.1 %
relative; more than two failures discard the run. Fusion on GPU (< 1 GPU-h), scoring on CPU.

## The readings, fixed now

Welch's *t*, two-sided 5 %, exact df; Holm within each family; TOST at ±1.75 pooled σ; sample SD.

**M1 — does Run 17 hold on MSRS?** Tied d8 vs untied d8 (62–73), SF, AG, VIF, SCD, Holm across four.
- **M1-HOLDS:** ≥ 1 survives favouring tied, none favouring untied. **M1-NULL:** none survives. **M1-REVERSED:** any
  survives favouring untied.

**M2 — tied d8 against the published configuration on MSRS.** Tied d8 (62–73) vs published d4 (**seeds 50–61**, the same
baseline seeds as Run 17b's Q2), the seven control-passing metrics (EN, MI, SD, VIF, SCD, Qabf, SSIM), Holm across seven.
- **M2-BETTER:** ≥ 1 favouring tied, none against. **M2-MIXED:** survivors both ways (a trade-off). **M2-NULL:** none.
  **M2-WORSE:** only against.

The published seeds 42–49 are scored too but **not read here**: they are Run 20's baseline, and Run 20's reading is on
TNO and RoadScene only; whether to add MSRS to Run 20 is a separate decision, registered separately if made.

## Kill criteria

1. Every checkpoint loads at its expected parameter count (20,014 tied; 88,124 untied d8; 49,204 published) and fuses
   361 MSRS + 25 TNO pairs.
2. TNO re-score within 0.1 % relative for all but at most two checkpoints.
3. No identical rows.
