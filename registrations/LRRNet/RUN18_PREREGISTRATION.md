# Run 18 — pre-registration · the improvement screen (S0–S4), exploratory by design

**Written 2026-09-26, committed on its own before any Run 18 script, patch or kernel exists.** Run 17 is at 4 of 12
seeds per arm and has not been analysed. Nothing below may be edited after the first result; corrections go in a
dated block at the end, only before data is read.

---

## Why this run exists

Diagnostics D2 and D3 (CPU, test images, `LRRNET_RESULTS.md` and `TIER3_RESULTS.md`) found that trained LRRNet and
CrossFuse networks do not beat trivial fusion rules on their own training objectives, and that six LRRNet
configurations sit within 5 % of each other on theirs. The account this suggests — quality is limited by the
objective and its optimisation, not the architecture — predicts that changing *those* moves the metrics where size,
depth, tying and budget did not. This run **screens** the candidate changes. It is exploratory and labelled so
everywhere: **no improvement is claimed from it.** A claim needs the separate confirmatory run it selects.

## What is run

Base: released LRRNet, `num_block = 4`, untied, 4 epochs, LLVIP, 128×128, batch 8, the published loss — each arm
changes only what its row names. Seeds **42, 43, 44** for every arm, so each arm is compared with the published
configuration's existing rows at the same seeds (`kernel_csv/lrrnet-stage1`, seeds 42–44).

| arm | change | tests |
|---|---|---|
| **S0** | none — CPU kernel: the released loss on 400 random LLVIP training pairs for copy-VI, max, mean, copy-IR and the stored checkpoints nb4 seeds 42–44 | whether D2 holds on the training corpus |
| **S1a** | learning rate 1e-5 → **1e-4** | under-optimisation |
| **S1b** | learning rate 1e-5 → **3e-4** | under-optimisation, further |
| **S2** | middle-loss weights rescaled to sum to 1, ratio kept: w_ir 3.0 → **0.857**, w_vi 0.5 → **0.143** | the unreachable target |
| **S3u** | S1a + S2, untied | whether the two add |
| **S3t** | S1a + S2, **tied** (weight tying as in M1) | whether tying's saving survives them |
| **S4** | training images: random 128×128 **crops** of frames downscaled 2× (to 640×512), instead of whole frames resized 10× | train/test scale mismatch |

Seven training arms × 3 seeds = 18 runs, ≈ 20 GPU-h; S0 on CPU. Every arm also logs its own final-epoch training loss
beside copy-VI's on the same batches, so each arm's position against the trivial solution is recorded.

## The reading, fixed before the run

Metrics: the seven that **pass LRRNet's positive control** — EN, MI, SD, VIF, SCD, Qabf, SSIM. SF and AG are reported
but not used to select, because they cannot distinguish an untrained LRRNet from a trained one. TNO, 25 pairs, the
same evaluator as every LRRNet row.

For each arm against the published configuration at the same three seeds: mean difference in pooled σ per metric.

**An arm ADVANCES if all three hold:**
1. it is ahead on at least **5 of the 7** selection metrics;
2. no selection metric is behind by more than **1.0 pooled σ**;
3. on TNO pair 02 the person visible only in the infrared band is visible in the arm's fused output (seed 42),
   checked by eye and by the mean intensity of the person's region against the published output's — an arm that
   loses the person does not advance whatever the numbers say.

**Of the arms that advance, exactly one goes to confirmation**: the largest mean standardised gain over the seven
selection metrics. If none advances, the screen's reading is **"no candidate"**, the account in D2/D3 is reported as
not supported by intervention, and no confirmatory run follows from this screen.

**S0 reading:** D2 **holds on the training corpus** if the mean loss of the three stored checkpoints is above
copy-VI's on the 400 LLVIP pairs; otherwise D2 is recorded as a test-set artefact and the S2 rationale is withdrawn
(its result is still reported).

No p-values are computed for selection: three seeds cannot carry them, and a screen that tests for significance
invites reading it as confirmation.

## Kill criteria — before any reading

1. `[CONFIG]` of every run shows the arm's settings (the patch prints lr, w_ir, w_vi, crop mode, tied), 4 epochs,
   `num_block = 4`; parameter count 49,204 (untied) or 20,014 (S3t).
2. 4 epochs completed, exit 0, one final checkpoint per seed; 25 TNO pairs scored.
3. A run whose training loss becomes NaN or diverges (final epoch-mean loss above the first epoch's) is excluded and
   named, and counts as the arm **not advancing** if it leaves fewer than 3 seeds.

## What follows

The advancing arm, if any, is tested in a **separate, separately registered** confirmatory run: new seeds (≥ 10 per
arm), Welch at exact df, Holm across the seven selection metrics, on **TNO and RoadScene** — an improvement must hold
on both. Nothing from this screen enters the paper as a finding; it enters as the selection step of that run.
