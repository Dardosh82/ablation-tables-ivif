# Run 22 — pre-registration · Run 18's collaborator-account arms, again on our own account

**Written 2026-09-28, committed on its own before any Run 22 kernel is built or pushed.** Nothing below may be edited
after the first result; corrections go in a dated block at the end, only before data is read.

---

## Why this run exists

Three of Run 18's six screening arms — **s1a** (lr 1e-4), **s4** (crops at test scale) and **s2** (middle-loss weights
rescaled to sum to 1) — ran on a collaborator's Kaggle account. Their results are valid: the same code, dataset
versions and torch build, and the seed fingerprint matched the baseline. But their notebooks were deleted in a
clean-up, so the committed CSVs and logs are the only record, and all work since 2026-09-26 runs on our own account
(`mostafanada82`) only. This run repeats the three arms there, so that every row the thesis reads comes from a kernel
on our account that anyone with access can open.

## Design

Exactly Run 18's arms. The builder is `build_run18_kernels.py` (the version with the `[LOSSCMP]` print fixed), with
kernels `mostafanada82/lrrnet-run22-{s1a,s4,s2}`. Seeds **42, 43, 44**. Base: untied, `num_block = 4`, 4 epochs,
LLVIP. Changes, as registered in `RUN18_PREREGISTRATION.md`:

| arm | change |
|---|---|
| s1a | lr 1e-5 → 1e-4 |
| s4 | random 128×128 crops of frames downscaled 2× |
| s2 | w_ir 3.0 → 0.857, w_vi 0.5 → 0.143 |

9 runs, ≈ 10.5 GPU-h (measured: ≈ 68 min per run in Run 18).

## The reading, fixed now

1. **Run 18's rule, unchanged**, is applied to the own-account rows against the published configuration at seeds
   42–44 (`kernel_csv/lrrnet-stage1`). The seven selection metrics are EN, MI, SD, VIF, SCD, Qabf and SSIM, with pooled
   σ. An arm advances only if it is ahead on ≥ 5 of 7 **and** no metric is behind by more than 1.0 σ **and** it passes
   the person check.
   - **R1 — CONSISTENT:** no arm advances. Run 18's reading, "no candidate", stands, now on our own account.
   - **R2 — CHANGED:** an arm that failed on the collaborator's account advances here. Run 18's reading is re-opened in
     a dated correction. The arm is named as a candidate, with no claim, and it goes to a separately registered
     confirmation.
2. **Agreement with the first execution (descriptive).** For each arm and metric, the own-account mean minus the
   collaborator mean, in units of the published configuration's seed σ. cuDNN is not deterministic, so exact equality
   is not expected. Any difference above 1 σ is named.
3. **The record.** Own-account rows replace the collaborator rows in every analysis and document. The collaborator
   rows stay in `kernel_csv/` as the first execution and are cited as such.

## Kill criteria — before any reading

1. The `[CONFIG]` line shows the arm's settings, 4 epochs, `num_block = 4`, 49,204 parameters.
2. 4 epochs, exit 0, one checkpoint per seed, 25 TNO pairs scored, and the `[LOSSCMP]` line printed.
3. Seed fingerprint equal to the baseline's (as in Run 18).
4. A kernel that fails is re-pushed once. If it fails again, that arm is reported as not re-run and its collaborator
   rows stay the record, named as such.
