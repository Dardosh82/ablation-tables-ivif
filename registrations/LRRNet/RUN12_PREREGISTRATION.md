# Run 12 — pre-registration · M1 (weight tying) from three seeds to five

**Written 2026-09-24, committed on its own before any Run 12 kernel exists.** Nothing below may be
edited after the first result; corrections go in a dated block at the end.

---

## Why this run exists

M1 — tying the weights across LRRNet's unrolled iterations — is the one *design* result this project
has. It is reported as **2.46× fewer parameters (49,204 → 20,014) at indistinguishable quality**.
Two things about how that was measured need repair, and neither is the fault of the result:

1. **It is 5 against 3.** The untied baseline has five seeds (42–46, stage 1); the tied model has
   three (42–44). The registered critical value was a fixed **2.776**, which is correct for three
   against three and **too strict** for five against three, where the right value is about 2.49. A
   stricter ruler makes "nothing moves" easier to obtain — and "nothing moves" is exactly what was
   reported. Re-measured with the correct value, **one metric of nine crosses: SSIM, *p* = 0.038, in
   M1's favour** (recorded 2026-09-13 beside the registered result, not in place of it).
2. **Three seeds is the weakest sample in the chapter.** Every other LRRNet result now stands at
   n = 8 or n = 20. The design claim should not rest on the smallest sample in the thesis.

Adding seeds 45 and 46 to the tied arm makes the comparison **5 against 5 at each comparison's own
degrees of freedom**, which is what the protocol requires everywhere else.

**This run can damage our own claim.** If SSIM (or anything else) crosses at n = 5, the honest
reading is *not* "M1 is better" — see outcome B2 below — and if a metric crosses **against** the tied
model, the efficiency claim acquires a measured quality cost. Both readings are written here first.

## What is run

The configuration of stage 1 and the M1 run, unchanged: `num_block = 4`, `epochs = 4`, batch 8,
128×128 grayscale, LLVIP for training, CrossFuse's `Evaluator`, the same **25 TNO pairs**, nine
metrics. Only the seed changes.

| arm | seeds already run | new seeds | runs | h/run (M1 log) | GPU-h |
|---|---|---|---|---|---|
| tied (M1, 20,014 par.) | 42, 43, 44 | **45, 46** | 2 | ~1.10 | ~2.2 |
| untied (published, 49,204 par.) | 42–46 | none | 0 | — | 0 |

Seeds 45 and 46 have never been used on the tied model. Fusing and scoring add ~6 minutes.

---

## The readings, fixed before the run

### A. The test

Nine metrics, tied (n = 5) against untied (n = 5), **Welch's *t*, two-sided 5 %, at each
comparison's own degrees of freedom** — not a fixed critical value. σ is the **sample** standard
deviation (ddof = 1); the population SD that inflated earlier figures by 1.22 (withdrawn claim #13)
is not used anywhere in this analysis. Multiplicity: nine correlated metrics, **Holm** across the
nine, reported beside the uncorrected count.

### B. The four outcomes

- **B1 — 0 of 9 cross.** The standing claim holds and is now measured at 5 against 5 with the right
  ruler: *the same algorithm at 2.46× fewer parameters, indistinguishable on all nine metrics.*
  The efficiency claim stays categorical; the quality claim stays **indistinguishable**, not better.
- **B2 — exactly 1 of 9 crosses, favouring tied (the SSIM case).** Reported as **one cell of nine,
  at the rate chance produces** (0.45 expected at 5 %), and **not** as a quality improvement. The
  wording is fixed now: *"one metric of nine crosses uncorrected and none after Holm; M1 is
  indistinguishable, not better."* This is the same rule applied to LRRNet's own 1 of 27, and it is
  applied here against our own interest.
- **B3 — 2 or more cross, all favouring tied, and at least one survives Holm.** Reported as a
  measured quality gain at 2.46× fewer parameters, with the surviving metrics named and the rest
  reported as null. This is the only outcome that upgrades the design claim.
- **B4 — any metric crosses favouring untied.** Reported as a **measured cost of the reduction**:
  the tied model is smaller and worse on that metric, named in the abstract of the chapter, not
  buried. If this happens, the headline becomes "2.46× smaller at a cost on *X*", and §10.6 is
  rewritten.

A cell that crosses uncorrected but not after Holm is never reported as a finding in either
direction, whichever way it points.

### C. Equivalence, stated as well as significance

A non-significant test is not evidence of equivalence. Alongside the nine *t*-tests, **TOST at
±1.75 pooled σ** (the margin used for Run 9 and Run 10) is run on all nine metrics at n = 5. Cells
that are neither different nor equivalent are reported as **inconclusive at this sample size**, by
name, exactly as Run 9's four cells were.

### D. Power, stated so a null is not over-read

At 5 against 5, a two-sided 5 % Welch test resolves about **2.0 pooled σ** at 80 % power. Effects
smaller than that are not detectable here, and B1 therefore means *"no effect of about 2 σ or larger
was found"*, not *"no effect exists"*. The equivalence test in C is what carries the positive claim.

---

## Kill criteria — checked before any statistic is computed

Any failure stops the run and the readings are discarded; nothing is analysed after a repair.

1. **Parameter counts exactly as designed:** tied = **20,014**, untied = **49,204**. A tied run that
   prints the untied count means the tying did not take, and the run is a healthy untied network
   trained for nothing.
2. **Epochs completed = 4** on every new run, from the kernel log, not the config.
3. **25 TNO pairs scored** per run, the same image list as seeds 42–44.
4. **Seed fingerprint audit passes:** seed 45 ≠ seed 46 on the first-batch fingerprint, and a repeat
   of seed 45 reproduces itself, as in the M1 run's Cell 4.
5. **No checkpoint reuse:** each seed writes to its own output directory; the checkpoint file is
   cleared before each run (`--resume auto` does not work in this container).
6. **No duplicate runs:** the two new seeds must differ on every metric. Identical rows mean one run
   was scored twice.

## What is *not* changed

- The registered n = 3 result stays in the record as it was written. This run is reported **beside**
  it, not in place of it.
- Nothing about the training budget changes here. Whether the tied model benefits from a longer
  budget is a **separate, separately registered experiment** (the training-length curve); mixing the
  two would confound the sample-size repair with a design change.

---

## Cost

2 runs × ~1.10 h + ~0.1 h scoring ≈ **2.4 GPU-h**, inside one weekly quota with wide margin.
