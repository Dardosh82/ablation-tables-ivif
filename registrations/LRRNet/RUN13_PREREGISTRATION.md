# Run 13 — pre-registration · the training-length curve: does the tied network have headroom the published one lacks?

**Written 2026-09-24, committed on its own before any Run 13 kernel exists.** Nothing below may be
edited after the first result; corrections go in a dated block at the end.

---

## Why this run exists

M1 is measured at **the published training budget and nowhere else**. LRRNet trains for 4 epochs, and
the paper's own numbers saturate early — §1 of the paper names the epoch-2 saturation. So the question
M1 has never been asked is the one a design chapter has to answer:

> The tied network has 2.46× fewer parameters. Under a budget that is not saturated, does it gain more
> than the network it replaces — or does it hit the same ceiling?

This is the difference between *an efficiency result* (what Part III has) and *a design result* (what a
thesis half called **Design** owes). Nothing in Run 12 answers it: Run 12 repaired the sample size at
the published budget, and its outcome (B2 — one cell of nine, none after Holm) is a statement about
4 epochs only.

**This run can close the design half in the negative, and that outcome is written down here as
plainly as the positive one.** If neither arm gains at twice the budget, the honest chapter says the
published budget already saturates both networks, M1 remains an efficiency result, and the next
question is depth rather than time.

## What is run

The configuration of stage 1, the M1 run, and Run 12 — `num_block = 4`, batch 8, 128×128 grayscale,
LLVIP, CrossFuse's `Evaluator`, the same **25 TNO pairs**, nine metrics — with **one variable changed:
`LRR_EPOCHS = 8` instead of 4.** Both arms get it, so the budget is not confounded with the tying.

| arm | `LRR_TIED` | seeds | runs | h/run (est. 2× the 4-epoch 1.10 h) | GPU-h |
|---|---|---|---|---|---|
| tied @ 8 epochs | 1 | 42–46 | 5 | ~2.2 | ~11 |
| untied @ 8 epochs | 0 | 42–46 | 5 | ~2.2 | ~11 |

≈ **22 GPU-h**, split into kernels of three and two seeds so no session approaches the 12-hour cap.
The 4-epoch arms are **not** re-run: they are stage 1 (untied, seeds 42–46) and M1 + Run 12 (tied,
seeds 42–46), both already measured at n = 5 under this same code, container and evaluator.

---

## The readings, fixed before the run

σ is the **sample** standard deviation (ddof = 1). Every test is **Welch's *t*, two-sided 5 %, at each
comparison's own degrees of freedom** — never a fixed critical value; that error has now been found
twice in this project (M1 at 5 against 3, and the half-budget arm at 5 against 5). **Holm within each
family of nine**, reported beside the uncorrected count. **TOST at ±1.75 pooled σ** for every cell that
does not cross, so a null is stated as equivalent or as inconclusive.

Three families, 27 tests in all. The count of families is stated wherever a count of crossings is.

| family | comparison | what it answers |
|---|---|---|
| **C1** | tied @ 8 vs untied @ 8 | the design question at an equal, unsaturated budget |
| **C2** | tied @ 8 vs tied @ 4 | does the smaller network use more training? |
| **C3** | untied @ 8 vs untied @ 4 | does the published network use more training? |

### The four outcomes

- **D1 — tied gains and untied does not** (C2 has at least one metric surviving Holm, C3 has none, and
  C1 does not favour untied). The design result: *the tied network has headroom the published
  configuration does not, at 2.46× fewer parameters.* Reported with the metrics named, the interaction
  stated as what it is — a difference of differences measured on two arms, not a claim about one — and
  the next experiment (depth at constant size) declared.
- **D2 — both gain.** More training helps both networks. The tying claim stays **efficiency only**, and
  the finding is reported *about the paper*: its published budget leaves measurable quality unclaimed,
  which its own single-run table could not have shown.
- **D3 — neither gains.** Saturation at the published budget is confirmed at twice the budget on both
  arms. Reported as a categorical negative: *the budget is not where the headroom is.* Part III then
  closes as an efficiency-and-measurability result, and the free-depth experiment — impossible in the
  untied network, where depth drags size — becomes the next and final design step.
- **D4 — tied is worse at 8 epochs** (any C1 or C2 metric crossing against the tied arm). Reported as a
  **measured cost of tying under longer training**, named in the chapter's headline. A smaller model
  that degrades when trained longer is a real limitation of the design, and it is stated as one.

A cell that crosses uncorrected but not after Holm is not a finding in either direction. One cell of
nine is the rate chance produces; the same rule refused SSIM in Run 12 and refuses it here.

### Power, stated so a null is not over-read

At 5 against 5 this design resolves about **2.0 pooled σ** at 80 %. D3 therefore means *no effect of
about 2 σ or larger was found at twice the budget*, not that the tied and untied networks are identical
under any budget. The equivalence test carries whatever positive claim D3 makes.

---

## Kill criteria — checked before any statistic is computed

1. **`[CONFIG]` on every run** reads `epochs=8`, `num_block=4`, and `tied=1` (tied arm) or `tied=0`
   (untied arm). A mislabelled arm invalidates the whole comparison, not one row.
2. **Parameter counts**: tied **20,014**, untied **49,204**.
3. **The longer run must actually train**: each arm's final training loss at 8 epochs must be **lower
   than the same arm's final loss at 4 epochs**. If it is not, the extra budget did nothing mechanical
   and the comparison is reported as a failed manipulation, not as a null result about quality.
4. **25 TNO pairs** scored per run, the same image list as the 4-epoch arms.
5. **No duplicate runs**: the five seeds in each arm differ on every metric.
6. **Seed fingerprint audit passes**, and each seed writes its own checkpoint path.

## What is *not* done here

- **No third budget.** If 8 epochs shows an effect, the curve is extended by a **separately registered**
  run; choosing a budget after seeing which one helps is the error this thesis documents.
- **No learning-rate change.** Tuning the schedule to rescue an arm would make the arms incomparable.
- **No depth change.** Free depth is its own experiment, registered separately.

---

## Cost

10 runs ≈ **22 GPU-h**, inside one weekly quota alongside Run 11's k0 arm.
