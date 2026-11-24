# Run 15 — pre-registration · does the MI/VIF inversion follow the input skip? Two published methods without one

**Written 2026-09-25, committed on its own before any Run 15 script or kernel exists.** Nothing below may be
edited after the first result; corrections go in a dated block at the end.

---

## Why this run exists

The paper's most consequential finding is that **MI and VIF score an untrained network above a trained
one** — on CrossFuse, and on CDDFuse's released model. The paper's account of *why* is mechanical: both
decoders end by adding the visible image back to the output (`sigmoid(conv(features) + inp_img)`), so an
untrained network returns a copy of its visible input (r = 0.997), and MI and VIF reward that copy.

That account makes a **prediction a reviewer can check and we have not**: in a published method whose
output has **no path from the visible input**, an untrained network cannot return a copy, and MI and VIF
should then detect it like the other seven metrics. So far the only such case is LRRNet — one architecture,
and one we trained ourselves. A reviewer's first objection to the paper is "two architectures"; the second
will be "is the mechanism real or post hoc?". This run answers both with **released** models and no training.

**Two methods, chosen before any output exists, because both have public code and weights in their own
repositories and neither has an input-to-output skip** (read from the source, 2026-09-25):

| method | venue | forward pass, as released | skip from input to output |
|---|---|---|---|
| **DenseFuse** (`hli1221/densefuse-pytorch`, `models/densefuse_gray.model`) | IEEE TIP 2019 | encoder → `(en_ir + en_vi) / 2` → decoder of plain convolutions | **none** |
| **SeAFusion** (`Linfeng-Tang/SeAFusion`, `model/Fusion/fusionmodel_final.pth`) | Information Fusion 2022 | two encoders → concatenation → four decoder convolutions → `tanh(conv)/2 + 0.5` | **none** (residuals only inside encoder blocks) |

**This run can overturn the paper's explanation.** If MI or VIF rank an untrained network above the released
one *without* a copy, the explanation in §7 is incomplete and the metric critique becomes broader — that
outcome is written down here as plainly as the one that confirms it.

## What is run

The design of the CDDFuse transfer test (`studied_papers/CDDFuse/MECHANISM_RESULTS.md`), unchanged in every respect
that affects a reading:

- **Arms.** RELEASED: the checkpoint above, loaded with `strict=True`. UNTRAINED: the same network built
  after seeding `torch`, `numpy` and `random` with **42, 43, 44, 45, 46** — no weight loaded, no gradient step.
- **Images.** The same **25 TNO pairs** the CDDFuse test used, fused with **each method's own test-time
  preprocessing and post-processing**, as its released test script does it.
- **Scoring.** The same evaluator as the CDDFuse test (CDDFuse's `utils/Evaluator.py`, SSIM default as
  pinned there), nine metrics.
- **No training, no tuning.** One forward pass per image per arm. Expected cost: well under 1 GPU-h.

---

## The readings, fixed before the run

Applied **to each method separately**; the two are reported side by side.

**Copy criterion** (as for CDDFuse): mean Pearson r over the 25 pairs between the output and each of VI, IR,
(IR+VI)/2 and max(IR, VI). A copy is declared when **VI is the most-resembled reference in at least 4 of 5
untrained seeds, with mean r ≥ 0.90**.

**Inversion criterion** (as for CDDFuse): the untrained mean over five seeds exceeds the released value on
MI or on VIF by a **one-sample t-test across the five seeds, two-sided 5 %, df = 4, |t| > 2.776**. (Five
untrained seeds against one deterministic released model: df = 4 is the exact value here, not a fixed
threshold carried over from another design.)

- **N1 — CONSISTENT: no copy, and neither MI nor VIF inverts.** Reported as: *the inversion follows the input
  skip.* With LRRNet, three architectures without a skip do not invert and two with one do. §7's explanation
  gains an out-of-family test on released models, and the paper's scope sentence names all five.
- **N2 — INVERSION WITHOUT A COPY: MI or VIF ranks the untrained network higher although its output is not a
  copy.** Reported as: *the skip account is incomplete* — the metrics reward something else an untrained
  network produces. §7 is rewritten to say so, the metric critique is stated as broader than the mechanism,
  and the untrained outputs' properties (contrast, mean intensity, r to each input) are reported to show
  what the metrics are rewarding.
- **N3 — A COPY WITHOUT A SKIP.** Reported as: *the copy does not need an explicit skip in these networks*;
  the mechanism sentence is rewritten around what produces the copy, from the source.
- **Mixed** (the two methods land in different outcomes): each is reported under its own outcome, and the
  paper's sentence covers only what both support.

Reported either way, not used to decide: all nine metrics' *t* for each method; the released checkpoint's own
r to VI and IR; whether the other seven metrics detect the untrained network (they did on every architecture
so far except SF and AG on LRRNet).

---

## Kill criteria — checked before any metric is read

1. **The released checkpoint loads with `strict=True`**: no missing or unexpected keys. A partial load would
   make "released" a half-untrained network.
2. **The released model fuses**: its mean r to VI **and** to IR are both ≥ 0.30 over the 25 pairs. A released
   model that ignores a modality means the preprocessing is wrong, and nothing is read.
3. **25 TNO pairs** fused and scored per arm, the same list as the CDDFuse test.
4. **The five untrained seeds give distinct outputs** (no two identical on any image).
5. **Outputs finite and in range** after each method's own post-processing.

**What this cannot do**: reproduce DenseFuse's or SeAFusion's published numbers — their papers score other
test sets with other evaluators (the scale mismatch this paper documents). Kill criterion 2 is what stands in
for K1 here, and the report says so.

## Cost

Two methods × six arms × 25 forward passes: **well under 1 GPU-h**.

---

## Correction — 2026-09-25, after a local CPU test tripped kill criterion 4 for DenseFuse, before any metric was read

**What happened.** The five untrained DenseFuse seeds gave identical outputs on up to 25 of 25 images. It is not
a seeding failure: the raw float outputs differ by seed (maximum 1.33, 0.00 and 0.28 of 255 for seeds 42, 43,
44 on the first pair; seed 43's last ReLU is dead everywhere). DenseFuse's released post-processing is
`clamp(0, 255)` then `uint8`, with no normalisation, so an untrained DenseFuse is rendered as an all-black image
(values 0 and 1 at most) — the same image for every seed.

**Decision, fixed now, before any metric is read.** Kill criterion 4 is kept as written and it fails for
DenseFuse, so **DenseFuse receives no N1/N2/N3 reading.** The question this run asks — does MI or VIF prefer an
untrained output that is *not* a copy? — cannot be put to an output that carries no image. It is reported as a
property of the architecture under its own pipeline: *an untrained DenseFuse emits no image at all, so it
cannot pass for a fused one.* No normalisation is added after the fact to rescue the arm; doing so would
change the method's released pipeline because the registered one gave an inconvenient answer.

**SeAFusion is unaffected** (its released pipeline min-max normalises every output) and carries the registered
reading alone. If a second no-skip method is wanted, it is a separately registered run.
