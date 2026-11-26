# Run 32 — pre-registration · do the metric verdicts agree with a downstream task?

**Written 2026-11-25, committed on its own before any Run 32 kernel exists.** Nothing below may change after the
first fused image is written; corrections go in a dated block at the end.

## The question

The paper's central metric finding is that on CrossFuse, MI and VIF rank an untrained network — whose output is a
copy of the visible image — above a trained one, and the paper argues that a fused image which drops a person seen
only in the infrared band should not help a detector find them. That argument is untested. This run tests it, and
asks the same of LRRNet's positive control and of the paper's one quality gain (tied depth 8 over untied).

## What is run

**Data.** LLVIP's **test** split (night-time pedestrians; LRRNet was trained only on LLVIP's train split, CrossFuse
never on LLVIP), with its official person annotations. A fixed subset of **1,000 pairs**, drawn with
`random.Random(20261125).sample` over the sorted test file names **before any fusion**, downscaled 2× to 640×512 (the
same downscale LRRNet's training uses); boxes scaled with the images.

**Fusion (GPU kernel), the existing checkpoints and code paths only — nothing is trained:**

| family | arm | checkpoints |
|---|---|---|
| CrossFuse | trained | A1b, A1c, A1d, PC45, PC46 (the positive control's trained arm) |
| CrossFuse | untrained, skip on | U42–U46 |
| CrossFuse | untrained, skip off | U42–U46 with `--no-skip` (Run 30) |
| LRRNet | published d4 | seeds 42–46 |
| LRRNet | untrained d4 | seeds 400–404 (`torch.manual_seed`, zero steps, as Run 25) |
| LRRNet | tied d8 / untied d8 | seeds 62–66 each (Run 17) |

References (one image each, no variance): the infrared band alone and the visible band alone.

**Detection, fixed:** YOLOv8s pretrained on COCO (`ultralytics==8.3.0`, `yolov8s.pt`), **never fine-tuned**,
class *person* only, every fused image replicated to three channels at 640×512, confidence threshold 0.001, NMS IoU
0.6. Score: **AP50** for *person* (all-point interpolated precision–recall, IoU ≥ 0.5 against the scaled ground
truth), one value per network.

**Stated bias, fixed now:** the detector was trained on visible-light COCO images, so it favours images that look
visible. That bias works *for* the visible-copy arm, so it works against the paper's argument in D1.

## The readings, fixed before the run

Welch's *t* at exact df on per-network AP50, five networks per arm; Holm across the three registered contrasts.

**D1 — CrossFuse, trained vs untrained with the skip on (the copy MI and VIF prefer).**
- **AGREES WITH THE CONTROL**: trained AP50 higher, surviving Holm. The paper may say that a downstream detector
  ranks the trained network above the copy that MI and VIF prefer.
- **AGREES WITH MI/VIF**: untrained AP50 higher, surviving Holm. The paper's argument that the copy is worse for the
  task is withdrawn; the inversion is then reported as a disagreement between the measures and the control, not as
  the measures preferring a worse image.
- **INCONCLUSIVE**: neither survives Holm; reported as such.

**D2 — LRRNet, published d4 vs untrained d4.** TRAINED HIGHER / UNTRAINED HIGHER / INCONCLUSIVE, as in D1.

**D3 — LRRNet, tied d8 vs untied d8 (the paper's one quality gain).**
- **TRANSFERS**: tied higher, surviving Holm. **DOES NOT TRANSFER**: no difference survives; the paper says the metric
  gain is not shown to carry to detection. **REVERSES**: untied higher, surviving Holm; reported plainly.

**Also reported, descriptively, no verdict:** the skip-off arm's AP50; the infrared-only and visible-only references;
whether each arm's ordering on AP50 matches its ordering on MI and VIF from the paper.

**Kill criteria:** a checkpoint that fails its parameter-count or integrity check (as in Runs 16 and 25); any fused
image that is non-finite or blank for every pair of a network; fewer than five networks scored in any arm.

## Correction — 2026-11-26, after the first push stopped and before any network is fused

The first push (`lrrnet-run32-detect`, version 1) stopped at the checkpoint integrity check, as the kill criteria
require: the copy of `CrossFuse_RSM3_A1b.pth` that the attached kernel output (`tier3-a1b`) now holds is not the file
the paper's A1b rows were scored from (its SHA-256 differs from that file, `3859d38d…`). No network had been fused. The
only values printed were the two single-band references (infrared-only and visible-only AP50), which are descriptive
and carry no verdict.

**Decision, fixed now:** the four checkpoints that came from kernel outputs (A1b, A1c, A1d, PC46) are attached instead
as a private dataset, `crossfuse-run32-ckpts`, holding the byte-identical files the paper's rows were scored from (the
same SHA-256 values, still checked in the kernel). Nothing else in the registration or the kernel changes.
