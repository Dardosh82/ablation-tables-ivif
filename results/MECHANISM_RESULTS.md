# CDDFuse transfer test of the residual-copy mechanism — result of record

**Registered** in the docstring of `scripts/cddfuse_untrained_mechanism.py`, committed alone (1670e41) before
the first run. **Run** on Kaggle (`mostafanada82/cddfuse-untrained-mechanism`, Tesla T4, torch 2.10.0+cu128)
against MMIF-CDDFuse at commit `73681d28`. Output: `results_mechanism/kaggle/` (log and summary CSV).

## Before the result: three runs, two crashes, no reading taken from either

| push | outcome | fix (commit) |
|---|---|---|
| 1 | `KeyError: 'CDDF_Encoder'` while loading the checkpoint — nothing computed | the released file names its modules `DIDF_Encoder`/`DIDF_Decoder` while CDDFuse's own `test_IVF.py` asks for `CDDF_*`; accept either. The PNG round-trip also now rounds as skimage 0.19.2 → imageio does, instead of truncating (2f33d40) |
| 2 | `ValueError` inside `Evaluator.SSIM` on the first image — nothing computed | current skimage refuses float SSIM without `data_range`; restored 0.19.2's default (dtype range, 2 for float) (e4fefff) |
| 3 | completed | — |

The readings were not edited. K1 is what decides whether these environment shims reproduce CDDFuse.

## K1 — the released checkpoint reproduces the published TNO row: **PASSED on all eight**

| | EN | SD | SF | MI | SCD | VIF | Qabf | SSIM |
|---|---|---|---|---|---|---|---|---|
| ours | 7.1203 | 46.0024 | 13.1520 | 2.1864 | 1.7607 | 0.7683 | 0.5393 | 1.0295 |
| published (README) | 7.12 | 46.0 | 13.15 | 2.19 | 1.76 | 0.77 | 0.54 | 1.03 |

No earlier record in this project reproduces CDDFuse's published TNO row; this does, to the printed precision
(SSIM here is the sum of two SSIMs, as withdrawn claim #11 records).

## Copy criterion — **MET**

Mean Pearson r over 25 pairs between the output and each reference:

| arm | VI | IR | (IR+VI)/2 | max(IR,VI) | most resembled |
|---|---|---|---|---|---|
| released | 0.782 | 0.243 | 0.914 | 0.947 | max |
| untrained, seed 42 | 0.998 | −0.247 | 0.668 | 0.660 | VI |
| untrained, seed 43 | 0.997 | −0.242 | 0.673 | 0.657 | VI |
| untrained, seed 44 | 0.998 | −0.282 | 0.640 | 0.636 | VI |
| untrained, seed 45 | 0.996 | −0.215 | 0.691 | 0.672 | VI |
| untrained, seed 46 | 0.996 | −0.242 | 0.671 | 0.662 | VI |

VI wins 5 of 5 untrained seeds, mean r = 0.997 (CrossFuse: 0.996–0.998; trained ≈ 0.79 on both).

## Inversion criterion — **MET**

Untrained mean ± SD over 5 seeds against the released checkpoint, one-sample t, df = 4, critical 2.776:

| metric | untrained | released | t |
|---|---|---|---|
| EN | 7.0145 ± 0.0325 | 7.1203 | −7.29 |
| SD | 46.4067 ± 0.9583 | 46.0024 | +0.94 |
| SF | 10.7923 ± 0.3857 | 13.1520 | −13.68 |
| **MI** | **3.4738 ± 0.1799** | **2.1864** | **+16.00** |
| SCD | 0.9278 ± 0.2198 | 1.7607 | −8.47 |
| **VIF** | **0.9450 ± 0.0260** | **0.7683** | **+15.17** |
| Qabf | 0.5260 ± 0.0041 | 0.5393 | −7.23 |
| SSIM | 0.9036 ± 0.0199 | 1.0295 | −14.17 |

## Reading, as registered: **H1 TRANSFERS**

The residual-copy inversion is a property of this decoder design, found in two published methods: CDDFuse,
whose decoder ends `sigmoid(conv(features) + inp_img)` with the visible image, and CrossFuse, which inherits
it. An untrained network with no gradient step returns the visible image (r = 0.997) and MI and VIF score it
above the released, trained model; the other six metrics rank the trained model higher (SD does not
separate them). LRRNet, without the skip, does not invert. §7's scope sentence widens from one network
to two.

Reported either way, not used to decide: the released checkpoint's own r to VI is 0.782; its output most
resembles max(IR, VI) (0.947).

## Post-hoc observation, 2026-09-14 — the untrained side is not independent of CrossFuse's (does not change the reading)

Found while propagating the result, and recorded before any document was edited. The untrained CDDFuse arm
matches the untrained CrossFuse arm (`TIER3_RESULTS.md`, Run 7) on all eight shared metrics to within 0.002:
EN 7.0145 against 7.0140, SD 46.4067 / 46.4051, SF 10.7923 / 10.7923, MI 3.4738 / 3.4738, SCD 0.9278 ±0.2198 /
0.9278 ±0.2198, VIF 0.9450 / 0.9450, Qabf 0.5260 / 0.5258, SSIM 0.9036 / 0.9041 — the per-seed spread included.

The reason is in the code. CrossFuse's `network.py` carries CDDFuse's `Restormer_Encoder` and
`Restormer_Decoder`, and both `make_untrained.py` and this script build those two modules first after seeding
torch with the same seeds. The encoder and decoder therefore start from the same weights seed for seed (read
from the code and implied by the agreement; not compared tensor by tensor), only the fusion layers between them
differ, and the residual skip makes the output a near-copy of the visible image in both. The small differences
are the fusion layers and the uint8 round-trip.

What this does and does not change:

- **The registered reading stands: H1.** The copy and inversion criteria were met as written.
- **The untrained arm is not an independent replication.** It is, to within 0.002, the same object measured
  in Run 7. The new evidence is the **trained reference**: CDDFuse's released checkpoint, reproduced to its
  published row by K1, is ranked below that copy by MI (2.19 against 3.47) and VIF (0.768 against 0.945), as
  CrossFuse's trained models were.
- **So the correct wording is narrower than "the mechanism generalises".** It is: the inversion is a property
  of one decoder lineage, and holds against the released models of both published methods that use it. Whether
  an unrelated design with an input skip inverts the same way remains untested.
