# LRRNet — results of record

> ⚠️ **Correction, 2026-09-13 — withdrawn claim #13.** The screen that borrowed CrossFuse's σ used that
> method's population SD. With the sample SD the borrowed readings are **16.8 σ** (VIF, 4 vs 6) and
> **3.90 σ** (SD), not 20.6 and 4.77, and LRRNet is 26× (MI) and 25× (VIF) noisier than CrossFuse, not
> 32× and 31×. The borrowed estimate is still wrong by more than twentyfold (16.8 against 0.74).

**Case study 2** of *Variance-Aware Design and Evaluation of Deep Learning-Based Infrared and
Visible Image Fusion*.

Paper: H. Li, T. Xu, X.-J. Wu, J. Lu, J. Kittler, *"LRRNet: A Novel Representation Learning Guided
Fusion Network for Infrared and Visible Images"*, **IEEE TPAMI 45 (2023) 11040–11052** ·
arXiv:2304.05172 · code: <https://github.com/hli1221/imagefusion-LRRNet>

This file holds the **numbers of record** and how each was produced. Narrative lives in
[EXPLAIN.md](../../thesis/EXPLAIN.md) §9; operational state in [CONTEXT.md](../../CONTEXT.md).

**Last updated:** 2026-09-02

---

## Why this paper

Not because it was cheapest — because it is a **different architectural family**. CrossFuse and
CDDFuse are Restormer derivatives; LRRNet is a CNN built around a learnable low-rank representation
block. A result that holds in both families is a claim about the field rather than about one
architecture. CDDFuse was measured first and ruled out at **37.5 h per run**; SeAFusion was ruled
out from its config files alone, at zero cost.

| | LRRNet | CDDFuse |
|---|---|---|
| iterations per run | 4 × 1,503 = **6,012** | 120 × 813 = 97,560 |
| measured cost | **1.10 h** | 37.5 h |
| parameters | 49,204 | 1,188,272 |

---

## Configuration

The published one, unchanged: `num_block = 4`, `epochs = 4`, `batch_size = 8`, 128×128 grayscale,
`lr = 1e-5`, `lam2 = 1.5`, `w_ir = 3.0`, `lam3_gram = 2000`. Evaluation on the **25 TNO pairs**,
because Table IV is published on TNO, scored with **CrossFuse's `Evaluator`** — the same code behind
every CrossFuse number in this project.

### Deviations, to be stated wherever these numbers are used

| Deviation | Why it was unavoidable |
|---|---|
| **LLVIP (12,025 pairs) instead of KAIST (20,000)** | `load_data` takes `sorted(listdir(path))[:20000]` from a folder whose construction is never described, and no file list is released — **the authors' exact 20,000 images are unrecoverable even with the full KAIST download**. Kaggle also carries no KAIST in IR/VIS-separated form. |
| torchvision ImageNet VGG16 for the loss network | LRRNet ships no `vgg16.pth`; these are the standard weights and match the positional layout `init_vgg16` expects |
| seeding added to the source | the released code seeds **nothing**; without this, "five seeds" is not a meaningful phrase |
| cudnn left nondeterministic | deliberate — σ must include what a reader re-running the released code would meet |

---

## Zero-GPU findings (2026-09-01)

### The block-count ablation is a capacity ablation

`scripts/check_numblock_capacity.py`, CPU only:

| num_block | parameters | vs 4 |
|---|---|---|
| 2 | 29,744 | 60.5% |
| **4 (chosen)** | **49,204** | 100% |
| 6 | 68,664 | 139.5% |
| 8 | 88,124 | 179.1% |

**2.96× spread.** Block count and capacity move together, so no row difference can be attributed to
representational depth — the same defect as CrossFuse Table VI rows III/IV, in a different family.

### IVIF papers do not share a metric scale

LRRNet reports **MI = 13.7**; this project measures **2.13** on CrossFuse and **2.50** on LRRNet
itself for the same metric name. A 5.5–6.4× gap is a different normalisation convention, not a
different result. `SSIMm` and `VIFm` are explicitly *modified* variants; CDDFuse publishes
**SSIM = 1.03**, above standard SSIM's maximum. **Cross-paper comparison of published tables is
unsound** — which is why every row of Table IV is being re-run here rather than screened.

---

## Stage 1 — σ_LRRNet (2026-09-01, `mostafanada82/lrrnet-stage1`)

Five seeds, 66.3 min each. `scripts/analyse_stage1.py`. Per-run rows in `lrrnet_metrics.csv`.

| metric | mean | **σ** | CV % | σ vs CrossFuse TNO |
|---|---|---|---|---|
| EN | 6.9727 | **0.0254** | 0.36 | 1.9× |
| MI | 2.5021 | **0.1357** | 5.43 | **32.3×** |
| SD | 43.5032 | **0.8549** | 1.97 | 2.9× |
| SF | 10.7123 | 0.3715 | 3.47 | 1.6× |
| AG | 3.9128 | 0.1494 | 3.82 | 2.3× |
| VIF | 0.7646 | **0.0401** | 5.24 | **30.8×** |
| SCD | 1.2607 | 0.0469 | 3.72 | 1.5× |
| Qabf | 0.4325 | 0.0169 | 3.91 | 13.0× |
| SSIM | 0.9020 | 0.0152 | 1.68 | 2.5× |

A genuine sample standard deviation at **n = 5** (ddof = 1), not the range-based estimate used for
CrossFuse (n = 3, range/1.693). LRRNet is noisier on every metric — expected at 4 epochs against
CrossFuse's 160, and a property of the training budget rather than a defect of the method.

### Table IV under σ_LRRNet — preliminary: 15 of 15 cells inside 3σ

Judged by **coefficient of variation** applied to their level, because absolute levels differ.

| metric | 4 vs 6 | 4 vs 8 | 4 vs 2 |
|---|---|---|---|
| En | 1.19σ | 1.11σ | 2.66σ |
| SD | 0.90σ | 0.98σ | 2.24σ |
| MI | 0.08σ | 0.07σ | 0.18σ |
| SSIMm | 0.09σ | 0.32σ | 0.45σ |
| VIFm | 0.74σ | 0.49σ | 1.49σ |

The paper's justification is *"4 blocks obtains five best values"* — a count of wins with no
magnitude and no variance. The numbers support *"the network works"*; they do not support *"four
blocks is the right number"*, which is what the paper concludes and the setting it then fixes for
every other experiment in it.

⚠️ **Preliminary.** It assumes relative noise transfers between a KAIST-trained and an LLVIP-trained
model. Stages 2–4 retrain rows 6, 8 and 2 and remove the assumption.

### ⚠️ Our own proxy was wrong by more than an order of magnitude

| comparison | screened with CrossFuse's σ | measured with LRRNet's own σ |
|---|---|---|
| VIF, 4 vs 6 | 20.6σ — "clearly real" | **0.74σ — noise** |
| SD, 4 vs 6 | 4.77σ — "real" | **0.90σ — noise** |

The borrowed noise level made the problem look **smaller** than it is. This is the strongest
available argument for measuring σ per method rather than borrowing one — made against our own
earlier estimate, not somebody else's.

### ⭐ "Set the seed" is not a reproducibility protocol

The first stage-1 attempt crashed after seed 42 had trained and fused; its checkpoint survived in
the notebook output and was scored separately. The re-run trained seed 42 again. Two runs, same
seed, same code, same data:

| | EN | MI | SD | SCD |
|---|---|---|---|---|
| run A | 6.9433 | 2.3030 | 43.8387 | 1.2849 |
| run B | 6.9546 | 2.3353 | 43.9216 | 1.2653 |
| as % of σ | 44% | 24% | 10% | 42% |

Median 16% of σ, up to 44%. Identical weights and identical data order still diverge, because cudnn
is left free to choose algorithms. **A paper reporting a single run together with its seed would
still not be reproducible to the precision its own ablation table claims to resolve.**

---

## Stage 2 — `num_block = 6`, seeds 42–44 · ✅ complete (2026-09-02)

**The decisive comparison.** Table IV selects 4 blocks; 6 is its nearest rival, and that choice is
then fixed for every other experiment in the paper. Stage 1 could only judge the gap by borrowing
relative noise across training corpora. This borrows nothing: both rows are our own training on the
same corpus, scored by the same evaluator on the same 25 TNO pairs, against σ measured on those very
runs. `scripts/analyse_stage2.py`.

| metric | nb=4 mean (n=5) | nb=6 mean (n=3) | difference | pooled σ units | Welch *t* | verdict |
|---|---|---|---|---|---|---|
| EN | 6.9727 | 6.9874 | −0.0147 | 0.51σ | −0.64 | no difference |
| MI | 2.5021 | 2.4299 | +0.0723 | 0.64σ | 1.14 | no difference |
| SD | 43.5032 | 44.8192 | −1.3160 | 1.51σ | −2.04 | no difference |
| SF | 10.7123 | 10.2883 | +0.4240 | 1.20σ | 1.74 | no difference |
| AG | 3.9128 | 3.8232 | +0.0896 | 0.67σ | 1.05 | no difference |
| VIF | 0.7646 | 0.7198 | +0.0448 | 1.36σ | 2.45 | no difference |
| SCD | 1.2607 | 1.2364 | +0.0242 | 0.56σ | 0.82 | no difference |
| Qabf | 0.4325 | 0.4151 | +0.0174 | 1.21σ | 2.02 | no difference |
| SSIM | 0.9020 | 0.8968 | +0.0052 | 0.40σ | 0.68 | no difference |

> **0 of 9 metrics clear the 5% level. 9 of 9 sit inside 3 pooled σ.**
> Two-sided critical value 2.776 at the Welch–Satterthwaite df; nine metrics are tested at once, so
> a claim would need several agreeing, not one.

### ⭐ The finding, stated exactly

**Counting wins the way the paper does, 4 blocks leads on 7 of 9 metrics — and not one of those
leads is separable from run-to-run noise.** That is the whole problem with a table justified by
*"obtains five best values"*: with nine correlated metrics and one run each, a lead is nearly
guaranteed to exist and says nothing about whether it is real.

Six blocks is even nominally ahead on EN and SD — also inside the noise.

**What this does and does not show.** It does not show the architecture is wrong. It shows the
published table **cannot support the choice it is used to justify**, and that the choice propagates,
because every other experiment in the paper runs at `num_block = 4`.

**A point in the paper's favour, recorded as plainly as the criticism.** 6 blocks costs 68,664
parameters against 49,204 — **1.40× more capacity for no measurable gain**. Choosing the smaller
model is the right call. It is simply not the reason the paper gives, and not what its table shows.

**Calibration of stage 1's preliminary estimate.** Using relative noise borrowed across corpora,
stage 1 put this comparison at 1.19σ on EN. Measured directly: **0.51σ**. Same verdict, right
magnitude — unlike the CrossFuse-based proxy it replaced, which was out by more than 20×. Borrowing
σ from the *same architecture* under a different corpus is defensible; borrowing it from a different
architecture entirely is not.

**Noise levels are comparable between the two configurations** (e.g. σ_EN 0.0254 vs 0.0347,
σ_SD 0.855 vs 0.897), so pooling is sound and the n=3 rows are not systematically noisier.

## Stage 3 — `num_block = 8`, seeds 42–44 · 🟡 first attempt OOMed (2026-09-02)

```
nb8_seed42
train exit 1 after 0.1 min
torch.OutOfMemoryError: Tried to allocate 32.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 38.81 MiB is free.
```

### ⭐ This is a finding, not merely an obstacle

**`num_block = 8` — one of the four settings LRRNet's own Table IV compares — does not train on a
16 GB card as released.** Two properties of the released code combine to make the second GPU
useless: `training_fusion_lrr.py` pins `CUDA_VISIBLE_DEVICES = "0"` at import, and its
`DataParallel` line is **commented out**. So the published ablation contains a configuration the
published code cannot run on commonly available hardware — the same class of defect recorded for
CDDFuse's `batch_size = 8` (C1), now in the second case study.

Note this was not visible at nb=4 or nb=6: memory grows with block count because the whole
computation graph is held for the backward pass, on top of three VGG16 forward passes per
iteration.

### The fix preserves the published configuration exactly

Options considered, and why one was chosen:

| option | verdict |
|---|---|
| smaller batch | ❌ a real deviation — nb=8 would no longer be comparable to nb=4 and nb=6 |
| gradient checkpointing | ⚠️ identical arithmetic, but requires editing `net_lista.py` itself |
| **restore the author's own `DataParallel` line** | ✅ **chosen** |

**Why DataParallel is not a deviation.** Inspection of `net_lista.py` finds **no BatchNorm, no
InstanceNorm, no LayerNorm and no Dropout** — no batch-dependent layer of any kind. Splitting a
batch of 8 across two devices and averaging the gradients therefore produces the *arithmetically
identical* update to computing it on one device. The line is the authors' own, present in the source
and commented out.

Implemented as two further patches in `scripts/patch_lrrnet.py` (now 11, all dry-run verified),
both **defaulting to the released behaviour** so stages 1–2 are unaffected:
`LRR_GPUS` (visible devices, default `"0"`) and `LRR_MULTI_GPU` (`"1"` restores DataParallel).
The run prints `[GPUS] n parallel=True/False` so the configuration is on the record.

⚠️ DataParallel prefixes `state_dict` keys with `module.`; `fuse_lrrnet.py` already strips that
prefix — a defence written for the *opposite* bug in the released inference code, which now pays off
in the other direction.

### ✅ Re-run succeeded — stage 3 complete (2026-09-02)

`[GPUS] 2 parallel= True` on all three runs, **63.3 min each**, loss converging normally
(294,901 → 56,166 · 101,648 → 57,609 · 285,839 → 56,043). The fix held.

`scripts/analyse_table_iv.py` now supersedes `analyse_stage2.py` and judges every trained row at
once:

| metric | nb=4 (n=5) | nb=6 (n=3) | nb=8 (n=3) |
|---|---|---|---|
| EN | 6.9727 ±0.0254 | 6.9874 ±0.0347 | 6.9603 ±0.0301 |
| MI | 2.5021 ±0.1357 | 2.4299 ±0.0325 | 2.4665 ±0.0151 |
| SD | 43.5032 ±0.8549 | 44.8192 ±0.8969 | 43.5392 ±0.6714 |
| SF | 10.7123 ±0.3715 | 10.2883 ±0.3092 | 10.5957 ±0.3179 |
| AG | 3.9128 ±0.1494 | 3.8232 ±0.0908 | 3.8951 ±0.0842 |
| VIF | 0.7646 ±0.0401 | 0.7198 ±0.0065 | 0.7457 ±0.0124 |
| SCD | 1.2607 ±0.0469 | 1.2364 ±0.0357 | 1.2378 ±0.0405 |
| Qabf | 0.4325 ±0.0169 | 0.4151 ±0.0071 | 0.4226 ±0.0039 |
| SSIM | 0.9020 ±0.0152 | 0.8968 ±0.0061 | 0.9020 ±0.0028 |

**4 vs 8, in pooled σ:** EN 0.46 · MI 0.32 · SD 0.05 · SF 0.33 · AG 0.13 · VIF 0.57 · SCD 0.51 ·
Qabf 0.71 · **SSIM 0.00**. Largest Welch *t* = 1.25 against a 2.776 critical value.

> ### 0 of 9 against 6 blocks · 0 of 9 against 8 blocks
>
> **The ablation is uninformative across its whole upper range, not merely at one neighbour.**

8 blocks sits *closer* to the chosen setting than 6 does — most differences under 0.6σ, SSIM
identical to four decimal places — while carrying **1.79× the parameters**. Counting wins the way
the paper does, 4 blocks leads on **7 of 9** against each rival, and not one lead is separable from
noise.

**Still in the paper's favour, and now more clearly:** the chosen configuration is the smallest of
three indistinguishable ones, which is the right engineering call. It is not the reason the paper
gives, and not what its table demonstrates.

## Stage 4 — `num_block = 2`, seeds 42–44 · ✅ complete (2026-09-03)

45.1 min per run, back on the **released single-device path** — `num_block = 2` is the smallest model
in the ablation (29,744 parameters, below the chosen setting's 49,204), so it needs no exception.
Stages 1, 2 and 4 therefore all ran exactly as published; only the row that could not otherwise be
trained departs from it.

### The completed table

| metric | nb=2 (n=3) | nb=4 (n=5) | nb=6 (n=3) | nb=8 (n=3) |
|---|---|---|---|---|
| EN | 7.0016 ±0.0389 | 6.9727 ±0.0254 | 6.9874 ±0.0347 | 6.9603 ±0.0301 |
| MI | 2.4569 ±0.0826 | 2.5021 ±0.1357 | 2.4299 ±0.0325 | 2.4665 ±0.0151 |
| SD | 43.5300 ±0.7931 | 43.5032 ±0.8549 | 44.8192 ±0.8969 | 43.5392 ±0.6714 |
| SF | 10.9792 ±0.2055 | 10.7123 ±0.3715 | 10.2883 ±0.3092 | 10.5957 ±0.3179 |
| AG | 4.0782 ±0.0881 | 3.9128 ±0.1494 | 3.8232 ±0.0908 | 3.8951 ±0.0842 |
| VIF | 0.7758 ±0.0423 | 0.7646 ±0.0401 | 0.7198 ±0.0065 | 0.7457 ±0.0124 |
| SCD | 1.3085 ±0.0176 | 1.2607 ±0.0469 | 1.2364 ±0.0357 | 1.2378 ±0.0405 |
| Qabf | 0.4407 ±0.0165 | 0.4325 ±0.0169 | 0.4151 ±0.0071 | 0.4226 ±0.0039 |
| SSIM | 0.9136 ±0.0075 | 0.9020 ±0.0152 | 0.8968 ±0.0061 | 0.9020 ±0.0028 |

**4 vs 2, in pooled σ:** EN 0.95 · MI 0.37 · SD 0.03 · SF 0.82 · AG 1.25 · VIF 0.27 · SCD 1.21 ·
Qabf 0.49 · SSIM 0.89. Largest |Welch *t*| = 2.05 against a 2.776 critical value.

> ## ~~0 of 27 comparisons across the whole table reach the 5% level~~
>
> ⚠️ **Superseded 2026-09-13 by Run 9.** Every row taken to n = 8: **1 of 27** (SCD, 4 vs 2, Welch
> *t* = −2.452, critical 2.262) — at the chance rate: a table with at least one such cell appears 56 % of the time under random re-assignment of whole runs, and none survives a Holm or max-t correction. The count above is withdrawn as
> claim #10; the full reading is in [`RUN9_PREREGISTRATION.md`](RUN9_PREREGISTRATION.md).
>
> **Re-read 2026-09-13, after an external review** (`scripts/analyse_run9_multiplicity.py`; no number
> above changes). Holm **0 of 27** · Benjamini–Hochberg **0 of 27** · max-|*t*| permutation over whole
> runs, 20,000 draws, **0 of 27** (family-wise critical |*t*| = 3.72; SCD cell *p* = 0.40). Under that
> null a table shows ≥ 1 uncorrected cell **56 %** of the time (mean 1.26). TOST at ±1.75 σ — the
> resolution registered for Run 9, used here as the margin and labelled as such: **23 of 27
> equivalent**, 4 inconclusive (4v2 SF, AG, SCD; 4v6 SF). *Resolved by Run 10 (end of this file): at
> n = 20 all four are equivalent, and the SCD cell did not replicate on twelve new seeds.*
>
> Four settings spanning **2.96×** in model size (29,744 → 88,124 parameters), every row trained
> here, all judged against σ measured on these same runs. **No metric distinguishes any setting from
> the chosen one.** The paper selects 4 blocks and then fixes it for every other experiment it
> reports; that choice is not supported by any comparison in the table it comes from.

### ⭐ And the published ordering does not survive either

2 blocks is the setting LRRNet's own numbers place **clearly worst** — the one row the published
table separates from the rest. Retrained, it is nominally **ahead of the chosen configuration on 8
of 9 metrics** (at n = 3; **7 of 9** at n = 8 after Run 9) (4 blocks leads on just 1 of 9 against it), while using **60% of the parameters**.

Everything stays inside the noise band, so **nothing is claimed about 2 blocks being better** — that
would repeat the paper's own error in the opposite direction. The claim is narrower and firmer:
**retraining does not reproduce even the published ordering**, so the table's ranking carries no
more information than its gaps do.

**The paper's engineering instinct remains sound and is recorded as such:** with four
indistinguishable configurations, choosing a small one is right. It is simply not the reason the
paper gives, and not what its table demonstrates.

## Case study II — cost summary

| Stage | Runs | Hours |
|---|---|---|
| Screening probes (CDDFuse, LRRNet) | 2 | 0.2 |
| Pilot | 1 | 0.1 |
| Stage 1 — 4 blocks × 5 seeds | 5 | 5.5 |
| Stage 2 — 6 blocks × 3 seeds | 3 | 3.3 |
| Stage 3 — 8 blocks × 3 seeds | 3 | 3.2 |
| Stage 4 — 2 blocks × 3 seeds | 3 | 2.3 |
| **Total** | **17** | **≈14.6** |

Once all four rows exist, Table IV is fully re-measured with error bars: our rows, our evaluator,
our σ, and no proxy anywhere.

---

## Reproducing any of this

```bash
python patch_lrrnet.py --check          # dry-run the 9 source patches against upstream
python make_vgg16_weights.py --dry-run  # verify the 20 VGG tensors land in the right slots
python check_numblock_capacity.py       # the 2.96x capacity table, CPU only
python screen_table_iv.py               # the zero-GPU screen, and why range is the wrong statistic
python analyse_stage1.py                # sigma, the Table IV verdict, the nondeterminism floor
```

Kaggle cells: [kaggle_pilot_cells.md](kaggle_pilot_cells.md) ·
[kaggle_stage1_cells.md](kaggle_stage1_cells.md)

---

## ⭐ Run 10 — the SCD cell and the four inconclusive cells · ✅ **2026-09-14**

**Registered first.** `RUN10_PREREGISTRATION.md` and `scripts/analyse_run10.py` were committed alone
(`0c42239`) before any Run 10 kernel existed. Seeds **50–61** added to nb = 2, 4 and 6 (36 runs, 40.5 GPU-h
summed from the kernel logs), same configuration, T4, image and templates as Runs 1–9. nb = 8 was not
extended: no cell against it was inconclusive. **All five kill criteria pass for all three settings.** The
analysis output is saved unedited in [`RUN10_ANALYSIS_full.txt`](RUN10_ANALYSIS_full.txt).

### A — confirmatory: SCD, nb = 2 against nb = 4, new seeds only → **A2 NOT REPLICATED**

| | nb = 2 | nb = 4 |
|---|---|---|
| SCD, seeds 50–61 | 1.2933 ±0.0277 | 1.2704 ±0.0531 |

Difference +0.0229 = **0.54 pooled σ** (Run 9: 1.23 σ), Welch *t* = +1.325, df = 16.6, **p = 0.203**. The
direction repeats; the size does not. As registered: Run 9's cell was consistent with chance. Power was 82 % at
the selected 1.23 σ and 47 % at 0.8 σ, so this is not evidence the difference is zero.

### B — secondary: the four cells Run 9 left inconclusive, TOST ±1.75 σ at n = 20

| cell | d (pooled σ) | 90 % CI | Welch p | TOST p | reading |
|---|---|---|---|---|---|
| 4v2 SF | −0.77 | [−1.31, −0.24] | 0.019 | 0.002 | **equivalent and different** — a real difference smaller than the design resolution |
| 4v2 AG | −0.64 | [−1.17, −0.10] | 0.051 | 0.001 | **equivalent** |
| 4v2 SCD | −0.82 | [−1.35, −0.28] | 0.015 | 0.003 | **equivalent and different** |
| 4v6 SF | +0.62 | [+0.09, +1.15] | 0.058 | 0.001 | **equivalent** |

The 4v2 SCD row pools the eight seeds that produced the hypothesis; it is not independent evidence about that
cell. Test A is.

### The n = 20 table (descriptive, Holm across 18 cells, as registered)

**All 18 comparisons of nb = 2 and 6 against 4 are equivalent** within ±1.75 σ. **0 of 18 survive Holm**; 8 of
18 are nominally significant uncorrected. That count is not read against 18 × 0.05: the cells share the nb = 4
arm and the metrics are correlated. Descriptively, where a direction shows it runs 2 > 4 > 6 on SF, AG, VIF,
SCD and Qabf — fewer blocks nominally higher, the reverse of the published choice — every such gap inside the
design resolution. **The standing statement: settings 2, 4 and 6 are equivalent within ±1.75 σ, and nothing
supports 4 blocks as best.** Run 9's 1 of 27 at n = 8 is not replaced.

---

## ⭐ Run 12 — M1 (weight tying) at five seeds · ✅ **2026-09-24**

`mostafanada82/lrrnet-run12-m1tied` · seeds **45, 46** on the tied model · 2 runs, **2.2 GPU-h**
measured (65.8 and 65.2 min) · readings fixed in [`RUN12_PREREGISTRATION.md`](RUN12_PREREGISTRATION.md),
committed alone in `9c96a53` before the kernel existed · analysed by `scripts/analyse_run12.py`.

**Why it was run.** The design claim stood at **five untied seeds against three tied ones**, judged with a
fixed critical value of 2.776 — the right value for three against three and too strict for five against
three. A stricter ruler makes "nothing moves" easier to obtain, and "nothing moves" is what had been
reported. This run makes the comparison 5 against 5 at each comparison's own degrees of freedom.

### Kill criteria — all passed before any statistic

| check | result |
|---|---|
| `[CONFIG]` line of each run | `seed=45 … num_block=4 tied=1`, `seed=46 … tied=1` |
| parameter counts | untied **49,204** → tied **20,014** (2.46×), constant in depth |
| epochs completed | 4 on both runs (601 logged losses each) |
| images scored | 25 TNO pairs per run |
| duplicate runs | none — the two seeds differ on every metric |
| training exit code | 0 on both |

### The result — outcome **B2**, as pre-registered

Welch's *t*, two-sided 5 %, at each comparison's own df; Holm across the nine; TOST at ±1.75 pooled σ.

| metric | untied (n = 5) | M1 tied (n = 5) | diff (σ) | *t* (df) | *p* | verdict |
|---|---|---|---|---|---|---|
| EN | 6.9727 ± 0.0254 | 6.9850 ± 0.0267 | +0.47 | 0.75 (8.0) | 0.474 | equivalent |
| MI | 2.5021 ± 0.1357 | 2.4720 ± 0.0349 | −0.30 | −0.48 (4.5) | 0.653 | equivalent |
| SD | 43.5032 ± 0.8549 | 42.8601 ± 0.9984 | −0.69 | −1.09 (7.8) | 0.307 | inconclusive |
| SF | 10.7123 ± 0.3715 | 10.9125 ± 0.2943 | +0.60 | 0.94 (7.6) | 0.374 | inconclusive |
| AG | 3.9128 ± 0.1494 | 4.0605 ± 0.1031 | +1.15 | 1.82 (7.1) | 0.111 | inconclusive |
| VIF | 0.7646 ± 0.0401 | 0.7864 ± 0.0060 | +0.76 | 1.20 (4.2) | 0.293 | inconclusive |
| SCD | 1.2607 ± 0.0469 | 1.3172 ± 0.0411 | +1.28 | 2.03 (7.9) | 0.078 | inconclusive |
| Qabf | 0.4325 ± 0.0169 | 0.4464 ± 0.0073 | +1.07 | 1.70 (5.4) | 0.145 | inconclusive |
| **SSIM** | 0.9020 ± 0.0152 | **0.9236 ± 0.0080** | **+1.78** | **2.81 (6.1)** | **0.030** | **crosses, favours tied** |

**1 of 9 crosses uncorrected — SSIM, favouring the tied model — and 0 of 9 survive Holm.** Chance produces
0.45 such cells in nine tests at 5 %. By the reading written down first, this is **not** a quality
improvement: *M1 is indistinguishable from the untied network at 2.46× fewer parameters, not better than
it.* The same rule that refuses LRRNet's 1 of 27 refuses this cell, and it is applied here against our own
interest. (It is the same cell the corrected critical value surfaced at n = 3 on 2026-09-13, *p* = 0.038;
at five seeds it sits at *p* = 0.030 — stable, and still one cell.)

### What five seeds also show, and it is a limitation

**Only EN and MI are demonstrably equivalent within ±1.75 pooled σ.** The other six non-crossing metrics are
**inconclusive at this sample size** — neither different nor equivalent — and they are named rather than
folded into "no difference": SD, SF, AG, VIF, SCD, Qabf. At 5 against 5 the design resolves about 2.0 pooled
σ at 80 % power, so B1-style wording ("indistinguishable on all nine") overstates what the equivalence test
supports. The honest statement is the one above: **no effect of about 2 σ or larger was found, two metrics
are equivalent within the margin, six are unresolved, and the efficiency claim — a parameter count — is
categorical and needs no seeds at all.**

The registered n = 3 result stays in the record as written; this stands beside it.

---

## ⭐ Run 14 — free depth: the tied network at depth 8 against depth 4, same 20,014 parameters · ✅ **2026-09-25**

`mostafanada82/lrrnet-run14-m1tied-nb8` · seeds **45, 46** at depth 8 (64.2 and 63.6 min) · readings fixed in
[`RUN14_PREREGISTRATION.md`](RUN14_PREREGISTRATION.md), committed alone in `4b0f94a` · `scripts/analyse_run14.py`.

**Kill criteria — all passed.** `[CONFIG]` reads `epochs=4 num_block=8 tied=1` on both runs; 20,014 parameters;
exit 0; 601 logged losses; 25 TNO pairs; the two seeds differ on every metric; no out-of-memory stop.

### The result — outcome **E1**, as pre-registered

Depth 4 (n = 5: M1 seeds 42–44 + Run 12 seeds 45–46) against depth 8 (n = 5: `lrrnet-m1-nb8` seeds 42–44 +
Run 14 seeds 45–46). Welch at each comparison's own df; Holm across nine; TOST at ±1.75 pooled σ.

| metric | depth 4 | depth 8 | diff (σ) | *t* (df) | *p* | verdict |
|---|---|---|---|---|---|---|
| EN | 6.9850 ± 0.0267 | 6.9850 ± 0.0340 | −0.00 | −0.00 (7.6) | 0.997 | equivalent |
| MI | 2.4720 ± 0.0349 | 2.4980 ± 0.0356 | +0.74 | 1.16 (8.0) | 0.278 | inconclusive |
| SD | 42.8601 ± 0.9984 | 43.1430 ± 1.4229 | +0.23 | 0.36 (7.2) | 0.726 | equivalent |
| SF | 10.9125 ± 0.2943 | 10.9005 ± 0.2359 | −0.04 | −0.07 (7.6) | 0.945 | equivalent |
| AG | 4.0605 ± 0.1031 | 4.0463 ± 0.0889 | −0.15 | −0.23 (7.8) | 0.821 | equivalent |
| VIF | 0.7864 ± 0.0060 | 0.7847 ± 0.0130 | −0.17 | −0.26 (5.6) | 0.801 | equivalent |
| SCD | 1.3172 ± 0.0411 | 1.3031 ± 0.0386 | −0.35 | −0.56 (8.0) | 0.592 | equivalent |
| Qabf | 0.4464 ± 0.0073 | 0.4432 ± 0.0135 | −0.30 | −0.48 (6.1) | 0.650 | equivalent |
| SSIM | 0.9236 ± 0.0080 | 0.9172 ± 0.0136 | −0.58 | −0.91 (6.5) | 0.395 | inconclusive |

**0 of 9 cross, uncorrected or after Holm. 7 of 9 are demonstrably equivalent within ±1.75 pooled σ; MI and SSIM
are inconclusive at five seeds.** Doubling the number of solver iterations, with the parameter count held fixed
by construction, changes nothing the design can resolve — and unlike M1's comparison against the untied network
(2 of 9 equivalent), this null is carried by the equivalence test on seven metrics, not by the absence of
significance.

**What it says about the design.** The one ablation the untied network cannot produce — depth without size — has
now been run at five seeds, and at the published budget the answer is that depth is not where the tied network's
quality comes from. It is the cleanest null in Part III, and the ablation LRRNet's own Table IV was presented as
being. Whether depth matters under a longer budget is not answered here; Run 13 is measuring the budget itself.

### Run 14, secondary — the 2×2 of depth against tying, with the tied side at five seeds

Registered as a secondary, descriptive reading in `RUN14_PREREGISTRATION.md`; computed by
`scripts/analyse_2x2_n5.py`. The pre-registration did not fix the untied depth-4 sample, so both are shown.
Untied depth 8 stands at eight seeds (stage 3 + Run 9); tied depth 4 and 8 at five.

| contrast | untied d4 at n = 5 | untied d4 at n = 8 |
|---|---|---|
| tied d8 − tied d4 (depth, size constant) | 0 of 9 | 0 of 9 |
| untied d8 − untied d4 (depth as the paper measures it) | 0 of 9 | 0 of 9 |
| tied d4 − untied d4 (tying at depth 4) | 1 (SSIM, *p* = 0.030) | 4 at exact df (VIF, SCD, Qabf, SSIM), 1 at the fixed 2.776 |
| **tied d8 − untied d8 (tying at depth 8)** | **4 of 9: SF, AG, VIF, SCD — *p* = 0.021, 0.007, 0.008, 0.017** | same |
| **total of 36** | **5** (fixed 2.776) / **5** (exact df) | **5** / **8** |

**Every crossing cell favours the tied model. None survives Holm within its contrast** (the smallest *p*,
0.007, is above 0.05/9 = 0.0056).

**What this changes.** The paper's sentence — "1 of 36 comparisons reaches significance under the registered
threshold and 2 of 36 at exact degrees of freedom, against 1.8 expected by chance — both consistent with chance"
— described the 2×2 at three seeds on the tied side. At five it reads **5 of 36** (8 with untied depth 4 at eight
seeds), concentrated in one contrast: **at depth 8 the tied network is nominally ahead of the untied one on four
of nine metrics.** That is not a finding under the paper's own rule (nothing survives correction) — and it
favours our own model, which is exactly when the rule must be applied without exception. But "consistent with
chance" no longer describes the count, and the sentence is rewritten to say what the table shows: **the depth
contrasts are empty at every sample size; the tying contrast at depth 8 leans towards the tied model on four
metrics, uncorrected, and would need its own registered test to be claimed.**

---

## ⭐ Run 13 — the training-length curve: both arms at twice the published budget · ✅ **2026-09-25**

Kernels `lrrnet-run13-{tied,untied}-{a,b}` · 10 runs at 8 epochs (tied 130.6–134.4 min, untied 126.5–130.9 min
per run) · readings fixed in [`RUN13_PREREGISTRATION.md`](RUN13_PREREGISTRATION.md), committed alone in
`c149a84` before any kernel existed · `scripts/analyse_run13.py`, written before the last kernel landed.

**Kill criteria — all passed.** `[CONFIG]` on every run as designed (epochs 8, `num_block` 4, tied 1 or 0);
exit 0; 1,202 logged losses per run; 25 pairs; distinct seeds. **Criterion 3 — the longer run must actually
train:** arm-mean last logged loss tied 57,226 → 56,929, untied 56,644 → 56,260; seed 44's full epoch-mean curve
(tied) 57,151 at epoch 4 → 56,181 at epoch 8 (−1.7 %), its first four epochs reproducing the 4-epoch run to within
3 units. (untied_a ended with status ERROR from an assertion the builder appended, after all three seeds had
trained and been scored; recorded in `169bed9`.)

### The three families

| family | uncorrected | after Holm | equivalent (±1.75 σ) |
|---|---|---|---|
| **C1** tied@8 vs untied@8 | 3 — AG, SCD, SSIM, all favouring **tied** | **0** | 2 of 9 |
| **C2** tied@8 vs tied@4 | 2 — VIF, Qabf, both **lower at 8 epochs** | **1 — VIF: −3.43 σ, *p* = 0.0020, lower at 8** | 5 of 9 |
| **C3** untied@8 vs untied@4 | 0 | **0** (VIF −1.03 σ, *p* = 0.14) | 5 of 9 |

### Outcome — **D4**, by the rule as registered, and what it does and does not say

The pre-registration checks D4 first: *any C1 or C2 metric crossing against the tied arm*. One does, in C2: **the
tied network's VIF falls when its training is doubled** (0.7864 → 0.7704), and the fall survives Holm. That is
recorded as the registered outcome.

**What it is not** is a cost of tying relative to the published network. At the doubled budget the tied network
is behind the untied one on **no** metric (C1, 0 after Holm) and nominally *ahead* on three (AG, SCD, SSIM,
uncorrected). The untied network's VIF falls too, by more in absolute terms (−0.040 against −0.016), inside its
own much larger seed spread (σ 0.040 against 0.006). The accurate statement: **doubling the budget improves no
metric on either network; it lowers VIF on both, measurably on the tied one.**

**What it says about the design half.** Nothing gains from twice the budget — the D3 half of the outcome — so the
budget is not where the tied network's headroom is, and the extra training costs it VIF. Together with Run 14
(depth at constant size: 0 of 9, seven equivalent) and Run 12 (tying: no metric after Holm), three independent
directions — size, depth, budget — leave quality unchanged or slightly lower. **Part III closes as an efficiency
and measurability result**: the same algorithm at 2.46× fewer parameters, with the confound unbuildable, and no
setting found that buys quality.

**And a correction the paper owes.** §7 says optimisation *saturates by the second epoch*, from one seed's last
logged loss. Run 13 shows the training loss still falling at eight epochs on both arms — slowly — while no metric
rises. What saturates is the **metrics**, not the loss. The sentence is rewritten.

---

## ⭐ Run 16 — the LRRNet verdicts on a second test set (RoadScene, 40 pairs), no training · ✅ **2026-09-25**

Kernels `lrrnet-run16-fuse` (GPU, fusion only) and `lrrnet-run16-score` (CPU, evaluator) · 78 existing checkpoints
from dataset `mostafanada82/lrrnet-run16-ckpts` · readings fixed in [`RUN16_PREREGISTRATION.md`](RUN16_PREREGISTRATION.md),
committed alone in `ab54d03`, corrected before any data in `e5cbbd6` · `scripts/analyse_run16.py`, committed
(`47280fe`) before any Run 16 output existed · output in [`results_run16/`](results_run16/).

**One process failure, no data read.** The first fuse push died on `KeyError: 'LRR_DATA'`: the patched `args.py`
reads the training-corpus path at import. The kernel's own kill assertion stopped it before anything was scored.
Fusion never reads the corpus, so an empty folder is passed (`8c732e1`) and the kernel re-pushed.

### Kill criteria — all passed

| check | result |
|---|---|
| 1 — strict load at the expected parameter count | asserted in the fuse kernel for all 78 (29,744 / 49,204 / 68,664 / 88,124 / 20,014); kernel COMPLETE |
| 2 — images per checkpoint | 40 RoadScene + 25 TNO fused and scored; 156 evaluator calls, all exit 0 |
| 3 — TNO re-score against the stored 4-epoch rows (0.1 % relative) | **every one of the 702 TNO means is identical to the stored value to six decimals** — worst relative difference 0. The pipeline that read RoadScene is, bit for bit, the pipeline that produced the TNO record |
| 3 — no identical rows | none, on either set |

### The result — **T2 (DISAGREES)** for Table IV; **M agrees**; **D agrees**

| family | TNO (re-scored here) | RoadScene |
|---|---|---|
| T — nb 2 vs 4, n = 20 | 4 of 9 uncorrected, **0 after Holm**, 9 of 9 equivalent | 6 of 9 uncorrected, **2 after Holm — EN (*p* = 0.0046) and SCD (*p* = 0.0047), both higher for nb 2**; 9 of 9 equivalent |
| T — nb 6 vs 4, n = 20 | 4 of 9 uncorrected, **0 after Holm**, 9 of 9 equivalent | 5 of 9 uncorrected, **2 after Holm — VIF (*p* = 0.0029) and SCD (*p* = 0.0059), both lower for nb 6**; 9 of 9 equivalent |
| T — nb 8 vs 4, n = 8 | 0 of 9, 9 equivalent | 0 of 9, 8 equivalent (SF inconclusive) |
| M — tied vs untied, nb 4, n = 5 | 1 uncorrected (SSIM), 0 after Holm, 2 equivalent | 1 uncorrected (SSIM, *p* = 0.033, favours tied), **0 after Holm**, 2 equivalent |
| D — tied nb 8 vs nb 4, n = 5 | 0 of 9, 7 equivalent | **0 of 9, 9 of 9 equivalent** |

**Registered outcome T2**: 4 of 27 Table IV cells survive Holm on RoadScene. As the pre-registration requires, every
surviving cell's direction is reported, including whether it favours the published choice of 4 blocks:

- **2 blocks beat 4** on EN and SCD — against the published choice;
- **4 blocks beat 6** on VIF and SCD — for it;
- **8 against 4**: nothing.

**What this changes, and what it does not.**

* **The paper's Table IV statements are scoped to TNO.** "No comparison survives correction" is true of TNO at
  n = 8 and n = 20; on RoadScene, at twenty seeds, four cells do. The sentence becomes: *on TNO nothing survives
  correction; on a second test set, at twenty seeds, four of 27 cells do, and they order the settings 2 > 4 > 6 —
  fewer blocks better — which the published choice of 4 does not win.*
* **The detected differences are small — about 1 pooled σ — and all 18 twenty-seed comparisons remain equivalent
  within ±1.75 σ on RoadScene too.** Twenty seeds per arm resolve effects smaller than the design's equivalence
  margin, so the same cell is both *different* and *equivalent within the margin*. Both are stated: what is
  measured is a real, small, consistent preference for fewer blocks on RoadScene, not a practically large one.
* **The reading that damages our own earlier wording**: "2, 4 and 6 blocks are equivalent" (Run 10) still holds
  *as an equivalence statement* at ±1.75 σ, but "nothing separates them" is TNO-specific. "Nothing supports 4 as
  best" is **strengthened**: on RoadScene 2 blocks is measurably ahead of 4, at 60 % of the parameters.
* **The design half is unchanged and firmer.** M1 against the untied network: no metric after correction on
  either set. Depth at constant size: nothing on either set, and on RoadScene all nine metrics equivalent.
* **The TNO record is confirmed, bit for bit.** Every stored TNO row was re-produced exactly from its checkpoint
  in a fresh session, which answers the question of whether the analysis rows are the models' rows.

---

## 🔍 Diagnostic D4 — where the tied depth-8 network loses SD on TNO · exploratory · CPU · 2026-09-28

`diagnostics/d4_sd_loss.py` · output [`diagnostics/d4_sd_loss_output.txt`](diagnostics/d4_sd_loss_output.txt). It was
run **after** `RUN23_PREREGISTRATION.md` (`60f6e37`) was committed, so it cannot have shaped Run 23's arms. It does not
use Run 23's selection seeds 42–44.

**Inputs.**
- tied depth 8: Run 20 seeds 74–77;
- published configuration: seeds 45–49;
- the fused TNO images, downloaded from the kernels.

**Method.** Each image's variance is split exactly into three parts, with the infrared-hot pixels defined as IR above
the image's 90th percentile:
1. within the hot pixels;
2. within the rest (the background);
3. between the two groups.

| | SD | mean | p1 | p99 | hot − background gap |
|---|---|---|---|---|---|
| tied d8 | 41.89 | 111.27 | 44.81 | 208.50 | +3.68 |
| published | 43.87 | 109.41 | 40.38 | 208.59 | −0.66 |
| difference | **−1.99** | +1.87 | **+4.43** | −0.09 | **+4.35** |

**Reading (descriptive).**
- **93 % of the variance lost is within the background.** The hot-pixel part is 6 %, and the between-group part is 1 %.
- The infrared targets are **not** dimmed: they stand further from the background than in the published output
  (gap +4.35 grey levels).
- The loss comes from the **dark end being lifted** (1st percentile +4.4) while the bright end is unchanged. Shadows
  and dark background texture are flattened.
- The tied network has the lower SD on 21 of the 25 images.

So the SD cost is a property of the background — largely visible-band content — not of the infrared targets. Run 23's
registered arms A1–A3 strengthen the infrared side of the loss. D4 suggests that is not where the cost is. This is
recorded **before** any Run 23 row exists. Run 23 stays as registered.

---

## 🔍 Diagnostics D1–D2 — the released objective, read on CPU · exploratory · 2026-09-26

No GPU, no training, no Run 17 data. Scripts and outputs in [`diagnostics/`](diagnostics/); the reading and the plan
they produced are in [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md). Measured on the 25 TNO + 40 RoadScene pairs resized
to 128×128 as training resizes — **not** on the training corpus, which is why they are exploratory.

- **D1 — Adam's coupled `weight_decay = 0.9` is not what limits training.** The loss gradient exceeds the decay term
  in 86 % of weights of a trained checkpoint (75 % at init); the released loss is ~10⁵, and its gradients swamp 0.9·w.
  A hypothesis written into the first draft of the improvement plan the same morning is withdrawn by this.
- **D2 — every trained network scores *worse on its own loss* than copying the visible image.** Copy-VI 81,283;
  mean and max of the sources ≈ 90,300; six trained checkpoints (untied nb 2/4/8, tied nb 4/8) 98,333–103,070, within
  5 % of each other; untrained 130,095. 65 % of the loss is the relu3_3 term, whose target 3·Φ(IR) + 0.5·Φ(VI) no
  min-max-normalised image can reach.

**Reading, to be confirmed on the training corpus (step S0 of the plan):** the objective neither separates the
configurations Table IV and Part III compare nor is optimised, in four epochs at lr 1e-5, as far as a trivial
solution. That would account for every null in the design half at once.

---

## Run 18, arm S0 — D2 on the training corpus · ✅ 2026-09-26 · **D2 does NOT hold: a test-set artefact**

`mostafanada82/lrrnet-run18-s0` (CPU, no GPU quota) · reading fixed in [`RUN18_PREREGISTRATION.md`](RUN18_PREREGISTRATION.md)
(`ccee4ce`) before the kernel existed · output in [`results_run18_s0/`](results_run18_s0/) (400 LLVIP training pairs drawn
with `random.Random(0)` from the 12,025-file training list, listed in `run18_s0_sample.txt`).

| output F | total | pixel | relu1_2 | relu2_2 | relu3_3 | Gram |
|---|---|---|---|---|---|---|
| copy-VI | 61,450 | 0 | 0 | 1,716 | 47,647 | 12,088 |
| mean(IR,VI) | 65,893 | 11,936 | 3,932 | 1,477 | 41,752 | 6,796 |
| max(IR,VI) | 80,132 | 31,667 | 6,268 | 1,161 | 33,827 | 7,209 |
| copy-IR | 91,714 | 47,743 | 13,684 | 1,137 | 29,150 | 0 |
| **trained nb 4, seeds 42 / 43 / 44** | **56,178 / 56,793 / 56,549** | ~2,700 | ~1,000 | ~1,630 | ~45,200 | ~5,950 |

**Registered reading: D2 does not hold on the training corpus.** On LLVIP training pairs the trained networks are
**8 % below copy-VI** and below every trivial rule. D2 — "worse on its own loss than copying the visible image" — was a
property of the **test** images (TNO, RoadScene), not of the optimisation. As registered, **S2's rationale is
withdrawn**; S2 is still run and reported.

What survives, stated at its size: (i) the relu3_3 term is still ~80 % of the trained networks' loss and sits only
5 % below copy-VI's (45,200 against 47,647) — the unreachable-target observation is structural, not test-set-specific;
(ii) the objective the networks minimise on LLVIP does **not transfer** to the test sets, where the same networks fall
behind copy-VI — a train/test gap, which is what S4 (training at the test scale) addresses. Diagnostic D3 on CrossFuse
was likewise measured on test images only and carries the same caveat.

---

## Run 18 — the improvement screen · ✅ **2026-09-26** · reading: **NO CANDIDATE** (arm records and result below)

| arm | kernel | account | seeds | kill criteria |
|---|---|---|---|---|
| s4 (crop) | `collaborator-account/lrrnet-run18-s4` | collaborator (own account, own quota) | 42–44 | ✅ `crop=1`, 49,204 par., 4 epochs, exit 0, 25 pairs, seed fingerprint 267.13894641 = baseline |
| s1a (lr 1e-4) | `collaborator-account/lrrnet-run18-s1a` | collaborator | 42–44 | ✅ `lr=1e-4`, 49,204 par., 4 epochs, exit 0, 25 pairs, fingerprint = baseline |

**Two process defects, recorded; neither touches a number.**
1. Both collaborator kernels ended with status ERROR in their **last** cell, after all three seeds were trained, fused,
   scored and written: Kaggle's *Import Notebook* turned backslash-n escapes in the template's σ-summary cell into line
   breaks. The collaborator copies of s3u and s2 replace that cell with the completeness check alone
   (`make_collab_notebook.py`). Kernels pushed through the API are unaffected.
2. The registered `[LOSSCMP]` line (each arm's last-epoch loss beside copy-VI's) was computed but **not printed** by
   s1a, s4 and s1b: the templates print only `[CONFIG]`/`[EPOCH]` lines of the trainer's output. Fixed for s3t, s3u
   and s2 (`build_run18_kernels.py`). The selection rule does not use it; the three arms without it are named here.
3. **2026-09-26 ~14:15 UTC: the collaborator's notebooks for s4, s1a and s3u were deleted** during a clean-up of
   drafts. s4 and s1a had finished and their `lrrnet_metrics.csv` and full logs were already downloaded and committed
   here; those files are now the only record and are kept as the provenance. **s3u was running and was lost**; it is
   re-run. **Decision fixed before any s3u result exists (15:20 UTC):** the collaborator's re-run stayed queued for
   ~50 min while a slot on our account was free, so s3u was pushed on our account (`mostafanada82/lrrnet-run18-s3u`,
   same builder, seeds 42-44) and **that run is the record**. The collaborator's queued copy is to be cancelled; if it
   runs anyway, it is not used. No other arm is affected.

---

## ⭐⭐ Run 17 — the depth-8 tying lead, on new seeds · ✅ **2026-09-26** · outcome **L1: REPLICATES**

Kernels `lrrnet-run17-{tied,untied}-{a,b,c}` · new seeds **62–73**, 12 per arm · readings fixed in
[`RUN17_PREREGISTRATION.md`](RUN17_PREREGISTRATION.md), committed alone in `26c05eb` before any Run 17 kernel existed ·
`scripts/analyse_run17.py`, committed before any kernel finished · full output in
[`kernel_csv/run17_analysis_output.txt`](kernel_csv/run17_analysis_output.txt).

**Kill criteria — all passed, all 24 runs.** `[CONFIG]` `epochs=4 num_block=8` with `tied=1` / `tied=0`; 20,014 /
88,124 parameters at load; the untied arm on two GPUs as every earlier untied depth-8 row; exit 0, 4 epochs, 25 TNO
pairs per run; no identical rows. (Process notes: tied_c was started by a leftover queue at 04:32; s3t's version 3 was
created from the Kaggle UI — neither touched Run 17.)

### Confirmatory family — SF, AG, VIF, SCD, new seeds only, Holm across four

| metric | untied d8 (n = 12) | tied d8 (n = 12) | diff (σ) | *t* (df) | *p* |
|---|---|---|---|---|---|
| SF | 10.3603 ± 0.2083 | 10.8585 ± 0.2776 | **+2.03** | 4.97 (20.4) | **0.0001** |
| AG | 3.8203 ± 0.0836 | 4.0050 ± 0.1416 | **+1.59** | 3.89 (17.8) | **0.0011** |
| VIF | 0.7226 ± 0.0355 | 0.7821 ± 0.0285 | **+1.85** | 4.52 (21.0) | **0.0002** |
| SCD | 1.2131 ± 0.0531 | 1.3092 ± 0.0299 | **+2.23** | 5.46 (17.3) | **< 0.0001** |

**All four survive Holm, all favouring the tied network. Registered outcome L1 — REPLICATES:** at depth 8 the tied
network, at **4.40× fewer parameters** (20,014 against 88,124), is better on SF, AG, VIF and SCD — scoped, as registered,
to depth 8, TNO and the published 4-epoch budget, and stated beside Run 12's null at depth 4 (tying is not better in
general; it is better at depth 8). The effects sit at the size that raised the question (1.6–2.2 σ), not below it.

### Secondary, descriptive only (no outcome depends on these)

All nine on the new seeds, Holm across nine: **six favour tied** (SF, AG, VIF, SCD, **Qabf +1.48 σ, SSIM +2.63 σ**),
**one favours untied — SD, −1.57 σ (*p* = 0.0011)**, and EN and MI are equivalent within ±1.75 σ. The pooled arms
(tied 17 seeds, untied 20, including the seeds that raised the question) read the same way. **The SD deficit is a real
cost and is reported with the gain**: the tied depth-8 output has less global contrast.

**Caveats, stated now.** (i) One test set: the Run 17 checkpoints have not been scored on RoadScene. (ii) MI and VIF
reward copying a source (§7); VIF is one of the four here, but SF, AG, SCD, Qabf and SSIM do not share that failure on
LRRNet, and the positive control on LRRNet does not show MI/VIF inversion. (iii) The comparison is tied against untied
at the same depth; whether the tied depth-8 network is better than the *published* depth-4 configuration is a separate
question, not asked by this run.

---

## ⭐⭐⭐ Run 17b — Run 17's checkpoints on RoadScene · ✅ **2026-09-26** · **Q1-HOLDS; Q2-BETTER**

Kernels `lrrnet-run17b-fuse` (GPU) and `lrrnet-run17b-score` (CPU) · the 24 Run 17 checkpoints (dataset
`mostafanada82/lrrnet-run17-ckpts`) · readings in [`RUN17B_PREREGISTRATION.md`](RUN17B_PREREGISTRATION.md), committed
alone (`09f21ce`) before any Run 17 checkpoint was fused or scored on RoadScene · `scripts/analyse_run17b.py`, committed
before the kernels ran · output in [`results_run17b/`](results_run17b/).

**Kill criteria — passed.** All 24 checkpoints re-produced their stored TNO rows **exactly** (worst relative difference
0); no identical rows; 40 RoadScene pairs each.

**Q1 — Run 17 on RoadScene (tied d8 vs untied d8, seeds 62–73, Holm across four): HOLDS.** SF +1.94 σ (*p* = 0.0001),
AG +1.62 σ (0.0012), VIF +2.66 σ (< 0.0001), SCD +2.61 σ (< 0.0001) — all four favour tied, larger than on TNO.

**Q2 — tied d8 against the published configuration (untied d4, seeds 50–61), seven control-passing metrics, Holm
across seven: BETTER.**

| metric | published d4 (n = 12) | tied d8 (n = 12) | diff (σ) | *p* |
|---|---|---|---|---|
| EN | 7.2039 | 7.2502 | **+1.94** | 0.0001 |
| MI | 2.5457 | 2.6103 | **+1.12** | 0.0119 |
| SD | 48.5609 | 48.9021 | +0.38 | 0.36 (equivalent) |
| VIF | 0.6496 | 0.6900 | **+1.71** | 0.0005 |
| SCD | 1.4954 | 1.5590 | **+1.67** | 0.0009 |
| Qabf | 0.3583 | 0.3714 | **+2.05** | 0.0001 |
| SSIM | 0.7794 | 0.8226 | **+3.20** | < 0.0001 |

Six of seven survive Holm in the tied network's favour; none against. **On RoadScene, the weight-tied network at depth 8
— 20,014 parameters, 2.46× fewer than the published configuration — is better than the published configuration on six
of the seven control-passing metrics and worse on none.**

**Scope, stated with the result.** Q2 was registered on RoadScene because the TNO comparison had already been seen
(`kernel_csv/run17_exploratory_vs_published.txt`): on TNO it is a **trade-off**, not an improvement — VIF (+1.30 σ) and
SSIM (+1.72 σ) better after Holm, **SD worse (−1.46 σ)**. The accurate sentence is therefore: *better than the published
model on RoadScene; on TNO better on VIF and SSIM at a cost in SD.* MI and VIF reward copying a source (§7); the gain is
also carried by SCD, Qabf and SSIM, which do not share that failure. A claim of general improvement would need new seeds
on both sets; Run 19's design (candidate at seeds 50–61 against the existing published rows on TNO and RoadScene) is the
template for that test.

**Person check (the §7 standard), TNO pair 02, Figure 6's box, three seeds each.** Contrast of the person the infrared
band sees against the rest of the box: published d4 (seeds 42–44) **+25.4**, tied d8 (62–64) **+26.8 (1.05× published)**,
untied d8 (62–64) +22.9 (0.90×). By eye ([`results_run17b/person_check_tno02.png`](results_run17b/person_check_tno02.png):
IR | VI / published | tied d8 / untied d8) the person is visible in all three and at close to the published brightness
in the tied network. The tied depth-8 gain is not bought by discarding the infrared target.


### Run 18 — the result, by the rule registered in `RUN18_PREREGISTRATION.md` (`ccee4ce`)

`scripts/analyse_run18.py` (committed `66e3ce1` before the six arms were complete; its table was not read until they
were) · output in [`kernel_csv/run18_analysis_output.txt`](kernel_csv/run18_analysis_output.txt). Each arm minus the
published configuration at the same seeds 42–44, pooled σ:

| arm | change | EN | MI | SD | VIF | SCD | Qabf | SSIM | ahead / 7 | worst | rules 1+2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s1a | lr 1e-4 | −2.62 | −1.53 | −0.96 | −2.43 | −2.05 | −2.19 | −2.31 | 0 | −2.62 | fail |
| s1b | lr 3e-4 | −1.84 | −2.55 | −0.83 | −4.04 | −1.31 | −3.27 | −4.58 | 0 | −4.58 | fail |
| s2 | reachable middle target | +1.43 | +3.65 | +1.85 | +0.99 | **−13.78** | +0.40 | −1.10 | 5 | −13.78 | fail |
| s3u | lr 1e-4 + target | +1.49 | +3.04 | +2.17 | −0.33 | **−8.36** | −0.50 | −2.38 | 3 | −8.36 | fail |
| s3t | lr 1e-4 + target, tied | +1.66 | +5.46 | +3.61 | +1.49 | **−8.03** | +1.09 | −1.12 | 5 | −8.03 | fail |
| s4 | crops at test scale | −2.71 | −0.11 | −0.56 | −1.34 | −1.28 | −1.16 | −1.97 | 0 | −2.71 | fail |

**Registered reading: no arm passes rules 1 and 2 → "no candidate". Run 19 is not run, and the D2/D3 account —
quality limited by the optimiser or the objective — is reported as not supported by intervention.** Rule 3 (the person
check) is not needed.

**What the screen shows, stated descriptively.** A higher learning rate makes every selection metric worse. Training
on crops at test scale does not help either. The reachable-target arms (s2, s3u, s3t) move the output toward the visible
image: **MI rises by up to +5.5 σ while SCD collapses by −8 to −14 σ** — the pattern §7 documents, in which MI rewards
a fused image for resembling one source and SCD, which measures what the fused image takes from *both*, falls. The
selection rule, which forbids any metric falling by more than 1 σ, rejected every one of them; read on MI alone they
would have been reported as large improvements. This is the paper's protocol working on the thesis's own proposals.

---

## ⭐⭐ Run 21 — the depth-8 result on a third test set (MSRS, 361 pairs), no training · ✅ **2026-09-28** · **M1-HOLDS; M2-BETTER**

Kernels `lrrnet-run21-fuse` (GPU) and `lrrnet-run21-score` (CPU) · 44 checkpoints (Run 17's 24 + the published
configuration at seeds 42–61) · readings in [`RUN21_PREREGISTRATION.md`](RUN21_PREREGISTRATION.md) (`2dcb0d8`), committed
before any checkpoint was fused on MSRS · `scripts/analyse_run21.py`, committed before the kernels finished · output in
[`results_run21/`](results_run21/). MSRS's visible folder holds eight "(1)" duplicates that are not pairs; the 361 IR
names define the pairs, as in every script here.

**Kill criteria — passed.** All 44 checkpoints reproduced their stored TNO rows exactly (worst relative difference 0);
no identical rows; 361 MSRS pairs each.

**M1 — tied d8 vs untied d8 (seeds 62–73), Holm across four: HOLDS.** SF +2.62 σ, AG +2.74 σ, VIF +2.49 σ (all
*p* < 0.0001) favour tied; SCD +0.24 σ (*p* = 0.56) is equivalent on MSRS.

**M2 — tied d8 vs the published configuration (seeds 50–61), seven metrics, Holm across seven: BETTER.** EN +1.46 σ
(*p* = 0.0018), VIF +1.88 σ (0.0001), Qabf +1.60 σ (0.0008), SSIM +1.81 σ (0.0002) favour tied; MI +1.05 σ (*p* = 0.017)
crosses uncorrected only; **SD +0.77 σ and SCD −0.35 σ are equivalent**; none favours the published model.

**Reading with Runs 17 and 17b.** The tied depth-8 network beats the untied one at the same depth on **three test sets**
(TNO, RoadScene, MSRS), and is better than the published configuration — 2.46× larger — on **two of them** (RoadScene 6
of 7, MSRS 4 of 7, none against on either); on TNO, read before its registration, the published-model comparison is a
trade-off (SD). Run 20 (new seeds, a baseline outside this comparison, TNO and RoadScene together) is the general test.

---

## ⭐⭐ Run 20 — the general test: tied depth 8 vs the published configuration, new seeds, two test sets · ✅ **2026-09-28** · outcome **G2: TRADE-OFF**

Candidate: tied `num_block` = 8 at **new seeds 74–85** (kernels `lrrnet-run20-tied-{a,b,c}`, all kill criteria passed).
Baseline: the published configuration at **seeds 42–49**, never in this comparison (TNO: stage 1 + Run 9; RoadScene: Run 16).
Readings in [`RUN20_PREREGISTRATION.md`](RUN20_PREREGISTRATION.md) (`b0ca87e`) with two execution-only corrections before any
RoadScene row was read (`cac0b49`, `e344850`): the GPU quota ran out, so the RoadScene part was fused and scored on
**Kaggle CPU** kernels (`lrrnet-run20-cpu-{a,b,c}`, `fuse_lrrnet.py` with its three `.cuda()` calls removed); a partial
local attempt was discarded unread. `scripts/analyse_run20.py` (`01339f5`) · output in [`results_run20/`](results_run20/).

**Kill criteria — passed.** 12 of 12 candidate seeds; every candidate reproduced its stored TNO row within 0.1 % in the
CPU session (no failures); 40 RoadScene pairs each.

| test set | favouring tied d8 after Holm (across seven) | against it |
|---|---|---|
| **TNO** | VIF +1.77 σ, Qabf +1.41 σ, SSIM +1.86 σ | **SD −1.83 σ** (*p* = 0.0029) |
| **RoadScene** | EN +2.18 σ, VIF +1.62 σ, Qabf +1.98 σ, SSIM +2.60 σ | none |

**Registered outcome G2 — TRADE-OFF.** A metric survives against the candidate on TNO, so the result is reported metric by
metric and **not** as an improvement: *at new seeds, the tied depth-8 network — 2.46× smaller than the published
configuration — is better than it on RoadScene on four metrics with no cost, and on TNO better on VIF, Qabf and SSIM at a
cost in global contrast (SD).* This replicates, on independent seeds and baseline, exactly the pattern Runs 17b and 21 and
the exploratory TNO comparison showed. Combined with Run 21 (MSRS: better on four, none against), the defensible sentence
for the thesis is: **better than the published model on RoadScene and MSRS, a trade-off of SD for VIF, Qabf and SSIM on
TNO.**

**Process note (2026-09-28, found while building Run 22; no number affected).** `build_run20_kernel.py` rewrote each
template cell line by line, but most of the template's cells store their source as one string, so the three Run 20
training notebooks hold those cells as lists of single characters. Kaggle joins a cell's list before running it, so the
executed code is character-for-character the template's. The kernels ran, and every kill criterion passed. Both builders
that used the loop (`build_run18_kernels.py --run 22`, `build_run20_kernel.py`) now join the source first.

---

## Run 20 GPU consistency check: 2026-10-03 (own account, quota week of 03-10)

- Run 20's RoadScene rows were produced on three Kaggle **CPU** kernels when the GPU quota ran out.
- The GPU kernels Kaggle had refused then were pushed again: `lrrnet-run20-fuse` (GPU, 10 min, all 12 checkpoints
  loaded with the expected parameter count) and `lrrnet-run20-score` (CPU, 13 min).
- **Registered reading (WEEK_PLAN):** the GPU-fused rows must agree with the analysed CPU rows to within 0.1 %
  relative.
- **Result: they agree.** On RoadScene the worst relative difference is 1.6e-5 over 12 checkpoints × 9 metrics. On
  TNO it is 2.4e-5. No metric differs by more than 0.1 %.
- **Run 20's G2 reading stands unchanged.** Files: `results_run20_gpu_fuse/` (log; the fused PNGs are not tracked)
  and `results_run20_gpu_score/`.

## Run 22: the Run 18 arms re-run on our own account (2026-10-03): R1, CONSISTENT

Registered alone before any run (`RUN22_PREREGISTRATION.md`, daaec39). The kill criteria are checked from the kernel
logs. Kaggle's CLI on Windows needs `PYTHONUTF8=1` to save the logs, because of a progress-bar character. No
metric is read until all three arms are in.

| arm | kernel | seeds | kill criteria |
|---|---|---|---|
| s1a (lr 1e-4) | `mostafanada82/lrrnet-run22-s1a` | 42–44 | ✅ `lr=1e-4`, 49,204 par., 4 epochs, exit 0, 25 pairs, `[LOSSCMP]` printed, seed fingerprint 267.13894641 = baseline |
| s4 (crop) | `mostafanada82/lrrnet-run22-s4` | 42–44 | ✅ `crop=1`, 49,204 par., 4 epochs, exit 0, 25 pairs, `[LOSSCMP]` printed, fingerprint 267.13894641 = baseline |
| s2 (reachable target) | `mostafanada82/lrrnet-run22-s2` | 42–44 | ✅ `wir=0.857 wvi=0.143`, 49,204 par., 4 epochs, exit 0, 25 pairs, `[LOSSCMP]` printed, fingerprint 267.13894641 = baseline |

**Registered outcome: R1, CONSISTENT** (`scripts/analyse_run22.py`, output in `kernel_csv/run22_analysis_output.txt`).
Run 18's rule was applied again to the own-account rows, against the published configuration at seeds 42–44.

| arm | ahead on the 7 control-passing metrics | worst | verdict |
|---|---|---|---|
| s1a | 0/7 | −2.51 σ (VIF) | fail |
| s4 | 0/7 | −2.73 σ (EN) | fail |
| s2 | 5/7 | **−15.61 σ (SCD)**, the copy pattern | fail |

Run 18's "no candidate" stands on our own account, and the own-account rows are now the record of Run 18.

**Agreement with the first execution (descriptive).** Own-account mean minus collaborator mean, in published seed
σ:
- s4 and s2 agree within 0.65 σ on every metric.
- s1a differs on SD (−1.20) and SCD (+2.88). Its verdict is unchanged, but the arm with the 10× learning rate is the
  least reproducible across accounts.

## Runs 23 and 23b: the SD screen on tied depth 8 (2026-10-03): NO CANDIDATE, Run 24 not run

Registered alone before any run (`RUN23_PREREGISTRATION.md` 60f6e37, `RUN23B_PREREGISTRATION.md` 048956e). No metric
is read until all five arms are in.
- Kill criterion 3 (loss NaN or rising) is checked by `scripts/check_run23_losses.py`. It reads the loss files that
  LRRNet saves (`model/loss_v1/loss_all_*epoch_4_batch_1491*.mat`), because the kernel logs keep only summaries.

| arm | change | kill 1 (config, 20,014 par.) | kill 2 (4 epochs, 25 pairs) | kill 3 (loss) |
|---|---|---|---|---|
| A1 | `w_ir` 4.0 | ✅ `wir=4.0`, tied, nb 8 | ✅ | ✅ first-epoch mean → final-epoch mean: 114,880 → 96,012; 113,130 → 95,903; 107,929 → 95,890 |
| A2 | `lam3_gram` 2500 | ✅ `gram=2500`, tied, nb 8 | ✅ | ✅ first-epoch mean → final-epoch mean: 77,854 → 58,708; 75,282 → 58,479; 70,822 → 58,458 |
| A3 | `lam2_vi` 1.0 | ✅ `lam2=1.0`, tied, nb 8 | ✅ | ✅ first-epoch mean → final-epoch mean: 74,588 → 56,805; 71,537 → 56,645; 68,051 → 56,589 |
| A4 | tied depth 6 | ✅ nb 6, tied | ✅ | ✅ first-epoch mean → final-epoch mean: 76,730 → 57,115; 72,883 → 57,109; 69,902 → 56,977 |
| B1 | `lam2_vi` 2.0 | ✅ `lam2=2.0`, tied, nb 8 | ✅ | ✅ first-epoch mean → final-epoch mean: 77,374 → 57,390; 73,906 → 57,292; 70,034 → 57,246 |

**The registered reading** (`scripts/analyse_run23.py`, output in `kernel_csv/run23_analysis_output.txt`). Each arm
minus P at seeds 42–44, in pooled σ:

| arm | ahead/7 | worst | SD ≥ P | SD > A0 | rules 1–4 |
|---|---|---|---|---|---|
| A1 (`w_ir` 4.0) | 2/7 | SD −2.59 | no | no | fail (SCD +9.77, the copy pattern) |
| A2 (`gram` 2500) | 4/7 | SD −0.59 | no | no | fail |
| A3 (`lam2` 1.0) | 5/7 | SD −0.25 | no | no | fail |
| A4 (tied depth 6) | 6/7 | SD −0.47 | no | no | fail |
| **B1 (`lam2` 2.0)** | **6/7** | SCD −0.38 | **yes (+0.25)** | **yes** | **PASS** |

Only B1 passes rules 1–4, so it is the candidate, subject to rule 5.

**Rule 5, the person check** (`person_check_run18.py --run 23`; panel at `results_run23_person/person_panel.png`).
On TNO pair 02, the person's contrast at seed 42, relative to the published output:
- published 1.00;
- A0 (unchanged tied d8) 0.80;
- A1 1.27, A2 0.98, A3 0.94, A4 0.84;
- **B1 0.72**.

**B1 fails rule 5** (the threshold is ≥ 0.9). By eye the person is visibly fainter in B1's crop than in the
published one. No other arm passes rules 1–4.

**Registered outcome: NO CANDIDATE. Run 24 is not run.**

Reading:
- The stronger shallow pull toward the visible image (B1) restores global SD, as D4 predicted from the background.
- It does so by flattening the infrared target. "Improvement with no cost" was not found among one-step
  changes to the released loss.
- ~~The unchanged tied network already stands at 0.80 on this check, so the person's contrast is a cost that
  depth-8 tying carries on TNO, beside SD.~~ **Withdrawn 2026-10-03:** the registered follow-up below does not
  confirm it (C3). The paper states no person-contrast cost for tied depth 8.

## Person-contrast follow-up, seeds 43 and 44 (2026-10-03): **C3, NOT CONFIRMED**

Registered in `PERSON_FOLLOWUP_PREREGISTRATION.md` (bb20e95, committed alone before any measurement).
- No training and no GPU. Only the existing `02.png` outputs were downloaded.
- Command: `person_check_run18.py --run 23 --seed 43|44`. Seed 42 reproduces exactly.
- Panels: `results_run23_person/seed43/` and `seed44/`.

Person contrast on TNO pair 02, as a ratio to the published output at the same seed:

| arm | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| published (contrast, grey levels) | 26.7 | 23.3 | 26.3 |
| **A0 tied d8** | **0.80** | **1.17** | **1.08** |
| a1 | 1.27 | 1.67 | 1.56 |
| a2 | 0.98 | 1.44 | 1.23 |
| a3 | 0.94 | 1.37 | 1.23 |
| a4 | 0.84 | 1.21 | 1.10 |
| b1 | 0.72 | 1.09 | 0.96 |

**Reading as registered:**
- A0 is ≥ 0.9 at both new seeds, so the outcome is **C3, NOT CONFIRMED**. The seed-42 figure was not
  representative, and the paper does not state a person-contrast cost for tied depth 8.
- By eye, the target is clearly visible in every crop at seeds 43 and 44.

**Run 23's outcome is unchanged (NO CANDIDATE)**, as fixed beforehand.
- B1 would have passed rule 5 at seeds 43 and 44 (1.09 and 0.96).
- Rule 5 was registered on seed 42 alone and was applied as registered.
- Lesson for future pre-registrations: a single-seed, single-pair check is too noisy to gate a decision.
  Seed-to-seed spread for one arm reaches 0.4 in ratio. Future person checks use at least three seeds.
- One image pair only; nothing here is a claim about TNO as a whole.

## Correction (2026-10-06): the power label of RUN9_PREREGISTRATION.md's resolution table, withdrawn claim #19

- `RUN9_PREREGISTRATION.md` gives the smallest resolvable difference as 3.5 / 2.3 / 1.75 / 1.53 σ at n = 3 / 5 / 8 / 10,
  "at 80 % power".
- Recomputed on 2026-10-06, (t₀.₉₇₅ + t_power) · √(2/n) gives those values at **90 %** power. At 80 % they are 3.0 / 2.0 /
  1.5 / 1.3 σ, and the exact noncentral-t values agree to within 0.07.
- The **numbers, the registered ±1.75 σ margin and every verdict stand**. Only the label was wrong.
- The pre-registration is left as committed. The protocol text (paper §3, thesis Chapter 3) and the report now say
  90 %, and the error is withdrawn claim #19.
- RUN12/RUN14's "about 2.0 σ at 80 %" for 5 against 5 was correct as written.


## Run 25: the positive control at twenty seeds per arm (2026-10-06): **P3, DO NOT SEPARATE AT TWENTY**

Registered alone before any kernel (`RUN25_PREREGISTRATION.md`, eb09d45). Analysis committed before any score
existed (fbd301f). Output: `kernel_csv/run25_analysis_output.txt`.
- **Trained arm:** the published configuration at seeds 42–61 (existing rows).
- **Untrained arm:** 20 new random initialisations, seeds 400–419, zero gradient steps, fused on TNO.
- **Kill criteria 1–4 pass:**
  - 49,204 parameters, `num_block = 4`;
  - the initialisation is deterministic per seed and differs across seeds;
  - 25/25 pairs fused;
  - 40 rows, none identical.

| metric | trained (n = 20) | untrained (n = 20) | effect | p (Welch) | Holm (×2) | n per arm for 90 % power |
|---|---|---|---|---|---|---|
| **SF** | 10.70 ± 0.28 | 9.26 ± 2.73 | −0.75 σ | 0.029 | **does not survive** | **~40** |
| **AG** | 3.97 ± 0.12 | 3.50 ± 0.87 | −0.76 σ | 0.026 | **does not survive** | **~39** |

**Registered outcome: P3.** SF and AG did not separate an untrained LRRNet from a trained one even at twenty seeds per
arm, after Holm. Both cross the 5 % level uncorrected (p ≈ 0.03).
- At the observed means and spreads, 90 % power needs **about forty runs per arm**.
- The other seven metrics all separate (secondary):
  - EN, MI, SD, VIF, SCD, Qabf and SSIM, at p ≤ 6e-7;
  - their power requirement is **2–7 runs per arm**;
  - SD, which needed five seeds in Run 6, separates easily here.
- **Mechanism, as in Run 6:** the untrained arm's spread is 50–90× the trained arm's variance on SF and AG, which
  absorbs a mean difference that is not small in absolute terms.

**Consequence for the text:**
- The claim narrowed by #20 ("did not separate at five seeds") is now stated at twenty seeds, with the power
  requirement.
- It is stronger than before, and it is still a statement of non-detection at a stated n, not of equivalence.

## Run 26: deterministic mode (2026-10-06): **D1, A SEED REPRODUCES A RUN**

Registered alone before any kernel (`RUN26_PREREGISTRATION.md`, 890e11c). Analysis committed before the rows were
opened (2afaecb). Output: `kernel_csv/run26_analysis_output.txt`.
- Kill criteria pass: `[DET]` line in all four runs, seeds as registered, 4 epochs × 1,503 batches, exit 0, 25 pairs.
- PyTorch issued **no determinism warning**, so no operation in LRRNet's training lacked a deterministic
  implementation.

**Primary:** `nb4det_seed42a` and `nb4det_seed42b` are **bit-identical on all nine metrics** (relative difference 0).
Registered outcome **D1**.
- **Cost:** median epoch 1,134 s against 945 s for the nondeterministic Run 9 nb4 runs, so **1.20×**, or 20 % slower.
  The first deterministic run's first epoch took 1,844 s (warm-up).
- **Exploratory (n = 3):** the spread across deterministic seeds 42–44, relative to σ of record (n = 20), ranges from
  0.35 to 2.04, and every 95 % interval reaches 1 or more. Removing kernel nondeterminism did not visibly shrink the
  run-to-run variability, so on LRRNet the seed carries most of it.
- The deterministic seed-42 row differs from the nondeterministic seed-42 row of record by 0.02–0.55 σ, as expected
  for a different algorithm path.

**Consequence for the text:** "recording a seed is necessary but not sufficient" (wording as of 2026-10-11) now carries its fix: "unless
deterministic algorithms are enforced (on LRRNet: bit-identical reruns at a 20 % time cost)". Checklist step 1 gains
the same option.

## Run 27: the volume-controlled Top-K test at eight seeds per arm (2026-10-06): **S2, NONE DETECTED**

Registered alone before any kernel (`RUN27_PREREGISTRATION.md`, 9ce0c57). Analysis committed before any Run 27 row
was opened (e701800). Output: `kernel_csv/run27_analysis_output.txt`.
- **k0 kernel:** the log shows kill criterion 1 passed (corpora identical to Run 4b's: 59.595 / 6.552 / 1.0111 /
  overlap 192), and 5/5 runs. `results_run27_k0/lrrnet_metrics.csv` is the kernel's own output file (downloaded
  from the Kaggle web page on 2026-10-10, identical to the copy taken earlier from the output viewer), and the five
  checkpoints are in `results_run27_k0/ckpts/`.
- **topk kernel:** the kernel's full output archive and its log (both downloaded from the Kaggle web page, after the API rate-limited this account) are in hand. The log shows kill criterion 1 passed, with the same four values, and 5/5 runs of 67–72 min. The kernel ran 6.51 h; corpus building took about 42 min of that. `results_run27_topk/lrrnet_metrics.csv` is the archive's own file, and the five checkpoints are in `results_run27_topk/ckpts/`.
- **Pooled:** Run 4b's seeds 42–44 plus the new 45–49, so n = 8 per arm.

**Primary (stability):** σ²(k0)/σ²(topk) per deciding metric, F-test (7, 7), Holm across seven.

| metric | variance ratio | p |
|---|---|---|
| EN | 0.58 | 0.49 |
| MI | 0.24 | 0.081 |
| SD | 0.37 | 0.22 |
| VIF | 0.36 | 0.20 |
| SCD | 0.88 | 0.87 |
| Qabf | 0.79 | 0.76 |
| SSIM | 0.81 | 0.78 |

**Registered outcome: S2, none detected.** Every ratio is below 1 (Top-K nominally *more* variable) and none is
significant.
- **Means (secondary):** Top-K is lower on EN (−1.36 σ, p 0.018), MI (−1.32 σ, p 0.024) and VIF (−1.25 σ, p 0.028),
  uncorrected; none survives Holm.
- **Median spread ratio across nine:** 0.89, bootstrap 95 % interval 0.50–1.50.
- **Resolution:** at eight seeds per arm a variance ratio must exceed about 5 to reach significance (about 39 at three);
  90 % power needs a true ratio of about 14 (about 350 at three). *(Corrected 2026-10-13: first written as "about 5
  is detectable", which is the critical ratio, i.e. 50 % power.)*

**Consequence for the text:** "no effect of the mechanism itself was detected" now rests on eight seeds per arm.
Its scope is unchanged: LRRNet with MSRS, not CrossFuse with M3FD. That gap is what C2 (Saturday) addresses.

**Note on Run 4b's own cell:** its in-kernel verdict used 19.0 as the "two-sided" 5 % critical value of F(2,2).
19.0 is the one-sided value; the two-sided value is 39.0. No reading changes, because no ratio came near either.
