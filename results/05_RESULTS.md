# 05 — LRRFuse-VA results of record

Numbers of record for chapter 6. Design: [`02_DESIGN.md`](02_DESIGN.md) · patch:
[`04_PATCH_NOTES.md`](04_PATCH_NOTES.md) · plain language: [`03_EXPLAINED.md`](03_EXPLAINED.md)

**Last updated:** 2026-09-09

| Run | Configuration | Status |
|---|---|---|
| **1** | **M1 — weight-tied unrolling, `nb=4`, seeds 42–44** | ✅ **complete 2026-09-03** |
| **2** | **M1 at `nb=8` — a clean depth ablation** | ✅ **complete 2026-09-05** |
| **3** | **M1 + M2 — asymmetric refiners, capacity-matched to 49,204** | ❌ **complete 2026-09-06 — the pre-registered FAILURE branch** |
| **4** | **M3 — Top-K selective, distribution-aligned corpus augmentation** | ⚪ **complete 2026-09-07 — a clean null at the paper's own k=1,000** |
| 5 | M1 + M2 + M3 — the full method | ⛔ **not needed** — M2 and M3 both failed, so the full method *is* M1, already measured on 3 seeds |
| **6** | **Positive control — the sensitivity floor of the nine metrics** | ⭐ **complete 2026-09-07** |
| **7** | **The same control on CrossFuse — does the failure generalise?** | ⭐⭐⭐ **complete 2026-09-09 — it does NOT. 9 of 9 detect; SF/AG lead; MI/VIF invert** |

---

## Run 1 — M1, weight-tied unrolling · ✅ PASS

`mostafanada82/lrrnet-m1-tied` · 204.6 min · 3.09 GPU-h · `scripts/analyse_m1.py`

### Integrity, verified before any number was used

| check | result |
|---|---|
| kernel elapsed | **204.6 min** — a real run, not a Quick Save |
| preflight, on the training machine | **20,014 parameters · 1 distinct block** |
| seed audit | PASS — both generators respond to the seed |
| `[CONFIG]` line, all three runs | `num_block=4 **tied=1**` |
| training | exit 0, **65.5–65.7 min** each; loss 296,780 → 57,634 · 165,505 → 58,549 · 130,819 → 56,391 |
| checkpoint size | **93,307 B** — consistent with 20,014 parameters, not 49,204 |
| fusion | `loaded … (20,014 parameters, num_block=4)` on every seed |
| container | same pinned digest as stages 1–4, so the environment is not a second variable |

The preflight gate is new in this run and was written for one specific failure: a stale
`patch_lrrnet.py` in the attached dataset would have trained a perfectly healthy **untied** network
for 3.5 hours and returned a row labelled `m1tied`. It printed 20,014 and one block, so it did not.

### The measurement

Against stage 1's `nb=4` **untied** rows, n=5, our evaluator, the same 25 TNO pairs.

| metric | untied (n=5) | **M1 tied (n=3)** | diff | pooled σ | Welch *t* |
|---|---|---|---|---|---|
| EN | 6.9727 ±0.0254 | 6.9823 ±0.0363 | +0.0097 | 0.33σ | 0.41 |
| MI | 2.5021 ±0.1357 | 2.4626 ±0.0440 | −0.0395 | 0.35σ | −0.60 |
| SD | 43.5032 ±0.8549 | 42.8125 ±1.2559 | −0.6907 | 0.69σ | −0.84 |
| SF | 10.7123 ±0.3715 | 11.0253 ±0.3271 | +0.3130 | 0.88σ | 1.24 |
| AG | 3.9128 ±0.1494 | 4.1095 ±0.0810 | +0.1967 | 1.51σ | 2.41 |
| VIF | 0.7646 ±0.0401 | 0.7825 ±0.0013 | +0.0179 | 0.55σ | 1.00 |
| SCD | 1.2607 ±0.0469 | 1.3116 ±0.0388 | +0.0510 | 1.15σ | 1.66 |
| Qabf | 0.4325 ±0.0169 | 0.4441 ±0.0043 | +0.0117 | 0.83σ | 1.47 |
| SSIM | 0.9020 ±0.0152 | 0.9227 ±0.0061 | +0.0207 | 1.60σ | 2.70 |

> **0 of 9 reach the 5% level. 0 of 9 move beyond 3 pooled σ.**
> Two-sided critical value 2.776 at the Welch–Satterthwaite df.

### ⭐ The result, stated exactly

**Removing 29,190 of 49,204 parameters — 59% of the network — changes nothing measurable.**

This is the **pre-registered PASS branch**, applied as written. The efficiency claim stands at
**2.46×**, and it is *categorical*: a parameter count has no run-to-run variability to hide behind.

**M2's budget is open.** Its refiners may spend exactly 29,190 parameters, returning the model to
49,204 — identical to the published network — so the comparison is **capacity-matched by
construction**, not by a control run bolted on afterwards.

### The third independent demonstration that capacity is not the binding constraint

| change | parameters moved | metrics that moved |
|---|---|---|
| `nb` 4 → 6 (stage 2) | +19,460 | **0 of 9** |
| `nb` 4 → 8 (stage 3) | +38,920 | **0 of 9** |
| **M1 (this run)** | **−29,190** | **0 of 9** |

Adding capacity does nothing; removing 59% of it does nothing. Three independent runs, in both
directions. Together these say something stronger than any one of them: on this architecture and
budget, **parameter count is not what determines fusion quality.**

---

## ⚠️ Two things this run does NOT establish

Recorded here because the temptation to claim them is exactly the error this thesis documents.

### 1. M1 is **not** "better", despite leading on 7 of 9

M1 is nominally ahead on **7 of 9** metrics — the *same count* LRRNet's Table IV uses to justify
`num_block = 4`, and which chapters 4–5 spend their length refuting.

> With nine correlated metrics and one configuration each, a lead is nearly guaranteed to exist.
> **Zero of these leads reach significance.** The honest statement is *indistinguishable*, not
> *better*.

The standard is applied to our own method first. Any later draft that upgrades this to "M1 improves
quality" is reintroducing the defect the thesis exists to document.

### 2. The apparent σ reduction is **not** established

Relative spread fell from a mean CV of **3.29% to 1.66% — a 1.98× reduction** — which looks like
precisely the variance reduction **M3** is designed to deliver, arriving early and unbidden.

It does not survive a test:

| | |
|---|---|
| variance ratios beyond F(4,2) | **1 of 9** |
| two-sided 5% critical value | **39.25** |

At n=5 against n=3 a variance estimate is far too weak to carry a claim. **Logged as a hypothesis
for M3 to test at higher n** — not as a finding.

If it does replicate, it has a plausible mechanism worth stating in advance: a tied block receives
gradient from four iterations rather than one, which averages over more of the training signal per
step and could genuinely stabilise the solution. Writing the mechanism down *before* the test is
what stops it becoming a post-hoc story.

---

---

## Run 2 — M1 at `num_block = 8` · ✅ a clean depth ablation

`mostafanada82/lrrnet-m1-nb8` · 208.1 min · 3.47 GPU-h · `scripts/analyse_depth.py`

### Integrity

| check | result |
|---|---|
| kernel elapsed | **208.1 min** |
| preflight | `num_block=4: 20,014` · `num_block=8: 20,014` → **`DEPTH IS FREE`** |
| `[GPUS]` | `2 parallel= True` on all three runs |
| `[CONFIG]` | `num_block=8 **tied=1**`, all three |
| training | exit 0, **63.9–65.1 min**; loss 310,184 → 57,527 · 160,281 → 58,541 · 137,658 → 56,058 |
| fusion | `loaded … (20,014 parameters, num_block=8)` |

The preflight assertion is the one this run stands on: `counts[4] == counts[8] == 20_014`. Had it
failed, this would not have been a depth ablation and the 3.5 hours would have bought nothing.

### ⭐ The 2×2 the published table needed and never had

|  | depth 4 | depth 8 |
|---|---|---|
| **untied** | 49,204 (stage 1, n=5) | 88,124 (stage 3, n=3) |
| **tied** | **20,014** (Run 1, n=3) | **20,014** (Run 2, n=3) |

| metric | untied d4 | untied d8 | tied d4 | **tied d8** |
|---|---|---|---|---|
| EN | 6.9727 ±0.0254 | 6.9603 ±0.0301 | 6.9823 ±0.0363 | 6.9836 ±0.0455 |
| MI | 2.5021 ±0.1357 | 2.4665 ±0.0151 | 2.4626 ±0.0440 | 2.4957 ±0.0501 |
| SD | 43.5032 ±0.8549 | 43.5392 ±0.6714 | 42.8125 ±1.2559 | 43.4045 ±1.7558 |
| SF | 10.7123 ±0.3715 | 10.5957 ±0.3179 | 11.0253 ±0.3271 | 10.9962 ±0.2557 |
| AG | 3.9128 ±0.1494 | 3.8951 ±0.0842 | 4.1095 ±0.0810 | 4.0758 ±0.0883 |
| VIF | 0.7646 ±0.0401 | 0.7457 ±0.0124 | 0.7825 ±0.0013 | 0.7789 ±0.0122 |
| SCD | 1.2607 ±0.0469 | 1.2378 ±0.0405 | 1.3116 ±0.0388 | 1.2945 ±0.0423 |
| Qabf | 0.4325 ±0.0169 | 0.4226 ±0.0039 | 0.4441 ±0.0043 | 0.4374 ±0.0118 |
| SSIM | 0.9020 ±0.0152 | 0.9020 ±0.0028 | 0.9227 ±0.0061 | 0.9103 ±0.0085 |

### The four contrasts

| # | contrast | what it isolates | result |
|---|---|---|---|
| **1** | tied d4 → tied d8 | **depth, parameter count constant by construction** | **0 of 9** |
| 2 | untied d4 → untied d8 | depth *as the published table measures it* (+38,920 params) | 0 of 9 |
| 3 | untied d4 → tied d4 | capacity, −29,190 params | 0 of 9 |
| 4 | untied d8 → tied d8 | capacity, −68,110 params | 1 of 9 (VIF, *t* = 3.30) |

> ### 1 of 36 — against 1.8 expected by chance
>
> Four configurations spanning **4.40× in size** (20,014 → 88,124) and **2× in depth**. With no
> correction for multiplicity, 36 tests at the 5% level would produce **1.8 false positives on
> average**. We got **one**, ~~isolated, with nothing agreeing with it in the neighbouring contrast~~.
>
> ⚠️ **Corrected 2026-09-13.** The critical value 2.776 fixed here is the value at df = 4; five seeds
> against three give df ≈ 5.6 (exact 2.49), so it was conservative. At exact df one more cell
> crosses — SSIM in contrast 3, *t* = 2.70, *p* = 0.038, favouring tied — making **2 of 36**, still
> at the chance rate. Both cells favour the tied model, in different metrics, so they are not
> agreeing cells, but "isolated" does not survive. The registered verdict above stands as registered.
>
> **A count below the chance rate is not evidence of an effect**, and the VIF cell is not reported
> as one. This is the same rule this project applies to other people's tables, applied here to a
> cell that would have flattered our own method.

### What Run 2 establishes

**Unrolling depth does not measurably matter on this architecture and budget — and now that is a
measurement rather than an inference.** Contrast 1 is the experiment LRRNet's Table IV was
presented as being: 8 iterations of the solver against 4, with *nothing else different*, because
the parameter count is identical by construction rather than by matching after the fact.

Contrast 2 shows why the published version could not have answered it: the same depth change,
run their way, drags 38,920 parameters along.

**In the paper's favour, as plainly as the criticism:** with everything indistinguishable, choosing
a small model is right. Our tied model at depth 4 is smaller still — **2.46× fewer parameters than
the published configuration** — and equally indistinguishable. That is the whole efficiency claim
of chapter 6, now resting on four rows instead of one.

---

## Run 3 — M1 + M2 · ❌ the pre-registered failure branch

`mostafanada82/lrrnet-m2` · 237.2 min · 3.92 GPU-h · `scripts/analyse_m2.py`

### Integrity

| check | result |
|---|---|
| kernel elapsed | **237.2 min** · `torch 2.10.0+cu128` — the same pinned container as every other row |
| preflight | `total 49,204` · `refiners 29,190` · `1 distinct block` → capacity-matched **by construction** |
| width solver, on the training machine | **1 exact solution in 8,649 pairs**: d=16, e=69 |
| `[CONFIG]` | `num_block=4 tied=1`, all three |
| checkpoints | **19 refiner tensors** in each — M2 really trained |
| fusion | `loaded … (49,204 parameters)` on every seed |
| training | exit 0, **76.4–76.6 min**; loss 296,583 → 57,467 · 165,350 → 58,325 · 130,781 → 56,278 |

**The zero-init design is visible in the logs.** Initial loss 296,583 (M1+M2) against 296,780 (M1)
at the same seed — 0.07% apart. The refiners are exact identities at step 0 and are constructed
last, so the two runs genuinely start from the same weights.

### 1. The primary test: M1+M2 vs M1, **paired**

Because both runs start from identical weights in identical data order, the seed-wise differences
are paired. That removes the run-to-run noise the two runs *share* and is far more sensitive than
comparing means.

| metric | seed42 | seed43 | seed44 | mean Δ | paired *t* |
|---|---|---|---|---|---|
| EN | −0.0059 | −0.0107 | +0.0128 | −0.0013 | −0.18 |
| **MI** | **−0.0028** | **−0.0137** | **−0.0135** | **−0.0100** | **−2.77** |
| SD | −0.2481 | −0.2566 | +0.0564 | −0.1494 | −1.45 |
| SF | +0.0022 | −0.1448 | +0.1525 | +0.0033 | 0.04 |
| AG | +0.0004 | −0.0433 | +0.0655 | +0.0075 | 0.24 |
| VIF | +0.0003 | −0.0093 | +0.0059 | −0.0010 | −0.23 |
| SCD | +0.0059 | −0.0032 | +0.0238 | +0.0088 | 1.11 |
| Qabf | −0.0010 | −0.0053 | +0.0073 | +0.0003 | 0.09 |
| **SSIM** | **+0.0029** | **+0.0008** | **+0.0056** | **+0.0031** | **2.24** |

> **0 of 9 reach the 5% level** (critical value 4.303 at df=2). **M2 did not earn its 29,190
> parameters.**

### ⭐ The detail that decides the direction

The hypothesis was **specific**, not "something will move": chapter 4 measured that attention buys
**VIF, Qabf and MI**. Only two metrics move consistently across all three seeds — and one of them is
**MI, moving the wrong way in every seed.**

That is evidence *against* the transfer hypothesis, not merely an absence of evidence for it.

### 2. M1+M2 vs the published network, at exactly equal parameters

| metric | LRRNet (n=5) | M1+M2 (n=3) | diff | pooled σ | *t* |
|---|---|---|---|---|---|
| EN | 6.9727 ±0.0254 | 6.9811 ±0.0328 | +0.0084 | 0.30σ | 0.38 |
| MI | 2.5021 ±0.1357 | 2.4526 ±0.0461 | −0.0495 | 0.43σ | −0.75 |
| SD | 43.5032 ±0.8549 | 42.6631 ±1.1914 | −0.8401 | 0.86σ | −1.07 |
| SF | 10.7123 ±0.3715 | 11.0286 ±0.1788 | +0.3163 | 0.99σ | 1.62 |
| **AG** | 3.9128 ±0.1494 | 4.1170 ±0.0287 | +0.2042 | 1.66σ | **2.97** |
| VIF | 0.7646 ±0.0401 | 0.7815 ±0.0070 | +0.0169 | 0.51σ | 0.92 |
| SCD | 1.2607 ±0.0469 | 1.3204 ±0.0343 | +0.0598 | 1.39σ | 2.07 |
| Qabf | 0.4325 ±0.0169 | 0.4444 ±0.0058 | +0.0120 | 0.84σ | 1.45 |
| **SSIM** | 0.9020 ±0.0152 | 0.9258 ±0.0058 | +0.0238 | 1.85σ | **3.14** |

**2 of 9 clear the threshold, both in our favour — and neither is claimed.** Three reasons, all of
which would be applied to somebody else's table:

1. **They are not M2's doing.** The paired test says M2 contributed nothing. AG and SSIM stood at
   *t* = 2.41 and 2.70 for **M1 alone** in Run 1; they have drifted just over the line, not appeared.
2. **Welch *t* is inflated here.** Our n=3 σ is very small (AG ±0.029 against the baseline's ±0.149).
   A small-sample variance underestimate inflates *t* — the same fragility flagged against our own
   σ ratio in Run 1.
3. **They are not the predicted metrics.** VIF, Qabf and MI were the pre-registered targets. None
   moved.

### The two explanations, and why this run cannot separate them

| | |
|---|---|
| **(a)** | The chapter-4 assignment does not transfer to a low-rank/sparse split — because **that split is never enforced architecturally**. `L` and `S` are two channel slices of one tensor with no rank constraint anywhere in the forward pass. There may be no low/high-frequency structure there for an asymmetric assignment to exploit. |
| **(b)** | `d=16` is too narrow for attention over 128 channels. The width was **forced**: an exact parameter match has exactly one solution, and it spends 80% of the budget on the CNN. |

⛔ **What must not happen next: widening `d` until the result improves.** That would tune against the
very budget that makes this comparison capacity-matched, converting a clean negative into an
unfalsifiable positive. If (b) is to be tested it becomes a separate, declared experiment with its
own parameter accounting — and it stops being capacity-matched.

⭐ **If (a) is right it is a substantial finding in its own right**, and it closes a loop opened by
reading the source: LRRNet's low-rank/sparse decomposition would be a **name without an empirical
referent**. The consistent *downward* movement of MI points this way.

### 💰 A cost that belongs in the thesis

| | minutes per seed |
|---|---|
| M1 | ~66 |
| **M1 + M2** | **~77 (+17%)** |

**At identical parameter count.** Matching parameters is not matching compute: attention carries
matrix products and depthwise convolution is parameter-cheap but bandwidth-expensive. A paper
claiming "the same number of parameters" while omitting runtime is hiding a real cost — and this
project is now obliged to report its own.

---

## Run 4 — M3, Top-K corpus augmentation · ❌ failed, in the opposite direction

`mostafanada82/lrrnet-m3` · 6.76 GPU-h · six trainings (two corpora × three seeds), 65.4–67.5 min each

### ⚠️ First: the design brief for M3 was wrong, and the correction changed the run

`02_DESIGN.md` §2.5 described Top-K as an **architectural** module aligning channels between the two
encoder towers. Reading `top_k_alignment.py` shows it is nothing of the kind — it is a **data**
procedure that selects the k auxiliary patches closest in channel statistics, gamma-corrects them to
the primary corpus's mean, and merges them in. The description had been inferred from the *name*.

Ported at C=1: primary **LLVIP** 8,415 · auxiliary **MSRS** cropped to 5,415 patches of 128×128 ·
merged to **12,025**, so iterations per epoch are unchanged (`batches=1503`, asserted).

### The corpus construction worked exactly as intended

```
primary target visible mean = 39.631
random  : 3,610 patches, no alignment.  mean 58.999   (off by 19.368)
top-k   : mean |diff| of kept = 19.602  (all patches: 38.930)
          gamma = 1.1588   mean 31.494 -> 39.136      (off by  0.495)
```

**A 39× reduction in distribution mismatch.** The two corpora were verified identical in size
(12,025/12,025) and to share their primary half exactly (8,415/8,415), so the only difference is how
the auxiliary patches were chosen and treated.

### The result

| metric | k=0 (random) | **Top-K aligned** | σ ratio (k=0 / Top-K) |
|---|---|---|---|
| EN | 6.6683 ±0.0375 | 6.2294 ±0.1535 | 0.24× |
| MI | 2.2086 ±0.1407 | 1.8760 ±0.0483 | 2.92× |
| SD | 35.4892 ±1.5747 | 27.7713 ±5.0297 | 0.31× |
| SF | 10.1534 ±0.4188 | 8.6394 ±1.0733 | 0.39× |
| AG | 3.7734 ±0.1883 | 3.0583 ±0.5335 | 0.35× |
| VIF | 0.6211 ±0.0161 | 0.4968 ±0.0302 | 0.53× |
| SCD | 1.2976 ±0.0207 | 1.2036 ±0.0212 | 0.98× |
| Qabf | 0.3730 ±0.0024 | 0.2525 ±0.0591 | 0.04× |
| SSIM | 0.8279 ±0.0014 | 0.6823 ±0.0461 | 0.03× |

> **Median σ ratio 0.35× — Top-K is ~2.9× *noisier*, not more stable.** Less noisy on **1 of 9**.
> **6 of 9 means separate, and Top-K is worse on all nine.**

### 🔴 And both arms are far worse than adding nothing at all

| | EN | SD | VIF | SSIM |
|---|---|---|---|---|
| **M1 — LLVIP only** | **6.98** | **42.8** | **0.78** | **0.923** |
| k=0 — + random MSRS | 6.67 | 35.5 | 0.62 | 0.828 |
| Top-K — + aligned MSRS | 6.23 | 27.8 | 0.50 | 0.682 |

Replacing 3,610 LLVIP images with MSRS patches **damaged the model badly, and the alignment
machinery made it worse still**.

### ⚠️ A flaw in our own design, stated before any interpretation

**k = 3,610 out of a 5,415-patch pool is 67% — barely selective.** "Top-K" kept two thirds of
everything, and even that kept set sits **19.6 grey levels** from the target. MSRS is simply too
distant from LLVIP for 3,610 close patches to exist.

The gamma step then did what it was asked and not what was wanted: it matched the *mean* (31.5 →
39.1) while **SD fell from 35.5 to 27.8** — brightening dark patches compresses the histogram. The
statistic being optimised was matched; the one that mattered was not.

**Re-run at k = 1,000** — 18% of the pool, genuinely selective, and the value CrossFuse itself
publishes. Results below; they supersede this run, which is kept because the contrast between the
two is itself the evidence that the flaw was in our `k`, not in the mechanism.

---

## Run 4b — M3 at **k = 1,000**, the paper's own value · ⚪ a clean null

`mostafanada82/lrrnet-m3-k1000` · 6.98 GPU-h · six trainings, 69.2–69.7 min each

### The mechanism now behaves as designed

```
primary: 11,025 pairs   target visible mean = 44.743
random : 1,000 patches, no alignment.  mean 59.595          (off by 14.852)
top-k  : mean |diff| of kept = 6.552   (all patches: 38.575)
         gamma = 1.0111   mean 44.066 -> 44.157             (off by  0.586)
auxiliary overlap between the two arms: 192 of 1,000
```

**Selectivity improved from 2.0× to 5.9×** (6.55 against 38.58). And the gamma is now **1.0111 —
essentially neutral**, because good selection leaves almost nothing to correct. That is what the
mechanism looks like when it works: the previous run's `gamma = 1.1588` was dragging dark patches
upward and flattening their histograms, which is why SD collapsed.

### The result

| metric | k=0 (random, n=3) | Top-K aligned (n=3) | σ ratio |
|---|---|---|---|
| EN | 6.8800 ±0.0421 | 6.8362 ±0.0492 | 0.86× |
| **MI** | 2.3609 ±0.0155 | 2.2913 ±0.0174 | 0.89× |
| SD | 39.9692 ±1.4896 | 39.2143 ±1.5655 | 0.95× |
| SF | 10.5953 ±0.2382 | 10.5442 ±0.3047 | 0.78× |
| AG | 3.9933 ±0.0804 | 3.9536 ±0.1041 | 0.77× |
| VIF | 0.7310 ±0.0099 | 0.7155 ±0.0086 | 1.16× |
| SCD | 1.3747 ±0.0209 | 1.4082 ±0.0293 | 0.71× |
| Qabf | 0.4339 ±0.0055 | 0.4334 ±0.0049 | 1.13× |
| SSIM | 0.9133 ±0.0092 | 0.9101 ±0.0088 | 1.05× |

> **Median σ ratio 0.89× — no stability effect in either direction.** Top-K is less noisy on **3 of
> 9**, which is what a coin flip looks like. **0 of 9** variance ratios reach F(2,2). **1 of 9** means
> separate: MI, where Top-K is *worse* (*t* ≈ 5.2).

### How the two k values compare, and what that settles

| | k = 3,610 (67% of pool) | **k = 1,000 (18%)** |
|---|---|---|
| median σ ratio | 0.35× (Top-K 2.9× noisier) | **0.89×** |
| means separating | 6 of 9, all worse | **1 of 9** |
| variance ratios beyond F(2,2) | 2 of 9 | **0 of 9** |
| gamma | 1.1588 | **1.0111** |

The first run was not measuring the mechanism. It was measuring **the damage of swapping 30% of the
corpus for distributionally distant data**. With `k` set to the published value the damage
disappears — and so does any effect at all.

### ⭐ The finding

| | corpus size | what varied | σ |
|---|---|---|---|
| **chapter 4** (`k=0` vs released) | 1,718 → 918, **changes** | the mechanism **+ 47% of the data** | rises 3.35× without it |
| **Run 4b** | 12,025, **fixed** | **the mechanism alone** | **0.89× — nothing** |

> **The benefit CrossFuse attributes to Top-K Selective Channel Alignment is attributable to the
> extra training data, not to the selection-and-alignment mechanism.** Adding an auxiliary corpus
> matters. *Which* patches you pick, and whether you gamma-align them, does not.

**This is, as far as we can establish, the first volume-controlled test of the mechanism.** The
paper's own design cannot be one: `k=0` versus `k=1000` changes the corpus size and the mechanism
together, so it can never separate them.

⚠️ **Stated with its limit.** This is LRRNet, not CrossFuse, and MSRS, not M3FD. The strict claims
are two: chapter 4's stability effect **cannot be attributed to the mechanism**, because its design
confounds it with volume; and a **volume-controlled test in a second architecture family finds
nothing**. The inference back to CrossFuse is suggestive, not proven, and is written that way in
`TIER3_RESULTS.md` §0.2.

---

## Run 6 — the positive control · ⭐ the sensitivity floor of the nine metrics

`mostafanada82/lrrnet-poscontrol` · 3.69 GPU-h · nine trainings/evaluations in three arms

### Why this run had to exist

Every chapter-6 result up to here is a null — capacity, depth, asymmetric refiners, Top-K. Each is
ambiguous between *"the intervention did nothing"* and *"the metrics cannot see"*, and nothing in
this work had separated them. An examiner would ask, and we had no answer.

Three deliberately degraded configurations, in increasing subtlety, each on three seeds:

| arm | intervention |
|---|---|
| **A** | **untrained** — random weights, zero gradient steps, verified tensor-by-tensor against a freshly seeded model |
| **B** | **half the training budget** — 2 epochs instead of 4 |
| **C** | **`num_block = 1`** — a quarter of the depth at **exactly the same 20,014 parameters** |

### The ladder

| arm | detected |
|---|---|
| untrained (0 epochs) | **7 of 9** at n=5 · 6 of 9 at n=3 |
| half budget (2 epochs) | **0 of 9** |
| depth 1 (same parameters) | **0 of 9** |

> ## 🔄 SUPERSEDED IN SCOPE — 2026-09-09
>
> **The untrained arm was repeated on CrossFuse and the finding below narrowed again.** On the
> second architecture **9 of 9 metrics detect**, and SF and AG — which fail here — are the
> **strongest** detectors there (*t* = 16.34 and 15.04). Two different metrics fail instead:
> **MI and VIF rank the untrained network *above* the trained one** on CrossFuse.
>
> The finding below stands **for LRRNet**. What does not stand is any reading of it as a statement
> about the metrics in general. See "Run 7" at the end of this file.
>
> **The replacement claim is stronger:** whether a metric responds is a property of the metric *and*
> the network being scored, so **~~a control cannot be inherited~~ *(narrowed 2026-09-13: cannot be **assumed to carry over** — shown for two architectures)* from another paper — every table
> needs its own.**

### ⭐ Finding 1 — at n=5, two of the nine standard metrics did not separate an untrained LRRNet at five seeds

Reported at **n=5 per arm**. The first pass used n=3 and is superseded; both are shown, because the
change between them is itself the result. See "Hardened, not defended" below.

| metric | trained (n=5) | untrained (n=5) | *t* @ n=5 | *t* @ n=3 | |
|---|---|---|---|---|---|
| Qabf | 0.4464 ±0.0072 | 0.1754 ±0.0199 | **−28.58** | −41.78 | ✅ |
| VIF | 0.7865 ±0.0063 | 0.1365 ±0.1137 | **−12.77** | −14.33 | ✅ |
| SSIM | 0.9235 ±0.0076 | 0.0127 ±0.2709 | −7.52 | −5.10 | ✅ |
| SCD | 1.3171 ±0.0407 | 0.2790 ±0.3469 | −6.65 | −5.19 | ✅ |
| MI | 2.4728 ±0.0350 | 1.3929 ±0.4452 | −5.41 | −6.10 | ✅ |
| EN | 6.9862 ±0.0269 | 6.2815 ±0.3378 | −4.65 | −3.20 | ✅ |
| **SD** | 42.8841 ±0.9908 | 29.5112 ±9.3537 | **−3.18** | −2.73 | ✅ **reversed** |
| **SF** | 10.9088 ±0.2980 | 9.0165 ±2.9436 | −1.43 | −1.59 | ❌ |
| **AG** | 4.0585 ±0.1049 | 3.3860 ±0.9809 | −1.52 | −1.58 | ❌ |

> **SF and AG did not separate random weights from a fully trained LRRNet, even at five seeds per
> arm. A third, SD, requires five — it fails at three.** (Scoped to LRRNet: on CrossFuse these two
> lead — see Run 7.)

The mechanism matters: the means *are* far apart (SD 42.9 → 29.5), but the untrained arm's variance
is enormous, and larger in the five-seed estimate (±9.35 on SD, ±2.94 on SF), so the statistic absorbs the
difference. One untrained seed scored **AG = 4.24 against the trained model's 4.06**.

**Both statements are damaging, in different ways.** A metric that cannot separate an untrained
model from a trained one is not an ablation instrument at all; a metric that needs five training
runs to notice one is not usable in a table that reports a single run.

### 🛡️ Hardened, not defended

The n=3 version of this table was already written into the reports when it was noticed that the
paper's own standard condemned it: a headline claim resting on three seeds, in a paper about
claims resting on too few seeds. Two further seeds were added to **both** arms, with the reading of
every outcome — including "the finding narrows" — written into the notebook before the run.

It narrowed. **SD moved from `ns` to significant**, and the wording "three metrics cannot detect"
was withdrawn everywhere it appeared, as the pre-registration required.

> Running the experiment that could damage our own headline, and then applying its result, is the
> only version of this paper's argument that is worth making.

### ⚠️ Finding 2 — and a correction to our own first reading of arms B and C

The initial reading of the 0-of-9 results was that the metrics are too coarse to see them. **The
training losses refute that**, and the corrected reading is the stronger one:

| seed | loss @ 2 epochs | loss @ 4 epochs |
|---|---|---|
| 42 | 60,479 | 57,634 |
| 43 | **58,219** | 58,549 |
| 44 | 59,808 | 56,390 |

The two models are genuinely equivalent — the metrics stop registering further training by epoch 2 (the loss itself
keeps falling slowly, claim #14), and at seed 43 half the budget reached a *lower* loss. Likewise, at seed 42, `nb=1`
trains to **57,909** against `nb=4`'s **57,634**.

**So arms B and C show no detected difference, with comparable training losses (non-detection, not demonstrated equivalence).** That distinction is what
makes them usable.

### ⭐ Finding 3 — the unrolling does nothing

With the parameter count identical at 1, 4 and 8 by construction, the depth ladder is now complete:

```
depth 1  ≈  depth 4  ≈  depth 8      no metric separates them, and all three reach the same loss
```

> **No advantage of eight iterations over one is detected** at the tested sample sizes — in fused-image
> metrics and in training loss (non-detection, not demonstrated equivalence). This is a result about the idea LRRNet is named after, not about a setting inside it.

`nb=1` also trains in **522 s/epoch against 980 s** — the extra iterations cost 88% more compute and
return nothing.

### What this settles for the rest of the chapter

| question | answer |
|---|---|
| Do the metrics detect a real difference when one exists? | ✅ yes — **7 of 9** on arm A at n=5 |
| Are the four earlier nulls artefacts of a blind instrument? | ❌ no — and arms B and C are corroborated by their loss curves |
| Where is the sensitivity floor? | between **total failure to train** and **half the training budget** — a very wide gap |

The four nulls therefore stand as measurements rather than as absences of evidence. What must be
stated alongside them is the resolution: on this architecture and budget these nine metrics resolve
catastrophe reliably and little else, and three of them do not resolve even that.

---

## Cost (superseded section below retained for the k=3,610 run)

### ⭐ What this already establishes, regardless of the re-run

Chapter 4 concluded that Top-K buys stability, from `k=0` runs that removed the mechanism **and 47%
of the training corpus**. Run 4 holds corpus size constant and finds the mechanism alone does not
buy stability — it costs it.

| | corpus size | σ verdict |
|---|---|---|
| chapter 4 (A1b/c/d) | **1,718 → 918, changes** | σ rises 3.35× without Top-K |
| **Run 4** | **12,025, fixed** | σ *falls* without Top-K |

So the chapter-4 stability effect is **more plausibly attributable to training-set size than to
Top-K**. Different architecture and different corpora, so this is "consistent with", not "proves" —
and the caveat is now recorded in `TIER3_RESULTS.md` §0.2 against our own earlier claim.

---

## Cost

| | GPU-h |
|---|---|
| Run | GPU-h | training runs |
|---|---|---|
| 1 — M1 at `nb=4` | 3.09 | 3 |
| 2 — M1 at `nb=8` | 3.47 | 3 |
| 3 — M1+M2 (incl. 0.04 for the aborted v1) | 3.96 | 3 |
| 4 — M3 at k=3,610 (incl. 1.20 for the aborted v1) | 7.96 | 6 |
| 4b — M3 at k=1,000 | 7.04 | 6 |
| 6 — positive control | 3.58 | 6 (+3 untrained) |
| **chapter 6** | **29.1** | **27** |

**Project total: 244 training runs, ≈374 GPU-hours** (Run 31 added 5 CrossFuse runs / 19.7 h measured from its three kernel logs, 2026-11-25/26; Runs 29, 30 and 32 trained nothing) (239, not 238: corrected 2026-10-13 -- Tier 3 had eight runs and was counted as seven; the hours were right); every earlier total below is likewise one short, and CrossFuse is 25 runs, not 24. (228 / ≈315 before Run 28, which added 10 runs and 39.4 h on 2026-10-10/11; 214 / ≈297 before Runs 25-27, which added 14 runs and about 17.6 h on 2026-10-06; 190 / ≈269 before Runs 22, 23 and 23b, which added 24 runs and 28.3 h measured from their kernel logs, 2026-10-03; 136 / ≈209 before Runs 17, 18 and 20, which added 54 runs and 60.2 h measured from their kernel logs: 24 / 26.8 h, 18 / 20.3 h, 12 / 13.1 h; 118 / ≈171 before Runs 11–14, which added 18 runs and 37.6 h measured from their kernel logs: CrossFuse 4 runs / 11.5 h, chapter 6 14 runs / 26.1 h) — CrossFuse 24 runs / ≈78.3 h, LRRNet 71 runs /
≈75.2 h (17 / ≈14.6 h before Run 9, which added 18 runs and 20.1 h measured from its kernel logs; Run 10
added 36 runs and 40.5 h, its seed loops summed from the kernel logs),
chapter 6 41 runs / ≈55.2 h. The chapter-6 figures come from the weekly quota trail rather
than from summing kernel elapsed times, so they include the two aborted attempts and Kaggle's own
session overhead; kernel-only times are ≈1.2 h lower.

> ⚠️ **Corrected 2026-09-07.** The reports had carried "33 runs, ≈91 GPU-hours" in five places. That
> figure was never derived; it is replaced above by a per-run accounting that can be checked.

Weekly quota after Run 6: **3.75 h remaining**, refreshing 2026-09-12.

### The aborted Run 3 v1, recorded because the gate is the point

The first launch died in the preflight after **0.01 GPU-h**. `check_tied_capacity.py` was
**inheriting `LRR_TIED` and `LRR_M2` from its caller**: the notebook exports both so that
`fuse_lrrnet.py` can load the checkpoint, and the verifier then built a model *with* refiners while
checking that the default was unchanged.

A verifier whose answer depends on who invoked it verifies nothing. Fixed by setting both flags
explicitly at every stage, and — the part that mattered — **re-verified under the polluted
environment that broke it** (`LRR_TIED=1 LRR_M2=1` exported locally), because running it cleanly
again would have reproduced the original false pass.

Total cost of the defect: **0.04 GPU-h**. The failure it prevented was 3.9 h and a mislabelled row.

---

## Reproducing

```bash
python patch_lrrnet.py --check        # 14 patches, dry-run against upstream
python check_tied_capacity.py         # 49,204 -> 20,014; default unchanged tensor-for-tensor
python analyse_m1.py --csv lrrnet_metrics.csv
```

Kaggle cells: [`kaggle_m1_cells.md`](../../../studied_papers/LRRNet/kaggle_m1_cells.md)

---

## Run 7 — the positive control on the second architecture · ⭐⭐⭐ the claim narrows, the recommendation widens

`mostafanada82/tier3-poscontrol-cf` v2 · CrossFuse · 5 seeds per arm · 2026-09-09

### Why this run had to exist

Run 6 concluded that two of the nine metrics cannot detect an untrained network. That was measured
on **one architecture**, and §9 of the paper said so and named this experiment as what would settle
it. Every other result in this work is measured on two architecture families precisely so that a
defect in one design is not mistaken for a property of the field; the control was the exception, at
the paper's most quotable point.

**Both readings were fixed before the run**, in the source of `analyse_poscontrol_cf.py`.

### The result — OUTCOME 2, the one that costs us the general claim

| metric | trained (n=5) | untrained (n=5) | *t* | |
|---|---|---|---|---|
| **SF** | 14.4548 ±0.3199 | 10.7923 ±0.3859 | **+16.34** | ✅ |
| **AG** | 5.3203 ±0.1609 | 3.9354 ±0.1285 | **+15.04** | ✅ |
| MI | 2.2967 ±0.0280 | **3.4738 ±0.1795** | −14.49 | ✅ ⚠️ untrained higher |
| VIF | 0.7843 ±0.0083 | **0.9450 ±0.0260** | −13.18 | ✅ ⚠️ untrained higher |
| EN | 7.1787 ±0.0360 | 7.0140 ±0.0329 | +7.55 | ✅ |
| SCD | 1.5420 ±0.0504 | 0.9278 ±0.2198 | +6.09 | ✅ |
| SSIM | 0.9524 ±0.0117 | 0.9041 ±0.0196 | +4.74 | ✅ |
| SD | 49.3629 ±1.3853 | 46.4051 ±0.9611 | +3.92 | ✅ |
| Qabf | 0.5396 ±0.0082 | 0.5258 ±0.0041 | +3.38 | ✅ |

**9 of 9 detect.** SF and AG — which fail outright on LRRNet — are the **strongest** detectors here.

### ⭐ Finding 1 — the failure belongs to the metric-architecture PAIR

The two architectures fail differently because their untrained outputs differ:

| | untrained LRRNet | untrained CrossFuse |
|---|---|---|
| output | high-frequency noise | near-copy of the visible image (r = 0.997, measured 2026-09-13) |
| SF spread across seeds | ±2.94 (huge) | ±0.39 (tiny) |
| SF / AG verdict | ❌ cannot detect | ✅ strongest detectors |

We had written the opposite reading down as plausible — SF and AG are gradient statistics, so the
mechanism looked general. **This run refutes that.**

### ⭐⭐ Finding 2 — two metrics rank the untrained network ABOVE the trained one

On CrossFuse, MI and VIF do not merely fail to prefer the trained model; they prefer the untrained
one, in every seed. Both measure information shared with the sources, and an untrained network's
output ~~is close to a passthrough~~ *(withdrawn 2026-09-13, claim #12: refuted for an averaging passthrough; see RUN 7 note)*. **Training deliberately discards source information to produce a
better fusion, and these two measures penalise it for that.**

> ⚠️ **The mechanism in the paragraph above is WITHDRAWN (claim #12, 2026-09-13).** Tested against
> zero-network baselines on the same 25 TNO pairs, reading written into the script's docstring
> (committed together with the result in 785b024, so its precedence is our statement, not a record)
> (`studied_papers/CrossFuse/Tier3_CrossFuse/scripts/passthrough_baseline.py`):
>
> | | MI | VIF | SF | AG |
> |---|---|---|---|---|
> | (IR + VI) / 2 | 1.60 | 0.57 | 6.95 | 2.67 |
> | max(IR, VI) | 4.71 | 0.92 | 10.06 | 3.77 |
> | IR copied | 4.96 | 1.03 | 7.54 | 2.89 |
> | VI copied | 5.16 | 1.04 | 10.82 | 3.90 |
> | CrossFuse trained (n=5) | 2.30 | 0.784 | 14.45 | 5.32 |
> | CrossFuse untrained (n=5) | 3.47 | 0.945 | 10.79 | 3.94 |
>
> The average — the literal passthrough — scores MI and VIF **far below** the trained network, so
> the mechanism fails in its averaging form. The pixel-wise maximum (VIF 0.924) lands close to the
> untrained network (0.945), so a source-*selecting* passthrough is not refuted. What MI and VIF reward is **copying a source**: either band returned
> unchanged beats both trained networks and the untrained one. The inversion stands, and is a
> property of the measures; why an untrained CrossFuse lands where it does is not established.
> SCD is undefined for the two copies (a zero difference image, 0/0) and is excluded there only.

A metric that ranks an untrained network above a trained one is worse than one that cannot tell them
apart — and it is a failure a single-architecture study could not have found, because both metrics
pass cleanly on LRRNet.

### What this costs and what it buys

| | |
|---|---|
| **lost** | "two of the nine metrics cannot detect an untrained network" as a claim about *the metrics* |
| **kept** | the same claim, scoped to LRRNet, where it is measured |
| **gained** | **~~a control cannot be inherited~~ *(narrowed 2026-09-13: cannot be **assumed to carry over** — shown for two architectures)*. Whether a measure responds is a property of the measure and the network together, so every table needs its own** |

The recommendation is stronger than the one it replaces, and it now rests on a reversal we ran
ourselves rather than on an argument.

### Run cost, and two failures worth recording

**v1 failed.** PC45 and PC46 shared an `--outdir`, and `train.py` writes resume checkpoints as
`ckpt_ep<NNN>.pth` with **no tag in the name**. `--resume auto` globs that pattern and takes the
last, so PC46 loaded PC45's final epoch, set `start_epoch = 299`, and would have saved a byte-for-byte
copy of PC45 — a "seed replicate" that was the same network twice. PyTorch 2.6's `weights_only=True`
default turned that into a crash instead of a silent copy. **Relying on that is luck, not
engineering**; v2 gives each tag its own `--outdir`.

Consequence worth carrying: **`--resume auto` does not work in this container at all**, so a long
run that is interrupted is lost rather than resumable.

| | |
|---|---|
| v1 | **3.20 GPU-h** (11,523 s) — PC45 to 299 epochs + 5 untrained checkpoints; PC46 crashed. Salvaged as a dataset |
| v2 | **3.34 GPU-h** (12,016 s) — PC46 to 299 epochs, plus inference and evaluation of all seven checkpoints |
| total | **6.54 GPU-h** |

⚠️ An earlier version of this table put v2 at "~1.0 GPU-h". That was an estimate written before the
run finished and never checked against the log; the kernel actually ran 12,016 s. Both figures here
are now read from the kernel logs. **An unchecked estimate in a results file is the same defect this
work documents in other people's tables.**

Twelve defects were found in the analysis pipeline before it was run on real data; eight were silent
and nine biased towards *finding* significance. They are listed in `CONTEXT.md`.
