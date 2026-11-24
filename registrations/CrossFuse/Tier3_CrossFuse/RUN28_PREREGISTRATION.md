# Run 28 (= C2): the volume-controlled Top-K test on CrossFuse itself. Pre-registration

**Written 2026-10-10, committed on its own before any Run 28 kernel or corpus exists.**
- Nothing below may change after the first Run 28 output.
- A correction may be added only as a dated block at the end, and only before any metric is read.

## Why

On CrossFuse, removing Top-K (k = 0, Run 11 arm A) raised seed-to-seed spread by a median factor of **2.17** over the
released configuration (21 of 27 cells larger). But k = 0 also removes the 800 auxiliary patches, **47 % of the
1,718-patch corpus**, so "Top-K buys stability" cannot be separated from "more data buys stability".

Runs 4b and 27 separated them on LRRNet, with MSRS as the auxiliary corpus, and detected no stability effect of the
mechanism at eight seeds per arm (S2). The paper says plainly that the same result on CrossFuse is "the natural
reading, not a measurement". Run 28 is that measurement: CrossFuse, its own auxiliary source (M3FD), at fixed corpus
volume.

## Design

Two arms of 1,718 patches each. The **only** difference is how the 800 auxiliary M3FD patches were chosen.

| arm | corpus (1,718 patches) | seeds |
|---|---|---|
| `topk` | the released `RSM3_imgsize_64_stride_200_k_1000.h5`, unchanged: 918 RoadScene + 800 M3FD patches selected and gamma-aligned by the authors | **47–51** |
| `rand` | the same 918 RoadScene entries, copied byte for byte from the released file (as `build_k0_h5.py` does), + **800 M3FD patches drawn uniformly at random**, without gamma alignment | **47–51** |

**How the random patches are made.**
- Source: raw M3FD_Fusion (300 pairs, the TarDAL release).
- Cropping: the released `crop_dataset` (64 × 64, stride 200).
- Drawing: 800 patches from the full pool, with `numpy.random.default_rng(2028)`.
- Processing: the released script's path to the `.h5` (visible to Y, infrared single channel), with the gamma step
  skipped. Nothing else differs.

**Training.** The released code and the R1 configuration:
- `coeff_tv 5.0`, code SSL schedule;
- 160 epochs, `epoch_gap 40`, `optim_step 20`, batch 8.

Both arms have 1,718 patches, so both take 215 steps per epoch and need no step matching. They use the same pinned
container image as Tier 3. Seeds 47–51 have not been used on CrossFuse before.

**Scoring.** TNO (25), RoadScene (40) and MSRS (361), with the same evaluator as Run 11.

**Kernels.** Four on `mostafanada82`, each well under the 12-hour cap:
- `rand` 47–49, `rand` 50–51, `topk` 47–49, `topk` 50–51;
- two at a time.

There are no automatic status polls: the outputs are fetched once, when the user asks.

## Kill criteria (before any metric is read)

1. **The random corpus is what it claims to be.** The builder prints:
   - pool size;
   - the number of patches drawn (800);
   - total patches (1,718);
   - `max |copy − released|` over the 918 RoadScene entries, which **must be 0.0**;
   - the mean channel distance to the RoadScene mean for the drawn patches and for the whole pool (descriptive: random
     draws should sit near the pool);
   - how many of the 800 drawn patches are pixel-identical to a released auxiliary patch (descriptive: the overlap).

   If the total is not 1,718, or the RoadScene copy is not exact, the run is reported as failed.
2. **For each run:**
   - `[CONFIG]` shows the seed and the corpus file;
   - 160 epochs completed, exit 0, one final checkpoint;
   - 25 / 40 / 361 images scored.
3. **No two of the ten runs are identical** on any test set.
4. **Minimum n.** If quota or a crash ends a kernel early, the reading is made at n ≥ 4 per arm, with the n stated.
   Below that, the run is reported as incomplete and not read.

## The reading (fixed now)

### Primary: stability, as in Run 27, so the two architectures are read the same way

- For each of the seven deciding metrics (EN, MI, SD, VIF, SCD, Qabf, SSIM) on **TNO**: the variance ratio
  σ²(rand)/σ²(topk), with an F-test, two-sided 5 %, df (n−1, n−1).
- Holm across the seven.

| outcome | condition | what the paper and thesis will say |
|---|---|---|
| **S1: STABILITY EFFECT** | ≥ 1 survives with `rand` more variable, none the other way | "On CrossFuse, at fixed corpus volume, Top-K selection reduced run-to-run spread on [metrics]." The LRRNet result is then stated as architecture-dependent |
| **S2: NONE DETECTED** | nothing survives | "No stability effect of the mechanism was detected on CrossFuse either, at five seeds per arm (a variance ratio of about 9.6 is detectable)." |
| **S3: DESTABILISES** | ≥ 1 survives with `topk` more variable, none the other way | "At fixed volume, Top-K increased run-to-run spread on CrossFuse on [metrics]." |
| **S4: MIXED** | survivors in both directions | the metrics and directions are listed, with no single-direction claim |

### Attribution: does the 2.17× of Run 11 belong to the mechanism or to the volume?

Fixed now and reported under every outcome:
- the median σ(rand)/σ(topk) over all **27 cells** (9 metrics × 3 test sets), with sample SDs;
- a bootstrap 95 % interval over runs (10,000 resamples, seed 2028).

This is the same statistic that gave 2.17 for k = 0 against the released configuration.

| reading | condition | statement |
|---|---|---|
| **V: VOLUME** | median ≤ 1.25 **and** the interval excludes 2.17 | "Most of the instability of the k = 0 ablation is explained by the smaller corpus, not by Top-K." |
| **M: MECHANISM** | median ≥ 1.75 **and** the interval excludes 1.0 | "Random auxiliary patches at the same volume are about as unstable as removing them: the stability is attributable to Top-K." |
| **U: UNRESOLVED** | anything else | "At five seeds per arm the attribution cannot be made; the interval is [a, b]." |

### Secondary, under every outcome

- **Means:** Welch's t on all 27 cells, two-sided 5 %, at each comparison's own df, Holm within each test set.
- **Equivalence:** reported where the registered ±1.75 σ margin applies.
- **Same-seed check (descriptive, not counted):** none is planned. If a session crash forces a same-seed rerun, the
  pair is reported beside the result, as before.

### What does not change under any outcome

The published ablation (Table VI) cannot attribute anything to Top-K, because k = 0 moves corpus size with the
mechanism. That is a property of the published design, not of this run.

## Precondition

The raw **M3FD_Fusion** set has never been held in this project (CONTEXT.md lists it as needed only for a K-sweep).
- How it reaches the kernel (a Kaggle dataset, or a download inside the kernel) is an implementation choice, not a
  reading.
- The builder prints the number of image pairs (expected 300) and a SHA-256 over the sorted file list. Both are
  recorded with the result.

## Cost

Ten training runs at about 3.3 h each (R1 measured about 3.3 h at 1,718 patches and 160 epochs), plus corpus
building and scoring: about **35 GPU-hours**. That fits one week's 45 h quota.

## Correction, 2026-10-10, before any kernel ran or any corpus existed

These are implementation changes only. No reading, threshold or outcome above changes.

1. **The kernel layout changes to six kernels.**
   - `cf-run28-corpus` (CPU, no GPU quota, internet on) builds both corpora **once** and asserts kill criterion 1
     before any GPU time is spent.
   - Then five GPU kernels, `cf-run28-s47` … `s51`. Each runs **one seed in both arms** (`TK<s>`, `RD<s>`) in the
     same session, alternating which arm runs first by seed parity.
   - Reason: R1b measured 3.57 h for one run, so three runs plus scoring would sit near the 12-hour cap. Pairing the
     arms within a session also spreads session-to-session differences evenly over the two arms.
2. **Where the `topk` corpus comes from.**
   - It is downloaded from the authors' Google Drive link in the CrossFuse README, and its SHA-256 is recorded.
   - It is accepted only if its 918 RoadScene entries are identical (max |diff| = 0) to Run 11's k = 0 file, which was
     cut from the released file.
   - Both arms then read the files this one kernel wrote.
3. **The descriptive overlap is counted by infrared patch.** The released auxiliary visible patches are
   gamma-corrected, so they cannot match a random draw pixel for pixel. Their infrared patches are not corrected, so
   the count compares those.
