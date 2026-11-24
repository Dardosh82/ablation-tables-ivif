# Run 9 — pre-registration

**Written 2026-09-09, before any run was launched.** Nothing below may be edited after the first
result exists; corrections go in a dated block at the end, as they did for Runs 6, 7 and 8.

---

## Why this run exists

The paper's own limitation section says sample sizes are small. It is more specific than that: at
`n = 3` per ablated row, a two-sided Welch test at 5 % with 80 % power cannot resolve an effect
smaller than **3.5 σ**. At `n = 5` it cannot resolve one smaller than **2.3 σ**.

| n per arm | df | minimum detectable difference |
|---|---|---|
| **3** — every ablated row of Table IV | 4 | **3.5 σ** |
| **5** — the reference configuration | 8 | **2.3 σ** |
| 8 | 14 | 1.75 σ |
| 10 | 18 | 1.53 σ |

So the paper's headline — *zero of 27 comparisons reach significance* — currently rests on a design
that **could not have found a 1 σ effect**, and 1 σ is large by this field's standards. A reviewer is
entitled to read every null here as *no power* rather than *no effect*, and the positive control does
not fully answer that: it shows the metrics resolve an **untrained** network, which is an enormous
difference, not that they would resolve a plausible ablation.

**This run buys power, and nothing else.** Every row of Table IV goes to `n = 8`.

## ⚠️ This run can damage the paper's headline

It is a genuine risk, not a formality. More power means a cell that currently reads *not
significant* may cross the threshold. If that happens, **zero of 27 becomes one of 27** — or more —
and the abstract, §5 and the conclusion all change. That is precisely what happened in Run 7, when
SD moved from *cannot detect* to *detects* at `n = 5` and the claim "three metrics cannot detect an
untrained network" was withdrawn everywhere.

The readings below are fixed now so that the outcome cannot be chosen after seeing it.

---

## What is run

`num_block` is LRRNet's Table IV variable. Existing seeds:

| setting | seeds now | n | seeds to add | new n |
|---|---|---|---|---|
| `nb = 2` | 42–44 | 3 | **45, 46, 47, 48, 49** | 8 |
| `nb = 4` (published choice) | 42–46 | 5 | **47, 48, 49** | 8 |
| `nb = 6` | 42–44 | 3 | **45, 46, 47, 48, 49** | 8 |
| `nb = 8` | 42–44 | 3 | **45, 46, 47, 48, 49** | 8 |

**18 new training runs.** Everything else is held exactly as in stages 1–4: 4 epochs, batch 8,
128×128 grayscale, LLVIP corpus, the same nine metrics through CrossFuse's `Evaluator`, TNO for
scoring. `nb = 8` trains across both GPUs, as it must — it OOMs on one 16 GB card — and that
deviation is already recorded wherever those rows are used.

**Cost, from measured per-run times:** nb2 5 × 0.77 h, nb4 3 × 1.10 h, nb6 5 × 1.10 h,
nb8 5 × 1.07 h ≈ **18.0 GPU-h**.

---

## The readings, fixed before the run

### OUTCOME 1 — Table IV still yields 0 of 27 at n = 8

The headline survives a design that can now resolve **1.75 σ** instead of 3.5 σ. The claim gets
materially stronger and the limitation shrinks from "we may lack power" to "we had power to 1.75 σ
and found nothing". §9's sample-size paragraph is rewritten to state the new bound. **This is the
outcome that helps us, which is why the alternative is written below with equal weight.**

### OUTCOME 2 — one or more comparisons reach significance at n = 8

**"Zero of 27" is withdrawn** and replaced by the true count, everywhere it appears: abstract, §1
contribution 2, §5, §8, §9, the long report, the short report, the deck and the speaker report. It
becomes **withdrawn claim #9**, and it is recorded in the register with the same prominence as the
eight before it.

The paper does **not** thereby lose its argument, and this must not be spun as if it did. Any cell
that emerges will be read against the same two rules already in force:

- **Multiplicity.** 27 comparisons at 5 % expect **1.35** false positives. **One significant cell is
  at the chance rate and is not a finding in either direction** — it is reported as one cell, named,
  with the expected count beside it. Two or more agreeing cells would be a finding.
- **Direction.** If a cell favours the published choice `nb = 4`, that is evidence *for* the paper
  under study and is reported at the same length as evidence against it.

The wider claim — that a single-run table cannot support its conclusions — survives either way,
because it rests on σ being comparable to the published differences, not on the count being zero.

### OUTCOME 3 — a run fails to train, or the arm is incomplete

Nothing is reported for that setting. A partial arm is not a measurement. The existing `n` stands
and the failure is recorded.

---

## Kill criteria

Any of these voids the run for the affected setting:

1. **Fewer than 8 completed seeds** for a setting → that setting keeps its current `n`, and the
   shortfall is stated rather than quietly averaged over.
2. **A new seed's loss trajectory outside the range of the existing seeds by more than 3×** →
   that run is excluded and named. Its exclusion is reported.
3. **`nb = 8` failing to engage both GPUs** — the existing assertion on `[GPUS] parallel= True`
   stays. A single-card `nb = 8` run is not comparable with the rows already collected.
4. **Any non-finite metric value** → the run is a broken evaluation, not a metric failing to
   respond, and the analysis must halt rather than treat it as data.
5. **A new seed reproducing an existing seed's metrics to within 1e-12** → a copied checkpoint,
   not a new run.

---

## The statistic, not negotiable afterwards

Welch's *t*, two-sided, 5 %. At `n = 8` versus `n = 8` the critical value is **2.145** (df = 14),
replacing 2.776. **This is not a loosened threshold** — it is the correct value for the new design,
and it is written here before any number exists. The comparison set is unchanged: each ablated row
against `nb = 4`, nine metrics, three rows = **27 comparisons**.

The pooled σ used for reporting is recomputed at the new n. No existing number is re-used.

---

## What this run is not

It is **not** an attempt to make a null more convincing by adding data until it holds. The outcome
that would follow from a cell crossing the threshold is written above, in the same detail, and will
be applied. It is also not a new experiment: no setting, corpus, metric or evaluator changes. Only
`n` changes.

---
---

## RESULT — recorded 2026-09-13, after all 18 runs completed

### Kill criteria, checked before any statistic

| # | criterion | result |
|---|---|---|
| 1 | 8 completed seeds per setting | ✅ nb = 2, 4, 6, 8 all at n = 8, seeds 42–49 |
| 2 | loss trajectory within 3× of the existing range | ✅ all 18 — largest excursion 2,069 against a tolerance of 6,470 (`check_run9_losses.py`) |
| 3 | nb = 8 on both GPUs | ✅ `[GPUS] 2 parallel= True` in all 5 runs |
| 4 | every metric finite | ✅ 288 of 288 |
| 5 | no copied checkpoint | ✅ no two of the 32 runs share a metric vector |

Criterion 2 needed an operational reading the text above did not give — *which* point of the
trajectory, and 3× of *what*. It was fixed in `check_run9_losses.py` (final logged loss; distance
outside the existing seeds' range against three times that range's width) and **run before
`analyse_run9.py` computed anything**. The first logged loss was rejected as the comparison point
because it is a property of the initialisation: it already spans 107k–197k within one setting.

### The statistic

**1 of 27** comparisons reaches the 5 % level.

| cell | Welch *t* | critical (df floor) | exact critical | *p* | favours |
|---|---|---|---|---|---|
| **SCD, nb = 4 vs nb = 2** | **−2.452** | 2.262 (df 9.7 → 9) | 2.237 | 0.035 | **nb = 2** |

Next largest: AG 4 vs 2 (−1.77, crit 2.160), SF 4 vs 2 (−1.77, crit 2.201). Nothing else within
reach. Cross-checked with `scipy.stats.ttest_ind(equal_var=False)`: identical *t*.

### The reading applied — OUTCOME 2

- **"Zero of 27" is withdrawn** and replaced by **"one of 27"** everywhere it appears.
- **Multiplicity:** 27 tests expect **1.35** false positives. One significant cell is **at the
  chance rate and is not a finding in either direction.** It is reported as one named cell with
  the expected count beside it.
- **Direction:** it favours **nb = 2** — the setting the published table ranks clearly *worst* —
  over the published choice. It is therefore not evidence *for* the paper's selection.
- The wider claim survives, as written above: it rests on σ being comparable to the published
  differences, not on the count being zero.

### Reported alongside, not instead of, the registered reading

Two observations that do **not** change the verdict and must not be used to upgrade it:

1. **The cell did not cross because *t* grew into it — mostly because the threshold fell.** At the
   old design (n = 5 vs 3) the same comparison stood at *t* = −2.05, *p* = 0.090, against a critical
   value of 2.776. Adding seeds took *t* to −2.45 and the correct critical value to 2.262.
2. **The gap reappears in the new seeds alone.** SCD mean, old seeds: nb2 1.3085 vs nb4 1.2607;
   Run 9 seeds only: nb2 1.3060 vs nb4 1.2525. An effect that is pure noise would not be expected to
   repeat its size in an independent sample — so this is the one cell in the table that deserves a
   targeted follow-up, **registered as its own hypothesis**, rather than a claim now.

A Bonferroni correction over 27 tests gives *p* × 27 = 0.94. That is **context, not the registered
rule** — the rule above is the expected-count reading, and it was chosen before the data.

### Correction to the text above (dated, not edited in place)

**2026-09-13.** OUTCOME 2 above says the withdrawal becomes *"withdrawn claim #9"*. Between writing
that and the result, claim #9 was assigned to a different withdrawal (§2's *"several of these are by
the authors of LRRNet"*, withdrawn 2026-09-09). This withdrawal is therefore **claim #10**. The
substance of the reading is unchanged.
