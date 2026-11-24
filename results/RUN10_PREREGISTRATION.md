# Run 10 — pre-registration

**Written 2026-09-13, committed on its own before any Run 10 kernel exists.** Nothing below may be
edited after the first result; corrections go in a dated block at the end.

---

## Why this run exists

Run 9 left two questions open, and the paper states both as open.

1. **One cell crossed.** SCD, 4 blocks against 2, reached *p* = 0.035 uncorrected at eight seeds per
   row (Welch *t* = −2.452), in favour of 2 blocks — the setting the published table ranks worst. It
   does not survive correction for 27 tests, but its gap repeated in Run 9's five new seeds alone
   (1.306 against 1.253, as in the old seeds' 1.309 against 1.261). A cell that repeats its size in a
   fresh sample deserves a test of its own, not an adjective.
2. **Four cells are inconclusive.** At the registered design resolution of ±1.75 pooled σ, TOST found
   23 of 27 differences equivalent and four neither equivalent nor different: 4 vs 2 on SF, AG and SCD,
   and 4 vs 6 on SF.

## What is run

The configuration of Runs 1–9, unchanged: 4 epochs, batch 8, 128×128 grayscale, LLVIP, CrossFuse's
`Evaluator`, the same 25 TNO pairs. Only new seeds are added, **50 to 61**, none used before.

| setting | new seeds | runs | h/run (Run 9 logs) | GPU-h | when |
|---|---|---|---|---|---|
| `nb = 2` | 50–61 | 12 | 0.79 | 9.5 | this week |
| `nb = 4` | 50–61 | 12 | 1.09 | 13.1 | this week, two kernels of six |
| `nb = 6` | 50–61 | 12 | 1.46 | 17.5 | next week, two kernels of six |

`nb = 8` is not extended: no cell against it is inconclusive.

---

## The readings, fixed before the run

### A. Confirmatory — the SCD cell

**One pre-specified test**, so no multiplicity correction applies to it: SCD, `nb = 2` against
`nb = 4`, **on the 24 new runs only** (12 against 12), Welch's *t*, two-sided, 5 %. The old seeds are
excluded from this test by design — they produced the hypothesis and cannot also confirm it.

- **A1 — REPLICATED:** *p* < 0.05 with `nb = 2` higher. Reported as: a difference on one metric of
  nine, found at eight seeds, survived an independent registered replication. It is **not** reported
  as "2 blocks is better": eight other metrics show no such difference. It is reported as evidence that
  the published ordering is not merely unreproduced but, on SCD, reversed.
- **A2 — NOT REPLICATED:** *p* ≥ 0.05, or `nb = 4` higher. Reported as: Run 9's cell was consistent
  with chance, as the correction for 27 tests already indicated; the repetition in five seeds was not
  enough to make it real.

**Power, stated so a null is not over-read.** Run 9's effect, 1.23 pooled σ, gives 82 % power at
12 against 12. That estimate comes from the cell selected for being largest and is inflated by that
selection; at a true effect of 0.8 σ the power is 47 %. A2 therefore does not show the effect
is zero, only that it did not replicate at this size.

### B. Secondary — the four inconclusive cells

For each of 4 vs 2 on SF, AG, SCD and 4 vs 6 on SF: TOST at the same **±1.75 pooled σ** margin Run 9
used, 5 %, on **all seeds pooled** (8 + 12 = 20 per arm), pooled σ recomputed at n = 20. Each cell is
read as **equivalent**, **different** (a two-sided Welch test at 5 % rejects), or still **inconclusive**.
A cell where both reject is read as both — **a real difference smaller than the design resolution** — and
is reported that way, not rounded to either word.
The 4 vs 6 cell is read only after `nb = 6` completes; until then it stays inconclusive and is said to.

The full table at n = 20 (settings 2, 4, 6 against 4) is reported descriptively with Holm correction
across its 18 comparisons. It is not a new headline and does not replace Run 9's count.

---

## Kill criteria

Any of these voids the affected setting:

1. **Fewer than 12 completed new seeds** for a setting.
2. **A new run's final training loss outside the existing seeds' range by more than 3× that range's
   width** — the operational reading `check_run9_losses.py` used.
3. **Any non-finite metric.**
4. **Two runs with identical metric vectors** to within 1e-12.
5. **A new seed not in 50–61, or a seed appearing twice.**

---

## What this run is not

It is not a search for significance: the confirmatory test is one cell, fixed now, on data that do not
yet exist. It is not a new experiment: nothing but the seeds changes. The analysis is
`scripts/analyse_run10.py`, committed alongside this file.
