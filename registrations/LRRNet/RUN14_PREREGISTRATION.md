# Run 14 — pre-registration · free depth: the tied network at depth 8, from three seeds to five

**Written 2026-09-25, committed on its own before any Run 14 kernel exists.** Nothing below may be
edited after the first result; corrections go in a dated block at the end.

---

## Why this run exists

Weight tying makes depth **free**: the tied network has **20,014 parameters at 2, 4, 6 and 8 blocks
alike**. That makes possible the one ablation LRRNet's own table was presented as being and could not
be — *more iterations of the solver, with nothing else changing*. In the published (untied) network,
going from 4 to 8 blocks also adds 38,920 parameters, so depth and size can never be separated there.

The paper already reports this contrast inside a 2×2 (untied/tied × depth 4/8), but **the tied depth-8
arm stands at three seeds** (42–44, `lrrnet-m1-nb8`), while the tied depth-4 arm now stands at five
(M1 + Run 12). This run takes tied depth 8 to five seeds so the capacity-controlled depth contrast is
measured 5 against 5, like everything else, and at each comparison's own degrees of freedom.

**It is run at the published budget (4 epochs), deliberately.** Run 13 is still measuring whether a
longer budget helps; choosing the budget for this run from Run 13's result would couple the two
experiments, and choosing it before that result is known is only clean at the budget every existing arm
of the 2×2 already uses. If Run 13 finds headroom, depth at the longer budget is a separately
registered run.

## What is run

The `lrrnet-m1-nb8` kernel, unchanged except its seed list: `num_block = 8`, `LRR_TIED = 1`,
`epochs = 4`, batch 8, LLVIP, CrossFuse's `Evaluator`, the same 25 TNO pairs.

| arm | seeds already run | new seeds | runs | h/run (est.) | GPU-h |
|---|---|---|---|---|---|
| tied, depth 8 | 42, 43, 44 | **45, 46** | 2 | ~1.8 | ~3.6 |

---

## The readings, fixed before the run

**The primary contrast: tied depth 8 (n = 5) against tied depth 4 (n = 5, M1 + Run 12).** Same
parameter count by construction, so this is depth and nothing else. Welch's *t*, two-sided 5 %, at each
comparison's own degrees of freedom; Holm across the nine; TOST at ±1.75 pooled σ for every cell that
does not cross. σ is the sample SD (ddof = 1).

- **E1 — nothing crosses.** *Doubling the number of solver iterations at constant size changes no
  metric the design can resolve.* Equivalence is claimed only for the metrics TOST supports, by name;
  the rest are reported as inconclusive at five seeds.
- **E2 — one metric crosses, none survives Holm.** Reported as one cell of nine at the chance rate, the
  metric named; not a finding in either direction. The same rule that refused SSIM in Run 12.
- **E3 — at least one metric survives Holm, favouring depth 8.** *Depth helps at constant size* — a
  design result that the untied network cannot produce, because there depth drags size with it. Reported
  with the metrics named and the 2.46× parameter figure unchanged.
- **E4 — at least one metric survives Holm, favouring depth 4.** *More iterations hurt at constant size*
  — reported as a limit of the unrolled design, named in the chapter.

**Secondary, descriptive:** the full 2×2 at five seeds on the tied side (untied depth 8 stands at eight
seeds from Run 9), with the counts stated as they were in the paper — the registered reading and the
exact-df reading side by side.

**Power:** at 5 against 5 the design resolves about 2.0 pooled σ at 80 %. E1 means no effect of that
size was found.

## Kill criteria — before any statistic

1. `[CONFIG]` on both runs reads `num_block=8`, `epochs=4`, `tied=1`.
2. Parameter count 20,014.
3. 25 TNO pairs scored per run, the same image list as seeds 42–44.
4. The two new seeds differ on every metric; each writes its own checkpoint.
5. **An out-of-memory stop is a failed run, not a result.** The untied depth-8 setting exhausted a 16 GB
   card once; tying shares weights but not activations, so the risk is the same, and a run that stops
   early is never scored.

## Cost

2 runs ≈ **3.6 GPU-h**.
