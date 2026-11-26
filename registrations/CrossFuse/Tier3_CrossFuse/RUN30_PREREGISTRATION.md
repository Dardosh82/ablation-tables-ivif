# Run 30 — pre-registration · does removing the decoder's skip remove the MI/VIF inversion?

**Written 2026-11-25, committed on its own before any output of the run exists.** Nothing below may change after the
first fused image is written.

## The question

On CrossFuse, MI and VIF rank an untrained network above a trained one (Section 7). The account in the paper is
mechanistic: the decoder adds the visible input back before its output (`sigmoid(output(features) + inp_img)`), so an
untrained network returns a near-copy of the visible image (r = 0.996–0.998), and both measures reward that copy.
Two observations support it across lineages (the inversion appears in CrossFuse and CDDFuse, which have the skip,
and not in SeAFusion, which has none), but different architectures differ in more than the skip. This run removes
the skip **and nothing else**, within one architecture and one set of weights.

## What is run

CrossFuse's decoder already takes the skip as an argument: `forward(inp_img, base, detail)` adds `inp_img` only when
it is not `None`. `run_inference.py` gains a `--no-skip` flag that passes `None` there; nothing else changes.

| arm | weights | decoder skip | source |
|---|---|---|---|
| **U-ON** | untrained, seeds 42–46 (`make_untrained.py`, zero gradient steps) | on | recorded (U42–U46) |
| **U-OFF** | **the same five untrained networks, rebuilt from the same seeds** | **off** | new |
| **T** | trained, the five runs of the positive control (A1b, A1c, A1d, PC45, PC46) | on | recorded |

Test set: TNO, the same 25 pairs; the same evaluator (CrossFuse's `Evaluator`, nine metrics) and image pipeline.
**Integrity check:** in the same kernel, U-ON is rebuilt and re-scored; its nine TNO means must reproduce the
recorded U42–U46 means to four decimals, or nothing is read. Cost: inference only, minutes of GPU.

A trained no-skip network is **not** part of this run: removing the skip from a network trained with it would test
a broken model, and training one without it is a separate question (whether the skip helps a trained network), not
this one.

## The readings, fixed before the run

Statistics as in the paper: Welch's *t* at each comparison's own degrees of freedom, Holm across the nine metrics,
five against five.

**P1 — the copy.** Mean correlation of each U-OFF output with the visible input, against 0.996–0.998 for U-ON.
- **COPY GONE**: below 0.9 for every seed. **COPY STAYS**: 0.9 or above for any seed.

**P2 — the inversion (the registered test).** MI and VIF, T against U-OFF.
- **CAUSAL**: on both MI and VIF the trained network now scores higher than U-OFF, and the difference survives Holm.
  The paper may say that removing the skip, with nothing else changed, removes the inversion, so the skip produces it.
- **PARTIAL**: one of the two reverses after Holm, or both reverse without surviving Holm. Reported per metric; the
  paper says the skip contributes to the inversion, not that it alone produces it.
- **NOT THE SKIP**: on both, U-OFF still scores at or above the trained network. The mechanism in the paper is
  withdrawn as the explanation, and the paper says so in Section 7 and the limitations.

**Also reported, descriptively, no verdict:** the other seven metrics, T against U-OFF and U-ON against U-OFF, and
the hierarchical bootstrap of Run 29 applied to T against U-OFF.

**Kill criteria:** the integrity check above; any non-finite score (a pipeline failure, never evidence); fewer than
five U-OFF seeds scored.
