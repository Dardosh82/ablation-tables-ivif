# Person-contrast follow-up: pre-registration (seeds 43 and 44)

**Written 2026-10-03, committed on its own before any seed-43 or seed-44 crop is measured.** It needs no training
and no GPU. The fused TNO images already exist in the kernel outputs of every arm, for every seed. Only pair 02 is
downloaded.

## Why

Run 23's rule 5 (the person check) was registered on **seed 42 only**. It found:
- the unchanged tied depth-8 network (A0) at 0.80 of the published configuration's person contrast;
- the candidate B1 at 0.72, which failed rule 5 (≥ 0.9).

The paper would report the A0 figure as a cost of tied d8 on TNO. One seed cannot carry that sentence.

## What is measured (identical to `person_check_run18.py`)

- **Image:** TNO pair 02, region (24, 359, 372, 480).
- **Person mask:** infrared pixels above the box's 90th percentile.
- **Contrast:** the mean fused intensity on the mask minus the mean on the rest of the box.
- **Ratio:** an arm's contrast divided by the published configuration's (`lrrnet-stage1`, tag `nb4_seed<s>`) **at
  the same seed**.
- **Seeds:** 43 and 44, added to seed 42's figures already recorded.
- **Arms:** A0 (`lrrnet-m1-nb8`, tied d8), A1–A4 and B1.

## Reading, fixed now

**For A0, the claim the paper would make:**

| outcome | condition | reported as |
|---|---|---|
| **C1: CONFIRMED** | A0's ratio is < 0.9 at seeds 43 **and** 44 | *tied depth 8 lowers the infrared target's contrast on TNO pair 02 at all three seeds (ratios given)* |
| **C2: MIXED** | < 0.9 at exactly one of 43 and 44 | the three ratios, with no general claim |
| **C3: NOT CONFIRMED** | ≥ 0.9 at both | the seed-42 figure was not representative, and the paper does not state the cost |

**B1 and A1–A4:** their three ratios are reported descriptively. **Run 23's outcome (NO CANDIDATE) does not change**
whatever these seeds show, because rule 5 was registered on seed 42 and was applied as registered.

**Limit, stated now:** this is one image pair. The result is described as such, and never as "on TNO".
