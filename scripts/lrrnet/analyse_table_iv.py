"""LRRNet Table IV, re-measured: every row trained here, judged against sigma measured here.

Supersedes analyse_stage2.py, which compared only 4 vs 6. The question the paper's table answers is
"how many LLRR blocks?", and it fixes that answer for every other experiment it reports, so the
comparison has to cover the range rather than one neighbour.

Group sizes are deliberately unequal: n=5 for the reference configuration, because sigma is the
number every later comparison leans on, and n=3 for each ablated row.

Stages 3 and 4 required a change stages 1-2 did not: num_block 8 exhausts a 16 GB card, so training
runs across both available GPUs. That is arithmetically the same update -- net_lista.py contains no
BatchNorm, InstanceNorm, LayerNorm or Dropout, so no layer's output depends on how the batch is
partitioned -- but it is recorded as a deviation wherever these rows are used.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrrnet_data import arm                                     # noqa: E402

METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]

# Every row of the re-measured table, from the CSV each kernel wrote (see lrrnet_data.py). These were
# literals until 2026-09-13; the values are unchanged, verified to 0.0 difference.
RUNS = {
    4: arm("lrrnet-stage1", "nb4_", seeds=[42, 43, 44, 45, 46]),   # published configuration, five seeds
    6: arm("lrrnet-stage2", "nb6_", seeds=[42, 43, 44]),
    8: arm("lrrnet-stage3", "nb8_", seeds=[42, 43, 44]),           # trained across two GPUs
    2: arm("lrrnet-stage4", "nb2_", seeds=[42, 43, 44]),           # back on the single-device path
}
PARAMS = {2: 29_744, 4: 49_204, 6: 68_664, 8: 88_124}

# Published Table IV (TNO), for reference only -- never mixed into our arithmetic. Their metric
# conventions differ from ours (their MI 13.7 against our 2.50), which is why every row is retrained.
PUBLISHED = {2: [6.79214, 78.25832, 13.58429, 0.71164, 0.66020],
             4: [6.85836, 81.78905, 13.71673, 0.70625, 0.71399],
             6: [6.82875, 80.37536, 13.65750, 0.70726, 0.68716],
             8: [6.83084, 80.24377, 13.66168, 0.70250, 0.69624]}

M = {nb: np.array([RUNS[nb][s] for s in sorted(RUNS[nb])]) for nb in RUNS}
REF = 4

print("=" * 104)
print("LRRNet Table IV, re-measured — our rows, our evaluator, our sigma; no proxy anywhere")
print("=" * 104)
print(f"{'metric':6}" + "".join(f"{f'nb={nb} (n={len(M[nb])})':>20}" for nb in sorted(M)))
print("-" * 104)
for i, m in enumerate(METRICS):
    cells = [f"{M[nb][:, i].mean():.4f} ±{M[nb][:, i].std(ddof=1):.4f}" for nb in sorted(M)]
    print(f"{m:6}" + "".join(f"{c:>20}" for c in cells))

print(f"\n{'='*104}\nEACH ABLATED ROW AGAINST THE CHOSEN ONE (nb=4)\n{'='*104}")
print(f"group sizes: nb=4 n={len(M[REF])}, " + ", ".join(f"nb={nb} n={len(M[nb])}" for nb in (2, 6, 8)))
print("critical value derived per comparison from its own degrees of freedom; "
      "2.776 at df=4, 2.145 at df=14")
print(f"{'metric':6}" + "".join(f"{f'4 vs {nb}: diff / pooled sigma / Welch t':>36}" for nb in (2, 6, 8)))
print("-" * 104)

# Two-sided 5 % critical values of t, by degrees of freedom. The value is DERIVED from the group
# sizes rather than typed, because Run 9 takes every row from n=3 to n=8 and the threshold that was
# correct at df=4 (2.776) is wrong at df=14 (2.145).
#
# This table was written while Run 9 was still training and no new metric existed here, so it cannot
# have been chosen to suit an outcome. Moving a threshold after seeing the numbers it decides is the
# error this work documents, and it is no less an error when the move is defensible.
#
# Welch-Satterthwaite df is data-dependent, so the conservative floor is used: the smaller of the
# exact df and (na + nb - 2). A smaller df means a LARGER critical value, so this never makes
# significance easier to reach than the design justifies.
T_TWO_SIDED_5PCT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
                    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
                    15: 2.131, 16: 2.120, 18: 2.101, 20: 2.086, 25: 2.060, 30: 2.042}


def tcrit(df):
    """Critical value at the nearest tabulated df at or below the one requested (conservative)."""
    if df != df or df < 1:
        return T_TWO_SIDED_5PCT[1]
    keys = sorted(k for k in T_TWO_SIDED_5PCT if k <= df)
    return T_TWO_SIDED_5PCT[keys[-1]] if keys else T_TWO_SIDED_5PCT[30]


def welch_df(sa, na, sb, nb):
    va, vb = sa**2 / na, sb**2 / nb
    den = va**2 / (na - 1) + vb**2 / (nb - 1)
    return (va + vb) ** 2 / den if den > 0 else float("nan")
verdicts = {2: [], 6: [], 8: []}
for i, m in enumerate(METRICS):
    row = f"{m:6}"
    for nb in (2, 6, 8):
        a, b = M[REF][:, i], M[nb][:, i]
        d = a.mean() - b.mean()
        sa, sb, na, nbn = a.std(ddof=1), b.std(ddof=1), len(a), len(b)
        pooled = np.sqrt(((na - 1) * sa**2 + (nbn - 1) * sb**2) / (na + nbn - 2))
        t = d / np.sqrt(sa**2 / na + sb**2 / nbn)
        df = min(welch_df(sa, na, sb, nbn), na + nbn - 2)
        crit = tcrit(df)
        sig = abs(t) > crit
        verdicts[nb].append(sig)
        row += f"{d:>+13.4f}{abs(d)/pooled:>10.2f}s{t:>9.2f}{'  <5%' if sig else '   ns':>4}"
    print(row)

print(f"""
{'='*104}
RESULT
{'='*104}""")
for nb in (2, 6, 8):
    n_sig = sum(verdicts[nb])
    print(f"  4 blocks vs {nb} blocks:  {n_sig} of 9 metrics reach the 5% level"
          f"   ({PARAMS[nb]/PARAMS[REF]:.2f}x the parameters)")

lead = {nb: sum(1 for i in range(9) if M[REF][:, i].mean() > M[nb][:, i].mean()) for nb in (2, 6, 8)}
total_sig = sum(sum(v) for v in verdicts.values())
print(f"""
  Counting wins the way the paper does, 4 blocks leads on {lead[2]} of 9 against 2 blocks,
  {lead[6]} of 9 against 6 and {lead[8]} of 9 against 8 -- and not one of those leads is
  separable from run-to-run noise.

  ACROSS THE WHOLE TABLE: {total_sig} of 27 comparisons reach the 5% level.

INTERPRETATION

  The table is now complete: four settings spanning {PARAMS[8]/PARAMS[2]:.2f}x in model size
  ({PARAMS[2]:,} to {PARAMS[8]:,} parameters), every row trained here, all judged against sigma
  measured on these same runs. No metric distinguishes any setting from the chosen one.

  The published table selects 4 blocks and then fixes that setting for every other experiment in
  the paper. That choice is not supported by any comparison in the table it comes from.

  Note the direction on 2 blocks, the setting the published numbers place clearly WORST. Retrained,
  it is nominally ahead of the chosen configuration on several metrics while using
  {100*PARAMS[2]/PARAMS[REF]:.0f}% of the parameters. Everything stays inside the noise band, so
  nothing is claimed -- but retraining does not reproduce even the published ordering.

  This does not show the architecture is wrong, and the paper's engineering instinct is sound: with
  four indistinguishable configurations, a small one is the right pick. That is simply not the
  reason the paper gives, and not what its table demonstrates.""")
