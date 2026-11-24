"""M2: do asymmetric refiners do anything, and is the full method better than the published one?

Two questions, two baselines, and they are NOT the same test.

  M1+M2 vs M1        did the refiners earn their 29,190 parameters? Both sides are weight-tied and
                     start from IDENTICAL weights at a given seed, so this comparison is PAIRED.
  M1+M2 vs published is the full method better than LRRNet at EXACTLY equal parameters (49,204)?
                     Different architectures, unpaired.

WHY THE FIRST ONE IS PAIRED, and why that matters. M2's refiners are residual with zero-initialised
output projections, and they are constructed LAST so module creation does not shift the global RNG
stream for any other layer. At seed 42 the M1 run and the M1+M2 run therefore begin from the same
weights, in the same data order, and this is visible in the logs: initial training loss 296,780 vs
296,583 (0.07% apart). Treating them as independent samples would throw that away and bury a small
real effect under seed noise the two runs SHARE. The paired test removes it.

The paired test is the primary one for M1+M2 vs M1. It is also weak at n=3 (df=2, two-sided 5%
critical value 4.303), which is stated rather than glossed.

Usage:
    python analyse_m2.py --csv /path/to/lrrnet_metrics.csv
"""
import argparse
import csv
import pathlib

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrrnet_data import arm                                     # noqa: E402

METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
SEEDS = [42, 43, 44]

# Run 1: M1 tied, num_block=4, seeds 42/43/44 -- 20,014 parameters
M1 = arm("lrrnet-m1-tied", "m1tied_nb4_", seeds=SEEDS)
# Stage 1: published LRRNet, untied, num_block=4, five seeds -- 49,204 parameters
PUB = list(arm("lrrnet-stage1", "nb4_", seeds=[42, 43, 44, 45, 46]).values())

TCRIT_WELCH = 2.776     # two-sided 5%, Welch-Satterthwaite df ~ 4
TCRIT_PAIRED = 4.303    # two-sided 5%, df = 2 (three pairs)


def unpaired(a, b):
    na, nb = len(a), len(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    d = b.mean() - a.mean()
    return d, abs(d) / pooled, d / np.sqrt(sa**2 / na + sb**2 / nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--prefix", default="m1m2_")
    a = ap.parse_args()

    rows = {int(r["tag"].split("seed")[-1]): [float(r[m]) for m in METRICS]
            for r in csv.DictReader(open(pathlib.Path(a.csv), encoding="utf-8-sig"))
            if r["tag"].startswith(a.prefix)}
    missing = [s for s in SEEDS if s not in rows]
    if missing:
        raise SystemExit(f"missing seeds {missing} in {a.csv}")

    X = np.array([rows[s] for s in SEEDS])          # M1+M2
    A = np.array([M1[s] for s in SEEDS])            # M1, same seed order
    P = np.array(PUB)

    print("=" * 92)
    print("M2 -- asymmetric refiners, capacity-matched to the published network by construction")
    print("=" * 92)
    print(f"  M1       20,014 par   n={len(A)}")
    print(f"  M1 + M2  49,204 par   n={len(X)}      (refiners: 29,190 = exactly what M1 freed)")
    print(f"  LRRNet   49,204 par   n={len(P)}")

    # ---- 1. PAIRED: M1+M2 vs M1 --------------------------------------------------------------
    print(f"\n{'=' * 92}")
    print("1. M1+M2 vs M1 -- PAIRED (identical initial weights and data order at each seed)")
    print("=" * 92)
    print(f"{'metric':7}{'seed42':>11}{'seed43':>11}{'seed44':>11}{'mean d':>11}"
          f"{'sd(d)':>10}{'paired t':>11}")
    print("-" * 92)
    n_paired = 0
    consistent = 0
    for i, m in enumerate(METRICS):
        d = X[:, i] - A[:, i]
        sd = d.std(ddof=1)
        t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else 0.0
        sig = abs(t) > TCRIT_PAIRED
        n_paired += sig
        consistent += abs(np.sign(d).sum()) == len(d)      # all three the same direction
        print(f"{m:7}" + "".join(f"{v:>+11.4f}" for v in d)
              + f"{d.mean():>+11.4f}{sd:>10.4f}{t:>11.2f}{'  <5%' if sig else '   ns'}")
    print(f"  --> {n_paired} of 9 reach the 5% level (paired, critical value {TCRIT_PAIRED})")
    print(f"  --> {consistent} of 9 move the same direction in all three seeds")

    # ---- 2. UNPAIRED: M1+M2 vs the published network ------------------------------------------
    print(f"\n{'=' * 92}")
    print("2. M1+M2 vs published LRRNet -- at EXACTLY equal parameters, unpaired")
    print("=" * 92)
    print(f"{'metric':7}{'LRRNet (n=5)':>22}{'M1+M2 (n=3)':>22}{'diff':>10}{'pooled s':>10}{'t':>9}")
    print("-" * 92)
    n_pub = 0
    for i, m in enumerate(METRICS):
        d, s, t = unpaired(P[:, i], X[:, i])
        sig = abs(t) > TCRIT_WELCH
        n_pub += sig
        print(f"{m:7}{P[:, i].mean():>13.4f} +-{P[:, i].std(ddof=1):<7.4f}"
              f"{X[:, i].mean():>13.4f} +-{X[:, i].std(ddof=1):<7.4f}"
              f"{d:>+10.4f}{s:>9.2f}s{t:>9.2f}{'  <5%' if sig else '   ns'}")
    print(f"  --> {n_pub} of 9 reach the 5% level")

    # ---- verdict -------------------------------------------------------------------------------
    print(f"""
{'=' * 92}
RESULT
{'=' * 92}
  M1+M2 vs M1        (paired)   : {n_paired} of 9
  M1+M2 vs published (unpaired) : {n_pub} of 9

READING (pre-registered in 02_DESIGN.md and lrrfuse_m2.py, before this run existed)

  The hypothesis was SPECIFIC, not "something will move": chapter 4 measured that attention buys
  perceptual fidelity (VIF, Qabf, MI) and a CNN buys contrast and sharpness (EN, SD, SF, AG). A
  genuine transfer to LRRNet's low-rank/sparse split should appear on the perceptual metrics, with
  several agreeing. One metric out of nine is not a result -- that is the rule this project applies
  to LRRNet's own table, and it applies here first.""")

    if n_paired == 0:
        print("""
  M2 DID NOT EARN ITS PARAMETERS. Reported as the pre-registered failure, with BOTH candidate
  explanations, because this run cannot separate them:

    (a) the chapter-4 block assignment does not transfer to a low-rank/sparse split. That split is
        named after a decomposition the network never actually enforces -- L and S are two channel
        slices of one tensor, with no rank constraint anywhere in the forward pass -- so there may
        be no low/high-frequency structure there for an asymmetric assignment to exploit. If so,
        the negative result is itself evidence about what LRRNet's decomposition really is.

    (b) d=16 is too narrow for attention over 128 channels. The width was forced: demanding an
        exact parameter match leaves exactly one solution in the search range, and it spends 80% of
        the budget on the CNN.

  What must NOT happen next: widening d until the result improves. That would tune against the very
  budget that makes this comparison capacity-matched, and would convert a clean negative into an
  unfalsifiable positive. If (b) is to be tested, it is tested as a SEPARATE, declared experiment
  with its own parameter accounting -- and it stops being a capacity-matched comparison.""")
    else:
        print(f"""
  {n_paired} metric(s) moved. Before this is called a transfer, check WHICH: the pre-registered
  prediction is VIF, Qabf and MI together. Scattered hits on unrelated metrics are what nine
  correlated tests produce by chance, and would not support the hypothesis.""")

    print("""
  Independent of either verdict, and worth reporting on its own: M2 costs ~17% more training time
  at IDENTICAL parameter count (~77 min vs ~66 min per seed). Matching parameters is not matching
  compute. A paper claiming "same number of parameters" while omitting this is hiding a real cost.""")


if __name__ == "__main__":
    main()
