"""The depth question, asked properly: a 2x2 of unrolling depth against weight tying.

LRRNet's Table IV is presented as a depth ablation. It is not one -- untied, going from 4 blocks to
8 also adds 38,920 parameters, so depth and capacity move together and neither can be attributed
the difference (finding L1).

Weight tying (M1) makes depth free: 20,014 parameters at either depth. Crossing the two factors
gives the design the published table needed and never had:

                       depth 4              depth 8
    untied      49,204 (stage 1, n=5)   88,124 (stage 3, n=3)
    tied        20,014 (run 1,  n=3)    20,014 (run 2,  n=3)

    tied@8  - tied@4    = the DEPTH effect at constant capacity   <- the clean question
    untied@8- untied@4  = what the published table actually measured
    tied@4  - untied@4  = the CAPACITY effect at constant depth
    tied@8  - untied@8  = the same, at the other depth

All rows: our training, our evaluator, the same 25 TNO pairs, the same pinned container.
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
TCRIT = 2.776          # two-sided 5%, Welch-Satterthwaite df ~ 4

CELLS = {
    ("untied", 4): list(arm("lrrnet-stage1", "nb4_", seeds=[42, 43, 44, 45, 46]).values()),  # stage 1
    ("untied", 8): list(arm("lrrnet-stage3", "nb8_", seeds=[42, 43, 44]).values()),          # stage 3
    ("tied", 4): list(arm("lrrnet-m1-tied", "m1tied_nb4_", seeds=[42, 43, 44]).values()),    # run 1
}
PARAMS = {("untied", 4): 49_204, ("untied", 8): 88_124,
          ("tied", 4): 20_014, ("tied", 8): 20_014}


def welch(a, b):
    """diff (b - a), difference in pooled sigma, Welch t."""
    na, nb = len(a), len(b)
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    d = b.mean() - a.mean()
    return d, abs(d) / pooled, d / np.sqrt(sa**2 / na + sb**2 / nb)


def contrast(M, key_a, key_b, title, note):
    a_, b_ = M[key_a], M[key_b]
    print(f"\n{title}")
    print(f"  {note}")
    print(f"  {'metric':7}{'diff':>12}{'pooled s':>11}{'Welch t':>10}")
    print("  " + "-" * 40)
    n_sig = 0
    for i, m in enumerate(METRICS):
        d, s, t = welch(a_[:, i], b_[:, i])
        sig = abs(t) > TCRIT
        n_sig += sig
        print(f"  {m:7}{d:>+12.4f}{s:>10.2f}s{t:>10.2f}{'  <5%' if sig else '   ns'}")
    print(f"  --> {n_sig} of 9 reach the 5% level")
    return n_sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="lrrnet_metrics.csv from the nb=8 tied run")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(pathlib.Path(a.csv), encoding="utf-8-sig"))
            if r["tag"].startswith("m1tied_nb8_")]
    if not rows:
        raise SystemExit(f"no rows tagged m1tied_nb8_ in {a.csv}")
    CELLS[("tied", 8)] = [[float(r[m]) for m in METRICS] for r in rows]

    M = {k: np.array(v) for k, v in CELLS.items()}

    print("=" * 92)
    print("DEPTH x TYING -- the 2x2 LRRNet's Table IV needed and never had")
    print("=" * 92)
    print(f"{'':10}{'depth 4':>28}{'depth 8':>28}")
    for tie in ("untied", "tied"):
        cells = [f"{PARAMS[(tie, d)]:,} par (n={len(M[(tie, d)])})" for d in (4, 8)]
        print(f"{tie:10}" + "".join(f"{c:>28}" for c in cells))

    print(f"\n{'metric':7}" + "".join(f"{f'{t} d{d}':>20}" for t, d in
                                      (("untied", 4), ("untied", 8), ("tied", 4), ("tied", 8))))
    print("-" * 92)
    for i, m in enumerate(METRICS):
        cells = [f"{M[k][:, i].mean():.4f} ±{M[k][:, i].std(ddof=1):.4f}"
                 for k in (("untied", 4), ("untied", 8), ("tied", 4), ("tied", 8))]
        print(f"{m:7}" + "".join(f"{c:>20}" for c in cells))

    print(f"\n{'=' * 92}\nTHE FOUR CONTRASTS\n{'=' * 92}")
    depth_clean = contrast(M, ("tied", 4), ("tied", 8),
                           "1. DEPTH, capacity held constant BY CONSTRUCTION  (tied 4 -> tied 8)",
                           "20,014 parameters on both sides. The clean question, asked for the first time.")
    depth_conf = contrast(M, ("untied", 4), ("untied", 8),
                          "2. DEPTH as the published table measures it  (untied 4 -> untied 8)",
                          "+38,920 parameters come along for the ride. Depth and capacity confounded.")
    cap4 = contrast(M, ("untied", 4), ("tied", 4),
                    "3. CAPACITY at depth 4  (untied -> tied)",
                    "-29,190 parameters, depth unchanged.")
    cap8 = contrast(M, ("untied", 8), ("tied", 8),
                    "4. CAPACITY at depth 8  (untied -> tied)",
                    "-68,110 parameters, depth unchanged. The largest single cut in this project.")

    total = depth_clean + depth_conf + cap4 + cap8
    print(f"""
{'=' * 92}
RESULT
{'=' * 92}
  depth, clean (tied 4 vs 8)      : {depth_clean} of 9
  depth, confounded (untied 4 vs 8): {depth_conf} of 9
  capacity at depth 4             : {cap4} of 9
  capacity at depth 8             : {cap8} of 9
  ACROSS THE WHOLE 2x2            : {total} of 36

  Expected purely by chance at the 5% level across 36 tests: {36 * 0.05:.1f}.

INTERPRETATION

  Four configurations spanning {PARAMS[('untied', 8)] / PARAMS[('tied', 4)]:.2f}x in model size ({PARAMS[('tied', 4)]:,} to {PARAMS[('untied', 8)]:,}
  parameters) and 2x in unrolled depth. Every row trained here, judged against sigma measured on
  these same runs.

  {total} of 36 comparisons reach the 5% level, against {36 * 0.05:.1f} expected from chance alone with no
  correction for multiplicity. A count at or below the chance rate is not evidence of an effect,
  and the single cell involved is not treated as one. Applying the rule this project applies to
  other people's tables: a hit among 36 correlated tests at the chance rate is what noise looks like.
  (Corrected 2026-09-13: TCRIT = 2.776 is the df = 4 value and is conservative for five seeds against
  three. At exact df a second cell crosses -- SSIM, untied -> tied at d4, p = 0.038 -- giving 2 of 36,
  both favouring tied, in different metrics. The registered verdict is the 2.776 one; see 05_RESULTS.md.)

  This is the experiment LRRNet's Table IV was presented as being. Run properly, with capacity
  held constant by construction rather than by hope, the answer is that unrolling depth does not
  measurably matter on this architecture and training budget -- and neither does capacity, across
  a {PARAMS[('untied', 8)] / PARAMS[('tied', 4)]:.2f}x range.

  The published table therefore selected 4 blocks on the basis of a comparison that could not
  have resolved the question either way, and then fixed that setting for every experiment in the
  paper.

  IN THE PAPER'S FAVOUR, stated as plainly as the criticism: with everything indistinguishable,
  the smallest model is the right engineering choice. Our tied model at depth 4 is smaller still
  -- {PARAMS[('untied', 4)] / PARAMS[('tied', 4)]:.2f}x fewer parameters than the published configuration -- and equally
  indistinguishable, which is the entire efficiency claim of chapter 6.""")


if __name__ == "__main__":
    main()
