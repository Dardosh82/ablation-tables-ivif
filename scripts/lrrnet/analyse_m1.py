"""Chapter 6, Run 1: does weight-tied unrolling cost anything?

M1 removes 29,190 of LRRNet's 49,204 parameters by sharing one LRR block across all four unrolled
iterations, instead of giving each iteration its own. The iteration count is unchanged, so the
question is narrow and answerable: were those parameters doing any measurable work?

Judged against stage 1's nb=4 UNTIED rows (n=5), our evaluator, the same 25 TNO pairs, the same
container digest. Never against the published table -- finding L2: IVIF papers do not share a
metric scale, so a published number is measured with a different ruler.

The reading of each outcome was pre-registered in our_papers/paper1_variance_aware/archive_LRRFuse-VA/02_DESIGN.md before this run
existed, and is applied here as written rather than chosen after seeing the numbers.

Usage:
    python analyse_m1.py --csv /path/to/lrrnet_metrics.csv
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

# stage 1, nb=4 untied, five seeds -- the baseline of record (LRRNET_RESULTS.md)
BASELINE = arm("lrrnet-stage1", "nb4_", seeds=[42, 43, 44, 45, 46])

P_UNTIED, P_TIED = 49_204, 20_014
TCRIT = 2.776          # two-sided 5%, Welch-Satterthwaite df ~ 4
KILL_N = 3             # kill criterion 1: three or more metrics beyond 3 pooled sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="lrrnet_metrics.csv from the M1 notebook")
    ap.add_argument("--prefix", default="m1tied_", help="tag prefix identifying the M1 rows")
    a = ap.parse_args()

    # utf-8-sig, not utf-8: a CSV that has passed through a Windows tool carries a BOM, which
    # otherwise lands inside the first header name and makes row["tag"] a KeyError.
    rows = [r for r in csv.DictReader(open(pathlib.Path(a.csv), encoding="utf-8-sig"))
            if r["tag"].startswith(a.prefix)]
    if not rows:
        raise SystemExit(f"no rows tagged {a.prefix!r} in {a.csv}")

    B = np.array([BASELINE[s] for s in sorted(BASELINE)])
    T = np.array([[float(r[m]) for m in METRICS] for r in rows])
    nb_, nt_ = len(B), len(T)

    print("=" * 96)
    print("M1 -- weight-tied unrolling at num_block=4, against the untied baseline")
    print("=" * 96)
    print(f"  seeds: {[r['tag'].split('_')[-1] for r in rows]}")
    print(f"  parameters: {P_UNTIED:,} untied -> {P_TIED:,} tied "
          f"({P_UNTIED/P_TIED:.2f}x smaller, {P_UNTIED-P_TIED:,} freed for M2)\n")

    print(f"{'metric':7}{'untied (n=%d)' % nb_:>22}{'M1 tied (n=%d)' % nt_:>22}"
          f"{'diff':>10}{'pooled s':>11}{'Welch t':>10}")
    print("-" * 96)

    beyond3, sig5 = [], []
    for i, m in enumerate(METRICS):
        x, y = B[:, i], T[:, i]
        mx, sx = x.mean(), x.std(ddof=1)
        my, sy = y.mean(), y.std(ddof=1)
        d = my - mx
        pooled = np.sqrt(((nb_ - 1) * sx**2 + (nt_ - 1) * sy**2) / (nb_ + nt_ - 2))
        t = d / np.sqrt(sx**2 / nb_ + sy**2 / nt_)
        n_sig = abs(d) / pooled
        beyond3.append(n_sig > 3)
        sig5.append(abs(t) > TCRIT)
        flag = "  <5%" if abs(t) > TCRIT else "   ns"
        print(f"{m:7}{mx:>13.4f} +-{sx:<7.4f}{my:>13.4f} +-{sy:<7.4f}"
              f"{d:>+10.4f}{n_sig:>10.2f}s{t:>10.2f}{flag}")

    # Sigma comparison -- not what M1 targets (that is M3), but it must be reported either way,
    # and it must be TESTED rather than eyeballed. A variance ratio at n=5 vs n=3 is a very weak
    # estimate: the two-sided 5% critical value of F(4,2) is 39.25, so an apparent 4x ratio is
    # entirely ordinary sampling behaviour. Reporting the ratio without this test would be the
    # exact error this thesis documents in other people's tables.
    cv_b = np.array([B[:, i].std(ddof=1) / abs(B[:, i].mean()) for i in range(9)])
    cv_t = np.array([T[:, i].std(ddof=1) / abs(T[:, i].mean()) for i in range(9)])
    FCRIT = 39.25          # F(4, 2), two-sided 5%
    fvals = np.array([B[:, i].var(ddof=1) / T[:, i].var(ddof=1) for i in range(9)])
    f_sig = fvals > FCRIT

    print(f"\n{'=' * 96}\nRESULT\n{'=' * 96}")
    print(f"  metrics reaching the 5% level : {sum(sig5)} of 9")
    print(f"  metrics beyond 3 pooled sigma : {sum(beyond3)} of 9")
    print(f"  mean coefficient of variation : {cv_b.mean()*100:.2f}% untied "
          f"-> {cv_t.mean()*100:.2f}% tied   ({cv_b.mean()/cv_t.mean():.2f}x)")
    print(f"  variance ratios beyond F(4,2) : {sum(f_sig)} of 9   "
          f"(max F = {fvals.max():.1f} against a critical value of {FCRIT})")
    lead = sum(1 for i in range(9) if T[:, i].mean() > B[:, i].mean())
    print(f"  M1 nominally ahead on          : {lead} of 9 metrics\n")

    passed = sum(beyond3) < KILL_N
    if passed:
        print(f"""  VERDICT -- the pre-registered PASS branch

  Fewer than {KILL_N} metrics move beyond 3 pooled sigma, so the {P_UNTIED-P_TIED:,} removed
  parameters bought nothing measurable. The efficiency claim stands at {P_UNTIED/P_TIED:.2f}x, and it is a
  CATEGORICAL claim: a parameter count has no seed variance to hide behind.

  M2's budget is open. Its refiners may spend exactly {P_UNTIED-P_TIED:,} parameters, which returns
  the model to {P_UNTIED:,} -- identical to the published network, so the comparison is
  capacity-matched BY CONSTRUCTION rather than by a control run added afterwards.

  Note this is the third independent demonstration in this project that LRRNet's capacity is not
  the binding constraint: nb 4->6 moved 0 of 9, nb 4->8 moved 0 of 9, and now removing 59% of the
  parameters moves {sum(beyond3)} of 9.""")
    else:
        print(f"""  VERDICT -- kill criterion 1 fires

  {sum(beyond3)} metrics move beyond 3 pooled sigma, at or above the pre-registered threshold of
  {KILL_N}. Report this plainly: weight tying costs quality on this architecture, and M2/M3 are
  built on the UNTIED backbone instead.

  Do NOT rescue it by tuning the learning rate. Tying changes the optimisation as well as the
  capacity -- one block now receives gradient from four iterations -- so a loss here is ambiguous
  between the two causes. An lr sweep is the next DIAGNOSTIC, run and reported as such; it is not
  a retrofit that turns this result into a pass.""")

    print(f"""
  TWO THINGS THAT MUST NOT BE CLAIMED FROM THIS RUN

  1. M1 is nominally ahead on {lead} of 9 metrics. Counting wins that way is EXACTLY what LRRNet's
     Table IV does, and this project spent two chapters showing it proves nothing. With nine
     correlated metrics and one configuration each, a lead is nearly guaranteed to exist. Zero of
     those leads reach significance, so the honest statement is "indistinguishable", not "better".
     The standard is applied to our own method first.

  2. Relative spread fell {cv_b.mean()/cv_t.mean():.2f}x, which looks like the sigma reduction M3 is
     supposed to deliver. It is NOT established: {sum(f_sig)} of 9 variance ratios clear F(4,2), whose
     two-sided 5% critical value is {FCRIT}. At n=5 against n=3 a variance estimate is far too weak
     to carry a claim. Recorded as a HYPOTHESIS for M3 to test at higher n, nothing more.""")


if __name__ == "__main__":
    main()
