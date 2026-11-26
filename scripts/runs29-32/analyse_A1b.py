"""Tier 3, Ablation I: does removing Top-K Selective Channel Alignment change anything?

A1b is the released configuration trained on the RoadScene-only (k=0, 918-patch) set with the
optimiser-step count held equal to R1's, so the only variable is the presence of the 800 M3FD
Top-K patches.

The baseline is the **R1 configuration mean over seeds 42 and 43** -- not a single run -- and every
delta is quoted in units of the pooled seed sigma measured in Tier 2. A delta inside ~3 sigma is
not distinguishable from the run-to-run noise of this training setup, which is the whole reason
Tier 2 ran five seeds before Tier 3 ran anything.
"""

import csv
import os

import numpy as np
from scipy.stats import wilcoxon

# relative to this file, so the script runs wherever the tree is unpacked (an absolute path to one machine until
# 2026-11-26, found by an external audit of the release)
T3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T2 = os.path.join(os.path.dirname(T3), "Tier2_CrossFuse")

METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
DATASETS = [("RoadScene", 40), ("TNO", 25), ("MSRS", 361)]

RUNS = {
    "A1b":     os.path.join(T3, "results_A1b", "tables_A1b_raw"),
    "R1":      os.path.join(T2, "results_perimage", "tables_R1_raw"),
    "R1b":     os.path.join(T2, "results_R1b", "tables_R1b_raw"),
    "shipped": os.path.join(T2, "results_perimage", "tables_shipped_raw"),
}

# Pooled seed sigma from Tier 2 (R3 configuration, n=3) -- REPORT_TABLES.md Appendix B.
# Seed sigma: SAMPLE standard deviation (n-1) of the three R3-configuration seeds. Until 2026-09-13 this
# block held the POPULATION SD (statistics.pstdev, dividing by n), which with n = 3 is 0.816 of the
# sample SD and inflated every 'x sigma' by 1.225. Values computed from the per-image tables, not typed.
SIGMA = {
    "RoadScene": dict(zip(METRICS, [0.0229, 0.0570, 0.6674, 0.1287, 0.0484, 0.0034, 0.0644, 0.0022, 0.0087])),
    "TNO":       dict(zip(METRICS, [0.0166, 0.0052, 0.3627, 0.2866, 0.0793, 0.0016, 0.0374, 0.0016, 0.0074])),
    "MSRS":      dict(zip(METRICS, [0.0443, 0.0976, 0.8294, 0.1172, 0.0543, 0.0094, 0.0757, 0.0045, 0.0262])),
}


def load(run, ds):
    path = os.path.join(RUNS[run], f"{ds}_gray_per_image.csv")
    with open(path, newline="", encoding="utf-8") as f:
        return {r["image"]: {m: float(r[m]) for m in METRICS} for r in csv.DictReader(f)}


print("=" * 96)
print("TIER 3 -- ABLATION I : Top-K Selective Channel Alignment removed (k=0), steps held equal")
print("=" * 96)
print("baseline = R1 configuration, mean of seeds 42 and 43 (released 1718-patch data)")
print("sigma    = pooled seed sigma from Tier 2 (n=3)")

verdicts = []

for ds, n in DATASETS:
    data = {r: load(r, ds) for r in RUNS}
    keys = sorted(set.intersection(*[set(d) for d in data.values()]))
    assert len(keys) == n, f"{ds}: {len(keys)} common images, expected {n}"

    print(f"\n\n### {ds}  (N={len(keys)})\n")
    print(f"{'metric':6} {'shipped':>9} {'R1':>9} {'R1b':>9} {'R1cfg':>9} "
          f"{'A1b':>9} {'delta':>9} {'sigma':>8} {'|d|/sig':>8}  verdict")
    print("-" * 96)

    for m in METRICS:
        col = {r: np.array([data[r][k][m] for k in keys]) for r in RUNS}
        r1cfg = (col["R1"].mean() + col["R1b"].mean()) / 2
        a1b = col["A1b"].mean()
        d = a1b - r1cfg
        sig = SIGMA[ds][m]
        u = abs(d) / sig
        mark = "**REAL**" if u >= 3 else ("marginal" if u >= 2 else "noise")
        verdicts.append((ds, m, d, u, mark))
        print(f"{m:6} {col['shipped'].mean():9.4f} {col['R1'].mean():9.4f} "
              f"{col['R1b'].mean():9.4f} {r1cfg:9.4f} {a1b:9.4f} "
              f"{d:+9.4f} {sig:8.4f} {u:8.1f}  {mark}")

    # Paired test against each R1-configuration seed separately. Agreement between the two is
    # the check that a "significant" result is not an artefact of which seed we compared against
    # -- the mistake that produced the withdrawn G10 claim in Tier 2.
    print(f"\n  Wilcoxon signed-rank, A1b vs each R1-config seed (per-image, N={len(keys)}):")
    print(f"  {'metric':6} {'p vs R1':>12} {'win% vs R1':>12} {'p vs R1b':>12} {'win% vs R1b':>12}")
    for m in METRICS:
        out = []
        for base in ("R1", "R1b"):
            a = np.array([data["A1b"][k][m] for k in keys])
            b = np.array([data[base][k][m] for k in keys])
            p = wilcoxon(a, b).pvalue if not np.allclose(a, b) else 1.0
            out += [p, 100.0 * (a > b).mean()]
        print(f"  {m:6} {out[0]:12.3g} {out[1]:11.1f}% {out[2]:12.3g} {out[3]:11.1f}%")

print("\n\n" + "=" * 96)
print("SUMMARY")
print("=" * 96)
real = [v for v in verdicts if v[3] >= 3]
if not real:
    print("\nNo metric on any dataset moves beyond 3 sigma.")
    print("Removing Top-K Selective Channel Alignment -- the paper's headline contribution --")
    print("is NOT distinguishable from seed noise in this reproduction.")
else:
    print(f"\n{len(real)} of {len(verdicts)} metric/dataset cells exceed 3 sigma:\n")
    for ds, m, d, u, _ in sorted(real, key=lambda v: -v[3]):
        direction = "k=0 HIGHER" if d > 0 else "k=0 lower"
        print(f"  {ds:10} {m:6} {d:+8.4f}  = {u:5.1f} sigma   ({direction})")
    print(f"\n{len(verdicts)-len(real)} cells are within 3 sigma of the R1 configuration.")
