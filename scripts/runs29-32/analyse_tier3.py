"""Tier 3 ablations, scored against the Tier 2 seed-noise floor.

    A1b  k = 0        Top-K Selective Channel Alignment removed (external augmentation),
                      optimiser steps held equal to R1 so data volume is the only variable.
    A2   coeff_ssl=0  Weak-Aggressive self-supervision removed (internal augmentation).
                      Corresponds to Table VI row II, "w/o Internal-view DA".

Baseline is the **R1 configuration mean over seeds 42 and 43**, never a single run, and every delta
is quoted in units of the pooled seed sigma measured in Tier 2. A delta inside ~3 sigma is not
distinguishable from this training setup's run-to-run noise -- the lesson that forced the G10
retraction and the reason five seeds were spent before any ablation was run.

Caveat carried into the report: the sigma estimate comes from runs on the released 1,718-patch set.
Tier 2 established that seed noise is independent of *loss and schedule* configuration; it did not
establish independence from *dataset size*. The yardstick is therefore firm for A2 and approximate
for A1b, where only the large deltas should be treated as settled.
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
VARIANTS = ["A1b", "A2", "A3"]

RUNS = {
    "A1b":     os.path.join(T3, "results_A1b", "tables_A1b_raw"),
    "A2":      os.path.join(T3, "results_A2", "tables_A2_raw"),
    "A3":      os.path.join(T3, "results_A3", "tables_A3_raw"),
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

# Paper Table VI row II ("w/o Internal-view DA") minus the full model, on TNO -- the authors'
# own reported ablation effect, for direct comparison with what A2 measures.
PAPER_ROW_II_TNO = {"EN": +0.01, "MI": -0.05, "SCD": -0.01, "VIF": -0.02, "Qabf": -0.01, "SSIM": -0.03}


def load(run, ds):
    with open(os.path.join(RUNS[run], f"{ds}_gray_per_image.csv"), newline="",
              encoding="utf-8") as f:
        return {r["image"]: {m: float(r[m]) for m in METRICS} for r in csv.DictReader(f)}


def bar(u):
    return "**REAL**" if u >= 3 else ("marginal" if u >= 2 else "noise")


summary = {v: [] for v in VARIANTS}

print("=" * 104)
print("TIER 3 ABLATIONS vs. the R1 configuration (seeds 42+43), in units of Tier 2 pooled seed sigma")
print("=" * 104)
print("""
!! The sigma used here was measured on the RELEASED 1,718-patch set. It is valid for A2 and A3,
!! which train on that set and change only the loss or the architecture. It is NOT valid for the
!! A1 family, which halves the training data: measured k=0 noise is a median 3.35x larger
!! (22/27 cells). The A1b column below therefore OVERSTATES significance -- read A1 results from
!! analyse_a1_seed.py, which uses the three-seed k=0 sigma. Keeping the column here only so the
!! variants sit side by side.""")

for ds, n in DATASETS:
    data = {r: load(r, ds) for r in RUNS}
    keys = sorted(set.intersection(*[set(d) for d in data.values()]))
    assert len(keys) == n, f"{ds}: {len(keys)} common images, expected {n}"
    col = {r: {m: np.array([data[r][k][m] for k in keys]) for m in METRICS} for r in RUNS}

    print(f"\n\n### {ds}  (N={len(keys)})\n")
    head = f"{'metric':6} {'shipped':>9} {'R1cfg':>9} |"
    for v in VARIANTS:
        head += f" {v:>9} {'delta':>9} {'sig':>6} {'verdict':>9} |"
    print(head)
    print("-" * len(head))

    for m in METRICS:
        r1cfg = (col["R1"][m].mean() + col["R1b"][m].mean()) / 2
        sig = SIGMA[ds][m]
        line = f"{m:6} {col['shipped'][m].mean():9.4f} {r1cfg:9.4f} |"
        for v in VARIANTS:
            val = col[v][m].mean()
            d = val - r1cfg
            u = abs(d) / sig
            summary[v].append((ds, m, d, u))
            line += f" {val:9.4f} {d:+9.4f} {u:6.1f} {bar(u):>9} |"
        print(line)

    for v in VARIANTS:
        print(f"\n  Wilcoxon signed-rank, {v} vs each R1-config seed (per-image, N={len(keys)}):")
        print(f"  {'metric':6} {'p vs R1':>11} {'win%':>7} {'p vs R1b':>11} {'win%':>7}   consistent?")
        for m in METRICS:
            out, sigs = [], []
            for base in ("R1", "R1b"):
                a, b = col[v][m], col[base][m]
                p = wilcoxon(a, b).pvalue if not np.allclose(a, b) else 1.0
                w = 100.0 * (a > b).mean()
                out += [p, w]
                sigs.append(p < 0.05 and (w > 50))
            # A result that flips sign depending on which baseline seed it is compared against is
            # exactly the artefact that produced the withdrawn G10 claim.
            agree = "yes" if sigs[0] == sigs[1] else "** DISAGREES **"
            print(f"  {m:6} {out[0]:11.3g} {out[1]:6.1f}% {out[2]:11.3g} {out[3]:6.1f}%   {agree}")

print("\n\n" + "=" * 104)
print("SUMMARY")
print("=" * 104)

for v in VARIANTS:
    real = [s for s in summary[v] if s[3] >= 3]
    print(f"\n--- {v}: {len(real)} of {len(summary[v])} metric/dataset cells beyond 3 sigma ---")
    if not real:
        print("    Nothing beyond seed noise. The removed component has no measurable effect here.")
    for ds, m, d, u in sorted(real, key=lambda s: -s[3]):
        print(f"    {ds:10} {m:6} {d:+9.4f}  {u:6.1f} sigma   "
              f"{'variant HIGHER' if d > 0 else 'variant lower'}")

# Does A2 reproduce the paper's own Table VI row II?
print("\n\n" + "=" * 104)
print("A2 vs. the paper's own Table VI row II (TNO) -- does their reported ablation reproduce?")
print("=" * 104)
data = {r: load(r, "TNO") for r in RUNS}
keys = sorted(set.intersection(*[set(d) for d in data.values()]))
print(f"\n{'metric':6} {'paper delta':>12} {'our delta':>12} {'our sigma':>10} {'sig units':>10}  agreement")
print("-" * 104)
for m, pd_ in PAPER_ROW_II_TNO.items():
    a = np.array([data["A2"][k][m] for k in keys]).mean()
    r1cfg = (np.array([data["R1"][k][m] for k in keys]).mean()
             + np.array([data["R1b"][k][m] for k in keys]).mean()) / 2
    od = a - r1cfg
    sig = SIGMA["TNO"][m]
    same_sign = (pd_ > 0) == (od > 0)
    note = ("same direction" if same_sign else "OPPOSITE direction")
    if abs(od) / sig < 3:
        note += ", but ours is within noise"
    print(f"{m:6} {pd_:+12.3f} {od:+12.4f} {sig:10.4f} {abs(od)/sig:10.1f}  {note}")
