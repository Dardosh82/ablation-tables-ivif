"""Every sigma-scaled figure of case study I, recomputed with sigma_8 (Run 31) in place of sigma_3.

    python recompute_with_sigma8.py > ../results_run31/SIGMA8_RECOMPUTE.txt

Each Tier 3 analysis script that holds a SIGMA block is run twice, unchanged except for that block -- once with the
sample SD of the three original seeds (the published basis) and once with the sample SD of all eight -- and every
output line that differs is printed side by side. Then the Tier 2 deficits (MI and VIF of retraining vs the released
checkpoint) and the capacity decomposition, also as a t-test at the estimate's own degrees of freedom (df 2 -> 7).
"""
import glob
import math
import os
import statistics as st
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "Tier3_CrossFuse", "scripts"))
import analyse_run31 as a31  # noqa: E402

old = {r: {ds: a31.table(os.path.join(a31.T2, x, y, f"{ds}_gray_per_image.csv")) for ds in a31.DS}
       for r, (x, y) in a31.OLD.items()}
new = {s: {ds: a31.table(glob.glob(os.path.join(a31.R31, "cf-run31-*", f"tables_R3s{s}_raw",
                                                f"{ds}_gray_per_image.csv"))[0]) for ds in a31.DS} for s in a31.NEW}
allr = list(old.values()) + list(new.values())
S3 = {ds: {m: st.stdev([x[ds][m] for x in old.values()]) for m in a31.M} for ds in a31.DS}
S8 = {ds: {m: st.stdev([x[ds][m] for x in allr]) for m in a31.M} for ds in a31.DS}

import contextlib, io  # noqa: E401,E402
with contextlib.redirect_stdout(io.StringIO()):
    import audit_sigma_estimator as aud  # noqa: E402  (prints its own audit on import; silenced)

print("=" * 100)
print("A. Tier 3 scripts: output lines that change when sigma_3 is replaced by sigma_8")
print("=" * 100)
for fn in ("analyse_tier3.py", "analyse_a4.py", "analyse_a3_a5.py", "analyse_A1b.py"):
    o3, o8 = aud.run_with(fn, S3).splitlines(), aud.run_with(fn, S8).splitlines()
    print(f"\n--- {fn}: {len(o3)} lines, {sum(x != y for x, y in zip(o3, o8))} differ")
    for x, y in zip(o3, o8):
        if x != y:
            print(f"  s3: {x.rstrip()[:150]}\n  s8: {y.rstrip()[:150]}")

print("\n" + "=" * 100)
print("B. Tier 2 deficits of retraining vs the released checkpoint, |mean of 5 retrained - released| / sigma")
print("=" * 100)
ship = ("results_perimage", "tables_shipped_raw")
five = [("results_perimage", "tables_R1_raw"), ("results_R1b", "tables_R1b_raw"), ("results_R3", "tables_R3_raw"),
        ("results_R3b", "tables_R3b_raw"), ("results_R3c", "tables_R3c_raw")]
for ds in a31.DS:
    s = a31.table(os.path.join(a31.T2, *ship, f"{ds}_gray_per_image.csv"))
    f5 = [a31.table(os.path.join(a31.T2, x, y, f"{ds}_gray_per_image.csv")) for x, y in five]
    dev = {m: st.mean(t[m] for t in f5) - s[m] for m in a31.M}
    print(f"  {ds:9s} " + "  ".join(f"{m} {abs(dev[m]) / S3[ds][m]:.1f}->{abs(dev[m]) / S8[ds][m]:.1f}"
                                     for m in a31.M))
    print(f"  {'':9s} >= 3 sigma_3: {[m for m in a31.M if abs(dev[m]) / S3[ds][m] >= 3]}   "
          f">= 3 sigma_8: {[m for m in a31.M if abs(dev[m]) / S8[ds][m] >= 3]}")

print("\n" + "=" * 100)
print("C. capacity decomposition: 3-sigma rule and t-test at the estimate's df (sigma_3: df 2; sigma_8: df 7)")
print("=" * 100)
src = open(os.path.join(aud.HERE, "analyse_a3_a5.py"), encoding="utf-8").read()
ns = {}
exec(src[src.index("RUNS = {"):src.index("SIGMA = {")], {"os": os, "T3": aud.T3, "T2": aud.T2}, ns)
RUNS = ns["RUNS"]
for lab, S, df in (("sigma_3", S3, 2), ("sigma_8", S8, 7)):
    tc = stats.t.ppf(0.975, df)
    c = {k: 0 for k in ("total_rule", "arch_rule", "cap_rule", "total_t", "arch_t", "cap_t")}
    side = {"A5 (attention) higher": [], "A3 (deep CNN) higher": []}
    for ds in a31.DS:
        data = {r: aud.load(r, ds) for r in ("A3", "A5", "R1", "R1b")} if hasattr(aud, "load") else None
        keys = sorted(set.intersection(*(set(d) for d in data.values())))
        for m in a31.M:
            mean = {r: np.mean([data[r][k][m] for k in keys]) for r in data}
            r1 = (mean["R1"] + mean["R1b"]) / 2
            tot, arch, cap = mean["A3"] - r1, mean["A5"] - r1, mean["A5"] - mean["A3"]
            s = S[ds][m]
            c["total_rule"] += abs(tot) / s >= 3
            c["arch_rule"] += abs(arch) / s >= 3
            c["cap_rule"] += abs(cap) / s >= 3
            c["total_t"] += abs(tot) / (s * math.sqrt(1.5)) > tc
            c["arch_t"] += abs(arch) / (s * math.sqrt(1.5)) > tc
            c["cap_t"] += abs(cap) / (s * math.sqrt(2)) > tc
            if abs(cap) / s >= 3:
                side["A5 (attention) higher" if cap > 0 else "A3 (deep CNN) higher"].append(f"{ds}:{m}")
    print(f"  {lab}: rule total {c['total_rule']}, arch {c['arch_rule']}, cap {c['cap_rule']} of 27 | "
          f"t-test (t_crit {tc:.3f}) total {c['total_t']}, arch {c['arch_t']}, cap {c['cap_t']} of 27")
    for k, v in side.items():
        print(f"     capacity cells >= 3 sigma, {k}: {v}")

print("\n" + "=" * 100)
print("D. architecture term (A5 widened CNN vs R1/R1b attention) cells >= 3 sigma, by direction")
print("=" * 100)
for lab, S in (("sigma_3", S3), ("sigma_8", S8)):
    att, cnn = [], []
    for ds in a31.DS:
        data = {k: aud.load(k, ds) for k in ("A5", "R1", "R1b")}
        keys = sorted(set.intersection(*(set(d) for d in data.values())))
        for m in a31.M:
            mean = {k: np.mean([data[k][i][m] for i in keys]) for k in data}
            arch = mean["A5"] - (mean["R1"] + mean["R1b"]) / 2
            if abs(arch) / S[ds][m] >= 3:
                (cnn if arch > 0 else att).append(f"{ds}:{m}")
    print(f"  {lab}: attention higher {att}\n  {'':8} deep CNN higher {cnn}")

print("\n" + "=" * 100)
print("E. k = 0 spread (Run 11's registered rows, n = 5) against the released configuration (NOT registered:")
print("   a re-reading of Run 11's A-3 statistic with sigma_8, plus an F test per cell, Holm across 27)")
print("=" * 100)
with contextlib.redirect_stdout(io.StringIO()):
    import analyse_run11_k0 as k0  # noqa: E402
for lab, runs_rel in (("sigma_3", list(old.values())), ("sigma_8", allr)):
    ps, cells = [], []
    for ds, n in (("RoadScene", 40), ("TNO", 25), ("MSRS", 361)):
        means = k0.run_means(k0.ROWS[ds], ds, n)
        K = np.array([means[x] for x in k0.ROWS[ds]])
        for i, m in enumerate(a31.M):
            rel = np.array([x[ds][m] for x in runs_rel])
            F = K[:, i].var(ddof=1) / rel.var(ddof=1)
            ps.append(2 * min(stats.f.sf(F, 4, len(rel) - 1), stats.f.cdf(F, 4, len(rel) - 1)))
            cells.append((ds, m, round(F ** 0.5, 2)))
    ps = np.array(ps)
    surv = []
    for rank, i in enumerate(np.argsort(ps)):
        if ps[i] > 0.05 / (27 - rank):
            break
        surv.append(cells[i])
    print(f"  {lab}: median sd ratio {np.median([c[2] for c in cells]):.2f}, larger in "
          f"{sum(c[2] > 1 for c in cells)} of 27; F test surviving Holm: {surv}")

print("\n" + "=" * 100)
print("F. Table VI rows I and II (the authors' printed values, TNO) under the two-decimal rounding rule")
print("   noise: whole range < 3 sigma; robust: low end >= 3 sigma; else not resolvable at two decimals")
print("=" * 100)
FULL = {"EN": 7.21, "MI": 2.13, "SCD": 1.70, "VIF": 0.77, "Qabf": 0.54, "SSIM": 0.99}
ROWS = {"I  w/o Top-K": {"EN": 7.20, "MI": 2.19, "SCD": 1.67, "VIF": 0.77, "Qabf": 0.55, "SSIM": 0.99},
        "II w/o L_ssl": {"EN": 7.22, "MI": 2.08, "SCD": 1.69, "VIF": 0.75, "Qabf": 0.54, "SSIM": 0.96}}
for row, abl in ROWS.items():
    for lab, S in (("sigma_3", S3["TNO"]), ("sigma_8", S8["TNO"])):
        cells = []
        for m in FULL:
            d = FULL[m] - abl[m]
            pt, lo, hi = abs(d) / S[m], max(0.0, abs(d) - 0.01) / S[m], (abs(d) + 0.01) / S[m]
            v = "noise" if hi < 3 else ((("full" if d > 0 else "ablated") + " robust") if lo >= 3 else "unresolvable")
            cells.append(f"{m} {pt:.1f} ({lo:.1f}-{hi:.1f}) {v}")
        print(f"  row {row} {lab}: " + "; ".join(cells))
