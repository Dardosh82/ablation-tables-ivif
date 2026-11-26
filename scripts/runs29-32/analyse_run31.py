"""Run 31 -- case study I's sigma from three seeds to eight.  Readings: ../RUN31_PREREGISTRATION.md (23b2d9b),
committed alone before any kernel existed.

    python analyse_run31.py

Order, fixed: (0) kill criteria -- five new seeds, 160 epochs each, every score finite; if not, nothing is read.
(1) guard -- sigma_3 (sample SD of R3, R3b, R3c) must reproduce every published sigma_3 figure (Row I table, VIF
deficits 6.0 / 12.4 / 7.7, capacity 14 of 27 against 8 for the total effect); if not, nothing is read.
(2) R1 consistency, (3) R2 sigma_8 with its chi-square 95 % interval, (4) R3 Row I in sigma_8, (5) R4 the VIF
deficits and the capacity decomposition in sigma_8. Definitions are those of audit_sigma_estimator.py (sections 3, 4)
and analyse_tablevi_direct.py, unchanged except for which sigma divides.
"""
import csv
import glob
import math
import os
import statistics as st
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
T2 = os.path.dirname(HERE)
T3 = os.path.join(os.path.dirname(T2), "Tier3_CrossFuse")
R31 = os.path.join(T2, "results_run31")
M = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
DS = ["RoadScene", "TNO", "MSRS"]
NPAIRS = {"RoadScene": 40, "TNO": 25, "MSRS": 361}
OLD = {"R3": ("results_R3", "tables_R3_raw"), "R3b": ("results_R3b", "tables_R3b_raw"),
       "R3c": ("results_R3c", "tables_R3c_raw")}
NEW = [45, 46, 47, 48, 49]
# analyse_tablevi_direct.py, Row I (the authors' printed values, TNO)
FULL = {"EN": 7.21, "MI": 2.13, "SCD": 1.70, "VIF": 0.77, "Qabf": 0.54, "SSIM": 0.99}
ROW_I = {"EN": 7.20, "MI": 2.19, "SCD": 1.67, "VIF": 0.77, "Qabf": 0.55, "SSIM": 0.99}
ROW_I_PUBLISHED = {"EN": (0.6, 0.0, 1.2), "SCD": (0.8, 0.5, 1.1), "SSIM": (0.0, 0.0, 1.4), "VIF": (0.0, 0.0, 6.3),
                   "Qabf": (6.3, 0.0, 12.5), "MI": (11.5, 9.6, 13.5)}
VIF_PUBLISHED = {"RoadScene": 6.0, "TNO": 12.4, "MSRS": 7.7}
COUNTS_PUBLISHED = {"total": 8, "cap": 14}


def table(path, n=None):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    assert n is None or len(rows) == n, f"{path}: {len(rows)} rows, expected {n}"
    v = {m: [float(r[m]) for r in rows] for m in M}
    assert all(math.isfinite(x) for m in M for x in v[m]), f"{path}: non-finite score -- kill criterion"
    return {m: st.mean(v[m]) for m in M}


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, df, 2 * stats.t.sf(abs(t), df)


def holm(ps):
    order, adj, running = np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def row_i(sigma):
    """(point, low, high) in sigma and the published verdict rule: noise if the whole rounding range is below 3,
    robust if its low end is at least 3, else not resolvable at two decimals."""
    out = {}
    for m in FULL:
        d = FULL[m] - ROW_I[m]
        pt, lo, hi = abs(d) / sigma[m], max(0.0, abs(d) - 0.01) / sigma[m], (abs(d) + 0.01) / sigma[m]
        v = "noise" if hi < 3 else ("robust, " + ("full wins" if d > 0 else "ablated wins") if lo >= 3
                                    else "not resolvable at 2 d.p.")
        out[m] = (pt, lo, hi, v)
    return out


def vif_deficits(sigma):
    ship = ("results_perimage", "tables_shipped_raw")
    five = [("results_perimage", "tables_R1_raw"), ("results_R1b", "tables_R1b_raw"), ("results_R3", "tables_R3_raw"),
            ("results_R3b", "tables_R3b_raw"), ("results_R3c", "tables_R3c_raw")]
    out = {}
    for ds in DS:
        s = table(os.path.join(T2, *ship, f"{ds}_gray_per_image.csv"))
        f5 = [table(os.path.join(T2, a, b, f"{ds}_gray_per_image.csv")) for a, b in five]
        out[ds] = abs(st.mean(t["VIF"] for t in f5) - s["VIF"]) / sigma[ds]["VIF"]
    return out


def capacity_counts(sigma):
    src = open(os.path.join(T3, "scripts", "analyse_a3_a5.py"), encoding="utf-8").read()
    ns = {}
    exec(src[src.index("RUNS = {"):src.index("SIGMA = {")], {"os": os, "T3": T3, "T2": T2}, ns)
    runs = ns["RUNS"]

    def load(run, ds):
        with open(os.path.join(runs[run], f"{ds}_gray_per_image.csv"), newline="", encoding="utf-8") as f:
            return {r["image"]: {m: float(r[m]) for m in M} for r in csv.DictReader(f)}
    c = {"total": 0, "arch": 0, "cap": 0}
    for ds in DS:
        data = {r: load(r, ds) for r in ("A3", "A5", "R1", "R1b")}
        keys = sorted(set.intersection(*(set(d) for d in data.values())))
        for m in M:
            mean = {r: np.mean([data[r][k][m] for k in keys]) for r in data}
            r1 = (mean["R1"] + mean["R1b"]) / 2
            c["total"] += abs(mean["A3"] - r1) / sigma[ds][m] >= 3
            c["arch"] += abs(mean["A5"] - r1) / sigma[ds][m] >= 3
            c["cap"] += abs(mean["A5"] - mean["A3"]) / sigma[ds][m] >= 3
    return c


def main():
    L = ["Run 31 -- case study I's sigma from three seeds (R3, R3b, R3c) to eight (+ seeds 45-49)", ""]
    # (0) kill criteria
    new = {}
    for s in NEW:
        loss = glob.glob(os.path.join(R31, "cf-run31-*", f"loss_R3s{s}.csv"))
        tbl = glob.glob(os.path.join(R31, "cf-run31-*", f"tables_R3s{s}_raw"))
        if not loss or not tbl:
            continue
        ep = len(list(csv.DictReader(open(loss[0], encoding="utf-8"))))
        assert ep == 160, f"seed {s}: {ep} epochs -- kill criterion"
        new[s] = {ds: table(os.path.join(tbl[0], f"{ds}_gray_per_image.csv"), NPAIRS[ds]) for ds in DS}
    L.append(f"kill criteria: new seeds scored {sorted(new)} (160 epochs each, every score finite)")
    if len(new) < 5:
        L.append("INCOMPLETE -- fewer than five new seeds; nothing is read (registered)")
        print("\n".join(L))
        return 1
    old = {r: {ds: table(os.path.join(T2, a, b, f"{ds}_gray_per_image.csv"), NPAIRS[ds]) for ds in DS}
           for r, (a, b) in OLD.items()}

    # (1) guard: sigma_3 reproduces the published figures
    s3 = {ds: {m: st.stdev([old[r][ds][m] for r in OLD]) for m in M} for ds in DS}
    ri3, vd3, cc3 = row_i(s3["TNO"]), vif_deficits(s3), capacity_counts(s3)
    # the published Row I table divided by sigma_3 rounded to four decimals (SIGMA_TNO of analyse_tablevi_direct.py);
    # unrounded, Q^AB/F is 6.1 rather than 6.3 and MI 11.6 rather than 11.5 -- no verdict changes
    ri3r = row_i({m: round(v, 4) for m, v in s3["TNO"].items()})
    bad = [m for m in ROW_I_PUBLISHED
           if any(abs(x - y) > 0.051 for x, y in zip(ri3r[m][:3], ROW_I_PUBLISHED[m]))]
    bad += [ds for ds in DS if abs(vd3[ds] - VIF_PUBLISHED[ds]) > 0.05]
    bad += [k for k in COUNTS_PUBLISHED if cc3[k] != COUNTS_PUBLISHED[k]]
    L.append("guard: sigma_3 reproduces the published Row I table, VIF deficits and capacity counts"
             if not bad else f"guard FAILED on {bad} -- nothing is read")
    if bad:
        print("\n".join(L))
        return 1

    # (2) R1 consistency, per test set
    L += ["", "R1  five new seeds vs R3-R3c, Welch exact df, Holm across nine metrics"]
    drift = False
    for ds in DS:
        res = [welch([new[s][ds][m] for s in NEW], [old[r][ds][m] for r in OLD]) for m in M]
        adj = holm(np.array([p for _, _, p in res]))
        hits = [f"{m} (new {np.mean([new[s][ds][m] for s in NEW]):.4f} vs old "
                f"{np.mean([old[r][ds][m] for r in OLD]):.4f}, Holm p {h:.3g})" for m, h in zip(M, adj) if h < 0.05]
        L.append(f"  {ds:9s} surviving Holm: " + (", ".join(hits) if hits else "none"))
        if ds == "TNO" and hits:
            drift = True
    L.append("  -> " + ("DRIFT (on TNO): sigma is also reported from the five new seeds alone" if drift
                        else "CONSISTENT (nothing survives on TNO)"))

    # (3) R2 sigma_8
    allr = [old[r] for r in OLD] + [new[s] for s in NEW]
    s8 = {ds: {m: st.stdev([x[ds][m] for x in allr]) for m in M} for ds in DS}
    s5 = {ds: {m: st.stdev([new[s][ds][m] for s in NEW]) for m in M} for ds in DS}
    lo_f, hi_f = math.sqrt(7 / stats.chi2.ppf(0.975, 7)), math.sqrt(7 / stats.chi2.ppf(0.025, 7))
    L += ["", f"R2  sigma_8 (sample SD of eight seeds), 95 % interval x[{lo_f:.2f}, {hi_f:.2f}]",
          f"  {'':9s} " + " ".join(f"{m:>9s}" for m in M)]
    for ds in DS:
        L.append(f"  {ds:9s} sigma_3 " + " ".join(f"{s3[ds][m]:>9.4f}" for m in M))
        L.append(f"  {'':9s} sigma_8 " + " ".join(f"{s8[ds][m]:>9.4f}" for m in M))
        L.append(f"  {'':9s} ratio   " + " ".join(f"{s8[ds][m] / s3[ds][m]:>9.2f}" for m in M))
        if drift:
            L.append(f"  {'':9s} sigma_5n" + " ".join(f"{s5[ds][m]:>9.4f}" for m in M))

    # (4) R3 Row I
    sets = [("sigma_8", s8)] + ([("sigma_5 new only", s5)] if drift else [])
    for lab, sg in sets:
        ri = row_i(sg["TNO"])
        L += ["", f"R3  Row I in {lab} (TNO)   [sigma_3 published value in brackets]"]
        for m in ROW_I_PUBLISHED:
            pt, lo, hi, v = ri[m]
            L.append(f"  {m:5s} {pt:5.1f}  range {lo:4.1f}-{hi:4.1f}  {v:28s} [{ri3[m][0]:.1f}, {ri3[m][3]}]")
        pt, lo, _, _ = ri["MI"]
        L.append("  -> MI: " + ("STANDS" if lo >= 3 else ("NARROWED" if pt >= 3 else "WITHDRAWN")))

    # (5) R4
    for lab, sg in sets:
        vd, cc = vif_deficits(sg), capacity_counts(sg)
        L += ["", f"R4  in {lab}",
              "  VIF deficits of retraining: " + ", ".join(f"{ds} {vd[ds]:.1f} (was {vd3[ds]:.1f})" for ds in DS),
              f"  capacity term >= 3 sigma in {cc['cap']} of 27 cells; total effect {cc['total']}; "
              f"architecture term {cc['arch']}   (sigma_3: {cc3['cap']}, {cc3['total']}, {cc3['arch']})",
              "  -> " + ("STANDS (capacity term exceeds 3 sigma in more cells than the effect)"
                         if cc["cap"] > cc["total"] else "WITHDRAWN")]
    text = "\n".join(L)
    print(text)
    open(os.path.join(R31, "RUN31_ANALYSIS.txt"), "w", encoding="utf-8").write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
