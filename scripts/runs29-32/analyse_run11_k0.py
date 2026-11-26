"""Run 11, arm A: the Top-K-removed (k = 0) arm at five seeds, read against the R1 configuration.

Readings: ../RUN11_PREREGISTRATION.md (committed 9c96a53) and its dated correction (c5779df), both before any
arm A statistic was computed. Applied as written:

  * Rows, fixed in the correction: TNO uses A1b, A1c, A1d, PC45, PC46 (the arm as it already stood -- PC45 and
    PC46 are k = 0 runs, scored on TNO only); RoadScene and MSRS use A1b, A1c, A1d, A1e, A1f. Every TNO reading
    is also printed with A1e/A1f in place of PC45/PC46, and a verdict that changes between the two is unstable.
  * Baseline: the R1 configuration (R1 and R1b, optimiser steps equal to the k = 0 arm), as in
    analyse_a1_seed.py. It has TWO runs, so every Welch df here is small and the test is conservative; that is
    reported, not repaired.
  * Welch's t, two-sided 5 %, at each comparison's own df; Holm across the nine metrics of each test set.
  * sigma is the SAMPLE standard deviation everywhere (withdrawn claim #13).

    python analyse_run11_k0.py
"""
import csv
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
T3 = os.path.dirname(HERE)
T2 = os.path.join(T3, "..", "Tier2_CrossFuse")
METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
DATASETS = [("RoadScene", 40), ("TNO", 25), ("MSRS", 361)]
ALPHA = 0.05
TABLES = {
    "A1b": os.path.join(T3, "results_A1b", "tables_A1b_raw"),
    "A1c": os.path.join(T3, "results_A1c", "tables_A1c_raw"),
    "A1d": os.path.join(T3, "results_A1d", "tables_A1d_raw"),
    "A1e": os.path.join(T3, "results_run11_k0", "tables_A1e_raw"),
    "A1f": os.path.join(T3, "results_run11_k0", "tables_A1f_raw"),
    "PC45": os.path.join(T3, "results_halfbudget_cf", "tables_PC45_raw"),
    "PC46": os.path.join(T3, "results_halfbudget_cf", "tables_PC46_raw"),
    "R1": os.path.join(T2, "results_perimage", "tables_R1_raw"),
    "R1b": os.path.join(T2, "results_R1b", "tables_R1b_raw"),
}
# Tier 2's released-data sigma (sample SD of three R3-configuration seeds), as in analyse_a1_seed.py
SIGMA_RELEASED = {
    "RoadScene": dict(zip(METRICS, [0.0229, 0.0570, 0.6674, 0.1287, 0.0484, 0.0034, 0.0644, 0.0022, 0.0087])),
    "TNO":       dict(zip(METRICS, [0.0166, 0.0052, 0.3627, 0.2866, 0.0793, 0.0016, 0.0374, 0.0016, 0.0074])),
    "MSRS":      dict(zip(METRICS, [0.0443, 0.0976, 0.8294, 0.1172, 0.0543, 0.0094, 0.0757, 0.0045, 0.0262])),
}
ROWS = {"RoadScene": ["A1b", "A1c", "A1d", "A1e", "A1f"],
        "TNO": ["A1b", "A1c", "A1d", "PC45", "PC46"],
        "MSRS": ["A1b", "A1c", "A1d", "A1e", "A1f"]}
TNO_ALT = ["A1b", "A1c", "A1d", "A1e", "A1f"]


def load(run, ds):
    path = os.path.join(TABLES[run], f"{ds}_gray_per_image.csv")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["image"]: {m: float(r[m]) for m in METRICS} for r in csv.DictReader(f)}


def run_means(runs, ds, n):
    d = {r: load(r, ds) for r in runs}
    keys = sorted(set.intersection(*(set(v) for v in d.values())))
    assert len(keys) == n, f"{ds}: {len(keys)} common images across {runs}, expected {n}"
    return {r: np.array([np.mean([d[r][k][m] for k in keys]) for m in METRICS]) for r in runs}


def welch(x, y):
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    d = x.mean() - y.mean()
    se = np.sqrt(vx / nx + vy / ny)
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    t = d / se
    return d, t, df, 2 * stats.t.sf(abs(t), df)


def holm(p):
    order, out = np.argsort(p), np.zeros(len(p), dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= ALPHA / (len(p) - rank):
            out[i] = True
        else:
            break
    return out


def family(ds, n, k0_runs, label):
    means = run_means(k0_runs + ["R1", "R1b"], ds, n)
    K = np.array([means[r] for r in k0_runs])
    B = np.array([means["R1"], means["R1b"]])
    print(f"\n### {ds} ({n} images) -- k=0 {k0_runs} against R1 configuration [R1, R1b]  {label}")
    print(f"{'metric':6}{'k=0 mean':>11}{'sd':>8}{'R1cfg':>10}{'diff':>10}{'t':>8}{'df':>6}{'p':>9}  verdict")
    ps, rows = [], []
    for i, m in enumerate(METRICS):
        d, t, df, p = welch(K[:, i], B[:, i])
        ps.append(p)
        rows.append((m, K[:, i].mean(), K[:, i].std(ddof=1), B[:, i].mean(), d, t, df, p))
    surv = holm(np.array(ps))
    out = {}
    for (m, km, ks, bm, d, t, df, p), s in zip(rows, surv):
        v = "crosses" if p < ALPHA else "ns"
        print(f"{m:6}{km:11.4f}{ks:8.4f}{bm:10.4f}{d:+10.4f}{t:8.2f}{df:6.1f}{p:9.4f}  {v}{' (Holm)' if s else ''}"
              f"{'  k=0 higher' if p < ALPHA and d > 0 else '  k=0 lower' if p < ALPHA else ''}")
        out[m] = (d, p, s)
    ratios = [K[:, i].std(ddof=1) / SIGMA_RELEASED[ds][m] for i, m in enumerate(METRICS)]
    return out, ratios


print("=" * 100)
print("RUN 11 arm A -- Top-K removed (k = 0) at five seeds against the R1 configuration (n = 2)")
print("=" * 100)
res, ratios = {}, []
for ds, n in DATASETS:
    res[ds], r = family(ds, n, ROWS[ds], "(registered rows)")
    ratios += r
alt, _ = family("TNO", 25, TNO_ALT, "(TNO with A1e/A1f in place of PC45/PC46 -- stability check)")

print("\n" + "=" * 100 + "\nREADINGS (RUN11_PREREGISTRATION.md, arm A)\n" + "=" * 100)


def crossed(ds, m, sign):
    d, p, _ = res[ds][m]
    return p < ALPHA and np.sign(d) == sign


a1 = all(crossed(ds, m, +1) for ds in ("RoadScene", "TNO") for m in ("VIF", "MI"))
print(f"  A-1 removing Top-K improves VIF and MI on RoadScene and TNO: "
      f"{'HOLDS' if a1 else 'REVISED'} -- " + ", ".join(
          f"{ds} {m} p={res[ds][m][1]:.4f}{'+' if res[ds][m][0] > 0 else '-'}" for ds in ("RoadScene", "TNO") for m in ("VIF", "MI")))
a2_cells = [("TNO", "SCD"), ("TNO", "SSIM"), ("RoadScene", "SF")]
a2 = all(crossed(ds, m, -1) for ds, m in a2_cells)
print(f"  A-2 removing Top-K costs SCD and SSIM on TNO and SF on RoadScene: {'HOLDS' if a2 else 'REVISED'} -- "
      + ", ".join(f"{ds} {m} p={res[ds][m][1]:.4f}{'+' if res[ds][m][0] > 0 else '-'}" for ds, m in a2_cells))
med = float(np.median(ratios))
print(f"  A-3 k=0 markedly less stable: median sigma(k=0, n=5) / sigma(released) = {med:.2f} over {len(ratios)} cells, "
      f"{sum(r > 1 for r in ratios)} of {len(ratios)} larger -> {'HOLDS (> 2)' if med > 2 else 'REVISED (<= 2)'}")
n_msrs = sum(1 for m in METRICS if res["MSRS"][m][1] < ALPHA)
print(f"  A-4 on MSRS almost nothing is distinguishable: {n_msrs} of 9 cross -> {'HOLDS (<= 2)' if n_msrs <= 2 else 'REVISED'}")
unstable = [m for m in METRICS if (res["TNO"][m][1] < ALPHA) != (alt[m][1] < ALPHA)]
print(f"  TNO verdicts that change when A1e/A1f replace PC45/PC46: {unstable or 'none'}")

print("\n" + "=" * 100 + "\nSAME-SEED REPLICATE PAIRS on TNO (descriptive, never counted as seeds)\n" + "=" * 100)
m_tno = run_means(["A1e", "PC45", "A1f", "PC46", "A1b", "A1c", "A1d"], "TNO", 25)
sd_k0 = np.array([m_tno[r] for r in ["A1b", "A1c", "A1d", "PC45", "PC46"]]).std(axis=0, ddof=1)
for a, b in (("A1e", "PC45"), ("A1f", "PC46")):
    diff = np.abs(m_tno[a] - m_tno[b]) / sd_k0 * 100
    print(f"  {a} vs {b} (same seed, config, code): |diff| as % of k=0 sigma -- "
          + "  ".join(f"{m} {x:.0f}%" for m, x in zip(METRICS, diff)) + f"   max {diff.max():.0f}%")
