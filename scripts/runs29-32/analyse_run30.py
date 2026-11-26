"""Run 30 -- does removing the decoder's skip remove the MI/VIF inversion?  Readings: ../RUN30_PREREGISTRATION.md
(06bacff), committed before any output existed.

    python analyse_run30.py

P1 the copy: mean r(output, visible) per U-OFF seed; COPY GONE if below 0.9 for every seed, else COPY STAYS.
P2 the inversion (registered): MI and VIF, trained (A1b, A1c, A1d, PC45, PC46, recorded) against U-OFF, Welch at
exact df, Holm across nine metrics. CAUSAL if on both MI and VIF the trained network scores higher and survives Holm;
NOT THE SKIP if on both U-OFF still scores at or above trained; PARTIAL otherwise.
Integrity (checked in the kernel): U-ON rebuilt must reproduce the recorded U42-U46 means to four decimals.
"""
import csv
import glob
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
T3 = os.path.dirname(HERE)
R30 = os.path.join(T3, "results_run30")
M = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
TRAINED = ["A1b", "A1c", "A1d", "PC45", "PC46"]
SEEDS = ["U42", "U43", "U44", "U45", "U46"]


def run_means(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    assert len(rows) == 25, f"{path}: {len(rows)} rows"
    v = np.array([[float(r[m]) for m in M] for r in rows])
    assert np.isfinite(v).all(), f"{path}: non-finite score -- a pipeline failure, never evidence"
    return v.mean(axis=0)


def welch(a, b):
    ma, mb, va, vb = a.mean(), b.mean(), a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (ma - mb) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, df, 2 * stats.t.sf(abs(t), df)


def holm(ps):
    order, adj, running = np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def main():
    lines = ["Run 30 -- the decoder's skip removed from the same five untrained CrossFuse networks", ""]
    integ = open(os.path.join(R30, "integrity.txt"), encoding="utf-8").read().strip().splitlines()
    ok = len(integ) == 5 and all(l.endswith(": yes") for l in integ)
    lines += ["integrity: " + ("U-ON reproduces the record for all five seeds" if ok else "FAILED -- nothing is read")]
    if not ok:
        lines += integ
        print("\n".join(lines))
        return
    T = np.array([run_means(os.path.join(T3, "results_poscontrol_cf", f"tables_{t}_raw", "TNO_gray_per_image.csv"))
                  for t in TRAINED])
    ON = np.array([run_means(os.path.join(R30, "tables", f"{s}_ON", "TNO_gray_per_image.csv")) for s in SEEDS])
    OFF = np.array([run_means(os.path.join(R30, "tables", f"{s}_OFF", "TNO_gray_per_image.csv")) for s in SEEDS])

    # P1 -- the copy
    rv = {}
    for r in csv.DictReader(open(os.path.join(R30, "resemblance.csv"), encoding="utf-8")):
        rv.setdefault((r["tag"], r["skip"]), []).append(float(r["r_visible"]))
    lines += ["", "P1  mean r(output, visible), 25 TNO pairs"]
    off_r = []
    for s in SEEDS:
        on_m, off_m = np.nanmean(rv[(s, "ON")]), np.nanmean(rv[(s, "OFF")])
        off_r.append(off_m)
        lines.append(f"  {s}  skip ON {on_m:.4f}   skip OFF {off_m:.4f}")
    p1 = "COPY GONE" if all(r < 0.9 for r in off_r) else "COPY STAYS"
    lines.append(f"  -> {p1}")

    # P2 -- the inversion, trained against U-OFF; also U-ON for reference
    res = [welch(T[:, j], OFF[:, j]) for j in range(len(M))]
    adj = holm(np.array([r[2] for r in res]))
    lines += ["", "P2  trained (5) vs untrained with the skip OFF (5), TNO; Welch exact df, Holm across nine",
              f"  {'metric':6} {'trained':>10} {'U-ON':>10} {'U-OFF':>10} {'t':>8} {'df':>6} {'p':>9} {'Holm p':>9}"]
    for j, m in enumerate(M):
        t, df, p = res[j]
        lines.append(f"  {m:6} {T[:, j].mean():>10.4f} {ON[:, j].mean():>10.4f} {OFF[:, j].mean():>10.4f} "
                     f"{t:>8.2f} {df:>6.2f} {p:>9.2e} {adj[j]:>9.2e}")
    up = {m: (T[:, M.index(m)].mean() > OFF[:, M.index(m)].mean(), adj[M.index(m)] < 0.05) for m in ("MI", "VIF")}
    if all(higher and sig for higher, sig in up.values()):
        p2 = "CAUSAL"
    elif all(not higher for higher, _ in up.values()):
        p2 = "NOT THE SKIP"
    else:
        p2 = "PARTIAL"
    lines.append(f"  MI: trained higher {up['MI'][0]}, survives Holm {up['MI'][1]};  "
                 f"VIF: trained higher {up['VIF'][0]}, survives Holm {up['VIF'][1]}")
    lines.append(f"  -> {p2}")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(R30, "RUN30_ANALYSIS.txt"), "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
