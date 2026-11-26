"""Audit: what the population-SD estimator did to every CrossFuse count stated in units of sigma.

Tier 2's report_tables.py and variance_full.py estimate seed sigma with statistics.pstdev over the three
R3-configuration seeds -- the POPULATION standard deviation, dividing by n rather than n-1. With n = 3 that
is sqrt(2/3) = 0.816 of the sample SD, so every "x sigma" derived from it is inflated by 1/0.816 = 1.225.
Those sigmas were then copied as SIGMA constants into analyse_tier3, analyse_a4, analyse_a3_a5 and
analyse_A1b. The A1 family (analyse_a1a, analyse_a1_seed) uses range/d2 instead and is not affected by
this particular error.

A second, separate question: every Tier 3 count applies ">= 3 sigma" to the difference between SINGLE
runs, with sigma estimated from three seeds. That is a rule of thumb, not a test. This script reports, for
the headline A3/A5 counts, what a Welch-style test with the sigma estimate's own degrees of freedom gives.

Read-only. Nothing here changes a result file; it prints the three readings side by side.

    python studied_papers/CrossFuse/Tier3_CrossFuse/scripts/audit_sigma_estimator.py
"""
import contextlib
import csv
import io
import os
import re
import statistics as st

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T3 = os.path.dirname(HERE)
T2 = os.path.join(os.path.dirname(T3), "Tier2_CrossFuse")
METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
DATASETS = ["RoadScene", "TNO", "MSRS"]
R3CFG = {"R3": ("results_R3", "tables_R3_raw"), "R3b": ("results_R3b", "tables_R3b_raw"),
         "R3c": ("results_R3c", "tables_R3c_raw")}


def mean_table(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {m: st.mean(float(r[m]) for r in rows) for m in METRICS}


vals = {ds: {m: [] for m in METRICS} for ds in DATASETS}
for run, (d1, d2) in R3CFG.items():
    for ds in DATASETS:
        t = mean_table(os.path.join(T2, d1, d2, f"{ds}_gray_per_image.csv"))
        for m in METRICS:
            vals[ds][m].append(t[m])

POP = {ds: {m: st.pstdev(vals[ds][m]) for m in METRICS} for ds in DATASETS}
SAMPLE = {ds: {m: st.stdev(vals[ds][m]) for m in METRICS} for ds in DATASETS}

print("=" * 100)
print("1. Is the SIGMA copied into the Tier 3 scripts the population SD?")
print("=" * 100)
for fn in ("analyse_tier3.py", "analyse_a4.py", "analyse_a3_a5.py", "analyse_A1b.py"):
    src = open(os.path.join(HERE, fn), encoding="utf-8").read()
    m = re.search(r"SIGMA = \{(.*?)\n\}", src, re.S)
    if not m:
        print(f"  {fn:20} no SIGMA block")
        continue
    ns = {"METRICS": METRICS}
    exec("SIGMA = {" + m.group(1) + "\n}", ns)
    worst_pop = max(abs(ns["SIGMA"][ds][k] - POP[ds][k]) for ds in DATASETS for k in METRICS)
    worst_smp = max(abs(ns["SIGMA"][ds][k] - SAMPLE[ds][k]) for ds in DATASETS for k in METRICS)
    print(f"  {fn:20} max |SIGMA - pstdev| {worst_pop:.5f}   max |SIGMA - sample SD| {worst_smp:.5f}   "
          f"-> {'POPULATION SD' if worst_pop < 5e-5 else ('sample SD' if worst_smp < 5e-5 else 'neither')}")


def run_with(fn, sigma):
    """Execute an analysis script with its SIGMA block replaced; return what it printed."""
    src = open(os.path.join(HERE, fn), encoding="utf-8").read()
    rep = "SIGMA = {\n" + "".join(
        f'    "{ds}": dict(zip(METRICS, {[round(sigma[ds][k], 6) for k in METRICS]})),\n' for ds in DATASETS) + "}"
    src2 = re.sub(r"SIGMA = \{.*?\n\}", lambda _: rep, src, count=1, flags=re.S)
    buf = io.StringIO()
    cwd = os.getcwd()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(src2, os.path.join(HERE, fn), "exec"), {"__file__": os.path.join(HERE, fn), "__name__": "__main__"})
    finally:
        os.chdir(cwd)
    return buf.getvalue()


print(f"\n{'=' * 100}\n2. Counts each script prints -- population SD (as published) vs sample SD\n{'=' * 100}")
COUNT = re.compile(r"(\d+) (?:of|/) ?(\d+)")
for fn in ("analyse_tier3.py", "analyse_a4.py", "analyse_a3_a5.py", "analyse_A1b.py"):
    a = run_with(fn, POP)
    b = run_with(fn, SAMPLE)
    la = [l for l in a.splitlines() if re.search(r"beyond 3|> ?3 sigma|real \(|term\)|No metric", l)]
    lb = [l for l in b.splitlines() if re.search(r"beyond 3|> ?3 sigma|real \(|term\)|No metric", l)]
    print(f"\n  {fn}")
    for x, y in zip(la, lb):
        mark = "" if x == y else "   <-- CHANGES"
        print(f"    published: {x.strip()[:92]}")
        if x != y:
            print(f"    sample SD: {y.strip()[:92]}{mark}")

print(f"\n{'=' * 100}\n3. Tier 2 deficits (shipped vs mean of 5 retrained), sigma units\n{'=' * 100}")
SHIP = ("results_perimage", "tables_shipped_raw")
FIVE = [("results_perimage", "tables_R1_raw"), ("results_R1b", "tables_R1b_raw"), ("results_R3", "tables_R3_raw"),
        ("results_R3b", "tables_R3b_raw"), ("results_R3c", "tables_R3c_raw")]
for ds in DATASETS:
    s = mean_table(os.path.join(T2, *SHIP, f"{ds}_gray_per_image.csv"))
    f5 = [mean_table(os.path.join(T2, a_, b_, f"{ds}_gray_per_image.csv")) for a_, b_ in FIVE]
    row = []
    for m in METRICS:
        dev = st.mean(t[m] for t in f5) - s[m]
        up, us = abs(dev) / POP[ds][m], abs(dev) / SAMPLE[ds][m]
        row.append((m, up, us))
    rp = [m for m, up, us in row if up >= 3]
    rs = [m for m, up, us in row if us >= 3]
    print(f"  {ds:9} robust (pop SD, published): {rp}")
    print(f"  {'':9} robust (sample SD)        : {rs}{'   <-- CHANGES' if rp != rs else ''}")
    print(f"  {'':9} VIF: {dict((m, (round(up, 1), round(us, 1))) for m, up, us in row)['VIF']} (published, sample)")

print(f"\n{'=' * 100}\n4. A3/A5 headline as a TEST rather than a 3-sigma rule (sample SD, df = 2 from three seeds)\n{'=' * 100}")
print("  total, arch: single run vs mean of R1/R1b  -> SE = s*sqrt(1 + 1/2);  cap: A5 vs A3 -> SE = s*sqrt(2)")
print("  two-sided 5 %, t(df=2) = 4.303.  Reported beside the rule, not in place of it.")
src = open(os.path.join(HERE, "analyse_a3_a5.py"), encoding="utf-8").read()
ns = {}
exec(src[src.index("RUNS = {"):src.index("SIGMA = {")], {"os": os, "T3": T3, "T2": T2}, ns)
RUNS = ns["RUNS"]


def load(run, ds):
    with open(os.path.join(RUNS[run], f"{ds}_gray_per_image.csv"), newline="", encoding="utf-8") as f:
        return {r["image"]: {m: float(r[m]) for m in METRICS} for r in csv.DictReader(f)}


counts = {k: 0 for k in ("total_rule_pop", "cap_rule_pop", "total_rule_smp", "cap_rule_smp", "total_t", "cap_t",
                         "arch_rule_pop", "arch_rule_smp", "arch_t")}
for ds in DATASETS:
    data = {r: load(r, ds) for r in ("A3", "A5", "R1", "R1b")}
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    for m in METRICS:
        mean = {r: np.mean([data[r][k][m] for k in keys]) for r in data}
        r1 = (mean["R1"] + mean["R1b"]) / 2
        total, arch, cap = mean["A3"] - r1, mean["A5"] - r1, mean["A5"] - mean["A3"]
        sp, ss = POP[ds][m], SAMPLE[ds][m]
        counts["total_rule_pop"] += abs(total) / sp >= 3
        counts["arch_rule_pop"] += abs(arch) / sp >= 3
        counts["cap_rule_pop"] += abs(cap) / sp >= 3
        counts["total_rule_smp"] += abs(total) / ss >= 3
        counts["arch_rule_smp"] += abs(arch) / ss >= 3
        counts["cap_rule_smp"] += abs(cap) / ss >= 3
        counts["total_t"] += abs(total) / (ss * np.sqrt(1.5)) > 4.303
        counts["arch_t"] += abs(arch) / (ss * np.sqrt(1.5)) > 4.303
        counts["cap_t"] += abs(cap) / (ss * np.sqrt(2)) > 4.303
print(f"  {'':34}{'published rule':>16}{'rule, sample SD':>17}{'t-test, df=2':>14}")
for lab, k in (("variant III total effect", "total"), ("architecture term (A5-R1)", "arch"), ("capacity term (A5-A3)", "cap")):
    print(f"  {lab:34}{counts[k + '_rule_pop']:>13}/27{counts[k + '_rule_smp']:>14}/27{counts[k + '_t']:>11}/27")
