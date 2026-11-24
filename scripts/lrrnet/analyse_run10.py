"""Run 10 -- read exactly as ../RUN10_PREREGISTRATION.md fixed it. Written and committed with that file,
before any Run 10 kernel existed.

Nothing here chooses a threshold, a comparison set or a reading:

  A. CONFIRMATORY. SCD, nb = 2 against nb = 4, on the NEW seeds 50-61 only (12 v 12), Welch's t,
     two-sided, 5 %. A1 REPLICATED if p < 0.05 with nb = 2 higher; otherwise A2 NOT REPLICATED.
     The eight old seeds generated the hypothesis and are excluded from this test by design.
  B. SECONDARY. TOST at +/-1.75 pooled sigma, 5 %, on all seeds pooled (8 + 12 = 20 per arm), for
     4v2 SF, 4v2 AG, 4v2 SCD and 4v6 SF. Each is equivalent, different (Welch two-sided 5 % rejects), or
     inconclusive; where both TOST and Welch reject, both are reported (a difference smaller than the margin). 4v6 SF stays inconclusive, and is printed as pending, until nb = 6 completes.
     The full n = 20 table (2 and 6 against 4, nine metrics, 18 cells) is descriptive, Holm-corrected,
     and is printed only when both arms are in.
  KILL CRITERIA 1-5 are checked per setting BEFORE any statistic. A setting that fails one is void;
  a setting whose kernels have not been downloaded yet is PENDING, which is not a failure.
  Criterion 2 compares each new run's FINAL logged loss with the eight existing seeds' final losses
  (stage log + Run 9 log): excluded if it lies outside their [min, max] by more than 3x the width --
  the operational reading check_run9_losses.py fixed for Run 9, applied to all eight existing seeds.

Layout expected (downloaded kernel outputs, one folder per kernel):
    ../results_run10_nb2/   lrrnet_metrics.csv, lrrnet-run10-nb2.log
    ../results_run10_nb4a/  ../results_run10_nb4b/  (seeds 50-55, 56-61)
    ../results_run10_nb6a/  ../results_run10_nb6b/

    python studied_papers/LRRNet/scripts/analyse_run10.py
"""
import argparse
import contextlib
import csv
import io
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
LRR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
_ap = argparse.ArgumentParser()
_ap.add_argument("--results", default=LRR, help="folder holding results_run10_* (default: studied_papers/LRRNet); "
                 "used only to test this script on synthetic data outside the repository")
RESULTS = os.path.abspath(_ap.parse_args().results)
with contextlib.redirect_stdout(io.StringIO()):
    import analyse_run9 as r9          # the eight existing seeds per setting; its kill criteria run on import

METRICS = r9.METRICS
NEW_SEEDS = list(range(50, 62))
KERNELS = {2: ["nb2"], 4: ["nb4a", "nb4b"], 6: ["nb6a", "nb6b"]}
OLD_LOG = {2: "stage4", 4: "stage1", 6: "stage2"}
MARGIN = 1.75
ALPHA = 0.05
FACTOR = 3.0
LOSS = re.compile(r"loss ([\d.]+) -> ([\d.]+)\s+\((\d+) logged\)")
SECONDARY = [(2, "SF"), (2, "AG"), (2, "SCD"), (6, "SF")]


def final_losses(path):
    text = "".join(e.get("data", "") for e in json.load(open(path, encoding="utf-8")))
    return [(float(b), int(n)) for _, b, n in LOSS.findall(text)]


def welch(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return t, df, 2 * stats.t.sf(abs(t), df)


# ------------------------------------------------------------------------------------ load and kill
print("=" * 100 + "\nRUN 10 -- KILL CRITERIA, checked per setting before any statistic\n" + "=" * 100)
new, status = {}, {}
for nb, kernels in KERNELS.items():
    dirs = [os.path.join(RESULTS, f"results_run10_{k}") for k in kernels]
    present = [d for d in dirs if os.path.exists(os.path.join(d, "lrrnet_metrics.csv"))]
    if not present:
        status[nb] = "PENDING"
        print(f"  nb={nb}: PENDING -- no Run 10 output downloaded yet")
        continue
    fails, runs, seen = [], {}, []
    for d in present:
        with open(os.path.join(d, "lrrnet_metrics.csv"), newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m = re.fullmatch(rf"nb{nb}_seed(\d+)", row["tag"])
                if not m:
                    fails.append(f"criterion 5: tag {row['tag']!r} in {os.path.basename(d)}")
                    continue
                s = int(m.group(1))
                seen.append(s)
                if int(row["n"]) != 25:
                    fails.append(f"seed {s} scored on {row['n']} images, not 25")
                runs[s] = np.array([float(row[k]) for k in METRICS])
    # 5. only 50-61, none twice (within Run 10, or colliding with an existing seed)
    bad = sorted({s for s in seen if s not in NEW_SEEDS or seen.count(s) > 1 or s in r9.runs[nb]})
    if bad:
        fails.append(f"criterion 5: seeds {bad} outside 50-61, repeated, or already used")
    # 1. twelve completed seeds (a kernel folder not yet downloaded counts as incomplete only once the
    #    other half is in: a half-downloaded setting is PENDING, not void)
    missing = [s for s in NEW_SEEDS if s not in runs]
    if missing and len(present) < len(dirs):
        status[nb] = "PENDING"
        print(f"  nb={nb}: PENDING -- {len(present)} of {len(dirs)} kernels downloaded, seeds {sorted(runs)}")
        continue
    if missing:
        fails.append(f"criterion 1: missing seeds {missing}")
    # 3. finite
    if any(not np.all(np.isfinite(v)) for v in runs.values()):
        fails.append("criterion 3: non-finite metric")
    # 2. final loss against the eight existing seeds
    old = final_losses(os.path.join(LRR, "stage_logs", OLD_LOG[nb], f"lrrnet-{OLD_LOG[nb]}.log")) + \
        final_losses(os.path.join(LRR, f"results_run9_nb{nb}", f"lrrnet-run9-nb{nb}.log"))
    if len(old) != 8:
        sys.exit(f"nb={nb}: expected 8 existing loss trajectories, found {len(old)} -- fix the log paths")
    lo, hi = min(f for f, _ in old), max(f for f, _ in old)
    n_logged = {n for _, n in old}
    newl = []
    for d, k in zip(dirs, kernels):
        if d in present:
            newl += final_losses(os.path.join(d, f"lrrnet-run10-{k}.log"))
    if len(newl) != len(runs):
        fails.append(f"criterion 2: {len(newl)} loss trajectories in the logs for {len(runs)} metric rows")
    for i, (fl, n) in enumerate(newl):
        outside = max(lo - fl, fl - hi, 0.0)
        if outside > FACTOR * (hi - lo) or n not in n_logged:
            fails.append(f"criterion 2: run {i + 1} final loss {fl:,.1f} ({n} logged) vs existing "
                         f"{lo:,.1f}-{hi:,.1f}")
    new[nb] = runs
    status[nb] = "VOID" if fails else "OK"
    for msg in fails:
        print(f"  [KILL] nb={nb} {msg}")
    if not fails:
        print(f"  [OK  ] nb={nb}: 12 seeds (50-61), finite, final losses {min(newl)[0]:,.0f}-{max(newl)[0]:,.0f} "
              f"within {FACTOR:.0f}x of existing {lo:,.0f}-{hi:,.0f}")

# 4. no two runs anywhere (old or new, any setting) share a metric vector
flat = [(nb, s, v) for nb in new if status[nb] != "PENDING" for s, v in new[nb].items()]
flat += [(nb, s, np.array(v)) for nb in (2, 4, 6) for s, v in r9.runs[nb].items()]
dupes = [(a[:2], b[:2]) for i, a in enumerate(flat) for b in flat[i + 1:] if np.max(np.abs(a[2] - b[2])) < 1e-12]
for a, b in dupes:
    for (nb, s), other in ((a, b), (b, a)):
        if s in NEW_SEEDS:
            print(f"  [KILL] nb={nb} criterion 4: seed {s} identical to {other}")
            status[nb] = "VOID"

ok = {nb for nb, st in status.items() if st == "OK"}
print("  status: " + ", ".join(f"nb={nb} {status[nb]}" for nb in (2, 4, 6)))

NEW = {nb: np.array([new[nb][s] for s in NEW_SEEDS]) for nb in ok}
POOL = {nb: np.vstack([np.array([r9.runs[nb][s] for s in sorted(r9.runs[nb])]), NEW[nb]]) for nb in ok}

# ------------------------------------------------------------------------------------ A. confirmatory
i_scd = METRICS.index("SCD")
print(f"\n{'=' * 100}\nA. CONFIRMATORY -- SCD, nb=2 against nb=4, new seeds 50-61 only, Welch two-sided 5 %\n{'=' * 100}")
if {2, 4} <= ok:
    a, b = NEW[2][:, i_scd], NEW[4][:, i_scd]
    t, df, p = welch(a, b)
    sp = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    print(f"  nb=2 {a.mean():.4f} +/-{a.std(ddof=1):.4f}   nb=4 {b.mean():.4f} +/-{b.std(ddof=1):.4f}   "
          f"diff {a.mean() - b.mean():+.4f} = {(a.mean() - b.mean()) / sp:+.2f} pooled sigma")
    print(f"  Welch t = {t:+.3f}, df = {df:.1f}, p = {p:.4f}")
    if p < ALPHA and a.mean() > b.mean():
        print("  A1 REPLICATED: a difference on one metric of nine, found at eight seeds, survived an independent\n"
              "  registered replication. Not '2 blocks is better' -- eight other metrics show no such difference;\n"
              "  on SCD the published ordering is reversed.")
    else:
        why = "p >= 0.05" if p >= ALPHA else "nb=4 higher"
        print(f"  A2 NOT REPLICATED ({why}): Run 9's cell was consistent with chance, as the correction for 27 tests\n"
              "  indicated. Power was 82 % at the selected 1.23 sigma and 47 % at 0.8 sigma, so this does not show\n"
              "  the effect is zero, only that it did not replicate at that size.")
else:
    print("  not read: needs nb=2 and nb=4 both OK -- " + ", ".join(f"nb={nb} {status[nb]}" for nb in (2, 4)))

# ------------------------------------------------------------------------------------ B. secondary
print(f"\n{'=' * 100}\nB. SECONDARY -- TOST at +/-{MARGIN} pooled sigma on all seeds pooled (n = 20 per arm)\n{'=' * 100}")


def cell(nb, m):
    j = METRICS.index(m)
    a, b = POOL[4][:, j], POOL[nb][:, j]
    na, n2 = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (na + n2 - 2))
    t, df, p = welch(a, b)
    d = (a.mean() - b.mean()) / sp
    se = math.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / n2) / sp
    q = stats.t.ppf(1 - ALPHA, df)
    p_tost = max(stats.t.sf((d + MARGIN) / se, df), stats.t.cdf((d - MARGIN) / se, df))
    return d, d - q * se, d + q * se, t, p, p_tost


for nb, m in SECONDARY:
    if not ({4, nb} <= ok):
        word = "VOID, not read" if "VOID" in (status[4], status[nb]) else "INCONCLUSIVE, pending"
        print(f"  4v{nb} {m:4}: {word} -- " + ", ".join(f"nb={k} {status[k]}" for k in sorted({4, nb})))
        continue
    d, lo, hi, t, p, pt = cell(nb, m)
    reading = ("EQUIVALENT AND DIFFERENT -- a real difference smaller than the design resolution"
               if pt < ALPHA and p < ALPHA else "EQUIVALENT" if pt < ALPHA else "DIFFERENT" if p < ALPHA
               else "INCONCLUSIVE")
    print(f"  4v{nb} {m:4}: d = {d:+.2f} sigma, 90 % CI [{lo:+.2f}, {hi:+.2f}], Welch p = {p:.4f}, "
          f"TOST p = {pt:.4f} -> {reading}")

print(f"\n{'=' * 100}\nFULL TABLE AT n = 20 -- descriptive, Holm across 18 cells (2 and 6 against 4)\n{'=' * 100}")
if {2, 4, 6} <= ok:
    rows = [(nb, m, *cell(nb, m)) for nb in (2, 6) for m in METRICS]
    ps = np.array([r[6] for r in rows])
    order, holm, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        holm[i] = run
    print(f"{'cell':>10}{'d (sigma)':>11}{'t':>8}{'p':>9}{'Holm':>8}{'TOST p':>9}")
    for (nb, m, d, lo, hi, t, p, pt), h in zip(rows, holm):
        print(f"{f'4v{nb} {m}':>10}{d:>+11.2f}{t:>+8.2f}{p:>9.4f}{h:>8.3f}{pt:>9.4f}")
    print(f"  {int(np.sum(ps < ALPHA))} of 18 uncorrected, {int(np.sum(holm < ALPHA))} of 18 after Holm. "
          "Descriptive; Run 9's count is not replaced.")
else:
    print("  not printed until nb=2, nb=4 and nb=6 are all OK -- " + ", ".join(f"nb={nb} {status[nb]}" for nb in (2, 4, 6)))
