"""Run 9 -- LRRNet Table IV at n = 8 per setting, read exactly as ../RUN9_PREREGISTRATION.md fixed it.

Nothing in this file chooses a threshold, a comparison set or a reading. Those were written on
2026-09-09, before any of the eighteen runs existed:

  * Welch's t, two-sided, 5 %. At n = 8 vs 8 the critical value is 2.145 (df = 14), taken per
    comparison from the conservative df floor in analyse_table_iv.tcrit().
  * Each ablated row (nb = 2, 6, 8) against the published choice nb = 4, nine metrics: 27 tests.
  * Multiplicity: 27 tests at 5 % expect 1.35 false positives. One significant cell is at the
    chance rate and is not a finding in either direction; two or more agreeing cells would be.
  * Kill criteria 1, 3, 4 and 5 are checked below, BEFORE any statistic is computed, and the script
    halts rather than analyse a setting that fails one. Criterion 2 (loss trajectories) is checked
    from the logs by check_run9_losses.py.

    python studied_papers/LRRNet/scripts/analyse_run9.py
"""
import contextlib
import csv
import io
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LRR = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# The seeds already in hand live in analyse_table_iv.RUNS -- the numbers every earlier report was
# built from. Importing them, rather than retyping, is what makes the pooled arms the SAME data.
with contextlib.redirect_stdout(io.StringIO()):
    import analyse_table_iv as t4

METRICS = t4.METRICS
PARAMS = t4.PARAMS
REF = 4
TARGET_N = 8
EXPECTED_NEW = {2: [45, 46, 47, 48, 49], 4: [47, 48, 49], 6: [45, 46, 47, 48, 49], 8: [45, 46, 47, 48, 49]}

failures = []


def kill(msg):
    failures.append(msg)
    print("  [KILL]", msg)


# ---------------------------------------------------------------------------------- load new seeds
runs = {nb: {s: list(v) for s, v in t4.RUNS[nb].items()} for nb in t4.RUNS}
for nb, want in EXPECTED_NEW.items():
    path = os.path.join(LRR, f"results_run9_nb{nb}", "lrrnet_metrics.csv")
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tag = row["tag"]
            seed = int(tag.split("seed")[1])
            if not tag.startswith(f"nb{nb}_"):
                kill(f"{path}: row tagged {tag} inside the nb={nb} arm")
            if seed in runs[nb]:
                kill(f"nb={nb} seed {seed} appears twice (existing and Run 9)")
            if int(row["n"]) != 25:
                kill(f"nb={nb} seed {seed} scored on {row['n']} images, not 25 TNO pairs")
            runs[nb][seed] = [float(row[m]) for m in METRICS]

print("=" * 100)
print("KILL CRITERIA -- checked before any statistic")
print("=" * 100)

# 1. fewer than 8 completed seeds
for nb in sorted(runs):
    seeds = sorted(runs[nb])
    missing = [s for s in EXPECTED_NEW[nb] if s not in runs[nb]]
    if len(seeds) < TARGET_N or missing:
        kill(f"criterion 1: nb={nb} has n={len(seeds)} (missing {missing})")
    else:
        print(f"  [OK  ] criterion 1: nb={nb} n={len(seeds)}  seeds {seeds[0]}-{seeds[-1]}")

# 3. nb = 8 must have engaged both GPUs, as stage 3 did
log8 = os.path.join(LRR, "results_run9_nb8", "lrrnet-run9-nb8.log")
text8 = open(log8, encoding="utf-8").read() if os.path.exists(log8) else ""
n_parallel = text8.count("[GPUS] 2 parallel= True")
if n_parallel < len(EXPECTED_NEW[8]):
    kill(f"criterion 3: '[GPUS] 2 parallel= True' seen {n_parallel} times in the nb=8 log, "
         f"expected {len(EXPECTED_NEW[8])}")
else:
    print(f"  [OK  ] criterion 3: nb=8 trained on two GPUs in all {n_parallel} runs")

# 4. every metric finite
bad = [(nb, s) for nb in runs for s, v in runs[nb].items() if not all(math.isfinite(x) for x in v)]
if bad:
    kill(f"criterion 4: non-finite metrics in {bad}")
else:
    print(f"  [OK  ] criterion 4: all {sum(len(r) for r in runs.values()) * 9} metric values finite")

# 5. no new seed a copy of another run (any setting) to within 1e-12
flat = [(nb, s, np.array(v)) for nb in runs for s, v in runs[nb].items()]
dupes = [(a[:2], b[:2]) for i, a in enumerate(flat) for b in flat[i + 1:]
         if np.max(np.abs(a[2] - b[2])) < 1e-12]
if dupes:
    kill(f"criterion 5: identical metric vectors {dupes}")
else:
    print(f"  [OK  ] criterion 5: no two of the {len(flat)} runs share a metric vector")

if failures:
    print(f"\nHALTED: {len(failures)} kill criterion failure(s). No statistic is computed.")
    sys.exit(1)

# ------------------------------------------------------------------------------------ the analysis
M = {nb: np.array([runs[nb][s] for s in sorted(runs[nb])]) for nb in runs}

print(f"\n{'=' * 100}\nTABLE IV AT n = 8 -- mean +/- sample SD, TNO\n{'=' * 100}")
print(f"{'metric':6}" + "".join(f"{f'nb={nb} ({PARAMS[nb]:,})':>23}" for nb in sorted(M)))
for i, m in enumerate(METRICS):
    print(f"{m:6}" + "".join(f"{M[nb][:, i].mean():>13.4f} +/-{M[nb][:, i].std(ddof=1):<8.4f}"
                             for nb in sorted(M)))

print(f"\n{'=' * 100}\nEACH ABLATED ROW AGAINST nb=4 -- Welch t, two-sided 5 %, df floor\n{'=' * 100}")
print(f"{'metric':6}" + "".join(f"{f'4 vs {nb}:  d/sd_p     t   crit':>31}" for nb in (2, 6, 8)))
sig_cells, all_t = [], []
for i, m in enumerate(METRICS):
    row = f"{m:6}"
    for nb in (2, 6, 8):
        a, b = M[REF][:, i], M[nb][:, i]
        sa, sb, na, nn = a.std(ddof=1), b.std(ddof=1), len(a), len(b)
        d = a.mean() - b.mean()
        pooled = math.sqrt(((na - 1) * sa**2 + (nn - 1) * sb**2) / (na + nn - 2))
        t = d / math.sqrt(sa**2 / na + sb**2 / nn)
        df = min(t4.welch_df(sa, na, sb, nn), na + nn - 2)
        crit = t4.tcrit(df)
        sig = abs(t) > crit
        all_t.append((abs(t), m, nb, t, crit, df))
        if sig:
            sig_cells.append((m, nb, d, t, crit, df))
        row += f"{d / pooled:>+12.2f}{t:>+8.2f}{crit:>7.3f}{' *' if sig else '  '}"
    print(row)

k = len(sig_cells)
top = sorted(all_t, reverse=True)[:3]
print(f"\n{'=' * 100}\nRESULT\n{'=' * 100}")
print(f"  {k} of 27 comparisons reach the 5 % level   (expected by chance alone: 27 x 0.05 = 1.35)")
for m, nb, d, t, crit, df in sig_cells:
    fav = "nb=4 (the published choice)" if d > 0 else f"nb={nb}"
    print(f"    * {m} 4 vs {nb}: t = {t:+.3f}, critical {crit:.3f} (df {df:.1f}) -- favours {fav}")
print("  largest |t| overall: " + ", ".join(f"{m} 4 vs {nb} = {t:+.2f} (crit {c:.3f})"
                                           for _, m, nb, t, c, _ in top))

print("\n  READING, AS PRE-REGISTERED")
if k == 0:
    print("  OUTCOME 1. Zero of 27 holds at n = 8, on a design that resolves 1.75 sigma instead of 3.5 sigma.")
elif k == 1:
    print("  OUTCOME 2. 'Zero of 27' is WITHDRAWN and replaced by the true count, everywhere it appears.")
    print("  One cell is at the chance rate (1.35 expected) and is not a finding in either direction:")
    print("  it is reported as one named cell with the expected count beside it.")
else:
    same_dir = len({(nb, d > 0) for _, nb, d, *_ in sig_cells}) == 1
    print(f"  OUTCOME 2. 'Zero of 27' is WITHDRAWN; {k} cells cross, above the 1.35 expected by chance.")
    print("  They " + ("agree in direction and setting -- that is a finding, and is reported as one."
                       if same_dir else "do not agree in direction/setting; read each against multiplicity."))
