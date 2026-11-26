"""A4 -- Table VI variant IV: the CNN in the high-frequency path H(.) replaced by a Transformer.

The mirror image of variant III's confound. Where A3 CUT the encoder to 48.51%, A4 RAISES it to
151.49% -- the Transformer block is 363,016 parameters against the CNN's 54,528. And unlike A3,
this one cannot be capacity-controlled: the substituted block has no width knob that reaches down
to the CNN's size. Narrowing ffn_expansion_factor from 1.0 to 0.1 only moves it 363,016 -> 349,792,
still 6.4x the block it replaces, because the attention projections dominate and are fixed by dim.

So the interpretation was PRE-REGISTERED before the run, in TIER3_RESULTS.md and in the notebook
brief, precisely so the reading could not be bent to fit the outcome:

    A4 WORSE  -> clean and strong. More parameters AND worse performance means the block type is
                 genuinely worse in H(.); the capacity confound works against the conclusion, so it
                 cannot manufacture it.
    A4 BETTER -> confounded and unusable as evidence for the Transformer. Cannot separate the block
                 from the extra 51% of parameters.

Judged against the Tier 2 released-data sigma: A4 trains on the released 1,718-patch set, so that
yardstick applies directly (unlike the A1 family, which halves the data).
"""

import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T3 = os.path.join(HERE, "..")
T2 = os.path.join(T3, "..", "Tier2_CrossFuse")

METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
DATASETS = [("RoadScene", 40), ("TNO", 25), ("MSRS", 361)]

RUNS = {
    "A4":  os.path.join(T3, "results_A4", "tables_A4_raw"),
    "A3":  os.path.join(T3, "results_A3", "tables_A3_raw"),
    "A5":  os.path.join(T3, "results_A5", "tables_A5_raw"),
    "R1":  os.path.join(T2, "results_perimage", "tables_R1_raw"),
    "R1b": os.path.join(T2, "results_R1b", "tables_R1b_raw"),
}

# Seed sigma: SAMPLE standard deviation (n-1) of the three R3-configuration seeds. Until 2026-09-13 this
# block held the POPULATION SD (statistics.pstdev, dividing by n), which with n = 3 is 0.816 of the
# sample SD and inflated every 'x sigma' by 1.225. Values computed from the per-image tables, not typed.
SIGMA = {
    "RoadScene": dict(zip(METRICS, [0.0229, 0.0570, 0.6674, 0.1287, 0.0484, 0.0034, 0.0644, 0.0022, 0.0087])),
    "TNO":       dict(zip(METRICS, [0.0166, 0.0052, 0.3627, 0.2866, 0.0793, 0.0016, 0.0374, 0.0016, 0.0074])),
    "MSRS":      dict(zip(METRICS, [0.0443, 0.0976, 0.8294, 0.1172, 0.0543, 0.0094, 0.0757, 0.0045, 0.0262])),
}

# Higher is better for all nine metrics.
def load(run, ds):
    with open(os.path.join(RUNS[run], f"{ds}_gray_per_image.csv"), newline="",
              encoding="utf-8") as f:
        return {r["image"]: {m: float(r[m]) for m in METRICS} for r in csv.DictReader(f)}


print("=" * 100)
print("A4 -- Table VI variant IV: Transformer replaces the CNN in H(.)")
print("=" * 100)
print("  encoder:  published 599,144 (100.00%)   A4 907,632 (151.49%)  <- +51% capacity")
print("  PRE-REGISTERED: worse => clean result (capacity works against it);"
      " better => confounded.")

worse = better = 0
rows_real = []

for ds, n in DATASETS:
    data = {r: load(r, ds) for r in RUNS}
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    assert len(keys) == n, f"{ds}: {len(keys)} common images, expected {n}"
    col = {r: {m: np.array([data[r][k][m] for k in keys]) for m in METRICS} for r in RUNS}

    print(f"\n\n### {ds}  (N={n})\n")
    print(f"{'metric':6} {'R1cfg':>9} {'A4':>9} {'delta':>9} {'sigma':>7} {'verdict':>9}"
          f"   | {'A3':>9} {'A5':>9}")
    print("-" * 88)

    for m in METRICS:
        sig = SIGMA[ds][m]
        r1 = (col["R1"][m].mean() + col["R1b"][m].mean()) / 2
        v4 = col["A4"][m].mean()
        d = v4 - r1
        u = abs(d) / sig
        verdict = "**REAL**" if u >= 3 else ("marginal" if u >= 2 else "noise")
        if u >= 3:
            rows_real.append((ds, m, d, u))
            if d < 0:
                worse += 1
            else:
                better += 1
        print(f"{m:6} {r1:9.4f} {v4:9.4f} {d:+9.4f} {u:7.1f} {verdict:>9}"
              f"   | {col['A3'][m].mean():9.4f} {col['A5'][m].mean():9.4f}")

print("\n\n" + "=" * 100)
print("VERDICT")
print("=" * 100)
print(f"  cells beyond 3 sigma : {len(rows_real)} / 27")
print(f"    WORSE than the published configuration : {worse}")
print(f"    BETTER                                 : {better}")

print("\n  Significant cells, largest first:")
for ds, m, d, u in sorted(rows_real, key=lambda x: -x[3]):
    print(f"    {ds:10} {m:6} {d:+9.4f}  {u:6.1f} sigma   "
          f"{'WORSE' if d < 0 else 'BETTER'}")

if worse and worse >= 2 * max(better, 1):
    print("""
  READING (per the pre-registration): predominantly WORSE.
  A4 carries 51% MORE parameters than the published model and still loses. The capacity
  confound points the opposite way to the result, so it cannot have manufactured it --
  a lower bound on the harm, not an artefact. The CNN in H(.) is doing something the
  Transformer does not, and Table VI variant IV understates it.""")
elif better and better >= 2 * max(worse, 1):
    print("""
  READING (per the pre-registration): predominantly BETTER -- CONFOUNDED.
  A4 also carries 51% more parameters, so this cannot be attributed to the block type.
  Not usable as evidence for the Transformer in H(.); reported as inconclusive.""")
else:
    print("""
  READING (per the pre-registration): MIXED -- direction-dependent.
  Metrics that improve are confounded with the +51% capacity and cannot be credited to the
  block; metrics that degrade are clean, since capacity works against them. Report the two
  halves separately rather than as a single verdict.""")
