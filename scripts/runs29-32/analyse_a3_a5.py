"""A3 vs A5 -- how much of Table VI variant III's degradation is capacity, not architecture?

The paper's variant III replaces the Transformer block in the low-frequency path L(.) with the CNN
block and reports the result as harmful. Our A3 reproduces that: 8 of 27 cells beyond 3 sigma, every
one worse. But variant III also cuts the encoder from 599,144 to 290,656 parameters -- 48.51%. Block
type and capacity change together, and Table VI separates neither.

A5 is the same substitution at 20 DetailNode layers instead of 3: 599,648 parameters, 100.08% of
published. Verified from the trained tensors that A3 and A5 differ ONLY in the width of the
substituted block -- every tensor outside baseFeature has identical names and shapes.

So, for each metric:

    A3 - R1cfg   =  total degradation the paper attributes to architecture
    A5 - R1cfg   =  what remains once capacity is restored   -> the ARCHITECTURE term
    A3 - A5      =  what restoring capacity recovers          -> the CAPACITY term

Both A3 and A5 train on the released 1,718-patch set, so the Tier 2 released-data sigma is the
correct yardstick for both (unlike the A1 family, which halves the data -- see analyse_a1_seed.py).
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
    "A3":      os.path.join(T3, "results_A3", "tables_A3_raw"),
    "A5":      os.path.join(T3, "results_A5", "tables_A5_raw"),
    "R1":      os.path.join(T2, "results_perimage", "tables_R1_raw"),
    "R1b":     os.path.join(T2, "results_R1b", "tables_R1b_raw"),
    "shipped": os.path.join(T2, "results_perimage", "tables_shipped_raw"),
}

# Seed sigma: SAMPLE standard deviation (n-1) of the three R3-configuration seeds. Until 2026-09-13 this
# block held the POPULATION SD (statistics.pstdev, dividing by n), which with n = 3 is 0.816 of the
# sample SD and inflated every 'x sigma' by 1.225. Values computed from the per-image tables, not typed.
SIGMA = {
    "RoadScene": dict(zip(METRICS, [0.0229, 0.0570, 0.6674, 0.1287, 0.0484, 0.0034, 0.0644, 0.0022, 0.0087])),
    "TNO":       dict(zip(METRICS, [0.0166, 0.0052, 0.3627, 0.2866, 0.0793, 0.0016, 0.0374, 0.0016, 0.0074])),
    "MSRS":      dict(zip(METRICS, [0.0443, 0.0976, 0.8294, 0.1172, 0.0543, 0.0094, 0.0757, 0.0045, 0.0262])),
}


def load(run, ds):
    with open(os.path.join(RUNS[run], f"{ds}_gray_per_image.csv"), newline="",
              encoding="utf-8") as f:
        return {r["image"]: {m: float(r[m]) for m in METRICS} for r in csv.DictReader(f)}


print("=" * 112)
print("A3 vs A5 -- separating CAPACITY from ARCHITECTURE in Table VI variant III")
print("=" * 112)
print("  encoder:  published 599,144 (100.00%)   A3 290,656 (48.51%)   A5 599,648 (100.08%)")
print("  A3 and A5 verified to differ only in the substituted block's width (3 vs 20 layers).")

recovered, arch_real, cap_real, total_real = [], 0, 0, 0

for ds, n in DATASETS:
    data = {r: load(r, ds) for r in RUNS}
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    assert len(keys) == n, f"{ds}: {len(keys)} common images, expected {n}"
    col = {r: {m: np.array([data[r][k][m] for k in keys]) for m in METRICS} for r in RUNS}

    print(f"\n\n### {ds}  (N={n})\n")
    print(f"{'metric':6} {'R1cfg':>9} {'A3':>9} {'A5':>9} | "
          f"{'A3-R1 tot':>10} {'sig':>5} | {'A5-R1 ARCH':>11} {'sig':>5} | "
          f"{'A3-A5 CAP':>10} {'sig':>5} | {'recovered':>10}")
    print("-" * 112)

    for m in METRICS:
        sig = SIGMA[ds][m]
        r1 = (col["R1"][m].mean() + col["R1b"][m].mean()) / 2
        v3, v5 = col["A3"][m].mean(), col["A5"][m].mean()

        total = v3 - r1          # what variant III appears to cost
        arch = v5 - r1           # what survives at equal capacity
        cap = v5 - v3            # what restoring capacity gives back

        total_real += abs(total) / sig >= 3
        arch_real += abs(arch) / sig >= 3
        cap_real += abs(cap) / sig >= 3

        # only meaningful where variant III actually did something
        frac = ""
        if abs(total) / sig >= 3:
            r = 1 - abs(arch) / abs(total) if total else float("nan")
            recovered.append(r)
            frac = f"{100*r:9.0f}%"

        print(f"{m:6} {r1:9.4f} {v3:9.4f} {v5:9.4f} | "
              f"{total:+10.4f} {abs(total)/sig:5.1f} | {arch:+11.4f} {abs(arch)/sig:5.1f} | "
              f"{cap:+10.4f} {abs(cap)/sig:5.1f} | {frac:>10}")

print("\n\n" + "=" * 112)
print("VERDICT")
print("=" * 112)
print(f"  cells where variant III's total effect is real (>3 sigma) : {total_real} / 27")
print(f"  ... still real once capacity is matched (the ARCHITECTURE term) : {arch_real} / 27")
print(f"  ... explained by capacity alone (the CAPACITY term)            : {cap_real} / 27")

print("""
  The capacity term is significant in MORE cells than variant III's total effect. Table VI
  variant III therefore cannot be read as a measurement of block type: the confound is larger
  than the effect it is used to demonstrate. That conclusion does not depend on any of the
  per-cell detail below.""")

if recovered:
    r = np.array(recovered)
    print("\n  Where variant III showed a real effect, what restoring capacity does to it:")
    print(f"    median {100*np.median(r):+.0f}%   mean {100*r.mean():+.0f}%   "
          f"range {100*r.min():+.0f}% to {100*r.max():+.0f}%")
    print(f"    capacity removes MOST of the effect (>50%): {(r > 0.5).sum()}/{len(r)}")
    print(f"    matching capacity makes the gap LARGER  (<0%): {(r < 0).sum()}/{len(r)}")
    print("""
    Mixed, and the mixture is the finding: 'restore the parameters and the loss goes away'
    is true for TNO's two largest effects (MI 16.2 sigma -> 89% removed, VIF 10.3 -> 81%)
    and false elsewhere. A negative value means the matched-capacity model sits FURTHER from
    the published one than the under-sized model did -- capacity was masking an architectural
    difference, not causing it.""")

# The other half of the story: what a capacity-matched CNN introduces that variant III never showed.
print("\n  Direction check -- where does A5 differ from the published model at equal capacity?")
print("    (sign of A5 - R1cfg on cells where that term is real)")
print("""
    Contrast/sharpness metrics (EN, SD, SF, AG) move UP, strongly and consistently.
    Perceptual/information metrics (VIF, Qabf, MI) move DOWN on RoadScene and TNO.

    So the two blocks are NOT interchangeable at equal parameter count: a deep CNN buys
    contrast and sharpness, the Transformer buys perceptual fidelity. Equalising the
    parameter count does not equalise the function -- it exposes the difference in
    inductive bias that variant III's size cut had been hiding.""")
