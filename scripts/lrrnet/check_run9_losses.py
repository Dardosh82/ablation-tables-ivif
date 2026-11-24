"""Run 9, kill criterion 2: is any new seed's training trajectory out of family with the old ones?

The pre-registration says: "A new seed's loss trajectory outside the range of the existing seeds by
more than 3x -> that run is excluded and named." It does not say which point of the trajectory, or
3x of what. Both are fixed HERE, and this was written and run BEFORE analyse_run9.py computed a
single statistic, so the reading cannot have been chosen to suit a result:

  * The FINAL logged loss is compared. The first logged loss is a property of the random
    initialisation -- it already spans 107k to 197k across five seeds of one setting -- and says
    nothing about whether training went normally.
  * "Outside the range by more than 3x" = the distance from the existing seeds' [min, max] exceeds
    three times that range's width. A new seed inside the range, or near it, passes.

The existing seeds' logs are pulled from the stage kernels that produced them (stage_logs/), with
only the .log matched, so this compares against the runs the published analysis actually used.

    python studied_papers/LRRNet/scripts/check_run9_losses.py      # exit 1 if any run is excluded
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LRR = os.path.dirname(HERE)

OLD = {4: "stage1", 6: "stage2", 8: "stage3", 2: "stage4"}      # as analyse_table_iv.RUNS records
LOSS = re.compile(r"loss ([\d.]+) -> ([\d.]+)\s+\((\d+) logged\)")
FACTOR = 3.0


def losses(path):
    text = "".join(e.get("data", "") for e in json.load(open(path, encoding="utf-8")))
    return [(float(a), float(b), int(n)) for a, b, n in LOSS.findall(text)]


excluded = []
for nb in (2, 4, 6, 8):
    old = losses(os.path.join(LRR, "stage_logs", OLD[nb], f"lrrnet-{OLD[nb]}.log"))
    new = losses(os.path.join(LRR, f"results_run9_nb{nb}", f"lrrnet-run9-nb{nb}.log"))
    lo, hi = min(f for _, f, _ in old), max(f for _, f, _ in old)
    width = hi - lo
    print(f"nb={nb}: existing final loss {lo:,.1f} - {hi:,.1f} (width {width:,.1f}, n={len(old)}); "
          f"tolerance {FACTOR:.0f}x width = {FACTOR * width:,.1f}")
    for i, (first, final, n_logged) in enumerate(new):
        outside = max(lo - final, final - hi, 0.0)
        ok = outside <= FACTOR * width and n_logged == old[0][2]
        tag = "OK  " if ok else "EXCL"
        print(f"   [{tag}] new run {i + 1}: {first:>10,.1f} -> {final:>9,.1f}   "
              f"outside range by {outside:>8,.1f}   ({n_logged} points logged)")
        if not ok:
            excluded.append((nb, i + 1, final))

print()
if excluded:
    print(f"criterion 2: {len(excluded)} run(s) EXCLUDED -- name them in the write-up: {excluded}")
    sys.exit(1)
print("criterion 2: every new run's final loss lies within tolerance of its setting's existing seeds")
