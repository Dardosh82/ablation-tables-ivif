"""Sanity-check the derived critical value before Run 9's data arrives.

Three properties it must have, none of which the analysis output would reveal:
  1. it returns the textbook value at the two designs this project actually uses
  2. it is monotone -- more degrees of freedom never demands a larger t
  3. a df below the table floors at the STRICTEST value, not the loosest
"""
import io
import os
import re

# the analysis script sits beside this one, here and in the public release (was an absolute path, 2026-11-25)
src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyse_table_iv.py"),
              encoding="utf-8").read()
block = src[src.index("T_TWO_SIDED_5PCT = {"):src.index("verdicts = {")]
ns = {}
exec(block, ns)   # one namespace, so the functions see the table
tcrit, welch_df = ns["tcrit"], ns["welch_df"]

print(f"{'design':>24} {'df':>4} {'t_crit':>7}   expected")
for label, df, want in (("n=3 vs n=3 (ablated rows)", 4, 2.776),
                        ("n=5 vs n=3 (reference)", 6, 2.447),
                        ("n=5 vs n=5", 8, 2.306),
                        ("n=8 vs n=8 (Run 9)", 14, 2.145)):
    got = tcrit(df)
    print(f"{label:>24} {df:>4} {got:>7.3f}   {want}  {'OK' if abs(got - want) < 1e-9 else 'MISMATCH'}")

mono = all(tcrit(d) >= tcrit(d + 1) for d in range(1, 30))
print(f"\nmonotone in df: {mono}")
print(f"df below the table floors strictest: {tcrit(0.5) == 12.706}")

# a df BETWEEN tabulated points must round DOWN, giving the larger (stricter) value
print(f"df=13.7 -> {tcrit(13.7):.3f} (must be 2.160, the df=13 value, not 2.145)")

# Welch df for the two designs, at equal variance
import math
print(f"\nWelch df, equal variances: n=3 vs 3 -> {welch_df(1, 3, 1, 3):.1f} "
      f"| n=8 vs 8 -> {welch_df(1, 8, 1, 8):.1f}")
