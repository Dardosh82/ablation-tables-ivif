"""Step 0, zero GPU: is LRRNet's Table IV worth measuring sigma for?

Table IV (arXiv:2304.05172, TNO) varies the number of LLRR blocks over {2,4,6,8} and the paper
selects 4: "When the number of LLRR blocks is set to 4, our fusion network delivers the best fusion
performance (five best values, En, SD, MI, VIFm and Nabf)."

This screen does NOT claim their table is wrong. We do not have LRRNet's sigma -- measuring it is
the proposed experiment. We apply CrossFuse's TNO sigma as a PROXY to answer one question only:

    do the decisive comparisons sit near a plausible noise floor, or far above it?

If they sit far above, sigma would change nothing and the 8.8 GPU-hours are wasted. If they sit
near it, the answer depends on a number nobody has measured -- which is the case for spending them.

A second, unexpected obstacle surfaced while doing this: the two papers do not report metrics on a
common scale, so most columns cannot be screened at all. That is recorded, not worked around.
"""

# CrossFuse TNO sigma, from three runs differing only in seed.
# Full nine-metric vector, from Tier3_CrossFuse/scripts/analyse_tier3.py. An earlier version of
# this file used the six-metric subset carried by analyse_tablevi_direct.py (which only holds the
# metrics Table VI reports) and wrongly recorded SD as unmeasured. SD *was* measured: 0.2962.
# CrossFuse seed sigma: SAMPLE SD (n-1) of three R3-configuration seeds. Until 2026-09-13 this held the
# POPULATION SD (0.816 of the sample value) -- withdrawn claim #13. Computed from the tables, not typed.
SIGMA_TNO = {"EN": 0.0166, "MI": 0.0052, "SD": 0.3627, "SF": 0.2866, "AG": 0.0793, "VIF": 0.0016, "SCD": 0.0374, "Qabf": 0.0016, "SSIM": 0.0074}

METRICS = ["En", "SD", "MI", "SSIMm", "VIFm", "Nabf"]
LOWER_IS_BETTER = {"Nabf"}

TABLE_IV = {                # blocks -> values, transcribed from the published Table IV
    2: [6.79214, 78.25832, 13.58429, 0.71164, 0.66020, 0.14770],
    4: [6.85836, 81.78905, 13.71673, 0.70625, 0.71399, 0.14168],
    6: [6.82875, 80.37536, 13.65750, 0.70726, 0.68716, 0.14669],
    8: [6.83084, 80.24377, 13.66168, 0.70250, 0.69624, 0.16993],
}
PARAMS = {2: 29_744, 4: 49_204, 6: 68_664, 8: 88_124}   # measured, check_numblock_capacity.py

# Which of our sigmas may stand in for which of their columns, and whether the scales agree.
# CrossFuse TNO reference values: EN 7.21, MI 2.13, VIF 0.77, SSIM 0.99.
PROXY = {
    "En":    ("EN",   True,  "same scale (6.8 vs 7.2) -- usable as a proxy"),
    "SD":    ("SD",   False, "pixel-intensity std, a convention-stable quantity, but the levels "
                             "differ (78-82 here vs 48.7 in our TNO runs) -- indicative only"),
    "MI":    (None,   False, "SCALE MISMATCH: they report 13.7, we measure 2.13 (6.4x) -- "
                             "a different MI convention, so our sigma is meaningless here"),
    "SSIMm": ("SSIM", False, "'SSIMm' is a modified SSIM: 0.70 here vs 0.99 for standard SSIM -- "
                             "treat as indicative only"),
    "VIFm":  ("VIF",  False, "'VIFm' is a modified VIF: 0.71 vs 0.77 -- indicative only"),
    "Nabf":  (None,   False, "artifact metric, never measured in our runs"),
}

print("=" * 88)
print("LRRNet Table IV -- capacity, then a proxy noise screen")
print("=" * 88)
print(f"{'blocks':>7} {'params':>9} {'vs 4':>7}  " + "".join(f"{m:>10}" for m in METRICS))
print("-" * 88)
for b, vals in TABLE_IV.items():
    mark = " <- chosen" if b == 4 else ""
    print(f"{b:>7} {PARAMS[b]:>9,} {PARAMS[b]/PARAMS[4]*100:>6.0f}%  "
          + "".join(f"{v:>10.5f}" for v in vals) + mark)

print(f"\nCAPACITY: the ablated range spans {max(PARAMS.values())/min(PARAMS.values()):.2f}x in "
      "parameters.\nBlock count and capacity move together, so no row difference can be attributed "
      "to representational\ndepth alone -- the same defect established for CrossFuse Table VI rows "
      "III and IV, in a different\narchitecture family.\n")

print("=" * 88)
print("PROXY SCREEN -- chosen configuration (4 blocks) against each alternative")
print("=" * 88)
for m in METRICS:
    ours, clean, note = PROXY[m]
    if ours is None:
        print(f"\n{m:6} SKIPPED -- {note}")
        continue
    s = SIGMA_TNO[ours]
    tag = "proxy sigma" if clean else "INDICATIVE ONLY"
    print(f"\n{m:6} {tag} = {s} (our {ours}); {note}")
    i = METRICS.index(m)
    for b in (2, 6, 8):
        d = TABLE_IV[4][i] - TABLE_IV[b][i]
        if m in LOWER_IS_BETTER:
            d = -d
        u = abs(d) / s
        # Deliberately conditional wording. This is NOT a significance test for LRRNet: the sigma
        # is CrossFuse's, measured on a different model, training corpus and codebase. No cell here
        # may be called "real" or "significant" -- only "would be, IF LRRNet's sigma resembles ours".
        verdict = ("would be NOISE if sigma_LRRNet ~ sigma_CrossFuse"
                   if u < 3 else "would be RESOLVABLE if sigma_LRRNet ~ sigma_CrossFuse")
        print(f"        4 vs {b}:  delta {d:+9.5f}  = {u:6.2f} ref-units   {verdict}")

print(f"\n{'='*88}")
print("WHY THE TABLE-WIDE RANGE IS THE WRONG STATISTIC")
print("=" * 88)
print("A natural shortcut is to take each column's min-to-max range and divide by sigma. It is\n"
      "wrong, and Table IV shows why: the range is dominated by the WORST row, while the claim\n"
      "under test is that the chosen row beats its RUNNER-UP.\n")
print(f"{'metric':7} {'range (min..max)':>18} {'range/sigma':>13} {'4 vs runner-up':>16} {'verdict':>12}")
print("-" * 88)
for m in METRICS:
    ours, _, _ = PROXY[m]
    if ours is None:
        continue
    i, s = METRICS.index(m), SIGMA_TNO[PROXY[m][0]]
    col = [TABLE_IV[b][i] for b in TABLE_IV]
    rng = max(col) - min(col)
    others = [TABLE_IV[b][i] for b in (2, 6, 8)]
    best_other = min(others) if m in LOWER_IS_BETTER else max(others)
    gap = abs(TABLE_IV[4][i] - best_other)
    print(f"{m:7} {rng:>18.5f} {rng/s:>12.1f}x {gap/s:>15.2f}x "
          f"{'noise?' if gap/s < 3 else 'resolvable?':>12}")
print("\nOn En the range reads 4.0 sigma -- 'a large effect' -- while the gap between 4 blocks and\n"
      "the next-best setting is 1.66 sigma, inside the noise band. The range-based reading and the\n"
      "pairwise reading give OPPOSITE answers on the same column. Only the pairwise one addresses\n"
      "the paper's actual claim, which is that 4 is the right number of blocks.")

print(f"""
{'='*88}
READING
{'='*88}
Two things fall out, and only one of them was expected.

1. The screen does NOT settle Table IV -- and that is precisely the argument for measuring sigma.
   On En, 4 beats 2 by 4.9 proxy-sigma (real), but beats 6 by 2.2 and 8 by 2.0 -- both inside the
   noise band. On VIFm the same comparisons look decisively real; on SSIMm the 6-block model is
   marginally ahead. So the published numbers support "at least 4 blocks", while the paper's
   claim is "exactly 4", and whether that stronger claim survives depends entirely on a sigma
   nobody has measured. That is the case for spending 8.8 GPU-hours, not against it.

2. Unexpected obstacle: THE TWO PAPERS DO NOT SHARE A METRIC SCALE. LRRNet reports MI = 13.7 where
   we measure 2.13 on the same metric name -- a 6.4x difference that can only be a different
   normalisation convention. SSIMm and VIFm are likewise modified variants. Of six columns, only
   En can be screened cleanly and two more indicatively; three cannot be screened at all.
   This is the same class of problem as CDDFuse publishing SSIM = 1.03, above standard SSIM's
   maximum. It is worth its own subsection: published IVIF metric values are not on a common
   scale, so cross-paper comparison of tables -- which the field does routinely -- is unsound.

VERDICT: GO. The decisive comparisons sit in the region where the answer depends on sigma.""")
