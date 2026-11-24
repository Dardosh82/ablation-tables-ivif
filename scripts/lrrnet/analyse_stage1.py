"""Stage 1: sigma_LRRNet, and how much of it survives fixing the seed.

Five runs of LRRNet's published configuration (num_block = 4, 4 epochs, batch 8, 128x128, lr 1e-5)
differing only in the random seed, trained on LLVIP and scored on the 25 TNO pairs with CrossFuse's
Evaluator -- the same evaluator behind every CrossFuse number in this project.

From here on, THIS is the sigma used to judge LRRNet. The CrossFuse reference level that made the
zero-GPU screen possible is retired: it came from a different model on different data and could
only ever answer "is this worth measuring?".

The second block is an accident worth keeping. The first stage-1 attempt crashed after seed 42 had
already trained and fused; its checkpoint and fused images survived in the notebook output and were
scored separately. The re-run then trained seed 42 again from the same seed. Two runs, same seed,
same code -- so their difference isolates the hardware nondeterminism we deliberately did not
suppress, with the seed's contribution removed.
"""
import numpy as np

METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]

# Five seeds, one Kaggle session, one code path (lrrnet_metrics.csv).
RUNS = {
    42: [6.943322, 2.303000, 43.838683, 10.230884, 3.804604, 0.695046, 1.284920, 0.410110, 0.881323],
    43: [6.979207, 2.501652, 43.199640, 11.004628, 4.015288, 0.772657, 1.299631, 0.427562, 0.904119],
    44: [6.975376, 2.464511, 42.938198, 11.054159, 4.023671, 0.793158, 1.297634, 0.457161, 0.921820],
    45: [6.955591, 2.668204, 42.707371, 10.405193, 3.703211, 0.789962, 1.198266, 0.431998, 0.894515],
    46: [7.009818, 2.573222, 44.832242, 10.866538, 4.017142, 0.772155, 1.222802, 0.435437, 0.908198],
}

# The earlier, independently-scored seed-42 run recovered from the crashed session.
SEED42_REPLICATE = [6.9546, 2.3353, 43.9216, 10.2004, 3.7769, 0.6980, 1.2653, 0.4074, 0.8799]

# CrossFuse TNO sigma (analyse_tier3.py), shown only to compare noise LEVELS between two methods --
# never to judge an LRRNet difference.
# CrossFuse seed sigma: SAMPLE SD (n-1) of three R3-configuration seeds. Until 2026-09-13 this held the
# POPULATION SD (0.816 of the sample value) -- withdrawn claim #13. Computed from the tables, not typed.
SIGMA_CROSSFUSE_TNO = dict(zip(METRICS,
                               [0.0166, 0.0052, 0.3627, 0.2866, 0.0793, 0.0016, 0.0374, 0.0016, 0.0074]))

M = np.array([RUNS[s] for s in sorted(RUNS)])
n = len(M)

print("=" * 94)
print(f"sigma_LRRNet  --  {n} seeds, published configuration, TNO (25 pairs)")
print("=" * 94)
print(f"{'metric':7}{'mean':>11}{'sigma':>10}{'min':>11}{'max':>11}{'range':>10}{'CV %':>8}"
      f"{'vs CrossFuse':>14}")
print("-" * 94)
sigma = {}
for i, m in enumerate(METRICS):
    v = M[:, i]
    sigma[m] = v.std(ddof=1)
    ratio = sigma[m] / SIGMA_CROSSFUSE_TNO[m]
    print(f"{m:7}{v.mean():>11.4f}{sigma[m]:>10.4f}{v.min():>11.4f}{v.max():>11.4f}"
          f"{v.ptp():>10.4f}{100*sigma[m]/abs(v.mean()):>8.2f}{ratio:>13.1f}x")

print(f"\nsigma is a genuine sample standard deviation at n={n} (ddof=1), not a range-based estimate\n"
      "-- CrossFuse's sigma is the sample SD of n=3 (it was the population SD until 2026-09-13; claim #13).")

print(f"\n{'='*94}\nHOW NOISY IS THIS METHOD, RELATIVE TO CROSSFUSE\n{'='*94}")
worst = sorted(METRICS, key=lambda m: -sigma[m] / SIGMA_CROSSFUSE_TNO[m])
print("  LRRNet's seed noise exceeds CrossFuse's on every metric, most severely:")
for m in worst[:4]:
    print(f"    {m:5} {sigma[m]:.4f} vs {SIGMA_CROSSFUSE_TNO[m]:.4f}  "
          f"({sigma[m]/SIGMA_CROSSFUSE_TNO[m]:.0f}x)")
print("""
  Read this as a property of the training setup, not a defect of the method. LRRNet trains for
  4 epochs where CrossFuse trains for 160, on a corpus assembled differently, with a VGG-based
  perceptual loss. Less training means less time for runs to converge on each other. It does mean
  any ablation table built on single LRRNet runs carries a wider noise band than CrossFuse's did.""")

print(f"\n{'='*94}\nA FLOOR THAT REMAINS AFTER THE SEED IS FIXED\n{'='*94}")
print("Two independent runs at seed 42, identical code and configuration:\n")
print(f"{'metric':7}{'run A':>11}{'run B':>11}{'|diff|':>10}{'as % of sigma':>16}")
print("-" * 55)
frac = []
for i, m in enumerate(METRICS):
    a, b = RUNS[42][i], SEED42_REPLICATE[i]
    d = abs(a - b)
    frac.append(d / sigma[m])
    print(f"{m:7}{a:>11.4f}{b:>11.4f}{d:>10.4f}{100*d/sigma[m]:>15.0f}%")

print(f"""
  median |diff| = {np.median(frac)*100:.0f}% of sigma.

  Fixing the seed does NOT make a run reproducible here. Identical weights and identical data
  order still diverge, because cudnn is left free to choose algorithms -- a choice this project
  made deliberately, so that sigma reflects what a reader re-running the released code meets
  rather than a laboratory ideal. The consequence is worth stating plainly in the thesis:

      "set the seed" is not a reproducibility protocol.

  A paper that reported a single run WITH its seed would still not be reproducible to the
  precision its own ablation table claims to resolve.""")

# ----------------------------------------------------------------------------------------------
# Table IV under sigma_LRRNet -- a preliminary reading, by relative noise
# ----------------------------------------------------------------------------------------------
# The absolute levels differ between their numbers and ours (their MI 13.7 vs our 2.50, their SD
# 80 vs our 43.5), because the metric conventions differ and because our models trained on LLVIP.
# Dividing their gaps by our absolute sigma would therefore be meaningless. The coefficient of
# variation -- sigma as a FRACTION of the mean -- does transfer across a level difference, so it
# is applied to their own level. That is an assumption, and stages 2-4 replace it with direct
# measurement of every row. It is stated here, not buried.

TABLE_IV = {                       # published, TNO: blocks -> En, SD, MI, SSIMm, VIFm
    2: [6.79214, 78.25832, 13.58429, 0.71164, 0.66020],
    4: [6.85836, 81.78905, 13.71673, 0.70625, 0.71399],
    6: [6.82875, 80.37536, 13.65750, 0.70726, 0.68716],
    8: [6.83084, 80.24377, 13.66168, 0.70250, 0.69624],
}
THEIR = ["En", "SD", "MI", "SSIMm", "VIFm"]
OURS = {"En": "EN", "SD": "SD", "MI": "MI", "SSIMm": "SSIM", "VIFm": "VIF"}

print(f"\n{'='*94}\nTABLE IV JUDGED BY sigma_LRRNet (relative noise) -- PRELIMINARY\n{'='*94}")
print("The paper: \"When the number of LLRR blocks is set to 4, our fusion network delivers the")
print("best fusion performance (five best values, En, SD, MI, VIFm and Nabf).\"\n")
print(f"{'metric':7}{'CV %':>7}{'their level':>13}{'implied sigma':>15}"
      f"{'4 vs 6':>10}{'4 vs 8':>10}{'4 vs 2':>10}")
print("-" * 82)
inside = total = 0
for j, tm in enumerate(THEIR):
    cv = sigma[OURS[tm]] / abs(M[:, METRICS.index(OURS[tm])].mean())
    level = np.mean([TABLE_IV[b][j] for b in TABLE_IV])
    s = cv * level
    cells = []
    for b in (6, 8, 2):
        u = abs(TABLE_IV[4][j] - TABLE_IV[b][j]) / s
        total += 1
        inside += (u < 3)
        cells.append(f"{u:.2f}s")
    print(f"{tm:7}{100*cv:>7.2f}{level:>13.3f}{s:>15.4f}" + "".join(f"{c:>10}" for c in cells))

print(f"""
  cells inside 3 sigma: {inside} / {total}

READING -- preliminary, and it points one way

  Every comparison between 4 blocks and ANY other setting falls inside the noise band, most of
  them inside 1.2 sigma. The published numbers support "the network works"; they do not support
  "four blocks is the right number", which is what the paper concludes and what it fixes for
  every other experiment in the paper.

  Note how far this moves from the zero-GPU screen. Using CrossFuse's noise level as a stand-in,
  VIF looked resolvable at 16.8 sigma and SD at 3.90. Measured on LRRNet itself, the same gaps are
  0.7 and 0.9 sigma. The proxy was wrong by more than an order of magnitude, and wrong in the
  direction that would have made us understate the problem. That is the argument for measuring
  sigma per method rather than borrowing one -- made here against our own earlier estimate.

  What is still assumed: that relative noise transfers between a KAIST-trained and an
  LLVIP-trained model. Stages 2-4 retrain rows 2, 6 and 8 and remove the assumption entirely.""")
