"""Zero-network baselines: what do the nine metrics give a "fusion" that no model produced at all?

Section 7 explains the MI/VIF inversion on CrossFuse by a mechanism -- an untrained CrossFuse emits
something close to a passthrough of its inputs, and MI and VIF reward agreement with the sources --
and section 9 admits that mechanism is post-hoc. An external review asked for it to be measured
rather than argued. This is the cheapest measurement that can contradict it: no training, no
checkpoint, no GPU. Four fixed images per TNO pair, scored by the same Evaluator on the same 25 pairs
as every trained and untrained row in the paper.

    mean   (IR + VI) / 2      the literal passthrough
    max    max(IR, VI)        the classical hand-crafted rule
    ir     IR alone           no fusion, one source copied
    vi     VI alone           no fusion, the other source copied

READINGS, WRITTEN BEFORE THIS SCRIPT WAS FIRST RUN
  (The script and its first output were committed together, in 785b024, so git cannot show this
  order. It is the author's statement, not a verifiable record. Future readings are committed on
  their own before the run, as RUN9_PREREGISTRATION.md was.)

  SUPPORTS the mechanism: `mean` scores MI and VIF at or above TRAINED CrossFuse (2.30, 0.784), and
      its SF and AG sit BELOW trained CrossFuse (14.45, 5.32) -- a smooth blend that the information
      metrics reward and the gradient metrics do not. Reported as: the inversion is reproduced by an
      image no network produced, so it is a property of the metrics.
  CONTRADICTS the mechanism: `mean` scores MI or VIF BELOW trained CrossFuse. Then "an untrained
      network is close to a passthrough, and passthroughs are rewarded" is false as stated, and
      section 7's explanation is withdrawn and replaced by "demonstrated, not explained".
  EITHER WAY, reported: whether `ir` or `vi` alone -- no fusion at all -- beats a trained network on
      any metric. That would be a stronger statement than the inversion itself, and is not assumed.

Comparison values are transcribed from 05_RESULTS.md Run 7 (CrossFuse, n=5 per arm) and Run 6
(LRRNet trained arm, n=5). They are means over seeds; the baselines here are deterministic, so they
have no sigma and are compared against the trained arm's mean and its seed spread.

    python studied_papers/CrossFuse/Tier3_CrossFuse/scripts/passthrough_baseline.py
"""
import argparse
import os
import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CF_ROOT = os.path.dirname(os.path.dirname(HERE))                     # studied_papers/CrossFuse
ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.join(CF_ROOT, "Tier2_CrossFuse", "CrossFuse"),
                help="CrossFuse repo holding utils/Evaluator.py -- the evaluator Kaggle ran")
ap.add_argument("--tno", default=os.path.join(CF_ROOT, "data set", "crossfuse-testsets", "TNO"),
                help="TNO test set with ir/ and vi/ subfolders, 25 pairs")
args = ap.parse_args()
REPO, TNO = args.repo, args.tno

sys.path.insert(0, REPO)
from utils.Evaluator import Evaluator                                   # noqa: E402

METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]

# 05_RESULTS.md, Run 7 (CrossFuse) and Run 6 (LRRNet), mean and sample SD over five seeds
REF = {
    "CrossFuse trained":   ([7.1787, 2.2967, 49.3629, 14.4548, 5.3203, 0.7843, 1.5420, 0.5396, 0.9524],
                            [0.0360, 0.0280, 1.3853, 0.3199, 0.1609, 0.0083, 0.0504, 0.0082, 0.0117]),
    "CrossFuse untrained": ([7.0140, 3.4738, 46.4051, 10.7923, 3.9354, 0.9450, 0.9278, 0.5258, 0.9041],
                            [0.0329, 0.1795, 0.9611, 0.3859, 0.1285, 0.0260, 0.2198, 0.0041, 0.0196]),
    "LRRNet trained":      ([6.9862, 2.4728, 42.8841, 10.9088, 4.0585, 0.7865, 1.3171, 0.4464, 0.9235],
                            [0.0269, 0.0350, 0.9908, 0.2980, 0.1049, 0.0063, 0.0407, 0.0072, 0.0076]),
}


def read_gray(path):
    """Identical to eval_fused.read_gray, so these rows are on the evaluator's own scale."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return np.round(cv2.cvtColor(bgr.astype("float32"), cv2.COLOR_BGR2GRAY))


def score(F, A, B):
    return [float(Evaluator.EN(F)), float(Evaluator.MI(F, A, B)), float(Evaluator.SD(F)),
            float(Evaluator.SF(F)), float(Evaluator.AG(F)), float(Evaluator.VIFF(F, A, B)),
            float(Evaluator.SCD(F, A, B)), float(Evaluator.Qabf(F, A, B)),
            float(Evaluator.SSIM(F, A, B))]


names = sorted(os.listdir(os.path.join(TNO, "ir")))
assert len(names) == 25, f"expected the 25 TNO pairs every row in the paper uses, found {len(names)}"
assert names == sorted(os.listdir(os.path.join(TNO, "vi"))), "TNO ir/vi filenames differ"

RULES = {"mean": lambda a, b: np.round((a + b) / 2),
         "max": np.maximum,
         "ir": lambda a, b: a,
         "vi": lambda a, b: b}
rows = {k: [] for k in RULES}
for n in names:
    A = read_gray(os.path.join(TNO, "ir", n))
    B = read_gray(os.path.join(TNO, "vi", n))
    assert A.shape == B.shape, f"{n}: ir {A.shape} vs vi {B.shape}"
    for k, rule in RULES.items():
        F = np.clip(rule(A, B), 0, 255)
        vals = score(F, A, B)
        # SCD = corr(F - B, A) + corr(F - A, B). When F IS one of the sources, one difference image
        # is identically zero and its correlation is 0/0: undefined by the formula, not a pipeline
        # fault. Found on the first run of this script; it is the only exemption, and only for the
        # two copy baselines. Every other non-finite value still stops the analysis.
        if k in ("ir", "vi") and not np.isfinite(vals[METRICS.index("SCD")]):
            vals[METRICS.index("SCD")] = float("nan")
            if all(np.isfinite(v) for i, v in enumerate(vals) if METRICS[i] != "SCD"):
                rows[k].append(vals)
                continue
        if not all(np.isfinite(vals)):
            sys.exit(f"FAIL: {k} on {n} gave a non-finite metric {vals}. A broken evaluation is not a "
                     f"baseline; nothing is reported.")
        rows[k].append(vals)
base = {k: np.mean(v, axis=0) for k, v in rows.items()}

print("=" * 104)
print("ZERO-NETWORK BASELINES on the 25 TNO pairs -- same Evaluator, same grayscale read")
print("=" * 104)
print(f"{'metric':7}" + "".join(f"{k:>11}" for k in RULES)
      + "".join(f"{k:>22}" for k in REF))
print("-" * 104)
for i, m in enumerate(METRICS):
    print(f"{m:7}" + "".join(f"{base[k][i]:>11.4f}" for k in RULES)
          + "".join(f"{REF[r][0][i]:>13.4f} ±{REF[r][1][i]:<7.4f}" for r in REF))

cf_mu, cf_sd = (np.array(v) for v in REF["CrossFuse trained"])
lr_mu, _ = (np.array(v) for v in REF["LRRNet trained"])
iMI, iVIF, iSF, iAG = (METRICS.index(m) for m in ("MI", "VIF", "SF", "AG"))
mean = base["mean"]

print(f"\n{'=' * 104}\nREADING (as written in this file's docstring -- see the note there on what git can show)\n{'=' * 104}")
info_up = mean[iMI] >= cf_mu[iMI] and mean[iVIF] >= cf_mu[iVIF]
grad_down = mean[iSF] < cf_mu[iSF] and mean[iAG] < cf_mu[iAG]
print(f"  mean vs trained CrossFuse:  MI {mean[iMI]:.3f} vs {cf_mu[iMI]:.3f} "
      f"({(mean[iMI] - cf_mu[iMI]) / cf_sd[iMI]:+.1f} seed-SD) · VIF {mean[iVIF]:.3f} vs {cf_mu[iVIF]:.3f} "
      f"({(mean[iVIF] - cf_mu[iVIF]) / cf_sd[iVIF]:+.1f} seed-SD)")
print(f"                              SF {mean[iSF]:.2f} vs {cf_mu[iSF]:.2f} · AG {mean[iAG]:.2f} vs {cf_mu[iAG]:.2f}")
if info_up and grad_down:
    print("  SUPPORTS: a plain average of the sources, which no network produced, scores MI and VIF at\n"
          "  or above a trained CrossFuse and SF/AG below it. The inversion is a property of the metrics.")
elif not (mean[iMI] >= cf_mu[iMI] or mean[iVIF] >= cf_mu[iVIF]):
    print("  CONTRADICTS: the average scores MI and VIF below trained CrossFuse. Section 7's mechanism is\n"
          "  withdrawn; the inversion stands as demonstrated, not explained.")
else:
    print("  MIXED: the mechanism is supported on one information metric and not the other. Report per\n"
          "  metric; do not state the mechanism as established.")

print("\n  Metrics on which a baseline with NO network beats the trained network's mean:")
for k in RULES:
    for label, mu in (("CrossFuse", cf_mu), ("LRRNet", lr_mu)):
        defined = [i for i in range(9) if np.isfinite(base[k][i])]
        wins = [METRICS[i] for i in defined if base[k][i] > mu[i]]
        print(f"    {k:>4} beats trained {label:9}: {len(wins)} of {len(defined)}  "
              f"{', '.join(wins) if wins else '-'}")
print("\n  No sigma exists for a deterministic baseline, so none of these is a significance test. A win\n"
      "  is reported as a count against a mean, with the trained arm's seed spread printed beside it.")
