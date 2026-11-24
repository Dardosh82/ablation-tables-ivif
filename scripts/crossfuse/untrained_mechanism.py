"""Why does an untrained CrossFuse score higher MI and VIF than a trained one? Measured, not argued.

Withdrawn claim #12 left this open. The zero-network baselines (passthrough_baseline.py) ruled out an
AVERAGING passthrough -- (IR+VI)/2 scores MI 1.60 / VIF 0.57, far below trained CrossFuse -- and showed
that a source copied unchanged scores above every network (VI: MI 5.16 / VIF 1.04). They could not say
what the untrained network itself emits. This script fuses the 25 TNO pairs with the same five
untrained checkpoints (U42-U46) and two trained ones (PC45, PC46) that the paper's §7 table scored,
and measures which zero-network image each output most resembles.

WHY A SPECIFIC HYPOTHESIS EXISTS BEFORE ANY OUTPUT DOES
    CrossFuse's decoder ends  sigmoid(conv(features) + inp_img),  and test.py passes the VISIBLE
    image as inp_img (network.py, Restormer_Decoder.forward). The visible band is added back after the
    learned branch. With random weights the learned branch contributes structured noise, so the
    output should be close to a monotone transform of the visible image: a source-SELECTING
    passthrough, delivered by a residual connection, not a blend. That is read from the code; whether
    it holds is what this run measures.

READINGS -- WRITTEN AND COMMITTED ON THEIR OWN, BEFORE THIS SCRIPT IS FIRST RUN

  Decision statistic: Pearson r over all pixels between the network's fused output F (the grayscale
  image the evaluator scores) and each zero-network candidate, averaged over the 25 pairs, per seed.
  Candidates: VI, IR, MEAN = (IR+VI)/2, MAX = max(IR, VI).

  K1 KILL -- local inference must reproduce the kernel. For every checkpoint, the MI and VIF this script
     computes on its own fused images must match the kernel's per-image values
     (results_poscontrol_cf/tables_<TAG>_raw/TNO_gray_per_image.csv) with median absolute relative
     difference below 1 % on both. If any checkpoint fails, NOTHING below is read: the images would not
     be the images the paper scored.

  H1 SELECTS VISIBLE -- in at least 4 of the 5 untrained seeds, VI is the candidate with the highest mean
     r, and that r is at least 0.90. Reported as: an untrained CrossFuse emits close to a copy of its
     visible input, through the decoder's residual connection; since a copied visible image is exactly
     what MI and VIF score highest, that is the mechanism of the inversion. §7's "not established" is
     replaced by this, stated as a measured resemblance plus an architectural cause, not as proof.
  H2 SELECTS MAXIMUM -- the same rule with MAX in place of VI. Reported as a pixel-wise selection
     mechanism, and the residual-connection reading is NOT used.
  H3 NEITHER -- no candidate wins in 4 of 5 seeds, or the winner's mean r is below 0.90. §7's
     "not established" stands, and the correlations are reported as what was measured.

  Reported either way, and not used to choose between H1-H3:
    * the trained checkpoints' r against the same candidates -- if H1 holds, a trained network should
      resemble the visible image less than an untrained one does;
    * SKIP = min-max(sigmoid(VI_Y)), the network with its learned branch removed entirely, as an
      upper reference for how far a pure residual path would go;
    * SSIM (data_range 255) beside every r, as a second view that is not read.

    python studied_papers/CrossFuse/Tier3_CrossFuse/scripts/untrained_mechanism.py
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
T3 = os.path.dirname(HERE)                                             # studied_papers/CrossFuse/Tier3_CrossFuse
CF = os.path.dirname(T3)                                               # studied_papers/CrossFuse

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.join(T3, "results_poscontrol_cf_rawdownload", "CrossFuse"),
                help="CrossFuse code the kernel ran (network.py, utils/); read, never written")
ap.add_argument("--models", default=os.path.join(T3, "results_poscontrol_cf_rawdownload", "CrossFuse", "models"))
ap.add_argument("--tno", default=os.path.join(CF, "data set", "crossfuse-testsets", "TNO"))
ap.add_argument("--kernel_tables", default=os.path.join(T3, "results_poscontrol_cf"))
ap.add_argument("--out", default=os.path.join(T3, "results_mechanism_cf"))
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()

TAGS = ["U42", "U43", "U44", "U45", "U46", "PC45", "PC46"]
UNTRAINED = [t for t in TAGS if t.startswith("U")]
CANDS = ["VI", "IR", "MEAN", "MAX"]
TOOLS = os.path.join(CF, "Tier1_CrossFuse", "scripts")                # the kernel's own run_inference/evaluate

# ---- a scratch copy of the repo, so the downloaded record is never written to -----------------------
work = os.path.join(args.out, "work")
wrepo = os.path.join(work, "CrossFuse")
if not os.path.isdir(wrepo):
    os.makedirs(wrepo)
    shutil.copy(os.path.join(args.repo, "network.py"), wrepo)
    shutil.copytree(os.path.join(args.repo, "utils"), os.path.join(wrepo, "utils"))
    shutil.copytree(args.tno, os.path.join(wrepo, "test_image", "TNO"))


def read_gray(path):
    """Identical to evaluate.py / eval_fused.read_gray."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return np.round(cv2.cvtColor(bgr.astype("float32"), cv2.COLOR_BGR2GRAY))


# ---- 1. fuse and score every checkpoint with the kernel's own scripts ------------------------------
for tag in TAGS:
    fused = os.path.join(args.out, "fused", tag, "TNO")
    table = os.path.join(args.out, "tables", tag, "TNO_gray_per_image.csv")
    if len(glob.glob(os.path.join(fused, "*.png"))) != 25:
        # Wiped when the checkpoint CHANGES, as the kernel did -- so no image from one checkpoint is scored
        # as another's. Not wiped on a restart of the same checkpoint: run_inference.py skips images already
        # fused, and at ~90 s per image on CPU a run interrupted near the end would otherwise start over.
        marker = os.path.join(work, "current_tag.txt")
        if not os.path.exists(marker) or open(marker).read().strip() != tag:
            for d in ("test_results", "test_results_gray"):
                shutil.rmtree(os.path.join(wrepo, d), ignore_errors=True)
            with open(marker, "w") as mf:
                mf.write(tag)
        subprocess.run([sys.executable, "-u", os.path.join(TOOLS, "run_inference.py"), "--repo", wrepo,
                        "--ckpt", os.path.join(args.models, f"CrossFuse_RSM3_{tag}.pth"),
                        "--datasets", "TNO", "--threads", str(args.threads)], check=True)
        shutil.rmtree(os.path.dirname(fused), ignore_errors=True)
        shutil.copytree(os.path.join(wrepo, "test_results_gray"), os.path.dirname(fused))
    assert len(glob.glob(os.path.join(fused, "*.png"))) == 25, f"{tag}: expected 25 fused images"
    if not os.path.exists(table):
        subprocess.run([sys.executable, "-u", os.path.join(TOOLS, "evaluate.py"), "--repo", wrepo,
                        "--results", os.path.dirname(fused), "--datasets", "TNO",
                        "--out", os.path.dirname(table)], check=True)


def per_image(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["image"]: r for r in csv.DictReader(f)}


# ---- 2. K1: local reproduction of the kernel ------------------------------------------------------
print("=" * 100)
print("K1 -- does local inference reproduce the kernel's per-image MI and VIF?")
print("=" * 100)
k1_ok = True
for tag in TAGS:
    mine = per_image(os.path.join(args.out, "tables", tag, "TNO_gray_per_image.csv"))
    kern = per_image(os.path.join(args.kernel_tables, f"tables_{tag}_raw", "TNO_gray_per_image.csv"))
    assert sorted(mine) == sorted(kern) and len(mine) == 25, f"{tag}: image sets differ"
    med = {}
    for m in ("MI", "VIF"):
        rel = [abs(float(mine[k][m]) - float(kern[k][m])) / abs(float(kern[k][m])) for k in kern]
        med[m] = float(np.median(rel))
    ok = med["MI"] < 0.01 and med["VIF"] < 0.01
    k1_ok &= ok
    print(f"  {tag:5} median |rel diff|  MI {100 * med['MI']:.3f} %   VIF {100 * med['VIF']:.3f} %   "
          f"{'OK' if ok else 'FAIL'}")
if not k1_ok:
    print("\nK1 FAILED: the local images are not the images the paper scored. Nothing below is read.")
    sys.exit(1)

# ---- 3. resemblance to the zero-network candidates -------------------------------------------------
names = sorted(os.listdir(os.path.join(args.tno, "ir")))
src = {}
for n in names:
    A = read_gray(os.path.join(args.tno, "ir", n))
    B = read_gray(os.path.join(args.tno, "vi", n))
    y = cv2.cvtColor(cv2.imread(os.path.join(args.tno, "vi", n)).astype("float32"), cv2.COLOR_BGR2YCrCb)[..., 0] / 255.0
    s = 1.0 / (1.0 + np.exp(-y))
    skip = np.round((s - s.min()) / (s.max() - s.min()) * 255.0)
    src[n] = {"VI": B, "IR": A, "MEAN": np.round((A + B) / 2), "MAX": np.maximum(A, B), "SKIP": skip}


def pearson(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


rows, summary = [], {}
for tag in TAGS:
    rs = {c: [] for c in CANDS + ["SKIP"]}
    ss = {c: [] for c in CANDS + ["SKIP"]}
    for n in names:
        F = read_gray(os.path.join(args.out, "fused", tag, "TNO", os.path.splitext(n)[0] + ".png"))
        for c in CANDS + ["SKIP"]:
            G = src[n][c]
            r = pearson(F, G)
            s_ = float(structural_similarity(F, G, data_range=255))
            rs[c].append(r)
            ss[c].append(s_)
            rows.append({"tag": tag, "image": os.path.splitext(n)[0], "candidate": c, "r": r, "ssim": s_})
    summary[tag] = {c: (float(np.mean(rs[c])), float(np.mean(ss[c]))) for c in rs}

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "resemblance_per_image.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["tag", "image", "candidate", "r", "ssim"])
    w.writeheader()
    w.writerows(rows)

print(f"\n{'=' * 100}\nMEAN PEARSON r (SSIM in brackets) between each network's output and each candidate\n{'=' * 100}")
print(f"{'tag':6}" + "".join(f"{c:>19}" for c in CANDS + ["SKIP"]) + "   winner (of VI/IR/MEAN/MAX)")
winners = {}
for tag in TAGS:
    best = max(CANDS, key=lambda c: summary[tag][c][0])
    winners[tag] = best
    print(f"{tag:6}" + "".join(f"{summary[tag][c][0]:>10.3f} ({summary[tag][c][1]:.3f})" for c in CANDS + ["SKIP"])
          + f"   {best}")

# ---- 4. the registered reading --------------------------------------------------------------------
print(f"\n{'=' * 100}\nREADING (registered in this file's docstring, committed before the first run)\n{'=' * 100}")
votes = {c: sum(1 for t in UNTRAINED if winners[t] == c) for c in CANDS}
top = max(votes, key=votes.get)
mean_r = float(np.mean([summary[t][top][0] for t in UNTRAINED if winners[t] == top])) if votes[top] else 0.0
print(f"  untrained winners: {votes}   leading candidate {top}, mean r where it wins {mean_r:.3f}")
if votes[top] >= 4 and mean_r >= 0.90 and top == "VI":
    print("  H1 SELECTS VISIBLE: an untrained CrossFuse emits close to a copy of its visible input.")
elif votes[top] >= 4 and mean_r >= 0.90 and top == "MAX":
    print("  H2 SELECTS MAXIMUM: an untrained CrossFuse behaves like a pixel-wise maximum.")
else:
    print("  H3 NEITHER: no candidate meets the registered rule; 'not established' stands.")
tr_vi = [summary[t]["VI"][0] for t in TAGS if t.startswith("PC")]
un_vi = [summary[t]["VI"][0] for t in UNTRAINED]
print(f"  reference, not used to decide: mean r to VI -- untrained {np.mean(un_vi):.3f}, trained {np.mean(tr_vi):.3f}")
