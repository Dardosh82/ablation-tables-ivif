"""Run 30 -- the same five untrained CrossFuse networks with and without the decoder's visible-input skip.
Readings: ../RUN30_PREREGISTRATION.md (06bacff), committed before any output existed.

For each untrained seed (U42-U46): fuse TNO with the skip ON (the integrity check: it must reproduce the recorded
U-ON means to four decimals) and with the skip OFF (run_inference.py --no-skip, the only change), score both with the
kernel's own evaluate.py, and measure each output's Pearson correlation with the visible input. Writes, under --out:
    tables/<TAG>_<ON|OFF>/TNO_gray_per_image.csv      the evaluator's per-image scores
    resemblance.csv                                   tag, skip, image, r_visible
    integrity.txt                                     the ON-vs-record check
It reads nothing about the verdicts; analyse_run30.py does that, after this run.
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

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="CrossFuse code (network.py, utils/)")
ap.add_argument("--models", required=True, help="folder holding CrossFuse_RSM3_U4x.pth")
ap.add_argument("--tno", required=True, help="TNO folder with ir/ and vi/")
ap.add_argument("--tools", required=True, help="folder with run_inference.py (with --no-skip) and evaluate.py")
ap.add_argument("--recorded", required=True, help="results_poscontrol_cf with tables_U4x_raw/")
ap.add_argument("--out", required=True)
args = ap.parse_args()
M = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
TAGS = ["U42", "U43", "U44", "U45", "U46"]

work = os.path.join(args.out, "work")
wrepo = os.path.join(work, "CrossFuse")
if not os.path.isdir(wrepo):
    os.makedirs(wrepo)
    shutil.copy(os.path.join(args.repo, "network.py"), wrepo)
    shutil.copytree(os.path.join(args.repo, "utils"), os.path.join(wrepo, "utils"))
    shutil.copytree(args.tno, os.path.join(wrepo, "test_image", "TNO"))


def read_gray(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return np.round(cv2.cvtColor(bgr.astype("float32"), cv2.COLOR_BGR2GRAY))


def means(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    assert len(rows) == 25, f"{path}: {len(rows)} rows"
    vals = np.array([[float(r[m]) for m in M] for r in rows])
    assert np.isfinite(vals).all(), f"{path}: non-finite score -- a pipeline failure, never evidence"
    return vals.mean(axis=0)


res = []
integrity = []
for tag in TAGS:
    for skip in ("ON", "OFF"):
        for d in ("test_results", "test_results_gray"):
            shutil.rmtree(os.path.join(wrepo, d), ignore_errors=True)
        cmd = [sys.executable, "-u", os.path.join(args.tools, "run_inference.py"), "--repo", wrepo,
               "--ckpt", os.path.join(args.models, f"CrossFuse_RSM3_{tag}.pth"), "--datasets", "TNO", "--threads", "4"]
        if skip == "OFF":
            cmd.append("--no-skip")
        subprocess.run(cmd, check=True)
        fused = os.path.join(args.out, "fused", f"{tag}_{skip}", "TNO")
        shutil.rmtree(os.path.dirname(fused), ignore_errors=True)
        shutil.copytree(os.path.join(wrepo, "test_results_gray"), os.path.dirname(fused))
        pngs = sorted(glob.glob(os.path.join(fused, "*.png")))
        assert len(pngs) == 25, f"{tag} {skip}: {len(pngs)} fused images"
        table = os.path.join(args.out, "tables", f"{tag}_{skip}")
        subprocess.run([sys.executable, "-u", os.path.join(args.tools, "evaluate.py"), "--repo", wrepo,
                        "--results", os.path.dirname(fused), "--datasets", "TNO", "--out", table], check=True)
        for p in pngs:
            stem = os.path.splitext(os.path.basename(p))[0]
            vi = glob.glob(os.path.join(args.tno, "vi", stem + ".*"))
            assert vi, f"no visible image for {stem}"
            f, v = read_gray(p).ravel(), read_gray(vi[0]).ravel()
            r = float(np.corrcoef(f, v)[0, 1]) if f.std() > 0 else float("nan")
            res.append({"tag": tag, "skip": skip, "image": stem, "r_visible": r})
        if skip == "ON":
            new = means(os.path.join(table, "TNO_gray_per_image.csv"))
            rec = means(os.path.join(args.recorded, f"tables_{tag}_raw", "TNO_gray_per_image.csv"))
            bad = [f"{m} {n:.5f} vs {r:.5f}" for m, n, r in zip(M, new, rec) if abs(n - r) > 5e-5]
            integrity.append(f"{tag} ON reproduces the record to four decimals: {'yes' if not bad else 'NO -- ' + '; '.join(bad)}")
        print(tag, skip, "done", flush=True)

with open(os.path.join(args.out, "resemblance.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["tag", "skip", "image", "r_visible"])
    w.writeheader()
    w.writerows(res)
open(os.path.join(args.out, "integrity.txt"), "w", encoding="utf-8").write("\n".join(integrity) + "\n")
print("\n".join(integrity))
shutil.rmtree(work, ignore_errors=True)
