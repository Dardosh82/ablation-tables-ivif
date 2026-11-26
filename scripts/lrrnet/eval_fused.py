"""Nine-metric evaluation of a folder of fused images, using the SAME evaluator as CrossFuse.

One evaluator for every row of the re-measured Table IV is the whole point of re-running the
ablation rather than screening the published one: LRRNet reports MI = 13.7 where this evaluator
measures ~2.1 for the same metric name, so their numbers and ours are not on a common scale and
cannot be compared. Ours can only be compared with ours.

Metric implementations come from CrossFuse's `utils/Evaluator.py` -- the authors' own code, already
used for every CrossFuse number in this project.

Must run as its own process, NOT inside the LRRNet repo: LRRNet has a top-level `utils.py` that
shadows CrossFuse's `utils` package, so the import would silently resolve to the wrong module.

    python eval_fused.py --crossfuse <path to CrossFuse repo> --ir ... --vi ... --fused ... \
                         --tag seed42 --csv results.csv
"""
import argparse
import csv
import os
import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]

ap = argparse.ArgumentParser()
ap.add_argument("--crossfuse", required=True, help="CrossFuse repo root (holds utils/Evaluator.py)")
ap.add_argument("--ir", required=True)
ap.add_argument("--vi", required=True)
ap.add_argument("--fused", required=True)
ap.add_argument("--tag", required=True, help="row label, e.g. nb4_seed42")
ap.add_argument("--csv", required=True, help="CSV to append one row to")
# Run 29 (RUN29_PREREGISTRATION.md, 2026-11-25): also keep one row per image, for the runs-by-images bootstrap.
# The mean row above is written exactly as before and is the integrity check against the record.
ap.add_argument("--per-image", default=None, help="optional CSV to append one row per image to (tag, image, metrics)")
a = ap.parse_args()

# Python puts THIS SCRIPT'S OWN DIRECTORY at sys.path[0] -- not the working directory. When the
# script lives in the LRRNet repo, that repo's top-level utils.py shadows CrossFuse's utils
# package and the import fails with "'utils' is not a package". Checking os.getcwd() does not
# catch it, because cwd is irrelevant here. Remove the script's directory from the path, drop any
# utils already imported, then put CrossFuse first.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.modules.pop("utils", None)
sys.path.insert(0, os.path.abspath(a.crossfuse))

from utils.Evaluator import Evaluator          # noqa: E402

# Check __path__, not __file__: CrossFuse's utils/ has no __init__.py, so it resolves as a
# namespace package whose __file__ is None. A package has __path__; LRRNet's plain utils.py
# does not -- which is exactly the distinction that matters here.
_paths = [os.path.abspath(p) for p in getattr(sys.modules["utils"], "__path__", [])]
assert _paths and all(p.startswith(os.path.abspath(a.crossfuse)) for p in _paths), \
    f"resolved the wrong 'utils': {_paths or 'a plain module, not a package'}"
print(f"evaluator package: {_paths[0]}")


def read_gray(path):
    """0-255 float32 grayscale, matching Evaluator.image_read_cv2(mode='GRAY')."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    return np.round(cv2.cvtColor(bgr.astype("float32"), cv2.COLOR_BGR2GRAY))


rows = []
names = sorted(os.listdir(a.ir))
for name in names:
    stem = os.path.splitext(name)[0]
    fused = os.path.join(a.fused, stem + ".png")
    if not os.path.exists(fused):
        raise SystemExit(f"missing fused image for {stem} at {fused}")
    F = read_gray(fused)
    A = read_gray(os.path.join(a.ir, name))     # infrared
    B = read_gray(os.path.join(a.vi, name))     # visible
    rows.append({
        "image": stem,
        "EN":   float(Evaluator.EN(F)),
        "MI":   float(Evaluator.MI(F, A, B)),
        "SD":   float(Evaluator.SD(F)),
        "SF":   float(Evaluator.SF(F)),
        "AG":   float(Evaluator.AG(F)),
        "VIF":  float(Evaluator.VIFF(F, A, B)),
        "SCD":  float(Evaluator.SCD(F, A, B)),
        "Qabf": float(Evaluator.Qabf(F, A, B)),
        "SSIM": float(Evaluator.SSIM(F, A, B)),
    })

assert rows, "no images evaluated"
mean = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}

print(f"\n{a.tag}  ({len(rows)} image pairs)")
print("  " + "".join(f"{m:>10}" for m in METRICS))
print("  " + "".join(f"{mean[m]:>10.4f}" for m in METRICS))

new = not os.path.exists(a.csv)
with open(a.csv, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["tag", "n"] + METRICS)
    w.writerow([a.tag, len(rows)] + [f"{mean[m]:.6f}" for m in METRICS])
print(f"appended to {a.csv}")

if a.per_image:
    new = not os.path.exists(a.per_image)
    with open(a.per_image, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "image"] + METRICS)
        for r in rows:
            w.writerow([a.tag, r["image"]] + [f"{r[m]:.9g}" for m in METRICS])
    print(f"appended {len(rows)} per-image rows to {a.per_image}")
