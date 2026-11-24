"""Does the residual-copy mechanism transfer? An untrained CDDFuse against the released one. No training.

On CrossFuse, an untrained network's output matched its visible input at r = 0.997 (mechanism run, H1),
because the decoder ends sigmoid(conv(features) + inp_img) with the visible image as inp_img -- and MI and
VIF score a copied visible image above any trained network. CrossFuse inherits that decoder from CDDFuse.
If the account is architectural rather than a CrossFuse accident, CDDFuse must show the same thing:
its decoder is identical (net.py, Restormer_Decoder.forward) and its test_IVF.py passes data_VIS the same
way. LRRNet, which has no such skip, already showed no MI/VIF inversion -- the contrast case exists.

This uses only public material: CDDFuse's repository at a pinned commit, its released IVF checkpoint, its
own test_IVF.py preprocessing and utils/Evaluator.py, and its own 25 TNO pairs.

READINGS -- WRITTEN AND COMMITTED ON THEIR OWN, BEFORE THIS SCRIPT IS FIRST RUN

  Arms. TRAINED: models/CDDFuse_IVF.pth, as released. UNTRAINED: the four modules built in train.py's
  construction order (lines 56-59) after torch/numpy/random seeding with 42, 43, 44, 45, 46 -- no weight
  loaded, no gradient step. Every image is fused exactly as test_IVF.py does it (gray VIS and IR, min-max
  normalised output) and scored by CDDFuse's own Evaluator.

  K1 KILL -- the released checkpoint must reproduce CDDFuse's published TNO row. The README prints
     EN 7.12, SD 46.0, SF 13.15, MI 2.19, SCD 1.76, VIF 0.77, Qabf 0.54, SSIM 1.03. Our means over the 25
     pairs must match within 0.02 on each (0.5 on SD). If not, NOTHING below is read.

  Copy criterion: Pearson r between the fused output and each of VI, IR, (IR+VI)/2, max(IR,VI), averaged
  over the 25 pairs; VI must be the most-resembled in at least 4 of 5 untrained seeds with that mean r >= 0.90.
  Inversion criterion: the untrained mean over 5 seeds exceeds the released value on BOTH MI and VIF, each by
  a one-sample t-test across the 5 seeds with |t| > 2.776 (two-sided 5 %, df = 4).

  H1 TRANSFERS -- copy AND inversion. Reported as: the mechanism is a property of this decoder design, found
     in two published methods; LRRNet, without the skip, does not invert. §7's scope sentence is widened from
     "a decoder that adds one modality back" on one network to two.
  H2 COPY WITHOUT INVERSION -- the output copies VI but MI or VIF does not rank it higher. Reported as: the
     skip produces the copy, but the copy is not sufficient for the inversion on CDDFuse; §7 keeps the
     mechanism scoped to CrossFuse and says why.
  H3 NO COPY -- the copy criterion fails. Reported as: the residual-skip account does not transfer even to the
     method whose decoder CrossFuse reuses; §7 states it as CrossFuse-specific.

  Reported either way, not used to decide: all nine metrics' t, the trained checkpoint's own r to VI.

    python studied_papers/CDDFuse/scripts/cddfuse_untrained_mechanism.py --repo <MMIF-CDDFuse checkout>
"""
import argparse
import csv
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True, help="MMIF-CDDFuse at commit 73681d28f6100461f1b214ecf77a93b495be227b")
ap.add_argument("--out", required=True)
args = ap.parse_args()
REPO = os.path.abspath(args.repo)
os.makedirs(args.out, exist_ok=True)
sys.path.insert(0, REPO)
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction  # noqa: E402
from utils.Evaluator import Evaluator                                                                    # noqa: E402
from utils.img_read_save import image_read_cv2                                                           # noqa: E402
import utils.Evaluator as _ev                                                                             # noqa: E402
from skimage.metrics import structural_similarity as _ssim                                               # noqa: E402

# Fixed 2026-09-13 after the second push crashed in Evaluator.SSIM, before any metric was read. CDDFuse pins
# scikit-image 0.19.2, whose structural_similarity takes data_range from the dtype when it is not given: for
# float images dtype_range is (-1, 1), so data_range = 2. Current skimage raises instead. This restores the
# pinned default exactly; K1 decides whether the pipeline reproduces the published row.
_DTYPE_RANGE = {np.dtype("float16"): 2.0, np.dtype("float32"): 2.0, np.dtype("float64"): 2.0, np.dtype("uint8"): 255.0}


def _ssim_0_19_2(im1, im2, **kw):
    kw.setdefault("data_range", _DTYPE_RANGE[np.asarray(im1).dtype])
    return _ssim(im1, im2, **kw)


_ev.ssim = _ssim_0_19_2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TNO = os.path.join(REPO, "test_img", "TNO")
NAMES = sorted(os.listdir(os.path.join(TNO, "ir")))
assert len(NAMES) == 25, f"expected CDDFuse's 25 TNO pairs, found {len(NAMES)}"
METRICS = ["EN", "SD", "SF", "MI", "SCD", "VIF", "Qabf", "SSIM"]           # test_IVF.py's order
PUBLISHED = {"EN": 7.12, "SD": 46.0, "SF": 13.15, "MI": 2.19, "SCD": 1.76, "VIF": 0.77, "Qabf": 0.54, "SSIM": 1.03}
SEEDS = [42, 43, 44, 45, 46]
print("device", DEVICE, "| torch", torch.__version__)


def build():
    enc = nn.DataParallel(Restormer_Encoder()).to(DEVICE)                     # train.py lines 56-59, in order
    dec = nn.DataParallel(Restormer_Decoder()).to(DEVICE)
    base = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(DEVICE)
    detail = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(DEVICE)
    return enc, dec, base, detail


def fuse_all(mods):
    enc, dec, base, detail = mods
    for m in mods:
        m.eval()
    out = {}
    with torch.no_grad():
        for n in NAMES:                                                        # test_IVF.py, line for line
            ir = image_read_cv2(os.path.join(TNO, "ir", n), mode="GRAY")[np.newaxis, np.newaxis, ...] / 255.0
            vi = image_read_cv2(os.path.join(TNO, "vi", n), mode="GRAY")[np.newaxis, np.newaxis, ...] / 255.0
            ir, vi = torch.FloatTensor(ir).to(DEVICE), torch.FloatTensor(vi).to(DEVICE)
            fvb, fvd, _ = enc(vi)
            fib, fid, _ = enc(ir)
            fused, _ = dec(vi, base(fvb + fib), detail(fvd + fid))
            lo, hi = torch.min(fused), torch.max(fused)
            fused = (fused - lo) / (hi - lo).clamp_min(1e-8)
            fi = np.squeeze((fused * 255).cpu().numpy())
            # test_IVF.py saves this as a uint8 PNG and scores the re-read file; reproduce that rounding
            out[n] = np.round(cv2_roundtrip(fi))
    return out


def cv2_roundtrip(fi):
    # Fixed 2026-09-13 after the first push crashed before any metric was computed (nothing was read).
    # test_IVF.py saves with skimage 0.19.2 (its requirements.txt), which hands a float array to imageio 2.x;
    # imageio's image_as_uint min-max rescales to 0..255 and adds 0.499999999 before the uint8 cast, i.e. it
    # rounds. The first version truncated. Modern skimage refuses to write a float PNG, so this is reproduced.
    fi = fi.astype(np.float64)
    lo, hi = np.nanmin(fi), np.nanmax(fi)
    fi = (fi - lo) / max(hi - lo, 1e-12) * 255.0 + 0.499999999
    return fi.astype(np.uint8).astype("float32")


def pearson(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float((a @ b) / math.sqrt((a @ a) * (b @ b)))


src = {n: (image_read_cv2(os.path.join(TNO, "ir", n), "GRAY"), image_read_cv2(os.path.join(TNO, "vi", n), "GRAY"))
       for n in NAMES}


def score(fused):
    rows, res = [], {c: [] for c in ("VI", "IR", "MEAN", "MAX")}
    for n in NAMES:
        ir, vi = src[n]
        fi = fused[n]
        rows.append([float(Evaluator.EN(fi)), float(Evaluator.SD(fi)), float(Evaluator.SF(fi)),
                     float(Evaluator.MI(fi, ir, vi)), float(Evaluator.SCD(fi, ir, vi)),
                     float(Evaluator.VIFF(fi, ir, vi)), float(Evaluator.Qabf(fi, ir, vi)),
                     float(Evaluator.SSIM(fi, ir, vi))])
        for c, g in (("VI", vi), ("IR", ir), ("MEAN", np.round((ir + vi) / 2)), ("MAX", np.maximum(ir, vi))):
            res[c].append(pearson(fi, g))
    return np.mean(rows, axis=0), {c: float(np.mean(v)) for c, v in res.items()}, rows


results = {}
ckpt = torch.load(os.path.join(REPO, "models", "CDDFuse_IVF.pth"), map_location=DEVICE)
mods = build()
# The released file names its first two modules DIDF_Encoder / DIDF_Decoder (inherited from DIDFuse), while
# test_IVF.py at this commit asks for CDDF_*; the first push died on that KeyError. Accept either spelling.
for m, key in zip(mods, ("CDDF_Encoder", "CDDF_Decoder", "BaseFuseLayer", "DetailFuseLayer")):
    m.load_state_dict(ckpt[key] if key in ckpt else ckpt[key.replace("CDDF_", "DIDF_")])
results["trained"] = score(fuse_all(mods))
for s in SEEDS:
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    results[f"U{s}"] = score(fuse_all(build()))

with open(os.path.join(args.out, "cddfuse_mechanism_summary.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["arm"] + METRICS + ["r_VI", "r_IR", "r_MEAN", "r_MAX"])
    for arm, (m, r, _) in results.items():
        w.writerow([arm] + [f"{x:.6f}" for x in m] + [f"{r[c]:.6f}" for c in ("VI", "IR", "MEAN", "MAX")])
print("wrote", os.path.join(args.out, "cddfuse_mechanism_summary.csv"))

print("\n" + "=" * 100 + "\nK1 -- does the released checkpoint reproduce CDDFuse's published TNO row?\n" + "=" * 100)
tm = dict(zip(METRICS, results["trained"][0]))
k1 = True
for k in METRICS:
    tol = 0.5 if k == "SD" else 0.02
    ok = abs(tm[k] - PUBLISHED[k]) <= tol
    k1 &= ok
    print(f"  {k:5} ours {tm[k]:8.4f}   published {PUBLISHED[k]:6.2f}   {'OK' if ok else 'FAIL'}")
if not k1:
    print("\nK1 FAILED: our pipeline does not reproduce the released model. Nothing below is read.")
    sys.exit(1)

print("\n" + "=" * 100 + "\nRESEMBLANCE (mean Pearson r over 25 pairs)\n" + "=" * 100)
winners = {}
for arm, (_, r, _) in results.items():
    winners[arm] = max(r, key=r.get)
    print(f"  {arm:8} " + "  ".join(f"{c} {r[c]:.3f}" for c in ("VI", "IR", "MEAN", "MAX")) + f"   winner {winners[arm]}")

print("\n" + "=" * 100 + "\nUNTRAINED (mean of 5 seeds) vs RELEASED -- one-sample t, df = 4\n" + "=" * 100)
ut = np.array([results[f"U{s}"][0] for s in SEEDS])
tstat = {}
for i, k in enumerate(METRICS):
    col = ut[:, i]
    sd = col.std(ddof=1)
    t = (col.mean() - tm[k]) / (sd / math.sqrt(len(col))) if sd > 0 else float("inf")
    tstat[k] = t
    print(f"  {k:5} untrained {col.mean():8.4f} +/-{sd:.4f}   released {tm[k]:8.4f}   t = {t:+8.2f}"
          f"{'  *' if abs(t) > 2.776 else ''}")

print("\n" + "=" * 100 + "\nREADING (registered in this file's docstring, committed before the first run)\n" + "=" * 100)
votes = sum(1 for s in SEEDS if winners[f"U{s}"] == "VI")
r_vi = float(np.mean([results[f"U{s}"][1]["VI"] for s in SEEDS if winners[f"U{s}"] == "VI"])) if votes else 0.0
copy = votes >= 4 and r_vi >= 0.90
inversion = all(tstat[k] > 2.776 for k in ("MI", "VIF"))
print(f"  copy criterion: VI wins {votes} of 5, mean r {r_vi:.3f} -> {'MET' if copy else 'NOT MET'}")
print(f"  inversion criterion: MI t = {tstat['MI']:+.2f}, VIF t = {tstat['VIF']:+.2f} -> {'MET' if inversion else 'NOT MET'}")
print(f"  released checkpoint's own r to VI: {results['trained'][1]['VI']:.3f}")
if copy and inversion:
    print("  H1 TRANSFERS: the residual-copy inversion appears in CDDFuse too.")
elif copy:
    print("  H2 COPY WITHOUT INVERSION: the skip copies VI, but MI/VIF do not rank the copy higher here.")
else:
    print("  H3 NO COPY: the residual-skip account does not transfer to CDDFuse.")
