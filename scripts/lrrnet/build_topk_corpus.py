"""M3: build LRRNet's training corpus the way CrossFuse's Top-K Selective Channel Alignment does.

Ported from the released `top_k_alignment.py`, function by function, at C=1 instead of C=3:

    compute_channel_means      -> mean luminance of the primary corpus            (the target)
    compute_patch_differences  -> |mean(patch) - target| for every auxiliary patch
    top_k_patches              -> keep the k patches with the SMALLEST difference
    optimize_gamma             -> the gamma whose corrected means match the target
    apply_gamma_correction     -> table = ((i/255) ** (1/gamma)) * 255, cv2.LUT

CrossFuse's version loops over three BGR channels and sums the per-channel differences. LRRNet is
grayscale, so the loop has one iteration. Nothing else about the mechanism changes.

TWO MODES, and the contrast between them is the experiment:

    --mode topk    the k most distribution-similar patches, gamma-aligned   (the mechanism)
    --mode random  k patches chosen at random, NOT aligned                  (the k=0 control)

Both emit exactly --total pairs, so iterations per epoch are identical to every previous run and to
each other. The only difference between the two corpora is how the auxiliary patches were chosen
and whether their intensity was aligned. `topk - random` is therefore the mechanism alone.

Evaluation is TNO-only in every run of this thesis, so using MSRS as auxiliary TRAINING data
contaminates no test set.
"""
import argparse
import os
import pathlib
import random

import cv2
import numpy as np


def crop_patches(ir_dir, vi_dir, size, stride):
    """Every size x size patch of every pair, on a stride grid. Mirrors crop_dataset()."""
    ir_dir, vi_dir = pathlib.Path(ir_dir), pathlib.Path(vi_dir)
    out = []
    for name in sorted(os.listdir(ir_dir)):
        ir = cv2.imread(str(ir_dir / name), cv2.IMREAD_GRAYSCALE)
        vi = cv2.imread(str(vi_dir / name), cv2.IMREAD_GRAYSCALE)
        if ir is None or vi is None or ir.shape != vi.shape:
            continue
        h, w = ir.shape
        for y in range(0, h - size + 1, stride):
            for x in range(0, w - size + 1, stride):
                out.append((f"{pathlib.Path(name).stem}_p{y}_{x}.png",
                            ir[y:y + size, x:x + size], vi[y:y + size, x:x + size]))
    return out


def gamma_correction(src, gamma):
    """Exactly the released gammaCorrection(), which is why the LUT is built this way."""
    table = np.array([((i / 255) ** (1 / gamma)) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(src, table)


def optimise_gamma(patches, target_mean, lo=0.1, hi=5.0, steps=200):
    """The gamma whose corrected mean is closest to the target.

    The released code calls scipy.optimize on a 3-vector; at C=1 this is a 1-D problem, so a dense
    sweep is used instead -- it cannot get stuck in a local minimum and is trivially reproducible.
    """
    best = (None, float("inf"))
    for g in np.linspace(lo, hi, steps):
        m = np.mean([gamma_correction(v, g).mean() for _, _, v in patches])
        loss = abs(m - target_mean)
        if loss < best[1]:
            best = (float(g), loss)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary-ir", required=True)
    ap.add_argument("--primary-vi", required=True)
    ap.add_argument("--aux-ir", required=True)
    ap.add_argument("--aux-vi", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("topk", "random"), required=True)
    ap.add_argument("--k", type=int, default=3610)
    ap.add_argument("--total", type=int, default=12025)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0,
                    help="corpus construction seed; FIXED across training seeds so every run "
                         "sees the same corpus and only the training seed varies")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    (out / "infrared").mkdir(parents=True, exist_ok=True)
    (out / "visible").mkdir(parents=True, exist_ok=True)

    n_primary = a.total - a.k
    primary = sorted(os.listdir(a.primary_ir))[:n_primary]
    assert len(primary) == n_primary, f"primary has {len(primary)}, need {n_primary}"

    # target = mean luminance of the primary corpus, over the visible images (compute_channel_means)
    target = float(np.mean([cv2.imread(os.path.join(a.primary_vi, f),
                                       cv2.IMREAD_GRAYSCALE).mean() for f in primary]))
    print(f"primary: {len(primary):,} pairs   target visible mean = {target:.3f}")

    patches = crop_patches(a.aux_ir, a.aux_vi, a.size, a.stride)
    print(f"auxiliary: {len(patches):,} patches of {a.size}x{a.size}")
    assert len(patches) >= a.k, f"only {len(patches)} auxiliary patches, need k={a.k}"

    if a.mode == "topk":
        # compute_patch_differences + top_k_patches, at C=1
        diffs = [abs(vi.mean() - target) for _, _, vi in patches]
        idx = sorted(range(len(patches)), key=lambda i: diffs[i])[:a.k]
        chosen = [patches[i] for i in idx]
        before = float(np.mean([v.mean() for _, _, v in chosen]))
        gamma, loss = optimise_gamma(chosen, target)
        chosen = [(n, ir, gamma_correction(vi, gamma)) for n, ir, vi in chosen]
        after = float(np.mean([v.mean() for _, _, v in chosen]))
        print(f"top-k: mean |diff| of kept = {np.mean([diffs[i] for i in idx]):.3f} "
              f"(all patches: {np.mean(diffs):.3f})")
        print(f"gamma = {gamma:.4f}   visible mean {before:.3f} -> {after:.3f} "
              f"(target {target:.3f}, residual {loss:.4f})")
    else:
        # the k=0 control: same count, chosen at random, intensity left alone
        rng = random.Random(a.seed)
        chosen = rng.sample(patches, a.k)
        m = float(np.mean([v.mean() for _, _, v in chosen]))
        print(f"random: {a.k:,} patches, no alignment. visible mean {m:.3f} "
              f"(target {target:.3f}, off by {abs(m - target):.3f})")

    for f in primary:
        for sub, src in (("infrared", a.primary_ir), ("visible", a.primary_vi)):
            os.symlink(os.path.join(src, f), out / sub / f"p_{f}")
    for name, ir, vi in chosen:
        cv2.imwrite(str(out / "infrared" / f"a_{name}"), ir)
        cv2.imwrite(str(out / "visible" / f"a_{name}"), vi)

    n_ir = len(os.listdir(out / "infrared"))
    n_vi = len(os.listdir(out / "visible"))
    print(f"\nwrote {out}   infrared {n_ir:,}   visible {n_vi:,}")
    assert n_ir == n_vi == a.total, f"corpus is {n_ir}/{n_vi}, expected {a.total}"
    assert sorted(os.listdir(out / "infrared")) == sorted(os.listdir(out / "visible")), \
        "infrared and visible filenames differ -- the loader pairs them by sorted order"
    print(f"CORPUS OK -- {a.total:,} pairs, mode={a.mode}, "
          f"{n_primary:,} primary + {a.k:,} auxiliary")


if __name__ == "__main__":
    main()
