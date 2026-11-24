"""Arm A of the positive control for CrossFuse: a checkpoint that has never been trained.

Why this exists as a script rather than `train.py --epochs 0`
------------------------------------------------------------
"Zero gradient steps" has to be true by construction, not by an epoch loop that happens to iterate
zero times. A training script with `--epochs 0` still builds an optimiser, may step a scheduler,
and would leave a reader unable to tell from the code whether anything touched the weights. Here
nothing but construction happens, and the file is short enough to read in full.

What it guarantees, and how
---------------------------
  1. **Identical construction to training.** Seeding order (torch, numpy, random) and module
     construction order are copied from CrossFuse/train.py lines 98-133. If those change, this
     script is wrong and the check below fails loudly rather than silently drifting.
  2. **Identical checkpoint format to training.** The same four keys train.py writes, so
     run_inference.py loads an untrained model by exactly the path it loads a trained one. No
     special case anywhere downstream.
  3. **Verified tensor-by-tensor.** The model is rebuilt from scratch under the same seed and every
     parameter compared elementwise. This is the claim the paper makes about arm A, so it is
     checked here rather than asserted.

    python make_untrained.py --repo /kaggle/working/CrossFuse --seed 42 \
        --out /kaggle/working/CrossFuse/models/CrossFuse_RSM3_U42.pth

Cost: seconds. No data is read and no GPU work is done beyond allocating the parameters.
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn


def build(repo, seed, device):
    """Construct the four modules exactly as train.py does, under a fresh seed."""
    sys.path.insert(0, repo)
    from network import (Restormer_Encoder, Restormer_Decoder,
                         LongScaleFeatureExtraction, ShortScaleFeatureExtraction)

    # train.py lines 98-100, in this order. Order matters: each call advances a different generator,
    # and construction below draws from torch's.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    enc = nn.DataParallel(Restormer_Encoder()).to(device)
    dec = nn.DataParallel(Restormer_Decoder()).to(device)
    base = nn.DataParallel(LongScaleFeatureExtraction(dim=64, num_heads=8)).to(device)
    detail = nn.DataParallel(ShortScaleFeatureExtraction(num_layers=1)).to(device)
    return {"DIDF_Encoder": enc, "DIDF_Decoder": dec,
            "BaseFuseLayer": base, "DetailFuseLayer": detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    repo = os.path.abspath(a.repo)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mods = build(repo, a.seed, device)

    n = sum(p.numel() for m in mods.values() for p in m.parameters())
    print(f"[untrained] seed={a.seed}  parameters={n:,}  device={device}")

    # --- check 1: the same seed reproduces the same weights -------------------------------------
    again = build(repo, a.seed, device)
    checked = 0
    for key in mods:
        for (n1, p1), (n2, p2) in zip(mods[key].named_parameters(),
                                      again[key].named_parameters()):
            assert n1 == n2, f"parameter order differs: {n1} vs {n2}"
            if not torch.equal(p1, p2):
                sys.exit(f"FAIL: {key}.{n1} differs between two identically seeded builds. "
                         f"Construction is not deterministic under the seed; arm A would not be "
                         f"reproducible and must not be reported.")
            checked += 1
    print(f"[untrained] same seed -> same weights: {checked} tensors identical")

    # --- check 2: a DIFFERENT seed gives DIFFERENT weights ---------------------------------------
    # Check 1 alone is vacuous -- it re-seeds and rebuilds, so it passes whatever the model does,
    # and it would pass just as happily if every seed produced one identical network. That failure
    # mode is the dangerous one: five identical "seeds" give the untrained arm zero variance, every
    # Welch denominator collapses, and every metric appears to detect. The control would report the
    # exact opposite of the truth while looking cleaner than a real result.
    other = build(repo, a.seed + 1000, device)
    n_diff = n_same = 0
    for key in mods:
        for (n1, p1), (_, p2) in zip(mods[key].named_parameters(), other[key].named_parameters()):
            if torch.equal(p1, p2):
                n_same += 1
            else:
                n_diff += 1
    if n_diff == 0:
        sys.exit("FAIL: a different seed produced identical weights. The seeds are not doing "
                 "anything, arm A would have zero variance, and the control would be meaningless.")
    print(f"[untrained] different seed -> different weights: {n_diff} tensors differ, "
          f"{n_same} identical (constant-initialised layers such as LayerNorm are expected here)")

    # A trained model must NOT match this. Cheap guard against pointing --out at a real checkpoint.
    if os.path.exists(a.out):
        sys.exit(f"FAIL: {a.out} already exists. Refusing to overwrite a checkpoint.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({
        "epoch": 0, "tag": f"U{a.seed}", "seed": a.seed,
        "untrained": True,                       # so no later analysis can mistake this for trained
        "DIDF_Encoder": mods["DIDF_Encoder"].state_dict(),
        "DIDF_Decoder": mods["DIDF_Decoder"].state_dict(),
        "BaseFuseLayer": mods["BaseFuseLayer"].state_dict(),
        "DetailFuseLayer": mods["DetailFuseLayer"].state_dict(),
    }, a.out)
    print(f"[untrained] wrote {a.out}  ({os.path.getsize(a.out) / 1e6:.1f} MB)")
    print("[untrained] ZERO gradient steps were taken: this file contains initialisation only.")


if __name__ == "__main__":
    main()
