"""Build the vgg16.pth that LRRNet's utils.init_vgg16 expects.

Why this script exists
----------------------
`utils.init_vgg16(vgg, model_dir)` does NOT match layers by name. It iterates the loaded
state_dict and assigns the first 20 tensors **positionally**:

    count 0,1 -> conv1_1.weight, conv1_1.bias      count 10,11 -> conv3_2.*
    count 2,3 -> conv1_2.*                         count 12,13 -> conv3_3.*
    count 4,5 -> conv2_1.*                         count 14,15 -> conv4_1.*
    count 6,7 -> conv2_2.*                         count 16,17 -> conv4_2.*
    count 8,9 -> conv3_1.*                         count 18,19 -> conv4_3.*

So a file whose tensors are in a different order loads silently and wrongly -- no exception, just
a corrupted loss network. That is the failure this script prevents: it writes the 20 tensors in
exactly that order and asserts every shape before saving.

torchvision's VGG16 `features` submodule happens to enumerate conv layers in precisely this order
(features.0, 2, 5, 7, 10, 12, 14, 17, 19, 21 = conv1_1 .. conv4_3), so the mapping is a direct
take-the-first-20 -- but that is verified here, never assumed.

Usage
-----
    python make_vgg16_weights.py --out model/vgg/vgg16.pth     # downloads ImageNet weights
    python make_vgg16_weights.py --dry-run                     # verify layout only, no download
"""
import argparse
import collections
import os

import torch
import torchvision

# (name, expected weight shape) in the exact order init_vgg16 consumes them
EXPECTED = [
    ("conv1_1", (64, 3, 3, 3)),
    ("conv1_2", (64, 64, 3, 3)),
    ("conv2_1", (128, 64, 3, 3)),
    ("conv2_2", (128, 128, 3, 3)),
    ("conv3_1", (256, 128, 3, 3)),
    ("conv3_2", (256, 256, 3, 3)),
    ("conv3_3", (256, 256, 3, 3)),
    ("conv4_1", (512, 256, 3, 3)),
    ("conv4_2", (512, 512, 3, 3)),
    ("conv4_3", (512, 512, 3, 3)),
]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="model/vgg/vgg16.pth")
ap.add_argument("--dry-run", action="store_true",
                help="verify the layer ordering without downloading pretrained weights")
args = ap.parse_args()

weights = None if args.dry_run else torchvision.models.VGG16_Weights.IMAGENET1K_V1
vgg = torchvision.models.vgg16(weights=weights)
sd = vgg.features.state_dict()          # keys: '0.weight', '0.bias', '2.weight', ...

items = list(sd.items())[:20]
assert len(items) == 20, f"expected at least 20 tensors in features, got {len(sd)}"

print(f"{'#':>3}  {'torchvision key':<16} {'-> LRRNet slot':<18} {'shape':<22} ok")
print("-" * 74)
out = collections.OrderedDict()
for i, (k, v) in enumerate(items):
    name, wshape = EXPECTED[i // 2]
    is_weight = (i % 2 == 0)
    slot = f"{name}.{'weight' if is_weight else 'bias'}"
    want = wshape if is_weight else (wshape[0],)
    good = tuple(v.shape) == want
    assert good, f"SHAPE MISMATCH at position {i}: {k} is {tuple(v.shape)}, expected {want}"
    assert k.endswith(".weight") if is_weight else k.endswith(".bias"), \
        f"ORDER BROKEN at position {i}: {k} is not a {'weight' if is_weight else 'bias'}"
    print(f"{i:>3}  {k:<16} {slot:<18} {str(tuple(v.shape)):<22} {'OK' if good else 'FAIL'}")
    out[slot] = v

print("\nAll 20 tensors are in the order init_vgg16 consumes them, with matching shapes.")

if args.dry_run:
    print("\nDRY RUN -- layout verified, nothing written. Re-run without --dry-run to save.")
else:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"\nwrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB, {len(out)} tensors)")
    print("Deviation to record: LRRNet ships no vgg16.pth, so these are torchvision's ImageNet "
          "VGG16 weights.\nThey are the standard source and match the positional layout the code "
          "expects, but the authors'\nexact file was not published and cannot be verified against.")
