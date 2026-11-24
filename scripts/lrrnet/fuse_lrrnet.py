"""Fuse a folder of IR/VIS pairs with a trained LRRNet checkpoint.

Replaces `testing_fusion_lrr.py`, whose `main()` hard-codes the test folder, the checkpoint name
and a KAIST/TNO-specific way of deriving the visible path from the infrared one. The fusion itself
is copied from its `run()` unchanged: same grayscale load at full resolution, same
`normalize_tensor` on both inputs, same `output['fuse']`, same `utils.save_image`.

One deliberate difference, and it is a bug fix. `testing_fusion_lrr.load_model` does:

    if num_block <= 4: model = model_or
    else:              model = torch.nn.DataParallel(model_or, ...)
    model.load_state_dict(torch.load(path))

but `training_fusion_lrr.py` has its `DataParallel` line commented out and therefore saves a
**plain** state_dict. So for num_block 6 or 8 -- two of the four rows of Table IV -- the released
test path looks for "module."-prefixed keys that were never written, and cannot load its own
checkpoints. Here the model is always plain and any "module." prefix is stripped defensively.

Run from inside the cloned LRRNet repo (it imports net_lista, args and utils from there).
"""
import argparse
import os
import sys

import torch
from torch.autograd import Variable

sys.path.insert(0, ".")
from net_lista import LRR_NET          # noqa: E402
from args import Args as cfg           # noqa: E402
import utils                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ir", required=True, help="folder of infrared test images")
ap.add_argument("--vi", required=True, help="folder of visible test images, same filenames")
ap.add_argument("--out", required=True, help="folder to write fused PNGs into")
ap.add_argument("--num_block", type=int, default=4)
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
img_flag = cfg.channel != 1            # False -> grayscale, matching the published config

model = LRR_NET(cfg.s, cfg.n, cfg.channel, cfg.stride, a.num_block, "cat")
sd = torch.load(a.ckpt, map_location="cpu")
sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
model.load_state_dict(sd)
model.eval().cuda()
print(f"loaded {a.ckpt}  ({sum(p.numel() for p in model.parameters()):,} parameters, "
      f"num_block={a.num_block})")

names = sorted(os.listdir(a.ir))
done = 0
with torch.no_grad():
    for name in names:
        ir_path, vi_path = os.path.join(a.ir, name), os.path.join(a.vi, name)
        if not os.path.exists(vi_path):
            raise SystemExit(f"no visible counterpart for {name} at {vi_path}")

        img_ir = utils.get_train_images(ir_path, height=None, width=None, flag=img_flag)
        img_vi = utils.get_train_images(vi_path, height=None, width=None, flag=img_flag)
        img_ir = Variable(img_ir, requires_grad=False).cuda()
        img_vi = Variable(img_vi, requires_grad=False).cuda()
        img_ir = utils.normalize_tensor(img_ir)
        img_vi = utils.normalize_tensor(img_vi)

        out = model(img_ir, img_vi)["fuse"]
        utils.save_image(out, os.path.join(a.out, os.path.splitext(name)[0] + ".png"))
        done += 1

print(f"fused {done}/{len(names)} pairs -> {a.out}")
assert done == len(names), "not every pair was fused"
