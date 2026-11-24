"""Apply every source change the LRRNet seed-variance study needs, and verify each one.

Produces `training_seeded.py` from the released `training_fusion_lrr.py`, and rewrites `args.py`
and `net_lista.py` in place. Both the pilot and the real runs use this, so there is exactly one
definition of what was changed -- which is what the thesis has to document.

Everything is driven by environment variables so that no source edit is needed between runs:

    LRR_SEED        random seed                       (default 42)
    LRR_TRAIN_NUM   number of training pairs          (default 20000, the published value)
    LRR_EPOCHS      epochs                            (default 4, the published value)
    LRR_NUM_BLOCK   number of LLRR blocks             (default 4, the published choice)
    LRR_DATA        folder of infrared training images
    LRR_IR_DIR      substring naming the IR folder    (default 'lwir')
    LRR_VI_DIR      substring naming the VIS folder   (default 'visible')
    LRR_GPUS        visible CUDA devices              (default '0', the released value)
    LRR_MULTI_GPU   '1' enables the author's own commented-out DataParallel line
    LRR_TIED        '1' enables weight-tied unrolling (default '0', the released behaviour)
    LRR_LR          learning rate                     (default 1e-5, the published value)      Run 18
    LRR_WIR/LRR_WVI middle-loss weights w_ir, w_vi    (default 3.0 / 0.5, the published values) Run 18
    LRR_CROP        '1': random 128x128 crops of frames downscaled 2x, instead of whole frames
                    resized to 128x128 (default '0', the released behaviour)             Run 18
    LRR_CMP         '1': in the last epoch, every 20th batch, log the loss of copying the visible
                    image beside the model's on the same batch -> one [LOSSCMP] line   Run 18
    LRR_GRAM        lam3_gram, the infrared Gram-loss weight (default 2000, the published value) Run 23
    LRR_LAM2        lam2_vi, the shallow visible feature weight (default 1.5, the published value) Run 23

The Run 18 and Run 23 switches (RUN18_/RUN23_PREREGISTRATION.md) are off by default and leave every earlier run
reproducible from this file; with all of them unset the training arithmetic is unchanged.

The last two exist because num_block=8 -- one of the four settings the paper's own Table IV
compares -- exhausts a 16 GB card during the backward pass. The released code pins
CUDA_VISIBLE_DEVICES to a single device and leaves its DataParallel line commented out, so a second
card cannot help it. Enabling that line is NOT a deviation from the published configuration: the
network contains no BatchNorm, Dropout or other batch-dependent layer (verified by inspection), so
splitting a batch of 8 across two devices and averaging the gradients is arithmetically the same
update as computing it on one. Defaults keep the released behaviour, so stages 1-2 are unaffected.

LRR_IR_DIR / LRR_VI_DIR exist because the released code derives the visible path by string
replacement, `x.replace('lwir', 'visible')`, which is hard-wired to KAIST's folder naming. LLVIP
calls the same folder `infrared`, so without this the visible images silently fail to load.

LRR_TIED is modification M1 of chapter 6 (our_papers/paper1_variance_aware/archive_LRRFuse-VA/02_DESIGN.md) and is the FIRST patch
here that changes the network rather than the scaffolding around it. Everything above this line
leaves the published mathematics byte-for-byte intact; this one deliberately does not, and is
therefore off by default so stages 1-4 remain reproducible from the same file.

Run with --check to dry-run the patches against the upstream files without writing anything.
"""
import argparse
import pathlib
import urllib.request

BASE = "https://raw.githubusercontent.com/hli1221/imagefusion-LRRNet/e489fc325ea6acb9b7327b1de2c0cc4a09172ba4/"
T = "\t"

SEED_BLOCK = f"""EPSILON = 1e-5

# ---- REPLICATION PATCH: controlled seeding -------------------------------------------------
# The released code seeds nothing anywhere and calls random.shuffle twice. Data order comes from
# Python's `random`, weight initialisation from torch's, so seeding one and not the other means
# two runs differ by more than the seed and the resulting sigma is meaningless.
import numpy as np
_SEED = int(os.environ.get('LRR_SEED', 42))
random.seed(_SEED)
np.random.seed(_SEED)
torch.manual_seed(_SEED)
torch.cuda.manual_seed_all(_SEED)
# Deliberately NOT setting cudnn.deterministic or use_deterministic_algorithms. Sigma must
# represent what a reader re-running the released code actually sees, and that includes cudnn
# nondeterminism. Forcing determinism would understate sigma -- and a smaller sigma makes more
# published differences look real, which weakens the very criticism this measurement supports.
print(f'[CONFIG] seed={{_SEED}} '
      f'train_num={{os.environ.get("LRR_TRAIN_NUM", 20000)}} '
      f'epochs={{os.environ.get("LRR_EPOCHS", 4)}} '
      f'num_block={{os.environ.get("LRR_NUM_BLOCK", 4)}} '
      f'tied={{os.environ.get("LRR_TIED", "0")}} '
      f'lr={{os.environ.get("LRR_LR", "1e-05")}} '
      f'wir={{os.environ.get("LRR_WIR", "3.0")}} wvi={{os.environ.get("LRR_WVI", "0.5")}} '
      f'crop={{os.environ.get("LRR_CROP", "0")}} '
      f'gram={{os.environ.get("LRR_GRAM", "2000")}} lam2={{os.environ.get("LRR_LAM2", "1.5")}}', flush=True)

# ---- RUN 18 switches (all off by default) ---------------------------------------------------
_CROP = os.environ.get('LRR_CROP', '0') == '1'
_CMP = os.environ.get('LRR_CMP', '0') == '1'
_CMP_ROWS = []


def _crop_pair(paths_ir, paths_vi, size=128, down=2):
    # same random crop from both modalities, after a 2x downscale (LLVIP 1280x1024 -> 640x512)
    import cv2
    irs, vis = [], []
    for a, b in zip(paths_ir, paths_vi):
        ia, ib = cv2.imread(a, cv2.IMREAD_GRAYSCALE), cv2.imread(b, cv2.IMREAD_GRAYSCALE)
        ia = cv2.resize(ia, (ia.shape[1] // down, ia.shape[0] // down))
        ib = cv2.resize(ib, (ia.shape[1], ia.shape[0]))
        y = random.randint(0, ia.shape[0] - size)
        x = random.randint(0, ia.shape[1] - size)
        irs.append(ia[y:y + size, x:x + size][None])
        vis.append(ib[y:y + size, x:x + size][None])
    return torch.from_numpy(np.stack(irs)).float(), torch.from_numpy(np.stack(vis)).float()
"""

# ---- M1: weight-tied unrolling -----------------------------------------------------------------
# GetLS_Net creates one LRR_Block_lista per iteration, each with its own conv_Wdz/conv_Wdtz
# (4,865 parameters). That is a departure from the LISTA algorithm the network is named after, in
# which the block IS one iteration of one solver and its weights are shared across iterations. The
# departure is also what makes the paper's own Table IV unreadable: num_block moves depth and
# capacity together, a 2.96x spread over the ablated range (finding L1).
#
# Tying is applied ACROSS ITERATIONS ONLY, never across the two modality towers. LRR_NET holds two
# independent GetLS_Net instances and this patch leaves that untouched, so exactly one thing
# changes. Tying across modalities as well is a separate hypothesis (M1b, deferred) and folding it
# in here would commit the two-variables-at-once error this thesis exists to criticise.
#
# Default '0' reproduces the released behaviour exactly: the loop below runs num_block times and
# every lookup resolves to its own block, as upstream.
TIED_INIT = """        self.tied = os.environ.get('LRR_TIED', '0') == '1'
        for i in range(1 if self.tied else num_block):
            self.add_module('lrrblock' + str(i), LRR_Block_lista(s, 2 * n, channel, stride))"""

# ---- M2: asymmetric refiners ---------------------------------------------------------------------
# LRRNet computes a low-rank half L and a sparse half S, then processes both with an IDENTICAL
# layer (conv_ReZx and conv_ReUx are the same type, shape and settings) and collapses 128 channels
# to 1 in a single 3x3 convolution. Chapter 4 measured, at matched capacity, that this symmetry is
# wrong: attention buys perceptual fidelity, a CNN buys contrast and sharpness, and substituting
# either way costs. M2 applies that assignment to the halves LRRNet already computes.
#
# Widths are NOT chosen by taste. M1 frees 29,190 parameters; the pair (d, e) is the one that spends
# exactly that, so M1+M2 returns to the published 49,204. Over d,e in [4,96] that constraint has
# EXACTLY ONE solution -- d=16, e=69 -- so there is no width tuning to accuse us of.
#
# Both refiners are residual with zero-initialised output projections, so at step zero the network
# is bit-identical to M1. M2 can only be credited with what it learns, and a failure cannot be
# blamed on an initialisation that destroyed the M1 baseline.
#
# This source is the SINGLE definition: lrrfuse_m2.py execs this same string to run its budget
# search, so the thing measured is the thing that trains.
M2_SOURCE = '''

class LowRankRefiner(nn.Module):
    """Channel attention (MDTA-style) on a bottleneck -- for the LOW-RANK half.

    Channel attention is O(C^2), not O(HW^2), so it is affordable at 128x128.
    """

    def __init__(self, n, d):
        super(LowRankRefiner, self).__init__()
        self.down = nn.Conv2d(n, d, 1)
        self.qkv = nn.Conv2d(d, 3 * d, 1)
        self.qkv_dw = nn.Conv2d(3 * d, 3 * d, 3, padding=1, groups=3 * d)
        self.proj = nn.Conv2d(d, d, 1)
        self.up = nn.Conv2d(d, n, 1)
        self.temperature = nn.Parameter(torch.ones(1))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = self.down(x)
        b, c, h, w = z.shape
        q, k, v = self.qkv_dw(self.qkv(z)).chunk(3, dim=1)
        q = F.normalize(q.flatten(2), dim=-1)
        k = F.normalize(k.flatten(2), dim=-1)
        attn = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        z = (attn @ v.flatten(2)).reshape(b, c, h, w)
        return x + self.up(self.proj(z))


class SparseRefiner(nn.Module):
    """Depthwise-separable CNN on a bottleneck -- for the SPARSE half."""

    def __init__(self, n, e):
        super(SparseRefiner, self).__init__()
        self.down = nn.Conv2d(n, e, 1)
        self.dw = nn.Conv2d(e, e, 3, padding=1, groups=e)
        self.pw = nn.Conv2d(e, e, 1)
        self.up = nn.Conv2d(e, n, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = F.leaky_relu(self.down(x), 0.2)
        z = F.leaky_relu(self.pw(self.dw(z)), 0.2)
        return x + self.up(z)


'''

# The refiners are constructed LAST, after every other layer in the network. That is deliberate and
# not cosmetic: module construction draws from the global RNG, so building them earlier would shift
# the random stream for all the layers after them, and M1+M2 at seed 42 would start from different
# base weights than M1 at seed 42. Constructed last, the two runs begin from the SAME M1
# initialisation, and the only difference between them is the refiners themselves.
M2_INIT = """            self.conv_ReH = ConvLayer(channel, channel, s, stride)

        self.m2 = os.environ.get('LRR_M2', '0') == '1'
        if self.m2:
            self.ref_L = LowRankRefiner(n, int(os.environ.get('LRR_M2_D', 16)))
            self.ref_S = SparseRefiner(n, int(os.environ.get('LRR_M2_E', 69)))"""

# One refiner per ROLE, shared across the two modality towers -- so the only asymmetry M2 introduces
# is between L and S, which is the hypothesis under test. A per-modality variant would change two
# things at once.
M2_FORWARD = """        if self.m2:
            Z_x, Z_y = self.ref_L(Z_x), self.ref_L(Z_y)
            U_x, U_y = self.ref_S(U_x), self.ref_S(U_y)
        # get loww parts and sparse parts"""

NET_EDITS = [
    ("os import", "import numpy as np\nimport torch\nimport torch.nn as nn",
                  "import os\n\nimport numpy as np\nimport torch\nimport torch.nn as nn"),
    ("m2 classes", "\nclass Decoder_simple(nn.Module):", M2_SOURCE + "class Decoder_simple(nn.Module):"),
    ("m2 init", "            self.conv_ReH = ConvLayer(channel, channel, s, stride)", M2_INIT),
    ("m2 forward", "        # get loww parts and sparse parts", M2_FORWARD),
    ("tied init", "        for i in range(num_block):\n"
                  "            self.add_module('lrrblock' + str(i), "
                  "LRR_Block_lista(s, 2 * n, channel, stride))",
                  TIED_INIT),
    ("tied forward", "            lrrblock = getattr(self, 'lrrblock' + str(i))",
                     "            lrrblock = getattr(self, 'lrrblock' + "
                     "('0' if self.tied else str(i)))"),
]

ARGS_EDITS = [
    ("os import", "class Args():", "import os\n\n\nclass Args():"),
    ("epochs", T + "epochs = 4",
               T + "epochs = int(os.environ.get('LRR_EPOCHS', 4))"),
    ("path_ir", T + "path_ir = '/your/training/datasets/path/to/lwir/'",
                T + "path_ir = os.environ['LRR_DATA']"),
    ("lr", T + "lr = 0.00001", T + "lr = float(os.environ.get('LRR_LR', 0.00001))"),
]

TRAIN_EDITS = [
    ("gpu visibility", 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
                       'os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("LRR_GPUS", "0")'),
    ("seeding", "EPSILON = 1e-5\n", SEED_BLOCK),
    ("train_num", T + "train_num = 20000",
                  T + "train_num = int(os.environ.get('LRR_TRAIN_NUM', 20000))"),
    ("num_block", T + "num_block = num_block_list[1]",
                  T + "num_block = int(os.environ.get('LRR_NUM_BLOCK', 4))"),
    # The author's own DataParallel line, two lines above this one, is commented out. Restoring it
    # under a flag is the only way to fit num_block=8 without changing the batch size -- and with
    # no batch-dependent layer in the network it computes the identical update.
    ("data parallel", T + "model = model_or",
                      T + "model = (torch.nn.DataParallel(model_or,\n"
                      + T * 4 + "list(range(torch.cuda.device_count())))\n"
                      + T * 3 + "if os.environ.get('LRR_MULTI_GPU') == '1' and "
                                "torch.cuda.device_count() > 1 else model_or)\n"
                      + T + "print('[GPUS]', torch.cuda.device_count(), 'parallel=',\n"
                      + T * 3 + "isinstance(model, torch.nn.DataParallel), flush=True)"),
    # KAIST-specific folder naming, hard-wired; LLVIP names the same folder 'infrared'
    ("ir->vi path", T * 3 + "image_paths_vi = [x.replace('lwir', 'visible') for x in image_paths_ir]",
                    T * 3 + "image_paths_vi = [x.replace(os.environ.get('LRR_IR_DIR', 'lwir'), "
                            "os.environ.get('LRR_VI_DIR', 'visible')) for x in image_paths_ir]"),
    ("w_vi", T + "w_vi = 0.5", T + "w_vi = float(os.environ.get('LRR_WVI', 0.5))"),
    ("w_ir", T + "w_ir = w_ir_list[2]", T + "w_ir = float(os.environ.get('LRR_WIR', w_ir_list[2]))"),
    ("lam2_vi", T + "lam2_vi = lam2_vi_list[3]", T + "lam2_vi = float(os.environ.get('LRR_LAM2', lam2_vi_list[3]))"),
    ("lam3_gram", T + "lam3_gram = lam3_gram_list[1]",
                  T + "lam3_gram = int(os.environ.get('LRR_GRAM', lam3_gram_list[1]))"),   # int: it is also in the checkpoint name
    ("crop", T * 3 + "img_vi = utils.get_train_images(image_paths_vi, height=args.Height, width=args.Width, flag=img_flag)",
             T * 3 + "img_vi = utils.get_train_images(image_paths_vi, height=args.Height, width=args.Width, flag=img_flag)\n"
             + T * 3 + "if _CROP:\n" + T * 4 + "img_ir, img_vi = _crop_pair(image_paths_ir, image_paths_vi)"),
    ("loss cmp", T * 3 + "total_loss = loss_pixel_vi + loss_fea_vi + loss_fea_ir + loss_gram_ir",
                 T * 3 + "total_loss = loss_pixel_vi + loss_fea_vi + loss_fea_ir + loss_gram_ir\n"
                 + T * 3 + "if _CMP and e == args.epochs - 1 and idx % 20 == 0:\n"
                 + T * 4 + "with torch.no_grad():\n"
                 + T * 5 + "_vo = vgg(batch_vi)\n"
                 + T * 5 + "_l = weights_fea[0] * mse_loss(_vo[0], vgg_vis[0])\n"
                 + T * 5 + "for _k in (1, 2):\n"
                 + T * 6 + "_l = _l + weights_fea[_k] * mse_loss(_vo[_k], w_ir * vgg_irs[_k] + w_vi * vgg_vis[_k])\n"
                 + T * 5 + "_l = _l + weights_gram[3] * mse_loss(utils.gram_matrix(_vo[3]), utils.gram_matrix(vgg_irs[3]))\n"
                 + T * 5 + "_CMP_ROWS.append((total_loss.item(), _l.item()))"),
    # per-epoch wall clock, placed before the checkpoint save so save time is excluded
    ("timer start", T * 2 + "img_paths, batch_num = utils.load_dataset(data, batch_size)",
                    T * 2 + "img_paths, batch_num = utils.load_dataset(data, batch_size)\n"
                    + T * 2 + "_ep_t0 = time.time()"),
    ("timer print", "\n" + T * 2 + "# save model\n",
                    "\n" + T * 2 + 'print(f"\\n[EPOCH] {e} seconds={time.time()-_ep_t0:.1f} '
                    'batches={batch_num}", flush=True)\n'
                    + T * 2 + "if _CMP and _CMP_ROWS and e == args.epochs - 1:\n"
                    + T * 3 + "print(f'[LOSSCMP] batches={len(_CMP_ROWS)} model={np.mean([r[0] for r in _CMP_ROWS]):.1f} '\n"
                    + T * 4 + "f'copyVI={np.mean([r[1] for r in _CMP_ROWS]):.1f}', flush=True)\n"
                    + T * 2 + "# save model\n"),
]


def apply(src, edits, label):
    ok = True
    for name, old, new in edits:
        n = src.count(old)
        if n != 1:
            print(f"  [FAIL] {label}/{name}: {n} matches, expected 1")
            ok = False
            continue
        src = src.replace(old, new)
        print(f"  [OK  ] {label}/{name}")
    return src, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="path to the cloned imagefusion-LRRNet")
    ap.add_argument("--check", action="store_true",
                    help="dry-run against the upstream files, write nothing")
    a = ap.parse_args()

    def read(name):
        if a.check:
            return urllib.request.urlopen(BASE + name).read().decode("utf-8")
        return (pathlib.Path(a.repo) / name).read_text(encoding="utf-8")

    all_ok = True
    results = {}
    for fname, edits, out in (("args.py", ARGS_EDITS, "args.py"),
                              ("net_lista.py", NET_EDITS, "net_lista.py"),
                              ("training_fusion_lrr.py", TRAIN_EDITS, "training_seeded.py")):
        print(f"\n{fname}:")
        src, ok = apply(read(fname), edits, fname)
        all_ok &= ok
        try:
            compile(src, out, "exec")
            print(f"  [OK  ] {out} compiles")
        except SyntaxError as e:
            print(f"  [FAIL] {out} does not compile: {e}")
            all_ok = False
        results[out] = src

    if not all_ok:
        raise SystemExit("\nPATCHING FAILED -- nothing written")

    if a.check:
        print("\nCHECK PASSED -- all patches apply cleanly. Nothing written.")
        return

    for out, src in results.items():
        (pathlib.Path(a.repo) / out).write_text(src, encoding="utf-8")
        print(f"\nwrote {out}")

    # visdom is only used for live loss plots; a local stub shadows the installed package
    (pathlib.Path(a.repo) / "visdom.py").write_text(
        "# REPLICATION stub: the released code opens a live visdom server for loss plots.\n"
        "class Visdom:\n"
        "    def __init__(self, *a, **k): pass\n"
        "    def line(self, *a, **k): pass\n", encoding="utf-8")
    print("wrote visdom.py (stub)")


if __name__ == "__main__":
    main()
