"""M1 verification, zero GPU: what weight tying costs, and proof that it is off by default.

Two questions, both answerable on a laptop before a single GPU-hour is spent.

1. Does LRR_TIED=0 leave the published network bit-for-bit unchanged? If it does not, stages 1-4
   are no longer reproducible from the same source file and the whole patch is unusable. This is
   checked by parameter NAME and SHAPE against the unpatched upstream module, not by a count --
   two different networks can share a count.

2. What does tying actually buy? The design (our_papers/paper1_variance_aware/archive_LRRFuse-VA/02_DESIGN.md) claims 49,204 ->
   20,014 at num_block=4, a 2.46x reduction, and claims the freed 29,190 parameters are exactly
   what M2's refiners are allowed to spend. If that arithmetic is wrong, the capacity-matching-by-
   construction argument collapses, so it is verified here rather than asserted.

Also prints the tied column across the whole ablated range. Under tying the parameter count is
constant in num_block by construction, which is what makes a clean depth ablation possible --
finding L1 says the untied range spans 2.96x.

Run:  python check_tied_capacity.py
"""
import importlib
import os
import pathlib
import sys
import urllib.request

import torch

BASE = "https://raw.githubusercontent.com/hli1221/imagefusion-LRRNet/e489fc325ea6acb9b7327b1de2c0cc4a09172ba4/"
HERE = pathlib.Path(__file__).parent
SRC = HERE / "_lrrnet_src"          # pristine upstream, fetched by check_numblock_capacity.py
PATCHED = HERE / "_lrrnet_tied"     # upstream + the M1 patch, built here

# values from args.py: s=3 (filter size), n=128 (filters), channel=1 (gray), stride=1
S, N, CH, STRIDE, FUSION = 3, 128, 1, 1, "cat"


def load(path, name):
    """Import net_lista.py from `path` under a private module name."""
    sys.path.insert(0, str(path))
    for stale in [m for m in sys.modules if m.startswith("net_lista")]:
        del sys.modules[stale]
    mod = importlib.import_module("net_lista")
    sys.modules[name] = mod
    sys.path.pop(0)
    return mod


def build(mod, num_block):
    return mod.LRR_NET(S, N, CH, STRIDE, num_block, FUSION)


def signature(model):
    """Names and shapes, in order -- a far stronger identity check than a parameter count."""
    return [(n, tuple(p.shape)) for n, p in model.named_parameters()]


def main():
    SRC.mkdir(exist_ok=True)
    for name in ("net_lista.py", "args.py", "utils.py"):
        dst = SRC / name
        if not dst.exists():
            dst.write_bytes(urllib.request.urlopen(BASE + name).read())
            print(f"fetched {name}")

    # Build the patched copy from the same patch definition the runs use -- never a second,
    # hand-written implementation that could drift away from what actually trains.
    sys.path.insert(0, str(HERE))
    from patch_lrrnet import NET_EDITS, apply           # noqa: E402
    sys.path.pop(0)

    src, ok = apply((SRC / "net_lista.py").read_text(encoding="utf-8"), NET_EDITS, "net_lista.py")
    if not ok:
        raise SystemExit("patch did not apply -- run patch_lrrnet.py --check first")
    PATCHED.mkdir(exist_ok=True)
    (PATCHED / "net_lista.py").write_text(src, encoding="utf-8")

    upstream = load(SRC, "_upstream")

    # ---- 1. the default must be the published network, exactly -----------------------------
    # Every stage below sets BOTH flags explicitly. Inheriting either from the ambient environment
    # would make this script's verdict depend on who invoked it: run from the M2 notebook, which
    # exports LRR_M2=1 for the trainer and fusion step, stage 1 would build a model WITH refiners
    # and report the default as broken. A verifier whose answer depends on its caller verifies
    # nothing. (Caught on Kaggle, 2026-09-05, at a cost of 0.01 GPU-h -- which is what the gate
    # is for.)
    print("\n1. LRR_TIED unset -> is the published network unchanged?")
    os.environ.pop("LRR_TIED", None)
    os.environ.pop("LRR_M2", None)
    patched = load(PATCHED, "_patched")
    all_ok = True
    for nb in (2, 4, 6, 8):
        a, b = signature(build(upstream, nb)), signature(build(patched, nb))
        same = a == b
        all_ok &= same
        print(f"   num_block={nb}: {len(a):>3} tensors upstream, {len(b):>3} patched -> "
              f"{'IDENTICAL' if same else 'DIFFERENT  <-- FAIL'}")

    # ---- 2. what tying costs ----------------------------------------------------------------
    print("\n2. LRR_TIED=1, LRR_M2 off -> the M1 parameter budget")
    os.environ["LRR_TIED"] = "1"
    os.environ.pop("LRR_M2", None)
    tied = load(PATCHED, "_tied")

    print(f"\n   {'num_block':>10} {'untied':>10} {'tied':>10} {'reduction':>11}")
    print("   " + "-" * 43)
    rows = {}
    for nb in (2, 4, 6, 8):
        u = sum(p.numel() for p in build(upstream, nb).parameters())
        t = sum(p.numel() for p in build(tied, nb).parameters())
        rows[nb] = (u, t)
        print(f"   {nb:>10} {u:>10,} {t:>10,} {u / t:>10.2f}x")

    u4, t4 = rows[4]
    freed = u4 - t4
    tied_counts = {t for _, t in rows.values()}
    untied_spread = max(u for u, _ in rows.values()) / min(u for u, _ in rows.values())

    print(f"\n   published (num_block=4, untied) : {u4:,}")
    print(f"   M1        (num_block=4, tied)   : {t4:,}")
    print(f"   freed for M2's refiners         : {freed:,}")
    print(f"   parameter spread, untied range  : {untied_spread:.2f}x   (finding L1)")
    print(f"   parameter spread, tied range    : "
          f"{max(tied_counts) / min(tied_counts):.2f}x")

    # ---- 3. M2: does the freed budget get spent back exactly? --------------------------------
    # The whole capacity-matching-by-construction argument is this one number. It is checked on the
    # patched file itself, and with a real forward pass, because a parameter count says nothing
    # about whether the network runs.
    print("\n3. LRR_TIED=1 + LRR_M2=1 -> the full M1+M2 model")
    os.environ["LRR_M2"] = "1"
    m2 = load(PATCHED, "_m2")
    model = build(m2, 4)
    m2_total = sum(p.numel() for p in model.parameters())
    ref = sum(p.numel() for n_, p in model.named_parameters() if n_.startswith("decoder.ref_"))

    print(f"   M1 + M2 total                   : {m2_total:,}")
    print(f"     of which the two refiners     : {ref:,}")
    print(f"   published                       : {u4:,}")
    print(f"   difference                      : {m2_total - u4:+,}")

    x = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        out = model(x, torch.randn(1, 1, 128, 128))
    shape_ok = out["fuse"].shape == x.shape
    print(f"   forward pass                    : fuse {tuple(out['fuse'].shape)} "
          f"{'OK' if shape_ok else '<-- FAIL'}")

    # At initialisation the refiners are exact identities, so M1+M2 must fuse identically to M1.
    os.environ["LRR_M2"] = "0"
    m1_only = load(PATCHED, "_m1only")
    torch.manual_seed(0)
    a_model = build(m1_only, 4)
    os.environ["LRR_M2"] = "1"
    torch.manual_seed(0)
    b_model = build(load(PATCHED, "_m2b"), 4)
    with torch.no_grad():
        ir, vi = torch.randn(1, 1, 64, 64), torch.randn(1, 1, 64, 64)
        identity_ok = torch.allclose(a_model(ir, vi)["fuse"], b_model(ir, vi)["fuse"], atol=1e-6)
    print(f"   identity at init (M2 == M1)     : {'OK' if identity_ok else '<-- FAIL'}")
    os.environ.pop("LRR_M2", None)

    # ---- verdict ------------------------------------------------------------------------------
    checks = [
        ("default reproduces the published network", all_ok),
        ("published count is 49,204", u4 == 49_204),
        ("tied count at num_block=4 is 20,014", t4 == 20_014),
        ("tying is capacity-neutral in num_block", len(tied_counts) == 1),
        ("M1+M2 lands exactly on the published 49,204", m2_total == u4),
        ("M1+M2 forward pass produces the right shape", shape_ok),
        ("M1+M2 is identical to M1 at initialisation", identity_ok),
    ]
    print()
    for label, passed in checks:
        print(f"   [{'OK  ' if passed else 'FAIL'}] {label}")

    if not all(p for _, p in checks):
        raise SystemExit("\nFAILED -- do not spend GPU time until this passes")

    print(
        "\nREADING\n"
        "The default is byte-equivalent to upstream, so stages 1-4 remain reproducible from this\n"
        f"same file. Tying costs {u4 / t4:.2f}x fewer parameters and frees exactly {freed:,} for\n"
        "M2, which is the budget the capacity-matching-by-construction argument spends. And the\n"
        "tied column is constant in num_block, so depth can finally be ablated without moving\n"
        "capacity -- the experiment LRRNet's Table IV was presented as being, and was not."
    )


if __name__ == "__main__":
    main()

