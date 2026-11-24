"""M2 width budget: find the refiner widths that spend exactly what M1 freed, and verify them.

The refiner classes are NOT defined here. They live as `M2_SOURCE` inside `patch_lrrnet.py`, which
is the string patched into `net_lista.py` and therefore the code that actually trains. This module
execs that same string, so the budget measured here cannot drift away from the network that runs --
the same discipline as `check_tied_capacity.py`, which imports `NET_EDITS` rather than
re-implementing the patch.

    M1 frees      49,204 - 20,014 = 29,190 parameters
    M2 must spend exactly that, so M1 + M2 returns to the published 49,204
    => the comparison against published LRRNet is capacity-matched BY CONSTRUCTION

Run:  python lrrfuse_m2.py
"""
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from patch_lrrnet import M2_SOURCE                      # noqa: E402

# exec the patched source in a namespace holding exactly what net_lista.py has in scope
_ns = {"nn": nn, "torch": torch, "F": F}
exec(M2_SOURCE, _ns)
LowRankRefiner, SparseRefiner = _ns["LowRankRefiner"], _ns["SparseRefiner"]

N_CHANNELS = 128            # args.py: n = 128
PUBLISHED_TOTAL = 49_204    # the untied published network, measured
M1_TOTAL = 20_014           # the weight-tied network, measured


def count(module):
    return sum(p.numel() for p in module.parameters())


def solve_widths(budget=PUBLISHED_TOTAL - M1_TOTAL, n=N_CHANNELS,
                 d_range=range(4, 97), e_range=range(4, 97)):
    """Every (d, e) whose two refiners together cost exactly `budget`, by BUILDING them."""
    lo = {d: count(LowRankRefiner(n, d)) for d in d_range}
    so = {e: count(SparseRefiner(n, e)) for e in e_range}
    exact = [(d, e, lo[d], so[e]) for d in lo for e in so if lo[d] + so[e] == budget]
    return sorted(exact), lo, so


if __name__ == "__main__":
    budget = PUBLISHED_TOTAL - M1_TOTAL
    exact, lo, so = solve_widths(budget)

    print(f"budget to spend : {budget:,}   ({PUBLISHED_TOTAL:,} published - {M1_TOTAL:,} tied)")
    print(f"search space    : d, e in [4, 96]  ->  {len(lo) * len(so):,} pairs\n")
    print(f"{len(exact)} EXACT solution(s):\n")
    print(f"{'d':>4}{'e':>5}{'L params':>11}{'S params':>11}{'L share':>10}")
    print("-" * 41)
    for d, e, l, s in exact:
        print(f"{d:>4}{e:>5}{l:>11,}{s:>11,}{l / (l + s) * 100:>9.1f}%")

    assert len(exact) == 1, f"expected a unique solution, found {len(exact)}"
    d, e, l, s = exact[0]
    assert M1_TOTAL + l + s == PUBLISHED_TOTAL

    print(f"\nM1 + M2 = {M1_TOTAL:,} + {l + s:,} = {M1_TOTAL + l + s:,}"
          f"   vs published {PUBLISHED_TOTAL:,}   EXACT")
    print("\nThe constraint has ONE solution in the search range, so the widths are not a tuning\n"
          "knob: they are determined by the budget. Nothing here was chosen to flatter a result.")

    lr, sr = LowRankRefiner(N_CHANNELS, d), SparseRefiner(N_CHANNELS, e)
    x = torch.randn(2, N_CHANNELS, 32, 32)
    with torch.no_grad():
        assert torch.allclose(lr(x), x), "LowRankRefiner is not identity at init"
        assert torch.allclose(sr(x), x), "SparseRefiner is not identity at init"
    print("\nboth refiners are shape-preserving and EXACTLY identity at initialisation,\n"
          "so training starts from M1 and M2 is credited only with what it learns.")

    print(f"""
CAVEAT, PRE-REGISTERED BEFORE THE RUN

  The unique solution gives the attention refiner {l / (l + s) * 100:.0f}% of the budget (d={d}) and the CNN
  refiner {s / (l + s) * 100:.0f}% (e={e}). That split is a CONSEQUENCE of demanding an exact match, not a
  functional choice, and d={d} is a narrow bottleneck for attention over {N_CHANNELS} channels.

  So if M2 fails, two explanations are confounded: the chapter-4 assignment may not transfer to an
  LRR-style split, OR the attention branch may simply be too narrow. That ambiguity is stated here,
  before the result, and a failure will be reported with it rather than resolved by widening d
  afterwards -- which would be tuning against the very budget that makes the comparison clean.

  It is not unreasonable as it stands: attention mixes globally through the down/up projections and
  is parameter-efficient per unit of function, whereas a CNN's capacity scales with its width. A5
  needed 20 DetailNode layers to match one Transformer block, which points the same way.""")
