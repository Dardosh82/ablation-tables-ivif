"""Step 0, zero GPU: does LRRNet's num_block ablation confound depth with capacity?

LRRNet's ablation varies the number of LRR blocks over {2, 4, 6, 8} (see the list `num_block_list`
in training_fusion_lrr.py). CrossFuse's Table VI had exactly this defect: its architecture
ablations moved block type and parameter count together, so no difference could be attributed to
the architecture alone (CONTEXT.md, capacity term 15/27 cells).

If parameter count moves with num_block here too, then LRRNet's block-count ablation measures
capacity, not representational depth -- and it is the same finding in a completely different
architecture family, which is exactly what case study 2 is for.

Runs on CPU. Downloads only the two source files it needs.
"""
import urllib.request
import pathlib
import sys

BASE = "https://raw.githubusercontent.com/hli1221/imagefusion-LRRNet/e489fc325ea6acb9b7327b1de2c0cc4a09172ba4/"
here = pathlib.Path(__file__).parent / "_lrrnet_src"
here.mkdir(exist_ok=True)

for name in ("net_lista.py", "args.py", "utils.py"):
    dst = here / name
    if not dst.exists():
        dst.write_bytes(urllib.request.urlopen(BASE + name).read())
        print(f"fetched {name}")

sys.path.insert(0, str(here))
import net_lista                                    # noqa: E402

# signature: LRR_NET(s, n, channel, stride, num_block, fusion_type)
# values from args.py: s=3 (filter size), n=128 (filters), channel=1 (gray), stride=1
S, N, CH, STRIDE, FUSION = 3, 128, 1, 1, "cat"

print(f"\n{'num_block':>10} {'parameters':>12} {'vs num_block=4':>16}")
print("-" * 42)
counts = {}
for nb in (2, 4, 6, 8):
    m = net_lista.LRR_NET(S, N, CH, STRIDE, nb, FUSION)
    counts[nb] = sum(p.numel() for p in m.parameters())

for nb, c in counts.items():
    print(f"{nb:>10} {c:>12,} {c / counts[4] * 100:>15.1f}%")

spread = max(counts.values()) / min(counts.values())
print(f"\nparameter spread across the ablated range: {spread:.2f}x")
print(
    "\nREADING\n"
    "If the spread is ~1.00x the block count is capacity-neutral and the ablation is clean.\n"
    "Anything else means the published num_block comparison moves depth and capacity together,\n"
    "so a difference between rows cannot be attributed to the number of LRR blocks alone --\n"
    "the same defect established for CrossFuse Table VI rows III and IV."
)
