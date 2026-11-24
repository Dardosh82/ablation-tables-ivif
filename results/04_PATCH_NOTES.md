# 04 — M1 patch notes (weight-tied unrolling)

**Status:** ✅ **written and verified on CPU, 2026-09-03. No GPU time spent yet.**
Design: [`02_DESIGN.md`](02_DESIGN.md) §2.1–2.3 · plain language: [`03_EXPLAINED.md`](03_EXPLAINED.md) §5

---

## What changed

Patch #12 in `studied_papers/LRRNet/scripts/patch_lrrnet.py`, gated on `LRR_TIED` (default `0` = released
behaviour). It is the **first patch in this project that changes the network rather than the
scaffolding around it** — every earlier one left the published mathematics byte-for-byte intact.
That is why it is off by default and why the check below exists.

Three edits to `net_lista.py`, all inside `GetLS_Net`:

| edit | what |
|---|---|
| `os import` | `net_lista.py` does not import `os`; the flag needs it |
| `tied init` | build `1 if tied else num_block` blocks |
| `tied forward` | every iteration resolves to `lrrblock0` when tied |

```python
# init
self.tied = os.environ.get('LRR_TIED', '0') == '1'
for i in range(1 if self.tied else num_block):
    self.add_module('lrrblock' + str(i), LRR_Block_lista(s, 2 * n, channel, stride))

# forward
lrrblock = getattr(self, 'lrrblock' + ('0' if self.tied else str(i)))
```

**The loop still runs `num_block` times either way.** Tying changes *which weights* each iteration
uses, never *how many iterations run* — so depth and capacity are finally independent.

### Tied across iterations only — never across modalities

`LRR_NET` holds two `GetLS_Net` towers, one per sensor, and this patch does not touch that. Tying
across modalities as well (**M1b**, 15,149 parameters) is a separate hypothesis, deferred: folding it
in here would change two things in one run, which is the exact error this thesis exists to
criticise. See [`02_DESIGN.md`](02_DESIGN.md) §2.1b.

---

## Verification — `scripts/check_tied_capacity.py`, CPU, zero GPU

Two questions had to be answered *before* spending GPU time, and both now are.

### 1. Does the default still reproduce the published network?

Checked by parameter **name and shape**, tensor by tensor, against the unpatched upstream module —
not by a parameter count, since two different networks can share a count.

```
num_block=2:  36 tensors upstream,  36 patched -> IDENTICAL
num_block=4:  52 tensors upstream,  52 patched -> IDENTICAL
num_block=6:  68 tensors upstream,  68 patched -> IDENTICAL
num_block=8:  84 tensors upstream,  84 patched -> IDENTICAL
```

✅ **Stages 1–4 remain reproducible from this same file.** This mattered: a network-level patch that
silently altered the default would have invalidated eleven completed runs.

### 2. What does tying actually cost?

| num_block | untied | tied | reduction |
|---|---|---|---|
| 2 | 29,744 | 20,014 | 1.49× |
| **4 (published)** | **49,204** | **20,014** | **2.46×** |
| 6 | 68,664 | 20,014 | 3.43× |
| 8 | 88,124 | 20,014 | 4.40× |

```
published (num_block=4, untied) : 49,204
M1        (num_block=4, tied)   : 20,014
freed for M2's refiners         : 29,190
parameter spread, untied range  : 2.96x   (finding L1)
parameter spread, tied range    : 1.00x
```

> ### ⭐ Every figure the design predicted is confirmed, to the digit
>
> **49,204 → 20,014**, freeing **exactly 29,190** for M2 — so *capacity matching by construction*
> is now an arithmetic fact rather than a design intention.
>
> And **1.00× spread across the tied range**: under tying, `num_block` costs nothing. The capacity
> confound recorded as **L1** is not controlled for — it is **unconstructible**.

---

## How this was checked, and why that method matters

`check_tied_capacity.py` **imports `NET_EDITS` from `patch_lrrnet.py` itself** and applies it to a
freshly fetched upstream file. It never contains a second, hand-written copy of the patch, so the
thing verified here is literally the thing that will train — the two cannot drift apart.

The full 14-patch dry-run against upstream also passes:

```
python patch_lrrnet.py --check
  args.py                 3 edits  [OK] compiles
  net_lista.py            3 edits  [OK] compiles
  training_fusion_lrr.py  8 edits  [OK] compiles
CHECK PASSED -- all patches apply cleanly. Nothing written.
```

This discipline — never upload a source-patching cell without running it against the real upstream
file first — has now paid for itself three times in this project.

---

## What is not yet done

| | |
|---|---|
| ⬜ | **Run 1: M1 at `num_block=4`, seeds 42–44** (~3.3 h). `LRR_TIED=1` |
| ⬜ | **Run 2: M1 at `num_block=8`** (~3.3 h) — the first clean depth ablation |
| ⬜ | Kaggle cells for both, and a Cell-4 assertion on the `tied=1` line in `[CONFIG]` |
| ⬜ | `05_RESULTS.md`, and §6.2b of `03_EXPLAINED.md` (**both** language copies) |

`training_seeded.py` now prints `tied=` in its `[CONFIG]` banner, so the configuration is on the
record in every log — the same defence used for `[GPUS]` after the stage-3 OOM.

**Sequencing (`02_DESIGN.md` §2.1b):** LRRNet **stage 4** (`nb=2`, ×3, 4.4 h) runs first and in a
**separate notebook** — it needs no new code, and stage 3's OOM showed what sharing a notebook costs.
