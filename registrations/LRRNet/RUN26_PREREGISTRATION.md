# Run 26: can deterministic mode make a seed sufficient? Pre-registration

**Written 2026-10-06, committed on its own before any Run 26 kernel exists.**
- Nothing below may change after the first Run 26 output.
- A correction may be added only as a dated block at the end, and only before any metric is read.

## Why

Two LRRNet runs at the **same** seed, code, data and container differed by up to **44 % of σ** (median 16 %),
because GPU kernels were left free to choose nondeterministic algorithms. That was deliberate: σ should describe
what a reader re-running the released code meets. The paper therefore says that recording a seed is necessary and
far from sufficient.

A reviewer can reasonably ask:
- whether turning on PyTorch's deterministic mode would make a seed sufficient on this setup;
- what that would cost.

Run 26 measures both. It is a separate arm: **no existing row and no existing σ is changed.**

## Design

The published configuration (`num_block = 4`, untied, 49,204 parameters, 4 epochs, LLVIP, batch 8), with the Run 9
kernel unchanged except for one switch. When `LRR_DETERMINISTIC=1`, the training script sets:
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`;
- `torch.use_deterministic_algorithms(True, warn_only=True)`;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

It also prints a `[DET]` line to the log.

Runs, all in deterministic mode, on one T4:

| tag | seed |
|---|---|
| `nb4det_seed42a` | 42 |
| `nb4det_seed42b` | 42 (repeat) |
| `nb4det_seed43` | 43 |
| `nb4det_seed44` | 44 |

Each run is fused on TNO (25 pairs) and scored by the same evaluator as every other row.

## Kill criteria (before any metric is read)

1. Each log shows `[CONFIG]` with the run's seed, `num_block=4`, 4 epochs; the `[DET]` line; exit 0; one final
   checkpoint; 25 fused pairs.
2. The determinism switch was actually applied: the kernel asserts the patched line was found exactly once.

If a run crashes on an operation with no deterministic implementation, that operation is named and reported. The
`warn_only` setting is meant to prevent the crash and to log such operations instead.

## The reading (fixed now)

**Primary: do two deterministic runs at seed 42 reproduce each other?** Compare `nb4det_seed42a` with
`nb4det_seed42b` on all nine metrics.

| outcome | condition | what the paper and thesis will say |
|---|---|---|
| **D1: REPRODUCES** | all nine values agree to 1e-6 relative | "With deterministic algorithms enabled, a recorded seed reproduces an LRRNet run on this setup, at a training-time cost of X % (measured); without them, same-seed runs differ by up to 44 % of σ." |
| **D2: DOES NOT REPRODUCE** | any metric differs by more | "Even with deterministic algorithms enabled, two same-seed runs differed by up to Y σ", with the operations PyTorch warned about named |

**Secondary, descriptive only (no outcome depends on it):**
- training time per epoch, deterministic against the Run 9 nb4 runs (cost);
- the spread of the three deterministic seeds (42, 43, 44) against σ of the published configuration at its twenty
  seeds. Three values cannot estimate a variance well, so this is shown with its interval and labelled exploratory.

## Cost

Four training runs at about 1.1–1.5 GPU-hours each, so about 5 GPU-hours. Within the 18 hours left this week,
after Run 25 (under 1 hour) and before C1 (about 12 hours).
