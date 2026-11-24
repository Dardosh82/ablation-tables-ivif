# Kaggle cells — the positive control on CrossFuse

**Purpose.** §7's headline — *SF and AG cannot separate a randomly weighted network from a trained
one* — currently rests on **one architecture**. Every other result in the paper is measured on two
families precisely so a defect in one design cannot be mistaken for a property of the field; the
control is the exception, at the paper's most quotable point. §9 names this limitation and names
this experiment as what would settle it. This run is that experiment.

**The reading of both outcomes is fixed before the run**, in the source of
[`analyse_poscontrol_cf.py`](../scripts/analyse_poscontrol_cf.py). Read it before running anything.
Outcome 2 — SF and AG *succeeding* here — narrows our headline and is the more interesting result;
it is written down with the same weight as the outcome that helps us.

---

## Budget

| | seeds | cost |
|---|---|---|
| trained arm — seeds 42/43/44 | **already exist** (A1b, A1c, A1d) | 0 h |
| trained arm — seeds 45, 46 | 2 runs × ~3.9 h | **~7.8 h** |
| untrained arm — seeds 42–46 | 5 × construction only, no training | **~0.5 h** |
| inference + evaluation, 7 checkpoints | | included above |
| | | **≈ 8.5 h of the 16 h 28 m available** |

> **Why not stop at n = 3, which is free?** §7 contains a subsection called *Hardened, not defended*
> describing how this very table stood at three seeds, how our own standard condemned it, and how
> two seeds were added and the finding narrowed. Running the second architecture's control at n = 3
> would commit, in the same paper, the error that section exists to record correcting.
> **The 8 hours are a consistency requirement, not a refinement.**

---

## Before you start

> ⚠️ **Name the notebook `Tier3_PosControl_CF` BEFORE running.** Renaming afterwards forces a save,
> and that save creates a new version **which replaces the output** — every checkpoint and CSV
> disappears. Confirmed on 2026-08-25: renaming a notebook produced a 10.8-second version and wiped
> 3.9 hours of results. The name is free now and costs a full re-run later.

1. Open **`Tier3_A1c`** → **File → Copy & Edit**. This carries the inputs and the accelerator
   setting across, so nothing needs re-attaching.
2. **Rename to `Tier3_PosControl_CF` now.**
3. Confirm the sidebar reads **GPU T4 ×2** and **Internet Off**.
4. Confirm inputs: `crossfuse-tier2-code`, `crossfuse-testsets`, `crossfuse-k0-dataset`.
5. Upload `make_untrained.py` into the `crossfuse-tier2-code` dataset under `scripts/`, or paste it
   inline as Cell 1b below.
6. Replace the cells named below, **Run All**, then **Save Version → Save & Run All (Commit)**.
   ⚠️ Not *Quick Save*.

The trained arm reuses the **A1b family configuration** (`k=0`, step-matched) because that is the
configuration whose σ §4 already reports, at seeds 42–44. Seeds 45 and 46 extend the same
configuration — nothing else changes, or they are not seed replicates.

---

## Cell 1b — the untrained arm (new cell, insert after the setup cell)

```python
%cd /kaggle/working
import os, subprocess, sys
os.makedirs('/kaggle/working/CrossFuse/models', exist_ok=True)
for s in (42, 43, 44, 45, 46):
    out = f'/kaggle/working/CrossFuse/models/CrossFuse_RSM3_U{s}.pth'
    r = subprocess.run([sys.executable, '-u', 'scripts/make_untrained.py',
                        '--repo', '/kaggle/working/CrossFuse',
                        '--seed', str(s), '--out', out])
    assert r.returncode == 0, f'untrained seed {s} FAILED -- stop here, do not continue'
print('ARM A BUILT: five untrained checkpoints, zero gradient steps taken')
```

`make_untrained.py` rebuilds each model twice under the same seed and compares every parameter
tensor elementwise. If construction is not deterministic under the seed it exits non-zero and this
cell stops the notebook — an unreproducible arm A must not reach the analysis.

## Cell 2 — the two extra trained seeds

```python
%cd /kaggle/working/CrossFuse
for s, tag in ((45, 'PC45'), (46, 'PC46')):
    !python -u train.py --tag {tag} --data {src_h5} \
        --epochs 299 --epoch_gap 75 --optim_step 37 \
        --seed {s} --outdir ./models --resume auto --max_hours 5.0
```

Only `--tag` and `--seed` differ from A1b. Everything else must stay identical.

## Cell 4 — inference and evaluation, all seven checkpoints

```python
%cd /kaggle/working
CKPTS = [('A1b', 'CrossFuse_RSM3_A1b.pth'), ('A1c', 'CrossFuse_RSM3_A1c.pth'),
         ('A1d', 'CrossFuse_RSM3_A1d.pth'), ('PC45', 'CrossFuse_RSM3_PC45.pth'),
         ('PC46', 'CrossFuse_RSM3_PC46.pth')] + \
        [(f'U{s}', f'CrossFuse_RSM3_U{s}.pth') for s in (42, 43, 44, 45, 46)]

for tag, ck in CKPTS:
    p = f'/kaggle/working/CrossFuse/models/{ck}'
    if not os.path.exists(p):
        print(f'SKIP {tag}: {ck} not present'); continue
    !python -u scripts/run_inference.py --repo /kaggle/working/CrossFuse \
            --ckpt {p} --datasets TNO
    !python -u scripts/evaluate.py --repo /kaggle/working/CrossFuse \
            --results /kaggle/working/CrossFuse/test_results_gray \
            --out /kaggle/working/tables_{tag}_raw
```

**TNO only.** LRRNet's control was scored on the same 25 TNO pairs, and the two architectures have
to be compared on the same test set or the comparison is not one. Restricting to TNO also keeps
this cell inside the budget.

## Cell 5 — bundle

Change the tag list to `[t for t, _ in CKPTS]` so every table is collected, not just one.

---

## Afterwards, locally

```powershell
$env:KAGGLE_API_TOKEN=(Get-Content "$env:USERPROFILE\.kaggle\access_token" -Raw).Trim()
$k  = "<project>\studied_papers\CrossFuse\Tier1_CrossFuse\.venv\Scripts\kaggle.exe"
$t3 = "<project>\studied_papers\CrossFuse\Tier3_CrossFuse"

& $k kernels output "mostafanada82/<slug>" -p "$t3\results_poscontrol_cf" --file-pattern ".*\.csv$"
& "<project>\studied_papers\CrossFuse\Tier1_CrossFuse\.venv\Scripts\python.exe" `
    "$t3\scripts\analyse_poscontrol_cf.py" --dir "$t3\results_poscontrol_cf"
```

Note the regex form of `--file-pattern` — the CLI takes a regex, so `.*\.csv$`, not `*.csv`.

The analysis prints which of the three pre-registered outcomes arrived and what each requires of the
paper. **Apply it as written**, including outcome 2, which narrows our own headline.

---

## What changes in the paper, either way

| | |
|---|---|
| §9 | The limitation *"The positive control ran on one architecture"* is **removed** and replaced by the measured result, whichever it is |
| §7 | Gains a second column or a second row; the "on this architecture only" scoping either widens or hardens |
| Abstract, §1, §8 | Wording follows §7 — `check_paper.py` **fails the build** if any `SF and AG cannot …` sentence stops naming its scope |
| `05_RESULTS.md`, `CONTEXT.md`, `PAPER.md`, §6.2b of `03_EXPLAINED.md` (EN **and** AR) | updated the same day, per the standing rule |
