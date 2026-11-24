# Run 1 — M1, weight-tied unrolling at `num_block = 4`

**The first run of chapter 6.** Design: [`our_papers/paper1_variance_aware/archive_LRRFuse-VA/02_DESIGN.md`](../../our_papers/paper1_variance_aware/archive_LRRFuse-VA/02_DESIGN.md)
§2.1–2.3 · patch and CPU verification: [`04_PATCH_NOTES.md`](../../our_papers/paper1_variance_aware/archive_LRRFuse-VA/04_PATCH_NOTES.md)

**The question, in one line.** Stages 1–3 added 39,000 parameters (`nb` 4 → 6 → 8) and not one of
nine metrics moved. This asks the same question in reverse: **remove 29,190 and measure.**

**Configuration.** Identical to stage 1 in every respect — `num_block = 4`, `epochs = 4`,
`batch_size = 8`, 128×128 grayscale, `lr = 1e-5`, `lam2 = 1.5`, `w_ir = 3.0`, `lam3_gram = 2000`,
LLVIP, 25 TNO pairs, CrossFuse's `Evaluator`. **One variable changes: `LRR_TIED = 1`.**

| | published | M1 |
|---|---|---|
| parameters | 49,204 | **20,014** |
| blocks built | 4 | **1** |
| block applications per forward | 4 | **4 — unchanged** |

Tying changes *which weights* each iteration uses, never *how many iterations run*. FLOPs are
therefore unchanged, so expect **~1.10 h per seed**, as in stage 1.

**Baseline.** Stage 1's `nb=4` untied rows, **n = 5, σ already measured**. Never the published table
(finding **L2**).

**Budget.** 3 seeds × ~1.1 h + ~6 min fusing/scoring ≈ **3.5 h**. Check `kaggle quota` before
launching — this run has no retry margin in the current week.

**Notebook:** `lrrnet-m1-tied` · **Settings:** GPU T4 ×2, Internet On
**Attach:** LLVIP · `lrrnet-replication-scripts` · `crossfuse-testsets` · `crossfuse-tier2-code`

> ⚠️ **Update `lrrnet-replication-scripts` first.** It must carry the 14-patch `patch_lrrnet.py`
> (with `LRR_TIED`) and `check_tied_capacity.py`. Cell 3 fails loudly if it does not — but the
> failure costs a session slot, so verify the dataset version before launching.

---

## Cell 1 — GPU guard

Unchanged from stage 1.

```python
import subprocess, os
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    print("NO GPU ATTACHED --", e); os._exit(1)
import torch
print("torch", torch.__version__, "| GPUs:", torch.cuda.device_count())
```

---

## Cell 2 — clone, patch, VGG weights, locate every input

Unchanged from stage 1 except the last line, which sets `LRR_TIED` **in the notebook's own
environment** rather than only in the training call.

> ⚠️ **Why `LRR_TIED` must be global, not per-subprocess.** `fuse_lrrnet.py` builds its own
> `LRR_NET` to load the checkpoint. A tied checkpoint holds **one** block; an untied model expects
> **four**. If fusion ran without the flag, `load_state_dict` would raise a size mismatch. That is
> the loud failure — the same class of trap recorded for CrossFuse A3/A5, where the architecture
> flags had to be passed to inference or nothing loaded. Setting it in `os.environ` here means every
> child process inherits it.

```python
import os, shutil, subprocess, glob
os.chdir("/kaggle/working")
if not os.path.isdir("imagefusion-LRRNet"):
    subprocess.run(["git","clone","--depth","1",
                    "https://github.com/hli1221/imagefusion-LRRNet.git"], check=True)
REPO = "/kaggle/working/imagefusion-LRRNet"
os.chdir(REPO)
subprocess.run(["rm","-rf",".git"], check=False)

print("mounted inputs:", sorted(os.listdir("/kaggle/input")))

def find_one(pattern, what):
    hits = glob.glob(pattern, recursive=True)
    assert hits, f"{what} not found: {pattern}"
    return sorted(hits, key=len)[0]

for name in ("patch_lrrnet.py", "check_tied_capacity.py",
             "make_vgg16_weights.py", "fuse_lrrnet.py", "eval_fused.py"):
    src = find_one(f"/kaggle/input/**/{name}", name)
    shutil.copy(src, name); print(f"copied {name} from {src}")

subprocess.run(["python","patch_lrrnet.py","--repo","."], check=True)
subprocess.run(["python","make_vgg16_weights.py","--out","model/vgg/vgg16.pth"], check=True)

IR_TRAIN = find_one("/kaggle/input/**/LLVIP/infrared/train", "LLVIP infrared/train").rstrip("/")
VI_TRAIN = IR_TRAIN.replace("infrared", "visible")
n_tr = len(os.listdir(IR_TRAIN))
assert os.path.isdir(VI_TRAIN) and n_tr == len(os.listdir(VI_TRAIN))
print(f"\ntrain: {n_tr:,} pairs at {IR_TRAIN}")

TNO_IR = find_one("/kaggle/input/**/TNO/ir", "TNO/ir").rstrip("/")
TNO_VI = find_one("/kaggle/input/**/TNO/vi", "TNO/vi").rstrip("/")
assert sorted(os.listdir(TNO_IR)) == sorted(os.listdir(TNO_VI)), "TNO ir/vi filenames differ"
print(f"TNO test: {len(os.listdir(TNO_IR))} pairs")

CF = os.path.dirname(find_one("/kaggle/input/**/CrossFuse/utils/Evaluator.py", "Evaluator.py"))
CF = os.path.dirname(CF)
print(f"CrossFuse repo: {CF}")

os.environ.update(LRR_DATA=IR_TRAIN + "/", LRR_IR_DIR="infrared", LRR_VI_DIR="visible",
                  LRR_TIED="1")                      # <-- M1, inherited by every child process
print("\nready · LRR_TIED =", os.environ["LRR_TIED"])
```

---

## Cell 3 — 🆕 the M1 preflight gate

**This is the cell that did not exist in stages 1–4, and the reason it exists.** The patch was
verified on a laptop; that proves the *logic*. This proves the *file that is about to train on this
machine* is the one we think it is. A wrong `patch_lrrnet.py` version in the attached dataset would
otherwise train a perfectly healthy **untied** network for 3.5 hours and hand back a row silently
mislabelled `m1_tied`.

Two independent checks, in increasing strictness:

```python
import os, sys, subprocess, importlib

# (a) the standalone verifier, against a freshly fetched upstream copy -- prints the full table
os.makedirs("/kaggle/working/preflight", exist_ok=True)
for f in ("patch_lrrnet.py", "check_tied_capacity.py"):
    subprocess.run(["cp", f"{REPO}/{f}", "/kaggle/working/preflight/"], check=True)
p = subprocess.run([sys.executable, "check_tied_capacity.py"],
                   cwd="/kaggle/working/preflight", capture_output=True, text=True)
print(p.stdout[-2500:] or p.stderr[-2500:])
assert p.returncode == 0, "check_tied_capacity.py FAILED -- do not spend GPU time"

# (b) the strict one: the repo file that will ACTUALLY train, at the flag setting it will train under
sys.path.insert(0, REPO)
for stale in [m for m in sys.modules if m.startswith("net_lista")]:
    del sys.modules[stale]
import net_lista
n_par = sum(x.numel() for x in net_lista.LRR_NET(3, 128, 1, 1, 4, 'cat').parameters())
blocks = [n for n, _ in net_lista.LRR_NET(3, 128, 1, 1, 4, 'cat').named_parameters()
          if "lrrblock" in n]
n_blocks = len({b.split(".")[1] for b in blocks})

print(f"\nrepo net_lista.py under LRR_TIED={os.environ.get('LRR_TIED')}:")
print(f"  parameters      {n_par:,}   (published untied: 49,204)")
print(f"  distinct blocks {n_blocks}         (published untied: 4)")
assert n_par == 20_014, f"expected 20,014 tied parameters, got {n_par:,} -- the dataset is stale"
assert n_blocks == 1, f"expected 1 shared block, found {n_blocks}"
print("\nPREFLIGHT PASS -- this is M1")
```

> If either assertion fires, **stop and update the `lrrnet-replication-scripts` dataset.** The cost
> of ignoring it is a full session and a row that looks fine and is not.

---

## Cell 4 — seed audit

Unchanged from stage 1, and still not skippable: it re-proves on *this* container that both random
generators respond to the seed. It now also exercises the tied model, so a construction error would
surface here rather than 66 minutes into training.

```python
import random, numpy as np, torch, sys
sys.path.insert(0, ".")
import net_lista, utils

def fingerprint(seed, n=2000):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    m = net_lista.LRR_NET(3, 128, 1, 1, 4, 'cat')
    w = sum(p.detach().double().sum().item() for p in m.parameters())
    paths, _ = utils.list_images(os.environ["LRR_DATA"])
    paths = paths[:n]; random.shuffle(paths)
    return round(w, 8), [os.path.basename(p) for p in paths[:4]]

a1, a2, b = fingerprint(42), fingerprint(42), fingerprint(43)
for tag, f in (("seed 42 (1st)", a1), ("seed 42 (2nd)", a2), ("seed 43", b)):
    print(f"{tag:15} weight-sum {f[0]:>16}   first files {f[1]}")
assert a1 == a2 and a1[0] != b[0] and a1[1] != b[1], "seeding is not under control"
print("\nPASS")
```

---

## Cell 5 — the three runs

Stage 1's Cell 4, with `tag`, and one new assertion.

```python
import os, re, glob, shutil, subprocess, time

SEEDS      = [42, 43, 44]
NUM_BLOCK  = 4
CSV        = "/kaggle/working/lrrnet_metrics.csv"
os.makedirs("/kaggle/working/ckpts", exist_ok=True)

for seed in SEEDS:
    tag = f"m1tied_nb{NUM_BLOCK}_seed{seed}"
    print(f"\n{'='*78}\n{tag}\n{'='*78}", flush=True)
    t0 = time.time()

    # Every seed writes the SAME checkpoint filename: clear first, move the result out immediately.
    # Without this, seed 43 would be silently scored against seed 42's weights.
    for f in glob.glob(f"{REPO}/model/*.model"):
        os.remove(f)

    env = dict(os.environ, LRR_SEED=str(seed), LRR_TRAIN_NUM=str(n_tr),
               LRR_EPOCHS="4", LRR_NUM_BLOCK=str(NUM_BLOCK),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    p = subprocess.run(["python", "training_seeded.py"], cwd=REPO,
                       capture_output=True, text=True, env=env)
    print(f"train exit {p.returncode} after {(time.time()-t0)/60:.1f} min", flush=True)
    if p.returncode != 0:
        print("STDERR tail:\n", p.stderr[-3000:], flush=True)
        break

    cfg = [l for l in p.stdout.splitlines() if l.startswith("[CONFIG]")]
    print(cfg, flush=True)
    # The configuration must be on the record in the log of every run, not merely in the notebook
    # state. This is the defence written after the stage-3 OOM, applied to the new flag.
    assert cfg and "tied=1" in cfg[0], f"LRR_TIED did not reach the trainer: {cfg}"

    print([l for l in p.stdout.splitlines() if l.startswith("[EPOCH]")], flush=True)
    losses = [float(m) for m in re.findall(r"total loss: ([\d.]+)", p.stdout)]
    print(f"loss {losses[0]:.1f} -> {losses[-1]:.1f}  ({len(losses)} logged)", flush=True)

    final = glob.glob(f"{REPO}/model/final_lrr_net_*.model")
    assert len(final) == 1, f"expected one final checkpoint, found {final}"
    ckpt = f"/kaggle/working/ckpts/{tag}.model"
    shutil.copy(final[0], ckpt)

    # fuse_lrrnet.py inherits LRR_TIED from os.environ (Cell 2). If it did not, load_state_dict
    # would raise a size mismatch here -- loudly, which is the good failure mode.
    fused = f"/kaggle/working/fused/{tag}"
    subprocess.run(["python", "fuse_lrrnet.py", "--ckpt", ckpt, "--ir", TNO_IR, "--vi", TNO_VI,
                    "--out", fused, "--num_block", str(NUM_BLOCK)], cwd=REPO, check=True)

    # Evaluator runs OUTSIDE the repo: LRRNet's utils.py would shadow CrossFuse's utils package.
    subprocess.run(["python", f"{REPO}/eval_fused.py", "--crossfuse", CF, "--ir", TNO_IR,
                    "--vi", TNO_VI, "--fused", fused, "--tag", tag, "--csv", CSV],
                   cwd="/kaggle/working", check=True)
    print(f"{tag} done in {(time.time()-t0)/60:.1f} min", flush=True)
```

---

## Cell 6 — the verdict, in the session

Printed here so the answer survives even if the session dies before `/kaggle/working` is committed.
Full analysis happens locally afterwards with `analyse_table_iv.py`.

```python
import csv, numpy as np
METRICS = ["EN","MI","SD","SF","AG","VIF","SCD","Qabf","SSIM"]

# stage 1, nb=4 untied, n=5 -- the baseline of record (LRRNET_RESULTS.md)
BASE = {"EN":(6.9727,0.0254), "MI":(2.5021,0.1357), "SD":(43.5032,0.8549),
        "SF":(10.7123,0.3715), "AG":(3.9128,0.1494), "VIF":(0.7646,0.0401),
        "SCD":(1.2607,0.0469), "Qabf":(0.4325,0.0169), "SSIM":(0.9020,0.0152)}

rows = [r for r in csv.DictReader(open(CSV)) if r["tag"].startswith("m1tied_")]
print(f"{len(rows)} M1 runs\n")
print(f"{'metric':7}{'untied n=5':>22}{'M1 tied':>22}{'diff':>10}{'pooled s':>10}")
print("-" * 71)
n1, n2 = 5, len(rows)
for m in METRICS:
    v = np.array([float(r[m]) for r in rows])
    mu1, s1 = BASE[m]; mu2, s2 = v.mean(), v.std(ddof=1)
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    print(f"{m:7}{mu1:>13.4f} +-{s1:<7.4f}{mu2:>13.4f} +-{s2:<7.4f}"
          f"{mu2-mu1:>10.4f}{abs(mu2-mu1)/sp:>9.2f}s")

print("\nREADING (pre-registered, 02_DESIGN.md)\n"
      "All nine inside 3 pooled sigma -> the 29,190 removed parameters bought nothing, the\n"
      "efficiency claim stands at 2.46x, and M2's budget is open.\n"
      "M1 losing 3 or more metrics beyond 3 sigma -> kill criterion 1: report it, and build\n"
      "M2/M3 on the untied backbone instead. Do NOT rescue it by tuning the learning rate --\n"
      "an lr check is the next DIAGNOSTIC, not a retrofit.")
```

---

## After the run

| | |
|---|---|
| 1 | `kaggle kernels output mostafanada82/lrrnet-m1-tied` (set `PYTHONUTF8=1` first) |
| 2 | `analyse_table_iv.py`, extended with the M1 row |
| 3 | Fill **§6.2b of `03_EXPLAINED.md` — both the English and the Arabic copy** |
| 4 | `05_RESULTS.md`, `CONTEXT.md`, `EXPLAIN.md` |

**Run 2 (`nb = 8` tied) still needs `LRR_GPUS="0,1"` and `LRR_MULTI_GPU="1"`.** Tying removes
*parameters*, not *activations* — the backward pass still holds all eight block outputs, so the
stage-3 OOM would recur unchanged. Not a new problem; do not diagnose it as one.
