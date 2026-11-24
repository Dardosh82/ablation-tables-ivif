# LRRNet stage 1 — five seeds at the published configuration

**Goal.** Measure **sigma_LRRNet**: the run-to-run spread of the nine metrics when nothing changes
but the random seed. This is the thesis's central number for case study 2, and every later
judgement about Table IV rests on it.

**Configuration** — the published one, unchanged: `num_block = 4`, `epochs = 4`, `batch_size = 8`,
`128x128` grayscale, `lr = 1e-5`, loss weights `lam2 = 1.5`, `w_ir = 3.0`, `lam3_gram = 2000`.
Seeds **42, 43, 44, 45, 46**.

**Documented deviations from the paper** — state these wherever the number is used:

| deviation | why |
|---|---|
| **LLVIP (12,025 pairs) instead of KAIST (20,000)** | the paper's exact 20,000 images are unrecoverable: `load_data` takes `sorted(listdir(path))[:20000]` from a folder whose construction is never described, and no file list is released. KAIST is also unavailable on Kaggle in a usable IR/VIS-separated form. |
| **torchvision ImageNet VGG16 for the loss network** | LRRNet ships no `vgg16.pth`; these are the standard weights and match the positional layout `init_vgg16` expects |
| **seeding added** | the released code seeds nothing; without this "five seeds" is not a meaningful phrase |
| **cudnn left nondeterministic** | deliberate — sigma must include what a reader re-running the code would meet |

**Budget.** 657 ms/iteration measured in the pilot on real LLVIP images; 4 epochs x 1,503 batches =
6,012 iterations = **1.10 h per seed**, plus ~2 min to fuse and score 25 TNO pairs.
**~5.7 h total**, inside Kaggle's 12 h session limit.

**Evaluation.** TNO, because Table IV is published on TNO. Metrics come from **CrossFuse's own
`Evaluator`** — the same code behind every CrossFuse number in this project — so all four rows of
the re-measured table share one evaluator. Their published values are never mixed in.

**Notebook:** `lrrnet-stage1` — name it before the first run.
**Settings:** GPU T4 x2, Internet On.
**Attach:** LLVIP · `lrrnet-replication-scripts` · `crossfuse-testsets` · `crossfuse-tier2-code`.

> **Every result is printed to the log as it is produced.** A Kaggle session that dies loses
> `/kaggle/working`, but the log of a failed run is still retrievable — so a crash at seed 4 still
> leaves seeds 1–3 fully recoverable. Do not batch the printing to the end.

---

## Cell 1 — GPU guard

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

for name in ("patch_lrrnet.py", "make_vgg16_weights.py", "fuse_lrrnet.py", "eval_fused.py"):
    src = find_one(f"/kaggle/input/**/{name}", name)
    shutil.copy(src, name); print(f"copied {name} from {src}")

subprocess.run(["python","patch_lrrnet.py","--repo","."], check=True)
subprocess.run(["python","make_vgg16_weights.py","--out","model/vgg/vgg16.pth"], check=True)

# training data
IR_TRAIN = find_one("/kaggle/input/**/LLVIP/infrared/train", "LLVIP infrared/train").rstrip("/")
VI_TRAIN = IR_TRAIN.replace("infrared", "visible")
n_tr = len(os.listdir(IR_TRAIN))
assert os.path.isdir(VI_TRAIN) and n_tr == len(os.listdir(VI_TRAIN))
print(f"\ntrain: {n_tr:,} pairs at {IR_TRAIN}")

# TNO test set, from the CrossFuse test-set dataset
TNO_IR = find_one("/kaggle/input/**/TNO/ir", "TNO/ir").rstrip("/")
TNO_VI = find_one("/kaggle/input/**/TNO/vi", "TNO/vi").rstrip("/")
assert sorted(os.listdir(TNO_IR)) == sorted(os.listdir(TNO_VI)), "TNO ir/vi filenames differ"
print(f"TNO test: {len(os.listdir(TNO_IR))} pairs")

# CrossFuse repo, for utils/Evaluator.py
CF = os.path.dirname(find_one("/kaggle/input/**/CrossFuse/utils/Evaluator.py", "Evaluator.py"))
CF = os.path.dirname(CF)
print(f"CrossFuse repo: {CF}")

os.environ.update(LRR_DATA=IR_TRAIN + "/", LRR_IR_DIR="infrared", LRR_VI_DIR="visible")
print("\nready")
```

---

## Cell 3 — seed audit (repeat of the pilot check, seconds)

Cheap, and it re-proves on *this* session that both generators respond to the seed. Never skip it
because the pilot passed — the pilot ran in a different container.

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

## Cell 4 — the five runs

```python
import os, re, glob, shutil, subprocess, time

SEEDS      = [42, 43, 44, 45, 46]
NUM_BLOCK  = 4
CSV        = "/kaggle/working/lrrnet_metrics.csv"
os.makedirs("/kaggle/working/ckpts", exist_ok=True)

for seed in SEEDS:
    tag = f"nb{NUM_BLOCK}_seed{seed}"
    print(f"\n{'='*78}\n{tag}\n{'='*78}", flush=True)
    t0 = time.time()

    # Every seed writes the SAME checkpoint filename, so clear the folder first and move the
    # result out immediately. Without this, seed 43 would silently be scored against seed 42's
    # weights -- a failure that produces plausible numbers and no error.
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
    print([l for l in p.stdout.splitlines()
           if l.startswith("[CONFIG]") or l.startswith("[EPOCH]")], flush=True)
    losses = [float(m) for m in re.findall(r"total loss: ([\d.]+)", p.stdout)]
    print(f"loss {losses[0]:.1f} -> {losses[-1]:.1f}  ({len(losses)} logged)", flush=True)

    final = glob.glob(f"{REPO}/model/final_lrr_net_*.model")
    assert len(final) == 1, f"expected one final checkpoint, found {final}"
    ckpt = f"/kaggle/working/ckpts/{tag}.model"
    shutil.copy(final[0], ckpt)

    fused = f"/kaggle/working/fused/{tag}"
    subprocess.run(["python", "fuse_lrrnet.py", "--ckpt", ckpt, "--ir", TNO_IR, "--vi", TNO_VI,
                    "--out", fused, "--num_block", str(NUM_BLOCK)], cwd=REPO, check=True)

    # run the evaluator OUTSIDE the LRRNet repo: its utils.py would shadow CrossFuse's utils package
    subprocess.run(["python", f"{REPO}/eval_fused.py", "--crossfuse", CF, "--ir", TNO_IR,
                    "--vi", TNO_VI, "--fused", fused, "--tag", tag, "--csv", CSV],
                   cwd="/kaggle/working", check=True)
    print(f"{tag} done in {(time.time()-t0)/60:.1f} min", flush=True)
```

---

## Cell 5 — sigma

```python
import csv, numpy as np
METRICS = ["EN","MI","SD","SF","AG","VIF","SCD","Qabf","SSIM"]
rows = list(csv.DictReader(open(CSV)))
print(f"{len(rows)} runs\n")
print(f"{'metric':7}{'mean':>12}{'std':>12}{'min':>12}{'max':>12}{'range':>12}")
print("-" * 67)
sigma = {}
for m in METRICS:
    v = np.array([float(r[m]) for r in rows])
    sigma[m] = v.std(ddof=1)
    print(f"{m:7}{v.mean():>12.4f}{sigma[m]:>12.4f}{v.min():>12.4f}{v.max():>12.4f}"
          f"{v.max()-v.min():>12.4f}")
print("\nsigma_LRRNet (n=%d, sample std) =" % len(rows),
      {m: round(sigma[m], 5) for m in METRICS})
print("\nThis is a genuine sample standard deviation, not a range-based estimate. Every later\n"
      "judgement about Table IV uses THIS number -- CrossFuse's sigma is retired for LRRNet.")
```

---

## After stage 1

Stage 2 reuses Cell 4 with `SEEDS = [42, 43, 44]` and `NUM_BLOCK = 6`.

### ⚠️ Stages 3–4 need two extra environment variables

`num_block = 8` exhausts a 16 GB card during the backward pass. The released code pins
`CUDA_VISIBLE_DEVICES = "0"` and leaves its own `DataParallel` line commented out, so a second card
cannot help it. `patch_lrrnet.py` restores that line behind `LRR_MULTI_GPU`; this is **not** a
deviation, because `net_lista.py` contains no BatchNorm, InstanceNorm, LayerNorm or Dropout — no
batch-dependent layer at all — so splitting a batch of 8 across two devices and averaging gradients
is the arithmetically identical update. Defaults preserve the released behaviour, so stages 1–2 are
unaffected.

```python
import os, re, glob, shutil, subprocess, time

SEEDS      = [42, 43, 44]
NUM_BLOCK  = 8                       # stage 3; use 2 for stage 4
CSV        = "/kaggle/working/lrrnet_metrics.csv"
os.makedirs("/kaggle/working/ckpts", exist_ok=True)

for seed in SEEDS:
    tag = f"nb{NUM_BLOCK}_seed{seed}"
    print(f"\n{'='*78}\n{tag}\n{'='*78}", flush=True)
    t0 = time.time()

    # Every seed writes the SAME checkpoint filename: clear first, move the result out immediately.
    # Without this, seed 43 would be silently scored against seed 42's weights.
    for f in glob.glob(f"{REPO}/model/*.model"):
        os.remove(f)

    env = dict(os.environ,
               LRR_SEED=str(seed), LRR_TRAIN_NUM=str(n_tr),
               LRR_EPOCHS="4", LRR_NUM_BLOCK=str(NUM_BLOCK),
               LRR_GPUS="0,1", LRR_MULTI_GPU="1",          # <-- stages 3-4 only
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    p = subprocess.run(["python", "training_seeded.py"], cwd=REPO,
                       capture_output=True, text=True, env=env)
    print(f"train exit {p.returncode} after {(time.time()-t0)/60:.1f} min", flush=True)

    gpus = [l for l in p.stdout.splitlines() if l.startswith("[GPUS]")]
    print(gpus, flush=True)
    if p.returncode != 0:
        print("STDERR tail:\n", p.stderr[-3000:], flush=True)
        break
    # Fail loudly rather than silently training on one card: if the patch did not take effect the
    # run would either OOM again or produce a row that is not comparable with the others.
    assert gpus and "parallel= True" in gpus[0], \
        f"DataParallel did not engage: {gpus} — check the dataset was updated to the new patch_lrrnet.py"

    print([l for l in p.stdout.splitlines()
           if l.startswith("[CONFIG]") or l.startswith("[EPOCH]")], flush=True)
    losses = [float(m) for m in re.findall(r"total loss: ([\d.]+)", p.stdout)]
    print(f"loss {losses[0]:.1f} -> {losses[-1]:.1f}  ({len(losses)} logged)", flush=True)

    final = glob.glob(f"{REPO}/model/final_lrr_net_*.model")
    assert len(final) == 1, f"expected one final checkpoint, found {final}"
    ckpt = f"/kaggle/working/ckpts/{tag}.model"
    shutil.copy(final[0], ckpt)

    # DataParallel prefixes state_dict keys with "module."; fuse_lrrnet.py strips it.
    fused = f"/kaggle/working/fused/{tag}"
    subprocess.run(["python", "fuse_lrrnet.py", "--ckpt", ckpt, "--ir", TNO_IR, "--vi", TNO_VI,
                    "--out", fused, "--num_block", str(NUM_BLOCK)], cwd=REPO, check=True)

    # Evaluator runs OUTSIDE the repo: LRRNet's utils.py would shadow CrossFuse's utils package.
    subprocess.run(["python", f"{REPO}/eval_fused.py", "--crossfuse", CF, "--ir", TNO_IR,
                    "--vi", TNO_VI, "--fused", fused, "--tag", tag, "--csv", CSV],
                   cwd="/kaggle/working", check=True)
    print(f"{tag} done in {(time.time()-t0)/60:.1f} min", flush=True)
```

Success shows in the first minute as `[GPUS] 2 parallel= True`. If it still does not fit, the
honest report is that the row cannot be reproduced on this hardware — which is itself the result.

Once all four rows exist, Table IV is fully re-measured: our rows, our evaluator, our sigma, no
proxy anywhere.
