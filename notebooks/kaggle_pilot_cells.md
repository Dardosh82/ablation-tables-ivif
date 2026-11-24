# LRRNet pilot — Kaggle cells

**Purpose.** Prove the pipeline before spending ~20 GPU-hours on 14 runs. The pilot checks only the
things that would silently invalidate all of that work if they were wrong:

| # | check | what it prevents |
|---|---|---|
| 3 | **data audit** | the `lwir` -> `infrared` trap: visible images failing to load *without an error*, so training runs on garbage |
| 4 | **seed audit** | seeding one RNG and not the others, so runs differ by more than the seed and sigma is meaningless |
| 5 | **short training** | a crash in the real loss path — the timing probe skipped VGG entirely |

Inference and metrics are deliberately **not** in the pilot. If they fail they can be fixed and
re-run against saved checkpoints; no training is lost. The three checks above cannot be recovered
from after the fact.

**Notebook name:** `lrrnet-pilot` — set it **before** the first run.
**Settings:** Accelerator = GPU T4 x2, Internet = On.
**Attach dataset:** an LLVIP mirror (e.g. `afradhossain/llvip-dataset`, ~4 GB).

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

## Cell 2 — clone, patch, build the VGG loss weights

The two prep scripts come from the attached dataset **`mostafanada82/lrrnet-replication-scripts`**,
not retyped into a cell. Both were dry-run against the upstream sources locally (9 patches, all
matching exactly once and compiling; all 20 VGG tensors landing in the right positional slots);
re-typing them by hand would throw that verification away.

```python
import os, shutil, subprocess, glob
os.chdir("/kaggle/working")
if not os.path.isdir("imagefusion-LRRNet"):
    subprocess.run(["git","clone","--depth","1",
                    "https://github.com/hli1221/imagefusion-LRRNet.git"], check=True)
os.chdir("/kaggle/working/imagefusion-LRRNet")
subprocess.run(["rm","-rf",".git"], check=False)

# Locate the scripts by searching for them, never by assuming the mount path: Kaggle does not
# always mount a dataset under a folder matching its slug. Run 1 of this pilot failed here on a
# hard-coded "/kaggle/input/lrrnet-replication-scripts" while the dataset was correctly attached.
print("mounted inputs:", sorted(os.listdir("/kaggle/input")))
for name in ("patch_lrrnet.py", "make_vgg16_weights.py"):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, (f"{name} not found under /kaggle/input. Attach the dataset "
                  f"'LRRNet replication scripts' (mostafanada82/lrrnet-replication-scripts).")
    shutil.copy(hits[0], name)
    print(f"copied {name} from {hits[0]} ({os.path.getsize(name)} bytes)")

subprocess.run(["python","patch_lrrnet.py","--repo","."], check=True)
subprocess.run(["python","make_vgg16_weights.py","--out","model/vgg/vgg16.pth"], check=True)

assert os.path.exists("training_seeded.py") and os.path.exists("visdom.py")
assert os.path.exists("model/vgg/vgg16.pth")
print("\nprepared:", sorted(f for f in os.listdir(".") if f.endswith(".py")))
```

---

## Cell 3 — data audit (the silent-failure check)

The released code derives the visible path by string replacement. If the replacement produces a
path that does not exist, `cv2.imread` returns `None` and `np.reshape` fails — **or worse**, on a
partially-matching folder it loads the wrong images without complaint. This cell resolves the real
folder names, proves the pairing is 1:1, and confirms a sample pair is genuinely two modalities of
the same scene rather than the same file twice.

```python
import os, glob, numpy as np, cv2

roots = [d for d in glob.glob("/kaggle/input/**/", recursive=True)
         if os.path.basename(d.rstrip("/")) in ("infrared", "lwir")]
assert roots, "no infrared/lwir folder found under /kaggle/input -- attach an LLVIP dataset"
IR_ROOT = sorted(roots, key=len)[0]
VI_ROOT = IR_ROOT.replace("infrared", "visible").replace("lwir", "visible")
print("IR:", IR_ROOT, "\nVI:", VI_ROOT)
assert os.path.isdir(VI_ROOT), f"visible folder not found at {VI_ROOT}"

IR_DIR = IR_ROOT if glob.glob(f"{IR_ROOT}/*.png") else f"{IR_ROOT}/train"
VI_DIR = IR_DIR.replace("infrared", "visible").replace("lwir", "visible")

ir = sorted(os.path.basename(p) for p in glob.glob(f"{IR_DIR}/*.*"))
vi = sorted(os.path.basename(p) for p in glob.glob(f"{VI_DIR}/*.*"))
print(f"\nIR images {len(ir):,}   VI images {len(vi):,}   filenames identical: {ir == vi}")
assert ir and ir == vi, "IR and VIS filename sets differ -- pairing is not 1:1"

a = cv2.imread(f"{IR_DIR}/{ir[0]}", cv2.IMREAD_GRAYSCALE)
b = cv2.imread(f"{VI_DIR}/{vi[0]}", cv2.IMREAD_GRAYSCALE)
assert a is not None and b is not None, "a sample pair failed to decode"
diff = float(np.mean(np.abs(a.astype(float) - b.astype(float))))
print(f"sample {ir[0]}: IR {a.shape}, VIS {b.shape}, mean|IR-VIS| = {diff:.2f}")
assert a.shape == b.shape, "IR and VIS are not the same size -- not registered"
assert diff > 1.0, "IR and VIS are nearly identical -- the two folders are probably the same images"

# the substrings the patched code will use to turn an IR path into a VIS path
os.environ["LRR_IR_DIR"] = "infrared" if "infrared" in IR_DIR else "lwir"
os.environ["LRR_VI_DIR"] = "visible"
os.environ["LRR_DATA"]   = IR_DIR + "/"
print(f"\nLRR_DATA={os.environ['LRR_DATA']}")
print(f"LRR_IR_DIR={os.environ['LRR_IR_DIR']}  LRR_VI_DIR={os.environ['LRR_VI_DIR']}")
```

---

## Cell 4 — seed audit (no training, seconds)

**The single most important check in the pilot.** LRRNet seeds nothing and draws randomness from
*two* generators: data order from Python's `random`, weight initialisation from torch's. If only
one is seeded, two "different seed" runs also differ in ways we did not control, and the measured
sigma is not seed variance at all.

Note we do **not** compare training losses or checkpoints — cudnn nondeterminism is deliberately
left on, so two same-seed runs are *not* bit-identical and a bit-for-bit test would fail by design.
We compare the seeded **state**, which is exact.

**Every randomness source in this codebase, enumerated from the source rather than assumed:**

| source | present? | seeded by |
|---|---|---|
| Python `random` — `random.shuffle` in `load_data` **and** in `utils.load_dataset` | ✅ two call sites | `random.seed` |
| PyTorch CPU — weight init in `LRR_NET` / `Vgg16` | ✅ | `torch.manual_seed` |
| PyTorch CUDA | ✅ | `torch.cuda.manual_seed_all` |
| NumPy | not used for randomness in the training path | `np.random.seed` (defensive) |
| **`DataLoader` workers** | ❌ **none — there is no `DataLoader`** | n/a |
| **sampler / shuffle argument** | ❌ **none** | n/a |

The last two rows are the reason this list is short: LRRNet batches manually in a `for idx in
range(batch_num)` loop calling `utils.get_train_images`, so the usual worker-seeding and
`generator=` pitfalls simply do not arise. Confirmed by grep across all four source files: zero
occurrences of `DataLoader`, `manual_seed`, `random.seed` or `np.random.seed` in the released code.

```python
import random, numpy as np, torch, sys
sys.path.insert(0, ".")
import net_lista, utils

def fingerprint(seed, n=2000):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    m = net_lista.LRR_NET(3, 128, 1, 1, 4, 'cat')                 # torch RNG -> init weights
    w = sum(p.detach().double().sum().item() for p in m.parameters())
    paths, _ = utils.list_images(os.environ["LRR_DATA"])           # python RNG -> data order
    paths = paths[:n]; random.shuffle(paths)
    return round(w, 8), [os.path.basename(p) for p in paths[:4]]

a1 = fingerprint(42); a2 = fingerprint(42); b = fingerprint(43)
for tag, f in (("seed 42 (1st)", a1), ("seed 42 (2nd)", a2), ("seed 43", b)):
    print(f"{tag:15} weight-sum {f[0]:>16}   first files {f[1]}")

assert a1 == a2, "FAIL: same seed gave different state -- an RNG is still uncontrolled"
assert a1[0] != b[0], "FAIL: weight init did not change with the seed (torch RNG unseeded?)"
assert a1[1] != b[1], "FAIL: data order did not change with the seed (python RNG unseeded?)"
print("\nPASS: seeding is reproducible AND both generators respond to the seed.")
```

---

## Cell 5 — short training run with the real loss

The timing probe skipped `init_vgg16` entirely, so the VGG loss path has **never been executed**.
This is the first run that exercises it. 2 epochs over 2,000 pairs ~ 500 iterations, a few minutes.

```python
import os, subprocess, re, time
env = dict(os.environ,
           LRR_SEED="42", LRR_TRAIN_NUM="2000", LRR_EPOCHS="2", LRR_NUM_BLOCK="4",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")

t0 = time.time()
p = subprocess.run(["python","training_seeded.py"], capture_output=True, text=True, env=env)
print(f"exit {p.returncode} after {time.time()-t0:.0f}s")
if p.returncode != 0:
    print("STDOUT tail:\n", p.stdout[-2000:]); print("STDERR tail:\n", p.stderr[-4000:])
else:
    print([l for l in p.stdout.splitlines() if l.startswith("[CONFIG]") or l.startswith("[EPOCH]")])
    losses = [float(m) for m in re.findall(r"total loss: ([\d.]+)", p.stdout)]
    print(f"\nlogged losses: {len(losses)}  first {losses[:3]}  last {losses[-3:]}")
    assert losses, "no losses logged -- the training loop did not report"
    assert all(l == l for l in losses), "NaN in the loss -- the VGG path is misconfigured"
    ckpts = [f for f in os.listdir("model") if f.endswith(".model")]
    print(f"checkpoints written: {len(ckpts)} -> {sorted(ckpts)[-1] if ckpts else None}")
    assert ckpts, "no checkpoint saved"
    print("\nPILOT PASSED -- data, seeding, loss and checkpointing all work.")
```

---

## What the pilot deliberately does not do

* **No inference or metrics.** They run off saved checkpoints, so a failure there costs minutes, not
  the training. Wiring them is the next step, using **our own evaluator across every row** — the
  whole point of re-measuring Table IV rather than screening it is that one evaluator, one corpus
  and one codebase produce all four rows, so no cross-paper metric comparison is needed.
* **No claim about the loss values.** 2 epochs over 2,000 pairs is not a trained model. The only
  question is whether the machinery runs.

## Fixed before stage 1, not after

* **Checkpoint policy:** evaluate the `final_lrr_net_*.model` written at the end of training. The
  code also saves every epoch and every 2000 iterations; those are ignored. Choosing later, after
  seeing results, would make the checkpoint a free parameter that moves sigma.
* **Corpus:** LLVIP, fixed file set, `LRR_TRAIN_NUM` identical across all 14 runs.
* **cudnn left nondeterministic** on purpose — see `patch_lrrnet.py`.
