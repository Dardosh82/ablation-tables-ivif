"""Build (and optionally push) the Run 32 kernel: does a downstream detector agree with the metric verdicts?
Readings: ../RUN32_PREREGISTRATION.md (19940f4), committed alone before this builder existed.

    python build_run32_kernel.py [--push]

One T4 kernel, nothing trained:
  1. the subset -- LLVIP's test split, random.Random(20261125).sample(sorted names, 1000) drawn before any fusion,
     each pair downscaled 2x to 640x512 (cv2.INTER_AREA; infrared saved gray, visible saved colour, both PNG), the
     official person boxes scaled by 0.5;
  2. fusion -- CrossFuse with run_inference.py exactly as Runs 28/30 (embedded from HEAD; the build refuses if it
     differs), the 15 networks of the registration (5 trained, 5 untrained skip on, the same 5 skip off), every
     checkpoint SHA-256-checked against the copies the paper's rows came from; LRRNet with fuse_lrrnet.py exactly as
     Runs 16/25 (parameter count checked from the loader's own line), 20 networks: published d4 seeds 42-46, untrained
     d4 seeds 400-404 built as in Run 25, tied and untied d8 seeds 62-66;
  3. detection -- YOLOv8s (ultralytics==8.3.0, COCO weights, never fine-tuned), class person only, conf 0.001, NMS IoU
     0.6, every image replicated to three channels; AP50 (all-point interpolation, IoU >= 0.5) per network, plus the
     infrared-only and visible-only references. Each network is detected right after it is fused and its images are
     then deleted, so the disk never holds more than one network's output. Raw detections are kept (one .npz per
     network) so the AP can be recomputed locally.
Kill criteria (registered): a checkpoint failing its SHA-256 or parameter-count check stops the kernel before any
fusion; a network whose every output is blank (zero variance -- a NaN output min-max-normalised and cast to uint8 is
blank too) is recorded and excluded; fewer than five networks scored in any arm fails the kernel after the CSV is written.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LRR = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(LRR))
sys.path.insert(0, HERE)
SLUG = "mostafanada82/lrrnet-run32-detect"
OUT = os.path.join(LRR, "kernel_run32_detect")
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
RUN_INFERENCE = "studied_papers/CrossFuse/Tier1_CrossFuse/scripts/run_inference.py"

# SHA-256 of the checkpoint files the paper's CrossFuse rows were scored from (local copies of the same kernel outputs)
CF_SHA = {
    "A1b": "3859d38d080037d067da11fd8c339cf9b1b049aa1e549c9713c3e957ca5b75fe",
    "A1c": "76a584ed7a6e7b8a8f1997994ce70736cc6e8f7c607a8acaaa8a54039768528b",
    "A1d": "62069a4b58a08b3753a6b9cc610d125781f43e41a2464b242da109858a101783",
    "PC45": "4f702ba756b50dedc806d32bdbf5aa7dd3890c65672376eee367e9ac7aec5136",
    "PC46": "f65b847360b1b70a54a11562964ee1ab7f888b0c48b9fa6e5ce4b7ea3990ee0a",
    "U42": "7d807f0555bc35ae608f600e6ba47cf185a852d113190216822410d716d91107",
    "U43": "eba89caa1c00909e037aebbf91ed2cccb0dbe3eda9b066b3325be16adbfc3ee0",
    "U44": "cae1c1be986295b244e2928d42852c1fb40bb75e5b5596c6631e47020e4bcf10",
    "U45": "79b8b48faec6f9b286b83d114ff7d42bd0176503f496e3d95b8fca0a3e8705a0",
    "U46": "ec01fad7f615b46488445f6ee4955c013bb025601318795191fe084f195f8f89",
}

CODE = r'''
import csv, glob, hashlib, os, random, re, shutil, subprocess, sys, time
import xml.etree.ElementTree as ET
import numpy as np, cv2, torch
assert torch.cuda.is_available(), "NO GPU -- set the accelerator to GPU T4"
TORCH = torch.__version__
print("torch", TORCH, "| GPU", torch.cuda.get_device_name(0), flush=True)
W = "/kaggle/working"
OUTD = f"{W}/run32"
os.makedirs(f"{OUTD}/preds", exist_ok=True)
LOG = open(f"{OUTD}/checks.txt", "w")
def note(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()
def find(pattern, what=None):
    hits = sorted(glob.glob(pattern, recursive=True), key=len)
    assert hits, f"{what or pattern} not found   <-- input not attached"
    return hits[0]
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# ---- detector, installed first so a version problem stops the kernel before any GPU-hour is spent ---------------
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "--no-deps", "ultralytics==8.3.0",
                "ultralytics-thop==2.0.8", "py-cpuinfo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "kornia", "einops"], check=True)
import ultralytics
from ultralytics import YOLO
assert ultralytics.__version__ == "8.3.0", ultralytics.__version__
assert torch.__version__ == TORCH, "pip changed torch"
os.chdir(W)
DET = YOLO("yolov8s.pt")
note("ultralytics", ultralytics.__version__, "| yolov8s.pt sha256", sha256(f"{W}/yolov8s.pt"),
     "| person class name:", DET.names[0])
assert DET.names[0] == "person"

# ---- 1. the subset, drawn before any fusion ------------------------------------------------------------------------
IRT = find("/kaggle/input/**/LLVIP/infrared/test", "LLVIP/infrared/test")
VIT = find("/kaggle/input/**/LLVIP/visible/test", "LLVIP/visible/test")
ANN = find("/kaggle/input/**/LLVIP/Annotations", "LLVIP/Annotations")
names = sorted(os.listdir(IRT))
assert names == sorted(os.listdir(VIT)), "LLVIP test: infrared and visible names differ"
note("LLVIP test pairs:", len(names))
subset = random.Random(20261125).sample(names, 1000)
open(f"{OUTD}/subset.txt", "w").write("\n".join(subset) + "\n")
note("subset: 1000 pairs, first five", subset[:5], "| list sha256",
     hashlib.sha256(("\n".join(subset) + "\n").encode()).hexdigest())
SUB = f"{W}/LLVIP1000"
os.makedirs(f"{SUB}/ir", exist_ok=True); os.makedirs(f"{SUB}/vi", exist_ok=True)
GT = {}
for n in subset:
    stem = os.path.splitext(n)[0]
    ir = cv2.imread(f"{IRT}/{n}", cv2.IMREAD_GRAYSCALE)
    vi = cv2.imread(f"{VIT}/{n}", cv2.IMREAD_COLOR)
    assert ir is not None and vi is not None and ir.shape == (1024, 1280) and vi.shape[:2] == (1024, 1280), n
    cv2.imwrite(f"{SUB}/ir/{stem}.png", cv2.resize(ir, (640, 512), interpolation=cv2.INTER_AREA))
    cv2.imwrite(f"{SUB}/vi/{stem}.png", cv2.resize(vi, (640, 512), interpolation=cv2.INTER_AREA))
    root = ET.parse(f"{ANN}/{stem}.xml").getroot()
    boxes = []
    for o in root.iter("object"):
        assert o.findtext("name").strip() == "person", (stem, o.findtext("name"))
        b = o.find("bndbox")
        boxes.append([float(b.findtext(k)) * 0.5 for k in ("xmin", "ymin", "xmax", "ymax")])
    GT[stem] = np.array(boxes, dtype=np.float64).reshape(-1, 4)
STEMS = sorted(GT)
NGT = sum(len(v) for v in GT.values())
note("ground truth: person boxes", NGT, "| images with none", sum(len(v) == 0 for v in GT.values()))
with open(f"{OUTD}/ground_truth.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["image", "x1", "y1", "x2", "y2"])
    for s in STEMS:
        for b in GT[s]:
            w.writerow([s] + [f"{x:.2f}" for x in b])

# ---- detection and AP50 ----------------------------------------------------------------------------------------
def iou(b, B):
    x1 = np.maximum(b[0], B[:, 0]); y1 = np.maximum(b[1], B[:, 1])
    x2 = np.minimum(b[2], B[:, 2]); y2 = np.minimum(b[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter)
def ap50(img_idx, boxes, scores):
    order = np.argsort(-scores, kind="stable")
    used = {s: np.zeros(len(GT[s]), bool) for s in STEMS}
    tp = np.zeros(len(order))
    for k, i in enumerate(order):
        s = STEMS[img_idx[i]]; g = GT[s]
        if len(g) == 0:
            continue
        o = iou(boxes[i], g); j = int(np.argmax(o))
        if o[j] >= 0.5 and not used[s][j]:
            used[s][j] = True; tp[k] = 1
    ctp = np.cumsum(tp); fp = np.arange(1, len(tp) + 1) - ctp
    rec = ctp / NGT; prec = ctp / np.maximum(ctp + fp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]]); mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    k = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[k + 1] - mrec[k]) * mpre[k + 1]))
ROWS = []
def detect(tag, family, arm, folder):
    stds, ims = [], []
    for s in STEMS:
        p = f"{folder}/{s}.png"
        assert os.path.exists(p), f"{tag}: missing {s}.png"
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        assert g is not None and g.shape == (512, 640), f"{tag}: {s} shape {None if g is None else g.shape}"
        stds.append(float(g.std())); ims.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    blank = all(v == 0 for v in stds)
    ii, bb, ss = [], [], []
    for start in range(0, len(ims), 32):
        res = DET.predict(ims[start:start + 32], imgsz=640, conf=0.001, iou=0.6, classes=[0], verbose=False)
        for off, r in enumerate(res):
            xy = r.boxes.xyxy.cpu().numpy().astype(np.float64); sc = r.boxes.conf.cpu().numpy().astype(np.float64)
            ii.append(np.full(len(sc), start + off, np.int32)); bb.append(xy); ss.append(sc)
    ii, bb, ss = np.concatenate(ii), np.concatenate(bb).reshape(-1, 4), np.concatenate(ss)
    np.savez_compressed(f"{OUTD}/preds/{tag}.npz", img=ii, boxes=bb, score=ss)
    a = ap50(ii, bb, ss)
    ROWS.append(dict(family=family, arm=arm, tag=tag, n_images=len(STEMS), n_gt=NGT, n_det=len(ss),
                     ap50=f"{a:.6f}", mean_std=f"{np.mean(stds):.4f}", blank=int(blank)))
    note(f"{tag:28s} {family:9s} {arm:16s} AP50 {a:.4f}  detections {len(ss):7d}  mean std {np.mean(stds):.2f}"
         + ("   BLANK -- excluded (kill criterion)" if blank else ""))
    with open(f"{OUTD}/run32_ap50.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROWS[0])); w.writeheader(); w.writerows(ROWS)

# references: each band alone, gray as both networks read it
REF = f"{W}/ref_vis"; os.makedirs(REF, exist_ok=True)
for s in STEMS:
    cv2.imwrite(f"{REF}/{s}.png", cv2.imread(f"{SUB}/vi/{s}.png", cv2.IMREAD_GRAYSCALE))
detect("REF_IR", "reference", "infrared only", f"{SUB}/ir")
detect("REF_VIS", "reference", "visible only", REF)
shutil.rmtree(REF)

# ---- 2a. CrossFuse ---------------------------------------------------------------------------------------------
CF_SHA = __CF_SHA__
TOOLS = f"{W}/tools"; os.makedirs(TOOLS, exist_ok=True)
open(f"{TOOLS}/run_inference.py", "w", encoding="utf-8").write(__RUN_INFERENCE__)
CFREPO = f"{W}/CrossFuse"
shutil.copytree(os.path.dirname(find("/kaggle/input/**/crossfuse-tier2-code/**/CrossFuse/network.py")), CFREPO,
                dirs_exist_ok=True)
os.makedirs(f"{CFREPO}/test_image", exist_ok=True)
os.symlink(SUB, f"{CFREPO}/test_image/LLVIP1000")
CK = {}
for t, want in CF_SHA.items():
    hits = [p for p in glob.glob(f"/kaggle/input/**/CrossFuse_RSM3_{t}.pth", recursive=True) if sha256(p) == want]
    assert hits, f"kill: no copy of CrossFuse_RSM3_{t}.pth with the recorded SHA-256 -- nothing is fused"
    CK[t] = hits[0]
note("CrossFuse: all 10 checkpoints present with the recorded SHA-256")

# ---- 2b. LRRNet: clone, patch, build the untrained networks, check every checkpoint before any fusion -------------
if not os.path.isdir(f"{W}/imagefusion-LRRNet"):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/hli1221/imagefusion-LRRNet.git"], cwd=W,
                   check=True)
LREPO = f"{W}/imagefusion-LRRNet"
for name in ("patch_lrrnet.py", "fuse_lrrnet.py"):
    shutil.copy(find(f"/kaggle/input/**/lrrnet-replication-scripts/**/{name}", name), f"{LREPO}/{name}")
subprocess.run([sys.executable, "patch_lrrnet.py", "--repo", "."], cwd=LREPO, check=True)
os.makedirs(f"{W}/_no_training_corpus", exist_ok=True)
open(f"{LREPO}/make_untrained.py", "w").write(__MAKE__)
p = subprocess.run([sys.executable, "make_untrained.py", "400,401,402,403,404"], cwd=LREPO, capture_output=True,
                   text=True, env=dict(os.environ, LRR_TIED="0", LRR_DATA=f"{W}/_no_training_corpus"))
print(p.stdout, p.stderr[-2000:])
assert p.returncode == 0, "kill: an untrained LRRNet failed its parameter-count or determinism check"
LRR = []   # (tag, arm, ckpt, num_block, tied, expected parameters)
for s in range(42, 47):
    LRR.append((f"nb4_seed{s}", "published d4", find(f"/kaggle/input/**/lrrnet-run16-ckpts/**/nb4_seed{s}.model"),
                4, "0", "49,204"))
for s in range(400, 405):
    LRR.append((f"nb4_untrained_seed{s}", "untrained d4", f"{W}/ckpts/nb4_untrained_seed{s}.model", 4, "0", "49,204"))
for s in range(62, 67):
    LRR.append((f"m1tied_nb8_seed{s}", "tied d8", find(f"/kaggle/input/**/lrrnet-run17-ckpts/**/m1tied_nb8_seed{s}.model"),
                8, "1", "20,014"))
    LRR.append((f"nb8_seed{s}", "untied d8", find(f"/kaggle/input/**/lrrnet-run17-ckpts/**/nb8_seed{s}.model"),
                8, "0", "88,124"))
note("LRRNet: all 20 checkpoints present")

# ---- fusion + detection, one network at a time; LRRNet first (minutes), so a format problem shows before CrossFuse's
# four GPU-hours are spent --------------------------------------------------------------------------------------------
t0 = time.time()
for tag, arm, ck, nb, tied, nparam in LRR:
    out = f"{W}/fused_lrr/{tag}"
    shutil.rmtree(out, ignore_errors=True)
    q = subprocess.run([sys.executable, "fuse_lrrnet.py", "--ckpt", ck, "--ir", f"{SUB}/ir", "--vi", f"{SUB}/vi",
                        "--out", out, "--num_block", str(nb)], cwd=LREPO, capture_output=True, text=True,
                       env=dict(os.environ, LRR_TIED=tied, LRR_DATA=f"{W}/_no_training_corpus"))
    assert q.returncode == 0, f"{tag}: fuse exit {q.returncode}: {q.stderr[-1500:]}"
    got = re.search(r"\(([\d,]+) parameters, num_block=(\d)\)", q.stdout)
    assert got and got.group(1) == nparam and int(got.group(2)) == nb, \
        f"kill: {tag} loaded {got.group(0) if got else 'nothing'}, expected {nparam}"
    detect("LRR_" + tag, "LRRNet", arm, out)
    shutil.rmtree(out)
    note(f"   elapsed {(time.time() - t0) / 3600:.2f} h")
CF_RUNS = [(t, "trained", False) for t in ("A1b", "A1c", "A1d", "PC45", "PC46")]
CF_RUNS += [(f"U{s}", "untrained skip on", False) for s in range(42, 47)]
CF_RUNS += [(f"U{s}", "untrained skip off", True) for s in range(42, 47)]
for t, arm, noskip in CF_RUNS:
    tag = f"CF_{t}" + ("_OFF" if noskip else ("_ON" if t.startswith("U") else ""))
    for d in ("test_results", "test_results_gray"):
        shutil.rmtree(f"{CFREPO}/{d}", ignore_errors=True)
    cmd = [sys.executable, "-u", f"{TOOLS}/run_inference.py", "--repo", CFREPO, "--ckpt", CK[t],
           "--datasets", "LLVIP1000"] + (["--no-skip"] if noskip else [])
    r = subprocess.run(cmd, cwd=W, capture_output=True, text=True)
    assert r.returncode == 0, f"{tag}: run_inference exit {r.returncode}: {r.stderr[-1500:]}"
    print(r.stdout.strip().splitlines()[-1], flush=True)
    detect(tag, "CrossFuse", arm, f"{CFREPO}/test_results_gray/LLVIP1000")
    note(f"   elapsed {(time.time() - t0) / 3600:.2f} h")
for d in ("test_results", "test_results_gray"):
    shutil.rmtree(f"{CFREPO}/{d}", ignore_errors=True)

# ---- kill criterion: five scored networks per arm -------------------------------------------------------------
shutil.rmtree(SUB); shutil.rmtree(CFREPO, ignore_errors=True); shutil.rmtree(LREPO, ignore_errors=True)
shutil.rmtree(f"{W}/ckpts", ignore_errors=True); shutil.rmtree(f"{W}/fused_lrr", ignore_errors=True)
arms = {}
for r in ROWS:
    if r["family"] != "reference" and not r["blank"]:
        arms[(r["family"], r["arm"])] = arms.get((r["family"], r["arm"]), 0) + 1
note("scored networks per arm:", arms)
assert len(arms) == 7 and all(v == 5 for v in arms.values()), "kill: fewer than five networks scored in an arm"
note("RUN 32 KERNEL DONE: 35 networks and 2 references scored on 1000 LLVIP test pairs")
LOG.close()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    head = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{RUN_INFERENCE}"], capture_output=True, text=True,
                          encoding="utf-8")
    local = open(os.path.join(ROOT, RUN_INFERENCE), encoding="utf-8").read().replace("\r\n", "\n")
    if head.returncode != 0 or head.stdout.replace("\r\n", "\n") != local:
        sys.exit(f"{RUN_INFERENCE} differs from HEAD -- commit it first; the kernel runs what is committed.")
    assert "--no-skip" in local
    from build_run25_kernels import MAKE
    code = (CODE.replace("__CF_SHA__", repr(CF_SHA)).replace("__RUN_INFERENCE__", repr(local))
            .replace("__MAKE__", repr(MAKE)))
    os.makedirs(OUT, exist_ok=True)
    code_file = SLUG.split("/")[1] + ".ipynb"
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": code.strip("\n").splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(os.path.join(OUT, code_file), "w", encoding="utf-8", newline="\n"), indent=1)
    meta = {"id": SLUG, "title": "LRRNet_run32_detect", "code_file": code_file, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "keywords": [],
            "dataset_sources": ["afradhossain/llvip-dataset", "mostafanada82/crossfuse-tier2-code",
                                "mostafanada82/crossfuse-poscontrol-ckpts", "mostafanada82/lrrnet-replication-scripts",
                                "mostafanada82/lrrnet-run16-ckpts", "mostafanada82/lrrnet-run17-ckpts",
                                "mostafanada82/crossfuse-run32-ckpts"],   # correction f4c84fc
            "kernel_sources": [],
            "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4", "docker_image": IMAGE}
    json.dump(meta, open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("built", OUT)
    if a.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        from build_run9_kernel import push
        api = KaggleApi()
        api.authenticate()
        push(api, OUT, SLUG)


if __name__ == "__main__":
    sys.exit(main())
