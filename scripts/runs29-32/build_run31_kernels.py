"""Build (and optionally push) the Run 31 kernels: five new seeds (45-49) of case study I's reference configuration.
Readings: ../RUN31_PREREGISTRATION.md (23b2d9b), committed before this builder existed.

    python build_run31_kernels.py --kernel a [--push]     # seeds 45, 46   (~7.6 GPU-h)
    python build_run31_kernels.py --kernel b [--push]     # seeds 47, 48   (~7.6 GPU-h)
    python build_run31_kernels.py --kernel c [--push]     # seed  49       (~3.8 GPU-h)

The recipe is R3c's notebook (notebooks/kaggle_R1b_R3c_cells.md) unchanged -- train.py --coeff_tv 5.0
--ssl_schedule paper --seed S (160 epochs), then run_inference.py and evaluate.py on RoadScene, TNO and MSRS -- with
the two guards the project learned since: each seed gets its own --outdir, and --resume is never used (train.py's
resume checkpoints carry no tag, so a second run in one outdir continues the first). A run that does not finish its
160 epochs is a kill criterion: its loss file must hold 160 epochs or the kernel fails.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T2 = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(T2)))
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
KERNELS = {"a": [45, 46], "b": [47, 48], "c": [49]}

CODE = r'''
import glob, os, shutil, subprocess, sys, csv
import torch
assert torch.cuda.is_available(), "NO GPU"
print("torch", torch.__version__, "| GPUs", torch.cuda.device_count(), torch.cuda.get_device_name(0))
def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(pattern + "   <-- input not attached")
    return hits[0]
src_code = os.path.dirname(find("/kaggle/input/**/crossfuse-tier2-code/**/CrossFuse/network.py"))
src_scripts = os.path.dirname(find("/kaggle/input/**/crossfuse-tier2-code/**/scripts/run_inference.py"))
ir_dir = find("/kaggle/input/**/MSRS/ir")
src_h5 = find("/kaggle/input/**/RSM3.h5")
WORK = "/kaggle/working/CrossFuse"
shutil.copytree(src_code, WORK, dirs_exist_ok=True)
shutil.copytree(src_scripts, "/kaggle/working/scripts", dirs_exist_ok=True)
os.makedirs(f"{WORK}/test_image", exist_ok=True)
root = os.path.dirname(os.path.dirname(ir_dir))
for ds in ("MSRS", "RoadScene", "TNO"):
    dst = f"{WORK}/test_image/{ds}"
    if not os.path.exists(dst):
        os.symlink(f"{root}/{ds}", dst)
os.makedirs(f"{WORK}/dataset", exist_ok=True)
if not os.path.exists(f"{WORK}/dataset/RSM3.h5"):
    os.symlink(src_h5, f"{WORK}/dataset/RSM3.h5")
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "kornia", "einops"], check=True)
for s in __SEEDS__:
    tag = f"R3s{s}"
    outdir = f"/kaggle/working/models_{tag}"
    assert not glob.glob(f"{outdir}/ckpt_ep*.pth"), f"{outdir} holds resume checkpoints"
    r = subprocess.run([sys.executable, "-u", "train.py", "--tag", tag, "--coeff_tv", "5.0", "--ssl_schedule", "paper",
                        "--seed", str(s), "--outdir", outdir, "--max_hours", "5.0"], cwd=WORK)
    assert r.returncode == 0, f"{tag}: train.py exit {r.returncode}"
    with open(f"{outdir}/loss_{tag}.csv") as fh:
        epochs = len(list(csv.DictReader(fh)))
    assert epochs == 160, f"{tag}: {epochs} epochs, not 160 -- kill criterion"
    ckpt = f"{outdir}/CrossFuse_RSM3_{tag}.pth"
    assert os.path.exists(ckpt), ckpt
    for d in ("test_results", "test_results_gray"):
        shutil.rmtree(f"{WORK}/{d}", ignore_errors=True)
    subprocess.run([sys.executable, "-u", "/kaggle/working/scripts/run_inference.py", "--repo", WORK, "--ckpt", ckpt,
                    "--datasets", "RoadScene", "TNO", "MSRS"], cwd="/kaggle/working", check=True)
    subprocess.run([sys.executable, "-u", "/kaggle/working/scripts/evaluate.py", "--repo", WORK,
                    "--results", f"{WORK}/test_results_gray", "--out", f"/kaggle/working/tables_{tag}_raw"],
                   cwd="/kaggle/working", check=True)
    shutil.copy(f"{outdir}/loss_{tag}.csv", "/kaggle/working/")
    for f in glob.glob(f"{outdir}/ckpt_ep*.pth"):
        os.remove(f)                                   # keep the final checkpoint only
    print(tag, "done: 160 epochs, scored on RoadScene, TNO, MSRS", flush=True)
for d in ("test_results", "test_results_gray"):
    shutil.rmtree(f"{WORK}/{d}", ignore_errors=True)
shutil.rmtree(f"{WORK}/test_image", ignore_errors=True)
'''

ap = argparse.ArgumentParser()
ap.add_argument("--kernel", required=True, choices=sorted(KERNELS))
ap.add_argument("--push", action="store_true")
a = ap.parse_args()
seeds = KERNELS[a.kernel]
slug = f"cf-run31-{a.kernel}"
out = os.path.join(T2, f"kernel_run31_{a.kernel}")
os.makedirs(out, exist_ok=True)
code = CODE.replace("__SEEDS__", repr(seeds))
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": code.strip("\n").splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(out, f"{slug}.ipynb"), "w", encoding="utf-8", newline="\n"), indent=1)
meta = {"id": f"mostafanada82/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["mostafanada82/crossfuse-tier2-code", "mostafanada82/crossfuse-testsets"],
        "kernel_sources": ["mostafanada82/crossfuse-tier2"], "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4", "docker_image": IMAGE}
json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
print("built", out, "seeds", seeds)
if a.push:
    sys.path.insert(0, os.path.join(ROOT, "studied_papers", "LRRNet", "scripts"))
    from kaggle.api.kaggle_api_extended import KaggleApi
    from build_run9_kernel import push
    api = KaggleApi()
    api.authenticate()
    push(api, out, f"mostafanada82/{slug}")
