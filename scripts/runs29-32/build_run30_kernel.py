"""Build (and optionally push) the Run 30 kernel: the five untrained CrossFuse networks fused with and without the
decoder's skip. Readings: ../RUN30_PREREGISTRATION.md (06bacff).

    python build_run30_kernel.py [--push]

Same machine, image and inputs as the untrained-mechanism kernel (T4, the pinned image, the positive control's
checkpoints and per-image tables). The two scripts that changed -- run_inference.py (with --no-skip) and
no_skip_mechanism.py -- are embedded from the committed files, so the scripts dataset need not change; the build
refuses if either differs from HEAD.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
T3 = os.path.dirname(HERE)
CF = os.path.dirname(T3)
ROOT = os.path.dirname(os.path.dirname(CF))
SLUG = "crossfuse-run30-noskip"
OUT = os.path.join(T3, f"kernel_{SLUG}")
FILES = {"run_inference.py": "studied_papers/CrossFuse/Tier1_CrossFuse/scripts/run_inference.py",
         "no_skip_mechanism.py": "studied_papers/CrossFuse/Tier3_CrossFuse/scripts/no_skip_mechanism.py"}

ap = argparse.ArgumentParser()
ap.add_argument("--push", action="store_true")
a = ap.parse_args()

src = {}
for name, rel in FILES.items():
    head = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{rel}"], capture_output=True, text=True, encoding="utf-8")
    local = open(os.path.join(ROOT, rel), encoding="utf-8").read().replace("\r\n", "\n")
    if head.returncode != 0 or head.stdout.replace("\r\n", "\n") != local:
        sys.exit(f"{rel} differs from HEAD -- commit it first; the kernel runs what is committed.")
    src[name] = local

CODE = r'''
import glob, os, shutil, subprocess, sys, torch
assert torch.cuda.is_available(), "NO GPU -- set the accelerator to GPU T4"
print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0))
def find(pattern):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(pattern + "   <-- input not attached")
    return hits[0]
TOOLS, MODELS, REC = "/kaggle/working/tools", "/kaggle/working/models", "/kaggle/working/recorded"
for d in (TOOLS, MODELS, REC):
    os.makedirs(d, exist_ok=True)
shutil.copy(find("/kaggle/input/**/scripts/evaluate.py"), TOOLS)
open(f"{TOOLS}/run_inference.py", "w", encoding="utf-8").write(__RUN_INFERENCE__)
open(f"{TOOLS}/no_skip_mechanism.py", "w", encoding="utf-8").write(__NO_SKIP__)
code_src = os.path.dirname(find("/kaggle/input/**/crossfuse-tier2-code/**/CrossFuse/network.py"))
tno_src = find("/kaggle/input/**/TNO/ir").rsplit("/ir", 1)[0]
assert len(os.listdir(tno_src + "/ir")) == 25
for t in ["U42", "U43", "U44", "U45", "U46"]:
    shutil.copy(find(f"/kaggle/input/**/CrossFuse_RSM3_{t}.pth"), MODELS)
    os.makedirs(f"{REC}/tables_{t}_raw", exist_ok=True)
    shutil.copy(find(f"/kaggle/input/**/tables_{t}_raw/TNO_gray_per_image.csv"), f"{REC}/tables_{t}_raw/")
r = subprocess.run([sys.executable, "-u", f"{TOOLS}/no_skip_mechanism.py", "--repo", code_src, "--models", MODELS,
                    "--tno", tno_src, "--tools", TOOLS, "--recorded", REC, "--out", "/kaggle/working/results_run30"])
print("EXIT", r.returncode)
shutil.rmtree("/kaggle/working/results_run30/fused", ignore_errors=True)
shutil.make_archive("/kaggle/working/results_run30", "zip", "/kaggle/working/results_run30")
assert r.returncode == 0
'''
code = CODE.replace("__RUN_INFERENCE__", repr(src["run_inference.py"])).replace("__NO_SKIP__", repr(src["no_skip_mechanism.py"]))
os.makedirs(OUT, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                 "source": code.strip("\n").splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(OUT, f"{SLUG}.ipynb"), "w", encoding="utf-8", newline="\n"), indent=1)
meta = {"id": f"mostafanada82/{SLUG}", "title": "CF run30 noskip", "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": ["mostafanada82/crossfuse-tier2-code", "mostafanada82/crossfuse-testsets",
                            "mostafanada82/crossfuse-poscontrol-ckpts"],
        "kernel_sources": ["mostafanada82/tier3-poscontrol-cf"], "competition_sources": [], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}
json.dump(meta, open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
print("built", OUT)
if a.push:
    sys.path.insert(0, os.path.join(ROOT, "studied_papers", "LRRNet", "scripts"))
    from kaggle.api.kaggle_api_extended import KaggleApi
    from build_run9_kernel import push
    api = KaggleApi()
    api.authenticate()
    push(api, OUT, f"mostafanada82/{SLUG}")
