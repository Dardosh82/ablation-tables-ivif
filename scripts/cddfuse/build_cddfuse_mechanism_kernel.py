"""Build and push the Kaggle kernel for cddfuse_untrained_mechanism.py (CDDFuse transfer test).

Runs the script byte-for-byte as committed (read with `git show HEAD:`), against MMIF-CDDFuse cloned at the
pinned commit -- whose repository carries the released IVF checkpoint and its own 25 TNO pairs, so nothing
is uploaded. T4 and image pinned as in the CrossFuse mechanism kernel: an unpinned push was once given a P100
the image's torch cannot run on.

    python studied_papers/CDDFuse/scripts/build_cddfuse_mechanism_kernel.py --push
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SLUG = "cddfuse-untrained-mechanism"
OUT = os.path.join(os.path.dirname(HERE), "kernel_mechanism")
COMMIT = "73681d28f6100461f1b214ecf77a93b495be227b"
SRC = "studied_papers/CDDFuse/scripts/cddfuse_untrained_mechanism.py"

ap = argparse.ArgumentParser()
ap.add_argument("--push", action="store_true")
args = ap.parse_args()

script = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{SRC}"], capture_output=True, check=True,
                        text=True, encoding="utf-8").stdout
if script != open(os.path.join(ROOT, SRC), encoding="utf-8").read().replace("\r\n", "\n"):
    sys.exit(f"{SRC} differs from HEAD -- commit it first; the kernel runs what is committed.")

SETUP = f'''
import os, subprocess, sys, torch
assert torch.cuda.is_available(), "NO GPU"
print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0))
os.chdir("/kaggle/working")
subprocess.run(["git", "clone", "-q", "https://github.com/Zhaozixiang1228/MMIF-CDDFuse.git"], check=True)
subprocess.run(["git", "-C", "MMIF-CDDFuse", "checkout", "-q", "{COMMIT}"], check=True)
head = subprocess.run(["git", "-C", "MMIF-CDDFuse", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
assert head == "{COMMIT}", head
assert os.path.exists("MMIF-CDDFuse/models/CDDFuse_IVF.pth"), "released checkpoint missing from the clone"
print("CDDFuse at", head, "| TNO pairs:", len(os.listdir("MMIF-CDDFuse/test_img/TNO/ir")))
'''
RUN = '''
import subprocess, sys, shutil
r = subprocess.run([sys.executable, "-u", "/kaggle/working/cddfuse_untrained_mechanism.py",
                    "--repo", "/kaggle/working/MMIF-CDDFuse", "--out", "/kaggle/working/results_cddfuse_mechanism"])
print("EXIT", r.returncode)
shutil.rmtree("/kaggle/working/MMIF-CDDFuse", ignore_errors=True)
'''


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


nb = {"cells": [cell("# CDDFuse transfer test\n\nRuns the committed script unchanged. Readings: its docstring.", "markdown"),
                cell("!pip -q install einops kornia"),
                cell(SETUP),
                cell("%%writefile /kaggle/working/cddfuse_untrained_mechanism.py\n" + script),
                cell(RUN)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
os.makedirs(OUT, exist_ok=True)
json.dump(nb, open(os.path.join(OUT, f"{SLUG}.ipynb"), "w", encoding="utf-8", newline="\n"), indent=1)
meta = {"id": f"mostafanada82/{SLUG}", "title": "CDDFuse untrained mechanism", "code_file": f"{SLUG}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "kernel_sources": [],
        "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"}
json.dump(meta, open(os.path.join(OUT, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n"), indent=2)
assert meta["machine_shape"] == "NvidiaTeslaT4"
print(f"built {OUT} (script from HEAD, {len(script)} chars)")

if args.push:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    res = api.kernels_push(OUT)
    err = getattr(res, "error", None)
    if err:
        sys.exit(f"PUSH REFUSED: {err}")
    print("pushed", getattr(res, "url", res))
