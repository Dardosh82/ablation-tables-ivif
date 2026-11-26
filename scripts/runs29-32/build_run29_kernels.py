"""Build (and optionally push) the Run 29 per-image scoring kernels. Readings: ../RUN29_PREREGISTRATION.md (5b1e990).

    python build_run29_kernels.py --source 16   [--push]   # Table IV (nb 2/4/6/8) + M1 + free depth, TNO + RoadScene
    python build_run29_kernels.py --source 17b  [--push]   # tied vs untied d8, seeds 62-73, TNO + RoadScene
    python build_run29_kernels.py --source 20   [--push]   # tied d8, seeds 74-85, TNO + RoadScene
    python build_run29_kernels.py --source 25   [--push]   # untrained LRRNets, seeds 400-419, TNO

CPU only, no GPU quota: each kernel re-scores the fused images that the earlier run's FUSE kernel already wrote (its
output is attached as a kernel source), with eval_fused.py's new --per-image option. The eval script is embedded in
the notebook rather than taken from the scripts dataset, so the dataset (whose older copy the earlier runs used) need
not change. Every checkpoint's new mean row is compared with the record locally (the integrity stop rule) before any
bootstrap is computed.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LRR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
SOURCES = {   # source run -> (its fuse kernel, number of fused folders per test set, test sets)
    "16": ("mostafanada82/lrrnet-run16-fuse", 78, ["TNO", "RoadScene"]),
    "17b": ("mostafanada82/lrrnet-run17b-fuse", 24, ["TNO", "RoadScene"]),
    "20": ("mostafanada82/lrrnet-run20-fuse", 12, ["TNO", "RoadScene"]),
    "25": ("mostafanada82/lrrnet-run25-fuse", 20, ["TNO"]),
}
DATA = ["mostafanada82/crossfuse-testsets", "mostafanada82/crossfuse-tier2-code"]

SCORE = r'''
import os, glob, subprocess, sys, csv as _csv
from multiprocessing import Pool
def find_one(pattern, what):
    hits = glob.glob(pattern, recursive=True)
    assert hits, f"{what} not found: {pattern}"
    return sorted(hits, key=len)[0]
EVAL = "/kaggle/working/eval_fused.py"
open(EVAL, "w", encoding="utf-8").write(__EVAL_SOURCE__)
CF = os.path.dirname(os.path.dirname(find_one("/kaggle/input/**/CrossFuse/utils/Evaluator.py", "Evaluator.py")))
SETS = {}
for ds, n in (("TNO", 25), ("RoadScene", 40)):
    if ds not in __SETS__:
        continue
    ir = find_one(f"/kaggle/input/**/{ds}/ir", f"{ds}/ir").rstrip("/")
    vi = find_one(f"/kaggle/input/**/{ds}/vi", f"{ds}/vi").rstrip("/")
    assert sorted(os.listdir(ir)) == sorted(os.listdir(vi)) and len(os.listdir(ir)) == n, f"{ds}: not {n} pairs"
    SETS[ds] = (ir, vi)
FUSED = os.path.dirname(find_one("/kaggle/input/**/fused/TNO", "the fuse kernel's output"))
jobs = []
for ds, (ir, vi) in SETS.items():
    folders = sorted(glob.glob(f"{FUSED}/{ds}/*"))
    assert len(folders) == __N__, f"{ds}: {len(folders)} fused folders, expected __N__"
    jobs += [(ds, ir, vi, d, os.path.basename(d)) for d in folders]
os.makedirs("/kaggle/working/rows", exist_ok=True)
def score(job):
    ds, ir, vi, d, tag = job
    p = subprocess.run([sys.executable, EVAL, "--crossfuse", CF, "--ir", ir, "--vi", vi, "--fused", d, "--tag", tag,
                        "--csv", f"/kaggle/working/rows/{ds}__{tag}.csv",
                        "--per-image", f"/kaggle/working/rows/{ds}__{tag}__img.csv"],
                       cwd="/kaggle/working", capture_output=True, text=True)
    return job, p.returncode, p.stderr[-300:]
with Pool(os.cpu_count()) as pool:
    for (ds, _, _, _, tag), code, err in pool.imap_unordered(score, jobs):
        print(ds, tag, "exit", code, err if code else "", flush=True)
for ds in SETS:
    for kind, suffix in (("means", ".csv"), ("perimage", "__img.csv")):
        rows = []
        for f in sorted(glob.glob(f"/kaggle/working/rows/{ds}__*{suffix}")):
            if kind == "means" and f.endswith("__img.csv"):
                continue
            rows += list(_csv.DictReader(open(f)))
        expect = __N__ if kind == "means" else None
        assert expect is None or len(rows) == expect, f"{ds} {kind}: {len(rows)} rows, expected {expect}"
        out = f"/kaggle/working/run29_src__SRC___{ds}_{kind}.csv"
        with open(out, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("wrote", os.path.basename(out), len(rows), "rows")
import shutil; shutil.rmtree("/kaggle/working/rows", ignore_errors=True)
'''


def notebook(code):
    cell = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": code.strip("\n").splitlines(keepends=True)}
    return {"cells": [cell], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                         "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(SOURCES))
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    fuse_kernel, n, sets = SOURCES[a.source]
    slug = f"mostafanada82/lrrnet-run29-score-src{a.source}"
    out = os.path.join(LRR, f"kernel_run29_score_src{a.source}")
    os.makedirs(out, exist_ok=True)
    eval_src = open(os.path.join(HERE, "eval_fused.py"), encoding="utf-8").read()
    assert "--per-image" in eval_src, "eval_fused.py lacks the Run 29 --per-image option"
    code = (SCORE.replace("__EVAL_SOURCE__", repr(eval_src)).replace("__N__", str(n))
            .replace("__SETS__", repr(sets)).replace("__SRC__", a.source))
    code_file = slug.split("/")[1] + ".ipynb"
    json.dump(notebook(code), open(os.path.join(out, code_file), "w", encoding="utf-8"), indent=1)
    meta = {"id": slug, "title": f"LRRNet_run29_score_src{a.source}", "code_file": code_file, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_internet": False, "enable_gpu": False,
            "enable_tpu": False, "keywords": [], "dataset_sources": DATA, "kernel_sources": [fuse_kernel],
            "competition_sources": [], "model_sources": [], "docker_image": IMAGE}
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print("built", out)
    if a.push:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        from build_run9_kernel import push
        push(api, out, slug)


if __name__ == "__main__":
    sys.exit(main())
