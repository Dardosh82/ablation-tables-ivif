"""Rebuild, inside this release, the folder layout the Runs 29-32 analyses were written against, then run them.

    python scripts/runs29-32/make_tree.py            # builds _tree/studied_papers/... (copies, ~60 MB)
    python scripts/runs29-32/make_tree.py --run      # ... and runs the four analyses and the sigma recompute

The analyses resolve paths in the project's own layout (studied_papers/<area>/...). data/ holds the same files under
the same names, so this script copies data/<area> to _tree/studied_papers/<area> and places each script in the folder
it ran from. Nothing is trained, fused or scored: every analysis reads recorded CSV, text and detection files only.
Each analysis rewrites its own output (RUN29_ANALYSIS.txt ... RUN32_ANALYSIS.txt, SIGMA8_RECOMPUTE.txt) inside _tree;
the recorded copies in data/ are left untouched, so the two can be compared (--run does that and reports it).
"""
import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(os.path.dirname(HERE))
SCRIPT_DIRS = {   # script (shipped in scripts/runs29-32/) -> the folder under studied_papers/ it ran from
    "build_run29_kernels.py": "LRRNet/scripts", "analyse_run29.py": "LRRNet/scripts",
    "build_run32_kernel.py": "LRRNet/scripts", "analyse_run32.py": "LRRNet/scripts",
    "run_inference.py": "CrossFuse/Tier1_CrossFuse/scripts",
    "no_skip_mechanism.py": "CrossFuse/Tier3_CrossFuse/scripts", "build_run30_kernel.py": "CrossFuse/Tier3_CrossFuse/scripts",
    "analyse_run30.py": "CrossFuse/Tier3_CrossFuse/scripts",
    "build_run31_kernels.py": "CrossFuse/Tier2_CrossFuse/scripts", "analyse_run31.py": "CrossFuse/Tier2_CrossFuse/scripts",
    "recompute_with_sigma8.py": "CrossFuse/Tier2_CrossFuse/scripts",
    # case study I's sigma-scaled analyses, which recompute_with_sigma8.py re-runs with only the sigma block swapped
    "audit_sigma_estimator.py": "CrossFuse/Tier3_CrossFuse/scripts", "analyse_tier3.py": "CrossFuse/Tier3_CrossFuse/scripts",
    "analyse_a4.py": "CrossFuse/Tier3_CrossFuse/scripts", "analyse_a3_a5.py": "CrossFuse/Tier3_CrossFuse/scripts",
    "analyse_A1b.py": "CrossFuse/Tier3_CrossFuse/scripts", "analyse_run11_k0.py": "CrossFuse/Tier3_CrossFuse/scripts",
}
RUNS = [   # (script, recorded output in data/, output written in _tree)
    ("LRRNet/scripts/analyse_run29.py", "LRRNet/results_run29/RUN29_ANALYSIS.txt"),
    ("CrossFuse/Tier3_CrossFuse/scripts/analyse_run30.py", "CrossFuse/Tier3_CrossFuse/results_run30/RUN30_ANALYSIS.txt"),
    ("CrossFuse/Tier2_CrossFuse/scripts/analyse_run31.py", "CrossFuse/Tier2_CrossFuse/results_run31/RUN31_ANALYSIS.txt"),
    ("LRRNet/scripts/analyse_run32.py", "LRRNet/results_run32/RUN32_ANALYSIS.txt"),
    ("CrossFuse/Tier2_CrossFuse/scripts/recompute_with_sigma8.py",
     "CrossFuse/Tier2_CrossFuse/results_run31/SIGMA8_RECOMPUTE.txt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="also run the analyses and compare with the recorded outputs")
    a = ap.parse_args()
    tree = os.path.join(REL, "_tree", "studied_papers")
    shutil.rmtree(os.path.join(REL, "_tree"), ignore_errors=True)
    shutil.copytree(os.path.join(REL, "data"), tree)
    for name, sub in SCRIPT_DIRS.items():
        src = os.path.join(HERE, name)
        if not os.path.isfile(src):
            sys.exit(f"missing {src}")
        os.makedirs(os.path.join(tree, sub), exist_ok=True)
        shutil.copy(src, os.path.join(tree, sub, name))
    print("built", os.path.relpath(tree, REL))
    if not a.run:
        return 0
    bad = 0
    for script, rec in RUNS:
        out_rec = os.path.join(REL, "data", rec)
        out_new = os.path.join(tree, rec)
        if os.path.exists(out_new):
            os.remove(out_new)
        print(f"\n== {script}  (running ...)", flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(tree, script)], cwd=os.path.dirname(os.path.join(tree, script)),
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if script.endswith("recompute_with_sigma8.py"):     # it prints rather than writes; capture it
            open(out_new, "w", encoding="utf-8").write(r.stdout)
        same = os.path.isfile(out_new) and filecmp.cmp(out_rec, out_new, shallow=False) if os.path.isfile(out_rec) else None
        if not same and os.path.isfile(out_new) and os.path.isfile(out_rec):   # line endings may differ only
            same = open(out_rec, encoding="utf-8").read().split() == open(out_new, encoding="utf-8").read().split()
        print(f"   exit {r.returncode}; {time.time() - t0:.0f} s; output {'IDENTICAL to' if same else 'DIFFERS from'} "
              f"the recorded {rec}", flush=True)
        if r.returncode != 0:
            print(r.stderr[-1500:])
        bad += (r.returncode != 0) or not same
    print("\nall analyses reproduce their recorded outputs" if not bad else f"\n{bad} analysis/analyses did not reproduce")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
