"""Build and push the Run 10 kernels: seeds 50-61 for nb = 2, 4, 6. Readings: ../RUN10_PREREGISTRATION.md
and analyse_run10.py, both committed (0c42239) before this builder existed.

    python build_run10_kernel.py --list
    python build_run10_kernel.py --kernel nb2 --push

Same templates, same image, same T4 as Run 9 (asserted below), because the new seeds pool with the old ones.
Only SEEDS and the banner differ. Settings that would exceed Kaggle's 12-hour session cap at 12 runs are split
into two kernels of six seeds; nothing else about them differs.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LRR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from build_run9_kernel import load_template, push   # noqa: E402

IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
# kernel -> (setting, template, seeds, measured h/run from the Run 9 logs)
PLAN = {
    "nb2":  (2, "lrrnet-stage4", list(range(50, 62)), 0.79),
    "nb4a": (4, "lrrnet-stage1", list(range(50, 56)), 1.09),
    "nb4b": (4, "lrrnet-stage1", list(range(56, 62)), 1.09),
    "nb6a": (6, "lrrnet-stage2", list(range(50, 56)), 1.46),
    "nb6b": (6, "lrrnet-stage2", list(range(56, 62)), 1.46),
}
CAP_H = 11.0          # leave an hour of the 12-hour session for setup and scoring


def retarget(nb, setting, seeds, kernel):
    cells = nb["cells"]
    train = "".join(cells[3]["source"])
    new = re.sub(r"^SEEDS\s*=\s*\[[^\]]*\]", f"SEEDS      = {seeds!r}", train, count=1, flags=re.M)
    new = re.sub(r"^NUM_BLOCK\s*=\s*\d+", f"NUM_BLOCK  = {setting}", new, count=1, flags=re.M)
    if f"SEEDS      = {seeds!r}" not in new or f"NUM_BLOCK  = {setting}" not in new:
        sys.exit("FAIL: SEEDS / NUM_BLOCK did not take -- the template changed shape")
    banner = (
        "# ============================================================================\n"
        f"# RUN 10 ({kernel}) -- num_block = {setting}, seeds {seeds}.\n"
        "#\n"
        "# ONLY the seed list differs from the stage that produced this setting's existing rows,\n"
        "# so the new seeds pool with seeds 42-49. Readings fixed before this kernel existed in\n"
        "# studied_papers/LRRNet/RUN10_PREREGISTRATION.md and scripts/analyse_run10.py.\n"
        "# ============================================================================\n")
    cells[3]["source"] = (banner + new).splitlines(keepends=True)
    tail = (f"\n\nimport csv as _csv\n"
            f"_tags = [r['tag'] for r in _csv.DictReader(open(CSV))]\n"
            f"_missing = [t for t in [f'nb{setting}_seed{{s}}' for s in {seeds!r}] if t not in _tags]\n"
            f"assert not _missing, f'INCOMPLETE -- missing {{_missing}}. Kill criterion 1 applies.'\n"
            f"print('all {len(seeds)} Run 10 seeds present for nb={setting} ({kernel})')\n")
    cells[4]["source"] = list(cells[4]["source"]) + tail.splitlines(keepends=True)
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=sorted(PLAN))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    by_setting = {}
    for k, (setting, _, seeds, hpr) in PLAN.items():
        by_setting.setdefault(setting, []).extend(seeds)
        assert len(seeds) * hpr <= CAP_H, f"{k}: {len(seeds) * hpr:.1f} h exceeds the {CAP_H} h budget"
    for setting, seeds in by_setting.items():
        assert sorted(seeds) == list(range(50, 62)), f"nb={setting}: seeds {sorted(seeds)} are not exactly 50-61"

    if a.list or not a.kernel:
        for k, (setting, tmpl, seeds, hpr) in PLAN.items():
            print(f"{k:5} nb={setting} {tmpl:14} seeds {seeds[0]}-{seeds[-1]} ({len(seeds)})  {len(seeds) * hpr:4.1f} GPU-h")
        print(f"total {sum(len(s) * h for _, _, s, h in PLAN.values()):.1f} GPU-h; every setting gets exactly 50-61")
        if not a.kernel:
            return

    setting, tmpl, seeds, hpr = PLAN[a.kernel]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    nb, meta = load_template(api, tmpl)
    nb = retarget(nb, setting, seeds, a.kernel)

    slug = f"mostafanada82/lrrnet-run10-{a.kernel}"
    out = os.path.join(LRR, f"kernel_run10_{a.kernel}")
    os.makedirs(out, exist_ok=True)
    code_file = slug.split("/")[1] + ".ipynb"
    json.dump(nb, open(os.path.join(out, code_file), "w", encoding="utf-8"), indent=1)
    meta.update({"id": slug, "title": f"LRRNet_Run10_{a.kernel}", "code_file": code_file, "enable_gpu": True,
                 "is_private": True, "machine_shape": "NvidiaTeslaT4", "docker_image": IMAGE})
    meta.pop("id_no", None)
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
    print(f"built {out}: nb={setting}, seeds {seeds}, template {tmpl}, {len(seeds) * hpr:.1f} GPU-h expected, "
          f"data {meta.get('dataset_sources')}")
    if a.push:
        push(api, out, slug)
    else:
        print("NOT PUSHED. Re-run with --push.")


if __name__ == "__main__":
    sys.exit(main())
