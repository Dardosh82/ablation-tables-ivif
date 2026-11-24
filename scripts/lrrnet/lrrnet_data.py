"""The per-run metrics every LRRNet analysis reads, loaded from the kernels' own CSVs.

Until 2026-09-13 the analysis scripts carried these numbers as floating-point literals copied from the
kernel outputs. They matched -- verified to 0.0 difference against every CSV before this module
replaced them -- but a literal cannot show where it came from, and two external reviews asked, fairly,
why the analysis was decoupled from the record. Each value now comes from the file the Kaggle kernel
wrote, downloaded unchanged into ../kernel_csv/<kernel-slug>/lrrnet_metrics.csv.

    from lrrnet_data import arm
    arm("lrrnet-stage1", "nb4_")        -> {42: [EN, MI, SD, SF, AG, VIF, SCD, Qabf, SSIM], 43: [...], ...}

A missing file, a missing tag, a non-25-image row or a duplicated seed stops the analysis rather than
yielding a shorter arm: a short arm reads as a smaller sample, not as an error.
"""
import csv
import os
import re

METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
KERNEL_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kernel_csv")
TAG = re.compile(r"^(?P<prefix>.*?)seed(?P<seed>\d+)$")


def arm(kernel, prefix, seeds=None, n_images=25):
    """Rows of <kernel>'s lrrnet_metrics.csv whose tag starts with <prefix>, keyed by seed."""
    path = os.path.join(KERNEL_CSV, kernel, "lrrnet_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} -- download it from kaggle.com/code/mostafanada82/{kernel}")
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["tag"].startswith(prefix):
                continue
            m = TAG.match(row["tag"])
            if not m:
                raise ValueError(f"{path}: tag {row['tag']!r} does not end in seed<N>")
            seed = int(m.group("seed"))
            if seed in out:
                raise ValueError(f"{path}: seed {seed} appears twice under prefix {prefix!r}")
            if int(row["n"]) != n_images:
                raise ValueError(f"{path}: {row['tag']} scored on {row['n']} images, expected {n_images}")
            out[seed] = [float(row[k]) for k in METRICS]
    if seeds is not None and sorted(out) != sorted(seeds):
        raise ValueError(f"{path}: prefix {prefix!r} has seeds {sorted(out)}, expected {sorted(seeds)}")
    if not out:
        raise ValueError(f"{path}: no rows with prefix {prefix!r}")
    return dict(sorted(out.items()))
