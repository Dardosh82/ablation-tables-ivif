"""Run 29 -- do the headline results hold over test images as well as over runs?  Readings: ../RUN29_PREREGISTRATION.md
(5b1e990, committed before any per-image row existed).

    python analyse_run29.py            # every family whose data is present; LRRNet families wait for results_run29/

Hierarchical bootstrap, fixed in the registration: B = 10,000, seed 20261125; each replicate resamples runs with
replacement within each arm and test images with replacement, the SAME image draw for both arms (every run scores
the same images, so the comparison stays paired over images); statistic = difference of arm means (mean over images,
then over runs); 95 % percentile interval. A cell reported significant after Holm reads HOLDS OVER IMAGES if the
interval excludes zero in the reported direction, else RUNS ONLY. Nulls are read only as exploratory.

LRRNet families are read only after the integrity stop rule passes: every re-scored checkpoint's new mean row must
reproduce the run mean already on record to six decimals on every metric.
"""
import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LRR = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(LRR))
T3 = os.path.join(ROOT, "studied_papers", "CrossFuse", "Tier3_CrossFuse")
R29 = os.path.join(LRR, "results_run29")
M = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
B, SEED = 10_000, 20261125


# ---------------------------------------------------------------------------------------------------- bootstrap
def hier_boot(a, b, rng):
    """a, b: (runs, images, metrics). Returns (B, metrics) differences mean(a) - mean(b)."""
    ra, rb, ni = a.shape[0], b.shape[0], a.shape[1]
    out = np.empty((B, a.shape[2]))
    for k in range(B):
        img = rng.integers(0, ni, ni)
        ia, ib = rng.integers(0, ra, ra), rng.integers(0, rb, rb)
        out[k] = a[ia][:, img].mean(axis=(0, 1)) - b[ib][:, img].mean(axis=(0, 1))
    return out


def single_boot(a, b, rng, level):
    """Only runs (level='runs') or only images (level='images') resampled: for the variance-share description."""
    ra, rb, ni = a.shape[0], b.shape[0], a.shape[1]
    out = np.empty((B, a.shape[2]))
    for k in range(B):
        if level == "runs":
            ia, ib, img = rng.integers(0, ra, ra), rng.integers(0, rb, rb), np.arange(ni)
        else:
            ia, ib, img = np.arange(ra), np.arange(rb), rng.integers(0, ni, ni)
        out[k] = a[ia][:, img].mean(axis=(0, 1)) - b[ib][:, img].mean(axis=(0, 1))
    return out


def pooled_sigma(a, b):
    ma, mb = a.mean(axis=1), b.mean(axis=1)            # run means
    va, vb = ma.var(axis=0, ddof=1), mb.var(axis=0, ddof=1)
    return np.sqrt(((len(ma) - 1) * va + (len(mb) - 1) * vb) / (len(ma) + len(mb) - 2))


def read_cell(name, ds, a, b, cells, rng, lines):
    """cells: {metric: 'sig'|'null'} for this comparison. Appends report lines; returns per-cell verdicts."""
    d = hier_boot(a, b, rng)
    sr, si = single_boot(a, b, rng, "runs"), single_boot(a, b, rng, "images")
    sig = pooled_sigma(a, b)
    obs = a.mean(axis=(0, 1)) - b.mean(axis=(0, 1))
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    verdicts = {}
    for m, kind in cells.items():
        j = M.index(m)
        excl = lo[j] > 0 or hi[j] < 0
        same_dir = np.sign(obs[j]) == (1 if lo[j] > 0 else -1)
        if kind == "sig":
            v = "HOLDS OVER IMAGES" if excl and same_dir else "RUNS ONLY"
        else:
            v = "exploratory: interval excludes zero" if excl else "null, unchanged"
        share = sr[:, j].var() / (sr[:, j].var() + si[:, j].var())
        verdicts[m] = v
        lines.append(f"  {name:28s} {ds:9s} {m:5s} diff {obs[j]:+.4g} ({obs[j] / sig[j]:+.2f} sigma)  "
                     f"95% [{lo[j]:+.4g}, {hi[j]:+.4g}]  runs' share of variance {share:.0%}  -> {v}")
    return verdicts


# ---------------------------------------------------------------------------------------------------- data
def crossfuse_f5():
    trained, untrained = re.compile(r"^(A1[b-d]|PC4[56])$"), re.compile(r"^U4[2-6]$")
    runs = {"trained": [], "untrained": []}
    for p in sorted(glob.glob(os.path.join(T3, "results_poscontrol_cf", "tables_*_raw", "TNO_gray_per_image.csv"))):
        tag = os.path.basename(os.path.dirname(p))[len("tables_"):-len("_raw")]
        arm = "trained" if trained.match(tag) else "untrained" if untrained.match(tag) else None
        assert arm, f"unclassified run {tag}"
        with open(p, newline="", encoding="utf-8-sig") as f:
            rows = {r["image"]: [float(r[m]) for m in M] for r in csv.DictReader(f)}
        runs[arm].append((tag, rows))
    assert len(runs["trained"]) == 5 and len(runs["untrained"]) == 5, {k: len(v) for k, v in runs.items()}
    images = sorted(runs["trained"][0][1])
    for arm in runs.values():
        for tag, rows in arm:
            assert sorted(rows) == images, f"{tag}: a different image set"
            assert all(np.isfinite(rows[i]).all() for i in images), f"{tag}: non-finite score"
    arr = {k: np.array([[rows[i] for i in images] for _, rows in v]) for k, v in runs.items()}
    return arr["untrained"], arr["trained"], len(images)


def lrrnet_perimage(src, ds):
    p = os.path.join(R29, f"run29_src{src}_{ds}_perimage.csv")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["tag"], {})[r["image"]] = [float(r[m]) for m in M]
    return out


def stack(per, tags):
    missing = [t for t in tags if t not in per]
    assert not missing, f"tags missing from the per-image file: {missing}"
    images = sorted(per[tags[0]])
    for t in tags:
        assert sorted(per[t]) == images, f"{t}: a different image set"
    return np.array([[per[t][i] for i in images] for t in tags])


def integrity():
    """Every re-scored mean must equal the recorded mean to six decimals. Returns a list of failures."""
    fails, checked = [], 0
    for p in sorted(glob.glob(os.path.join(R29, "run29_src*_*_means.csv"))):
        src, ds = re.search(r"run29_src(\w+?)_(TNO|RoadScene)_means", p).groups()
        new = {r["tag"]: r for r in csv.DictReader(open(p, newline="", encoding="utf-8-sig"))}
        recorded = {}
        for q in RECORDED[(src, ds)]:
            for r in csv.DictReader(open(os.path.join(LRR, q), newline="", encoding="utf-8-sig")):
                recorded[r["tag"]] = r
        for tag, r in new.items():
            if tag not in recorded:
                fails.append(f"src{src} {ds} {tag}: no recorded row to check against")
                continue
            for m in M:
                new_v, rec_v = float(r[m]), float(recorded[tag][m])
                # registration correction f6a4cfa: Run 20's record was scored from CPU-fused images, the re-score
                # from the GPU fuse kernel's, so Run 20 alone is checked at 1e-4 relative; every other source exactly
                ok = (abs(new_v - rec_v) <= 1e-4 * abs(rec_v)) if src == "20" else (abs(new_v - rec_v) <= 5e-7)
                if not ok:
                    fails.append(f"src{src} {ds} {tag} {m}: new {r[m]} vs recorded {recorded[tag][m]}")
            checked += 1
    return fails, checked


# where each source's run means are already on record (written by the earlier runs' score kernels)
RECORDED = {("16", "TNO"): ["results_run16/run16_TNO.csv"], ("16", "RoadScene"): ["results_run16/run16_RoadScene.csv"],
            ("17b", "TNO"): ["results_run17b/run17b_TNO.csv"],
            ("17b", "RoadScene"): ["results_run17b/run17b_RoadScene.csv"],
            ("20", "TNO"): ["results_run20/run20_TNO.csv"], ("20", "RoadScene"): ["results_run20/run20_RoadScene.csv"],
            ("25", "TNO"): ["results_run25/run25_TNO.csv"]}


def main():
    rng = np.random.default_rng(SEED)
    lines = ["Run 29 -- hierarchical bootstrap (runs x images), B = 10,000, seed 20261125", ""]

    # F5: CrossFuse positive control, all nine metrics significant after Holm (untrained minus trained)
    u, t, n = crossfuse_f5()
    lines.append(f"F5  CrossFuse positive control, TNO, untrained (5) minus trained (5), {n} images")
    read_cell("F5 untrained - trained", "TNO", u, t, {m: "sig" for m in M}, rng, lines)

    present = [p for p in glob.glob(os.path.join(R29, "run29_src*_perimage.csv"))]
    if not present:
        lines += ["", "LRRNet families F1-F4: waiting for results_run29/ (the four Run 29 scoring kernels)."]
    else:
        fails, checked = integrity()
        lines += ["", f"integrity: {checked} re-scored checkpoint means compared with the record, {len(fails)} differ"]
        if fails:
            lines += ["STOP -- the registration's stop rule: nothing below is read."] + ["  " + f for f in fails[:40]]
        else:
            s = {k: lrrnet_perimage(*k) for k in [("16", "TNO"), ("16", "RoadScene"), ("17b", "TNO"),
                                                     ("17b", "RoadScene"), ("20", "TNO"), ("20", "RoadScene"),
                                                     ("25", "TNO")]}
            seeds = lambda prefix, lo_, hi_: [f"{prefix}seed{x}" for x in range(lo_, hi_ + 1)]
            four = {"SF": "sig", "AG": "sig", "VIF": "sig", "SCD": "sig"}
            lines.append("\nF1  tied d8 minus untied d8, seeds 62-73")
            for ds in ("TNO", "RoadScene"):
                read_cell("F1 tied-untied d8", ds, stack(s[("17b", ds)], seeds("m1tied_nb8_", 62, 73)),
                          stack(s[("17b", ds)], seeds("nb8_", 62, 73)), four, rng, lines)
            lines.append("\nF2  tied d8 (seeds 74-85) minus published d4 (seeds 42-49)")
            f2 = {"TNO": {"VIF": "sig", "Qabf": "sig", "SSIM": "sig", "SD": "sig"},
                  "RoadScene": {"EN": "sig", "VIF": "sig", "Qabf": "sig", "SSIM": "sig"}}
            for ds in ("TNO", "RoadScene"):
                read_cell("F2 tied d8-published", ds, stack(s[("20", ds)], seeds("m1tied_nb8_", 74, 85)),
                          stack(s[("16", ds)], seeds("nb4_", 42, 49)), f2[ds], rng, lines)
            lines.append("\nF3  Table IV at twenty seeds (42-61), setting minus the published 4")
            for ds in ("TNO", "RoadScene"):
                p4 = stack(s[("16", ds)], seeds("nb4_", 42, 61))
                for nb in (2, 6):
                    cells = {m: "null" for m in M}
                    if ds == "RoadScene":
                        cells.update({"EN": "sig", "SCD": "sig"} if nb == 2 else {"VIF": "sig", "SCD": "sig"})
                    read_cell(f"F3 nb{nb}-nb4", ds, stack(s[("16", ds)], seeds(f"nb{nb}_", 42, 61)), p4, cells, rng,
                              lines)
            lines.append("\nF4  LRRNet positive control, trained (42-61) minus untrained (400-419), TNO")
            f4 = {m: "sig" for m in M}
            f4.update({"SF": "null", "AG": "null"})
            read_cell("F4 trained-untrained", "TNO", stack(s[("16", "TNO")], seeds("nb4_", 42, 61)),
                      stack(s[("25", "TNO")], seeds("nb4_untrained_", 400, 419)), f4, rng, lines)
    text = "\n".join(lines)
    print(text)
    os.makedirs(R29, exist_ok=True)
    open(os.path.join(R29, "RUN29_ANALYSIS.txt"), "w", encoding="utf-8").write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
