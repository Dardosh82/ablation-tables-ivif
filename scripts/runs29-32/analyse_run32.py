"""Run 32 -- does a downstream detector agree with the metric verdicts?  Readings: ../RUN32_PREREGISTRATION.md
(19940f4), committed alone before any kernel existed.

    python analyse_run32.py            # reads ../results_run32/run32/run32_ap50.csv and preds/*.npz

Registered: Welch's t at exact df on per-network AP50, five networks per arm, Holm across D1-D3.
  D1 CrossFuse trained vs untrained skip on:  AGREES WITH THE CONTROL / AGREES WITH MI/VIF / INCONCLUSIVE
  D2 LRRNet published d4 vs untrained d4:     TRAINED HIGHER / UNTRAINED HIGHER / INCONCLUSIVE
  D3 LRRNet tied d8 vs untied d8:             TRANSFERS / DOES NOT TRANSFER / REVERSES
Descriptive, no verdict: the skip-off arm, the two single-band references, and whether each pair's AP50 ordering
matches its MI and VIF ordering on TNO (taken from the recorded tables, not typed in).
Integrity: AP50 is recomputed here from the kept raw detections and must match the kernel's to 1e-6.
"""
import csv
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
LRR = os.path.dirname(HERE)
SP = os.path.dirname(LRR)
R32 = os.path.join(LRR, "results_run32", "run32")
T3 = os.path.join(SP, "CrossFuse", "Tier3_CrossFuse")
ARMS = {"CF_T": ("CrossFuse", "trained"), "CF_ON": ("CrossFuse", "untrained skip on"),
        "CF_OFF": ("CrossFuse", "untrained skip off"), "L_PUB": ("LRRNet", "published d4"),
        "L_UNT": ("LRRNet", "untrained d4"), "L_TIED": ("LRRNet", "tied d8"), "L_UNTIED": ("LRRNet", "untied d8")}
CONTRASTS = [("D1", "CF_T", "CF_ON", ("AGREES WITH THE CONTROL", "AGREES WITH MI/VIF")),
             ("D2", "L_PUB", "L_UNT", ("TRAINED HIGHER", "UNTRAINED HIGHER")),
             ("D3", "L_TIED", "L_UNTIED", ("TRANSFERS", "REVERSES"))]


def welch(a, b):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, df, 2 * stats.t.sf(abs(t), df)


def holm(ps):
    order, adj, running = np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def _ap_from_tp(tp, ngt):
    ctp = np.cumsum(tp)
    rec, prec = ctp / ngt, ctp / np.arange(1, len(tp) + 1)
    mrec, mpre = np.concatenate([[0.0], rec, [1.0]]), np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    k = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[k + 1] - mrec[k]) * mpre[k + 1]))


def ap50(img, boxes, scores, gt, ngt):
    """AP50 as the kernel computed it, vectorised (2026-11-26: the detection-by-detection loop below took minutes per
    run of this script, over ten on an external reviewer's machine). Greedy VOC matching assigns each detection to the
    box it overlaps most and never to a second-best one, so the true positive for each (image, box) is simply the
    earliest detection in score order that points at it with IoU >= 0.5. Identical to ap50_loop on all 37 networks
    (run with --check-loop)."""
    order = np.argsort(-scores, kind="stable")
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    best_iou, best_j = np.zeros(len(scores)), np.full(len(scores), -1, np.int64)
    for i in np.unique(img):
        g = gt[i]
        if len(g) == 0:
            continue
        sel = np.nonzero(img == i)[0]
        b = boxes[sel]
        x1, y1 = np.maximum(b[:, None, 0], g[None, :, 0]), np.maximum(b[:, None, 1], g[None, :, 1])
        x2, y2 = np.minimum(b[:, None, 2], g[None, :, 2]), np.minimum(b[:, None, 3], g[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        o = inter / ((b[:, None, 2] - b[:, None, 0]) * (b[:, None, 3] - b[:, None, 1])
                     + (g[None, :, 2] - g[None, :, 0]) * (g[None, :, 3] - g[None, :, 1]) - inter)
        j = np.argmax(o, axis=1)
        best_j[sel], best_iou[sel] = j, o[np.arange(len(sel)), j]
    cand = np.nonzero(best_iou >= 0.5)[0]
    c = cand[np.lexsort((rank[cand], best_j[cand], img[cand]))]
    first = np.ones(len(c), bool)
    first[1:] = (img[c][1:] != img[c][:-1]) | (best_j[c][1:] != best_j[c][:-1])
    tp_det = np.zeros(len(scores))
    tp_det[c[first]] = 1
    return _ap_from_tp(tp_det[order], ngt)


def ap50_loop(img, boxes, scores, gt, ngt):
    """The kernel's own loop, kept as the reference the vectorised version is checked against."""
    order = np.argsort(-scores, kind="stable")
    used = [np.zeros(len(g), bool) for g in gt]
    tp = np.zeros(len(order))
    for k, i in enumerate(order):
        g = gt[img[i]]
        if len(g) == 0:
            continue
        b = boxes[i]
        x1, y1 = np.maximum(b[0], g[:, 0]), np.maximum(b[1], g[:, 1])
        x2, y2 = np.minimum(b[2], g[:, 2]), np.minimum(b[3], g[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        o = inter / ((b[2] - b[0]) * (b[3] - b[1]) + (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) - inter)
        j = int(np.argmax(o))
        if o[j] >= 0.5 and not used[img[i]][j]:
            used[img[i]][j] = True
            tp[k] = 1
    return _ap_from_tp(tp, ngt)


def tno_means(path, metric):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    return float(np.mean([float(r[metric]) for r in rows]))


def recorded_mi_vif():
    """Arm means of MI and VIF on TNO, from the recorded tables the paper reports."""
    pc = os.path.join(T3, "results_poscontrol_cf")
    cf = {"CF_T": [os.path.join(pc, f"tables_{t}_raw", "TNO_gray_per_image.csv") for t in
                   ("A1b", "A1c", "A1d", "PC45", "PC46")],
          "CF_ON": [os.path.join(pc, f"tables_U{s}_raw", "TNO_gray_per_image.csv") for s in range(42, 47)],
          "CF_OFF": [os.path.join(T3, "results_run30", "tables", f"U{s}_OFF", "TNO_gray_per_image.csv")
                     for s in range(42, 47)]}
    out = {k: {m: np.mean([tno_means(p, m) for p in v]) for m in ("MI", "VIF")} for k, v in cf.items()}
    r29 = os.path.join(LRR, "results_run29")
    means = {}
    for f in ("run29_src16_TNO_means.csv", "run29_src17b_TNO_means.csv", "run29_src25_TNO_means.csv"):
        for r in csv.DictReader(open(os.path.join(r29, f), encoding="utf-8")):
            means[r["tag"]] = r
    tags = {"L_PUB": [f"nb4_seed{s}" for s in range(42, 47)],
            "L_UNT": [f"nb4_untrained_seed{s}" for s in range(400, 405)],
            "L_TIED": [f"m1tied_nb8_seed{s}" for s in range(62, 67)],
            "L_UNTIED": [f"nb8_seed{s}" for s in range(62, 67)]}
    for k, v in tags.items():
        out[k] = {m: np.mean([float(means[t][m]) for t in v]) for m in ("MI", "VIF")}
    return out


def main():
    rows = list(csv.DictReader(open(os.path.join(R32, "run32_ap50.csv"), encoding="utf-8")))
    lines = ["Run 32 -- YOLOv8s (COCO, never fine-tuned) person AP50 on 1,000 LLVIP test pairs at 640x512", ""]

    # integrity: recompute every AP50 from the kept detections
    gtrows = list(csv.DictReader(open(os.path.join(R32, "ground_truth.csv"), encoding="utf-8")))
    stems = sorted(os.path.splitext(n.strip())[0] for n in open(os.path.join(R32, "subset.txt")) if n.strip())
    assert len(stems) == 1000
    idx = {s: i for i, s in enumerate(stems)}
    gt = [[] for _ in stems]
    for r in gtrows:
        gt[idx[r["image"]]].append([float(r[k]) for k in ("x1", "y1", "x2", "y2")])
    gt = [np.array(g, dtype=np.float64).reshape(-1, 4) for g in gt]
    ngt = sum(len(g) for g in gt)
    worst, loop_diff = 0.0, 0.0
    check_loop = "--check-loop" in sys.argv
    for n, r in enumerate(rows, 1):
        z = np.load(os.path.join(R32, "preds", r["tag"] + ".npz"))
        args = (z["img"], z["boxes"].astype(np.float64), z["score"].astype(np.float64), gt, ngt)
        a_fast = ap50(*args)
        worst = max(worst, abs(a_fast - float(r["ap50"])))
        if check_loop:
            loop_diff = max(loop_diff, abs(ap50_loop(*args) - a_fast))
            print(f"  [{n}/{len(rows)}] {r['tag']}: vectorised vs loop {abs(ap50_loop(*args) - a_fast):.1e}",
                  file=sys.stderr, flush=True)
    if check_loop:
        print(f"--check-loop: vectorised AP50 equals the kernel's loop on all {len(rows)} networks to {loop_diff:.1e}",
              file=sys.stderr)
    lines.append(f"integrity: AP50 recomputed from the raw detections, largest difference {worst:.1e} "
                 + ("(passes 1e-6)" if worst < 1e-6 else "-- FAILS, nothing is read"))
    lines.append(f"ground truth: {ngt} person boxes in 1000 images")
    if worst >= 1e-6:
        print("\n".join(lines))
        return 1

    ap = {k: np.array([float(r["ap50"]) for r in rows if (r["family"], r["arm"]) == fa and r["blank"] == "0"])
          for k, fa in ARMS.items()}
    blank = [r["tag"] for r in rows if r["blank"] == "1"]
    lines.append("blank networks excluded: " + (", ".join(blank) if blank else "none"))
    assert all(len(v) == 5 for v in ap.values()), {k: len(v) for k, v in ap.items()}
    lines += ["", f"  {'arm':28s} {'mean AP50':>10} {'sd':>8}   per network"]
    for k, (fam, arm) in ARMS.items():
        lines.append(f"  {fam + ' ' + arm:28s} {ap[k].mean():>10.4f} {ap[k].std(ddof=1):>8.4f}   "
                     + " ".join(f"{v:.4f}" for v in ap[k]))
    for r in rows:
        if r["family"] == "reference":
            lines.append(f"  {'reference ' + r['arm']:28s} {float(r['ap50']):>10.4f}")

    res = [welch(ap[a], ap[b]) for _, a, b, _ in CONTRASTS]
    adj = holm(np.array([p for _, _, p in res]))
    lines += ["", "Registered contrasts -- Welch exact df, Holm across three",
              f"  {'':3} {'A':>8} {'B':>8} {'A-B':>8} {'t':>7} {'df':>6} {'p':>9} {'Holm p':>9}  verdict"]
    for (name, a, b, (hi, lo)), (t, df, p), h in zip(CONTRASTS, res, adj):
        d = ap[a].mean() - ap[b].mean()
        verdict = (hi if d > 0 else lo) if h < 0.05 else ("DOES NOT TRANSFER" if name == "D3" else "INCONCLUSIVE")
        lines.append(f"  {name:3} {ap[a].mean():>8.4f} {ap[b].mean():>8.4f} {d:>+8.4f} {t:>7.2f} {df:>6.2f} "
                     f"{p:>9.2e} {h:>9.2e}  {verdict}")

    mv = recorded_mi_vif()
    lines += ["", "Descriptive: does the AP50 ordering match the MI and VIF ordering on TNO (recorded tables)?"]
    for a, b in (("CF_T", "CF_ON"), ("CF_T", "CF_OFF"), ("CF_ON", "CF_OFF"), ("L_PUB", "L_UNT"),
                 ("L_TIED", "L_UNTIED")):
        o = lambda x, y: ">" if x > y else "<"
        lines.append(f"  {a:8s} vs {b:8s}  AP50 {o(ap[a].mean(), ap[b].mean())}   "
                     f"MI {o(mv[a]['MI'], mv[b]['MI'])} ({mv[a]['MI']:.3f} vs {mv[b]['MI']:.3f})   "
                     f"VIF {o(mv[a]['VIF'], mv[b]['VIF'])} ({mv[a]['VIF']:.3f} vs {mv[b]['VIF']:.3f})")
    text = "\n".join(lines)
    print(text)
    open(os.path.join(os.path.dirname(R32), "RUN32_ANALYSIS.txt"), "w", encoding="utf-8").write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
