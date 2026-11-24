"""The positive control on CrossFuse — arm A only, trained against untrained.

    THE READING OF BOTH OUTCOMES, FIXED BEFORE THE RUN
    ==================================================
    The paper currently reports that SF and AG cannot separate a randomly weighted LRRNet from a
    trained one, and states in section 9 that this is demonstrated for one metric-architecture pair.
    This run tests the second architecture. Both outcomes are written here, now, and whichever
    arrives is what gets reported.

    OUTCOME 1 -- SF and AG fail here too.
        The claim generalises across two architecture families and the section 9 limitation is
        REPLACED by a result. Wording becomes "on both architectures tested". This is the outcome
        that helps us, which is exactly why the alternative is written below with equal weight.

    OUTCOME 2 -- SF and AG separate the arms on CrossFuse.
        The failure is a property of the metric-architecture PAIR, not of the metric. This is a
        strictly more interesting finding than outcome 1 and it must be reported as the headline
        rather than buried: it would mean a metric's ability to respond cannot be assumed from
        another paper's experiment, so EVERY table needs its own control. The abstract, section 7
        and section 8 all narrow to LRRNet, and section 9's limitation becomes a measured fact.
        The paper does not get weaker; the recommendation gets stronger.

    OUTCOME 3 -- a metric that passed on LRRNet fails here.
        Reported plainly and added to the failing set for that architecture. No claim is made about
        why without a further experiment.

    KILL CRITERION.
        If the trained arm's own five seeds do not give a usable sigma -- if any metric's
        coefficient of variation exceeds 25 %, indicating the training itself did not converge --
        the control is void and nothing from this run is reported. A control built on an unstable
        trained arm cannot establish anything.

    NOT NEGOTIABLE AFTERWARDS.
        Welch's t, two-sided, 5 %. n = 5 per arm, so df is Welch-Satterthwaite around 4-8 and the
        critical value is 2.776 at df = 4. Nine metrics are tested at once, so roughly 0.45 false
        positives are expected by chance: a single significant cell is not a result, in either
        direction. The same multiplicity rule that discarded two cells favouring our own model in
        section 6 applies here unchanged.

    python analyse_poscontrol_cf.py --dir results_poscontrol_cf
"""
import argparse
import csv
import glob
import math
import os
import re
import sys

METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
T_CRIT = 2.776          # two-sided 5 %, df = 4
CV_KILL = 0.25          # kill criterion on the trained arm


# Tags this script accepts, and nothing else. An unrecognised tag is a hard error rather than a
# default, because the first version of this function defaulted unknown tags to "trained" -- and
# silently classified all five untrained runs as trained, producing a clean-looking control that
# compared the trained arm against itself. A wrong arm assignment does not crash and does not look
# wrong; it just answers the question falsely. Never default this.
UNTRAINED = re.compile(r"^(U\d+|untrained.*)$", re.I)
TRAINED = re.compile(r"^(A1[bcd]|PC\d+|R\d+[a-z]?)$", re.I)


def tag_of(csv_path):
    """Directory name -> run tag.

    The evaluator writes  <run>/tables_<TAG>_raw/<DATASET>_gray_per_image.csv,  so the directory
    holding a CSV is 'tables_U42_raw', not 'U42'. Matching the arm patterns against the raw
    directory name fails on every single run -- which at least fails loudly. Normalise first.
    """
    d = os.path.basename(os.path.dirname(csv_path))
    if d.startswith("tables_"):
        d = d[len("tables_"):]
    if d.endswith("_raw"):
        d = d[:-len("_raw")]
    return d


def classify(tag):
    if UNTRAINED.match(tag):
        return "untrained"
    if TRAINED.match(tag):
        return "trained"
    sys.exit(f"FAIL: cannot tell which arm '{tag}' belongs to. Add it to TRAINED or UNTRAINED in "
             f"this script -- do not guess, and do not let it default.")


def read_per_image(path):
    """One evaluator CSV -> {image_id: {metric: value}}.

    Deliberately NOT collapsed to a mean here. analyse_tier3.py intersects the image keys across
    runs and asserts every run scored the same images before averaging anything; collapsing each
    file to its own mean would silently compare runs over different image populations if one
    inference wrote a different set. The first version of this function did exactly that.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:      # BOM: written on Windows
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"FAIL: {path} is empty")
    out = {}
    for r in rows:
        try:
            vals = {m: float(r[m]) for m in METRICS}
        except (KeyError, TypeError, ValueError) as e:
            sys.exit(f"FAIL: {path} row {r.get('image', '?')} is unreadable: {e}")
        # NaN / inf must stop the analysis, and this is the single most important guard here.
        # An untrained network is exactly the model most likely to emit non-finite output, and
        # Python parses 'nan' happily: the mean becomes nan, Welch's t becomes nan, and
        # `abs(nan) >= 2.776` evaluates to FALSE -- which this script would print as
        # "CANNOT DETECT". A broken pipeline would therefore report the strongest possible
        # version of this paper's claim. Non-finite values are a pipeline failure, never evidence.
        bad = [m for m, v in vals.items() if not math.isfinite(v)]
        if bad:
            sys.exit(f"FAIL: {path} row '{r.get('image','?')}' has non-finite {', '.join(bad)}. "
                     f"That is a broken evaluation, not a metric failing to detect. Fix the "
                     f"pipeline; do not report this run.")
        out[r["image"]] = vals
    return out


def epochs_completed(run_dir, tag):
    """Rows in loss_<tag>.csv == epochs actually finished. None if the file is not present."""
    for cand in (os.path.join(run_dir, f"loss_{tag}.csv"),
                 os.path.join(run_dir, "CrossFuse", "models", f"loss_{tag}.csv")):
        if os.path.exists(cand):
            with open(cand, newline="", encoding="utf-8-sig") as f:
                return len(list(csv.DictReader(f)))
    return None


def mean_sd(xs):
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, float("nan")
    return mu, math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def welch(a, b):
    ma, sa = mean_sd(a)
    mb, sb = mean_sd(b)
    na, nb = len(a), len(b)
    va, vb = sa ** 2 / na, sb ** 2 / nb
    if va + vb == 0:
        return ma, sa, mb, sb, float("inf")
    return ma, sa, mb, sb, (ma - mb) / math.sqrt(va + vb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding tables_<TAG>_raw/*.csv")
    ap.add_argument("--dataset", default="TNO", help="test set to report; TNO matches LRRNet's arm")
    ap.add_argument("--epochs", type=int, default=299,
                    help="epochs a trained-arm run must have completed (A1b family = 299)")
    a = ap.parse_args()

    # Exactly one file per run, named exactly. The evaluator writes BOTH <ds>_gray_per_image.csv
    # and <ds>_y_per_image.csv, so a substring match on the dataset name collects two rows per run,
    # silently doubling every n and shrinking every standard error. That inflates significance in
    # the direction that flatters the result, which is the worst direction for a bug to point.
    # The rest of this project scores on the _gray_ variant (analyse_tier3.py, verify_run.py).
    wanted = f"{a.dataset}_gray_per_image.csv"
    arms = {"trained": [], "untrained": []}
    seen = set()
    for path in sorted(glob.glob(os.path.join(a.dir, "**", wanted), recursive=True)):
        tag = tag_of(path)
        if tag in seen:
            sys.exit(f"FAIL: two files claim to be run '{tag}'. Refusing to count it twice.")
        seen.add(tag)
        arms[classify(tag)].append((tag, read_per_image(path)))

    if not seen:
        sys.exit(f"FAIL: no '{wanted}' found under {a.dir}. Nothing to analyse.")
    for k, v in arms.items():
        print(f"{k:>10}: {len(v)} run(s) -- {', '.join(t for t, _ in v)}")

    # --- every run must have scored the SAME images -------------------------------------------
    # analyse_tier3.py enforces this and it matters more here: the untrained arm produces garbage
    # images, so it is the arm most likely to lose one. Comparing means over different image sets
    # would not crash and would not look wrong.
    allruns = arms["trained"] + arms["untrained"]
    keysets = {t: set(d) for t, d in allruns}
    common = set.intersection(*keysets.values())
    for t, ks in keysets.items():
        if ks != common:
            sys.exit(f"FAIL: run '{t}' scored {len(ks)} images, common set is {len(common)}. "
                     f"Missing: {sorted(common ^ ks)[:5]}. Refusing to average over different "
                     f"image populations.")
    print(f"image sets identical across all {len(allruns)} runs: {len(common)} images")

    # --- a truncated training must not enter the trained arm -----------------------------------
    # train.py stops mid-run when --max_hours is hit and still writes a checkpoint. Nothing
    # downstream can tell a 200-epoch model from a 299-epoch one; verify_run.py checks this for
    # every other run in the project, so it is checked here too.
    for t, _ in arms["trained"]:
        n_ep = epochs_completed(a.dir, t)
        if n_ep is None:
            print(f"  ! {t}: no loss_{t}.csv -- epoch count UNVERIFIED")
        elif n_ep != a.epochs:
            sys.exit(f"FAIL: '{t}' completed {n_ep} epochs, expected {a.epochs}. A training cut "
                     f"short by --max_hours is not a member of the trained arm.")
        else:
            print(f"  {t}: {n_ep} epochs confirmed")

    # collapse to per-run metric means over the COMMON images only
    keys = sorted(common)
    arms = {arm: [(t, {m: sum(d[k][m] for k in keys) / len(keys) for m in METRICS})
                  for t, d in runs] for arm, runs in arms.items()}
    if len(arms["trained"]) < 3 or len(arms["untrained"]) < 3:
        sys.exit("FAIL: need at least 3 runs per arm; refusing to report a control on fewer")
    if len(arms["trained"]) < 5 or len(arms["untrained"]) < 5:
        print("\n!! WARNING: fewer than 5 seeds per arm. Section 7 of the paper narrowed a headline\n"
              "!! precisely because three seeds were not enough. Report n=3 results as provisional\n"
              "!! and say so, or run the remaining seeds.\n")

    # --- no arm may be a set of copies ----------------------------------------------------------
    # If five "seeds" produced one identical model, that arm's variance is zero, every Welch
    # denominator collapses and every metric appears to detect. The failure looks CLEANER than a
    # real result, which is why it needs an explicit guard rather than a glance at the output.
    for arm, runs in arms.items():
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                if all(abs(runs[i][1][m] - runs[j][1][m]) < 1e-12 for m in METRICS):
                    sys.exit(f"FAIL: '{runs[i][0]}' and '{runs[j][0]}' are numerically identical "
                             f"on all nine metrics. They are copies, not seeds. Arm '{arm}' has no "
                             f"variance and the control cannot be computed.")
        for m in METRICS:
            _, sd = mean_sd([t[m] for _, t in runs])
            if sd == 0:
                sys.exit(f"FAIL: arm '{arm}' has zero spread on {m}. Every t against it would be "
                         f"infinite. Refusing to report.")
    print("both arms vary across seeds: no duplicate runs, no zero-variance metric")

    # --- kill criterion -------------------------------------------------------------------------
    print(f"\n{'metric':>6}  {'trained CV':>11}")
    voided = []
    for m in METRICS:
        mu, sd = mean_sd([t[m] for _, t in arms["trained"]])
        cv = abs(sd / mu) if mu else float("inf")
        flag = "  <-- exceeds kill threshold" if cv > CV_KILL else ""
        if cv > CV_KILL:
            voided.append(m)
        print(f"{m:>6}  {cv:>10.1%}{flag}")
    if voided:
        sys.exit(f"\nVOID: trained arm unstable on {', '.join(voided)} (CV > {CV_KILL:.0%}). "
                 f"The pre-registered kill criterion applies: report nothing from this run.")
    print(f"\nkill criterion passed: every trained-arm CV is below {CV_KILL:.0%}")

    # --- the control ----------------------------------------------------------------------------
    print(f"\n{'metric':>6}  {'trained':>18}  {'untrained':>18}  {'t':>8}  verdict")
    detected = []
    for m in METRICS:
        ta = [t[m] for _, t in arms["trained"]]
        ua = [t[m] for _, t in arms["untrained"]]
        mt, st, mu_, su, t = welch(ta, ua)
        ok = abs(t) >= T_CRIT
        detected.append(m) if ok else None
        print(f"{m:>6}  {mt:>10.4f} +/-{st:<6.4f}  {mu_:>10.4f} +/-{su:<6.4f}  "
              f"{t:>8.2f}  {'detects' if ok else 'CANNOT DETECT'}")

    k = len(detected)
    print(f"\n{k} of 9 metrics separate an untrained CrossFuse from a trained one "
          f"(|t| >= {T_CRIT}).")
    print(f"detected      : {', '.join(detected) if detected else '(none)'}")
    failed = [m for m in METRICS if m not in detected]
    print(f"cannot detect : {', '.join(failed) if failed else '(none)'}")

    lrrnet_failed = {"SF", "AG"}
    both = lrrnet_failed & set(failed)
    print("\n--- against LRRNet, where SF and AG failed ---")
    if both == lrrnet_failed:
        print("OUTCOME 1: SF and AG fail on BOTH architectures. The claim generalises across two\n"
              "families. Replace the section 9 limitation with this result and reword to\n"
              "'on both architectures tested'.")
    elif not both:
        print("OUTCOME 2: SF and AG SEPARATE the arms here. The failure is a property of the\n"
              "metric-architecture pair, not of the metric. Narrow the abstract, section 7 and\n"
              "section 8 to LRRNet -- and promote the stronger recommendation: a metric's ability\n"
              "to respond does not transfer between architectures, so every table needs its own\n"
              "control.")
    else:
        print(f"OUTCOME 2 (partial): {', '.join(sorted(both))} fails on both; "
              f"{', '.join(sorted(lrrnet_failed - both))} does not. Report per metric; claim\n"
              "nothing general about which transfers without a third architecture.")
    new = set(failed) - lrrnet_failed
    if new:
        print(f"OUTCOME 3: {', '.join(sorted(new))} passed on LRRNet and fails here. Report plainly;\n"
              "offer no mechanism without a further experiment.")

    print(f"\nmultiplicity: 9 tests at 5 % expect ~0.45 false positives. "
          f"{'A single cell either way is not a result.' if k in (0, 1, 8, 9) else ''}")


if __name__ == "__main__":
    main()

