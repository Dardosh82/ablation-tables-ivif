"""The paper's figure set — 5 figures, publication quality.

Distinct from `thesis/tools/make_figures2.py`, which serves the long report. The paper's spine is
narrower: the positive control is the headline, the re-measured table is the evidence, and the
capacity-by-construction result is the constructive turn. Each figure below carries one of those and
nothing else.

Every number is transcribed from a results file and the source is named in a comment above it. No
figure computes a quantity that the text does not also state, and none plots data that was not
measured -- where only a summary count exists (control arms B and C), the figure shows the count.

Palette and chrome follow make_figures2.py: light surface only, first three categorical slots
(blue/orange/aqua, validated all-pairs), hairline grid on one axis, two spines.

    python our_papers/paper1_variance_aware/draft/make_paper_figures.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK2     = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
FAIL     = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.major.size": 0, "ytick.major.size": 0,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

METRICS = ["EN", "MI", "SD", "SF", "AG", "VIF", "SCD", "Qabf", "SSIM"]
T_CRIT = 2.776          # two-sided 5 %, df = 4


def frame(ax, xgrid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.grid(axis="x" if xgrid else "y", zorder=0)
    ax.set_axisbelow(True)


# Each figure carried its explanation as a footer baked into the PNG. Once the figures were placed
# in the paper, the LaTeX \caption said the same thing -- so the text appeared twice and the extra
# rows made every figure taller than it needed to be. The caption is the right home for it: it is
# selectable, searchable, and scales with the venue's font. Set FOOTERS = True to get standalone
# figures again (useful for slides, where there is no caption to carry the point).
FOOTERS = False


def footer(fig, text, y=-0.20):
    if FOOTERS:
        fig.text(0.008, y, text, fontsize=8, color=INK2, ha="left")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    # vector copy for the paper; no CreationDate, so an unchanged figure rebuilds byte-identically
    pdf = os.path.splitext(name)[0] + ".pdf"
    fig.savefig(os.path.join(OUT, pdf), format="pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  wrote {name} + {pdf}")


def pooled(s1, n1, s2, n2):
    return np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))


# ================================================================================================
# Figure 1 -- the positive control (paper section 7)
# Source: our_papers/paper1_variance_aware/archive_LRRFuse-VA/05_RESULTS.md, Run 6, "trained (n=5) / untrained (n=5)" table.
# ================================================================================================
# metric: (trained mean, trained sd, untrained mean, untrained sd, t at n=5)
CONTROL = {
    "Qabf": (0.4464, 0.0072, 0.1754, 0.0199, -28.58),
    "VIF":  (0.7865, 0.0063, 0.1365, 0.1137, -12.77),
    "SSIM": (0.9235, 0.0076, 0.0127, 0.2709,  -7.52),
    "SCD":  (1.3171, 0.0407, 0.2790, 0.3469,  -6.65),
    "MI":   (2.4728, 0.0350, 1.3929, 0.4452,  -5.41),
    "EN":   (6.9862, 0.0269, 6.2815, 0.3378,  -4.65),
    "SD":   (42.884, 0.991, 29.511, 9.354,    -3.18),
    "SF":   (10.909, 0.298,  9.0165, 2.9436,  -1.43),
    "AG":   (4.0585, 0.1049, 3.3860, 0.9809,  -1.52),
}

# The same arm on the second architecture (studied_papers/CrossFuse/.../results_poscontrol_cf, 2026-09-09).
# Sign convention matches CONTROL: t = (trained - untrained)/SE, so a NEGATIVE t means the
# untrained network scored HIGHER -- which is what MI and VIF do here.
CONTROL_CF = {
    "SF":   (14.4548, 0.3199, 10.7923, 0.3859,  16.34),
    "AG":   (5.3203, 0.1609,  3.9354, 0.1285,   15.04),
    "MI":   (2.2967, 0.0280,  3.4738, 0.1795,  -14.49),
    "VIF":  (0.7843, 0.0083,  0.9450, 0.0260,  -13.18),
    "EN":   (7.1787, 0.0360,  7.0140, 0.0329,    7.55),
    "SCD":  (1.5420, 0.0504,  0.9278, 0.2198,    6.09),
    "SSIM": (0.9524, 0.0117,  0.9041, 0.0196,    4.74),
    "SD":   (49.3629, 1.3853, 46.4051, 0.9611,   3.92),
    "Qabf": (0.5396, 0.0082,  0.5258, 0.0041,    3.38),
}


def fig1_positive_control():
    order = sorted(CONTROL, key=lambda m: abs(CONTROL[m][4]))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2, 3.9),
                                   gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.32})

    # (a) |t| against the critical value ---------------------------------------------------------
    y = np.arange(len(order))
    tvals = [abs(CONTROL[m][4]) for m in order]
    colors = [FAIL if t < T_CRIT else S1 for t in tvals]
    axL.barh(y, tvals, height=0.62, color=colors, zorder=3)
    axL.axvline(T_CRIT, color=INK2, lw=1.0, zorder=4)
    axL.text(T_CRIT * 1.08, len(order) - 0.35, f"$t_{{crit}}$ = {T_CRIT}", color=INK2, fontsize=8)
    axL.set_yticks(y)
    axL.set_yticklabels(order)
    axL.set_xscale("log")
    axL.set_xlim(0.8, 60)
    axL.set_xticks([1, 3, 10, 30])
    axL.set_xticklabels(["1", "3", "10", "30"])
    axL.set_xlabel("|t|, trained vs untrained  (log scale)")
    axL.set_title("(a)  LRRNet: SF and AG cannot see an untrained net", fontsize=9.5,
                  color=INK, loc="left", pad=9)
    for i, (m, t) in enumerate(zip(order, tvals)):
        if t < T_CRIT:
            axL.text(t * 1.12, i, "cannot", va="center", fontsize=8, color=FAIL)
    frame(axL, xgrid=True)

    # (b) the SAME arm on the second architecture, in the SAME metric order so the rows compare ---
    tvals2 = [abs(CONTROL_CF[m][4]) for m in order]
    inverted = [CONTROL_CF[m][4] < 0 for m in order]      # untrained scored HIGHER
    colors2 = [S2 if inv else (FAIL if t < T_CRIT else S1)
               for t, inv in zip(tvals2, inverted)]
    axR.barh(np.arange(len(order)), tvals2, height=0.62, color=colors2, zorder=3)
    axR.axvline(T_CRIT, color=INK2, lw=1.0, zorder=4)
    axR.text(T_CRIT * 1.08, len(order) - 0.35, f"$t_{{crit}}$ = {T_CRIT}", color=INK2, fontsize=8)
    axR.set_yticks(np.arange(len(order)))
    axR.set_yticklabels(order)
    axR.set_xscale("log")
    axR.set_xlim(0.8, 60)
    axR.set_xticks([1, 3, 10, 30])
    axR.set_xticklabels(["1", "3", "10", "30"])
    axR.set_xlabel("|t|, trained vs untrained  (log scale)")
    axR.set_title("(b)  CrossFuse: the same two metrics lead", fontsize=9.5,
                  color=INK, loc="left", pad=9)
    for i, (m, t, inv) in enumerate(zip(order, tvals2, inverted)):
        if inv:
            axR.text(t * 1.12, i, "untrained wins", va="center", fontsize=8, color=S2)
    # no legend: the two orange bars are labelled inline, and a swatch here collides with the SF bar
    frame(axR, xgrid=True)

    footer(fig,
             "The same untrained-network control on both architectures, metrics in the same order. "
             "SF and AG fail on LRRNet and are the strongest\ndetectors on CrossFuse; MI and VIF pass "
             "on LRRNet but rank the untrained network ABOVE the trained one on CrossFuse.\nWhether a "
             "metric responds is a property of the metric and the network together.",
             )
    save(fig, "fig_p1_positive_control.png")


# ================================================================================================
# Figure 2 -- the sensitivity ladder (paper section 7)
# Source: 05_RESULTS.md, Run 6 summary. Arms B and C are recorded as counts only, so the figure
# plots counts; it does not invent per-metric statistics for them.
# ================================================================================================
def fig2_sensitivity_ladder():
    """The ladder, on BOTH architectures.

    Until 2026-09-09 this figure asserted "between total failure to train and half the training
    budget, nothing is measurable" as a fact about the nine metrics. Run 8 measured that arm on
    the second architecture and got 2 of 9, so the sentence was true of LRRNet and false in
    general. A claim baked into a PNG is one no checker in this repository can read, which is why
    it survived a fully green build. Corrected here rather than caveated in the caption.

    Sources: LRRNet counts from our_papers/paper1_variance_aware/archive_LRRFuse-VA/05_RESULTS.md (Run 6); CrossFuse counts
    from studied_papers/CrossFuse/Tier3_CrossFuse/TIER3_RESULTS.md (Runs 7 and 8).
    """
    arms = [
        ("untrained" + chr(10) + "random weights, 0 steps", 7, 9),
        ("half the training budget" + chr(10) + "150 epochs instead of 299", 0, 2),
        ("a quarter of the depth" + chr(10) + "1 block instead of 4, same params", 0, None),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    y = np.arange(len(arms))[::-1].astype(float)
    h = 0.32
    for yi, (label, n_lrr, n_cf) in zip(y, arms):
        ax.barh(yi + h / 2 + 0.02, n_lrr, height=h, color=S1, zorder=3)
        ax.text(n_lrr + 0.16, yi + h / 2 + 0.02, str(n_lrr) + " of 9", va="center",
                fontsize=8.5, color=INK if n_lrr else MUTED,
                fontweight="bold" if n_lrr else "normal")
        if n_cf is None:
            ax.text(0.16, yi - h / 2 - 0.02, "not run on CrossFuse", va="center",
                    fontsize=8.5, color=MUTED, style="italic")
        else:
            ax.barh(yi - h / 2 - 0.02, n_cf, height=h, color=S2, zorder=3)
            ax.text(n_cf + 0.16, yi - h / 2 - 0.02, str(n_cf) + " of 9", va="center",
                    fontsize=8.5, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([a[0] for a in arms], fontsize=8.5)
    ax.set_xlim(0, 9.8)
    ax.set_xticks(range(0, 10, 3))
    ax.set_xlabel("metrics that detect the degradation (of 9)")
    ax.set_title("The same ladder, on two architectures", fontsize=10, color=INK,
                 loc="left", pad=10)
    # barh with empty data makes an EMPTY BarContainer, and the legend then draws both keys
    # in the default colour -- which is how this figure first rendered LRRNet and CrossFuse
    # as the same blue. Proxy patches carry the colour explicitly.
    ax.legend(handles=[mpatches.Patch(color=S1, label="LRRNet"),
                       mpatches.Patch(color=S2, label="CrossFuse")],
              loc="lower right", frameon=False, fontsize=8.5)
    frame(ax, xgrid=True)
    footer(fig,
           "On LRRNet nothing between total failure to train and half the training budget is "
           "measurable. On CrossFuse two metrics resolve the halved budget -- and they are MI and "
           "VIF," + chr(10) + "the same two that rank an untrained network ABOVE a trained one. "
           "The gap belongs to the metric-architecture pair, not to the metrics.",
           )
    save(fig, "fig_p2_sensitivity_ladder.png")


# ================================================================================================
# Figure 3 -- LRRNet's block-count table, re-measured (paper section 5)
# Source: studied_papers/LRRNet/LRRNET_RESULTS.md, "The completed table".
# ================================================================================================
# metric: {config: (mean, sd)}, n = 8 in every row after Run 9 (analyse_run9.py)
TABLE4 = {
    "EN":   {2: (6.9882, 0.0320), 4: (6.9791, 0.0247), 6: (6.9782, 0.0298), 8: (6.9840, 0.0298)},
    "MI":   {2: (2.4417, 0.0599), 4: (2.4874, 0.1060), 6: (2.4523, 0.0549), 8: (2.4886, 0.0394)},
    "SD":   {2: (43.4337, 0.9874), 4: (43.6686, 1.0372), 6: (43.8756, 1.0367), 8: (43.9778, 0.9496)},
    "SF":   {2: (10.9340, 0.1767), 4: (10.7195, 0.2934), 6: (10.4769, 0.2541), 8: (10.5277, 0.2216)},
    "AG":   {2: (4.0440, 0.1219), 4: (3.9365, 0.1207), 6: (3.8633, 0.1033), 8: (3.8700, 0.0697)},
    "VIF":  {2: (0.7682, 0.0311), 4: (0.7587, 0.0320), 6: (0.7370, 0.0207), 8: (0.7468, 0.0287)},
    "SCD":  {2: (1.3069, 0.0234), 4: (1.2576, 0.0519), 6: (1.2524, 0.0285), 8: (1.2373, 0.0412)},
    "Qabf": {2: (0.4374, 0.0137), 4: (0.4313, 0.0143), 6: (0.4233, 0.0088), 8: (0.4245, 0.0169)},
    "SSIM": {2: (0.9095, 0.0105), 4: (0.9049, 0.0134), 6: (0.9053, 0.0105), 8: (0.9086, 0.0169)},
}
N4 = 8
# The band is the test's own threshold expressed in pooled sigma, t_crit(df=14) * sqrt(2/8), not a
# fixed +/-3 sigma: at n = 8 a fixed band would contain the one cell the test calls significant.
BAND4 = 2.145 * (2 / N4) ** 0.5
SIG4 = {("SCD", 2)}          # from analyse_run9.py -- the figure marks it, it does not decide it
PARAMS4 = {2: 29744, 4: 49204, 6: 68664, 8: 88124}


def fig3_table4_remeasured():
    others = [2, 6, 8]
    marks = {2: "o", 6: "s", 8: "^"}
    cols = {2: S2, 6: S3, 8: S1}
    fig, ax = plt.subplots(figsize=(7.6, 3.6))

    ax.axvspan(-BAND4, BAND4, color=GRID, zorder=0)
    ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)
    for i, m in enumerate(METRICS):
        ref_mu, ref_sd = TABLE4[m][4]
        for j, nb in enumerate(others):
            mu, sd = TABLE4[m][nb]
            x = (mu - ref_mu) / pooled(ref_sd, N4, sd, N4)
            yy = i + (j - 1) * 0.22
            ax.plot(x, yy, marks[nb], ms=5, color=cols[nb], zorder=3)
            if (m, nb) in SIG4:
                ax.plot(x, yy, marks[nb], ms=11, mfc="none", mec=INK, mew=0.9, zorder=4)
                ax.annotate("p = 0.035 uncorrected — one of 27;\nnone after Holm or max-t", (x, yy),
                            xytext=(12, -2), textcoords="offset points", fontsize=7.5,
                            color=INK, va="center")
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels(METRICS)
    ax.set_ylim(-1.6, len(METRICS) + 0.05)
    ax.set_xlim(-2.2, 2.2)
    ax.set_xlabel("difference from the chosen setting, in pooled σ")
    ax.set_title("LRRNet's four-setting ablation, every row retrained at eight seeds", fontsize=10,
                 color=INK, loc="left", pad=10)
    ax.text(0, len(METRICS) - 0.28, f"±{BAND4:.2f} σ — the 5 % threshold at n = 8",
            ha="center", fontsize=8, color=MUTED)
    for nb in others:
        ax.plot([], [], marks[nb], ms=5, color=cols[nb],
                label=f"{nb} blocks · {PARAMS4[nb]:,} par.")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8, handletextpad=0.4, ncol=3,
              columnspacing=1.2)
    frame(ax, xgrid=True)
    footer(fig,
             "Eight seeds per setting, one evaluator, the same 25 TNO pairs. One of 27 comparisons "
             "reaches the 5 % level uncorrected (SCD, 4 blocks against 2),\nnone survives a Holm or max-t correction, "
             "and it favours the row the published table places worst. The band is the test's own "
             "threshold\nat n = 8 (equal-variance approximation); verdicts come from Welch's t per cell.",
             )
    save(fig, "fig_p3_table4_remeasured.png")


# ================================================================================================
# Figure 4 -- reading a published ablation row against a precision it never reported (section 4)
# Source: draft/04_case_study_I.md; the *values* are the authors' published numbers and the
# *sigma* is ours. That mixing is stated in the caption, as the section's revision note requires.
# ================================================================================================
# Delta in units of the SAMPLE SD of three seeds (analyse_tablevi_direct.py). Until 2026-09-13 these used
# the population SD and read 0.7 / 1.0 / 14.3 / 7.7 -- withdrawn claim #13.
ROW_I = [("EN", 7.21, 7.20, 0.6), ("SCD", 1.70, 1.67, 0.8), ("VIF", 0.77, 0.77, 0.0),
         ("SSIM", 0.99, 0.99, 0.0), ("MI", 2.13, 2.19, 11.5), ("Qabf", 0.54, 0.55, 6.3)]
# The published values carry two decimals, so each difference is uncertain by up to 0.01. Range in sigma
# consistent with that rounding, sigma = SIGMA_TNO of analyse_tablevi_direct.py, rounded half up (audit 2026-10-13).
ROW_I_RANGE = {"EN": (0.0, 1.2), "SCD": (0.5, 1.1), "VIF": (0.0, 6.3), "SSIM": (0.0, 1.4),
               "MI": (9.6, 13.5), "Qabf": (0.0, 12.5)}


def fig4_row_i():
    labels = [r[0] for r in ROW_I]
    deltas = [r[3] for r in ROW_I]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(labels))
    # robust = the whole rounding range lies above the 3 sigma line; only MI
    robust = [ROW_I_RANGE[l][0] > 3 for l in labels]
    colors = [S2 if r else MUTED for r in robust]
    ax.bar(x, deltas, width=0.55, color=colors, zorder=3)
    for xi, lbl in enumerate(labels):
        lo, hi = ROW_I_RANGE[lbl]
        ax.plot([xi, xi], [lo, hi], color=INK, lw=1.4, zorder=5)
        ax.plot([xi - 0.08, xi + 0.08], [hi, hi], color=INK, lw=1.4, zorder=5)
        ax.plot([xi - 0.08, xi + 0.08], [lo, lo], color=INK, lw=1.4, zorder=5)
    ax.axhline(3, color=INK2, lw=1.0, zorder=4)
    ax.text(-0.4, 3.4, "3 σ — a threshold for single published runs, not a test", color=INK2,
            fontsize=8, ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("published difference, in units of our σ")
    ax.set_ylim(0, 15)
    ax.set_title("The mechanism the method is named after, read against seed noise",
                 fontsize=10, color=INK, loc="left", pad=10)
    for xi, (lbl, r) in enumerate(zip(labels, robust)):
        hi = ROW_I_RANGE[lbl][1]
        if r:
            ax.text(xi + 0.12, hi - 1.2, "ablated\nmodel wins", ha="left", fontsize=8, color=S2)
        elif hi > 3:
            ax.text(xi + 0.12, hi - 1.2, "not resolvable\nat 2 d.p.", ha="left", fontsize=8, color=MUTED)
    frame(ax, xgrid=False)
    footer(fig,
             "Values are the authors' own published numbers; σ (sample SD of three retrained seeds of the paper's "
             "Eq. (16) schedule) is measured here.\nThe line on each bar is the range consistent with two-decimal "
             "rounding. Only MI exceeds the noise floor robustly, and it favours the model\nwithout the mechanism; "
             "the paper's stated justification rests on EN and SCD. This result cost no GPU time.",
             )
    save(fig, "fig_p4_row_i_vs_sigma.png")


# ================================================================================================
# Figure 5 -- the confound removed by construction (paper section 6)
# Source: studied_papers/LRRNet/LRRNET_RESULTS.md, Run 14 secondary (tied arms at five seeds, untied depth 8 at eight).
# ================================================================================================
CELLS = {                                    # (label, parameters, colour, marker)
    "untied d4": (49204, INK2, "o"), "untied d8": (88124, MUTED, "s"),
    "tied d4":   (20014, S1,   "D"), "tied d8":   (20014, S3,   "^"),
}
GRID2 = {   # computed by studied_papers/LRRNet/scripts/analyse_2x2_n5.py (Run 14 secondary reading, 2026-09-25)
    "EN":   {"untied d4": (6.9727, 0.0254), "untied d8": (6.9840, 0.0298), "tied d4": (6.9850, 0.0267), "tied d8": (6.9850, 0.0340)},
    "MI":   {"untied d4": (2.5021, 0.1357), "untied d8": (2.4886, 0.0394), "tied d4": (2.4720, 0.0349), "tied d8": (2.4980, 0.0356)},
    "SD":   {"untied d4": (43.5032, 0.8549), "untied d8": (43.9778, 0.9496), "tied d4": (42.8601, 0.9984), "tied d8": (43.1430, 1.4229)},
    "SF":   {"untied d4": (10.7123, 0.3715), "untied d8": (10.5277, 0.2216), "tied d4": (10.9125, 0.2943), "tied d8": (10.9005, 0.2359)},
    "AG":   {"untied d4": (3.9128, 0.1494), "untied d8": (3.8700, 0.0697), "tied d4": (4.0605, 0.1031), "tied d8": (4.0463, 0.0889)},
    "VIF":  {"untied d4": (0.7646, 0.0401), "untied d8": (0.7468, 0.0287), "tied d4": (0.7864, 0.0060), "tied d8": (0.7847, 0.0130)},
    "SCD":  {"untied d4": (1.2607, 0.0469), "untied d8": (1.2373, 0.0412), "tied d4": (1.3172, 0.0411), "tied d8": (1.3031, 0.0386)},
    "Qabf": {"untied d4": (0.4325, 0.0169), "untied d8": (0.4245, 0.0169), "tied d4": (0.4464, 0.0073), "tied d8": (0.4432, 0.0135)},
    "SSIM": {"untied d4": (0.9020, 0.0152), "untied d8": (0.9086, 0.0169), "tied d4": (0.9236, 0.0080), "tied d8": (0.9172, 0.0136)},
}


# The registered critical value for this 2x2 was a fixed 2.776 (df = 4). Expressed in pooled sigma
# for five seeds against three (equal-variance approximation) that is 2.776 * sqrt(1/5 + 1/3).
# A fixed +/-3 sigma band was drawn here before; it was never the test.
BAND5 = 2.306 * (1 / 5 + 1 / 5) ** 0.5   # t(0.975, df = 8) for 5 against 5
EXACT_ONLY5 = {("SSIM", "tied d4")}   # the one cell against the reference that crosses: p = 0.030, none after Holm
N_ARM = {"untied d4": 5, "untied d8": 8, "tied d4": 5, "tied d8": 5}

# Panel (b): Run 17's registered test of the depth-8 contrast on NEW seeds 62-73 (n = 12 per arm), tied d8 minus
# untied d8 in pooled sigma, on TNO (kernel_csv/lrrnet-run17-*) and RoadScene (results_run17b/run17b_RoadScene.csv).
RUN17 = {
    "EN": (+0.10, +2.26), "MI": (+0.11, +0.74), "SD": (-1.57, +0.67), "SF": (+2.03, +1.94), "AG": (+1.59, +1.62),
    "VIF": (+1.85, +2.66), "SCD": (+2.23, +2.61), "Qabf": (+1.48, +3.17), "SSIM": (+2.63, +5.58),
}
RUN17_CONFIRM = {"SF", "AG", "VIF", "SCD"}          # the registered family; all four survive Holm on both test sets
BAND12 = 2.074 * (1 / 12 + 1 / 12) ** 0.5            # t(0.975, df = 22) for 12 against 12


def fig5_capacity_by_construction():
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 3.9), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.08})
    ax.axvspan(-BAND5, BAND5, color=GRID, zorder=0)
    ax.axvline(0, color=BASELINE, lw=0.8, zorder=1)
    ref = "untied d4"
    offs = {"untied d8": 0.22, "tied d4": 0.0, "tied d8": -0.22}
    for i, m in enumerate(METRICS):
        ref_mu, ref_sd = GRID2[m][ref]
        for cfg, off in offs.items():
            mu, sd = GRID2[m][cfg]
            sp = pooled(ref_sd, 5, sd, N_ARM[cfg])
            _, col, mk = CELLS[cfg]
            x = (mu - ref_mu) / sp
            ax.plot(x, i + off, mk, ms=5, color=col, zorder=3)
            if (m, cfg) in EXACT_ONLY5:
                ax.plot(x, i + off, mk, ms=11, mfc="none", mec=INK, mew=0.9, zorder=4)
                ax.annotate("p = 0.030;\nnone after Holm", (x, i + off),
                            xytext=(10, -12), textcoords="offset points", fontsize=7.5, color=INK,
                            va="center")
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels(METRICS)
    ax.set_ylim(-1.6, len(METRICS) + 0.05)
    ax.set_xlim(-3.2, 3.2)
    ax.set_xlabel("difference from the published configuration, in pooled σ")
    ax.set_title("(a) 4.40× in capacity, 2× in depth, 5 seeds: 1 cell of 27 crosses",
                 fontsize=10, color=INK, loc="left", pad=10)
    ax.text(0, len(METRICS) - 0.28, f"±{BAND5:.2f} σ — 5 % threshold, 5 vs 5 seeds",
            ha="center", fontsize=8, color=MUTED)
    for cfg in offs:
        p, col, mk = CELLS[cfg]
        ax.plot([], [], mk, ms=5, color=col, label=f"{cfg} · {p:,} par.")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8, handletextpad=0.4, ncol=3,
              columnspacing=1.2)
    frame(ax, xgrid=True)

    # (b) the registered test of the one contrast (a) concentrated in
    bx.axvspan(-BAND12, BAND12, color=GRID, zorder=0)
    bx.axvline(0, color=BASELINE, lw=0.8, zorder=1)
    for i, m in enumerate(METRICS):
        for (x, off, col, mk) in ((RUN17[m][0], 0.14, S3, "^"), (RUN17[m][1], -0.14, S1, "o")):
            filled = m in RUN17_CONFIRM
            bx.plot(x, i + off, mk, ms=5.5, color=col, mfc=col if filled else "none", mew=1.1, zorder=3)
    bx.set_yticks(range(len(METRICS)))
    bx.set_yticklabels([])
    bx.set_ylim(-1.6, len(METRICS) + 0.05)
    bx.set_xlim(-2.5, 6.0)
    bx.set_xlabel("tied d8 − untied d8, in pooled σ")
    bx.set_title("(b) registered test, 12 new seeds: 4 of 4 after Holm, both sets",
                 fontsize=10, color=INK, loc="left", pad=10)
    bx.text(0, len(METRICS) - 0.28, f"±{BAND12:.2f} σ — 5 % threshold, 12 vs 12",
            ha="center", fontsize=8, color=MUTED)
    bx.plot([], [], "^", ms=5.5, color=S3, label="TNO")
    bx.plot([], [], "o", ms=5.5, color=S1, label="RoadScene")
    bx.plot([], [], "s", ms=5.5, mfc="none", mec=MUTED, label="not in the registered family")
    bx.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=8, handletextpad=0.4, ncol=3,
              columnspacing=1.0)
    frame(bx, xgrid=True)
    footer(fig,
             "(a) Weight tying makes the parameter count identical at depth 4 and depth 8 — 20,014 in both — so depth "
             "varies with the parameter count held fixed by construction; tied arms at five seeds,\nuntied depth 8 at eight; "
             "reference untied depth 4, n = 5. Across its four contrasts 5 of 36 cross uncorrected, four of them tied d8 "
             "against untied d8, none after Holm.\n(b) That contrast, registered and retested on twelve new seeds per arm "
             "(Run 17) and re-scored on RoadScene (Run 17b): filled markers are the registered family (SF, AG, VIF,\nSCD), "
             "all surviving Holm on both test sets, all favouring the 4.40x smaller tied network. Open markers are "
             "descriptive; SD on TNO is a cost (-1.57 σ).",
             )
    save(fig, "fig_p5_capacity_by_construction.png")


# ================================================================================================
# Figure 6 -- what MI and VIF actually prefer (paper section 7)
# Fused with the SAME checkpoints as the section 7 table: CrossFuse_RSM3_PC45.pth (trained, 299
# epochs) and CrossFuse_RSM3_U42.pth (epoch 0, untrained=True). Metric values are the per-image rows
# of results_poscontrol_cf, transcribed, not recomputed here.
# ================================================================================================
SRC = os.path.join(OUT, "sources")
QUAL_BOX = (24, 359, 372, 480)     # the hot figure, located by thresholding the infrared band
QUAL_PANELS = [
    ("Infrared", "tno02_ir.png", None, None),
    ("Visible", "tno02_vi.png", None, None),
    ("Trained (299 epochs)", "tno02_fused_trained_PC45.png", 2.762, 0.794),
    ("Untrained (0 steps)", "tno02_fused_untrained_U42.png", 3.940, 0.892),
]


def fig6_untrained_qualitative():
    import cv2
    x0, y0, x1, y1 = QUAL_BOX
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 4.1),
                             gridspec_kw={"height_ratios": [2.5, 1.0],
                                          "hspace": 0.10, "wspace": 0.04})
    for i, (title, fname, mi, vif) in enumerate(QUAL_PANELS):
        path = os.path.join(SRC, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit("missing figure source: " + path)
        edge = FAIL if title.startswith("Untrained") else INK2

        ax = axes[0, i]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(MUTED); sp.set_linewidth(0.6)
        ax.set_title(title, fontsize=9.5, color=INK, pad=5)
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                        edgecolor=edge, linewidth=1.1))

        az = axes[1, i]
        az.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
        az.set_xticks([]); az.set_yticks([])
        for sp in az.spines.values():
            sp.set_color(edge); sp.set_linewidth(1.1)
        if mi is not None:
            colour = S2 if title.startswith("Untrained") else INK2
            az.set_xlabel("MI " + format(mi, ".3f") + "    VIF " + format(vif, ".3f"),
                          fontsize=9, color=colour, labelpad=4,
                          fontweight="bold" if colour == S2 else "normal")

    axes[1, 0].set_xlabel("the person the infrared band sees", fontsize=8.5, color=INK2, labelpad=4)
    axes[1, 1].set_xlabel("not visible in the visible band", fontsize=8.5, color=MUTED, labelpad=4)
    footer(fig,
           "The trained network carries the person through; the untrained network loses them. "
           "MI and VIF score the image WITHOUT the person higher.")
    save(fig, "fig_p6_untrained_qualitative.png")


if __name__ == "__main__":
    print(f"writing to {OUT}")
    fig1_positive_control()
    fig2_sensitivity_ladder()
    fig3_table4_remeasured()
    fig4_row_i()
    fig5_capacity_by_construction()
    fig6_untrained_qualitative()
    print("done -- 6 figures")
