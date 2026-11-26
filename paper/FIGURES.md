# Figure set — 5 figures

Built by [`make_paper_figures.py`](make_paper_figures.py) into [`figures/`](figures/). Re-run the
script after any change to a source number; nothing here is hand-edited.

Style follows `thesis/tools/make_figures2.py`: light surface only, the first three validated
categorical slots plus one red reserved for *failure*, hairline grid on one axis, two spines.
**No figure plots a quantity the text does not also state.** Where only a summary count was recorded
(positive-control arms B and C), the figure shows the count rather than inventing per-metric
statistics.

| # | File | § | Carries |
|---|---|---|---|
| 1 | `fig_p1_positive_control.png` | 7 | **The headline.** Two panels: which metrics separate trained from untrained, and why SF and AG do not |
| 2 | `fig_p2_sensitivity_ladder.png` | 7 | The resolution gap on **both** architectures — LRRNet 7/0/0 against CrossFuse 9/2/(not run). The half-budget arm is where they part |
| 3 | `fig_p3_table4_remeasured.png` | 5 | LRRNet's four-setting table retrained at n = 8; one point of 27 crosses |
| 4 | `fig_p4_row_i_vs_sigma.png` | 4 | A published ablation row read against a precision its authors never measured |
| 5 | `fig_p5_capacity_by_construction.png` | 6 | (a) 4.40× in capacity, 2× in depth, five seeds: 1 of 27; (b) the depth-8 contrast retested on 12 new seeds: 4 of 4 after Holm on TNO and RoadScene |
| 6 | `fig_p6_untrained_qualitative.png` | 7 | **The picture behind the inversion.** The untrained network loses the person the infrared band sees, and MI and VIF score it higher for that. Built from four tracked frames in `figures/sources/` |

Figure 1 is the paper's headline and should be placed early — §7's closing line is the one a reader
quotes, and the figure is what makes it land.

---

## Draft captions

Each figure already carries its explanatory text as a footer inside the PNG, so the LaTeX caption
stays short. If the venue's style prefers everything in the caption, move the footer text there and
regenerate with the `fig.text(...)` calls removed.

**Figure 1.** *The positive control.* Seven of nine metrics separate a fully trained LRRNet from the
same architecture at random weights; SF and AG do not, at five seeds per arm, and SD reaches
significance only at five. Adapted from the parameter-randomisation test of Adebayo et al.

**Figure 2.** *What these nine metrics can and cannot resolve.* The sensitivity floor lies between
total failure to train and half the training budget. The two lower arms show no detected difference,
with comparable training losses (non-detection, not demonstrated equivalence).

**Figure 3.** *LRRNet's block-count ablation, every row retrained.* Eight seeds per setting, one
evaluator, the same 25 TNO pairs. One of 27 comparisons reaches the 5 % level — SCD, 4 blocks against
2 — uncorrected, at the chance rate, and none survives a Holm or max-*t* correction; that cell did not replicate on twelve new seeds. The
shaded band is the significance threshold at n = 8
(≈1.07 pooled σ), not a fixed ±3 σ rule.

**Figure 4.** *Reading a published row against seed noise.* **Values are the authors' published
numbers; σ is measured here** and was never reported by them — the caption must say so, or a reader
will take the whole row as our measurement. Since Run 31 σ is the eight-seed estimate; hollow markers show the three-seed values, under which MI
looked robust (11.5 σ, withdrawn claim #21).

**Figure 5.** *Depth with the parameter count fixed.* Weight tying makes the parameter count
identical at depth 4 and depth 8, so depth varies with the parameter count held fixed by construction rather
than controlled for afterwards. Tied arms at five seeds (Runs 12 and 14), untied depth 8 at eight.
Against the published configuration one cell of 27 crosses (SSIM, tied depth 4, p = 0.030, ringed);
across all four contrasts of the 2×2, 5 of 36 cross uncorrected — four of them tied against untied at
depth 8 — all favouring the tied model, none after Holm. The shaded band is the 5 % threshold for five
seeds against five (≈1.46 pooled σ), not a fixed ±3 σ rule. *(Updated 2026-09-25; at three tied seeds
the figure read 1 of 36 at the registered threshold, 2 at exact df.)*

*(Updated 2026-09-27: panel (b) added — Run 17's registered test of the depth-8 contrast on twelve new seeds per
arm, with Run 17b's RoadScene re-score; the registered family SF, AG, VIF, SCD survives Holm on both test sets.)*

---

## Open

| | |
|---|---|
| ✅ | Vector output — done 2026-09-14 after the fifth external review: `make_paper_figures.py` writes a PDF beside every PNG (byte-stable), `paper.tex` includes figures without an extension and prefers `.pdf`. The decks and `release/` keep the PNGs (release forbids `.pdf`), which LaTeX falls back to |
| ⬜ | If §7's two tables become Figure 1, decide whether the nine-metric table still appears in the body or moves to the appendix |
| ⚠️ | Figure 4 mixes published values with our σ. That must be in the printed caption, not only here |
