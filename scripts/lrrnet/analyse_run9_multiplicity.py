"""Run 9, read three further ways -- in answer to an external review, not to a new result.

The pre-registered reading of Run 9 (analyse_run9.py) is "1 of 27 uncorrected, against 1.35 expected
by chance". A reviewer raised two objections to that sentence, and both are fair:

  1. The nine metrics are correlated and the three comparisons share the nb=4 arm, so 27 tests are
     not 27 independent coin flips. The EXPECTED count is still 1.35 (linearity of expectation holds
     under any dependence), but "fewer than expected" is a comparison of one count against a mean,
     not a test -- and its distribution under correlation is unknown.
  2. A non-significant difference is not evidence of no difference. "Indistinguishable" needs an
     equivalence statement, not an absence of rejection.

THIS SCRIPT CHANGES NO REPORTED NUMBER AND NO PRE-REGISTERED READING. It adds three readings that do
not depend on independence or on reading a null as a positive:

  A. Holm and Benjamini-Hochberg over the 27 Welch p-values.
  B. A max-|t| permutation test (Westfall-Young): the 32 runs are re-assigned to the four settings at
     random, and each permutation records the largest |t| over all 27 cells. Correlation between
     metrics and the shared reference arm are preserved by construction, because whole runs move.
  C. TOST equivalence at a margin of +/-1.75 pooled sigma, with 90 % confidence intervals.

THE MARGIN WAS NOT CHOSEN HERE. 1.75 sigma is the resolution RUN9_PREREGISTRATION.md stated on
2026-09-09, before any Run 9 seed existed, as what n = 8 per row could detect at 80 % power. It was
registered as a POWER bound, not as an equivalence margin, and that difference is stated wherever
reading C is used. No other margin is reported, so no margin was shopped for.

READINGS, WRITTEN BEFORE THIS SCRIPT WAS FIRST RUN
  (The script and its first output were committed together, in 785b024, so git cannot show this
  order. It is the author's statement, not a verifiable record. Future readings are committed on
  their own before the run, as RUN9_PREREGISTRATION.md was.)
  A/B: if the SCD cell survives Holm or the max-t test, "not a finding" is WITHDRAWN and it is
       reported as the paper's one real cell. If it does not, the corrected count is 0 of 27 and is
       reported beside -- never instead of -- the uncorrected 1 of 27.
  C:   cells whose 90 % CI lies inside +/-1.75 sigma are "equivalent within the design's resolution".
       Cells whose CI crosses the margin are "inconclusive", and the paper's word "indistinguishable"
       must not be applied to them as if it meant equivalent.

    python studied_papers/LRRNet/scripts/analyse_run9_multiplicity.py
"""
import contextlib
import io
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    import analyse_run9 as r9          # kill criteria run on import; it exits if any fails

METRICS = r9.METRICS
REF, OTHERS = 4, (2, 6, 8)
MARGIN = 1.75                          # RUN9_PREREGISTRATION.md, table row n = 8
N_PERM = 20_000
ALPHA = 0.05

X = {nb: np.array([r9.runs[nb][s] for s in sorted(r9.runs[nb])]) for nb in (2, 4, 6, 8)}
for nb, a in X.items():
    assert a.shape == (8, 9), f"nb={nb}: expected 8 runs x 9 metrics, got {a.shape}"


def welch(a, b):
    """t (a - b), Welch-Satterthwaite df, two-sided p. Vectorised over metrics (columns)."""
    na, nb = len(a), len(b)
    va, vb = a.var(axis=0, ddof=1) / na, b.var(axis=0, ddof=1) / nb
    t = (a.mean(axis=0) - b.mean(axis=0)) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    return t, df, 2 * stats.t.sf(np.abs(t), df)


def all_t(groups):
    return np.concatenate([welch(groups[REF], groups[nb])[0] for nb in OTHERS])


cells = [(nb, m) for nb in OTHERS for m in METRICS]
t_obs, p_obs = [], []
for nb in OTHERS:
    t, _, p = welch(X[REF], X[nb])
    t_obs.extend(t)
    p_obs.extend(p)
t_obs, p_obs = np.array(t_obs), np.array(p_obs)
K = len(cells)

# ---------------------------------------------------------------------------------------- A. Holm, BH
order = np.argsort(p_obs)
holm = np.empty(K)
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (K - rank) * p_obs[i]))
    holm[i] = running
bh = np.empty(K)
running = 1.0
for rank in range(K - 1, -1, -1):
    i = order[rank]
    running = min(running, p_obs[i] * K / (rank + 1))
    bh[i] = running

# ------------------------------------------------------------------------------ B. max-|t| permutation
rng = np.random.default_rng(20260913)
pool = np.vstack([X[nb] for nb in (2, 4, 6, 8)])
max_null = np.empty(N_PERM)
for k in range(N_PERM):
    idx = rng.permutation(len(pool))
    g = {nb: pool[idx[j * 8:(j + 1) * 8]] for j, nb in enumerate((2, 4, 6, 8))}
    max_null[k] = np.abs(all_t(g)).max()
p_maxt = np.array([(1 + np.sum(max_null >= abs(t))) / (N_PERM + 1) for t in t_obs])
counts = np.empty(N_PERM // 4, dtype=int)
for k in range(len(counts)):
    idx = rng.permutation(len(pool))
    g = {nb: pool[idx[j * 8:(j + 1) * 8]] for j, nb in enumerate((2, 4, 6, 8))}
    ps = np.concatenate([welch(g[REF], g[nb])[2] for nb in OTHERS])
    counts[k] = int(np.sum(ps < ALPHA))

# ------------------------------------------------------------------------------------------ C. TOST
tost = []
for nb in OTHERS:
    a, b = X[REF], X[nb]
    na, nb_ = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(axis=0, ddof=1) + (nb_ - 1) * b.var(axis=0, ddof=1)) / (na + nb_ - 2))
    d = (a.mean(axis=0) - b.mean(axis=0)) / sp                        # in pooled sigma
    se = np.sqrt(a.var(axis=0, ddof=1) / na + b.var(axis=0, ddof=1) / nb_) / sp
    _, df, _ = welch(a, b)
    q = stats.t.ppf(1 - ALPHA, df)
    lo, hi = d - q * se, d + q * se
    p_tost = np.maximum(stats.t.sf((d + MARGIN) / se, df), stats.t.cdf((d - MARGIN) / se, df))
    for j, m in enumerate(METRICS):
        tost.append((nb, m, d[j], lo[j], hi[j], p_tost[j]))

# ------------------------------------------------------------------------------------------- report
print("=" * 100)
print("RUN 9 -- MULTIPLICITY AND EQUIVALENCE (27 cells: nb=4 against 2, 6, 8 on nine metrics, n=8 each)")
print("=" * 100)
print(f"{'cell':>14}{'t':>8}{'p':>9}{'Holm':>9}{'BH':>9}{'max-t':>9}   {'d (pooled s)':>13}"
      f"{'90% CI':>18}{'TOST p':>9}  reading")
print("-" * 100)
for i, (nb, m) in enumerate(cells):
    _, _, d, lo, hi, pt = tost[i]
    equiv = pt < ALPHA
    reading = "equivalent" if equiv else "inconclusive"
    flag = " *" if p_obs[i] < ALPHA else "  "
    print(f"{f'4v{nb} {m}':>14}{t_obs[i]:>8.2f}{p_obs[i]:>9.3f}{holm[i]:>9.3f}{bh[i]:>9.3f}"
          f"{p_maxt[i]:>9.3f}{flag} {d:>+12.2f}  [{lo:+.2f}, {hi:+.2f}]{pt:>9.3f}  {reading}")

n_raw = int(np.sum(p_obs < ALPHA))
n_holm, n_bh, n_maxt = (int(np.sum(v < ALPHA)) for v in (holm, bh, p_maxt))
n_eq = sum(1 for row in tost if row[5] < ALPHA)
crit_maxt = np.quantile(max_null, 1 - ALPHA)
p_count = float(np.mean(counts >= n_raw))

print(f"""
{'=' * 100}
RESULT
{'=' * 100}
  uncorrected (Welch, per cell)       : {n_raw} of 27
  Holm (family-wise error)            : {n_holm} of 27
  Benjamini-Hochberg (false discovery): {n_bh} of 27
  max-|t| permutation, {N_PERM:,} draws   : {n_maxt} of 27    (family-wise critical |t| = {crit_maxt:.2f})

  Under the permutation null the uncorrected count has mean {counts.mean():.2f} (analytic 1.35) and
  P(count >= {n_raw}) = {p_count:.2f}. Correlation widens that distribution; it does not move its mean.

  TOST at +/-{MARGIN} pooled sigma (the pre-registered design resolution, NOT a pre-registered margin):
    equivalent   : {n_eq} of 27
    inconclusive : {27 - n_eq} of 27
""")

scd = cells.index((2, "SCD"))
if holm[scd] < ALPHA or p_maxt[scd] < ALPHA:
    print("  READING A/B: the SCD cell SURVIVES correction. 'Not a finding' is withdrawn; report it as\n"
          "  the one real cell in the table.")
else:
    print(f"  READING A/B: the SCD cell does not survive (Holm {holm[scd]:.2f}, max-t {p_maxt[scd]:.2f}).\n"
          "  Corrected count 0 of 27, reported beside the uncorrected 1 of 27, never instead of it.")
print(f"""  READING C: {n_eq} of 27 cells are equivalent within the design's resolution. The remaining
  {27 - n_eq} are inconclusive at n = 8 -- 'indistinguishable' may describe them, 'equivalent' may not.""")
