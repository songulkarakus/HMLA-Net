"""Statistical analysis: Friedman+Nemenyi, Wilcoxon, McNemar, bootstrap CI, Cohen's d."""
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Bootstrap confidence interval (for any metric)
# --------------------------------------------------------------------------- #
def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    stat = [metric_fn(y_true[idx], y_pred[idx])
            for idx in (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric_fn(y_true, y_pred)), float(lo), float(hi)


# --------------------------------------------------------------------------- #
# McNemar test — the errors of two classifiers on the same test set
# --------------------------------------------------------------------------- #
def mcnemar_test(y_true, pred_a, pred_b) -> Dict:
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    n01 = int(np.sum(a_ok & ~b_ok))   # A correct, B wrong
    n10 = int(np.sum(~a_ok & b_ok))   # A wrong, B correct
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0, "test": "none"}
    # With few discordant pairs (n<25) the chi-square approximation is INVALID and the
    # continuity correction inflates p (conservative) — i.e. it works against the
    # proposed model. Between two strong models the number of discordant pairs is
    # routinely below 25, hence the EXACT binomial test is used by default.
    if n_disc < 25:
        try:
            p = float(stats.binomtest(n01, n_disc, 0.5).pvalue)   # scipy >= 1.7
        except AttributeError:
            p = float(stats.binom_test(n01, n_disc, 0.5))         # older scipy
        return {"n01": n01, "n10": n10, "statistic": float("nan"),
                "p_value": p, "test": "exact_binomial"}
    stat = (abs(n01 - n10) - 1) ** 2 / n_disc
    p = float(stats.chi2.sf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(stat),
            "p_value": p, "test": "chi2_continuity"}


# --------------------------------------------------------------------------- #
# Paired tests (scores across seeds/folds)
# --------------------------------------------------------------------------- #
def paired_tests(scores_a: Sequence[float], scores_b: Sequence[float]) -> Dict:
    a, b = np.asarray(scores_a, float), np.asarray(scores_b, float)
    out = {}
    try:
        t, p = stats.ttest_rel(a, b)
        out["paired_t_stat"], out["paired_t_p"] = float(t), float(p)
    except Exception:
        out["paired_t_stat"], out["paired_t_p"] = float("nan"), float("nan")
    try:
        w, p = stats.wilcoxon(a, b)
        out["wilcoxon_stat"], out["wilcoxon_p"] = float(w), float(p)
    except Exception:
        out["wilcoxon_stat"], out["wilcoxon_p"] = float("nan"), float("nan")
    out["cohens_dz"] = cohens_dz(a, b)
    out["cohens_d"] = cohens_d_pooled(a, b)
    out["mean_diff"] = float(a.mean() - b.mean())
    # With n=3 seeds the SMALLEST two-sided p Wilcoxon can reach is 0.25; after the Holm
    # correction no comparison can come out significant. The number of blocks is
    # reported too, so the table shows whether there was enough power.
    out["n_blocks"] = int(len(a))
    out["wilcoxon_min_possible_p"] = float(_wilcoxon_min_p(len(a)))
    return out


def _wilcoxon_min_p(n: int) -> float:
    """Smallest p the Wilcoxon signed-rank test can reach with n paired blocks."""
    if n < 1:
        return float("nan")
    return min(1.0, 2.0 ** (1 - n))          # all differences in the same direction -> 2/2^n


def cohens_dz(a, b) -> float:
    """Paired-sample effect size d_z = mean(diff)/sd(diff)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def cohens_d_pooled(a, b) -> float:
    """Classic Cohen's d (pooled sd).

    d_z used to be returned under the name `cohens_d`. For correlated scores measured
    on the same seed/split, d_z is far larger than the standardised mean difference;
    reporting it as 'Cohen's d' inflates the effect size and is an inconsistency anyone
    can check. Both are reported separately.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def holm_bonferroni(pvalues, alpha: float = 0.05):
    """Holm-Bonferroni correction (multiple comparisons). NaNs are preserved.
    Returns: (list of adjusted p, list of reject flags)."""
    p = np.asarray(pvalues, float)
    valid = ~np.isnan(p)
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    adj = np.full_like(p, np.nan)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)          # monotonicity
        adj[i] = min(running, 1.0)
    reject = np.where(valid, adj < alpha, False)
    return adj.tolist(), reject.tolist()


# --------------------------------------------------------------------------- #
# Friedman + Nemenyi post-hoc (the standard for multi-model comparisons)
#   perf_matrix: row = data set / seed block, column = model
# --------------------------------------------------------------------------- #
def friedman_nemenyi(perf_matrix: np.ndarray, model_names: List[str]) -> Dict:
    perf = np.asarray(perf_matrix, float)
    n_blocks, k = perf.shape
    result = {"n_blocks": n_blocks, "k_models": k, "model_names": list(model_names)}

    stat, p = stats.friedmanchisquare(*[perf[:, j] for j in range(k)])
    result["friedman_stat"] = float(stat)
    result["friedman_p"] = float(p)

    # average ranks (higher performance = lower rank number = better)
    ranks = np.zeros_like(perf)
    for i in range(n_blocks):
        ranks[i] = stats.rankdata(-perf[i])
    avg_ranks = ranks.mean(0)
    result["avg_ranks"] = {m: float(r) for m, r in zip(model_names, avg_ranks)}

    # Nemenyi critical difference (CD), alpha=0.05
    q_alpha = _nemenyi_q(k)
    cd = q_alpha * math_sqrt(k * (k + 1) / (6.0 * n_blocks))
    result["nemenyi_cd"] = float(cd)
    result["q_alpha"] = float(q_alpha)

    # full p-value matrix if scikit-posthocs is available
    try:
        import scikit_posthocs as sp
        import pandas as pd
        df = pd.DataFrame(perf, columns=model_names)
        pmat = sp.posthoc_nemenyi_friedman(df)
        result["nemenyi_pmatrix"] = pmat.to_dict()
    except Exception:
        result["nemenyi_pmatrix"] = None
    return result


def math_sqrt(x):
    import math
    return math.sqrt(x)


def _nemenyi_q(k: int) -> float:
    """Nemenyi q_0.05 critical values (for k models, infinite df).

    The table was extended to 30: this study uses 25 baselines + 1 proposed = 26 models
    and the old table ended at 25. The old code said "use the nearest UPPER value" but
    picked the NEAREST value with `min(..., key=abs(kk-k))`; for k=26 it returned
    q(25)=3.658 and showed the CD smaller than it is (anti-conservative).
    """
    table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
             8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
             14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517,
             20: 3.544, 21: 3.569, 22: 3.593, 23: 3.616, 24: 3.637, 25: 3.658,
             26: 3.678, 27: 3.696, 28: 3.714, 29: 3.732, 30: 3.749}
    if k in table:
        return table[k]
    keys = sorted(table)
    upper = [kk for kk in keys if kk >= k]
    return table[upper[0]] if upper else table[keys[-1]]
