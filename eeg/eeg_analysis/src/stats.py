"""Statistical helpers: paired Wilcoxon, LMM, Hedges' g, FDR."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def hedges_g_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g for paired samples (x - y)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = x - y
    n = d.size
    if n < 2:
        return np.nan
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0
    g = d.mean() / sd
    # small-sample correction
    J = 1 - 3 / (4 * n - 9)
    return float(g * J)


def wilcoxon_paired(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if x.size < 5:
        return {"n": int(x.size), "W": np.nan, "p": np.nan,
                "hedges_g": np.nan}
    W, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    return {"n": int(x.size), "W": float(W), "p": float(p),
            "hedges_g": hedges_g_paired(x, y)}


def fdr_bh(pvals: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    pvals = np.asarray(pvals, float)
    mask = ~np.isnan(pvals)
    q = np.full_like(pvals, np.nan)
    if mask.sum() == 0:
        return q
    _, qvals, _, _ = multipletests(pvals[mask], alpha=alpha, method="fdr_bh")
    q[mask] = qvals
    return q


def bootstrap_ci(x: np.ndarray, statistic=np.median,
                 n_boot: int = 2000, ci: float = 0.95,
                 rng: np.random.Generator | None = None
                 ) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = rng if rng is not None else np.random.default_rng(42)
    n = x.size
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = statistic(rng.choice(x, size=n, replace=True))
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(statistic(x)), float(lo), float(hi)


def lmm_stage_effect(df: pd.DataFrame, value_col: str,
                     stage_col: str = "stage",
                     subject_col: str = "subject",
                     ref: str = "Wake") -> pd.DataFrame:
    """Fit `value ~ C(stage, Treatment(ref)) + (1|subject)` and return a
    tidy coefficient table."""
    import statsmodels.formula.api as smf
    sub = df.dropna(subset=[value_col]).copy()
    sub[stage_col] = sub[stage_col].astype("category")
    formula = f"{value_col} ~ C({stage_col}, Treatment(reference='{ref}'))"
    model = smf.mixedlm(formula, sub, groups=sub[subject_col])
    res = model.fit(method="lbfgs", reml=True, disp=False)
    out = pd.DataFrame({
        "term": res.params.index,
        "beta": res.params.values,
        "se": res.bse.values,
        "z": res.tvalues.values,
        "p": res.pvalues.values,
    })
    return out
