"""Sensitivity: permutation importance, SHAP, partial dependence, one-at-a-time perturbations, monotonicity."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

Predict = Callable[[pd.DataFrame], np.ndarray]


def permutation_importance(predict: Predict, X: pd.DataFrame, y: np.ndarray, cols: list[str], reps: int = 3,
                           seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, predict(X))
    rows = []
    for c in cols:
        drops = []
        for _ in range(reps):
            Xp = X.copy()
            # keep the column dtype (LightGBM requires identical categorical dtypes at predict time)
            Xp[c] = pd.Series(Xp[c].to_numpy()[rng.permutation(len(Xp))], index=Xp.index, dtype=Xp[c].dtype)
            drops.append(base - roc_auc_score(y, predict(Xp)))
        rows.append({"variable": c, "auc_drop": float(np.mean(drops)), "auc_drop_sd": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("auc_drop", ascending=False).reset_index(drop=True)


def partial_dependence(predict: Predict, X: pd.DataFrame, col: str, grid: np.ndarray | list, sample: int = 20000,
                       seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xs = X.iloc[rng.choice(len(X), min(sample, len(X)), replace=False)].copy()
    rows = []
    for v in grid:
        Xg = Xs.copy()
        if isinstance(Xg[col].dtype, pd.CategoricalDtype):
            Xg[col] = pd.Categorical([v] * len(Xg), categories=Xg[col].cat.categories)
        else:
            Xg[col] = v
        p = predict(Xg)
        rows.append({"value": v, "mean_pred": float(p.mean()), "p10": float(np.quantile(p, 0.1)), "p90": float(np.quantile(p, 0.9))})
    return pd.DataFrame(rows)


def one_at_a_time(predict: Predict, X: pd.DataFrame, shocks: dict[str, list[float]]) -> pd.DataFrame:
    """Portfolio-level predicted default rate under additive shocks to one input at a time."""
    base = float(predict(X).mean())
    rows = [{"variable": "(none)", "shock": 0.0, "mean_pred": base, "rel_change": 0.0}]
    for c, deltas in shocks.items():
        for d in deltas:
            Xs = X.copy()
            Xs[c] = Xs[c] + d
            m = float(predict(Xs).mean())
            rows.append({"variable": c, "shock": d, "mean_pred": m, "rel_change": m / base - 1})
    return pd.DataFrame(rows)


def monotonicity_check(predict: Predict, X: pd.DataFrame, col: str, direction: int, grid: np.ndarray, sample: int = 5000,
                       seed: int = 0) -> dict:
    """Fraction of sampled loans whose prediction moves against `direction` along the grid."""
    rng = np.random.default_rng(seed)
    Xs = X.iloc[rng.choice(len(X), min(sample, len(X)), replace=False)].copy()
    preds = []
    for v in grid:
        Xg = Xs.copy(); Xg[col] = v
        preds.append(predict(Xg))
    P = np.vstack(preds)                     # (grid, n)
    diffs = np.diff(P, axis=0) * direction   # should be >= 0
    viol = (diffs < -1e-9).any(axis=0)
    return {"variable": col, "direction": direction, "violation_rate": float(viol.mean()), "n": len(Xs)}


def shap_summary(booster, X: pd.DataFrame, cols: list[str], sample: int = 20000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Mean |SHAP| per feature for a LightGBM booster; returns (table, shap_values, X_sample)."""
    import shap

    rng = np.random.default_rng(seed)
    Xs = X.iloc[rng.choice(len(X), min(sample, len(X)), replace=False)][cols].copy()
    ex = shap.TreeExplainer(booster)
    sv = ex.shap_values(Xs)
    if isinstance(sv, list):
        sv = sv[1]
    tab = pd.DataFrame({"variable": cols, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    return tab.reset_index(drop=True), sv, Xs
