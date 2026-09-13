"""Discrimination metrics: AUC/Gini/KS/PR-AUC with bootstrap CIs, DeLong test, segment and decile tables.

Pure functions over (y_true, p_pred[, segments]). No I/O, no plotting, no dataset-specific names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return {"n": len(y), "defaults": int(y.sum()), "auc": np.nan, "gini": np.nan, "ks": np.nan, "pr_auc": np.nan}
    auc = roc_auc_score(y, p)
    return {"n": len(y), "defaults": int(y.sum()), "auc": float(auc), "gini": float(2 * auc - 1),
            "ks": ks_stat(y, p), "pr_auc": float(average_precision_score(y, p))}


def bootstrap_auc(y: np.ndarray, p: np.ndarray, reps: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for AUC, resampling loans with replacement (stratified by outcome)."""
    y = np.asarray(y); p = np.asarray(p)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = np.empty(reps)
    for i in range(reps):
        ip = rng.choice(pos, len(pos), replace=True)
        ineg = rng.choice(neg, len(neg), replace=True)
        idx = np.concatenate([ip, ineg])
        aucs[i] = roc_auc_score(y[idx], p[idx])
    return {"auc": float(roc_auc_score(y, p)), "ci_low": float(np.quantile(aucs, alpha / 2)),
            "ci_high": float(np.quantile(aucs, 1 - alpha / 2)), "se": float(aucs.std(ddof=1)), "reps": reps}


# --- DeLong (1988) fast implementation after Sun & Xu (2014); structure follows
# --- github.com/yandexdataschool/roc_comparison (BSD), re-implemented here.
def _midrank(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def _delong_components(y: np.ndarray, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """preds: (k, n) array of k models' scores. Returns AUCs (k,) and covariance (k, k)."""
    order = np.argsort(-y, kind="stable")
    y = y[order]; preds = preds[:, order]
    m = int(y.sum()); n = len(y) - m
    k = preds.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(preds[r, :m]); ty[r] = _midrank(preds[r, m:]); tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_test(y: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> dict:
    """Two-sided DeLong test that AUC(p1) == AUC(p2) on the same loans (paired)."""
    y = np.asarray(y).astype(int)
    aucs, cov = _delong_components(y, np.vstack([np.asarray(p1), np.asarray(p2)]))
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else np.nan
    pval = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diff": float(diff), "se_diff": float(np.sqrt(var)),
            "z": float(z), "p_value": pval}


def by_segment(y: np.ndarray, p: np.ndarray, seg: pd.Series, min_defaults: int = 30, reps: int = 300, seed: int = 0) -> pd.DataFrame:
    """Point metrics + bootstrap AUC CI per segment value; CI only where defaults >= min_defaults."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "seg": seg.astype(str).to_numpy()})
    rows = []
    for k, g in df.groupby("seg", sort=True):
        m = point_metrics(g["y"].to_numpy(), g["p"].to_numpy())
        m["segment"] = k
        m["obs_rate"] = float(g["y"].mean()); m["mean_pred"] = float(g["p"].mean())
        if m["defaults"] >= min_defaults:
            b = bootstrap_auc(g["y"].to_numpy(), g["p"].to_numpy(), reps=reps, seed=seed)
            m["auc_ci_low"], m["auc_ci_high"] = b["ci_low"], b["ci_high"]
        else:
            m["auc_ci_low"] = m["auc_ci_high"] = np.nan
        rows.append(m)
    cols = ["segment", "n", "defaults", "obs_rate", "mean_pred", "auc", "auc_ci_low", "auc_ci_high", "gini", "ks", "pr_auc"]
    return pd.DataFrame(rows)[cols]


def decile_table(y: np.ndarray, p: np.ndarray, q: int = 10) -> pd.DataFrame:
    """Score-decile capture table (decile 1 = riskiest)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = q - pd.qcut(df["p"].rank(method="first"), q, labels=False)
    g = df.groupby("decile").agg(n=("y", "size"), defaults=("y", "sum"), mean_pred=("p", "mean"), obs_rate=("y", "mean")).sort_index()
    g["cum_capture"] = g["defaults"].cumsum() / g["defaults"].sum()
    g["cum_pop"] = g["n"].cumsum() / g["n"].sum()
    g["lift"] = g["obs_rate"] / df["y"].mean()
    return g.reset_index()
