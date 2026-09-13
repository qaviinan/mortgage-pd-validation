"""Calibration: reliability table, calibration-in-the-large, slope/intercept, Brier, ECE, Hosmer-Lemeshow.

Pure functions over (y_true, p_pred). Follows Van Calster et al. (2019) hierarchy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    if strategy == "quantile":
        df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    else:
        df["bin"] = pd.cut(df["p"], bins=np.linspace(0, df["p"].max(), bins + 1), labels=False, include_lowest=True)
    g = df.groupby("bin").agg(n=("y", "size"), defaults=("y", "sum"), mean_pred=("p", "mean"), obs_rate=("y", "mean")).reset_index()
    ci = g.apply(lambda r: wilson(int(r["defaults"]), int(r["n"])), axis=1)
    g["obs_ci_low"] = [c[0] for c in ci]; g["obs_ci_high"] = [c[1] for c in ci]
    g["expected"] = g["mean_pred"] * g["n"]
    return g


def calibration_in_the_large(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    obs, pred = float(y.mean()), float(p.mean())
    lo, hi = wilson(int(y.sum()), len(y))
    return {"n": len(y), "observed": obs, "predicted": pred, "ratio_obs_pred": obs / pred if pred > 0 else np.nan,
            "rel_error": (pred - obs) / obs if obs > 0 else np.nan, "obs_ci_low": lo, "obs_ci_high": hi}


def slope_intercept(y: np.ndarray, p: np.ndarray) -> dict:
    """Logistic recalibration: logit(y) ~ a + b*logit(p). Ideal a=0, b=1.
    Intercept reported with slope fixed at 1 (calibration-in-the-large on the logit scale) and free."""
    y = np.asarray(y); p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e9, max_iter=1000).fit(lp, y)
    b = float(lr.coef_[0, 0]); a = float(lr.intercept_[0])
    # standard errors via observed information
    z = a + b * lp[:, 0]
    q = 1 / (1 + np.exp(-z))
    w = q * (1 - q)
    X = np.column_stack([np.ones_like(lp[:, 0]), lp[:, 0]])
    info = X.T @ (X * w[:, None])
    try:
        cov = np.linalg.inv(info)
        se_a, se_b = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        se_a = se_b = np.nan
    # intercept with slope fixed at 1
    off = LogisticRegression(C=1e9, max_iter=1000, fit_intercept=True)
    a_fixed = float(np.log(y.mean() / (1 - y.mean())) - np.mean(lp[:, 0])) if 0 < y.mean() < 1 else np.nan
    _ = off
    return {"intercept": a, "intercept_se": se_a, "slope": b, "slope_se": se_b,
            "slope_ci_low": b - 1.96 * se_b, "slope_ci_high": b + 1.96 * se_b,
            "intercept_fixed_slope": a_fixed}


def brier(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    bs = brier_score_loss(y, p)
    base = y.mean()
    bs_ref = brier_score_loss(y, np.full_like(p, base))
    return {"brier": float(bs), "brier_ref": float(bs_ref), "brier_skill": float(1 - bs / bs_ref) if bs_ref > 0 else np.nan}


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    t = reliability_table(y, p, bins=bins)
    return float((t["n"] / t["n"].sum() * (t["obs_rate"] - t["mean_pred"]).abs()).sum())


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """Reported for completeness; known to be over-powered at large n and sensitive to binning."""
    t = reliability_table(y, p, bins=bins)
    o = t["defaults"].to_numpy(); e = t["expected"].to_numpy(); n = t["n"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.nansum((o - e) ** 2 / (e * (1 - e / n)))
    dof = bins - 2
    return {"chi2": float(chi), "dof": dof, "p_value": float(stats.chi2.sf(chi, dof))}


def summary(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    out = {}
    out.update({f"citl_{k}": v for k, v in calibration_in_the_large(y, p).items()})
    out.update(slope_intercept(y, p))
    out.update(brier(y, p))
    out["ece"] = ece(y, p, bins)
    hl = hosmer_lemeshow(y, p, bins)
    out.update({"hl_chi2": hl["chi2"], "hl_p": hl["p_value"]})
    return out
