"""Stability: PSI on scores, CSI on inputs, metric drift over time. Pure functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, edges: np.ndarray | None = None,
        eps: float = 1e-6) -> dict:
    """Population Stability Index on quantile bins of the reference distribution.
    The usual 0.10 / 0.25 cut-offs are convention, not derived from error rates (Yurdakul 2018)."""
    reference = np.asarray(reference, dtype=float); current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]; current = current[~np.isnan(current)]
    if edges is None:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, bins=edges)[0] / max(len(reference), 1)
    c = np.histogram(current, bins=edges)[0] / max(len(current), 1)
    r = np.clip(r, eps, None); c = np.clip(c, eps, None)
    contrib = (c - r) * np.log(c / r)
    return {"psi": float(contrib.sum()), "bins": len(r), "edges": edges, "ref_share": r, "cur_share": c, "contrib": contrib}


def psi_categorical(reference: pd.Series, current: pd.Series, eps: float = 1e-6) -> dict:
    r = reference.astype(str).value_counts(normalize=True)
    c = current.astype(str).value_counts(normalize=True)
    cats = sorted(set(r.index) | set(c.index))
    r = np.clip(r.reindex(cats).fillna(0).to_numpy(), eps, None)
    c = np.clip(c.reindex(cats).fillna(0).to_numpy(), eps, None)
    contrib = (c - r) * np.log(c / r)
    return {"psi": float(contrib.sum()), "bins": len(cats), "categories": cats, "ref_share": r, "cur_share": c, "contrib": contrib}


def csi_table(reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    rows = []
    for c in numeric:
        rows.append({"variable": c, "type": "numeric", "csi": psi(reference[c].to_numpy(), current[c].to_numpy())["psi"],
                     "ref_missing": float(reference[c].isna().mean()), "cur_missing": float(current[c].isna().mean())})
    for c in categorical:
        rows.append({"variable": c, "type": "categorical", "csi": psi_categorical(reference[c], current[c])["psi"],
                     "ref_missing": float((reference[c].astype(str) == "NA").mean()),
                     "cur_missing": float((current[c].astype(str) == "NA").mean())})
    return pd.DataFrame(rows).sort_values("csi", ascending=False)


def traffic_light(value: float, amber: float = 0.10, red: float = 0.25) -> str:
    if np.isnan(value):
        return "n/a"
    return "red" if value >= red else "amber" if value >= amber else "green"
