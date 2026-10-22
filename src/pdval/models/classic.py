"""Classic models: FICO-only logistic baseline and WoE scorecard challenger.

Kept in an importable module (not the training script) so pickled artifacts load anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression


class FicoBaseline:
    def __init__(self):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
        self.fill = None

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        f = df["fico"].to_numpy(dtype=float)
        miss = np.isnan(f).astype(float)
        f = np.where(np.isnan(f), self.fill, f)
        return np.column_stack([(f - 700.0) / 50.0, miss])

    def fit(self, df: pd.DataFrame, y: np.ndarray):
        self.fill = float(np.nanmedian(df["fico"]))
        self.lr.fit(self._x(df), y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.lr.predict_proba(self._x(df))[:, 1]


class WoeScorecard:
    """optbinning BinningProcess (monotonic trend for numerics) -> WoE -> logistic regression."""

    def __init__(self, numeric: list[str], categorical: list[str], monotone: dict[str, int], C: float = 1.0):
        self.numeric, self.categorical = numeric, categorical
        self.cols = numeric + categorical
        params = {}
        for c in numeric:
            trend = {1: "ascending", -1: "descending"}.get(monotone.get(c, 0), "auto_asc_desc")
            params[c] = {"monotonic_trend": trend, "max_n_bins": 8, "min_bin_size": 0.02}
        for c in categorical:
            params[c] = {"max_n_bins": 8, "min_bin_size": 0.01}
        self.bp = BinningProcess(self.cols, categorical_variables=categorical, binning_fit_params=params,
                                 min_prebin_size=0.01)
        self.lr = LogisticRegression(C=C, max_iter=2000)

    def _prep(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[self.cols].copy()
        for c in self.categorical:
            x[c] = x[c].astype(object).where(x[c].notna(), "NA")
        return x

    def fit(self, df: pd.DataFrame, y: np.ndarray):
        x = self._prep(df)
        self.bp.fit(x, y)
        w = self.bp.transform(x, metric="woe")
        self.lr.fit(w, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        w = self.bp.transform(self._prep(df), metric="woe")
        return self.lr.predict_proba(w)[:, 1]

    def iv_table(self) -> pd.DataFrame:
        return self.bp.summary()[["name", "dtype", "n_bins", "iv", "js"]].sort_values("iv", ascending=False)
