import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pdval.validation import calibration as cal
from pdval.validation import discrimination as disc
from pdval.validation import stability as stab


def _data(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-3 + 1.2 * x)))
    y = (rng.random(n) < p).astype(int)
    return y, p, x


def test_delong_matches_auc_and_is_symmetric():
    y, p, x = _data()
    p2 = 1 / (1 + np.exp(-(-3 + 0.6 * x + np.random.default_rng(1).normal(scale=0.5, size=len(x)))))
    t = disc.delong_test(y, p, p2)
    assert abs(t["auc_1"] - roc_auc_score(y, p)) < 1e-9
    assert abs(t["auc_2"] - roc_auc_score(y, p2)) < 1e-9
    assert t["p_value"] < 0.05           # p is clearly better
    same = disc.delong_test(y, p, p)
    assert same["diff"] == 0


def test_bootstrap_ci_contains_point():
    y, p, _ = _data()
    b = disc.bootstrap_auc(y, p, reps=100, seed=1)
    assert b["ci_low"] <= b["auc"] <= b["ci_high"]


def test_calibration_ideal_model():
    y, p, _ = _data(n=20000)
    s = cal.summary(y, p)
    assert abs(s["citl_ratio_obs_pred"] - 1) < 0.1
    assert 0.85 < s["slope"] < 1.15
    assert s["brier_skill"] > 0
    assert cal.reliability_table(y, p)["n"].sum() == len(y)


def test_psi_zero_on_identical_and_positive_on_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert stab.psi(a, a)["psi"] < 1e-6
    assert stab.psi(a, a + 1.0)["psi"] > 0.25
    s = pd.Series(list("aaabbc"))
    assert stab.psi_categorical(s, s)["psi"] < 1e-6
