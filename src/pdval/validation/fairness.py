"""Disparate-impact proxy analysis on geography-derived groups.

The data has no protected attributes, so groups come from the demographic make-up of the loan's
metro area (ACS 5-year table B03002, CBSA or metropolitan division). This follows the CFPB's 2014
proxy method without its surname component. The results are a screen and cannot support a
compliance determination.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import roc_auc_score

from pdval.validation.calibration import calibration_in_the_large

ACS_URL = "https://api.census.gov/data/2022/acs/acs5"
VARS = "NAME,B03002_001E,B03002_003E,B03002_004E,B03002_012E"   # total, NH white, NH black, hispanic


def fetch_acs_cbsa(cache: Path) -> pd.DataFrame:
    """Fetch (or load cached) ACS demographic shares for every CBSA and metropolitan division."""
    if cache.exists():
        return pd.read_parquet(cache)
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        raise RuntimeError("CENSUS_API_KEY not set (free key: https://api.census.gov/data/key_signup.html)")
    # Metropolitan divisions must be requested nested under their parent MSA; only these MSAs have divisions.
    MSAS_WITH_DIVISIONS = ["14460", "16980", "19100", "19820", "31080", "33100", "35620", "37980", "41860", "42660", "47900"]
    queries = [{"for": "metropolitan statistical area/micropolitan statistical area:*"}]
    queries += [{"for": "metropolitan division:*", "in": f"metropolitan statistical area/micropolitan statistical area:{m}"}
                for m in MSAS_WITH_DIVISIONS]
    frames = []
    for geo in queries:
        r = requests.get(ACS_URL, params={"get": VARS, **geo, "key": key}, timeout=60)
        r.raise_for_status()
        if "json" not in r.headers.get("content-type", ""):
            raise RuntimeError(f"Census API returned non-JSON ({r.url}); key rejected?")
        rows = r.json()
        df = pd.DataFrame(rows[1:], columns=rows[0])
        is_div = "division" in geo["for"]
        code_col = "metropolitan division" if is_div else "metropolitan statistical area/micropolitan statistical area"
        df = df.rename(columns={code_col: "geo_code"})
        df["geo_type"] = "division" if is_div else "cbsa"
        frames.append(df[["NAME", "B03002_001E", "B03002_003E", "B03002_004E", "B03002_012E", "geo_code", "geo_type"]])
    out = pd.concat(frames, ignore_index=True)
    for c in ("B03002_001E", "B03002_003E", "B03002_004E", "B03002_012E"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["geo_code"] = out["geo_code"].astype(str).str.zfill(5)
    out["share_nonwhite"] = 1 - out["B03002_003E"] / out["B03002_001E"]
    out["share_black"] = out["B03002_004E"] / out["B03002_001E"]
    out["share_hispanic"] = out["B03002_012E"] / out["B03002_001E"]
    out = out.rename(columns={"NAME": "geo_name", "B03002_001E": "population"})[
        ["geo_code", "geo_type", "geo_name", "population", "share_nonwhite", "share_black", "share_hispanic"]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def assign_groups(msa: pd.Series, acs: pd.DataFrame, n_groups: int = 3) -> pd.DataFrame:
    """Map loan MSA codes to area non-white share; return group labels (tertiles + 'non-metro')."""
    codes = msa.astype("Int64").astype(str).str.zfill(5).where(msa.notna(), None)
    # metropolitan divisions take precedence in Freddie's field, so try division then CBSA
    lut = {}
    for _, r in acs.sort_values("geo_type").iterrows():  # 'cbsa' < 'division' so division overwrites
        lut[r["geo_code"]] = r["share_nonwhite"]
    share = codes.map(lut)
    out = pd.DataFrame({"msa": msa, "area_share_nonwhite": share})
    matched = share.notna()
    q = pd.qcut(share[matched], n_groups, labels=[f"T{i + 1}" for i in range(n_groups)])
    out["group"] = "non-metro/unknown"
    out.loc[matched, "group"] = q.astype(str)
    out.loc[msa.notna() & ~matched, "group"] = "metro-unmatched"
    return out


def group_metrics(y: np.ndarray, p: np.ndarray, group: pd.Series, cutoff: float, reference: str) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": group.astype(str).to_numpy()})
    rows = []
    for g, d in df.groupby("g"):
        citl = calibration_in_the_large(d["y"].to_numpy(), d["p"].to_numpy())
        auc = roc_auc_score(d["y"], d["p"]) if d["y"].nunique() == 2 else np.nan
        rows.append({"group": g, "n": len(d), "defaults": int(d["y"].sum()), "obs_rate": citl["observed"],
                     "mean_pred": citl["predicted"], "pred_over_obs": citl["predicted"] / citl["observed"] if citl["observed"] > 0 else np.nan,
                     "auc": float(auc), "mean_score": float(d["p"].mean()),
                     "approval_rate": float((d["p"] < cutoff).mean())})
    t = pd.DataFrame(rows).set_index("group")
    ref_rate = t.loc[reference, "approval_rate"]
    t["adverse_impact_ratio"] = t["approval_rate"] / ref_rate
    pooled_sd = float(df["p"].std())
    t["std_mean_diff"] = (t["mean_score"] - t.loc[reference, "mean_score"]) / pooled_sd
    return t.reset_index()


def proxy_feature_audit(X: pd.DataFrame, share: pd.Series, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Association of each input with the area demographic share (Spearman for numerics, correlation ratio for categoricals)."""
    rows = []
    m = share.notna()
    for c in numeric:
        rows.append({"variable": c, "type": "numeric", "assoc": float(X.loc[m, c].corr(share[m], method="spearman"))})
    for c in categorical:
        s = share[m]; g = X.loc[m, c].astype(str)
        grand = s.mean()
        ss_between = sum(len(v) * (v.mean() - grand) ** 2 for _, v in s.groupby(g))
        ss_total = ((s - grand) ** 2).sum()
        rows.append({"variable": c, "type": "categorical", "assoc": float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("assoc", key=lambda s: s.abs(), ascending=False)


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))
