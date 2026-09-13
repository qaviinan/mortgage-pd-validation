"""Assemble the modelling frame: one row per observable loan across all vintages.

Usage:
    uv run python -m pdval.features --config configs/default.yaml

Writes data/processed/model_frame.parquet with:
    loan_id, vintage, role (train/tune/oot/stress/covid_exhibit), y,
    model inputs (config features.numeric + features.categorical, sentinels -> NaN),
    segmentation-only columns (config features.excluded_by_design + harp_indicator, default_age, ...)

Applies the population exclusions from config (HARP, amortization type) and records a waterfall.
No performance-file column other than the target ever enters this frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

SENTINELS = {"fico": 9999, "ltv": 999, "cltv": 999, "dti": 999, "mi_pct": 999, "num_units": 99,
             "num_borrowers": 99, "vantagescore4": 9999}
CAT_NA = {"first_time_homebuyer": "9", "occupancy": "9", "channel": "9", "property_type": "99",
          "loan_purpose": "9", "super_conforming": "", "special_program": "9"}

SEGMENT_COLS = ["state", "msa", "zip3", "seller_name", "harp_indicator", "special_program",
                "first_payment_date", "default_age", "default_type", "default_in_forbearance", "default_in_disaster"]


def role_map(cfg: dict) -> dict[int, str]:
    out: dict[int, str] = {}
    for role, years in cfg["vintages"].items():
        for y in years:
            out[int(y)] = role
    return out


def build_frame(cfg: dict, processed: Path) -> tuple[pd.DataFrame, list[dict]]:
    con = duckdb.connect()
    roles = role_map(cfg)
    num = cfg["features"]["numeric"]
    cat = cfg["features"]["categorical"]
    frames, waterfall = [], []
    for year, role in sorted(roles.items()):
        o = con.execute(f"SELECT * FROM '{(processed / f'orig_{year}.parquet').as_posix()}'").df()
        t = con.execute(
            f"SELECT loan_id, y, default_age, default_type, default_in_forbearance, default_in_disaster "
            f"FROM '{(processed / f'target_{year}.parquet').as_posix()}'"
        ).df()
        df = o.merge(t, on="loan_id", how="inner")
        w = {"vintage": year, "role": role, "loans": len(df)}
        df = df[df["y"].notna()]
        w["observable"] = len(df)
        if cfg["exclusions"].get("drop_harp", True):
            df = df[df["harp_indicator"].fillna("N") != "Y"]
            w["after_harp_exclusion"] = len(df)
        keep_amort = cfg["exclusions"].get("amortization_types")
        if keep_amort:
            df = df[df["amortization_type"].isin(keep_amort)]
            w["after_amortization_filter"] = len(df)
        w["defaults"] = int(df["y"].sum())
        waterfall.append(w)
        df = df.assign(vintage=year, role=role)
        frames.append(df)
    frame = pd.concat(frames, ignore_index=True)

    # Sentinel codes -> NaN for numerics; explicit "NA" level for categoricals
    for c in num:
        if c in SENTINELS:
            frame.loc[frame[c] == SENTINELS[c], c] = np.nan
        frame[c] = frame[c].astype("float64")
    for c in cat:
        s = frame[c].astype("string")
        na_code = CAT_NA.get(c)
        s = s.mask(s.isna() | (s == na_code) | (s == ""), "NA")
        frame[c] = pd.Categorical(s)
    frame["y"] = frame["y"].astype("int8")

    cols = ["loan_id", "vintage", "role", "y"] + num + cat + [c for c in SEGMENT_COLS if c in frame.columns]
    return frame[cols], waterfall


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    processed = Path(cfg["data"]["processed_dir"])
    frame, waterfall = build_frame(cfg, processed)
    out = processed / "model_frame.parquet"
    frame.to_parquet(out, index=False)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/feature_waterfall.json").write_text(json.dumps(waterfall, indent=2))
    print(f"wrote {out}: {len(frame):,} loans x {frame.shape[1]} cols")
    print(pd.DataFrame(waterfall).to_string(index=False))
    print("\nby role:")
    print(frame.groupby("role").agg(loans=("y", "size"), defaults=("y", "sum"), rate=("y", "mean")).to_string())


if __name__ == "__main__":
    main()
