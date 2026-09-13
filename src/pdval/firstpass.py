"""First pass over ingested vintages: targets, waterfall, data-quality summary.

Usage:
    uv run python -m pdval.firstpass --config configs/default.yaml [--years ...]

Writes:
    data/processed/target_<year>.parquet
    outputs/firstpass.md      human-readable report
    outputs/firstpass.json    machine-readable numbers
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd
import yaml

from pdval.target import build_target, waterfall

NA_CODES = {"fico": 9999, "cltv": 999, "dti": 999, "ltv": 999, "mi_pct": 999, "num_units": 99}


def vintage_summary(year: int, orig: pd.DataFrame, target: pd.DataFrame) -> dict:
    t = target.set_index("loan_id").drop(columns=["first_payment_date"])
    o = orig.set_index("loan_id").join(t, how="left")
    obs = o[o["y"].notna()]
    out: dict = {"vintage": year, "loans": len(o)}
    out["waterfall"] = waterfall(target).to_dict("records")
    out["default_rate_36m"] = float(obs["y"].mean()) if len(obs) else None
    out["defaults"] = int(obs["y"].sum()) if len(obs) else 0
    out["default_type"] = obs.loc[obs["y"] == 1, "default_type"].value_counts().to_dict()
    out["default_in_forbearance"] = int(obs.loc[obs["y"] == 1, "default_in_forbearance"].fillna(False).sum())
    out["harp_loans"] = int((o["harp_indicator"].fillna("") == "Y").sum())
    out["amortization_types"] = o["amortization_type"].value_counts(dropna=False).to_dict()
    # NA rates (Freddie sentinel codes + genuine nulls)
    na = {}
    for col, code in NA_CODES.items():
        s = o[col]
        na[col] = float(((s == code) | s.isna()).mean())
    for col in ("msa", "first_time_homebuyer", "channel", "loan_purpose", "occupancy", "property_type"):
        s = o[col]
        na[col] = float((s.isna() | s.astype(str).isin(["9", "99", "999", "9999", ""])).mean())
    out["na_rates"] = na
    # Distributions on the observable population
    def q(col):
        s = obs[col]
        s = s[(s != NA_CODES.get(col, -1))]
        return {f"{k:g}": float(v) for k, v in s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2).items()}
    out["quantiles"] = {c: q(c) for c in ("fico", "ltv", "cltv", "dti", "orig_upb", "orig_rate")}
    # Default rate by FICO band, LTV band — the monotonicity sanity check
    def band_rate(col, bins, labels):
        s = obs[col].where(obs[col] != NA_CODES.get(col, -1))
        g = obs.groupby(pd.cut(s, bins=bins, labels=labels), observed=True)["y"]
        return {str(k): {"n": int(v.count()), "rate": float(v.mean())} for k, v in g}
    out["by_fico"] = band_rate("fico", [0, 660, 700, 740, 780, 851], ["<660", "660-699", "700-739", "740-779", "780+"])
    out["by_ltv"] = band_rate("ltv", [0, 60, 70, 80, 90, 95, 999], ["<=60", "61-70", "71-80", "81-90", "91-95", ">95"])
    out["by_dti"] = band_rate("dti", [0, 25, 35, 43, 50, 66], ["<=25", "26-35", "36-43", "44-50", ">50"])
    out["by_purpose"] = {str(k): {"n": int(v.count()), "rate": float(v.mean())} for k, v in obs.groupby("loan_purpose")["y"]}
    out["by_occupancy"] = {str(k): {"n": int(v.count()), "rate": float(v.mean())} for k, v in obs.groupby("occupancy")["y"]}
    return out


def render_md(summaries: list[dict], cfg: dict) -> str:
    L = [
        "# First pass — Freddie Mac sample vintages\n",
        (
            f"Window {cfg['target']['window_months']}m, 90+ DPD{' (excluding months in forbearance plan)' if cfg['target'].get('exclude_forbearance_dpd', True) else ''} or credit-event termination. "
            f"Performance cutoff {cfg['data']['performance_cutoff']}.\n"
        ),
    ]
    L.append("## Waterfall and default rate\n")
    L.append("| vintage | loans | defect | censored | incomplete | observable | defaults | 36m default rate |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        w = {r["step"].strip(): r["loans"] for r in s["waterfall"]}
        rate = f"{s['default_rate_36m']:.2%}" if s["default_rate_36m"] is not None else "n/a"
        L.append(f"| {s['vintage']} | {s['loans']:,} | {-w['minus defect']:,} | {-w['minus censored']:,} | "
                 f"{-w['minus incomplete_window']:,} | {w['observable loans']:,} | {s['defaults']:,} | {rate} |")
    L.append("\n## NA / sentinel rates by field\n")
    cols = list(summaries[0]["na_rates"])
    L.append("| vintage | " + " | ".join(cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for s in summaries:
        L.append(f"| {s['vintage']} | " + " | ".join(f"{s['na_rates'][c]:.1%}" for c in cols) + " |")
    L.append("\n## Default rate by band (monotonicity sanity check)\n")
    for key, title in (("by_fico", "FICO"), ("by_ltv", "LTV"), ("by_dti", "DTI"), ("by_purpose", "Purpose"), ("by_occupancy", "Occupancy")):
        L.append(f"### {title}\n")
        bands = list(summaries[0][key])
        L.append("| vintage | " + " | ".join(bands) + " |")
        L.append("|---|" + "---|" * len(bands))
        for s in summaries:
            L.append(f"| {s['vintage']} | " + " | ".join(
                f"{s[key].get(b, {}).get('rate', float('nan')):.2%} (n={s[key].get(b, {}).get('n', 0):,})" for b in bands) + " |")
        L.append("")
    L.append("## Quantiles (observable loans)\n")
    for s in summaries:
        L.append(f"**{s['vintage']}**: " + "; ".join(
            f"{c} p5/p50/p95 = {v['0.05']:g}/{v['0.5']:g}/{v['0.95']:g}" for c, v in s["quantiles"].items()))
    L.append("\n## Other\n")
    for s in summaries:
        L.append(f"- {s['vintage']}: HARP loans {s['harp_loans']:,}; amortization {s['amortization_types']}; "
                 f"default types {s['default_type']}; defaults while in forbearance {s['default_in_forbearance']}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--years", nargs="*", type=int)
    ap.add_argument("--processed-dir", default=None)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    proc = Path(args.processed_dir or cfg["data"]["processed_dir"])
    years = args.years or sorted(int(p.stem.split("_")[1]) for p in proc.glob("orig_*.parquet"))
    if not years:
        raise SystemExit(f"No orig_*.parquet in {proc}. Run `make data` first.")
    tcfg = cfg["target"]
    con = duckdb.connect()
    summaries = []
    for y in years:
        orig_p, perf_p = proc / f"orig_{y}.parquet", proc / f"perf_{y}.parquet"
        target = build_target(
            str(orig_p), str(perf_p), window=tcfg["window_months"], dpd_threshold=tcfg["dpd_threshold"],
            credit_event_zb_codes=tcfg["credit_event_zb_codes"], payoff_zb_codes=tcfg["payoff_zb_codes"],
            censor_zb_codes=tcfg["censor_zb_codes"], defect_zb_codes=tcfg["defect_zb_codes"],
            performance_cutoff=cfg["data"]["performance_cutoff"],
            exclude_forbearance_dpd=tcfg.get("exclude_forbearance_dpd", True),
            forbearance_lookback=tcfg.get("forbearance_lookback_months", 3),
            exclude_disaster_dpd=tcfg.get("exclude_disaster_dpd", True), con=con,
        )
        target.to_parquet(proc / f"target_{y}.parquet", index=False)
        orig = con.execute(f"SELECT * FROM '{orig_p.as_posix()}'").df()
        s = vintage_summary(y, orig, target)
        summaries.append(s)
        print(f"  {y}: observable {s['waterfall'][-2]['loans']:,} | defaults {s['defaults']:,} | "
              f"rate {s['default_rate_36m'] if s['default_rate_36m'] is None else f'{s['default_rate_36m']:.2%}'}")
    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    (out / "firstpass.json").write_text(json.dumps(summaries, indent=2, default=str))
    (out / "firstpass.md").write_text(render_md(summaries, cfg))
    print(f"wrote {out / 'firstpass.md'}")


if __name__ == "__main__":
    main()
