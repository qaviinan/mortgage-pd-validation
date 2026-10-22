"""Run the validation battery on the scored model frame and emit tables + findings.

Usage:
    uv run python -m pdval.validate --config configs/default.yaml [--skip-fairness] [--fast]

Reads  data/processed/model_frame.parquet, data/processed/predictions.parquet, outputs/models/*
Writes outputs/validation/*.csv|json and outputs/findings.csv
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import yaml

from pdval.train import lgb_dataset
from pdval.validation import calibration as cal
from pdval.validation import discrimination as disc
from pdval.validation import fairness as fair
from pdval.validation import sensitivity as sens
from pdval.validation import stability as stab
from pdval.validation.findings import Register

MODELS = ["baseline", "challenger", "champion"]
SEGMENTS = {
    "fico_band": lambda d: pd.cut(d["fico"], [0, 660, 700, 740, 780, 851], labels=["<660", "660-699", "700-739", "740-779", "780+"]),
    "ltv_band": lambda d: pd.cut(d["ltv"], [0, 60, 70, 80, 90, 95, 999], labels=["<=60", "61-70", "71-80", "81-90", "91-95", ">95"]),
    "dti_band": lambda d: pd.cut(d["dti"], [0, 25, 35, 43, 50, 66], labels=["<=25", "26-35", "36-43", "44-50", ">50"]),
    "loan_purpose": lambda d: d["loan_purpose"],
    "occupancy": lambda d: d["occupancy"],
    "channel": lambda d: d["channel"],
    "property_type": lambda d: d["property_type"],
    "first_time_homebuyer": lambda d: d["first_time_homebuyer"],
    "num_borrowers": lambda d: d["num_borrowers"].map({1: "1", 2: "2"}).fillna("3+"),
    "region": lambda d: d["state"].map(REGION).fillna("other"),
    "metro": lambda d: np.where(d["msa"].isna(), "non-metro", "metro"),
    "vintage": lambda d: d["vintage"].astype(str),
}
REGION = {**{s: "West" for s in ["WA", "OR", "CA", "NV", "AZ", "UT", "ID", "MT", "WY", "CO", "NM", "AK", "HI"]},
          **{s: "South" for s in ["TX", "OK", "AR", "LA", "MS", "AL", "TN", "KY", "GA", "FL", "SC", "NC", "VA", "WV", "MD", "DE", "DC"]},
          **{s: "Midwest" for s in ["ND", "SD", "NE", "KS", "MN", "IA", "MO", "WI", "IL", "IN", "MI", "OH"]},
          **{s: "Northeast" for s in ["PA", "NY", "NJ", "CT", "RI", "MA", "VT", "NH", "ME"]}}


def load(cfg: dict) -> tuple[pd.DataFrame, dict]:
    processed = Path(cfg["data"]["processed_dir"])
    frame = pd.read_parquet(processed / "model_frame.parquet")
    pred = pd.read_parquet(processed / "predictions.parquet")
    df = frame.merge(pred[["loan_id"] + [f"p_{m}" for m in MODELS]], on="loan_id", how="inner")
    models = {"champion": lgb.Booster(model_file="outputs/models/champion.txt")}
    with open("outputs/models/challenger.pkl", "rb") as f:
        models["challenger"] = pickle.load(f)
    with open("outputs/models/baseline.pkl", "rb") as f:
        models["baseline"] = pickle.load(f)
    return df, models


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-fairness", action="store_true")
    ap.add_argument("--fast", action="store_true", help="fewer bootstrap reps")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    th = cfg["thresholds"]; seed = int(cfg["seed"])
    reps = 200 if args.fast else int(th["bootstrap_reps"])
    num, cat = cfg["features"]["numeric"], cfg["features"]["categorical"]
    cols = num + cat
    out = Path("outputs/validation"); out.mkdir(parents=True, exist_ok=True)
    df, models = load(cfg)
    reg = Register()
    R = {}  # everything we save

    roles = ["train", "tune", "oot", "stress", "covid_exhibit"]
    by_role = {r: df[df["role"] == r] for r in roles}
    oot = by_role["oot"]; train = by_role["train"]

    # ------------------------------------------------------------------ 7.2 discrimination
    print("[discrimination]")
    rows = []
    for m in MODELS:
        for r in roles:
            d = by_role[r]
            pm = disc.point_metrics(d["y"].to_numpy(), d[f"p_{m}"].to_numpy())
            b = disc.bootstrap_auc(d["y"].to_numpy(), d[f"p_{m}"].to_numpy(), reps=reps, seed=seed)
            rows.append({"model": m, "role": r, **pm, "auc_ci_low": b["ci_low"], "auc_ci_high": b["ci_high"], "auc_se": b["se"]})
    R["discrimination_by_role"] = pd.DataFrame(rows)
    print(R["discrimination_by_role"].pivot(index="role", columns="model", values="auc").round(4).to_string())

    # pairwise DeLong on OOT and on stress
    dl = []
    for r in ("tune", "oot", "stress"):
        d = by_role[r]
        for a, b_ in (("champion", "challenger"), ("champion", "baseline"), ("challenger", "baseline")):
            t = disc.delong_test(d["y"].to_numpy(), d[f"p_{a}"].to_numpy(), d[f"p_{b_}"].to_numpy())
            dl.append({"role": r, "model_1": a, "model_2": b_, **t})
    R["delong"] = pd.DataFrame(dl)
    print(R["delong"][["role", "model_1", "model_2", "diff", "p_value"]].round(4).to_string(index=False))

    # segments on OOT for champion + challenger
    seg_rows = []
    for m in ("champion", "challenger"):
        for name, fn in SEGMENTS.items():
            s = disc.by_segment(oot["y"].to_numpy(), oot[f"p_{m}"].to_numpy(), pd.Series(fn(oot), index=oot.index),
                                min_defaults=30, reps=max(100, reps // 5), seed=seed)
            s.insert(0, "dimension", name); s.insert(0, "model", m)
            seg_rows.append(s)
    R["segments_oot"] = pd.concat(seg_rows, ignore_index=True)
    R["deciles_oot"] = pd.concat([disc.decile_table(oot["y"].to_numpy(), oot[f"p_{m}"].to_numpy()).assign(model=m) for m in MODELS])

    # ------------------------------------------------------------------ 7.3 calibration
    print("[calibration]")
    crow, rel = [], []
    for m in MODELS:
        for r in roles:
            d = by_role[r]
            s = cal.summary(d["y"].to_numpy(), d[f"p_{m}"].to_numpy())
            crow.append({"model": m, "role": r, **s})
            t = cal.reliability_table(d["y"].to_numpy(), d[f"p_{m}"].to_numpy(), bins=10)
            t.insert(0, "role", r); t.insert(0, "model", m); rel.append(t)
        for v, d in df.groupby("vintage"):
            s = cal.summary(d["y"].to_numpy(), d[f"p_{m}"].to_numpy())
            crow.append({"model": m, "role": f"vintage_{v}", **s})
    R["calibration"] = pd.DataFrame(crow)
    R["reliability"] = pd.concat(rel, ignore_index=True)
    show = R["calibration"][R["calibration"]["role"].isin(roles)]
    print(show.pivot(index="role", columns="model", values="citl_ratio_obs_pred").round(3).to_string())

    # ------------------------------------------------------------------ 7.4 stability
    print("[stability]")
    srows = []
    ref_scores = {m: train[f"p_{m}"].to_numpy() for m in MODELS}
    for v, d in df.groupby("vintage"):
        row = {"vintage": v, "role": d["role"].iloc[0], "n": len(d)}
        for m in MODELS:
            row[f"psi_{m}"] = stab.psi(ref_scores[m], d[f"p_{m}"].to_numpy())["psi"]
            row[f"auc_{m}"] = disc.point_metrics(d["y"].to_numpy(), d[f"p_{m}"].to_numpy())["auc"]
        row["obs_rate"] = float(d["y"].mean()); row["mean_pred_champion"] = float(d["p_champion"].mean())
        srows.append(row)
    R["stability_by_vintage"] = pd.DataFrame(srows)
    csi = []
    for v, d in df.groupby("vintage"):
        t = stab.csi_table(train, d, num, cat); t.insert(0, "vintage", v); csi.append(t)
    R["csi_by_vintage"] = pd.concat(csi, ignore_index=True)
    print(R["stability_by_vintage"][["vintage", "role", "psi_champion", "auc_champion", "obs_rate", "mean_pred_champion"]].round(4).to_string(index=False))

    # ------------------------------------------------------------------ 7.6 sensitivity (champion)
    print("[sensitivity]")
    booster = models["champion"]
    def predict_champ(X: pd.DataFrame) -> np.ndarray:
        return booster.predict(X[cols], num_iteration=booster.best_iteration)
    X_oot = oot[cols]
    R["perm_importance_oot"] = sens.permutation_importance(predict_champ, X_oot, oot["y"].to_numpy(), cols, reps=3, seed=seed)
    shap_tab, shap_vals, shap_X = sens.shap_summary(booster, oot, cols, sample=20000, seed=seed)
    R["shap_importance_oot"] = shap_tab
    np.save(out / "shap_values_oot.npy", shap_vals); shap_X.to_parquet(out / "shap_X_oot.parquet", index=False)
    grids = {"fico": np.arange(620, 821, 20), "ltv": np.arange(40, 101, 5), "dti": np.arange(10, 56, 5),
             "orig_rate": np.arange(2.5, 7.1, 0.5), "orig_upb": np.arange(50e3, 800e3, 50e3)}
    pdp = []
    for c, g in grids.items():
        t = sens.partial_dependence(predict_champ, X_oot, c, g, seed=seed); t.insert(0, "variable", c); pdp.append(t)
    R["pdp_oot"] = pd.concat(pdp, ignore_index=True)
    R["oat_oot"] = sens.one_at_a_time(predict_champ, X_oot, {"fico": [-20, -50], "ltv": [5, 10], "dti": [5, 10], "orig_rate": [0.5, 1.0]})
    mono = cfg["features"]["monotone"]
    R["monotonicity"] = pd.DataFrame([sens.monotonicity_check(predict_champ, X_oot, c, d, grids[c], seed=seed) for c, d in mono.items() if c in grids])
    # same hyperparameters under each LightGBM constraint method
    card = json.loads(Path("outputs/model_card.json").read_text())
    dtrain = lgb_dataset(train, train["y"].to_numpy(), cols, cat)
    mrows = []
    for method in ("basic", "intermediate", "advanced"):
        b = lgb.train({**card["champion"]["params"], "monotone_constraints_method": method}, dtrain,
                      num_boost_round=card["champion"]["best_iteration"])
        for col, direction in mono.items():
            if col in grids:
                r = sens.monotonicity_check(lambda X, b=b: b.predict(X[cols]), X_oot, col, direction, grids[col], seed=seed)
                mrows.append({"method": method, **r})
    R["monotone_methods"] = pd.DataFrame(mrows)
    print(R["perm_importance_oot"].head(6).round(4).to_string(index=False))
    print(R["oat_oot"].round(4).to_string(index=False))

    # ------------------------------------------------------------------ 7.8 fairness proxy
    if not args.skip_fairness:
        print("[fairness proxy]")
        try:
            acs = fair.fetch_acs_cbsa(Path("data/processed/acs_cbsa_b03002.parquet"))
            grp = fair.assign_groups(oot["msa"], acs, n_groups=3)
            grp.index = oot.index
            cutoff = float(np.quantile(oot["p_champion"], 0.90))  # illustrative policy: decline riskiest 10%
            frows = []
            for m in ("champion", "challenger"):
                t = fair.group_metrics(oot["y"].to_numpy(), oot[f"p_{m}"].to_numpy(), grp["group"], cutoff=cutoff, reference="T1")
                t.insert(0, "model", m); frows.append(t)
            R["fairness_groups_oot"] = pd.concat(frows, ignore_index=True)
            R["fairness_proxy_audit"] = fair.proxy_feature_audit(oot, grp["area_share_nonwhite"], num, cat)
            R["fairness_meta"] = pd.DataFrame([{"cutoff_p": cutoff, "matched_share": float(grp["area_share_nonwhite"].notna().mean()),
                                                "groups": "T1 = lowest tertile of area non-white share (reference), T3 = highest"}])
            print(R["fairness_groups_oot"].round(4).to_string(index=False))
        except (requests.RequestException, RuntimeError, KeyError, ValueError, OSError) as e:  # must not kill the battery
            print(f"  fairness proxy skipped: {e!r}")
            reg.add("Fairness", "Low", "Fairness proxy analysis could not be executed",
                    "Census ACS demographic shares unavailable at build time (API key required).", "n/a",
                    "Re-run with CENSUS_API_KEY set; treat the fairness section as incomplete until then.")

    # ------------------------------------------------------------------ findings against thresholds
    d = R["discrimination_by_role"].set_index(["model", "role"])
    c = R["calibration"].set_index(["model", "role"])
    auc_oot = d.loc[("champion", "oot"), "auc"]; auc_tr = d.loc[("champion", "train"), "auc"]; auc_tu = d.loc[("champion", "tune"), "auc"]
    if auc_oot < th["auc_oot_min"]:
        reg.add("Discrimination", "High", "Champion OOT AUC below floor", f"AUC={auc_oot:.3f}", f">= {th['auc_oot_min']}", "Redevelop.")
    gap = auc_tr - auc_oot
    if gap > th["auc_train_oot_gap_max"]:
        reg.add("Discrimination", "Medium", "Champion train-to-OOT AUC decay exceeds tolerance",
                f"train {auc_tr:.3f} -> tune {auc_tu:.3f} -> OOT {auc_oot:.3f} (gap {gap:.3f})", f"<= {th['auc_train_oot_gap_max']}",
                "Part in-sample optimism (train-tune gap), part regime drift (tune-OOT). Tighten regularisation; monitor AUC quarterly; "
                "re-estimate on rolling window.", ["discrimination_by_role"])
    seg = R["segments_oot"]
    weak = seg[(seg["model"] == "champion") & (seg["defaults"] >= th["segment_min_defaults"]) & (seg["auc"] < th["segment_auc_min"])]
    for _, s in weak.iterrows():
        reg.add("Discrimination", "Medium", f"Weak rank-ordering in segment {s['dimension']}={s['segment']}",
                f"AUC {s['auc']:.3f} (n={s['n']:,}, defaults={s['defaults']})", f">= {th['segment_auc_min']}", "Investigate segment-specific drivers; consider segment overlay.")
    for r in ("oot", "stress", "covid_exhibit"):
        rel = c.loc[("champion", r), "citl_rel_error"]; ratio = c.loc[("champion", r), "citl_ratio_obs_pred"]
        if abs(rel) > th["calib_in_large_rel_tol"]:
            sev = "High" if r == "oot" else "Medium"
            direction = "under-predicts" if ratio > 1 else "over-predicts"
            if r == "stress":
                rec = ("Model has no macro inputs and never saw a downturn; PD level is point-in-time for the training regime only. "
                       "Do not use for capital/pricing without a macro overlay or recalibration; add crisis-vintage stress to monitoring.")
            elif r == "covid_exhibit" and ratio < 1:
                rec = ("Observed defaults were suppressed by pandemic forbearance, deferral and disaster relief, which the hardship-guarded "
                       "target removes from the numerator; 2019 is not a valid calibration test. Treat forbearance-heavy periods as a "
                       "data-relevance trigger and suspend calibration monitoring on affected cohorts.")
            else:
                rec = "Recalibrate intercept on recent vintages; monitor calibration-in-the-large quarterly."
            reg.add("Calibration", sev, f"Calibration-in-the-large breach on {r} (model {direction})",
                    f"predicted {c.loc[('champion', r), 'citl_predicted']:.2%} vs observed {c.loc[('champion', r), 'citl_observed']:.2%} (obs/pred = {ratio:.2f}x)",
                    f"|rel error| <= {th['calib_in_large_rel_tol']:.0%}", rec, ["calibration"])
    oot_vintages = sorted(oot["vintage"].unique())
    for v in oot_vintages:
        rel = c.loc[("champion", f"vintage_{v}"), "citl_rel_error"]; ratio = c.loc[("champion", f"vintage_{v}"), "citl_ratio_obs_pred"]
        if abs(rel) > th["calib_in_large_rel_tol"]:
            direction = "under-predicts" if ratio > 1 else "over-predicts"
            reg.add("Calibration", "Medium", f"Calibration-in-the-large breach on vintage {v} (model {direction})",
                    f"predicted {c.loc[('champion', f'vintage_{v}'), 'citl_predicted']:.2%} vs observed {c.loc[('champion', f'vintage_{v}'), 'citl_observed']:.2%} (obs/pred = {ratio:.2f}x)",
                    f"|rel error| <= {th['calib_in_large_rel_tol']:.0%}", "The pooled OOT ratio hides offsetting vintage errors; monitor by vintage.", ["calibration"])
    lo, hi = th["calib_slope_range"]
    bad = {r: c.loc[("champion", r), "slope"] for r in ("train", "tune", "oot") if not lo <= c.loc[("champion", r), "slope"] <= hi}
    if bad:
        rec = ("Slope > 1: predictions are too compressed (early stopping at few trees). Recalibrate on the tune vintage."
               if np.mean(list(bad.values())) > 1 else "Slope < 1: predictions are too extreme (over-fit). Add shrinkage or recalibrate.")
        reg.add("Calibration", "Medium", "Calibration slope outside range", ", ".join(f"{r} {v:.2f}" for r, v in bad.items()),
                f"[{lo}, {hi}]", rec, ["calibration"])
    st = R["stability_by_vintage"]
    for _, s in st.iterrows():
        if s["role"] in ("oot", "covid_exhibit") and s["psi_champion"] >= th["psi_red"]:
            reg.add("Stability", "Medium", f"Score PSI red on vintage {s['vintage']}", f"PSI={s['psi_champion']:.3f}", f"< {th['psi_red']}",
                    "Population shift vs development sample; confirm inputs (CSI) and recalibrate if persistent.", ["stability_by_vintage", "csi_by_vintage"])
    stress_psi = st[st["role"] == "stress"]["psi_champion"]
    if (stress_psi >= th["psi_red"]).any():
        reg.add("Stability", "Low", "Crisis vintages are outside the development population",
                f"PSI 2006–08 = {', '.join(f'{v:.2f}' for v in stress_psi)}", f"< {th['psi_red']}",
                "Expected by design of the stress backtest; documents that the model would be extrapolating in a 2006-style book.",
                ["stability_by_vintage", "csi_by_vintage"])
    viol = R["monotonicity"][R["monotonicity"]["violation_rate"] > 0]
    for _, s in viol.iterrows():
        reg.add("Sensitivity", "Low", f"Monotonicity violations for {s['variable']}", f"{s['violation_rate']:.1%} of sampled loans", "0%", "Check constraint configuration.")
    if "fairness_groups_oot" in R:
        f = R["fairness_groups_oot"]; f = f[f["model"] == "champion"].set_index("group")
        for g in ("T2", "T3"):
            if g in f.index and f.loc[g, "adverse_impact_ratio"] < th["air_min"]:
                reg.add("Fairness", "Medium", f"Adverse impact ratio below 4/5ths for proxy group {g}",
                        f"AIR={f.loc[g, 'adverse_impact_ratio']:.2f} at illustrative decline cutoff", f">= {th['air_min']}",
                        "Proxy-only screen; escalate to a fair-lending review with BISG/HMDA data before any policy use.", ["fairness_groups_oot"])
            if g in f.index and abs(f.loc[g, "pred_over_obs"] - 1) > th["calib_in_large_rel_tol"]:
                reg.add("Fairness", "Low", f"Within-group calibration drift for proxy group {g}",
                        f"pred/obs = {f.loc[g, 'pred_over_obs']:.2f}", f"within {th['calib_in_large_rel_tol']:.0%}", "Monitor; consider group-level recalibration check.")
    mm = R["monotone_methods"].pivot(index="variable", columns="method", values="violation_rate")
    used = card["champion"]["params"]["monotone_constraints_method"]
    if mm["advanced"].max() > 0 and mm[used].max() == 0:
        bad = mm["advanced"][mm["advanced"] > 0].sort_values(ascending=False)
        reg.add("Sensitivity", "High", "Monotone constraints not enforced with LightGBM's 'advanced' method",
                f"Same hyperparameters: {', '.join(f'{v} {r:.1%}' for v, r in bad.items())} of sampled loans move against the declared "
                f"direction with 'advanced'; '{used}' gives 0%. See LightGBM issue #5358.",
                "0% violations on the test grid", f"Champion uses '{used}'.", ["monotone_methods"], status="Remediated")
    iv = {r["name"]: r["iv"] for r in card["challenger"]["iv"]}
    perm_rank = R["perm_importance_oot"].sort_values("auc_drop", ascending=False)["variable"].tolist().index("orig_rate") + 1
    rate = train.groupby("vintage")["orig_rate"].mean()
    reg.add("Conceptual soundness", "Medium", "Original interest rate is confounded with calendar time",
            f"orig_rate: scorecard IV {iv['orig_rate']:.2f}, #{perm_rank} in OOT permutation importance; mean rate by training vintage "
            f"ranges {rate.min():.2f}-{rate.max():.2f}%.", "n/a",
            "Replace with the spread to the month's PMMS 30-year rate (FRED MORTGAGE30US) at redevelopment.", ["perm_importance_oot"])
    fp = {d["vintage"]: d for d in json.loads(Path("outputs/firstpass.json").read_text())}
    guard = [(v, fp[v]["default_rate_36m"], fp[v]["default_rate_36m_unguarded"]) for v in sorted(fp)
             if v >= 2017 and fp[v].get("default_rate_36m_unguarded") is not None]
    if guard:
        reg.add("Data", "Medium", "Target definition depends on servicer-reported hardship flags",
                "36-month default rate with / without the forbearance and disaster guard: "
                + ", ".join(f"{v} {g:.2%} / {u:.2%}" for v, g, u in guard) + ".", "n/a",
                "Keep both definitions in monitoring; treat forbearance-heavy periods as a data-relevance trigger.", ["firstpass"])
    train_v = sorted(train["vintage"].unique())
    harp = [fp[v]["harp_loans"] / fp[v]["loans"] for v in train_v if v in fp]
    reg.add("Data", "Low", "Development sample excludes HARP loans and is the 50k-loan public sample",
            f"HARP is {min(harp):.0%}-{max(harp):.0%} of {train_v[0]}-{train_v[-1]} originations. Train has {int(train['y'].sum()):,} defaults "
            f"across {len(train):,} loans after exclusion.", "n/a",
            "State scope in intended use; use the Standard dataset for segment-level recalibration if the model is adopted.", ["feature_waterfall"])

    # ------------------------------------------------------------------ save
    for k, v in R.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"{k}.csv", index=False)
    reg.to_frame().to_csv("outputs/findings.csv", index=False)
    (out / "summary.json").write_text(json.dumps({
        "findings": reg.counts(),
        "champion_auc": {r: float(d.loc[("champion", r), "auc"]) for r in roles},
        "champion_citl_ratio": {r: float(c.loc[("champion", r), "citl_ratio_obs_pred"]) for r in roles},
    }, indent=2))
    print("\nfindings:", reg.counts())
    print(reg.to_frame()[["id", "area", "severity", "title"]].to_string(index=False))


if __name__ == "__main__":
    main()
