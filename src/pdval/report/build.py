"""Build the validation report and executive summary (Markdown -> PDF via pandoc + tectonic).

Every number in the prose is read from outputs/ so a re-run of the battery re-renders the report.

Usage:
    uv run python -m pdval.report.build [--no-pdf]
Writes report/report.md, report/executive_summary.md and the PDFs next to them.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

V = Path("outputs/validation")
R = Path("report")


def load():
    d = {}
    d["cfg"] = yaml.safe_load(Path("configs/default.yaml").read_text())
    d["disc"] = pd.read_csv(V / "discrimination_by_role.csv").set_index(["model", "role"])
    d["delong"] = pd.read_csv(V / "delong.csv")
    d["cal"] = pd.read_csv(V / "calibration.csv").set_index(["model", "role"])
    d["seg"] = pd.read_csv(V / "segments_oot.csv")
    d["dec"] = pd.read_csv(V / "deciles_oot.csv")
    d["stab"] = pd.read_csv(V / "stability_by_vintage.csv").set_index("vintage")
    d["csi"] = pd.read_csv(V / "csi_by_vintage.csv")
    d["perm"] = pd.read_csv(V / "perm_importance_oot.csv")
    d["shap"] = pd.read_csv(V / "shap_importance_oot.csv")
    d["oat"] = pd.read_csv(V / "oat_oot.csv")
    d["mono"] = pd.read_csv(V / "monotonicity.csv")
    d["rel"] = pd.read_csv(V / "reliability.csv")
    d["mm"] = pd.read_csv(V / "monotone_methods.csv") if (V / "monotone_methods.csv").exists() else None
    d["find"] = pd.read_csv("outputs/findings.csv")
    d["card"] = json.loads(Path("outputs/model_card.json").read_text())
    d["wf"] = pd.DataFrame(json.loads(Path("outputs/feature_waterfall.json").read_text()))
    d["fp"] = {s["vintage"]: s for s in json.loads(Path("outputs/firstpass.json").read_text())}
    d["manifest"] = json.loads(Path("data/manifest.json").read_text())
    d["ingest"] = json.loads(Path("outputs/ingest_log.json").read_text())
    for f in ("fairness_groups_oot", "fairness_proxy_audit", "fairness_meta"):
        p = V / f"{f}.csv"
        d[f] = pd.read_csv(p) if p.exists() else None
    return d


def pct(x, nd=2):
    return f"{100 * x:.{nd}f}%"


ROLE_LABEL = {"train": "train 2009–15", "tune": "tune 2016", "oot": "OOT 2017–18", "stress": "stress 2006–08",
              "covid_exhibit": "COVID 2019"}


def tex_symbols(md: str) -> str:
    # pandoc does not close $...$ when a digit follows, so keep a space after the symbol
    return re.sub(r"[≥≤]\s*", lambda m: ("$\\geq$ " if m.group(0)[0] == "≥" else "$\\leq$ "), md)


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def span(vals, fmt) -> str:
    lo, hi = min(vals), max(vals)
    return f"{lo:{fmt}}" if f"{lo:{fmt}}" == f"{hi:{fmt}}" else f"{lo:{fmt}}–{hi:{fmt}}"


def md_table(df: pd.DataFrame, floatfmt: str = ".3f", widths: list[int] | None = None) -> str:
    """Pipe table. `widths` = relative column widths (pandoc uses the dash counts when a row is wider than the page)."""
    cols = list(df.columns)
    dashes = ["-" * max(3, w) for w in widths] if widths else ["---" for _ in cols]
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(dashes) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:{floatfmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_report(d: dict) -> str:
    cfg, disc, cal, stab = d["cfg"], d["disc"], d["cal"], d["stab"]
    th = cfg["thresholds"]; tc = cfg["target"]; vin = cfg["vintages"]
    C = lambda r, k: disc.loc[("champion", r), k]
    L = lambda r, k: disc.loc[("challenger", r), k]
    B = lambda r, k: disc.loc[("baseline", r), k]
    K = lambda r, k: cal.loc[("champion", r), k]
    find = d["find"]
    counts = find["severity"].value_counts().to_dict()
    open_high = find[(find["severity"] == "High") & (find["status"] != "Remediated")]
    n_train = int(d["wf"].query("role=='train'")["after_amortization_filter"].sum())
    def_train = int(d["wf"].query("role=='train'")["defaults"].sum())
    n_oot = int(d["wf"].query("role=='oot'")["after_amortization_filter"].sum())
    def_oot = int(d["wf"].query("role=='oot'")["defaults"].sum())
    n_stress = int(d["wf"].query("role=='stress'")["after_amortization_filter"].sum())
    def_stress = int(d["wf"].query("role=='stress'")["defaults"].sum())
    dl_oot = d["delong"].query("role=='oot' and model_1=='champion' and model_2=='challenger'").iloc[0]
    seg_c = d["seg"][(d["seg"]["model"] == "champion") & (d["seg"]["defaults"] >= th["segment_min_defaults"])]
    weak = seg_c.sort_values("auc").head(3)
    top1 = d["dec"].query("model=='champion' and decile==1").iloc[0]
    top2 = d["dec"].query("model=='champion' and decile==2").iloc[0]
    stress_ratio = K("stress", "citl_ratio_obs_pred"); oot_ratio = K("oot", "citl_ratio_obs_pred")
    covid_ratio = K("covid_exhibit", "citl_ratio_obs_pred")
    psi18 = stab.loc[2018, "psi_champion"]; psi17 = stab.loc[2017, "psi_champion"]
    csi18 = d["csi"][d["csi"]["vintage"] == 2018].sort_values("csi", ascending=False).head(3)
    perm = d["perm"]
    mono_max = d["mono"]["violation_rate"].max()
    champ = d["card"]["champion"]["params"]
    iv = pd.DataFrame(d["card"]["challenger"]["iv"]).sort_values("iv", ascending=False)
    fp = d["fp"]
    rating = "Fit for intended use, with conditions" if len(open_high) == 0 and C("oot", "auc") >= th["auc_oot_min"] else "Not fit without remediation"
    tol = th["calib_in_large_rel_tol"]
    v_ratio = ", ".join(f"{v} {K(f'vintage_{v}', 'citl_ratio_obs_pred'):.2f}" for v in vin["oot"])
    ro = d["rel"].query("model=='champion' and role=='oot'")
    miss = int(((ro["mean_pred"] < ro["obs_ci_low"]) | (ro["mean_pred"] > ro["obs_ci_high"])).sum())
    breach = [ROLE_LABEL[r] for r in ROLE_LABEL if abs(K(r, "citl_rel_error")) > tol]
    band = lambda x: "red" if x >= th["psi_red"] else "amber" if x >= th["psi_amber"] else "green"
    dev_v = [v for v in stab.index if stab.loc[v, "role"] in ("train", "tune")]
    not_green = [f"{v} {band(stab.loc[v, 'psi_champion'])} ({stab.loc[v, 'psi_champion']:.2f})" for v in dev_v if band(stab.loc[v, "psi_champion"]) != "green"]
    dev_psi = "green for every training and tune vintage" if not not_green else "green for training and tune vintages except " + ", ".join(not_green)
    over = [str(v) for v in stab.index if stab.loc[v, "obs_rate"] < stab.loc[v, "mean_pred_champion"] * (1 - tol)]
    guard = [(v, fp[v]["default_rate_36m"], fp[v]["default_rate_36m_unguarded"]) for v in sorted(fp) if v >= 2017 and fp[v].get("default_rate_36m_unguarded") is not None]
    guard_years = ", ".join(str(v) for v, _, _ in guard)
    ung_txt = ", ".join(pct(u) for _, _, u in guard); g_txt = ", ".join(pct(g) for _, g, _ in guard)
    train_v = vin["train"]
    med_rate = [fp[v]["quantiles"]["orig_rate"]["0.5"] for v in train_v]
    perm_rank = perm["variable"].tolist().index("orig_rate") + 1
    harp = [fp[v]["harp_loans"] / fp[v]["loans"] for v in train_v if fp[v]["harp_loans"]]
    na = lambda k: [fp[v]["na_rates"][k] for v in fp]
    def_train_v = d["wf"].query("role=='train'")["defaults"]
    pre_v = [v for v in sorted(fp) if not fp[v]["harp_loans"]][:3]
    dti_pre = [fp[v]["na_rates"]["dti"] for v in pre_v]
    harp_v = [v for v in sorted(fp) if fp[v]["harp_loans"]]
    dti_harp = ("; from 2009 it is missing only on HARP loans" if all(abs(fp[v]["na_rates"]["dti"] - fp[v]["harp_loans"] / fp[v]["loans"]) < 0.002 for v in harp_v) else "")
    dl_base = d["delong"].query("role=='oot' and model_2=='baseline'")
    bench = [C(r, "auc") - L(r, "auc") for r in ROLE_LABEL]
    shap_same = set(d["shap"]["variable"].head(4)) == set(perm["variable"].head(4))
    weak_breach = weak.iloc[0]["auc"] < th["segment_auc_min"]
    mono_txt, mono_bullet = "", ""
    if d["mm"] is not None:
        mm = d["mm"].pivot(index="variable", columns="method", values="violation_rate")
        adv = mm["advanced"][mm["advanced"] > 0].sort_values(ascending=False)
        t_mm = (mm[["basic", "intermediate", "advanced"]] * 100).round(1).reset_index()
        t_mm.columns = ["input", "basic (%)", "intermediate (%)", "advanced (%)"]
        mono_txt = f"""**Monotone constraints.** The champion's hyperparameters were refit with each LightGBM constraint method, and each input was moved across its grid for 5,000 OOT loans. Share of loans whose prediction moves against the declared direction:

{md_table(t_mm, ".1f")}
"""
        if len(adv):
            mono_bullet = (f"- **Monotone constraints.** With LightGBM's `advanced` method, predictions move against the declared direction for "
                           f"{', '.join(f'{r:.1%} of loans on {v}' for v, r in adv.items())} (LightGBM issue #5358). The champion uses `intermediate`.")

    # helper tables
    t_disc = disc.reset_index()
    t_disc = t_disc[t_disc["role"].isin(["train", "tune", "oot", "stress", "covid_exhibit"])]
    t_disc["AUC (95% CI)"] = t_disc.apply(lambda r: f"{r['auc']:.3f} ({r['auc_ci_low']:.3f}–{r['auc_ci_high']:.3f})", axis=1)
    t_disc_p = t_disc.pivot(index="role", columns="model", values="AUC (95% CI)").loc[["train", "tune", "oot", "stress", "covid_exhibit"]]
    t_disc_p.insert(0, "n / defaults", [f"{int(C(r, 'n')):,} / {int(C(r, 'defaults')):,}" for r in t_disc_p.index])
    t_disc_p = t_disc_p.reset_index().rename(columns={"role": "sample"}); t_disc_p["sample"] = t_disc_p["sample"].map(ROLE_LABEL)

    t_cal = cal.reset_index()
    t_cal = t_cal[(t_cal["model"] == "champion") & t_cal["role"].isin(["train", "tune", "oot", "stress", "covid_exhibit"])]
    t_cal = t_cal[["role", "citl_observed", "citl_predicted", "citl_ratio_obs_pred", "intercept", "slope", "slope_se", "brier_skill", "ece"]].copy()
    t_cal["citl_observed"] = t_cal["citl_observed"].map(pct); t_cal["citl_predicted"] = t_cal["citl_predicted"].map(pct)
    t_cal.columns = ["sample", "observed", "predicted", "obs/pred", "intercept", "slope", "slope SE", "Brier skill", "ECE"]
    t_cal["sample"] = t_cal["sample"].map(ROLE_LABEL)

    t_stab = stab.reset_index()[["vintage", "role", "n", "psi_champion", "auc_champion", "obs_rate", "mean_pred_champion"]].copy()
    t_stab["obs_rate"] = t_stab["obs_rate"].map(pct); t_stab["mean_pred_champion"] = t_stab["mean_pred_champion"].map(pct)
    t_stab.columns = ["vintage", "role", "n", "score PSI", "AUC", "observed", "predicted"]

    t_seg = seg_c[["dimension", "segment", "n", "defaults", "obs_rate", "mean_pred", "auc", "auc_ci_low", "auc_ci_high"]].copy()
    t_seg["obs_rate"] = t_seg["obs_rate"].map(pct); t_seg["mean_pred"] = t_seg["mean_pred"].map(pct)
    t_seg["AUC (95% CI)"] = t_seg.apply(lambda r: f"{r['auc']:.3f} ({r['auc_ci_low']:.3f}–{r['auc_ci_high']:.3f})", axis=1)
    t_seg = t_seg[["dimension", "segment", "n", "defaults", "obs_rate", "mean_pred", "AUC (95% CI)"]]

    t_find = find[["id", "severity", "status", "title", "evidence", "recommendation"]].copy()
    t_find["severity"] = t_find["severity"] + " / " + t_find["status"]; t_find = t_find.drop(columns="status")
    t_find.columns = ["ID", "Severity / status", "Finding", "Evidence", "Recommendation"]

    t_wf = d["wf"][["vintage", "role", "loans", "observable", "after_harp_exclusion", "defaults"]].copy()
    t_wf["36m rate"] = (t_wf["defaults"] / t_wf["after_harp_exclusion"]).map(pct)
    t_wf.columns = ["vintage", "role", "sample loans", "observable", "after HARP excl.", "defaults", "36m rate"]

    t_oat = d["oat"].copy(); t_oat["shock"] = t_oat["shock"].map(lambda x: f"{x:+g}"); t_oat["mean_pred"] = t_oat["mean_pred"].map(pct); t_oat["rel_change"] = t_oat["rel_change"].map(lambda x: f"{100*x:+.0f}%")
    t_oat.columns = ["input", "shock", "portfolio mean PD", "change"]

    fair_section = ""
    if d["fairness_groups_oot"] is not None:
        fg = d["fairness_groups_oot"]; fg = fg[fg["model"] == "champion"]
        meta = d["fairness_meta"].iloc[0]
        t_fair = fg[["group", "n", "defaults", "obs_rate", "mean_pred", "pred_over_obs", "auc", "approval_rate", "adverse_impact_ratio", "std_mean_diff"]].copy()
        t_fair["obs_rate"] = t_fair["obs_rate"].map(pct); t_fair["mean_pred"] = t_fair["mean_pred"].map(pct); t_fair["approval_rate"] = t_fair["approval_rate"].map(lambda x: pct(x, 1))
        t_fair.columns = ["proxy group", "n", "defaults", "observed", "predicted", "pred/obs", "AUC", "pass rate", "AIR", "SMD"]
        aud = d["fairness_proxy_audit"].head(6)
        fair_section = f"""
Groups are tertiles of the loan's metro area non-white population share (ACS 2022 5-year, table B03002, CBSA or metropolitan division), with non-metro loans as a separate group; {meta['matched_share']:.0%} of OOT loans matched an ACS geography. T1 (lowest share) is the reference. The illustrative policy cutoff declines the riskiest 10% of the OOT book (PD > {meta['cutoff_p']:.2%}).

{md_table(t_fair)}

**Proxy-feature audit.** Association between each model input and the area demographic share (Spearman for numerics, correlation ratio for categoricals):

{md_table(aud, ".3f")}

**Reading.** Lowest adverse impact ratio {fg['adverse_impact_ratio'].min():.2f} ({'passes' if fg['adverse_impact_ratio'].min() >= th['air_min'] else 'fails'} the 4/5ths screen). Standardised mean score difference, T3 vs T1: {fg.set_index('group').loc['T3','std_mean_diff']:.2f} SD. Within-group AUC {fg['auc'].min():.2f}–{fg['auc'].max():.2f}. Predicted/observed by group: {', '.join(f"{g} {r:.2f}" for g, r in fg.set_index('group')['pred_over_obs'].items())}. The input most associated with area demographics is {aud.iloc[0]['variable']} ({aud.iloc[0]['assoc']:.2f}).

The adverse impact ratio is compared with the 4/5ths rule as a *screen*, and within-group calibration (pred/obs) is the fair-lending-relevant quantity for a pricing or limit model: a group that is systematically under- or over-predicted is being mis-priced. This is a geography-only proxy in the lineage of the CFPB's 2014 BISG methodology, without the surname component that carries most of BISG's power; it cannot support a compliance conclusion. A bank would run this with applicant-level protected-class data (HMDA for mortgages) and a documented proxy method for other products.
"""
    else:
        fair_section = """
**Not executed in this version.** The geography proxy requires ACS demographic shares from the Census API, which now requires a (free) API key that was not available at build time. The code path (`pdval.validation.fairness`) is complete and unit-covered: it maps each loan's MSA / metropolitan-division code to the area's non-white population share (ACS 2022 5-year, B03002), forms tertile groups with non-metro as a separate group, and reports adverse impact ratio at an illustrative decline cutoff, within-group calibration, AUC parity, standardised mean difference of scores, and a proxy-feature audit. The section is recorded as a Low finding.
"""

    stress_years = ", ".join(str(v) for v in vin["stress"])
    md = f"""---
title: "Validation of a Mortgage Probability-of-Default Model"
subtitle: "Champion (gradient-boosted trees) vs challenger (WoE scorecard), Freddie Mac single-family loan-level data"
author: "Qavi Inan"
date: "{datetime.now(tz=UTC):%B %d, %Y}"
geometry: margin=0.9in
fontsize: 10pt
colorlinks: true
toc: false
numbersections: true
header-includes:
  - \\usepackage{{booktabs}}
  - \\usepackage{{longtable}}
  - \\usepackage{{float}}
  - \\floatplacement{{figure}}{{H}}
  - \\usepackage{{caption}}
  - \\captionsetup{{font=small}}
---

# Executive summary {{-}}

**Model.** A 36-month probability-of-default (PD) model for conforming, fixed-rate, full-documentation mortgages at origination, built on Freddie Mac's public single-family loan-level *sample* dataset (50,000 loans per vintage). Inputs are sixteen origination-time borrower and loan attributes (§4); geography, seller and vintage are excluded by design. A LightGBM champion with monotone constraints is compared with a weight-of-evidence logistic scorecard challenger and a FICO-only baseline.

**Intended use.** Rank-ordering of new originations for acquisition screening and risk-based review, and point-in-time PD estimation within the post-2009 underwriting regime. Not intended for capital, stress-testing or downturn pricing without a macro overlay.

**Overall rating: {rating}.**

**Headline evidence.**

- **Discrimination holds out of time.** Champion AUC {C('oot','auc'):.3f} (95% CI {C('oot','auc_ci_low'):.3f}–{C('oot','auc_ci_high'):.3f}) on {n_oot:,} loans / {def_oot:,} defaults from the {vin['oot'][0]}–{vin['oot'][-1]} vintages, against {L('oot','auc'):.3f} for the scorecard and {B('oot','auc'):.3f} for FICO alone. The champion's edge over the scorecard is small but significant (ΔAUC {dl_oot['diff']:+.3f}, DeLong p = {dl_oot['p_value']:.1g}). The riskiest decile captures {top1['cum_capture']:.0%} of OOT defaults.
- **Calibration.** Pooled OOT observed/predicted = {oot_ratio:.2f}; by vintage {v_ratio}. On the {stress_years} crisis vintages the model still ranks (AUC {C('stress','auc'):.3f}) but predicts {K('stress','citl_predicted'):.1%} against {K('stress','citl_observed'):.1%} observed: an under-prediction of {stress_ratio:.1f}×. The model has no macro inputs and has never seen a downturn.
{mono_bullet}
- **Target definition.** Without the hardship guard, 36-month default rates for {guard_years} are {ung_txt} instead of {g_txt}.

**Findings.** {counts.get('High', 0)} High ({len(open_high)} open), {counts.get('Medium', 0)} Medium, {counts.get('Low', 0)} Low (Section 11). The conditions on the rating are: no use for PD *levels* outside the 2009–2018 regime without a macro overlay; quarterly monitoring with the triggers in Section 12; replacement of the origination-rate input by a rate spread at the next redevelopment.

![AUC and default rate by vintage.](figures/fig1_timeline.png){{width=78%}}

# Model overview, intended use and materiality

The model estimates the probability that a newly originated loan becomes 90 or more days delinquent, or terminates through a credit event, within 36 months of its first payment. It is an *acquisition-time* model: every input is known at origination and no performance information is used. Population: fixed-rate, fully amortising, full-documentation conforming loans sold to Freddie Mac, excluding HARP/Relief Refinance loans, government loans, ARMs and non-standard programs (these are absent from Freddie's Standard dataset or excluded here).

Under SR 26-2 the intensity of validation follows *materiality*, the product of model exposure and purpose. For the purposes of this exercise the model is treated as it would be if it screened a conforming purchase book: material by exposure, moderate by purpose (screening and review, not capital). That places it in the tier where independent validation with outcomes analysis, benchmarking and sensitivity testing is expected before use. The same person built and validated this model, so the review is not independent in SR 26-2's sense.

# Regulatory context

This validation follows the *Supervisory Guidance on Model Risk Management* issued jointly by the Federal Reserve, OCC and FDIC on April 17, 2026 (Federal Reserve SR 26-2; OCC Bulletin 2026-13), which supersedes SR 11-7 (2011) and SR 21-8 and rescinds OCC 2011-12. SR 26-2 names three components of validation, **conceptual soundness**, **outcomes analysis** and **ongoing monitoring**, and frames the whole exercise through *effective challenge* by objective experts with the standing to effect change. Sections 5–8 of this report map onto those components; Appendix D gives the cross-walk to SR 26-2 paragraphs and to the SR 11-7 headings most readers know.

Footnote 3 of SR 26-2 carves out generative and agentic AI: "Generative AI and agentic AI models are novel and rapidly evolving. As such, they are not within the scope of this guidance. […] However, the principles described in this guidance apply to traditional statistical and quantitative models and non-generative, non-agentic AI models." A gradient-boosted PD model is squarely in scope.

Two further references shape the tests: the Basel Committee's *Studies on the Validation of Internal Rating Systems* (Working Paper 14, 2005) for the discrimination and calibration test catalogue, and the PRA's SS1/23 Principle 4 as an independent checklist of what a validation must cover. Fair-lending context comes from ECOA/Regulation B and CFPB Circular 2022-03.

# Data

**Source and lineage.** Freddie Mac Single-Family Loan-Level Dataset, Sample files, Release 47 (July 2026), downloaded from Clarity Data Intelligence on {d['manifest']['files'][0]['mtime_utc'][:10]}. Each vintage zip is hashed (SHA-256) in `data/manifest.json`; the file layout is verified programmatically against Freddie's July 2026 layout spreadsheet ({d['ingest']['layout']['orig']} origination fields, {d['ingest']['layout']['perf']} performance fields) before any column is named. Fourteen vintages were ingested, {sum(v['orig']['rows'] for v in d['ingest']['vintages']):,} loans and {sum(v['perf']['rows'] for v in d['ingest']['vintages']):,} loan-months, with zero orphan records between the origination and performance files. Performance runs through {tc and cfg['data']['performance_cutoff']}.

**Target definition.** `default_36m = 1` if within months 1–36 after first payment the loan is ever 90+ days delinquent (Freddie delinquency status ≥ 03, or RA), or terminates through a third-party sale, short sale/charge-off or REO disposition (zero-balance codes 02/03/09). Loan age is computed from First Payment Date, not Freddie's Loan Age field, which resets on modification. A 90+ month is **not** counted when the loan is in, or within {tc['forbearance_lookback_months']} months after, a servicer-reported forbearance plan (Borrower Assistance Plan = F) or a disaster-related delinquency flag. Credit-event terminations always count. Loans are observable if they reach month 36 or pay off voluntarily before it; defects (ZB 96) and loans sold before month 36 without a default are excluded. The pure ever-90+ definition and the forbearance-only guard are retained as sensitivities.

Without the guard, the {guard_years} vintages show 36-month default rates of {ung_txt}, against {g_txt} with it. Under the guarded definition the out-of-time vintages sit at {fp[2017]['default_rate_36m']:.2%} and {fp[2018]['default_rate_36m']:.2%}, close to 2016's {fp[2016]['default_rate_36m']:.2%}. This is a *data-relevance* issue in SR 26-2's sense and is carried as a Medium finding. The target definition and the train/tune split were revised after the out-of-time default rates were first examined.

**Sample design.** Vintage = sample file year (origination year; about 17% of each file's loans have a first payment in the following calendar year).

{md_table(t_wf, widths=[6, 10, 8, 8, 10, 7, 7])}

Train = {n_train:,} loans / {def_train:,} defaults; tune = 2016 (early stopping and hyper-parameter selection only); OOT = {n_oot:,} / {def_oot:,}; stress = {n_stress:,} / {def_stress:,}. The 2019 vintage's window spans the pandemic and is shown as a monitoring exhibit, not a test.

**Data quality.** FICO is missing on at most {max(na('fico')):.1%} of loans per vintage. DTI is missing on {span(dti_pre, '.1%')} of {pre_v[0]}–{pre_v[-1]} loans{dti_harp}. MSA is null on {span(na('msa'), '.0%')} of loans (non-metro properties). HARP loans, {span(harp, '.0%')} of {train_v[0]}–{train_v[-1]} originations, are excluded; this is the largest reduction in the waterfall and leaves {def_train_v.min()}–{def_train_v.max()} defaults per training vintage.

# Model development summary

| | Baseline | Challenger | Champion |
|---|---|---|---|
| Form | Logistic on Classic FICO (+ missing flag) | WoE scorecard: optbinning monotonic bins → L2 logistic | LightGBM, binary objective |
| Inputs | 1 | 16 | 16 |
| Constraints | — | Monotonic binning trend per input | `monotone_constraints` on FICO (−), LTV, CLTV, DTI, rate (+); method = intermediate |
| Selection | — | C = 1 | Random search, 24 configs; early stop on tune AUC; best: {champ['num_leaves']} leaves, min_child_samples {champ['min_child_samples']}, lr {champ['learning_rate']}, L2 {champ['lambda_l2']}, {d['card']['champion']['best_iteration']} trees |
| Class handling | None (no resampling or re-weighting; see §5) | None | None |

Inputs (information value from the scorecard binning, training sample): {", ".join(f"{r['name']} ({r['iv']:.2f})" for _, r in iv.head(8).iterrows())}; the remaining inputs each have IV < {iv.iloc[8]['iv']:.2f}.

The search history in `outputs/model_card.json` shows tune AUC in a narrow {min(h['tune_auc'] for h in d['card']['champion_search']):.3f}–{max(h['tune_auc'] for h in d['card']['champion_search']):.3f} band across all 24 configurations, i.e. the result is not sensitive to the hyper-parameter draw. Seeds are fixed; `make all` reproduces every number in this report from the raw zips.

# Conceptual soundness

**Variable rationale.** Each input has an expected economic direction: FICO (borrower credit history, −), LTV/CLTV (equity cushion and incentive, +), DTI (payment capacity, +), origination rate (risk-based pricing, +), number of borrowers (income diversification, −), loan purpose (cash-out refinance carries more risk than purchase), occupancy (investor/second-home), MI %, property type, channel, first-time-buyer flag, super-conforming flag, units, term and UPB. Geography (state, MSA, ZIP3), seller and servicer are **excluded** from the model and used only for segmentation and the fairness proxy: geography is a fair-lending risk, and seller/servicer identity is not a borrower attribute.

**Method choice.** A boosted-tree champion is justified only if it beats a well-built scorecard on out-of-time data and stays monotone in the inputs an underwriter would expect; both are tested in §6 and §7. Monotone constraints are essential for a credit model that will be explained to borrowers (Reg B adverse-action reasons) and to examiners.

**No class-imbalance correction.** The default rate is ~1%. Resampling or re-weighting would inflate predicted probabilities and destroy calibration, which is the property a PD model exists for (van den Goorbergh et al., 2022). Both models are trained on the natural prior.

**Finding on the rate input.** Origination interest rate has IV {iv.set_index('name').loc['orig_rate','iv']:.2f} and ranks {ordinal(perm_rank)} by permutation importance. The median rate moves between {min(med_rate):.2f}% and {max(med_rate):.2f}% across training vintages, so the input partly encodes *calendar time* rather than the borrower's risk-based spread. Recommendation: replace with the spread to the month's PMMS 30-year rate at redevelopment.

{mono_txt}
# Outcomes analysis

## Discrimination

{md_table(t_disc_p)}

Both multivariate models beat the FICO-only baseline out of time (by {L('oot','auc') - B('oot','auc'):.3f} and {C('oot','auc') - B('oot','auc'):.3f} AUC; DeLong p ≤ {dl_base['p_value'].max():.1g}). The champion's edge over the scorecard, {dl_oot['diff']:+.3f} on OOT (p = {dl_oot['p_value']:.1g}), is small. Train-to-OOT decay for the champion is {C('train','auc') - C('oot','auc'):.3f} ({C('train','auc'):.3f} → {C('tune','auc'):.3f} → {C('oot','auc'):.3f}); the train-to-tune step is in-sample optimism, the tune-to-OOT step is regime drift shared by all three models. The decay exceeds the {th['auc_train_oot_gap_max']} tolerance and is carried as a Medium finding.

![AUC by sample with bootstrap intervals.](figures/fig2_auc_by_role.png){{width=85%}}

**Capture.** The champion's riskiest decile holds {top1['cum_capture']:.0%} of OOT defaults at a {top1['obs_rate']:.1%} observed rate (lift {top1['lift']:.1f}×); the top two deciles hold {top2['cum_capture']:.0%}.

**Segments (OOT, champion; segments with ≥ {th['segment_min_defaults']} defaults).** The weakest within-segment ranking is {weak.iloc[0]['dimension']} = {weak.iloc[0]['segment']} (AUC {weak.iloc[0]['auc']:.3f}, 95% CI {weak.iloc[0]['auc_ci_low']:.3f}–{weak.iloc[0]['auc_ci_high']:.3f}), then {weak.iloc[1]['dimension']} = {weak.iloc[1]['segment']} ({weak.iloc[1]['auc']:.3f}). {'This is below the ' + str(th['segment_auc_min']) + ' segment floor and is carried as a finding. ' if weak_breach else ''}Full table in Appendix B.

![Within-segment AUC on the out-of-time sample.](figures/fig4_segments.png){{width=80%}}

## Calibration

{md_table(t_cal)}

Pooled out of time, observed/predicted is {oot_ratio:.2f} and the slope {K('oot','slope'):.2f} ± {K('oot','slope_se'):.2f}; by vintage the ratio is {v_ratio}, so the pooled figure averages out errors in opposite directions. {miss} of 10 OOT deciles have mean predicted PD outside the observed 95% Wilson interval. The slope is {K('train','slope'):.2f} on train and {K('tune','slope'):.2f} on tune. Calibration-in-the-large is outside ±{tol:.0%} on: {', '.join(breach)}. The 2019 COVID exhibit (observed/predicted {covid_ratio:.2f}) is not a valid calibration test: forbearance, deferral and disaster relief suppressed defaults that the guarded target removes from the numerator.

![Reliability diagrams (deciles, Wilson intervals).](figures/fig3_reliability.png){{width=100%}}

The model's inputs describe the borrower, not the economy; a 2006 borrower with a 720 FICO and 80 LTV looks like a 2014 borrower with the same attributes, and the model prices them the same. Rank-ordering survives (AUC {C('stress','auc'):.3f}) because relative risk is stable; the *level* does not. This is a design limitation, not a defect, but it bounds the intended use: PD levels from this model are point-in-time for the post-2009 regime and must carry a macro overlay for any capital, pricing or stress application.

## Stability

{md_table(t_stab, widths=[6, 10, 6, 7, 6, 8, 8])}

Score PSI against the training distribution is {dev_psi}; {psi17:.2f} for 2017 and {psi18:.2f} for 2018. The largest 2018 input shifts are {csi18.iloc[0]['variable']} (CSI {csi18.iloc[0]['csi']:.2f}), {csi18.iloc[1]['variable']} ({csi18.iloc[1]['csi']:.2f}) and {csi18.iloc[2]['variable']} ({csi18.iloc[2]['csi']:.2f}). Vintages over-predicted by more than {tol:.0%}: {', '.join(over) or 'none'}. The crisis vintages are far outside the development population (PSI above {th['psi_red']}), which is the point of the stress test.

![Score PSI by vintage and input CSI heat-map.](figures/fig5_stability.png){{width=100%}}

# Sensitivity analysis

![Permutation importance, SHAP, partial dependence and one-at-a-time shocks (champion, OOT).](figures/fig6_sensitivity.png){{width=100%}}

FICO dominates (permutation AUC drop {perm.iloc[0]['auc_drop']:.3f}); {perm.iloc[1]['variable']} ({perm.iloc[1]['auc_drop']:.3f}), {perm.iloc[2]['variable']} ({perm.iloc[2]['auc_drop']:.3f}) and {perm.iloc[3]['variable']} ({perm.iloc[3]['auc_drop']:.3f}) follow; {'SHAP gives the same top four. ' if shap_same else ''}One-at-a-time shocks on the OOT book:

{md_table(t_oat)}

Champion monotonicity grid test on 5,000 OOT loans (FICO 620–820 by 20, LTV and DTI by 5, rate 2.5–7% by 0.5): maximum violation rate {mono_max:.1%}.

# Benchmarking and model recommendation

The champion's AUC lead over the scorecard ranges from {min(bench):+.3f} to {max(bench):+.3f} across samples. The scorecard is transparent (one coefficient per bin); the champion needs SHAP for reason codes. **Recommendation:** adopt the champion for ranking, with the scorecard retained as the production challenger and the FICO-only model as the floor in monitoring. If the institution's adverse-action process cannot consume SHAP-based reasons, the scorecard's cost is {abs(dl_oot['diff']):.3f} AUC, which is small enough that transparency should win.

# Fairness / disparate-impact proxy analysis
{fair_section}
# Limitations

- **Population.** Freddie Standard-eligible loans only: fixed-rate, full-doc, conforming, already approved. No rejected applicants exist in the data, so the model cannot be used for approve/decline without reject inference. HARP loans are excluded.
- **Sample.** 50,000 loans per vintage, not the full 1–2 million; segment-level estimates in OOT have wide intervals (Appendix B).
- **Target.** Ever-90+ in 36 months is a delinquency target, not a loss target; no LGD/EAD. The hardship guard depends on servicer-reported flags whose usage changed in 2020.
- **No macro inputs and no downturn in training** (2009–2015). PD levels are point-in-time for that regime.
- **Fairness** is a geography-only proxy: no applicant-level protected-class data, no surname component, no compliance conclusion.
- **Process.** Self-validation by the model's developer. The target definition and train/tune split were revised after the out-of-time default rates were first examined.
- **Freddie's dataset is "living"**: historical records can be restated between releases; the manifest pins Release 47.

# Findings register and overall rating

Severity definitions: **High** – materially affects fitness for intended use; must be remediated or use restricted before reliance. **Medium** – limits use or reliability in identifiable conditions; remediate within one cycle, monitor meanwhile. **Low** – documentation, minor deviation, or observation.

{md_table(t_find, widths=[3, 8, 20, 30, 30])}

**Overall rating: {rating}.** Conditions: (1) no reliance on PD levels outside the 2009–2018 regime without a macro overlay or recalibration; (2) quarterly monitoring per Section 12 with the hardship-guarded and pure targets both tracked; (3) replacement of the origination-rate input by a rate spread, and a fair-lending review with real protected-class data, before any policy use; {'(4) the ' + str(weak.iloc[0]['dimension']) + ' = ' + str(weak.iloc[0]['segment']) + ' segment to be reviewed with a segment overlay or a dedicated model.' if weak_breach else ''}

# Ongoing monitoring plan

| Metric | Frequency | Green / Amber / Red | Action on Red |
|---|---|---|---|
| Score PSI vs development | Quarterly | < {th['psi_amber']} / {th['psi_amber']}–{th['psi_red']} / ≥ {th['psi_red']} | Input CSI review; recalibrate if persistent 2 quarters |
| AUC on matured cohorts (12-month early-warning target as proxy) | Quarterly | ≥ {th['auc_oot_min']:.2f} and within {th['auc_train_oot_gap_max']} of development / – / below | Redevelopment trigger |
| Calibration-in-the-large (obs/pred) | Quarterly | within ±{th['calib_in_large_rel_tol']:.0%} / ±{th['calib_in_large_rel_tol']:.0%}–25% / > 25% | Intercept recalibration; macro overlay review |
| Segment AUC (FICO, LTV, purpose, occupancy) | Semi-annual | ≥ {th['segment_auc_min']} where ≥ {th['segment_min_defaults']} defaults | Segment overlay |
| Share of 90+ events under forbearance / disaster flags | Quarterly | < 10% / 10–30% / > 30% | Data-relevance review; suspend calibration monitoring on affected cohorts |
| Monotonicity grid test | On every re-fit | 0% violations | Block release |
| Full revalidation | Annual, or on any redevelopment | — | — |

\\newpage

# Appendix A: Data dictionary and cleaning log {{-}}

Inputs (origination file, Release 47 positions): Classic FICO (1), First Payment Date (2), First-Time Homebuyer (3), MSA (5), MI % (6), Units (7), Occupancy (8), CLTV (9), DTI (10), UPB (11), LTV (12), Rate (13), Channel (14), Amortization Type (16), State (17), Property Type (18), ZIP3 (19), Loan ID (20), Purpose (21), Term (22), Borrowers (23), Seller (24), Super-Conforming (25), HARP (28). Sentinel codes (FICO 9999; LTV/CLTV/DTI/MI 999; units 99; categorical 9/99) are mapped to missing; categorical missing becomes an explicit "NA" level. Performance file fields used only for the target: Period, Delinquency Status, Zero Balance Code, Borrower Assistance Plan, Delinquency Due to Disaster. Unit tests cover every branch of the target definition (15 hand-built loan histories) and assert that no performance column other than the target enters the feature frame.

# Appendix B: Segment tables (OOT, champion) {{-}}

{md_table(t_seg, widths=[12, 8, 6, 6, 7, 7, 14])}

# Appendix C: Reproducibility {{-}}

`make data && make target && make features && make train && make validate && make figures && make report` from the raw zips in `data/raw/`; seed {cfg['seed']}; Python 3.12, LightGBM {d['card'].get('lightgbm_version', '4.7')}, optbinning, scikit-learn, DuckDB; dependency lock in `uv.lock`. Raw file SHA-256 digests in `data/manifest.json`. Champion hyper-parameter search history in `outputs/model_card.json`.

# Appendix D: Guidance cross-walk {{-}}

| Report section | SR 26-2 (2026) | SR 11-7 (2011) lineage |
|---|---|---|
| 1 Intended use, materiality | §II Purpose and scope; §III model risk, materiality | §III Overview |
| 4 Model development | §IV Model development and use (testing, out-of-time) | §IV Model development, implementation and use |
| 5 Conceptual soundness | §V Components of validation – conceptual soundness | §V Validation – evaluation of conceptual soundness |
| 6 Outcomes analysis; 7 Sensitivity; 8 Benchmarking | §V – outcomes analysis; §IV testing (comparison of alternatives) | §V – outcomes analysis; ongoing monitoring (benchmarking, sensitivity) |
| 12 Monitoring plan | §V – ongoing model monitoring | §V – ongoing monitoring |
| 11 Findings register | §VI Governance – documentation, tracking of recommendations | §VI Governance, policies and controls |
| 9 Fairness proxy | out of scope of SR 26-2; ECOA/Reg B, CFPB Circular 2022-03 | — |
"""
    return md


def build_exec_summary(d: dict) -> str:
    disc, cal = d["disc"], d["cal"]
    C = lambda r, k: disc.loc[("champion", r), k]
    K = lambda r, k: cal.loc[("champion", r), k]
    find = d["find"]; counts = find["severity"].value_counts().to_dict()
    open_high = find[(find["severity"] == "High") & (find["status"] != "Remediated")]
    top = find[find["severity"].isin(["High", "Medium"])].head(3)
    n_oot = int(d["wf"].query("role=='oot'")["after_amortization_filter"].sum()); def_oot = int(d["wf"].query("role=='oot'")["defaults"].sum())
    rating = "Fit for intended use, with conditions" if len(open_high) == 0 else "Not fit without remediation"
    return f"""---
title: "Executive Summary: Validation of a Mortgage PD Model"
author: "Qavi Inan"
date: "{datetime.now(tz=UTC):%B %d, %Y}"
geometry: margin=0.65in
fontsize: 9pt
colorlinks: true
header-includes:
  - \\usepackage{{float}}
  - \\floatplacement{{figure}}{{H}}
  - \\pagenumbering{{gobble}}
  - \\setlength{{\\parskip}}{{3pt}}
---

**What was validated.** A 36-month probability-of-default model for conforming fixed-rate mortgages at origination: a LightGBM champion with monotone constraints, benchmarked against a weight-of-evidence scorecard and a FICO-only baseline, on Freddie Mac's public loan-level sample (fourteen vintages, 2006–2019, 700,000 loans). Validation follows SR 26-2 (April 2026): conceptual soundness, outcomes analysis and ongoing monitoring, against thresholds set in `configs/default.yaml`.

**Intended use.** Ranking new originations for screening and review; point-in-time PD within the post-2009 regime. Not for capital, pricing or stress without a macro overlay.

**Overall rating: {rating}.**

| | Champion | Scorecard | FICO-only |
|---|---|---|---|
| AUC, OOT 2017–18 ({n_oot:,} loans / {def_oot:,} defaults) | **{C('oot','auc'):.3f}** ({C('oot','auc_ci_low'):.3f}–{C('oot','auc_ci_high'):.3f}) | {disc.loc[('challenger','oot'),'auc']:.3f} | {disc.loc[('baseline','oot'),'auc']:.3f} |
| Observed/predicted PD, OOT | **{K('oot','citl_ratio_obs_pred'):.2f}** | {cal.loc[('challenger','oot'),'citl_ratio_obs_pred']:.2f} | {cal.loc[('baseline','oot'),'citl_ratio_obs_pred']:.2f} |
| Observed/predicted PD, 2006–08 crisis | **{K('stress','citl_ratio_obs_pred'):.1f}×** | {cal.loc[('challenger','stress'),'citl_ratio_obs_pred']:.1f}× | {cal.loc[('baseline','stress'),'citl_ratio_obs_pred']:.1f}× |
| AUC, crisis vintages | {C('stress','auc'):.3f} | {disc.loc[('challenger','stress'),'auc']:.3f} | {disc.loc[('baseline','stress'),'auc']:.3f} |

![](figures/fig1_timeline.png){{width=50%}}

**Top findings** ({counts.get('High', 0)} High, {counts.get('Medium', 0)} Medium, {counts.get('Low', 0)} Low):

{chr(10).join(f"- **{r['id']} ({r['severity']}{', ' + r['status'].lower() if r['status'] != 'Open' else ''}).** {r['title']}. {r['recommendation']}" for _, r in top.iterrows())}

**Conditions of use.** Macro overlay or recalibration before any use of PD levels outside 2009–2018; quarterly monitoring of score PSI, AUC, calibration-in-the-large and the share of delinquencies under hardship flags; replace the origination-rate input with a rate spread and complete a fair-lending review with protected-class data before policy use.
"""


def render_pdf(md_path: Path, pdf_path: Path) -> None:
    cmd = ["pandoc", str(md_path), "-o", str(pdf_path), "--pdf-engine=tectonic", "--from", "markdown+pipe_tables+yaml_metadata_block",
           "--resource-path", str(md_path.parent), "-V", "linkcolor=blue"]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args(argv)
    d = load()
    R.mkdir(exist_ok=True)
    (R / "report.md").write_text(tex_symbols(build_report(d)))
    (R / "executive_summary.md").write_text(tex_symbols(build_exec_summary(d)))
    print("wrote report/report.md, report/executive_summary.md")
    if not args.no_pdf:
        render_pdf(R / "report.md", R / "report.pdf")
        render_pdf(R / "executive_summary.md", R / "executive_summary.pdf")
        print("rendered report/report.pdf, report/executive_summary.pdf")


if __name__ == "__main__":
    main()
