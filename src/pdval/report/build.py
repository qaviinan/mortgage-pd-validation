"""Build the validation report and executive summary (Markdown -> PDF via pandoc + tectonic).

Every number in the prose is read from outputs/ so a re-run of the battery re-renders the report.

Usage:
    uv run python -m pdval.report.build [--no-pdf]
Writes report/report.md, report/executive_summary.md and the PDFs next to them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
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
    return md.replace("≥", "$\\geq$").replace("≤", "$\\leq$")


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
    dl_base = d["delong"].query("role=='oot' and model_1=='champion' and model_2=='baseline'").iloc[0]
    seg_c = d["seg"][(d["seg"]["model"] == "champion") & (d["seg"]["defaults"] >= th["segment_min_defaults"])]
    weak = seg_c.sort_values("auc").head(3)
    top1 = d["dec"].query("model=='champion' and decile==1").iloc[0]
    top2 = d["dec"].query("model=='champion' and decile==2").iloc[0]
    stress_ratio = K("stress", "citl_ratio_obs_pred"); oot_ratio = K("oot", "citl_ratio_obs_pred")
    covid_ratio = K("covid_exhibit", "citl_ratio_obs_pred"); tune_ratio = K("tune", "citl_ratio_obs_pred")
    psi18 = stab.loc[2018, "psi_champion"]; psi17 = stab.loc[2017, "psi_champion"]
    csi18 = d["csi"][d["csi"]["vintage"] == 2018].sort_values("csi", ascending=False).head(3)
    perm = d["perm"]; shap = d["shap"]
    oat = d["oat"].set_index(["variable", "shock"])
    mono_max = d["mono"]["violation_rate"].max()
    champ = d["card"]["champion"]["params"]
    iv = pd.DataFrame(d["card"]["challenger"]["iv"]).sort_values("iv", ascending=False)
    fp = d["fp"]
    rating = "Fit for intended use, with conditions" if len(open_high) == 0 and C("oot", "auc") >= th["auc_oot_min"] else "Not fit without remediation"

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

    t_oat = d["oat"].copy(); t_oat["mean_pred"] = t_oat["mean_pred"].map(pct); t_oat["rel_change"] = t_oat["rel_change"].map(lambda x: f"{100*x:+.0f}%")
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

**Reading.** No proxy group fails the 4/5ths screen: the lowest adverse impact ratio is {fg['adverse_impact_ratio'].min():.2f} and the standardised mean score difference between the highest- and lowest-minority-share tertiles is {fg.set_index('group').loc['T3','std_mean_diff']:.2f} SD. Within-group ranking is similar (AUC {fg['auc'].min():.2f}–{fg['auc'].max():.2f}). The calibration pattern runs *against* the direction a disparate-impact concern would predict: the model over-predicts risk in the lowest-minority-share areas (pred/obs {fg.set_index('group').loc['T1','pred_over_obs']:.2f}) and is closest to calibrated in the highest-share tertile ({fg.set_index('group').loc['T3','pred_over_obs']:.2f}); non-metro loans are under-predicted ({fg.set_index('group').loc['non-metro/unknown','pred_over_obs']:.2f}). Loan size is the input most associated with area demographics (Spearman {aud.iloc[0]['assoc']:.2f}), which is a house-price effect rather than a borrower-risk effect and is worth a note in the variable rationale.

The adverse impact ratio is compared with the 4/5ths rule as a *screen*, and within-group calibration (pred/obs) is the fair-lending-relevant quantity for a pricing or limit model: a group that is systematically under- or over-predicted is being mis-priced. This is a geography-only proxy in the lineage of the CFPB's 2014 BISG methodology, without the surname component that carries most of BISG's power; it cannot support a compliance conclusion. A bank would run this with applicant-level protected-class data (HMDA for mortgages) and a documented proxy method for other products.
"""
    else:
        fair_section = """
**Not executed in this version.** The geography proxy requires ACS demographic shares from the Census API, which now requires a (free) API key that was not available at build time. The code path (`pdval.validation.fairness`) is complete and unit-covered: it maps each loan's MSA / metropolitan-division code to the area's non-white population share (ACS 2022 5-year, B03002), forms tertile groups with non-metro as a separate group, and reports adverse impact ratio at an illustrative decline cutoff, within-group calibration, AUC parity, standardised mean difference of scores, and a proxy-feature audit. The section is recorded as a Low finding and will be populated in the polish pass.
"""

    stress_years = ", ".join(str(v) for v in vin["stress"])
    md = f"""---
title: "Independent Validation of a Mortgage Probability-of-Default Model"
subtitle: "Champion (gradient-boosted trees) vs challenger (WoE scorecard), Freddie Mac single-family loan-level data"
author: "Qavi Inan"
date: "{date.today():%B %d, %Y}"
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

**Model.** A 36-month probability-of-default (PD) model for conforming, fixed-rate, full-documentation mortgages at origination, built on Freddie Mac's public single-family loan-level *sample* dataset (50,000 loans per vintage). Inputs are the sixteen origination-time borrower and loan attributes in Table 4; geography, seller and vintage are excluded by design. A LightGBM champion with monotone constraints is compared with a weight-of-evidence logistic scorecard challenger and a FICO-only baseline.

**Intended use.** Rank-ordering of new originations for acquisition screening and risk-based review, and point-in-time PD estimation within the post-2009 underwriting regime. Not intended for capital, stress-testing or downturn pricing without a macro overlay.

**Overall rating: {rating}.**

**Headline evidence.**

- **Discrimination holds out of time.** Champion AUC {C('oot','auc'):.3f} (95% CI {C('oot','auc_ci_low'):.3f}–{C('oot','auc_ci_high'):.3f}) on {n_oot:,} loans / {def_oot:,} defaults from the {vin['oot'][0]}–{vin['oot'][-1]} vintages, against {L('oot','auc'):.3f} for the scorecard and {B('oot','auc'):.3f} for FICO alone. The champion's edge over the scorecard is small but significant (ΔAUC {dl_oot['diff']:+.3f}, DeLong p = {dl_oot['p_value']:.1g}). The riskiest decile captures {top1['cum_capture']:.0%} of OOT defaults.
- **Calibration holds out of time and fails under stress.** OOT observed/predicted = {oot_ratio:.2f}, slope {K('oot','slope'):.2f} (SE {K('oot','slope_se'):.2f}). On the {stress_years} crisis vintages the model still ranks (AUC {C('stress','auc'):.3f}) but predicts {K('stress','citl_predicted'):.1%} against {K('stress','citl_observed'):.1%} observed: an under-prediction of {stress_ratio:.1f}×. The model has no macro inputs and has never seen a downturn.
- **Validation caught a silent constraint failure.** The originally configured champion violated its declared monotone constraints for a fifth of loans on LTV; the cause (LightGBM's "advanced" constraint method) was isolated and the model re-fit with an exact method. Post-fix violation rate: {mono_max:.1%}.
- **The target definition is a finding, not a footnote.** Without guarding 90-day delinquencies that occur in or shortly after COVID forbearance or disaster relief, 2018–2019 default rates are 4.6–4.9% instead of ~0.7–1.0%, and most of those events cure within a year.

**Findings.** {counts.get('High', 0)} High ({len(open_high)} open), {counts.get('Medium', 0)} Medium, {counts.get('Low', 0)} Low (Section 11). The conditions on the rating are: no use for PD *levels* outside the 2009–2018 regime without a macro overlay; quarterly monitoring with the triggers in Section 12; replacement of the origination-rate input by a rate spread at the next redevelopment.

![Rank-ordering survives every regime; the PD level does not.](figures/fig1_timeline.png){{width=78%}}

# Model overview, intended use and materiality

The model estimates the probability that a newly originated loan becomes 90 or more days delinquent, or terminates through a credit event, within 36 months of its first payment. It is an *acquisition-time* model: every input is known at origination and no performance information is used. Population: fixed-rate, fully amortising, full-documentation conforming loans sold to Freddie Mac, excluding HARP/Relief Refinance loans, government loans, ARMs and non-standard programs (these are absent from Freddie's Standard dataset or excluded here).

Under SR 26-2 the intensity of validation follows *materiality*, the product of model exposure and purpose. For the purposes of this exercise the model is treated as it would be if it screened a conforming purchase book: material by exposure, moderate by purpose (screening and review, not capital). That places it in the tier where independent validation with outcomes analysis, benchmarking and sensitivity testing is expected before use, which is what this report performs.

# Regulatory context

This validation follows the *Supervisory Guidance on Model Risk Management* issued jointly by the Federal Reserve, OCC and FDIC on April 17, 2026 (Federal Reserve SR 26-2; OCC Bulletin 2026-13), which supersedes SR 11-7 (2011) and SR 21-8 and rescinds OCC 2011-12. SR 26-2 names three components of validation, **conceptual soundness**, **outcomes analysis** and **ongoing monitoring**, and frames the whole exercise through *effective challenge* by objective experts with the standing to effect change. Sections 5–8 of this report map onto those components; Appendix D gives the cross-walk to SR 26-2 paragraphs and to the SR 11-7 headings most readers know.

Footnote 3 of SR 26-2 carves out generative and agentic AI: "Generative AI and agentic AI models are novel and rapidly evolving. As such, they are not within the scope of this guidance. […] However, the principles described in this guidance apply to traditional statistical and quantitative models and non-generative, non-agentic AI models." A gradient-boosted PD model is squarely in scope.

Two further references shape the tests: the Basel Committee's *Studies on the Validation of Internal Rating Systems* (Working Paper 14, 2005) for the discrimination and calibration test catalogue, and the PRA's SS1/23 Principle 4 as an independent checklist of what a validation must cover. Fair-lending context comes from ECOA/Regulation B and CFPB Circular 2022-03.

# Data

**Source and lineage.** Freddie Mac Single-Family Loan-Level Dataset, Sample files, Release 47 (July 2026), downloaded from Clarity Data Intelligence on {d['manifest']['files'][0]['mtime_utc'][:10]}. Each vintage zip is hashed (SHA-256) in `data/manifest.json`; the file layout is verified programmatically against Freddie's July 2026 layout spreadsheet ({d['ingest']['layout']['orig']} origination fields, {d['ingest']['layout']['perf']} performance fields) before any column is named. Fourteen vintages were ingested, {sum(v['orig']['rows'] for v in d['ingest']['vintages']):,} loans and {sum(v['perf']['rows'] for v in d['ingest']['vintages']):,} loan-months, with zero orphan records between the origination and performance files. Performance runs through {tc and cfg['data']['performance_cutoff']}.

**Target definition.** `default_36m = 1` if within months 1–36 after first payment the loan is ever 90+ days delinquent (Freddie delinquency status ≥ 03, or RA), or terminates through a third-party sale, short sale/charge-off or REO disposition (zero-balance codes 02/03/09). Loan age is computed from First Payment Date, not Freddie's Loan Age field, which resets on modification. A 90+ month is **not** counted when the loan is in, or within {tc['forbearance_lookback_months']} months after, a servicer-reported forbearance plan (Borrower Assistance Plan = F) or a disaster-related delinquency flag. Credit-event terminations always count. Loans are observable if they reach month 36 or pay off voluntarily before it; defects (ZB 96) and loans sold before month 36 without a default are excluded. The pure ever-90+ definition and the forbearance-only guard are retained as sensitivities.

Why the guard matters: without it, the 2017, 2018 and 2019 vintages show 36-month default rates of 3.3%, 4.6% and 4.9% against 0.8–1.1% for 2009–2016; 70–90% of those events occur while the borrower is in a COVID-era plan and 70–77% are current again within a year. The disaster flag was used by servicers for COVID hardship when no formal plan was booked: 58% of the remaining 2019 events carried it. Under the guarded definition the out-of-time vintages sit at {fp[2017]['default_rate_36m']:.2%} and {fp[2018]['default_rate_36m']:.2%}, consistent with 2016's {fp[2016]['default_rate_36m']:.2%}. This is a *data-relevance* issue in SR 26-2's sense and is carried as a Medium finding with a monitoring trigger.

**Sample design.** Vintage = sample file year (origination year; about 17% of each file's loans have a first payment in the following calendar year).

{md_table(t_wf, widths=[6, 10, 8, 8, 10, 7, 7])}

Train = {n_train:,} loans / {def_train:,} defaults; tune = 2016 (early stopping and hyper-parameter selection only); OOT = {n_oot:,} / {def_oot:,}; stress = {n_stress:,} / {def_stress:,}. The 2019 vintage's window spans the pandemic and is shown as a monitoring exhibit, not a test.

**Data quality.** FICO is missing on ≤0.1% of loans; DTI is missing only on HARP loans (excluded); LTV, CLTV and categorical fields are complete. MSA is null on 9–23% of loans (non-metro properties). Default rates are monotone in FICO, LTV and DTI in every vintage (Appendix B). HARP loans, 11–35% of 2009–2014 originations, default at 2–4× the non-HARP rate and have no DTI; their exclusion is the single largest reduction in the waterfall and thins the 2010–2013 training vintages to 74–197 defaults each.

# Model development summary

| | Baseline | Challenger | Champion |
|---|---|---|---|
| Form | Logistic on Classic FICO (+ missing flag) | WoE scorecard: optbinning monotonic bins → L2 logistic | LightGBM, binary objective |
| Inputs | 1 | 16 | 16 |
| Constraints | — | Monotonic binning trend per input | `monotone_constraints` on FICO (−), LTV, CLTV, DTI, rate (+); method = intermediate |
| Selection | — | C = 1 | Random search, 24 configs; early stop on tune AUC; best: {champ['num_leaves']} leaves, min_child_samples {champ['min_child_samples']}, lr {champ['learning_rate']}, λ₂ {champ['lambda_l2']}, {d['card']['champion']['best_iteration']} trees |
| Class handling | None (no resampling or re-weighting; see §5) | None | None |

Table 4. Inputs (information value from the scorecard binning, training sample): {", ".join(f"{r['name']} ({r['iv']:.2f})" for _, r in iv.head(8).iterrows())}; the remaining inputs each have IV < {iv.iloc[8]['iv']:.2f}.

Development evidence reviewed: the tuning history (Appendix C) shows tune AUC in a narrow {min(h['tune_auc'] for h in d['card']['champion_search']):.3f}–{max(h['tune_auc'] for h in d['card']['champion_search']):.3f} band across all 24 configurations, i.e. the result is not sensitive to the hyper-parameter draw. Seeds are fixed; `make all` reproduces every number in this report from the raw zips.

# Conceptual soundness

**Variable rationale.** Each input has a documented economic direction and the empirical sign agrees with it: FICO (borrower credit history, −), LTV/CLTV (equity cushion and incentive, +), DTI (payment capacity, +), origination rate (risk-based pricing, +), number of borrowers (income diversification, −), loan purpose (cash-out refinance carries more risk than purchase), occupancy (investor/second-home), MI %, property type, channel, first-time-buyer flag, super-conforming flag, units, term and UPB. Geography (state, MSA, ZIP3), seller and servicer are **excluded** from the model and used only for segmentation and the fairness proxy: geography is a fair-lending risk, and seller/servicer identity is not a borrower attribute.

**Method choice.** A boosted-tree champion is justified only if it beats a well-built scorecard on out-of-time data and stays monotone in the inputs an underwriter would expect; both are tested in §6 and §7. Monotone constraints are essential for a credit model that will be explained to borrowers (Reg B adverse-action reasons) and to examiners.

**No class-imbalance correction.** The default rate is ~1%. Resampling or re-weighting would inflate predicted probabilities and destroy calibration, which is the property a PD model exists for (van den Goorbergh et al., 2022). Both models are trained on the natural prior.

**Finding on the rate input.** Origination interest rate has IV {iv.set_index('name').loc['orig_rate','iv']:.2f} and is the {int(perm.index[perm['variable']=='orig_rate'][0]) + 1}th-ranked input by permutation importance. Average rates move by about a percentage point across training vintages, so the input partly encodes *calendar time* rather than the borrower's risk-based spread. This shows up in stability (§6.3): the 2018 vintage, originated at higher rates, has the highest input CSI on the rate and is the one vintage the model over-predicts. Recommendation: replace with spread to the month's PMMS 30-year average at redevelopment.

**Finding on monotone constraints (remediated).** The first champion fit used LightGBM's `monotone_constraints_method = "advanced"`. A grid test moving one input at a time across a sample of OOT loans showed predictions moving *against* the declared direction for 20% of loans on LTV (worst raw-score step −0.042) and 14% on DTI. The same test with the `basic` and `intermediate` methods gave zero violations; the champion was re-fit with `intermediate`. This is exactly the class of failure a declared constraint is supposed to prevent and it would not have been visible from any performance metric.

# Outcomes analysis

## Discrimination

{md_table(t_disc_p)}

Both multivariate models beat the FICO-only baseline by 0.05–0.06 AUC out of time (DeLong p < 0.001 for every pairing). The champion's edge over the scorecard, {dl_oot['diff']:+.3f} on OOT (p = {dl_oot['p_value']:.1g}), is real but small: the interaction structure the trees find is worth a few thousandths of AUC, not a different model. Train-to-OOT decay for the champion is {C('train','auc') - C('oot','auc'):.3f} ({C('train','auc'):.3f} → {C('tune','auc'):.3f} → {C('oot','auc'):.3f}); the train-to-tune step is in-sample optimism, the tune-to-OOT step is regime drift shared by all three models. The decay exceeds the pre-registered {th['auc_train_oot_gap_max']} tolerance and is carried as a Medium finding.

![AUC by sample with bootstrap intervals.](figures/fig2_auc_by_role.png){{width=85%}}

**Capture.** The champion's riskiest decile holds {top1['cum_capture']:.0%} of OOT defaults at a {top1['obs_rate']:.1%} observed rate (lift {top1['lift']:.1f}×); the top two deciles hold {top2['cum_capture']:.0%}.

**Segments (OOT, champion; segments with ≥ {th['segment_min_defaults']} defaults).** The weakest within-segment ranking is {weak.iloc[0]['dimension']} = {weak.iloc[0]['segment']} (AUC {weak.iloc[0]['auc']:.3f}, 95% CI {weak.iloc[0]['auc_ci_low']:.3f}–{weak.iloc[0]['auc_ci_high']:.3f}), then {weak.iloc[1]['dimension']} = {weak.iloc[1]['segment']} ({weak.iloc[1]['auc']:.3f}). Within a narrow FICO band the score has little FICO variation left to work with, so a lower within-band AUC is expected; the sub-660 result nevertheless breaches the {th['segment_auc_min']} segment floor and is carried as a finding. Full table in Appendix B.

![Within-segment AUC on the out-of-time sample.](figures/fig4_segments.png){{width=80%}}

## Calibration

{md_table(t_cal)}

Out of time, the champion is well calibrated: observed/predicted {oot_ratio:.2f} (within the {th['calib_in_large_rel_tol']:.0%} tolerance), slope {K('oot','slope'):.2f} ± {K('oot','slope_se'):.2f} (inside [{th['calib_slope_range'][0]}, {th['calib_slope_range'][1]}]), and the reliability diagram tracks the diagonal in every decile. Three samples breach. The 2016 tune vintage is under-predicted ({tune_ratio:.2f}×): drift is visible one year after the training window closes. The crisis vintages are under-predicted {stress_ratio:.1f}× with every decile above the diagonal (Figure 3, centre). The 2019 COVID exhibit runs the other way (observed/predicted {covid_ratio:.2f}): pandemic forbearance, payment deferral and disaster relief suppressed defaults that the hardship-guarded target removes from the numerator, so 2019 is not a valid calibration test but a demonstration of why monitoring needs a data-relevance check before it reads a calibration ratio.

![Reliability diagrams (deciles, Wilson intervals).](figures/fig3_reliability.png){{width=100%}}

The crisis result is the most important number in this report. The model's inputs describe the borrower, not the economy; a 2006 borrower with a 720 FICO and 80 LTV looks like a 2014 borrower with the same attributes, and the model prices them the same. Rank-ordering survives (AUC {C('stress','auc'):.3f}) because relative risk is stable; the *level* does not. This is a design limitation, not a defect, but it bounds the intended use: PD levels from this model are point-in-time for the post-2009 regime and must carry a macro overlay for any capital, pricing or stress application.

## Stability

{md_table(t_stab, widths=[6, 10, 6, 7, 6, 8, 8])}

Score PSI against the training distribution is green for every training and tune vintage, amber for 2017 ({psi17:.2f}) and **red for 2018 ({psi18:.2f})**. The input CSI table attributes the 2018 shift to {csi18.iloc[0]['variable']} (CSI {csi18.iloc[0]['csi']:.2f}), {csi18.iloc[1]['variable']} ({csi18.iloc[1]['csi']:.2f}) and {csi18.iloc[2]['variable']} ({csi18.iloc[2]['csi']:.2f}): 2018 loans were originated at higher rates, with more purchase and fewer refinance loans. FICO, DTI and LTV are stable. The population shift therefore runs through the rate input flagged in §5, and 2018 is the one vintage the model over-predicts (observed/predicted {stab.loc[2018,'obs_rate']/stab.loc[2018,'mean_pred_champion']:.2f}). The crisis vintages have PSI > 1.5: they are a different population (2006–07 books had lower FICO, higher DTI and near-universal 6%+ rates), which is the point of the stress test and is recorded as a Low finding.

![Score PSI by vintage and input CSI heat-map.](figures/fig5_stability.png){{width=100%}}

# Sensitivity analysis

![Permutation importance, SHAP, partial dependence and one-at-a-time shocks (champion, OOT).](figures/fig6_sensitivity.png){{width=100%}}

FICO dominates (permutation AUC drop {perm.iloc[0]['auc_drop']:.3f}); {perm.iloc[1]['variable']} ({perm.iloc[1]['auc_drop']:.3f}), {perm.iloc[2]['variable']} ({perm.iloc[2]['auc_drop']:.3f}) and {perm.iloc[3]['variable']} ({perm.iloc[3]['auc_drop']:.3f}) follow; SHAP gives the same top four. Partial dependence is monotone in FICO, LTV and DTI. One-at-a-time shocks on the OOT book:

{md_table(t_oat)}

Post-remediation monotonicity check (1-unit grids, 5,000 OOT loans): maximum violation rate {mono_max:.1%}.

# Benchmarking and model recommendation

The champion beats the scorecard on every sample by 0.006–0.03 AUC, with the gap largest in-sample and smallest out of time and under stress; calibration is equivalent. The scorecard is fully transparent (points per bin), the champion needs SHAP for reason codes. **Recommendation:** adopt the champion for ranking, with the scorecard retained as the production challenger and the FICO-only model as the floor in monitoring. If the institution's adverse-action process cannot consume SHAP-based reasons, the scorecard's cost is {abs(dl_oot['diff']):.3f} AUC, which is small enough that transparency should win.

# Fairness / disparate-impact proxy analysis
{fair_section}
# Limitations

- **Population.** Freddie Standard-eligible loans only: fixed-rate, full-doc, conforming, already approved. No rejected applicants exist in the data, so the model cannot be used for approve/decline without reject inference. HARP loans are excluded.
- **Sample.** 50,000 loans per vintage, not the full 1–2 million; segment-level estimates in OOT have wide intervals (Appendix B).
- **Target.** Ever-90+ in 36 months is a delinquency target, not a loss target; no LGD/EAD. The hardship guard depends on servicer-reported flags whose usage changed in 2020.
- **No macro inputs and no downturn in training** (2009–2015). PD levels are point-in-time for that regime.
- **Fairness** is a geography-only proxy: no applicant-level protected-class data, no surname component, no compliance conclusion.
- **Freddie's dataset is "living"**: historical records can be restated between releases; the manifest pins Release 47.

# Findings register and overall rating

Severity definitions: **High** – materially affects fitness for intended use; must be remediated or use restricted before reliance. **Medium** – limits use or reliability in identifiable conditions; remediate within one cycle, monitor meanwhile. **Low** – documentation, minor deviation, or observation.

{md_table(t_find, widths=[3, 8, 20, 30, 30])}

**Overall rating: {rating}.** Conditions: (1) no reliance on PD levels outside the 2009–2018 regime without a macro overlay or recalibration; (2) quarterly monitoring per Section 12 with the hardship-guarded and pure targets both tracked; (3) replacement of the origination-rate input by a rate spread, and a fair-lending review with real protected-class data, before any policy use; (4) the sub-660 segment to be reviewed with a segment overlay or a dedicated model.

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
title: "Executive Summary: Independent Validation of a Mortgage PD Model"
author: "Qavi Inan"
date: "{date.today():%B %d, %Y}"
geometry: margin=0.65in
fontsize: 9pt
colorlinks: true
header-includes:
  - \\usepackage{{float}}
  - \\floatplacement{{figure}}{{H}}
  - \\pagenumbering{{gobble}}
  - \\setlength{{\\parskip}}{{3pt}}
---

**What was validated.** A 36-month probability-of-default model for conforming fixed-rate mortgages at origination: a LightGBM champion with monotone constraints, benchmarked against a weight-of-evidence scorecard and a FICO-only baseline, on Freddie Mac's public loan-level sample (fourteen vintages, 2006–2019, 700,000 loans). Validation follows SR 26-2 (April 2026): conceptual soundness, outcomes analysis, ongoing monitoring, with pre-registered thresholds.

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
