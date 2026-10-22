# Mortgage PD model + independent validation report

[![ci](https://github.com/qaviinan/mortgage-pd-validation/actions/workflows/ci.yml/badge.svg)](https://github.com/qaviinan/mortgage-pd-validation/actions/workflows/ci.yml) [![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) ![python 3.12](https://img.shields.io/badge/python-3.12-blue) ![SR 26-2](https://img.shields.io/badge/guidance-SR%2026--2%20(2026)-6f42c1)

Build a 36-month probability-of-default model on Freddie Mac single-family loan-level data, then validate it the way a bank model-risk team would under [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) (April 2026). The validation report is the deliverable.

**Deliverables:** [validation report (PDF)](report/report.pdf) · [one-page executive summary (PDF)](report/executive_summary.pdf) · [findings register](outputs/findings.csv) · [validation tables](outputs/validation/)

![Rank-ordering survives every regime; the PD level does not.](report/figures/fig1_timeline.png)

## What it does

- **Data.** 14 vintages (2006–2019) of Freddie Mac's public 50,000-loan-per-year sample: 700,000 loans, 44.7M loan-months, layout verified programmatically against Freddie's July 2026 spreadsheet, every raw file SHA-256-hashed.
- **Target.** Ever 90+ days delinquent or credit-event termination within 36 months, with an explicit hardship guard (COVID forbearance / disaster relief) and full censoring logic, unit-tested on 15 hand-built loan histories.
- **Models.** FICO-only baseline; weight-of-evidence scorecard challenger (optbinning → logistic); LightGBM champion with monotone constraints, early-stopped on a time-ordered tune vintage. No resampling.
- **Validation battery** (`src/pdval/validation/`, pure functions, data-agnostic): AUC/Gini/KS with bootstrap CIs and DeLong tests; reliability, calibration slope/intercept, Brier, ECE; PSI/CSI by vintage; permutation importance, SHAP, partial dependence, one-at-a-time shocks, monotonicity grid test; geography-proxy fairness screen (ACS demographics by metro area); findings register with pre-registered thresholds.
- **Design.** Train 2009–2015 · tune 2016 · out-of-time 2017–2018 · stress backtest 2006–2008 · COVID exhibit 2019.

## Headline results

| | Champion | Scorecard | FICO-only |
|---|---|---|---|
| AUC, out of time 2017–18 | 0.795 | 0.790 | 0.739 |
| Observed / predicted PD, out of time | 0.96 | 0.88 | 1.13 |
| AUC, 2006–08 crisis vintages | 0.783 | 0.775 | 0.738 |
| Observed / predicted PD, crisis vintages | 1.9× | 2.3× | 5.0× |

Validation also caught that LightGBM 4.7's `monotone_constraints_method="advanced"` silently violates declared constraints (20% of loans on LTV); the champion was re-fit with the exact `intermediate` method. See finding F07 in the register.

## Reproduce

```bash
brew install uv libomp pandoc tectonic   # macOS
make test                                 # unit tests on synthetic fixtures, no data needed
make all                                  # data -> target -> features -> train -> validate -> figures -> report
```

`make all` needs the raw sample zips in `data/raw/` (see below) and takes ~25 minutes on a laptop. The fairness proxy section needs `CENSUS_API_KEY` in the environment (free key from [api.census.gov](https://api.census.gov/data/key_signup.html)); without it the battery records a Low finding and continues.

### Getting the data (one-time, ~15 minutes)

1. Register / sign in to **Clarity Data Intelligence** from the [dataset page](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset), open the **SFLLD Data** tab (not CRT Data).
2. Under **Sample**, download `sample_YYYY.zip` for 2006–2019 into `data/raw/`, unrenamed.
3. `docs/reference/` carries the Release 47 (July 2026) layout. If Clarity serves a newer release, replace `file_layout_*.xlsx`, update `configs/default.yaml → data.layout_xlsx` and `performance_cutoff`, and re-run `make test` (the layout test fails loudly if a column moved).

## Layout

```
configs/default.yaml        vintages, target definition, exclusions, pre-registered thresholds
src/pdval/layout.py         canonical 31/35-column layout, verified against Freddie's spreadsheet
src/pdval/ingest.py         zip -> DuckDB -> typed parquet, sha256 manifest, orphan checks
src/pdval/target.py         hardship-guarded 36-month target with censoring (single SQL)
src/pdval/features.py       model frame, exclusions, waterfall
src/pdval/models/classic.py FICO baseline + WoE scorecard
src/pdval/train.py          fits all three models, scores every loan, writes model card
src/pdval/validation/       discrimination, calibration, stability, sensitivity, fairness, findings
src/pdval/validate.py       runs the battery -> outputs/validation/*.csv, outputs/findings.csv
src/pdval/report/           figures.py (house style) and build.py (Markdown -> PDF; every number read from outputs/)
tests/                      target branches, layout drift, leakage, ingest round-trip, validation metrics
docs/reference/              Freddie file layout (July 2026)
```

Data and model binaries are not committed (`data/` is git-ignored except the manifest; `outputs/models/` is ignored). Freddie Mac's dataset is free for non-commercial and academic use under its own terms. Code is MIT-licensed.
