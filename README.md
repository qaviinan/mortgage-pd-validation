# Mortgage PD model and validation report

[![ci](https://github.com/qaviinan/mortgage-pd-validation/actions/workflows/ci.yml/badge.svg)](https://github.com/qaviinan/mortgage-pd-validation/actions/workflows/ci.yml) [![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A 36-month probability-of-default model built on Freddie Mac's loan-level data (14 vintages, 700k loans), and a validation report written to [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm), the 2026 model risk guidance that replaced SR 11-7.

[Report (PDF)](report/report.pdf) · [Executive summary](report/executive_summary.pdf) · [Findings](outputs/findings.csv)

| | LightGBM | WoE scorecard | FICO only |
|---|---|---|---|
| AUC, out of time (2017–18) | 0.795 | 0.790 | 0.739 |
| AUC, crisis vintages (2006–08) | 0.783 | 0.775 | 0.738 |
| Observed / predicted PD, crisis vintages | 1.9× | 2.3× | 5.0× |

Ranking holds up in the 2006–08 crisis, but the model under-predicts default rates there by about 2×. It has no macro inputs and never saw a downturn in training.

![AUC and default rate by vintage](report/figures/fig1_timeline.png)

- **Target:** 90+ days delinquent or a credit-event termination within 36 months. Delinquencies during or just after forbearance or disaster relief don't count; without that rule, 2017–19 default rates jump several-fold because of COVID. The definition is tested on 15 hand-built loan histories.
- **Models:** FICO-only logistic, a weight-of-evidence scorecard, and LightGBM with monotone constraints. Train 2009–15, tune 2016, test 2017–18, backtest 2006–08.
- **Validation:** bootstrap and DeLong tests on AUC, calibration by sample and vintage, PSI/CSI drift, permutation importance and SHAP, monotonicity grid tests, a geography-based fairness screen, and a findings register.

## Running it

```bash
brew install uv libomp pandoc tectonic   # macOS
make test   # no data needed
make all    # full pipeline, ~25 min
```

`make all` needs the Freddie Mac sample files. Sign in to [Clarity](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset), open the SFLLD Data tab, and download `sample_2006.zip` through `sample_2019.zip` into `data/raw/`. The fairness screen also needs a free `CENSUS_API_KEY`.

Code is in `src/pdval/`: `target.py` builds the default flag, `train.py` fits the models, `validation/` holds the tests, and `report/` builds the PDF from the tables in `outputs/`.
