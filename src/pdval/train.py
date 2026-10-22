"""Fit the three models and score every loan in the model frame.

Usage:
    uv run python -m pdval.train --config configs/default.yaml

Models
    baseline    logistic regression on Classic FICO only (+ missing indicator)
    challenger  WoE scorecard: optbinning monotonic bins -> L2 logistic regression
    champion    LightGBM with monotone constraints, early-stopped on the tune vintage,
                small random search, no class re-weighting

Outputs
    data/processed/predictions.parquet   loan_id, vintage, role, y, p_baseline, p_challenger, p_champion
    outputs/models/*.pkl / champion.txt   fitted artifacts
    outputs/model_card.json              hyperparameters, search history, dev metrics
    outputs/dev_metrics.json             AUC / logloss by role for each model
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss, roc_auc_score

from pdval.models.classic import FicoBaseline, WoeScorecard


# ----------------------------------------------------------------------------- champion
def lgb_dataset(df: pd.DataFrame, y: np.ndarray | None, cols: list[str], cats: list[str], reference=None):
    x = df[cols]
    return lgb.Dataset(x, label=y, categorical_feature=cats, free_raw_data=False, reference=reference)


def search_space(rng: np.random.Generator, n: int) -> list[dict]:
    out = []
    for _ in range(n):
        out.append({
            "num_leaves": int(rng.choice([7, 15, 31, 63])),
            "min_child_samples": int(rng.choice([100, 250, 500, 1000])),
            "learning_rate": float(rng.choice([0.02, 0.03, 0.05])),
            "feature_fraction": float(rng.choice([0.6, 0.8, 1.0])),
            "bagging_fraction": float(rng.choice([0.7, 0.9, 1.0])),
            "lambda_l2": float(rng.choice([0.0, 1.0, 10.0])),
            "max_bin": int(rng.choice([63, 255])),
        })
    return out


def fit_champion(train: pd.DataFrame, tune: pd.DataFrame, cols: list[str], cats: list[str],
                 monotone: dict[str, int], seed: int, n_configs: int = 24) -> tuple[lgb.Booster, dict, list[dict]]:
    rng = np.random.default_rng(seed)
    dtrain = lgb_dataset(train, train["y"].to_numpy(), cols, cats)
    dtune = lgb_dataset(tune, tune["y"].to_numpy(), cols, cats, reference=dtrain)
    mono = [monotone.get(c, 0) for c in cols]
    base = {
        "objective": "binary", "metric": "auc", "verbosity": -1, "seed": seed, "bagging_freq": 1,
        # "advanced" breaks the constraints in LightGBM 4.7; see outputs/validation/monotone_methods.csv
        "monotone_constraints": mono, "monotone_constraints_method": "intermediate", "num_threads": 8,
        "cat_smooth": 20, "min_data_per_group": 200,
    }
    history, best = [], None
    for i, hp in enumerate(search_space(rng, n_configs)):
        params = {**base, **hp}
        evals: dict = {}
        booster = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dtune], valid_names=["tune"],
                            callbacks=[lgb.early_stopping(100, verbose=False), lgb.record_evaluation(evals)])
        auc = float(booster.best_score["tune"]["auc"])
        rec = {"config": i, **hp, "best_iteration": booster.best_iteration, "tune_auc": auc}
        history.append(rec)
        if best is None or auc > best[1]:
            best = (booster, auc, params, booster.best_iteration)
        print(f"  cfg {i:2d}: leaves={hp['num_leaves']:>2} mcs={hp['min_child_samples']:>4} lr={hp['learning_rate']} "
              f"ff={hp['feature_fraction']} l2={hp['lambda_l2']:>4} -> iters={booster.best_iteration:>4} tune AUC={auc:.4f}")
    booster, auc, params, iters = best
    return booster, {"params": params, "best_iteration": iters, "tune_auc": auc}, history


# ----------------------------------------------------------------------------- main
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"n": len(y), "defaults": int(y.sum()), "auc": float(roc_auc_score(y, p)),
            "logloss": float(log_loss(y, p)), "mean_pred": float(p.mean()), "obs_rate": float(y.mean())}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n-configs", type=int, default=24)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg["seed"])
    np.random.seed(seed)
    processed = Path(cfg["data"]["processed_dir"])
    frame = pd.read_parquet(processed / "model_frame.parquet")
    num, cat = cfg["features"]["numeric"], cfg["features"]["categorical"]
    cols = num + cat
    mono = cfg["features"]["monotone"]

    train = frame[frame["role"] == "train"].reset_index(drop=True)
    tune = frame[frame["role"] == "tune"].reset_index(drop=True)
    ytr = train["y"].to_numpy()
    print(f"train {len(train):,} loans / {ytr.sum():,} defaults; tune {len(tune):,} / {tune['y'].sum():,}")

    print("\n[baseline] FICO-only logistic")
    baseline = FicoBaseline().fit(train, ytr)

    print("[challenger] WoE scorecard")
    challenger = WoeScorecard(num, cat, mono).fit(train, ytr)
    iv = challenger.iv_table()
    print(iv.to_string(index=False))

    print(f"\n[champion] LightGBM random search ({args.n_configs} configs, early stop on tune AUC)")
    champion, champ_info, history = fit_champion(train, tune, cols, cat, mono, seed, args.n_configs)
    print(f"  best: iters={champ_info['best_iteration']} tune AUC={champ_info['tune_auc']:.4f}")

    # Score everything
    pred = frame[["loan_id", "vintage", "role", "y"]].copy()
    pred["p_baseline"] = baseline.predict_proba(frame)
    pred["p_challenger"] = challenger.predict_proba(frame)
    pred["p_champion"] = champion.predict(frame[cols], num_iteration=champion.best_iteration)
    pred.to_parquet(processed / "predictions.parquet", index=False)

    dev = {}
    for model in ("baseline", "challenger", "champion"):
        dev[model] = {}
        for role, g in pred.groupby("role"):
            dev[model][role] = metrics(g["y"].to_numpy(), g[f"p_{model}"].to_numpy())
    out = Path("outputs"); (out / "models").mkdir(parents=True, exist_ok=True)
    with open(out / "models/baseline.pkl", "wb") as f:
        pickle.dump(baseline, f)
    with open(out / "models/challenger.pkl", "wb") as f:
        pickle.dump(challenger, f)
    champion.save_model(str(out / "models/champion.txt"), num_iteration=champion.best_iteration)
    (out / "dev_metrics.json").write_text(json.dumps(dev, indent=2))
    (out / "model_card.json").write_text(json.dumps({
        "features": {"numeric": num, "categorical": cat, "monotone": mono},
        "train_vintages": cfg["vintages"]["train"], "tune_vintages": cfg["vintages"]["tune"],
        "train_rows": len(train), "train_defaults": int(ytr.sum()),
        "challenger": {"binning": "optbinning BinningProcess, monotonic trends per config, max 8 bins",
                        "iv": iv.to_dict("records"), "C": 1.0},
        "champion": champ_info, "champion_search": history, "seed": seed,
    }, indent=2, default=str))

    print("\nAUC by role:")
    rows = []
    for model in dev:
        for role in ("train", "tune", "oot", "stress", "covid_exhibit"):
            m = dev[model][role]
            rows.append({"model": model, "role": role, "n": m["n"], "defaults": m["defaults"], "auc": round(m["auc"], 4),
                         "mean_pred": round(m["mean_pred"], 4), "obs_rate": round(m["obs_rate"], 4)})
    print(pd.DataFrame(rows).pivot(index="role", columns="model", values="auc").loc[["train", "tune", "oot", "stress", "covid_exhibit"]].to_string())
    print("\ncalibration-in-the-large (mean predicted vs observed):")
    print(pd.DataFrame(rows).pivot(index="role", columns="model", values="mean_pred").join(
        pd.DataFrame(rows).drop_duplicates("role").set_index("role")["obs_rate"]).loc[["train", "tune", "oot", "stress", "covid_exhibit"]].to_string())


if __name__ == "__main__":
    main()
