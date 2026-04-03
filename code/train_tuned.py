#!/usr/bin/env python3

import os
import json
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import xgboost as xgb


def safe_pearson(y_true, y_pred):
    try:
        return float(pearsonr(y_true, y_pred)[0])
    except Exception:
        return np.nan


def safe_spearman(y_true, y_pred):
    try:
        return float(spearmanr(y_true, y_pred)[0])
    except Exception:
        return np.nan


def compute_metrics(y_true, y_pred):
    return {
        "pearson": safe_pearson(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_arrays(feature_dir, feature_prefix):
    X_train = np.load(os.path.join(feature_dir, f"X_train_{feature_prefix}.npy"))
    X_val = np.load(os.path.join(feature_dir, f"X_val_{feature_prefix}.npy"))
    X_test = np.load(os.path.join(feature_dir, f"X_test_{feature_prefix}.npy"))

    y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
    y_val = np.load(os.path.join(feature_dir, "y_val.npy"))
    y_test = np.load(os.path.join(feature_dir, "y_test.npy"))

    return X_train, X_val, X_test, y_train, y_val, y_test


def maybe_standardize(model_name, X_train, X_val, X_test):
    need_scaling = model_name in {"ridge", "svr"}

    if not need_scaling:
        return X_train, X_val, X_test, None

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_val_s, X_test_s, scaler


def get_param_grid(model_name):
    if model_name == "ridge":
        return {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        }

    if model_name == "svr":
        return {
            "C": [0.1, 1.0, 10.0, 100.0],
            "epsilon": [0.01, 0.1, 0.5, 1.0],
            "gamma": ["scale", "auto"],
        }

    if model_name == "rf":
        return {
            "n_estimators": [100, 300, 500],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2, 4],
        }

    if model_name == "xgb":
        return {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

    raise ValueError(f"No tuning grid for model_name: {model_name}")


def iter_grid(grid):
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def build_model(model_name, params, seed):
    if model_name == "ridge":
        return Ridge(alpha=params["alpha"])

    if model_name == "svr":
        return SVR(
            kernel="rbf",
            C=params["C"],
            epsilon=params["epsilon"],
            gamma=params["gamma"],
        )

    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "xgb":
        return xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def save_predictions(out_dir, split_name, y_true, y_pred):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": y_pred - y_true,
        "abs_error": np.abs(y_pred - y_true),
    })
    df.to_csv(os.path.join(out_dir, f"{split_name}_predictions.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--feature_prefix", type=str, required=True, choices=["basic", "esm"])
    parser.add_argument("--model", type=str, required=True, choices=["ridge", "svr", "rf", "xgb"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection_metric", type=str, default="pearson",
                        choices=["pearson", "spearman", "mae", "rmse", "r2"])

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_arrays(args.feature_dir, args.feature_prefix)
    X_train_p, X_val_p, X_test_p, scaler = maybe_standardize(args.model, X_train, X_val, X_test)

    grid = get_param_grid(args.model)
    all_rows = []

    best_score = None
    best_params = None
    best_model = None
    best_val_metrics = None

    maximize = args.selection_metric in {"pearson", "spearman", "r2"}

    for idx, params in enumerate(iter_grid(grid), start=1):
        model = build_model(args.model, params, args.seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_p, y_train)

        val_pred = model.predict(X_val_p)
        val_metrics = compute_metrics(y_val, val_pred)

        row = {
            "trial": idx,
            "feature_prefix": args.feature_prefix,
            "model": args.model,
            "selection_metric": args.selection_metric,
            **params,
            "val_pearson": val_metrics["pearson"],
            "val_spearman": val_metrics["spearman"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
        }
        all_rows.append(row)

        score = val_metrics[args.selection_metric]

        if best_score is None:
            take = True
        else:
            take = score > best_score if maximize else score < best_score

        if take:
            best_score = score
            best_params = params
            best_model = model
            best_val_metrics = val_metrics

        print(f"Trial {idx}: params={params} | val_{args.selection_metric}={score}")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(os.path.join(args.out_dir, "all_trials.csv"), index=False)

    test_pred = best_model.predict(X_test_p)
    test_metrics = compute_metrics(y_test, test_pred)

    final_result = {
        "feature_prefix": args.feature_prefix,
        "model": args.model,
        "seed": args.seed,
        "selection_metric": args.selection_metric,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "input_dim": int(X_train.shape[1]),
        "best_params": best_params,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }

    with open(os.path.join(args.out_dir, "best_result.json"), "w") as f:
        json.dump(final_result, f, indent=2)

    summary_row = {
        "feature_prefix": args.feature_prefix,
        "model": args.model,
        "selection_metric": args.selection_metric,
        "input_dim": int(X_train.shape[1]),
        **best_params,
        "best_val_pearson": best_val_metrics["pearson"],
        "best_val_spearman": best_val_metrics["spearman"],
        "best_val_mae": best_val_metrics["mae"],
        "best_val_rmse": best_val_metrics["rmse"],
        "best_val_r2": best_val_metrics["r2"],
        "test_pearson": test_metrics["pearson"],
        "test_spearman": test_metrics["spearman"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }
    pd.DataFrame([summary_row]).to_csv(os.path.join(args.out_dir, "best_metrics.csv"), index=False)

    val_pred = best_model.predict(X_val_p)
    save_predictions(args.out_dir, "val", y_val, val_pred)
    save_predictions(args.out_dir, "test", y_test, test_pred)

    print("DONE")
    print(json.dumps(final_result, indent=2))


if __name__ == "__main__":
    main()
