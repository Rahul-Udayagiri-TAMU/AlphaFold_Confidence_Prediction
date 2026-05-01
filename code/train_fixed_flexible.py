#!/usr/bin/env python3

import os
import json
import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--feature_prefix", type=str, required=True, choices=["basic", "esm", "combined"])
    parser.add_argument("--model", type=str, required=True, choices=["ridge", "svr", "rf", "xgb"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--alpha", type=float, default=None)

    parser.add_argument("--C", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--gamma", type=str, default=None)

    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--max_depth", type=str, default=None)
    parser.add_argument("--min_samples_split", type=int, default=None)
    parser.add_argument("--min_samples_leaf", type=int, default=None)

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--colsample_bytree", type=float, default=None)

    return parser.parse_args()


def collect_params(args):
    if args.model == "ridge":
        if args.alpha is None:
            raise ValueError("ridge requires --alpha")
        return {"alpha": args.alpha}

    if args.model == "svr":
        required = [args.C, args.epsilon, args.gamma]
        if any(x is None for x in required):
            raise ValueError("svr requires --C --epsilon --gamma")
        return {
            "C": args.C,
            "epsilon": args.epsilon,
            "gamma": args.gamma,
        }

    if args.model == "rf":
        required = [args.n_estimators, args.max_depth, args.min_samples_split, args.min_samples_leaf]
        if any(x is None for x in required):
            raise ValueError("rf requires --n_estimators --max_depth --min_samples_split --min_samples_leaf")

        max_depth = None if str(args.max_depth).lower() == "none" else int(args.max_depth)

        return {
            "n_estimators": args.n_estimators,
            "max_depth": max_depth,
            "min_samples_split": args.min_samples_split,
            "min_samples_leaf": args.min_samples_leaf,
        }

    if args.model == "xgb":
        required = [args.n_estimators, args.max_depth, args.learning_rate, args.subsample, args.colsample_bytree]
        if any(x is None for x in required):
            raise ValueError("xgb requires --n_estimators --max_depth --learning_rate --subsample --colsample_bytree")

        return {
            "n_estimators": args.n_estimators,
            "max_depth": int(args.max_depth),
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        }

    raise ValueError(f"Unknown model: {args.model}")


def save_predictions(out_dir, split_name, y_true, y_pred):
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "residual": y_pred - y_true,
        "abs_error": np.abs(y_pred - y_true),
    })
    df.to_csv(os.path.join(out_dir, f"{split_name}_predictions.csv"), index=False)


def main():
    args = parse_args()
    params = collect_params(args)

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_arrays(args.feature_dir, args.feature_prefix)
    X_train_p, X_val_p, X_test_p, scaler = maybe_standardize(args.model, X_train, X_val, X_test)

    model = build_model(args.model, params, args.seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train_p, y_train)

    val_pred = model.predict(X_val_p)
    test_pred = model.predict(X_test_p)

    val_metrics = compute_metrics(y_val, val_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    result = {
        "feature_prefix": args.feature_prefix,
        "model": args.model,
        "seed": args.seed,
        "input_dim": int(X_train.shape[1]),
        "params": params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    row = {
        "feature_prefix": args.feature_prefix,
        "model": args.model,
        "seed": args.seed,
        "input_dim": int(X_train.shape[1]),
        **params,
        "val_pearson": val_metrics["pearson"],
        "val_spearman": val_metrics["spearman"],
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "test_pearson": test_metrics["pearson"],
        "test_spearman": test_metrics["spearman"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }
    pd.DataFrame([row]).to_csv(os.path.join(args.out_dir, "metrics.csv"), index=False)

    save_predictions(args.out_dir, "val", y_val, val_pred)
    save_predictions(args.out_dir, "test", y_test, test_pred)

    print("DONE")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

