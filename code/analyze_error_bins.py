#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def compute_metrics(df):
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    return {
        "count": int(len(df)),
        "pearson": safe_pearson(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def add_quantile_bins(df, source_col, bin_col, n_bins, labels):
    ranked = df[source_col].rank(method="first")
    df[bin_col] = pd.qcut(ranked, q=n_bins, labels=labels)
    return df


def summarize_by_bin(df, bin_col):
    rows = []
    for bin_name, group in df.groupby(bin_col, observed=False):
        metrics = compute_metrics(group)
        rows.append({"bin": str(bin_name), **metrics})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    meta = pd.read_csv(args.metadata_csv)
    pred = pd.read_csv(args.predictions_csv)

    if len(meta) != len(pred):
        raise ValueError(f"Row mismatch: metadata={len(meta)} predictions={len(pred)}")

    df = meta.copy()
    df["y_true"] = pred["y_true"].to_numpy()
    df["y_pred"] = pred["y_pred"].to_numpy()
    df["residual"] = pred["residual"].to_numpy()
    df["abs_error"] = pred["abs_error"].to_numpy()

    df = add_quantile_bins(df, "seq_len", "length_bin", 3, ["short", "medium", "long"])
    df = add_quantile_bins(df, "mean_plddt", "confidence_bin", 3, ["low", "medium", "high"])

    overall = pd.DataFrame([compute_metrics(df)])
    by_length = summarize_by_bin(df, "length_bin")
    by_conf = summarize_by_bin(df, "confidence_bin")

    df.to_csv(os.path.join(args.out_dir, "joined_test_analysis.csv"), index=False)
    overall.to_csv(os.path.join(args.out_dir, "overall_metrics.csv"), index=False)
    by_length.to_csv(os.path.join(args.out_dir, "metrics_by_length_bin.csv"), index=False)
    by_conf.to_csv(os.path.join(args.out_dir, "metrics_by_confidence_bin.csv"), index=False)

    print("Wrote:")
    print(os.path.join(args.out_dir, "joined_test_analysis.csv"))
    print(os.path.join(args.out_dir, "overall_metrics.csv"))
    print(os.path.join(args.out_dir, "metrics_by_length_bin.csv"))
    print(os.path.join(args.out_dir, "metrics_by_confidence_bin.csv"))
    print("\nBy length bin:")
    print(by_length.to_string(index=False))
    print("\nBy confidence bin:")
    print(by_conf.to_string(index=False))


if __name__ == "__main__":
    main()
