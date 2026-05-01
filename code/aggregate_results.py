#!/usr/bin/env python3

import os
import argparse
import pandas as pd

DEFAULT_FILES = [
    ("10k_basic_rf", "outputs/final_10000/basic_rf/metrics.csv"),
    ("10k_basic_xgb", "outputs/final_10000/basic_xgb/metrics.csv"),
    ("10k_esm_small_ridge", "outputs/final_10000/esm_ridge/metrics.csv"),
    ("10k_esm_small_svr", "outputs/final_10000/esm_svr/metrics.csv"),
    ("10k_esm_small_xgb", "outputs/final_10000/esm_xgb/metrics.csv"),
    ("10k_combined_small_ridge", "outputs/final_10000/combined_ridge/metrics.csv"),
    ("10k_combined_small_svr", "outputs/final_10000/combined_svr/metrics.csv"),
    ("10k_combined_small_xgb", "outputs/final_10000/combined_xgb/metrics.csv"),
    ("2k_esm_large_ridge", "outputs/final_esm_large_2000/esm_ridge/metrics.csv"),
    ("2k_esm_large_svr", "outputs/final_esm_large_2000/esm_svr/metrics.csv"),
    ("2k_esm_large_xgb", "outputs/final_esm_large_2000/esm_xgb/metrics.csv"),
    ("10k_esm_large_ridge", "outputs/final_esm_large_10000/esm_ridge/metrics.csv"),
    ("10k_esm_large_svr", "outputs/final_esm_large_10000/esm_svr/metrics.csv"),
    ("10k_esm_large_xgb", "outputs/final_esm_large_10000/esm_xgb/metrics.csv"),
    ("10k_combined_large_ridge", "outputs/final_esm_large_10000/combined_ridge/metrics.csv"),
    ("10k_combined_large_svr", "outputs/final_esm_large_10000/combined_svr/metrics.csv"),
    ("10k_combined_large_xgb", "outputs/final_esm_large_10000/combined_xgb/metrics.csv"),
]

def infer_fields(run_name: str):
    dataset_scale = "unknown"
    if run_name.startswith("10k_"):
        dataset_scale = "10k"
    elif run_name.startswith("2k_"):
        dataset_scale = "2k"

    feature_family = "unknown"
    if "basic" in run_name and "combined" not in run_name:
        feature_family = "basic"
    elif "combined_small" in run_name:
        feature_family = "basic+esm_small"
    elif "combined_large" in run_name:
        feature_family = "basic+esm_large"
    elif "esm_small" in run_name:
        feature_family = "esm_small"
    elif "esm_large" in run_name:
        feature_family = "esm_large"

    return dataset_scale, feature_family

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    rows = []

    for run_name, rel_path in DEFAULT_FILES:
        path = os.path.join(args.project_root, rel_path)
        if not os.path.exists(path):
            print(f"SKIP missing: {path}")
            continue

        df = pd.read_csv(path)
        if len(df) != 1:
            raise ValueError(f"Expected one-row metrics file: {path}")

        row = df.iloc[0].to_dict()
        dataset_scale, feature_family = infer_fields(run_name)

        row["run_name"] = run_name
        row["dataset_scale"] = dataset_scale
        row["feature_family"] = feature_family
        row["source_metrics_path"] = rel_path

        rows.append(row)

    if not rows:
        raise ValueError("No metrics files found")

    out_df = pd.DataFrame(rows)

    preferred_cols = [
        "run_name",
        "dataset_scale",
        "feature_family",
        "feature_prefix",
        "model",
        "input_dim",
        "test_pearson",
        "test_spearman",
        "test_mae",
        "test_rmse",
        "test_r2",
        "val_pearson",
        "val_spearman",
        "val_mae",
        "val_rmse",
        "val_r2",
        "source_metrics_path",
    ]

    ordered_cols = [c for c in preferred_cols if c in out_df.columns] + [c for c in out_df.columns if c not in preferred_cols]
    out_df = out_df[ordered_cols]

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")
    print(out_df[["run_name", "test_pearson", "test_spearman", "test_mae", "test_rmse", "test_r2"]].sort_values("test_pearson", ascending=False).to_string(index=False))

if __name__ == "__main__":
    main()
