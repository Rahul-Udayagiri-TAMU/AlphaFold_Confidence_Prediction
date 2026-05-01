#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--predictions_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
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

    # Predicted vs true
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(df["y_true"], df["y_pred"], alpha=0.5, s=14)
    lo = min(df["y_true"].min(), df["y_pred"].min())
    hi = max(df["y_true"].max(), df["y_pred"].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True confidence")
    plt.ylabel("Predicted confidence")
    plt.title(f"Predicted vs True ({args.tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.tag}_pred_vs_true.png"), dpi=200)
    plt.close()

    # Residual vs true
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(df["y_true"], df["residual"], alpha=0.5, s=14)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("True confidence")
    plt.ylabel("Residual (pred - true)")
    plt.title(f"Residual vs True ({args.tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.tag}_residual_vs_true.png"), dpi=200)
    plt.close()

    # Absolute error vs sequence length
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(df["seq_len"], df["abs_error"], alpha=0.5, s=14)
    plt.xlabel("Sequence length")
    plt.ylabel("Absolute error")
    plt.title(f"Absolute Error vs Length ({args.tag})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.tag}_abs_error_vs_length.png"), dpi=200)
    plt.close()

    print("Wrote:")
    print(os.path.join(args.out_dir, f"{args.tag}_pred_vs_true.png"))
    print(os.path.join(args.out_dir, f"{args.tag}_residual_vs_true.png"))
    print(os.path.join(args.out_dir, f"{args.tag}_abs_error_vs_length.png"))


if __name__ == "__main__":
    main()
