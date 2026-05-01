#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

AA = "ACDEFGHIKLMNPQRSTVWY"

def basic_features(seq: str) -> np.ndarray:
    seq = seq.strip().upper()
    L = len(seq)
    if L == 0:
        return np.zeros(len(AA) + 9, dtype=np.float32)

    aa_comp = [seq.count(a) / L for a in AA]

    hydrophobic = set("AVILMFWY")
    polar = set("STNQCY")
    positive = set("KRH")
    negative = set("DE")
    aromatic = set("FWYH")
    sulfur = set("CM")
    small = set("AGSTPV")
    charged = positive | negative

    def frac(group):
        return sum(1 for x in seq if x in group) / L

    extra = [
        L,
        frac(hydrophobic),
        frac(polar),
        frac(charged),
        frac(positive),
        frac(negative),
        frac(aromatic),
        frac(sulfur),
        frac(small),
    ]

    return np.array(aa_comp + extra, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.window_csv)
    if len(df) == 0:
        raise ValueError("Window CSV is empty")

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=args.seed)

    train_df.to_csv(os.path.join(args.out_dir, "train_metadata.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "val_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test_metadata.csv"), index=False)

    X_train_basic = np.vstack(train_df["window_seq"].apply(basic_features).values)
    X_val_basic = np.vstack(val_df["window_seq"].apply(basic_features).values)
    X_test_basic = np.vstack(test_df["window_seq"].apply(basic_features).values)

    y_train = train_df["window_mean_plddt"].to_numpy(dtype=np.float32)
    y_val = val_df["window_mean_plddt"].to_numpy(dtype=np.float32)
    y_test = test_df["window_mean_plddt"].to_numpy(dtype=np.float32)

    np.save(os.path.join(args.out_dir, "X_train_basic.npy"), X_train_basic)
    np.save(os.path.join(args.out_dir, "X_val_basic.npy"), X_val_basic)
    np.save(os.path.join(args.out_dir, "X_test_basic.npy"), X_test_basic)

    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test)

    print("DONE")
    print(f"Train windows: {len(train_df)}")
    print(f"Val windows  : {len(val_df)}")
    print(f"Test windows : {len(test_df)}")
    print(f"Basic dim    : {X_train_basic.shape[1]}")


if __name__ == "__main__":
    main()
