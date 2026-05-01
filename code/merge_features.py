#!/usr/bin/env python3

import os
import argparse
import numpy as np


def load_arrays(feature_dir: str):
    X_train_basic = np.load(os.path.join(feature_dir, "X_train_basic.npy"))
    X_val_basic = np.load(os.path.join(feature_dir, "X_val_basic.npy"))
    X_test_basic = np.load(os.path.join(feature_dir, "X_test_basic.npy"))

    X_train_esm = np.load(os.path.join(feature_dir, "X_train_esm.npy"))
    X_val_esm = np.load(os.path.join(feature_dir, "X_val_esm.npy"))
    X_test_esm = np.load(os.path.join(feature_dir, "X_test_esm.npy"))

    y_train = np.load(os.path.join(feature_dir, "y_train.npy"))
    y_val = np.load(os.path.join(feature_dir, "y_val.npy"))
    y_test = np.load(os.path.join(feature_dir, "y_test.npy"))

    return (
        X_train_basic, X_val_basic, X_test_basic,
        X_train_esm, X_val_esm, X_test_esm,
        y_train, y_val, y_test
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    (
        X_train_basic, X_val_basic, X_test_basic,
        X_train_esm, X_val_esm, X_test_esm,
        y_train, y_val, y_test
    ) = load_arrays(args.feature_dir)

    if X_train_basic.shape[0] != X_train_esm.shape[0]:
        raise ValueError("Train row mismatch between basic and esm features")
    if X_val_basic.shape[0] != X_val_esm.shape[0]:
        raise ValueError("Val row mismatch between basic and esm features")
    if X_test_basic.shape[0] != X_test_esm.shape[0]:
        raise ValueError("Test row mismatch between basic and esm features")

    X_train_combined = np.concatenate([X_train_basic, X_train_esm], axis=1)
    X_val_combined = np.concatenate([X_val_basic, X_val_esm], axis=1)
    X_test_combined = np.concatenate([X_test_basic, X_test_esm], axis=1)

    np.save(os.path.join(args.out_dir, "X_train_combined.npy"), X_train_combined)
    np.save(os.path.join(args.out_dir, "X_val_combined.npy"), X_val_combined)
    np.save(os.path.join(args.out_dir, "X_test_combined.npy"), X_test_combined)

    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test)

    print("DONE")
    print("Train combined shape:", X_train_combined.shape)
    print("Val combined shape  :", X_val_combined.shape)
    print("Test combined shape :", X_test_combined.shape)


if __name__ == "__main__":
    main()

