#!/usr/bin/env python3

import os
import json
import time
import math
import argparse
import requests
import pandas as pd


def load_uniprot_ids(id_file: str, max_n: int | None = None):
    with open(id_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    ids = list(dict.fromkeys(ids))
    if max_n is not None:
        ids = ids[:max_n]
    return ids


def fetch_prediction_entry(pid: str, session: requests.Session):
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{pid}"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    entry = payload[0] if isinstance(payload, list) else payload
    return entry


def fetch_plddt_json(plddt_url: str, session: requests.Session):
    r = session.get(plddt_url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected pLDDT JSON type")
    if "confidenceScore" not in data:
        raise ValueError("confidenceScore missing from pLDDT JSON")
    return data


def make_windows(seq: str, scores: list[float], window_size: int, stride: int):
    rows = []
    L = len(seq)
    if L != len(scores):
        raise ValueError(f"Length mismatch: seq={L}, scores={len(scores)}")

    if L < window_size:
        return rows

    window_id = 0
    for start in range(0, L - window_size + 1, stride):
        end = start + window_size
        window_scores = scores[start:end]
        window_seq = seq[start:end]
        rows.append({
            "window_id": window_id,
            "window_start_1based": start + 1,
            "window_end_1based": end,
            "window_len": window_size,
            "window_seq": window_seq,
            "window_mean_plddt": float(sum(window_scores) / len(window_scores)),
        })
        window_id += 1

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id_file", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--max_proteins", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--sleep_sec", type=float, default=0.05)
    args = parser.parse_args()

    ids = load_uniprot_ids(args.id_file, args.max_proteins)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "ECEN766-window-label-builder/1.0"})

    rows = []
    success = 0
    failed = 0
    skipped_short = 0

    for idx, pid in enumerate(ids, start=1):
        try:
            entry = fetch_prediction_entry(pid, session)
            seq = entry.get("sequence") or entry.get("uniprotSequence")
            plddt_url = entry.get("plddtDocUrl")

            if not seq or not plddt_url:
                failed += 1
                continue

            plddt_data = fetch_plddt_json(plddt_url, session)
            scores = plddt_data["confidenceScore"]

            windows = make_windows(seq, scores, args.window_size, args.stride)
            if len(windows) == 0:
                skipped_short += 1
                continue

            for w in windows:
                rows.append({
                    "protein_id": pid,
                    "protein_len": len(seq),
                    **w
                })

            success += 1

        except Exception:
            failed += 1

        if idx % 10 == 0 or idx == len(ids):
            print(
                f"Processed {idx}/{len(ids)} | "
                f"success={success} | failed={failed} | skipped_short={skipped_short} | "
                f"windows={len(rows)}"
            )

        time.sleep(args.sleep_sec)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print("DONE")
    print(f"Output CSV: {args.out_csv}")
    print(f"Proteins with windows: {success}")
    print(f"Failed proteins: {failed}")
    print(f"Skipped short proteins: {skipped_short}")
    print(f"Total windows: {len(df)}")


if __name__ == "__main__":
    main()
