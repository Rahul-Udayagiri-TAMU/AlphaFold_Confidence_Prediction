#!/usr/bin/env python3

import os
import json
import time
import argparse
import requests
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


def load_uniprot_ids(id_file: str, max_n: int = None):
    with open(id_file, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    ids = list(dict.fromkeys(ids))
    if max_n is not None:
        ids = ids[:max_n]
    return ids


def load_json_dict(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def save_json_dict(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def parse_afdb_entry(pid: str, payload):
    if payload is None:
        return None, "empty_payload"

    if isinstance(payload, list):
        if len(payload) == 0:
            return None, "empty_list_payload"
        entry = payload[0]
    elif isinstance(payload, dict):
        entry = payload
    else:
        return None, f"unexpected_payload_type:{type(payload).__name__}"

    if not isinstance(entry, dict):
        return None, f"unexpected_entry_type:{type(entry).__name__}"

    seq = entry.get("sequence")
    mean_plddt = entry.get("globalMetricValue")

    if not isinstance(seq, str) or len(seq.strip()) == 0:
        return None, f"missing_or_bad_sequence keys={list(entry.keys())[:20]}"

    try:
        mean_plddt = float(mean_plddt)
    except Exception:
        return None, f"missing_or_bad_globalMetricValue keys={list(entry.keys())[:20]}"

    return {
        "protein_id": pid,
        "sequence": seq.strip().upper(),
        "mean_plddt": mean_plddt,
        "model_entity_id": entry.get("modelEntityId"),
        "provider_id": entry.get("providerId"),
        "tool_used": entry.get("toolUsed"),
        "latest_version": entry.get("latestVersion"),
        "fraction_plddt_very_low": entry.get("fractionPlddtVeryLow"),
        "fraction_plddt_low": entry.get("fractionPlddtLow"),
        "fraction_plddt_confident": entry.get("fractionPlddtConfident"),
        "fraction_plddt_very_high": entry.get("fractionPlddtVeryHigh"),
    }, None


def fetch_afdb_entry(pid: str, session: requests.Session, debug: bool = False):
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{pid}"
    try:
        r = session.get(url, timeout=20)

        if debug:
            print(f"[DEBUG] {pid} -> status {r.status_code}")

        if r.status_code != 200:
            if debug:
                print(f"[DEBUG] {pid} response preview: {r.text[:300]}")
            return None, f"http_{r.status_code}"

        try:
            payload = r.json()
        except Exception as e:
            if debug:
                print(f"[DEBUG] {pid} json parse failed: {repr(e)}")
                print(f"[DEBUG] raw response preview: {r.text[:500]}")
            return None, f"json_parse_error:{repr(e)}"

        entry, err = parse_afdb_entry(pid, payload)
        if entry is None:
            if debug:
                print(f"[DEBUG] {pid} parse failure: {err}")
            return None, err

        return entry, None

    except requests.exceptions.Timeout:
        return None, "timeout"
    except requests.exceptions.ConnectionError as e:
        return None, f"connection_error:{repr(e)}"
    except Exception as e:
        return None, f"unexpected_error:{repr(e)}"


def fetch_and_cache(ids, cache_file, fail_log_file, sleep_sec=0.1, debug=False, refresh=False):
    cache = load_json_dict(cache_file)
    fail_log = load_json_dict(fail_log_file)

    session = requests.Session()
    session.headers.update({"User-Agent": "ECEN766-AFDB-Fetch/1.0"})

    fetched = 0
    reused = 0
    failed = 0

    for idx, pid in enumerate(ids, start=1):
        if (not refresh) and (pid in cache):
            reused += 1
        else:
            entry, err = fetch_afdb_entry(pid, session, debug=debug)
            if entry is not None:
                cache[pid] = entry
                fetched += 1
                if pid in fail_log:
                    del fail_log[pid]
            else:
                failed += 1
                fail_log[pid] = err

        if idx % 10 == 0 or idx == len(ids):
            print(f"Scanned {idx}/{len(ids)} | reused={reused} | fetched_new={fetched} | failed={failed}")
            save_json_dict(cache, cache_file)
            save_json_dict(fail_log, fail_log_file)

        time.sleep(sleep_sec)

    save_json_dict(cache, cache_file)
    save_json_dict(fail_log, fail_log_file)

    print(f"Cache ready | reused={reused} | fetched_new={fetched} | failed={failed}")
    return cache, fail_log


def build_dataframe(cache, ids):
    rows = []

    for pid in ids:
        if pid not in cache:
            continue

        entry = cache[pid]
        seq = entry.get("sequence")
        mean_plddt = entry.get("mean_plddt")

        if not isinstance(seq, str) or len(seq) == 0:
            continue

        try:
            mean_plddt = float(mean_plddt)
        except Exception:
            continue

        rows.append({
            "protein_id": pid,
            "sequence": seq,
            "seq_len": len(seq),
            "mean_plddt": mean_plddt,
            "model_entity_id": entry.get("model_entity_id"),
            "provider_id": entry.get("provider_id"),
            "tool_used": entry.get("tool_used"),
            "latest_version": entry.get("latest_version"),
            "fraction_plddt_very_low": entry.get("fraction_plddt_very_low"),
            "fraction_plddt_low": entry.get("fraction_plddt_low"),
            "fraction_plddt_confident": entry.get("fraction_plddt_confident"),
            "fraction_plddt_very_high": entry.get("fraction_plddt_very_high"),
        })

    return pd.DataFrame(rows)


def mean_pool(last_hidden_state, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def load_esm_model(esm_model_name: str, force_cpu: bool = False):
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device for ESM: {device}")

    tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
    model = AutoModel.from_pretrained(esm_model_name).to(device)
    model.eval()
    return tokenizer, model, device


def embed_sequences(seqs, tokenizer, model, device, batch_size=8, max_length=2048):
    import torch

    all_embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]

            toks = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            toks = {k: v.to(device) for k, v in toks.items()}

            out = model(**toks)
            pooled = mean_pool(out.last_hidden_state, toks["attention_mask"])
            all_embs.append(pooled.cpu().numpy().astype(np.float32))

            print(f"Embedded {min(i + batch_size, len(seqs))}/{len(seqs)}")

    return np.vstack(all_embs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--id_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)

    parser.add_argument("--max_proteins", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--esm_model", type=str, default="facebook/esm2_t6_8M_UR50D")

    parser.add_argument("--sleep_sec", type=float, default=0.1)
    parser.add_argument("--min_valid_proteins", type=int, default=10)

    parser.add_argument("--skip_embeddings", action="store_true")
    parser.add_argument("--debug_fetch", action="store_true")
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--max_length", type=int, default=2048)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    cache_file = os.path.join(args.cache_dir, "afdb_cache.json")
    fail_log_file = os.path.join(args.cache_dir, "afdb_fail_log.json")

    np.random.seed(args.seed)

    ids = load_uniprot_ids(args.id_file, args.max_proteins)
    print(f"Loaded {len(ids)} UniProt IDs")

    if len(ids) == 0:
        raise ValueError("No UniProt IDs found")

    cache, fail_log = fetch_and_cache(
        ids,
        cache_file=cache_file,
        fail_log_file=fail_log_file,
        sleep_sec=args.sleep_sec,
        debug=args.debug_fetch,
        refresh=args.refresh_cache
    )

    df = build_dataframe(cache, ids)
    print(f"Usable proteins: {len(df)}")

    if len(df) < args.min_valid_proteins:
        sample_failures = dict(list(fail_log.items())[:10])
        print("Sample failures:")
        print(json.dumps(sample_failures, indent=2))
        raise ValueError(f"Too few valid proteins collected: {len(df)} < {args.min_valid_proteins}")

    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=args.seed)

    print(f"Split sizes | train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_df.to_csv(os.path.join(args.out_dir, "train_metadata.csv"), index=False)
    val_df.to_csv(os.path.join(args.out_dir, "val_metadata.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test_metadata.csv"), index=False)

    X_train_basic = np.vstack(train_df["sequence"].apply(basic_features).values)
    X_val_basic = np.vstack(val_df["sequence"].apply(basic_features).values)
    X_test_basic = np.vstack(test_df["sequence"].apply(basic_features).values)

    y_train = train_df["mean_plddt"].to_numpy(dtype=np.float32)
    y_val = val_df["mean_plddt"].to_numpy(dtype=np.float32)
    y_test = test_df["mean_plddt"].to_numpy(dtype=np.float32)

    np.save(os.path.join(args.out_dir, "X_train_basic.npy"), X_train_basic)
    np.save(os.path.join(args.out_dir, "X_val_basic.npy"), X_val_basic)
    np.save(os.path.join(args.out_dir, "X_test_basic.npy"), X_test_basic)

    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test)

    summary = {
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "basic_dim": int(X_train_basic.shape[1]),
        "esm_model": None if args.skip_embeddings else args.esm_model,
        "esm_dim": None,
        "skip_embeddings": bool(args.skip_embeddings),
        "cache_file": cache_file,
        "fail_log_file": fail_log_file,
    }

    if not args.skip_embeddings:
        tokenizer, model, device = load_esm_model(args.esm_model, force_cpu=args.force_cpu)

        X_train_esm = embed_sequences(
            train_df["sequence"].tolist(), tokenizer, model, device,
            batch_size=args.batch_size, max_length=args.max_length
        )
        X_val_esm = embed_sequences(
            val_df["sequence"].tolist(), tokenizer, model, device,
            batch_size=args.batch_size, max_length=args.max_length
        )
        X_test_esm = embed_sequences(
            test_df["sequence"].tolist(), tokenizer, model, device,
            batch_size=args.batch_size, max_length=args.max_length
        )

        np.save(os.path.join(args.out_dir, "X_train_esm.npy"), X_train_esm)
        np.save(os.path.join(args.out_dir, "X_val_esm.npy"), X_val_esm)
        np.save(os.path.join(args.out_dir, "X_test_esm.npy"), X_test_esm)

        summary["esm_dim"] = int(X_train_esm.shape[1])

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("DONE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
