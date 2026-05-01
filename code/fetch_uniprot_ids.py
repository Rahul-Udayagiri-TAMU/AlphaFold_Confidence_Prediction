#!/usr/bin/env python3

import argparse
import requests


BASE_URL = "https://rest.uniprot.org/uniprotkb/search"


def fetch_ids_reviewed_human(n_ids: int, page_size: int = 500):
    if page_size > 500:
        page_size = 500

    query = "reviewed:true AND organism_id:9606"
    fields = "accession"

    ids = []
    cursor = None

    while len(ids) < n_ids:
        params = {
            "query": query,
            "fields": fields,
            "format": "tsv",
            "size": page_size,
        }
        if cursor is not None:
            params["cursor"] = cursor

        r = requests.get(BASE_URL, params=params, timeout=60)
        r.raise_for_status()

        lines = r.text.strip().splitlines()
        if len(lines) <= 1:
            break

        batch = [line.strip() for line in lines[1:] if line.strip()]
        ids.extend(batch)

        next_link = r.links.get("next")
        if not next_link:
            break

        next_url = next_link["url"]
        if "cursor=" in next_url:
            cursor = next_url.split("cursor=", 1)[1].split("&", 1)[0]
        else:
            break

    ids = list(dict.fromkeys(ids))
    return ids[:n_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--n_ids", type=int, default=1000)
    args = parser.parse_args()

    ids = fetch_ids_reviewed_human(args.n_ids)

    with open(args.out_file, "w") as f:
        for pid in ids:
            f.write(pid + "\n")

    print(f"Wrote {len(ids)} IDs to {args.out_file}")


if __name__ == "__main__":
    main()
