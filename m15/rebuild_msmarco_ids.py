"""Rebuild the recorded MS MARCO diagnostic's passage IDs from cached public inputs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


CORPUS_REVISION = "a918e0d11a77ed33f42f29d98340b655593b96ad"
QRELS_REVISION = "253fbf8a3f8d4a0932b63882b5162bedc84779f5"
EXPECTED_SHA256 = "4a09d3d055f13ae8cfd3f3a9202574af0f084954cd51f482918ce31aaee181a3"
SEED = 20260930
SUBSET_SIZE = 1_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new JSON file to create")
    parser.add_argument("--hub-cache", type=Path, default=Path.home() / ".cache/huggingface/hub")
    args = parser.parse_args()

    corpus = args.hub_cache / "datasets--BeIR--msmarco/snapshots" / CORPUS_REVISION
    judgments = args.hub_cache / "datasets--BeIR--msmarco-qrels/snapshots" / QRELS_REVISION
    with (judgments / "dev.tsv").open(encoding="utf-8", newline="") as stream:
        positives = {row["corpus-id"] for row in csv.DictReader(stream, delimiter="\t")
                     if int(row["score"]) > 0}
    ids = pq.read_table(corpus / "corpus/corpus-00000-of-00001.parquet", columns=["_id"])
    all_ids = [str(value) for value in ids["_id"].to_pylist()]
    positive_rows = np.array([index for index, doc_id in enumerate(all_ids) if doc_id in positives])
    if len(positive_rows) != len(positives):
        raise ValueError("Judged positives do not match the corpus.")

    remaining = np.setdiff1d(np.arange(len(all_ids)), positive_rows)
    rng = np.random.default_rng(SEED)
    fill = rng.choice(remaining, SUBSET_SIZE - len(positive_rows), replace=False)
    selected = np.sort(np.concatenate([positive_rows, fill])).tolist()
    payload = json.dumps({
        "seed": SEED,
        "n_positives": len(positive_rows),
        "corpus_revision": CORPUS_REVISION,
        "qrels_revision": QRELS_REVISION,
        "corpus_rows": selected,
        "doc_ids": [all_ids[index] for index in selected],
    }).encode("utf-8")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != EXPECTED_SHA256:
        raise SystemExit(f"Hash mismatch: got {actual}. No file written.")
    with args.output.open("xb") as stream:
        stream.write(payload)
    print(f"{actual}  {args.output}")


if __name__ == "__main__":
    main()
