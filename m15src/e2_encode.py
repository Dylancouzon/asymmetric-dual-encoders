"""E2 preparation on the Mac MPS, one detached job: parity, NFCorpus (for E4), MS MARCO 1M subset.

Method: `m15/MEASUREMENTS.md`, E2 "Data" and "Encoding". Resumable: finished shards are kept.
"""
import json
import sys
import time

import numpy as np

from common import REPO, SEED, load_public, sha_file
import vectors15 as V

WORK = REPO / "work" / "m15"
IDS = WORK / "msmarco1m_ids.json"
N_SUBSET = 1_000_000
MSMARCO_REV = "a918e0d11a77ed33f42f29d98340b655593b96ad"
QRELS_REV = "253fbf8a3f8d4a0932b63882b5162bedc84779f5"


def build_ids():
    """Every dev-judged positive plus a seeded uniform fill to exactly 1,000,000 passages."""
    from datasets import load_dataset
    corpus = load_dataset("BeIR/msmarco", "corpus", revision=MSMARCO_REV)["corpus"]
    qrels = load_dataset("BeIR/msmarco-qrels", revision=QRELS_REV, split="validation")
    positives = {str(r["corpus-id"]) for r in qrels if int(r["score"]) > 0}
    all_ids = [str(x) for x in corpus["_id"]]
    pos_idx = np.array(sorted(i for i, d in enumerate(all_ids) if d in positives))
    if len(pos_idx) != len(positives):
        raise RuntimeError("some judged positives are missing from the corpus")
    rest = np.setdiff1d(np.arange(len(all_ids)), pos_idx)
    rng = np.random.default_rng(SEED)
    fill = rng.choice(rest, N_SUBSET - len(pos_idx), replace=False)
    idx = np.sort(np.concatenate([pos_idx, fill]))
    IDS.write_text(json.dumps({"seed": SEED, "n_positives": int(len(pos_idx)),
                               "corpus_revision": MSMARCO_REV, "qrels_revision": QRELS_REV,
                               "corpus_rows": idx.tolist(),
                               "doc_ids": [all_ids[i] for i in idx]}))
    print(f"subset: {len(pos_idx)} positives + {len(fill)} fill, sha {sha_file(IDS)}", flush=True)


def mps_parity():
    data = load_public("fiqa")
    local, _ = V.doc_vectors("fiqa", data)
    rng = np.random.default_rng(SEED)
    pick = np.sort(rng.choice(len(data["doc_ids"]), 256, replace=False))
    got = V.encode_docs([data["doc_texts"][i] for i in pick], WORK / "vecs" / "fiqa-parity256",
                        shard=256)
    cos = (got.astype(np.float32) * local[pick].astype(np.float32)).sum(1)
    blob = {"n": 256, "min_cos": float(cos.min()), "passed": bool(cos.min() >= 0.9999)}
    (WORK / "e2_mps_parity.json").write_text(json.dumps(blob))
    print("MPS parity", blob, flush=True)
    if not blob["passed"]:
        raise SystemExit("MPS parity failed")


def encode_subset():
    from datasets import load_dataset
    rows = json.loads(IDS.read_text())["corpus_rows"]
    corpus = load_dataset("BeIR/msmarco", "corpus", revision=MSMARCO_REV)["corpus"]
    sub = corpus.select(rows)
    texts = [f"{(t or '').strip()} {(x or '').strip()}".strip()
             for t, x in zip(sub["title"], sub["text"])]
    t0 = time.time()
    V.encode_docs(texts, WORK / "vecs" / "msmarco1m", shard=20_000)
    print(f"msmarco1m encoded in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    steps = sys.argv[1:] or ["parity", "nfcorpus", "ids", "encode"]
    if "parity" in steps:
        mps_parity()
    if "nfcorpus" in steps:
        data = load_public("nfcorpus")
        V.doc_vectors("nfcorpus", data)
        print("nfcorpus encoded", flush=True)
    if "ids" in steps and not IDS.exists():
        build_ids()
    if "encode" in steps:
        encode_subset()
