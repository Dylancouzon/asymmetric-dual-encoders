"""Vendor the six test sets' queries+qrels and freeze content hashes (Codex round-2 blocker 2).

Writes results/frozen_eval/{ds}.json (queries + qrels, the labels the final scorer reads)
and results/eval_manifest.json (content hashes the final loader must verify so a silent
HF dataset revision can't invalidate the frozen per-query comparator pairing).
The M7 session extends the manifest with its dev and untouched-final sets at kickoff.
"""
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ["BENCH_DATASETS"] = "scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid"
sys.path.insert(0, str(REPO / "bench"))

from core import DATASETS, load_beir  # noqa: E402


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


out_dir = REPO / "results" / "frozen_eval"
out_dir.mkdir(exist_ok=True)
pq = json.load(open(REPO / "results" / "perquery.json"))
manifest = {
    "_note": "Frozen 2026-08-25. Final-run loader must verify corpus hashes against a fresh "
    "HF download and read queries/qrels ONLY from results/frozen_eval/. Comparator pairing "
    "is invalid if any hash mismatches.",
    "versions": {"datasets": version("datasets"), "pytrec_eval": version("pytrec-eval-terrier")},
    "datasets": {},
}
for ds in DATASETS:
    doc_ids, doc_texts, q_ids, q_texts, qrels = load_beir(ds)
    (out_dir / f"{ds}.json").write_text(json.dumps({"queries": dict(zip(q_ids, q_texts)), "qrels": qrels}))
    manifest["datasets"][ds] = {
        "n_docs": len(doc_ids),
        "n_queries": len(q_ids),
        "corpus_ids_sha256": sha(doc_ids),
        "corpus_text_sha256": sha(doc_texts),
        "qids_sha256": sha(sorted(q_ids)),
        # The manifest pinned qids and qrels but not the query TEXT, so editing a query's text
        # while leaving its key alone passed every final-run check and was scored
        # (Codex review #4, MAJOR 1). Ordered by sorted qid, matching final_run.preflight.
        "qtexts_sha256": sha([dict(zip(q_ids, q_texts))[q] for q in sorted(q_ids)]),
        "qrels_sha256": sha(qrels),
        "comparator_vectors_sha256": {n: sha(v) for n, v in pq["datasets"][ds]["systems"].items()},
    }
    print(ds, manifest["datasets"][ds]["n_docs"], "docs,", manifest["datasets"][ds]["n_queries"], "queries", flush=True)
(REPO / "results" / "eval_manifest.json").write_text(json.dumps(manifest, indent=1))
print("wrote results/eval_manifest.json + results/frozen_eval/")
