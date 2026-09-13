"""Report-only top-10 comparison for bare Qdrant terms; intentionally has no qrels."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import evaluate
import system
from common import REPO, WORK, admit_read, registry, write_json

UNSUPPORTED_REQUESTED = ("k8s", "mmap", "cuda", "tls", "rocksdb", "wal")
SUMMARY_FIELDS = ("kind", "title", "path", "source_url", "artifact_id")


def _top(run, metadata, term, limit=10):
    ranked = sorted(run[term].items(), key=lambda x: (-x[1], str(x[0])))[:limit]
    return [{"rank": i, "doc_id": doc_id, "score": float(score),
             **{k: metadata.get(doc_id, {}).get(k) for k in SUMMARY_FIELDS}}
            for i, (doc_id, score) in enumerate(ranked, 1)]


def build(selected_bundle, out, limit=10):
    reg = registry()
    vocab = json.loads(admit_read(REPO / "results/m18_vocabulary_manifest.json").read_text())
    admitted = [row["term"] for row in vocab["terms"]["terms"]]
    terms = list(dict.fromkeys(admitted + list(UNSUPPORTED_REQUESTED)))
    memory = system.ProjectMemory(WORK / "derived/index", selected_bundle)
    bm25 = memory.bm25.run(terms, terms, 100)
    models = {"zero_v1": reg["models"]["zero_v1"]["source_path"],
              "selected": str(selected_bundle)}
    result = {"_schema": "m18-bare-term-probe-v1", "report_only": True,
              "qrels": None, "terms": terms, "admitted_terms": admitted,
              "unsupported_requested_terms": [x for x in UNSUPPORTED_REQUESTED if x not in admitted],
              "models": {}, "bm25": {t: _top(bm25, memory.metadata, t, limit) for t in terms}}
    for name, bundle in models.items():
        encoder = system._query_encoder(bundle)
        started = time.perf_counter()
        qv = encoder.encode(terms)
        elapsed = time.perf_counter() - started
        dense = evaluate.search(qv, memory.vectors, 100, memory.ids, terms)
        fused = evaluate.dbsf_at(dense, bm25, 100)
        result["models"][name] = {
            "bundle": str(bundle), "query_encode_milliseconds": elapsed * 1000,
            "dense": {t: _top(dense, memory.metadata, t, limit) for t in terms},
            "dbsf": {t: _top(fused, memory.metadata, t, limit) for t in terms}}
    write_json(out, result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args(argv)
    result = build(args.selected, args.out, args.limit)
    print(json.dumps({"output": args.out, "terms": len(result["terms"])}, sort_keys=True))


if __name__ == "__main__":
    main()
