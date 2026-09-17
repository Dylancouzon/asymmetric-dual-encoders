#!/usr/bin/env python3
"""Peak memory of the frozen BM25 function at two public corpus sizes.

The one hard unknown about running M20 on the local box is whether `m7src/fusion.py`'s BM25 fits
in 25 GB of RAM at MS MARCO's 8.8M documents. M7 recorded HotpotQA's 5.23M as "the single most
expensive repeated step on this box", which is evidence but not a measurement, and a fixed cost
divided by a row count is not a per-row cost. So measure at two sizes and extrapolate the
*marginal* cost, separating the fixed load from the per-document term.

Public corpora only, and only ones already scored or already development-contacted: no reserved
corpus, no query, no qrel. It builds the index and retrieves for a small fixed set of synthetic
non-benchmark queries, because indexing and retrieval have different peaks.

  PYTHONPATH=m20src:m8src:m7src:bench .venv/bin/python -u m20src/bm25_memory_probe.py \\
      --corpora nq hotpotqa
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

REPO = Path(__file__).resolve().parents[1]
for _p in ("bench", "m7src", "m8src", "m20src"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

import pre_encode as P            # claims the corpus-only guard entry and installs the guard

RESULT = REPO / "results" / "m20_bm25_memory_probe.json"
# Deliberately not benchmark queries: this probe measures memory, not quality.
PROBE_QUERIES = [f"synthetic memory probe query number {i} about storage and retrieval"
                 for i in range(64)]


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def host():
    import shutil

    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        meminfo[key] = value.strip()
    return {"MemTotal": meminfo.get("MemTotal"), "SwapTotal": meminfo.get("SwapTotal"),
            "free_bytes_root": shutil.disk_usage("/").free}


def probe(corpus_name):
    import fusion

    started = time.monotonic()
    corpus, source, revision = P.load_corpus(corpus_name)
    doc_ids = [str(value) for value in corpus["_id"]]
    doc_texts = [P._doc_text(row) for row in corpus]
    loaded_rss = peak_rss_bytes()
    load_seconds = time.monotonic() - started
    print(f"[bm25probe] {corpus_name}: {len(doc_ids):,} docs loaded, "
          f"peak RSS {loaded_rss / 1e9:.1f} GB, {load_seconds:.0f}s", flush=True)

    index_started = time.monotonic()
    run = fusion.bm25_run(doc_ids, doc_texts, [f"probe-{i}" for i in range(len(PROBE_QUERIES))],
                          PROBE_QUERIES)
    peak = peak_rss_bytes()
    index_seconds = time.monotonic() - index_started
    row = {
        "corpus": corpus_name, "source": source, "revision": revision,
        "n_docs": len(doc_ids),
        "corpus_text_bytes": sum(len(text) for text in doc_texts),
        "peak_rss_after_load_bytes": loaded_rss,
        "peak_rss_bytes": peak,
        "load_seconds": load_seconds,
        "index_and_retrieve_seconds": index_seconds,
        "n_probe_queries": len(PROBE_QUERIES),
        "nonempty_runs": int(sum(1 for value in run.values() if value)),
    }
    print(f"[bm25probe] {corpus_name}: peak RSS {peak / 1e9:.1f} GB after index+retrieve, "
          f"{index_seconds:.0f}s", flush=True)
    del run, doc_texts, doc_ids, corpus
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpora", nargs="+", default=["nq", "hotpotqa"])
    parser.add_argument("--allow-reserved-corpus", action="store_true",
                        help="permit a reserved CORPUS. This is corpus-only contact under the "
                             "corpus-only allowlist entry -- exactly what m8src/pre_encode.py "
                             "already does for these four datasets before the tag. No query, no "
                             "qrel, and the probe queries are synthetic. It is needed because "
                             "BM25 indexing happens inside the tagged transaction, so its memory "
                             "must be proven BEFORE the access is spent, on the real corpus "
                             "rather than by extrapolating from a similar one.")
    parser.add_argument("--project-to", nargs="+", type=int, default=[5_416_568, 8_841_823])
    args = parser.parse_args(argv)

    rows = []
    for corpus_name in args.corpora:
        if corpus_name in P.RESERVED_DATASETS and not args.allow_reserved_corpus:
            raise SystemExit(f"{corpus_name} is reserved; pass --allow-reserved-corpus if you "
                             f"intend corpus-only contact")
        rows.append(probe(corpus_name))

    record = {"status": "MEASURED", "host": host(), "rows": rows,
              "protected_evaluation_access": False,
              "reserved_corpus_contact": sorted(set(args.corpora) & set(P.RESERVED_DATASETS)),
              "_reserved_corpus_note": "corpus text only, under the corpus-only allowlist entry, "
                                       "which is the same contact m8src/pre_encode.py makes with "
                                       "these corpora before the tag. No query or qrel was "
                                       "opened; the probe queries are synthetic.",
              "purpose": "does m7src/fusion.py's BM25 fit in this box's RAM at MS MARCO scale"}
    if len(rows) >= 2:
        small, large = sorted(rows, key=lambda r: r["n_docs"])[:2]
        marginal = ((large["peak_rss_bytes"] - small["peak_rss_bytes"])
                    / (large["n_docs"] - small["n_docs"]))
        fixed = small["peak_rss_bytes"] - marginal * small["n_docs"]
        record["model"] = {
            "marginal_bytes_per_doc": marginal, "fixed_bytes": fixed,
            "_why": "Two sizes separate the fixed load from the per-document term. A single "
                    "measurement divided by its row count would blame the fixed cost on the rows.",
            "projection": {n: fixed + marginal * n for n in args.project_to},
        }
        total = int(host()["MemTotal"].split()[0]) * 1024
        record["model"]["memtotal_bytes"] = total
        record["model"]["fits"] = {n: bool(fixed + marginal * n < total * 0.85)
                                   for n in args.project_to}
        print(json.dumps(record["model"], indent=2), flush=True)
    RESULT.write_text(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
