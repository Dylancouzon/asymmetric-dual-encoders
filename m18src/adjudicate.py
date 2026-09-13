"""Export the bounded prospective M18 structural pool for source-only adjudication."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import protocol
from common import REPO, WORK, atomic_write_bytes, registry, sha_file, sha_json, write_json


def export(corpus_path=None, out_path=None):
    reg = registry()
    corpus_path = Path(corpus_path or WORK / "derived/corpus.jsonl")
    corpus = list(protocol._read_jsonl(corpus_path))
    by_id = {r["doc_id"]: r for r in corpus}
    structural = protocol.structural_candidates(corpus)
    pool = protocol.prospective_adjudication_pool(structural, reg)
    rows = []
    for q in pool:
        rows.append(protocol.adjudication_record(q, by_id))
    # Recompute the protocol candidates with the same deterministic identities available to the
    # later ingestion pass. The digest map is deliberately a separate, content-bound artifact.
    out_path = Path(out_path or REPO / "results/m18_adjudication_candidates.jsonl")
    payload = b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n").encode()
                       for r in rows)
    atomic_write_bytes(out_path, payload)
    identity_path = out_path.with_name(out_path.stem + "_identities.json")
    write_json(identity_path, {r["query_id"]: r["candidate_sha256"] for r in rows})
    manifest = {"_schema": "m18-adjudication-candidates-v1", "prospective": True,
                "candidate_count": len(rows), "pool_per_stratum":
                reg["evaluation"]["qrel_adjudication"]["pool_per_stratum"],
                "strata": dict(sorted(Counter(r["proposed_stratum"] for r in rows).items())),
                "candidates_sha256": sha_file(out_path),
                "identities_sha256": sha_file(identity_path),
                "corpus_sha256": sha_file(corpus_path),
                "instruction": ("Judge whether target clearly and directly answers query using only "
                                "source_context and target. Partial/ambiguous pairs are excluded. "
                                "Correct the stratum only when clearly mismatched. No retrieval output exists.")}
    manifest["sha256"] = sha_json(manifest)
    write_json(REPO / "results/m18_adjudication_candidates_manifest.json", manifest)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=None); ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    print(json.dumps(export(args.corpus, args.out), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
