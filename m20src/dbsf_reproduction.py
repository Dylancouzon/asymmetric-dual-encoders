#!/usr/bin/env python3
"""Prove the M20 roster code computes M12's DBSF@100 before it runs on anything new.

Registered as a run prerequisite in `m20/REGISTRATION.md`.  Two separate questions, because they
have different right answers:

  A. **Did extending the executor change the operator?**  Re-run M12's exact path -- the frozen M7
     table, dense at depth 1000 truncated to 100, BM25 at depth 1000 truncated to 100, DBSF -- but
     through `m20src/roster.py`.  This must reproduce `m12/six_dbsf.json` to 1e-9.  Anything larger
     is a code change, not a rounding difference.

  B. **What does substituting the RELEASED bundle for the frozen table cost?**  M20 scores
     `zero-dense` from the published `constella-zero` int8 bytes, whose query vectors differ from
     the frozen table's by ~1.5e-8 max-abs.  That difference is real and its effect on a reported
     number is measured here rather than assumed to be zero.

  C. **Is retrieving dense at depth 100 the same as retrieving at 1000 and truncating?**  The
     registration claims it is, because the only possible difference is which of two EQUAL-scoring
     documents sits at rank 100.  Measured, not asserted.

Six-set data only.  No reserved payload, no protected path.

  PYTHONPATH=m20src:m12src:m7src M7_ENCODER=stella-400M-v5 \\
      .venv/bin/python m20src/dbsf_reproduction.py --datasets scifact nfcorpus
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in ("m7src", "m11/release", "m12src", "m20src"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

RESULT = REPO / "results" / "m20_dbsf_reproduction.json"
EXACT_TOLERANCE = 1e-9          # A: extending the code must not move the operator at all
BUNDLE_TOLERANCE = 1e-6         # B: the released-bundle substitution, measured and bounded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    parser.add_argument("--out", default=str(RESULT))
    args = parser.parse_args(argv)

    import torch

    import fusion
    import qfusion
    import roster
    import run_six_dbsf as M12
    from evalkit import topk_ids_scores
    from table import Preproc, load_table, read_meta
    from teacher import encode_cached

    roster.assert_registered_identities()
    published = json.loads((REPO / "m12" / "six_dbsf.json").read_text())["per_dataset"]
    spec = json.loads((REPO / "m7" / "FREEZE.json").read_text())
    table_path = REPO / spec["table_relpath"]
    pre = Preproc(**read_meta(table_path)["preproc"])

    rows = {}
    for dataset in args.datasets:
        doc_ids, doc_texts, q_ids, q_texts, qrels = M12.load(dataset)
        doc_vectors = encode_cached(f"final-six-{dataset}-docs", doc_texts, prefix="",
                                    dtype=torch.float32, verify=True)
        frozen_table = load_table(table_path, variant="int8")
        frozen_q = np.asarray(frozen_table.encode(q_texts, pre), dtype=np.float32)
        del frozen_table
        bundle = roster.QueryEncoder("zero-dense")
        bundle_q = bundle.encode(q_texts)
        bundle.release()

        lexical = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts)
        row = {"query_vector_max_abs_frozen_vs_bundle": float(np.abs(frozen_q - bundle_q).max())}

        def dbsf_mean(qvectors, depth):
            dense = topk_ids_scores(qvectors, doc_vectors, doc_ids, k=depth, chunk=200_000,
                                    qids=q_ids)
            fused = roster.dbsf_at_depth(dense, lexical)
            return float(np.mean(list(roster.per_query_ndcg10(fused, qrels).values())))

        row["m20_frozen_table_depth1000"] = dbsf_mean(frozen_q, fusion.DEPTH)
        row["m20_frozen_table_depth100"] = dbsf_mean(frozen_q, roster.DENSE_DEPTH)
        row["m20_released_bundle_depth100"] = dbsf_mean(bundle_q, roster.DENSE_DEPTH)
        row["m12_published_dbsf100"] = float(published[dataset]["dbsf@100"])
        row["A_operator_delta"] = row["m20_frozen_table_depth1000"] - row["m12_published_dbsf100"]
        row["B_released_bundle_delta"] = (row["m20_released_bundle_depth100"]
                                          - row["m20_frozen_table_depth100"])
        row["C_depth_equivalence_delta"] = (row["m20_frozen_table_depth100"]
                                            - row["m20_frozen_table_depth1000"])
        row["A_passed"] = abs(row["A_operator_delta"]) <= EXACT_TOLERANCE
        row["B_passed"] = abs(row["B_released_bundle_delta"]) <= BUNDLE_TOLERANCE
        row["C_passed"] = abs(row["C_depth_equivalence_delta"]) <= EXACT_TOLERANCE
        rows[dataset] = row
        print(f"  {dataset:<12} A {row['A_operator_delta']:+.3e}  "
              f"B {row['B_released_bundle_delta']:+.3e}  "
              f"C {row['C_depth_equivalence_delta']:+.3e}", flush=True)
        del doc_vectors

    record = {
        "status": "PASSED" if all(r["A_passed"] and r["B_passed"] and r["C_passed"]
                                  for r in rows.values()) else "FAILED",
        "purpose": "m20/REGISTRATION.md run prerequisite: the extended roster code reproduces "
                   "M12's registered DBSF@100 operator before it runs on new data.",
        "protected_evaluation_access": False,
        "datasets": list(args.datasets),
        "tolerances": {"A_operator": EXACT_TOLERANCE, "B_released_bundle": BUNDLE_TOLERANCE,
                       "C_depth_equivalence": EXACT_TOLERANCE},
        "checks": {
            "A": "m20 roster + frozen M7 table, dense depth 1000 truncated to 100, vs "
                 "m12/six_dbsf.json dbsf@100",
            "B": "released constella-zero int8 bundle vs the frozen table, same everything else",
            "C": "dense retrieved at depth 100 vs retrieved at 1000 and truncated to 100",
        },
        "rows": rows,
        "bm25_config": __import__("fusion").BM25_CONFIG,
    }
    Path(args.out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"status": record["status"], "out": args.out}))
    return 0 if record["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
