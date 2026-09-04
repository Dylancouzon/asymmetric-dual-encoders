"""M12: re-measure the published fused row on the six under DBSF at prefetch 100.

Registered in m12/LEDGER.md (Amendment 2026-09-04) BEFORE this ran. Dylan's ruling: convex fusion
goes, because Qdrant cannot run it, and a 1000-deep prefetch to return 10 is not a realistic
configuration. DBSF at depth 100 is the replacement -- a shipping operator with ZERO fitted
parameters, so the released system loses a hyperparameter rather than gaining one.

DESCRIPTIVE and development-informed. C1/C2/C3 keep their registered convex0 basis and are not
recomputed. The table artifact is unchanged. The convex0 row is retained beside the new one, never
deleted. Nothing reserved is touched.

Loading mirrors `final_run.verify_and_load` -- same manifest hash checks, same frozen_eval labels --
but logs to m12, because an M12 descriptive pass is not a FINAL-RUN access and must not be written
into m7/LEDGER.md as one. `_m7_access_trail` still fires: the six-access log should record this.

  PYTHONPATH=m12src:m7src M7_ENCODER=stella-400M-v5 .venv/bin/python m12src/run_six_dbsf.py
"""
import json

import numpy as np
import torch

import fusion
import qfusion
from _paths import REPO
from evalkit import per_query_ndcg, topk_ids_scores
from hashing import sha_stream_list
from table import Preproc, load_table, read_meta
from teacher import QUERY_PREFIX, encode_cached

SIX = ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid"]
CLEAN4 = ["nfcorpus", "scidocs", "scifact", "trec-covid"]   # no disclosed stella overlap
HEADLINE_DEPTH = 100                                        # registered before any number
CURVE = [10, 50, 100, 1000]
FROZEN = REPO / "results" / "frozen_eval"
MANIFEST = REPO / "results" / "eval_manifest.json"


def load(ds):
    from core import _m7_access_trail, doc_text
    from datasets import load_dataset
    _m7_access_trail(ds)
    man = json.loads(MANIFEST.read_text())["datasets"][ds]
    corpus = load_dataset(f"BeIR/{ds}", "corpus")["corpus"]
    doc_ids, doc_texts = [str(x) for x in corpus["_id"]], [doc_text(r) for r in corpus]
    for field, got in (("corpus_ids_sha256", sha_stream_list(doc_ids)),
                       ("corpus_text_sha256", sha_stream_list(doc_texts)),
                       ("n_docs", len(doc_ids))):
        if man[field] != got:
            raise SystemExit(f"M12 ABORTED: {ds}.{field} mismatch vs the frozen manifest")
    froz = json.loads((FROZEN / f"{ds}.json").read_text())
    q_ids = sorted(froz["queries"])
    return doc_ids, doc_texts, q_ids, [froz["queries"][q] for q in q_ids], froz["qrels"]


def main():
    spec = json.loads((REPO / "m7" / "FREEZE.json").read_text())
    table_path = REPO / spec["table_relpath"]
    pre = Preproc(**read_meta(table_path)["preproc"])
    print(f"M12 six-set DBSF, headline depth {HEADLINE_DEPTH}\n")

    per, collisions = {}, {}
    for ds in SIX:
        doc_ids, doc_texts, q_ids, q_texts, qrels = load(ds)
        dv = encode_cached(f"final-six-{ds}-docs", doc_texts, prefix="", dtype=torch.float32,
                           verify=True)
        m = load_table(table_path, variant="int8")
        dense = topk_ids_scores(m.encode(q_texts, pre), dv, doc_ids, k=fusion.DEPTH,
                                chunk=200_000, qids=q_ids)
        del m
        torch.cuda.empty_cache()
        bm25 = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts)
        collisions[ds] = len(set(q_ids) & set(doc_ids))

        row = {}
        # the published operator, at its own depth: a reproduction check on 0.4911
        row["convex0@1000"] = per_query_ndcg(
            fusion.convex([dense, bm25], w=0.8, floor_zero=True), qrels)
        row["dense"] = per_query_ndcg(dense, qrels)
        row["bm25"] = per_query_ndcg(bm25, qrels)
        for d in CURVE:
            td, tb = qfusion.truncate(dense, d), qfusion.truncate(bm25, d)
            row[f"dbsf@{d}"] = per_query_ndcg(qfusion.dbsf([td, tb]), qrels)
        per[ds] = row
        mean = {k: float(np.mean(list(v.values()))) for k, v in row.items()}
        print(f"  {ds:<12} " + "  ".join(f"{k}={v:.4f}" for k, v in mean.items())
              + f"  (self-hits {collisions[ds]})", flush=True)

    def avg(system, sets):
        return float(np.mean([np.mean(list(per[d][system].values())) for d in sets]))

    systems = ["dense", "bm25", "convex0@1000"] + [f"dbsf@{d}" for d in CURVE]
    print(f"\n  {'system':<16}{'all 6':>9}{'clean 4':>10}{'contam. cost':>14}")
    table = {}
    for s in systems:
        a, c = avg(s, SIX), avg(s, CLEAN4)
        table[s] = {"all6": a, "clean4": c, "contamination_cost": a - c}
        print(f"  {s:<16}{a:>9.4f}{c:>10.4f}{a - c:>14.4f}")

    frozen_fused = 0.4911
    got = table["convex0@1000"]["all6"]
    print(f"\n  reproduction of the published fused row: {got:.4f} vs {frozen_fused} "
          f"(delta {got - frozen_fused:+.4f})")

    out = {"headline_depth": HEADLINE_DEPTH, "registered": "m12/LEDGER.md Amendment 2026-09-04",
           "status": "DESCRIPTIVE, development-informed, post-M7",
           "six": SIX, "clean4": CLEAN4, "self_hit_collisions": collisions,
           "per_dataset": {d: {s: float(np.mean(list(per[d][s].values()))) for s in systems}
                           for d in SIX},
           "aggregates": table,
           "published_convex0_all6": frozen_fused,
           "reproduction_delta": got - frozen_fused,
           "headline_all6": table[f"dbsf@{HEADLINE_DEPTH}"]["all6"],
           "headline_clean4": table[f"dbsf@{HEADLINE_DEPTH}"]["clean4"]}
    (REPO / "m12" / "six_dbsf.json").write_text(json.dumps(out, indent=2))
    print("\n  wrote m12/six_dbsf.json")


if __name__ == "__main__":
    main()
