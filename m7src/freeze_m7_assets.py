"""Extend results/eval_manifest.json with the M7 dev components and the UNTOUCHED-FINAL sets,
and vendor their queries+qrels into results/frozen_eval/ (the final scorer's only label source).

Run at kickoff, before any candidate result exists. Committing untouched-final qrels is not
reading them: the final scorer is the sole reader and logs every access to m7/LEDGER.md.
"""
import hashlib
import json

from datasets import load_dataset

import devsuite
from _paths import REPO
from core import doc_text

FROZEN = REPO / "results" / "frozen_eval"
MANIFEST = REPO / "results" / "eval_manifest.json"
UNTOUCHED = ["fever", "dbpedia-entity"]   # climate-fever dropped: no affirmative license


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def beir_test(ds):
    qrels = {}
    for r in load_dataset(f"BeIR/{ds}-qrels", split="test"):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    q = load_dataset(f"BeIR/{ds}", "queries")["queries"]
    q_ids, q_texts = [], []
    for i, t in zip(list(q["_id"]), list(q["text"])):
        if str(i) in qrels:
            q_ids.append(str(i))
            q_texts.append(t)
    corpus = load_dataset(f"BeIR/{ds}", "corpus")["corpus"]
    return [str(x) for x in corpus["_id"]], [doc_text(r) for r in corpus], q_ids, q_texts, qrels


def entry(doc_ids, doc_texts, q_ids, q_texts, qrels, construction):
    return {"n_docs": len(doc_ids), "n_queries": len(q_ids),
            "corpus_ids_sha256": sha(doc_ids), "corpus_text_sha256": sha(doc_texts),
            "qids_sha256": sha(sorted(q_ids)), "qrels_sha256": sha(qrels),
            "construction": construction}


man = json.loads(MANIFEST.read_text())
man.setdefault("m7_dev", {})
man.setdefault("m7_untouched_final", {})
man["m7_notes"] = {
    "partitions": "datasets = KNOWN-TEST (the six); m7_dev = DEV; m7_untouched_final = UNTOUCHED-FINAL",
    "climate_fever": "DROPPED from UNTOUCHED-FINAL: no affirmative license at any primary source "
                     "(see m7/LEDGER.md). The partition therefore holds two sets, not three.",
    "fever_caveat": "BEIR FEVER is scored as untouched-final but is IN-DOMAIN if fever-train pairs "
                    "stay in the training mix (same corpus, disjoint queries). Labeled in the report; "
                    "DBpedia-entity is the clean generalization probe.",
    "frozen_eval_is_authoritative": "the final scorer reads queries/qrels only from results/frozen_eval/ "
                                    "after verifying corpus hashes against a fresh HF download",
}

for c in devsuite.COMPONENTS:
    doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(c)
    (FROZEN / f"dev-{c}.json").write_text(json.dumps({"queries": dict(zip(q_ids, q_texts)), "qrels": qrels}))
    man["m7_dev"][c] = entry(doc_ids, doc_texts, q_ids, q_texts, qrels,
                             devsuite.manifest_entry(c)["construction"])
    print(f"dev {c:20s} {len(doc_ids):>9,} docs {len(q_ids):>6,} queries", flush=True)

for ds in UNTOUCHED:
    doc_ids, doc_texts, q_ids, q_texts, qrels = beir_test(ds)
    (FROZEN / f"untouched-{ds}.json").write_text(json.dumps({"queries": dict(zip(q_ids, q_texts)),
                                                            "qrels": qrels}))
    man["m7_untouched_final"][ds] = entry(doc_ids, doc_texts, q_ids, q_texts, qrels,
                                         f"BeIR/{ds} test split, full corpus")
    print(f"untouched {ds:15s} {len(doc_ids):>9,} docs {len(q_ids):>6,} queries", flush=True)

MANIFEST.write_text(json.dumps(man, indent=1))
print("extended results/eval_manifest.json with m7_dev and m7_untouched_final")
