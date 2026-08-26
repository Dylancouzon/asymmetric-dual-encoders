"""Extend results/eval_manifest.json with the M7 dev components and the UNTOUCHED-FINAL sets,
and vendor their queries+qrels into results/frozen_eval/ (the final scorer's only label source).

Run at kickoff, before any candidate result exists. Committing untouched-final qrels is not
reading them: the final scorer is the sole reader and logs every access to m7/LEDGER.md.
"""
import json

from datasets import load_dataset

import devsuite
from _paths import REPO
from core import doc_text
from hashing import sha, sha_stream_list

FROZEN = REPO / "results" / "frozen_eval"
MANIFEST = REPO / "results" / "eval_manifest.json"
UNTOUCHED = ["fever", "dbpedia-entity"]   # climate-fever dropped: no affirmative license
# untouched-final repair (LEDGER 2026-08-26, pre-freeze): the partition's two Wikipedia members
# carry 9-11%% TRAIN doc overlap, so two unused CQADupStack subforums -- picked by a rule fixed in
# the ledger (alphabetically first two outside dev's programmers/physics) -- are its only
# near-zero-overlap, non-Wikipedia members. Development-informed at FAMILY level; labelled.
UNTOUCHED_CQA = ["cqadup-android", "cqadup-english"]


def beir_test_labels(ds):
    qrels = {}
    for r in load_dataset(f"BeIR/{ds}-qrels", split="test"):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    q = load_dataset(f"BeIR/{ds}", "queries")["queries"]
    q_ids, q_texts = [], []
    for i, t in zip(list(q["_id"]), list(q["text"])):
        if str(i) in qrels:
            q_ids.append(str(i))
            q_texts.append(t)
    return q_ids, q_texts, qrels


def corpus_hashes_streamed(ds):
    """Hashes the corpus without materializing it: FEVER (5.4M) and DBpedia (4.6M) as Python
    string lists plus a json.dumps copy would be ~8 GB each, and this box has 25 GB total."""
    corpus = load_dataset(f"BeIR/{ds}", "corpus")["corpus"]
    n = len(corpus)
    ids_sha = sha_stream_list(str(x) for x in corpus["_id"])
    text_sha = sha_stream_list(doc_text(r) for r in corpus)
    return n, ids_sha, text_sha


def entry_streamed(n_docs, ids_sha, text_sha, q_ids, qrels, construction):
    return {"n_docs": n_docs, "n_queries": len(q_ids), "corpus_ids_sha256": ids_sha,
            "corpus_text_sha256": text_sha, "qids_sha256": sha(sorted(q_ids)),
            "qrels_sha256": sha(qrels), "construction": construction}


def entry(doc_ids, doc_texts, q_ids, q_texts, qrels, construction):
    return entry_streamed(len(doc_ids), sha_stream_list(doc_ids), sha_stream_list(doc_texts),
                          q_ids, qrels, construction)


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
    q_ids, q_texts, qrels = beir_test_labels(ds)
    (FROZEN / f"untouched-{ds}.json").write_text(json.dumps({"queries": dict(zip(q_ids, q_texts)),
                                                            "qrels": qrels}))
    n, ids_sha, text_sha = corpus_hashes_streamed(ds)
    man["m7_untouched_final"][ds] = entry_streamed(n, ids_sha, text_sha, q_ids, qrels,
                                                  f"BeIR/{ds} test split, full corpus")
    print(f"untouched {ds:15s} {n:>9,} docs {len(q_ids):>6,} queries", flush=True)

for c in UNTOUCHED_CQA:
    doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(c)
    (FROZEN / f"untouched-{c}.json").write_text(json.dumps({"queries": dict(zip(q_ids, q_texts)),
                                                            "qrels": qrels}))
    man["m7_untouched_final"][c] = entry(doc_ids, doc_texts, q_ids, q_texts, qrels,
                                         f"mteb/cqadupstack-{c.split('-',1)[1]}, full corpus + "
                                         "test qrels; added pre-freeze per LEDGER "
                                         "untouched-final repair 2026-08-26")
    print(f"untouched {c:15s} {len(doc_ids):>9,} docs {len(q_ids):>6,} queries", flush=True)

man["m7_notes"]["untouched_final_repair"] = (
    "cqadup-android and cqadup-english added 2026-08-26 before freeze: the only near-zero-overlap "
    "non-Wikipedia members (R3: fever 11.3%%, dbpedia 9.32%%, cqadupstack ~0). Same family as two "
    "dev components -- development-informed at family level, labelled at the row.")

MANIFEST.write_text(json.dumps(man, indent=1))
print("extended results/eval_manifest.json with m7_dev and m7_untouched_final")
