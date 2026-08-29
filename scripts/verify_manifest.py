"""Bring-up step 2: re-download the six and verify eval_manifest.json corpus hashes.

The frozen comparator pairing in results/perquery.json is only valid if the HF dataset
content still matches what it was frozen against. Also checks that the vendored
queries/qrels in results/frozen_eval/ match the manifest and the fresh download.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ["BENCH_DATASETS"] = "scifact,nfcorpus,fiqa,arguana,scidocs,trec-covid"
sys.path.insert(0, str(REPO / "bench"))

from core import DATASETS, load_beir  # noqa: E402


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


man = json.load(open(REPO / "results" / "eval_manifest.json"))["datasets"]
fail = []
for ds in DATASETS:
    doc_ids, doc_texts, q_ids, q_texts, qrels = load_beir(ds)
    m = man[ds]
    checks = {
        "n_docs": (len(doc_ids), m["n_docs"]),
        "n_queries": (len(q_ids), m["n_queries"]),
        "corpus_ids_sha256": (sha(doc_ids), m["corpus_ids_sha256"]),
        "corpus_text_sha256": (sha(doc_texts), m["corpus_text_sha256"]),
        "qids_sha256": (sha(sorted(q_ids)), m["qids_sha256"]),
        "qrels_sha256": (sha(qrels), m["qrels_sha256"]),
    }
    # vendored copy must agree with the fresh download too
    froz = json.loads((REPO / "results" / "frozen_eval" / f"{ds}.json").read_text())
    checks["frozen_queries"] = (sha(dict(zip(q_ids, q_texts))), sha(froz["queries"]))
    checks["frozen_qrels"] = (sha(qrels), sha(froz["qrels"]))
    bad = [k for k, (got, want) in checks.items() if got != want]
    print(f"{ds:11s} {'OK ' if not bad else 'FAIL'} {len(doc_ids)} docs / {len(q_ids)} queries"
          + (f"  mismatched: {bad}" if bad else ""), flush=True)
    for k in bad:
        fail.append(f"{ds}.{k}: {checks[k][0]} != {checks[k][1]}")

if fail:
    print("\nMANIFEST VERIFICATION FAILED:\n" + "\n".join(fail))
    sys.exit(1)
print("\nOK: all six datasets match the frozen manifest; comparator pairing is valid.")
