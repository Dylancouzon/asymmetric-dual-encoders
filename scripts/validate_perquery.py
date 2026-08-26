"""Executable check on the frozen comparators: structure, per-qid pairing, and cell means.

Three layers (Codex M-perquery: the original mean-only check passed under any permutation of
scores across qids, which silently destroys the pairing every CI depends on):
  1. structural -- qids sorted+unique, every system vector the right length;
  2. per-qid pairing -- sha256 over the canonical (qid, score) serialization per cell, against
     results/perquery.sha256.json (written once, 2026-08-26, from the then-validated file; the
     Mac caches behind perquery.json are gone, so the freeze pins the pairing we have, and the
     bm25 spot-check below is the independent evidence for it);
  3. cell means vs quality.json, with the FINAL_MATRIX.md allowlist.
`--bm25` additionally RECOMPUTES bm25 per-qid nDCG@10 from results/frozen_eval/ + the HF corpora
and compares element-wise -- an independent full-row pairing check (logged six-set access,
class (a): no new-model number is produced).
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
q = json.load(open(REPO / "results" / "quality.json"))
p = json.load(open(REPO / "results" / "perquery.json"))

# (system, dataset) -> known encode-time-fp32 vs fp16-at-rest deltas, documented in FINAL_MATRIX.md
ALLOW = {("potion-retrieval-32M", "arguana"), ("potion-retrieval-32M", "trec-covid"),
         ("mdbr-leaf-ir", "arguana"), ("arctic-embed-m-v1.5", "scidocs")}
QUALITY_SLUG = {"lr-dense-pertask": "lightretriever-qwen2.5-1.5b-dense",
                "lr-dense-websearch": "lightretriever-qwen2.5-1.5b-dense-websearch"}

failures = []

# layer 1: structure
for ds, blob in p["datasets"].items():
    qids = blob["qids"]
    if qids != sorted(qids) or len(set(qids)) != len(qids):
        failures.append(f"{ds}: qids not sorted+unique")
    for name, vec in blob["systems"].items():
        if len(vec) != len(qids):
            failures.append(f"{name}/{ds}: {len(vec)} scores for {len(qids)} qids")

# layer 2: per-qid pairing hashes (freeze file written on first run)
HP = REPO / "results" / "perquery.sha256.json"
hashes = {ds: {name: hashlib.sha256("\n".join(
              f"{q}\t{s:.6f}" for q, s in zip(blob["qids"], vec)).encode()).hexdigest()
              for name, vec in blob["systems"].items()}
          for ds, blob in p["datasets"].items()}
if HP.exists():
    frozen = json.load(open(HP))
    for ds in frozen:
        for name, h in frozen[ds].items():
            if hashes.get(ds, {}).get(name) != h:
                failures.append(f"{name}/{ds}: per-qid pairing hash mismatch vs perquery.sha256.json")
else:
    HP.write_text(json.dumps(hashes, indent=1))
    print(f"wrote {HP.name}: per-qid pairing frozen for "
          f"{sum(len(v) for v in hashes.values())} cells")

# layer 3: cell means
for ds, blob in p["datasets"].items():
    for name, vec in blob["systems"].items():
        mean = sum(vec) / len(blob["qids"])
        rec = q[QUALITY_SLUG.get(name, name)][ds]["ndcg@10"]
        if abs(mean - rec) > 5e-5 and (name, ds) not in ALLOW:
            failures.append(f"{name}/{ds}: perquery {mean:.6f} vs quality.json {rec:.6f}")
        elif (name, ds) in ALLOW and abs(mean - rec) > 5e-4:
            failures.append(f"{name}/{ds}: allowlisted cell drifted beyond 5e-4 ({mean:.6f} vs {rec:.6f})")

if "--bm25" in sys.argv:
    # independent per-qid recompute for the one cheaply-reproducible system
    sys.path.insert(0, str(REPO / "bench"))
    import os
    os.environ["BENCH_DATASETS"] = ",".join(p["datasets"])
    import numpy as np
    from core import load_beir, topk_run   # load_beir appends to m7/SIX_ACCESS.log
    import bm25s, Stemmer, pytrec_eval
    for ds, blob in p["datasets"].items():
        doc_ids, doc_texts, q_ids, q_texts, qrels = load_beir(ds)
        st = Stemmer.Stemmer("english")
        r = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
        r.index(bm25s.tokenize(doc_texts, stopwords="en", stemmer=st, show_progress=False),
                show_progress=False)
        ids, sc = r.retrieve(bm25s.tokenize(q_texts, stopwords="en", stemmer=st,
                                            show_progress=False),
                             k=min(1000, len(doc_ids)), show_progress=False)
        sims = np.zeros((len(q_ids), len(doc_ids)), dtype=np.float32)
        for i in range(len(q_ids)):
            sims[i, ids[i]] = sc[i]
        run = topk_run(doc_ids, sims, q_ids, k=1000)
        ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
        got = {q: v["ndcg_cut_10"] for q, v in ev.evaluate(run).items()}
        want = dict(zip(blob["qids"], blob["systems"]["bm25"]))
        bad = [q for q in want if abs(got.get(q, -1) - want[q]) > 5e-4]
        print(f"  bm25 per-qid {ds}: {len(want)-len(bad)}/{len(want)} match" +
              (f" -- MISMATCHES {bad[:5]}" if bad else ""))
        if bad:
            failures.append(f"bm25/{ds}: {len(bad)} per-qid mismatches")

if failures:
    print("PERQUERY VALIDATION FAILED:\n" + "\n".join(failures))
    sys.exit(1)
n = sum(len(b["systems"]) for b in p["datasets"].values())
print(f"OK: {n} cells validated ({len(ALLOW)} allowlisted, documented in FINAL_MATRIX.md)")
