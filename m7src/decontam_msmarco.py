"""R1/R2 for the RESEARCH-ONLY `msmarco-train` source (the clean-stack tax, m7/LEDGER.md).

Mirrors `decontam.run` for one source, with the same functions and thresholds, and MERGES the kept
qids into `work/decontam/kept.json` under their own key -- inert for every five-source run, since
`train.kept_pairs` filters by the requested sources. Counts land in
`results/m7_decontam_msmarco.json`.

Scope, per the arm-shape pre-registration:
  * R1 remove on protected-query overlap (six + dev + untouched-final, the shared index).
  * R2 remove on positive-document overlap with the six.
  * R3 disclosure sweeps for cqadupstack-dev and nq-250k only; the FEVER/DBpedia sweeps are
    SKIPPED (those corpora are M8-reserved and this arm never trains toward them; disclosed).
  * Pool-ban is VACUOUS for this source: msmarco rows enter as a side bank of positives only and
    can never be drawn as negatives (arm-shape constraint 2).

    M7_ENCODER=stella-400M-v5 PYTHONPATH=m7src .venv/bin/python m7src/decontam_msmarco.py
"""
import json
import time

import numpy as np

import mix
from _paths import REPO, WORK
from decontam import (DUP_SHARE, Inverted, exact_u64, protected_query_index, query_hits, sketch,
                      stream_cqa_dev_docs, stream_dev_component_docs, stream_six_docs)
from trainmix import heldout

SRC = "msmarco-train"
OUT = WORK / "decontam"


def main():
    t0 = time.time()
    blob = mix.load_source(SRC)
    pairs, store_name = blob["pairs"], blob["docstore"]
    tr = [p for p in pairs if not heldout(SRC, p["qid"])]
    print(f"[{SRC}] {len(pairs):,} pairs, {len(tr):,} on the TRAIN side of the mod-50 split",
          flush=True)

    q_ex, q_gram, q_whole, prot_counts = protected_query_index()
    print(f"[R1] protected-query index: {prot_counts} ({time.time()-t0:.0f}s)", flush=True)
    r1 = {"exact": 0, "near": 0, "contains": 0}
    survive = []
    for n, p in enumerate(tr):
        if n and n % 100_000 == 0:
            print(f"    R1 {n:,}/{len(tr):,} ({time.time()-t0:.0f}s)", flush=True)
        hit = query_hits(p["query"], q_ex, q_gram, q_whole)
        if hit:
            r1[hit] += 1
        else:
            survive.append(p)
    n_r1 = sum(r1.values())
    print(f"[R1] removed {n_r1:,} ({r1}); {len(survive):,} survive ({time.time()-t0:.0f}s)",
          flush=True)

    ids, texts = mix.load_store(store_name)
    text_of = dict(zip(ids, texts))
    doc_key = {}
    for p in survive:
        for d in p["pos"]:
            if d in text_of and d not in doc_key:
                doc_key[d] = len(doc_key)
    keys = list(doc_key)
    print(f"[index] {len(keys):,} unique positive documents ({time.time()-t0:.0f}s)", flush=True)
    sk = [sketch(text_of[k]) for k in keys]
    ex = [exact_u64(text_of[k]) for k in keys]
    inv = Inverted(sk, ex)
    del sk, ex
    print(f"  inverted index {inv.h.size:,} sketch hashes, {inv.nbytes/1e9:.2f} GB "
          f"({time.time()-t0:.0f}s)", flush=True)

    def sweep(label, it, total=None):
        flag = np.zeros(inv.n, dtype=bool)
        exact_flag = np.zeros(inv.n, dtype=bool)
        n = 0
        for text in it:
            e, near = inv.match(text, DUP_SHARE, want_sketch=True)
            if e.size:
                exact_flag[e] = True
            if near.size:
                flag[near] = True
            n += 1
            if n % 500_000 == 0:
                print(f"    {label} {n:,}{'/' + format(total, ',') if total else ''} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        res = {"protected_docs_scanned": n,
               "train_docs_exact_dup": int(exact_flag.sum()),
               "train_docs_near_dup": int(flag.sum()),
               "train_docs_total": inv.n,
               "near_dup_rate": round(float((flag | exact_flag).mean()), 5)}
        print(f"  [{label}] {json.dumps(res)}", flush=True)
        return res, (flag | exact_flag)

    six_res, six_hit = sweep("six", stream_six_docs(), 272_117)
    cqa_res, _ = sweep("cqadupstack-dev", stream_cqa_dev_docs(), 70_492)
    nq_res, _ = sweep("nq-250k-dev", stream_dev_component_docs("nq-250k"), 250_000)

    bad = {keys[i] for i in np.nonzero(six_hit)[0]}
    r2 = 0
    kept = []
    for p in survive:
        if any(d in bad for d in p["pos"]):
            r2 += 1
            continue
        kept.append(p["qid"])
    print(f"[R2] removed {r2:,} pairs on positive-doc overlap with the six; {len(kept):,} kept "
          f"({time.time()-t0:.0f}s)", flush=True)

    kp = OUT / "kept.json"
    keep = json.loads(kp.read_text())
    assert SRC not in keep or set(keep[SRC]) == set(kept), \
        f"kept.json already holds a DIFFERENT {SRC} entry; refusing to overwrite silently"
    keep[SRC] = sorted(kept)
    kp.write_text(json.dumps({k: sorted(v) for k, v in keep.items()}))
    res = {"_note": "RESEARCH-ONLY msmarco decontamination for the clean-stack-tax arm "
                    "(m7/LEDGER.md). R3 sweeps for FEVER/DBpedia skipped: M8-reserved. "
                    "Pool-ban vacuous: side-bank positives never enter negative sampling.",
           "source": SRC, "pairs_total": len(pairs), "pairs_train_side": len(tr),
           "r1_removed": {**r1, "total": n_r1}, "r2_removed_pairs": r2, "kept_pairs": len(kept),
           "r3_disclosure": {"six": six_res, "cqadupstack-dev": cqa_res, "nq-250k-dev": nq_res},
           "seconds": round(time.time() - t0, 1)}
    (REPO / "results" / "m7_decontam_msmarco.json").write_text(json.dumps(res, indent=1))
    print("wrote results/m7_decontam_msmarco.json and merged kept.json key", flush=True)


if __name__ == "__main__":
    main()
