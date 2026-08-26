"""The two R3 sweeps the first decontamination run was missing: TRAIN vs the nq-250k dev corpus
and TRAIN vs the FEVER untouched-final corpus.

decontam.py now covers all five protected corpora, so a fresh run produces one coherent summary.
This script exists so the already-completed 30-minute run need not be repeated: it rebuilds the
same TRAIN-side index (deterministic, ~4 min) and merges only the missing rows into
results/m7_decontam.json. Both paths give identical numbers.

Run it as a script, never import it: it is a memory-heavy job and must not share the box.
"""
import json
import time

import numpy as np

import mix
from _paths import REPO
from decontam import (DUP_SHARE, Inverted, exact_u64, sketch, stream_beir_docs,
                      stream_dev_component_docs)


def build_index(t0):
    store_of = {s: mix.load_source(s)["docstore"] for s in mix.available_sources()}
    tr, _ = mix.split_pairs()
    stores, doc_text_of, order = {}, {}, []
    for src, qid, query, pos, hneg in tr:
        st = store_of[src]
        if st not in stores:
            ids, texts = mix.load_store(st)
            stores[st] = dict(zip(ids, texts))
        for d in pos:
            t = stores[st].get(d)
            if t is not None and (st, d) not in doc_text_of:
                doc_text_of[(st, d)] = t
                order.append((st, d))
    del stores
    print(f"[index] {len(order):,} unique TRAIN positive documents ({time.time()-t0:.0f}s)", flush=True)
    sk = [sketch(doc_text_of[k]) for k in order]
    ex = [exact_u64(doc_text_of[k]) for k in order]
    del doc_text_of
    inv = Inverted(sk, ex)
    print(f"  inverted index {inv.h.size:,} sketch hashes ({time.time()-t0:.0f}s)", flush=True)
    return inv


def sweep(inv, label, it, total, t0):
    flag = np.zeros(inv.n, dtype=bool)
    ex_flag = np.zeros(inv.n, dtype=bool)
    n = 0
    for text in it:
        e, near = inv.match(text, DUP_SHARE, want_sketch=True)
        if e.size:
            ex_flag[e] = True
        if near.size:
            flag[near] = True
        n += 1
        if n % 500_000 == 0:
            print(f"    {label} {n:,}/{total:,} ({time.time()-t0:.0f}s)", flush=True)
    res = {"protected_docs_scanned": n, "train_docs_exact_dup": int(ex_flag.sum()),
           "train_docs_near_dup": int(flag.sum()), "train_docs_total": inv.n,
           "near_dup_rate": round(float((flag | ex_flag).mean()), 5)}
    print(f"  [{label}] {json.dumps(res)}", flush=True)
    return res


def main():
    t0 = time.time()
    inv = build_index(t0)
    rows = {
        "nq-250k-dev": sweep(inv, "nq-250k-dev", stream_dev_component_docs("nq-250k"), 250_000, t0),
        "fever-untouched": sweep(inv, "fever-untouched", stream_beir_docs("fever"), 5_416_568, t0),
    }
    p = REPO / "results" / "m7_decontam.json"
    blob = json.loads(p.read_text())
    blob["R3_disclosed_overlap"].update(rows)
    blob["R3_disclosed_overlap"]["fever-untouched-note"] = (
        "fever-pos is drawn from this corpus by construction; the rate above also covers the "
        "hotpotqa/squad/mrtydi Wikipedia positives against it")
    blob["R3_extra_sweep_seconds"] = round(time.time() - t0, 1)
    p.write_text(json.dumps(blob, indent=1))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
