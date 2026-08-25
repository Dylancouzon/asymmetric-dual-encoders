"""Rule R1 applied to the query-text-only TRAIN sources (nq-open, TriviaQA).

decontam.py scans pairs; these sources have no positives and feed objective B only, so they get
their own pass against the same protected-query index (six + dev + untouched-final).
"""
import json
import time

import numpy as np

from _paths import REPO, WORK
from decontam import (OUT, build_protected, dev_queries_and_cqa_docs, hits,
                      six_queries_and_docs, untouched_queries_and_dbpedia_docs)
from trainmix import heldout

TRAIN = WORK / "train"
SOURCES = ["nqopen", "triviaqa"]

t0 = time.time()
six_q, _ = six_queries_and_docs()
dev_q, _ = dev_queries_and_cqa_docs()
unt_q, _ = untouched_queries_and_dbpedia_docs()
q_ex, q_gram = build_protected(six_q + dev_q + unt_q, want_sketch=False)
print(f"query index: {len(q_ex):,} exact, {q_gram.size:,} 8-grams ({time.time()-t0:.0f}s)", flush=True)

summary, kept = {}, {}
for s in SOURCES:
    p = TRAIN / "querytext" / f"{s}.json"
    if not p.exists():
        continue
    qs = json.loads(p.read_text())
    train_idx = [i for i in range(len(qs)) if not heldout(s, str(i))]
    keep, drop = [], {"exact": 0, "near": 0}
    for i in train_idx:
        h = hits(qs[i], q_ex, q_gram, want_sketch=False, min_share=1)
        if h:
            drop[h] += 1
        else:
            keep.append(i)
    kept[s] = keep
    summary[s] = {"n_train": len(train_idx), "n_kept": len(keep), "dropped": drop,
                  "n_heldout": len(qs) - len(train_idx)}
    print(f"{s}: {len(train_idx):,} train -> {len(keep):,} kept  dropped {drop}", flush=True)

(OUT / "kept_querytext.json").write_text(json.dumps(kept))
(REPO / "results" / "m7_decontam_querytext.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
