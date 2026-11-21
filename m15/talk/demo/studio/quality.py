"""How good is a search setting? nDCG@10 on judged NQ questions, measured live in a few seconds."""
import math
import threading
import time

import numpy as np


def ndcg10(relevant: dict, ranked: list) -> float:
    """relevant: {doc_id: grade}; ranked: doc ids in rank order. Binary gain, as in the BEIR NQ judgments."""
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if relevant.get(d, 0) > 0)
    n_rel = sum(1 for g in relevant.values() if g > 0)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, n_rel)))
    return dcg / ideal if ideal else 0.0


class QualityBench:
    """Scores (encoder, ef, oversampling) on a fixed, seeded set of judged queries.

    Query vectors are encoded once per encoder and cached, so measuring a new search setting costs only
    searches. Those searches use the same shard as live traffic and count against its time, so they are paced.
    A measurement in progress is dropped if that encoder's setting changes again before it finishes.
    """

    def __init__(self, search, encoders: dict, queries, qids, qrels, doc_ids, n: int = 300, rate: float = 150.0):
        pick = np.random.default_rng(7).choice(len(queries), n, replace=False)
        self.texts = [queries[i] for i in pick]
        self.relevant = [qrels.get(qids[i], {}) for i in pick]
        self.search, self.encoders, self.doc_ids = search, encoders, doc_ids
        self.n, self.interval = n, 1.0 / rate
        self.vectors = {}                    # encoder -> cached query vectors (n x 1024)
        self.scores = {}                     # (encoder, ef, oversampling) -> nDCG@10
        self.search_ms = {}                  # (encoder, ef, oversampling) -> mean shard time per search
        self.progress = {}                   # (encoder, ef, oversampling) -> searches done so far
        self.wanted = {}                     # encoder -> the setting it should be measured at
        threading.Thread(target=self._run, daemon=True, name="quality-bench").start()

    def want(self, encoder: str, ef: int, oversampling: float):
        self.wanted[encoder] = (encoder, int(ef), float(oversampling))

    def status(self, encoder: str, ef: int, oversampling: float) -> dict:
        key = (encoder, int(ef), float(oversampling))
        return {"ndcg10": self.scores.get(key), "search_ms": self.search_ms.get(key),
                "done": self.progress.get(key, 0), "total": self.n}

    def _run(self):
        while True:
            todo = [k for k in list(self.wanted.values()) if k not in self.scores]
            if not todo:
                time.sleep(0.05)
                continue
            self._measure(todo[0])

    def _measure(self, key):
        encoder, ef, oversampling = key
        if encoder not in self.vectors:      # one-time cost per encoder (Stella: a few seconds)
            self.vectors[encoder] = self.encoders[encoder].encode(self.texts)
        total_ndcg, total_ms = 0.0, 0.0
        for i, vector in enumerate(self.vectors[encoder]):
            if self.wanted.get(encoder) != key:              # superseded by a newer setting: discard
                self.progress.pop(key, None)
                return
            t0 = time.perf_counter()
            ids, _, ms = self.search.search(vector, ef, oversampling)
            total_ndcg += ndcg10(self.relevant[i], [self.doc_ids[p] for p in ids])
            total_ms += ms
            self.progress[key] = i + 1
            time.sleep(max(0.0, self.interval - (time.perf_counter() - t0)))
        self.search_ms[key] = total_ms / self.n
        self.scores[key] = total_ndcg / self.n
