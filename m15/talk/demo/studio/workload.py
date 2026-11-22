"""Live traffic: replay judged questions at a chosen rate, send each to a query encoder, search, score.

Arrivals follow a fixed seeded trace (exponential gaps, query ids), shaped by the controls: rate, a steady /
daily-wave / bursts pattern, a spike switch, and agent tasks (one task = several searches issued together, done
when the slowest returns). One FIFO queue feeds a pool of worker threads; each request is encoded by the chosen
encoder, searched on the shared Edge shard, and scored against the dataset's judgments. Every query is served: when the
queue is full, the replay waits, and each query keeps its scheduled arrival time, so the wait counts in latency.
"""
import collections
import math
import random
import threading
import time

import numpy as np

ENCODERS = ("stella", "nano", "zero")
MAX_QUEUE = 5000                              # bounds memory; beyond it the replay falls behind schedule instead


def ndcg10(relevant: dict, ranked: list) -> float:
    """relevant: {doc_id: grade}; ranked: doc ids in rank order. Binary gain (BEIR's NQ and FiQA grades are 0/1)."""
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if relevant.get(d, 0) > 0)
    n_rel = sum(1 for g in relevant.values() if g > 0)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, n_rel)))
    return dcg / ideal if ideal else 0.0


class Workload:
    def __init__(self, search, encoders, queries, qids, qrels, doc_ids, state: dict, workers: int):
        self.search, self.encoders, self.state = search, encoders, state
        self.queries, self.qids, self.qrels, self.doc_ids = queries, qids, qrels, doc_ids
        self.workers = workers
        rng = np.random.default_rng(0)
        self.gaps, self.qidx = rng.exponential(1.0, 2_000_000), rng.integers(0, len(queries), 2_000_000)
        self.queue, self.cv = collections.deque(), threading.Condition()
        # (t, latency_ms, encoder, ndcg, queue_ms, encode_ms, search_ms, (ef, oversampling))
        self.done = collections.deque(maxlen=100_000)
        self.arrived = collections.deque(maxlen=100_000)
        self.tasks = collections.deque(maxlen=50_000)      # (t, latency of the slowest search in an agent task)
        self.shares = {"stella": 1.0, "nano": 0.0, "zero": 0.0}   # dispatch weights, set by the router
        self.behind_s = 0.0                                        # how far the replay lags its schedule
        self.epoch = time.time()                                   # origin of the wave / bursts pattern
        self._rng = random.Random(1)

    def start(self):
        threading.Thread(target=self._arrivals, daemon=True, name="arrivals").start()
        for i in range(self.workers):
            threading.Thread(target=self._serve, args=(i,), daemon=True, name=f"worker-{i}").start()

    # ---- traffic -------------------------------------------------------------------------------------
    def offered_rate(self, now: float) -> float:
        s = self.state
        if s["paused"]:
            return 0.0
        rate, phase = s["rate"], now - self.epoch
        if s["pattern"] == "wave":            # one "day" per minute, from rate up to rate x peak
            rate *= 1 + (s["peak"] - 1) * (1 - math.cos(2 * math.pi * phase / 60)) / 2
        elif s["pattern"] == "bursts":        # 3 s at rate x peak every 12 s
            rate *= s["peak"] if phase % 12 < 3 else 1
        if s["spike"]:
            rate *= s["peak"]
        return rate

    def _arrivals(self):
        k, nxt = 0, time.perf_counter()
        while True:
            now, wall = time.perf_counter(), time.time()
            rate = self.offered_rate(wall)
            if rate <= 0:                     # paused: the schedule restarts on resume
                nxt, self.behind_s = now, 0.0
                time.sleep(0.02)
                continue
            fanout = max(1, int(self.state["fanout"]))
            with self.cv:
                while nxt <= now and len(self.queue) + fanout <= MAX_QUEUE:
                    group = [fanout, 0.0] if fanout > 1 else None   # searches left, slowest latency
                    arrival = wall - (now - nxt)
                    for j in range(fanout):
                        self.arrived.append(arrival)
                        self.queue.append((int(self.qidx[(k + j) % len(self.qidx)]), nxt, group))
                    k += fanout
                    nxt += float(self.gaps[k % len(self.gaps)]) * fanout / rate
                self.behind_s = max(0.0, now - nxt)
                self.cv.notify_all()
            time.sleep(max(0.0, min(0.005, nxt - time.perf_counter())))

    # ---- serving -------------------------------------------------------------------------------------
    def pick_encoder(self) -> str:
        x, acc = self._rng.random() * (sum(self.shares.values()) or 1.0), 0.0
        for name in ENCODERS:
            acc += self.shares[name]
            if x < acc:
                return name
        return ENCODERS[-1]

    def _serve(self, idx: int):
        while True:
            with self.cv:
                while not self.queue:
                    self.cv.wait(0.05)
                qi, t_arrival, group = self.queue.popleft()
            t0 = time.perf_counter()
            name = self.pick_encoder()
            ef, oversampling = self.state["search"][name]
            vector = self.encoders[name].encode([self.queries[qi]])[0]
            t1 = time.perf_counter()
            ids, _, search_ms = self.search.search(vector, ef, oversampling)
            t2 = time.perf_counter()
            latency = (t2 - t_arrival) * 1000
            score = ndcg10(self.qrels.get(self.qids[qi], {}), [self.doc_ids[p] for p in ids])
            self.done.append((time.time(), latency, name, score, (t0 - t_arrival) * 1000,
                              (t1 - t0) * 1000, search_ms, (ef, oversampling)))
            if group is not None:             # an agent task finishes with its slowest search
                with self.cv:
                    group[0] -= 1
                    group[1] = max(group[1], latency)
                    if group[0] == 0:
                        self.tasks.append((time.time(), group[1]))

    # ---- recent windows ------------------------------------------------------------------------------
    @staticmethod
    def recent(records, seconds: float, now: float, key=lambda r: r[0], limit: int = 1_000_000) -> list:
        """Up to `limit` records from the last `seconds`, newest first, scanning from the newest end only."""
        for _ in range(5):                    # a worker may append mid-scan; then simply rescan
            try:
                out = []
                for r in reversed(records):
                    if now - key(r) > seconds or len(out) >= limit:
                        break
                    out.append(r)
                return out
            except RuntimeError:
                continue
        return [r for r in list(records) if now - key(r) <= seconds]
