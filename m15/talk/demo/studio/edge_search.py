"""Search one Qdrant Edge shard with per-request HNSW settings.

Qdrant Edge runs a shard inside this process: no server, no network hop. The shard here (built by
m15src/talk_edge.py) holds 2.68M Stella document vectors (1024-d) stored in fp16 on disk, with binary
quantization codes in RAM and an HNSW graph over them. Every query encoder searches this same index.

Two knobs trade speed for recall on each request, with no rebuild and no re-index:
  hnsw_ef        how many candidates the HNSW graph walk keeps; wider finds more of the true neighbors.
  oversampling   with binary quantization, the 1-bit codes shortlist oversampling x limit candidates,
                 which are then rescored with the full fp16 vectors.
Docs: https://qdrant.tech/documentation/edge/ and https://qdrant.tech/documentation/manage-data/quantization/
"""
import threading
import time
from pathlib import Path

import numpy as np
import qdrant_edge as E


class EdgeSearch:
    def __init__(self, shard_dir: Path, vector_name: str = "stella"):
        self.shard = E.EdgeShard.load(str(shard_dir))
        self.vector_name = vector_name
        self._params = {}
        # In this process the shard serves one search at a time (throughput does not grow with threads), so we
        # make that explicit: the lock lets us time waiting and searching separately, which the router needs.
        self._lock = threading.Lock()

    def params(self, ef: int, oversampling: float) -> E.SearchParams:
        key = (int(ef), float(oversampling))
        if key not in self._params:
            self._params[key] = E.SearchParams(
                hnsw_ef=key[0],
                quantization=E.QuantizationSearchParams(rescore=True, oversampling=key[1]))
        return self._params[key]

    def search(self, vector: np.ndarray, ef: int, oversampling: float, limit: int = 10):
        """Top `limit` point ids, plus milliseconds spent waiting for the shard and searching it."""
        request = E.QueryRequest(
            query=E.Query.Nearest(np.asarray(vector, dtype=np.float32), using=self.vector_name),
            limit=limit,
            params=self.params(ef, oversampling))
        t0 = time.perf_counter()
        with self._lock:
            t1 = time.perf_counter()
            hits = self.shard.query(request)
            t2 = time.perf_counter()
        return [h.id for h in hits], (t1 - t0) * 1000, (t2 - t1) * 1000
