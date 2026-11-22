"""Talk demo server: one Qdrant Edge shard, three query encoders plus BM25, all in this process.

  DATASET=nq .venv/bin/python m15/talk/demo/app.py      # then open http://127.0.0.1:8765
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "m15src"), str(REPO / "bench")]
import encoders15                     # noqa: E402
import qdrant_edge as E               # noqa: E402
from fastapi import FastAPI           # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

DATASET = os.environ.get("DATASET", "nq")
EDGE = REPO / "work/m15/edge"
HERE = Path(__file__).parent

shard = E.EdgeShard.load(str(EDGE / DATASET))
meta = json.loads((EDGE / f"{DATASET}.meta.json").read_text())
bm25 = E.Bm25(E.Bm25Config(avg_len=meta["bm25_avg_len"]))
ENC = {"stella": encoders15.stella_query(), "nano": encoders15.nano(), "zero": encoders15.ZeroEncoder()}
for e in ENC.values():
    e.encode(["warm up the encoder"])


def load_matrix():
    # Exact search uses the same fp16 document vectors the shard was built from (point id = row).
    if DATASET == "nq":
        d = REPO / "work/m15/vecs/nq_stella"
        return np.concatenate([np.load(p) for p in sorted(d.glob("shard_*.npy"))])
    return np.load(REPO / f"artifacts/demo-stella/{DATASET}/doc_vecs.npy")


DOCS = load_matrix()
RACE = REPO / f"results/m15_talk_edge_{DATASET}.json"
app = FastAPI()


def exact_top10(v, chunk=524_288):
    best = []
    for lo in range(0, len(DOCS), chunk):
        s = DOCS[lo:lo + chunk].astype(np.float32) @ v
        top = np.argpartition(-s, 10)[:10]
        best += [(float(s[i]), lo + int(i)) for i in top]
    return {i for _, i in sorted(best, reverse=True)[:10]}


def hits_json(hits):
    return [{"id": h.id, "score": round(float(h.score), 4), "title": h.payload.get("title", ""),
             "text": h.payload.get("text", "")[:260]} for h in hits]


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/race")
def race():
    if not RACE.exists():
        return JSONResponse({"error": f"{RACE.name} missing; run talk_edge.py bench {DATASET}"}, 404)
    r = json.loads(RACE.read_text())
    corpus = {"nq": ("Wikipedia passages", "NQ"), "fiqa": ("finance forum posts", "FiQA")}[DATASET]
    return {"dataset": DATASET, "n_docs": r["n_docs"], "n_queries": r["n_queries"],
            "corpus": corpus[0], "qset": corpus[1],
            "lanes": {k: r["results"][k] for k in ("stella", "nano", "zero", "bm25")}}


@app.get("/search")
def search(q: str, ef: int = 128, oversampling: float = 2.0, exact: int = 0, k: int = 5):
    params = E.SearchParams(hnsw_ef=ef, quantization=E.QuantizationSearchParams(
        rescore=True, oversampling=oversampling))
    out = {"n_docs": len(DOCS), "ef": ef, "oversampling": oversampling, "lanes": {}}
    for name, enc in ENC.items():
        t0 = time.perf_counter()
        v = enc.encode([q])[0].astype(np.float32)
        t1 = time.perf_counter()
        hits = shard.query(E.QueryRequest(query=E.Query.Nearest(v, using="stella"), limit=10,
                                          params=params, with_payload=True))
        t2 = time.perf_counter()
        lane = {"encode_ms": (t1 - t0) * 1000, "search_ms": (t2 - t1) * 1000,
                "hits": hits_json(hits[:k])}
        if exact:
            ex = exact_top10(v)
            lane["recovered"] = len(ex & {h.id for h in hits})
        out["lanes"][name] = lane
    t0 = time.perf_counter()
    hits = shard.query(E.QueryRequest(query=E.Query.Nearest(bm25.embed_query(q), using="bm25"),
                                      limit=k, with_payload=True))
    out["lanes"]["bm25"] = {"encode_ms": None, "search_ms": (time.perf_counter() - t0) * 1000,
                            "hits": hits_json(hits)}
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
