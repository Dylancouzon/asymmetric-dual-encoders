"""Traffic-spike demo: live NQ traffic against one Qdrant Edge shard, with a query-encoder dial.

Replays the 3,452 judged NQ test queries as Poisson arrivals at a chosen rate. Worker threads encode
with Stella, Nano or Zero and search the same Stella index. In "auto", each request takes the best
encoder whose measured service time still fits the p95 budget after its queue wait. Every completed
request is scored against the NQ judgments, so the nDCG@10 on screen is measured, not simulated.

  DATASET=nq .venv/bin/python m15/talk/demo/spike.py      # then open http://127.0.0.1:8766
Keys in the page: 1-4 mode, up/down rate, s spike, b budget.
"""
import os
# One thread per library call: 14 workers each spawning BLAS or tokenizer thread pools oversubscribe the cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import collections
import json
import math
import os
import sys
import resource
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "m15src"), str(REPO / "bench")]
import qdrant_edge as E                               # noqa: E402
from fastapi import FastAPI                           # noqa: E402
from fastapi.responses import FileResponse            # noqa: E402
from common import load_public                        # noqa: E402
from talk_throughput import make_encoder              # noqa: E402

DATASET = os.environ.get("DATASET", "nq")
HERE = Path(__file__).parent
EDGE = REPO / "work/m15/edge"
WORKERS = int(os.environ.get("WORKERS", "14"))
MAX_QUEUE = 1500                      # beyond this, arrivals are dropped (counted on screen)
ORDER = ("stella", "nano", "zero")
# Auto's ladder: (rung, encoder, ef, oversampling), best NQ nDCG@10 first, from results/m15_talk_edge_nq.json:
# 0.631, 0.579, 0.554, 0.510, 0.480. The first rung per encoder is its selected setting (fixed modes).
LADDER = (("stella", "stella", 256, 4.0), ("nano", "nano", 512, 4.0), ("nano-fast", "nano", 64, 4.0),
          ("zero", "zero", 1024, 8.0), ("zero-fast", "zero", 128, 4.0))
RUNG = {r: (e, ef, os_) for r, e, ef, os_ in LADDER}

data = load_public(DATASET, with_corpus=False)
QUERIES, QIDS, QRELS = data["q_texts"], data["q_ids"], data["qrels"]
DOC_IDS = json.loads((REPO / "work/m15/vecs/nq_stella/doc_ids.json").read_text())
shard = E.EdgeShard.load(str(EDGE / DATASET))
ENC = {k: make_encoder(k) for k in ORDER}
PARAMS = {r: E.SearchParams(hnsw_ef=ef, quantization=E.QuantizationSearchParams(rescore=True, oversampling=os_))
          for r, (e, ef, os_) in RUNG.items()}


def serve(rung, text):
    e = RUNG[rung][0]
    t0 = time.perf_counter()
    v = ENC[e].encode([text])[0].astype(np.float32)
    t1 = time.perf_counter()
    hits = shard.query(E.QueryRequest(query=E.Query.Nearest(v, using="stella"), limit=10, params=PARAMS[rung]))
    return hits, (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000


# Quiet-time service time per rung (encode + search, sequential), used when a rung has not run lately.
BASE = {}
for r in RUNG:
    for q in QUERIES[:5]:
        serve(r, q)
    ts = []
    for q in QUERIES[5:35]:
        t0 = time.perf_counter(); serve(r, q); ts.append((time.perf_counter() - t0) * 1000)
    BASE[r] = float(np.median(ts))

state = {"rate": 40.0, "mode": "stella", "budget_ms": 100.0, "workers": WORKERS}
queue = collections.deque()
cv = threading.Condition()
done = collections.deque(maxlen=200_000)       # (t_done, latency_ms, encoder, ndcg, rung, queue_ms, enc_ms, search_ms)
dropped = collections.deque(maxlen=200_000)    # t of each dropped arrival
arrivals = collections.deque(maxlen=200_000)   # [t_arrival_wall, ok, trial]; ok None until known
pins = []
trial = {"id": 0, "end": 0.0, "mode": None, "rate": 0.0}   # a timed, seeded trial; 0 = none running
SETTLE_S = 3.0                                  # warm-up before a trial, so the new mode starts warm
paused_until = [0.0]
cpu = {"cores": 0.0, "last": None}
TRIAL_S = float(os.environ.get("TRIAL_S", "12"))
TRIAL_LOG = REPO / "work/m15/logs/spike_trials.jsonl"
reseed = [False]
HEADROOM = 0.85                                 # auto plans to 85% of the budget
service = dict(BASE)                           # EWMA ms under current load
last_used = {r: 0.0 for r in RUNG}


def ndcg10(qid, ids):
    rel = QRELS.get(qid, {})
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ids[:10]) if rel.get(DOC_IDS[d], 0) > 0)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, sum(1 for v in rel.values() if v > 0))))
    return dcg / ideal if ideal else 0.0


# A fixed offered trace: unit-rate exponential gaps and query ids. Each arrival consumes one entry whether or
# not it is dropped, so every trial offers exactly the same traffic.
_trace_rng = np.random.default_rng(0)
GAPS = _trace_rng.exponential(1.0, 2_000_000)
QIDX = _trace_rng.integers(0, len(QUERIES), 2_000_000)


def generator():
    k, nxt = 0, time.perf_counter()
    while True:
        now = time.perf_counter()
        warming = time.time() < paused_until[0]
        if reseed[0] and not warming:     # trial starts: drop warm-up leftovers, replay the trace from its start
            with cv:
                for _, _, rec in queue:
                    rec[1] = False
                queue.clear()
            k, nxt, reseed[0] = 0, now, False
        # Warm-up before a trial: real traffic in the new mode at 30% of the trial rate (at most 300/s), so
        # per-thread model state and disk pages are warm when the measured trace starts.
        rate = min(0.3 * state["rate"], 300.0) if warming else state["rate"]
        with cv:
            while nxt <= now:
                rec = [time.time() - (now - nxt), None,
                       trial["id"] if trial.get("start", 0) <= time.time() < trial["end"] else 0]
                arrivals.append(rec)
                qi, gap = int(QIDX[k % len(QIDX)]), float(GAPS[k % len(GAPS)])
                k += 1
                if len(queue) >= MAX_QUEUE:
                    dropped.append(time.time())
                    rec[1] = False
                else:
                    queue.append((qi, nxt, rec))
                nxt += gap / rate
            cv.notify_all()
        time.sleep(max(0.0, min(0.005, nxt - time.perf_counter())))


def choose(wait_ms):
    if state["mode"] != "auto":
        return state["mode"]          # fixed modes name a ladder rung
    now = time.time()
    for r, *_ in LADDER:                # best quality that still fits the budget after queueing
        if now - last_used[r] >= 2.0:     # stale: forget overload history, restart from quiet time
            service[r] = BASE[r]
        if wait_ms + service[r] <= HEADROOM * state["budget_ms"]:
            return r
    return LADDER[-1][0]


def worker(idx):
    while True:
        with cv:
            while not queue or idx >= state["workers"]:   # workers above the live setting stay idle
                cv.wait(0.05)
            i, t_arr, rec = queue.popleft()
        t0 = time.perf_counter()
        r = choose((t0 - t_arr) * 1000)
        hits, enc_ms, srch_ms = serve(r, QUERIES[i])
        t1 = time.perf_counter()
        service[r] = 0.9 * service[r] + 0.1 * (t1 - t0) * 1000
        last_used[r] = time.time()
        lat, nd = (t1 - t_arr) * 1000, ndcg10(QIDS[i], [h.id for h in hits])
        # A good answer: at least one judged-relevant passage in the top 10, within the budget.
        rec[1] = lat <= state["budget_ms"] and nd > 0
        done.append((time.time(), lat, RUNG[r][0], nd, r, (t0 - t_arr) * 1000, enc_ms, srch_ms, rec[2]))


app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "spike.html")


@app.post("/set")
def set_state(rate: float = None, mode: str = None, budget: float = None, workers: int = None,
              enc: str = None, ef: int = None, os: float = None):
    if rate is not None:
        state["rate"] = max(1.0, min(5000.0, rate))
    if budget is not None:
        state["budget_ms"] = max(1.0, budget)
    if workers is not None:
        state["workers"] = max(1, min(WORKERS, workers))
        with cv:
            cv.notify_all()
    if enc in ORDER and ef and os:      # any encoder with any search setting, as a fixed "custom" mode
        ef, os = max(4, min(4096, int(ef))), max(1.0, min(16.0, float(os)))
        PARAMS["custom"] = E.SearchParams(hnsw_ef=ef, quantization=E.QuantizationSearchParams(rescore=True, oversampling=os))
        RUNG["custom"] = (enc, ef, os)
        BASE.setdefault("custom", BASE[enc]); service["custom"] = BASE[enc]; last_used.setdefault("custom", 0.0)
        mode = "custom"
    if mode in tuple(RUNG) + ("auto",):
        state["mode"] = mode
    return state


@app.post("/reset")
def reset():
    """Empty the queue so each fixed-encoder trial starts from the same state."""
    with cv:
        for _, _, rec in queue:
            rec[1] = False
        queue.clear()
    return {"queue": 0}


@app.post("/pin")
def pin():
    s = snapshot()
    pins.append({k: s[k] for k in ("mode", "rate", "good_pct", "p95_ms", "ndcg10", "served_per_s")})
    return pins


@app.post("/trial")
def start_trial():
    """Empty the queue, replay the seeded trace for TRIAL_S seconds, then pin this trial's own results."""
    reset()
    start = time.time() + SETTLE_S
    paused_until[0] = start
    trial.update(id=trial["id"] + 1, start=start, end=start + TRIAL_S, mode=state["mode"], rate=state["rate"],
                 setting=RUNG.get(state["mode"]), pinned=False)
    reseed[0] = True
    return {k: v for k, v in trial.items()}


def finish_trial():
    tid, now = trial["id"], time.time()
    arr = [a for a in list(arrivals) if a[2] == tid]
    comp = [d for d in list(done) if d[8] == tid]
    lat = [d[1] for d in comp]
    med = lambda j: float(np.median([d[j] for d in comp])) if comp else None
    row = {"mode": trial["mode"], "rate": trial["rate"], "trial": True, "budget_ms": state["budget_ms"],
           "setting": trial.get("setting"), "workers": state["workers"],
           "good_pct": 100.0 * sum(1 for a in arr if a[1]) / len(arr) if arr else 0.0,
           "p95_ms": float(np.percentile(lat, 95)) if lat else None,
           "ndcg10": float(np.mean([d[3] for d in comp])) if comp else 0.0,
           "served_per_s": len(comp) / TRIAL_S, "completed": len(comp), "offered": len(arr),
           "split_ms": {"queue": med(5), "encode": med(6), "search": med(7)}}
    pins.append(row)
    # Keep every trial across restarts (rehearsal and stage); demo observations, not registered results.
    with open(TRIAL_LOG, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "trial_s": TRIAL_S, "workers": WORKERS, **row}) + "\n")
    trial["pinned"] = True


@app.post("/unpin")
def unpin():
    pins.clear()
    return pins


HIST = collections.deque(maxlen=400)          # one /stats sample every 250 ms, last 100 s


def sampler():
    while True:
        # Pin a trial once its last arrival's deadline has passed.
        if trial["id"] and not trial.get("pinned") and time.time() > trial["end"] + state["budget_ms"] / 1000 + 0.5:
            finish_trial()
        t, r = time.time(), resource.getrusage(resource.RUSAGE_SELF)
        used = r.ru_utime + r.ru_stime
        if cpu["last"]:
            dt = t - cpu["last"][0]
            cpu["cores"] = 0.7 * cpu["cores"] + 0.3 * max(0.0, (used - cpu["last"][1]) / dt) if dt > 0 else cpu["cores"]
        cpu["last"] = (t, used)
        HIST.append(snapshot())
        time.sleep(0.25)


@app.get("/stats")
def stats():
    return {**(HIST[-1] if HIST else snapshot()), "history": list(HIST)[-240:]}


def snapshot():
    now = time.time()
    last = [d for d in list(done)[-20000:] if now - d[0] <= 2.0]
    lat = [d[1] for d in last]
    recent = list(done)[-600:]
    sec = [d for d in last if now - d[0] <= 1.0]
    med = lambda j: float(np.median([d[j] for d in sec])) if sec else 0.0
    # Last 500 arrivals whose deadline has passed: drops, late answers and still-queued count as failures.
    due = [a for a in list(arrivals)[-6000:] if now - a[0] >= state["budget_ms"] / 1000][-500:]
    good = 100.0 * sum(1 for a in due if a[1]) / len(due) if due else 0.0
    active = collections.Counter(d[4] for d in sec).most_common(1)
    mix = collections.Counter(d[2] for d in last if now - d[0] <= 1.0)
    return {**state, "t": now, "queue": len(queue),
            "served_per_s": sum(1 for d in last if now - d[0] <= 1.0),
            "dropped_per_s": sum(1 for t in list(dropped)[-20000:] if now - t <= 1.0),
            "p95_ms": float(np.percentile(lat, 95)) if lat else 0.0,
            "p50_ms": float(np.percentile(lat, 50)) if lat else 0.0,
            "ndcg10": float(np.mean([d[3] for d in recent])) if recent else 0.0,
            "mix": {k: mix.get(k, 0) for k in ORDER},
            "rungs": dict(collections.Counter(d[4] for d in sec)),
            "good_pct": good, "pins": pins,
            "split_ms": {"queue": med(5), "encode": med(6), "search": med(7)},
            "active": active[0][0] if active else None,
            "active_setting": (lambda r: f"{RUNG[r][0].capitalize()} · ef {RUNG[r][1]} · os {RUNG[r][2]:g}")(active[0][0]) if active else "",
            "trial_left": max(0.0, trial["end"] - now) if trial["id"] else 0.0,
            "settling": max(0.0, paused_until[0] - now),
            "cores_busy": round(cpu["cores"], 1), "cores_total": os.cpu_count(),
            "setting": {"enc": RUNG[state["mode"]][0], "ef": RUNG[state["mode"]][1], "os": RUNG[state["mode"]][2]} if state["mode"] in RUNG else None,
            "service_ms": {k: round(v, 2) for k, v in service.items()},
            "base_ms": {k: round(v, 2) for k, v in BASE.items()},
            "n_docs": len(DOC_IDS), "workers": WORKERS}


if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=generator, daemon=True).start()
    threading.Thread(target=sampler, daemon=True).start()
    for _ in range(WORKERS):
        threading.Thread(target=worker, args=(_,), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8766, log_level="warning")
