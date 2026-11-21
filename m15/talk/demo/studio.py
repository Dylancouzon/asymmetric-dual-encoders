"""Workload studio: live NQ traffic against one Qdrant Edge shard, with every serving choice on a control.

Replays the 3,452 judged NQ test queries from a fixed seeded trace. Traffic has a rate, a pattern (steady,
daily wave, bursts) and a spike switch. Requests are split across Stella, Nano and Zero by weight, all
running at once, each with its own HNSW ef and oversampling; or "auto" picks, per request, the best
(encoder, ef, oversampling) rung whose service time fits the latency target after queueing. Every answer
is scored against the NQ judgments, so nDCG@10 on screen is measured. Machine load is this process's CPU.

  DATASET=nq .venv/bin/python m15/talk/demo/studio.py      # then open http://127.0.0.1:8767
Demo observations only; nothing here is a registered result. Trials append to work/m15/logs/studio_trials.jsonl.
"""
import os
# One thread per library call: 14 workers each spawning BLAS or tokenizer thread pools oversubscribe the cores.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import collections   # noqa: E402
import json          # noqa: E402
import math          # noqa: E402
import random        # noqa: E402
import resource      # noqa: E402
import sys           # noqa: E402
import threading     # noqa: E402
import time          # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np   # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "m15src"), str(REPO / "bench")]
import qdrant_edge as E                               # noqa: E402
from fastapi import FastAPI                           # noqa: E402
from fastapi.responses import FileResponse            # noqa: E402
from common import load_public                        # noqa: E402
from talk_throughput import make_encoder              # noqa: E402

DATASET = os.environ.get("DATASET", "nq")
HERE = Path(__file__).parent
WORKERS = int(os.environ.get("WORKERS", "14"))
MAX_QUEUE = 1500                  # beyond this, arrivals are dropped
TRIAL_S, WARM_S = 12.0, 3.0       # a replay: 3 s warm-up in the new setup, then 12 s of the fixed trace
HEADROOM = 0.85                   # auto plans to 85% of the latency target
ORDER = ("stella", "nano", "zero")
TRIAL_LOG = REPO / "work/m15/logs/studio_trials.jsonl"

data = load_public(DATASET, with_corpus=False)
QUERIES, QIDS, QRELS = data["q_texts"], data["q_ids"], data["qrels"]
DOC_IDS = json.loads((REPO / "work/m15/vecs/nq_stella/doc_ids.json").read_text())
shard = E.EdgeShard.load(str(REPO / "work/m15/edge" / DATASET))
ENC = {k: make_encoder(k) for k in ORDER}

# Defaults: each encoder's selected NQ setting (fastest within 1% of its own exact nDCG@10; Zero: widest tested).
state = {"rate": 40.0, "pattern": "steady", "peak": 15.0, "spike": False, "route": "split",
         "split": {"stella": 1.0, "nano": 0.0, "zero": 0.0},
         "search": {"stella": [256, 4.0], "nano": [512, 4.0], "zero": [1024, 8.0]},
         "budget_ms": 100.0, "workers": WORKERS}
_params = {}


def params(ef, os_):
    key = (int(ef), float(os_))
    if key not in _params:
        _params[key] = E.SearchParams(hnsw_ef=key[0], quantization=E.QuantizationSearchParams(rescore=True, oversampling=key[1]))
    return _params[key]


def ladder():
    """Auto's rungs, best NQ nDCG@10 first: Stella, Nano, Nano ef 64, Zero, Zero ef 128 (0.631 ... 0.480 at defaults)."""
    s = state["search"]
    return (("stella", "stella", *s["stella"]), ("nano", "nano", *s["nano"]), ("nano-fast", "nano", 64, 4.0),
            ("zero", "zero", *s["zero"]), ("zero-fast", "zero", 128, 4.0))


def serve(enc, ef, os_, text):
    t0 = time.perf_counter()
    v = ENC[enc].encode([text])[0].astype(np.float32)
    t1 = time.perf_counter()
    hits = shard.query(E.QueryRequest(query=E.Query.Nearest(v, using="stella"), limit=10, params=params(ef, os_)))
    return hits, (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000


BASE = {}                 # quiet-time service ms per rung, for auto when a rung has not run lately
for r, e, ef, os_ in ladder():
    for q in QUERIES[:5]:
        serve(e, ef, os_, q)
    ts = []
    for q in QUERIES[5:35]:
        t0 = time.perf_counter(); serve(e, ef, os_, q); ts.append((time.perf_counter() - t0) * 1000)
    BASE[r] = float(np.median(ts))
service, last_used = dict(BASE), {r: 0.0 for r in BASE}

queue = collections.deque()
cv = threading.Condition()
done = collections.deque(maxlen=400_000)      # (t, latency, enc, ndcg, rung, queue_ms, enc_ms, search_ms, trial)
arrivals = collections.deque(maxlen=400_000)  # [t_wall, ok, trial]
dropped = collections.deque(maxlen=400_000)
trial = {"id": 0, "start": 0.0, "end": 0.0, "pinned": True}
pins, HIST = [], collections.deque(maxlen=400)
epoch = [time.time()]                         # pattern phase origin; a replay restarts it
reseed = [False]
cpu = {"cores": 0.0, "last": None}

# Fixed offered trace: unit-rate exponential gaps and query ids; each arrival consumes one entry, dropped or not.
_rng = np.random.default_rng(0)
GAPS, QIDX = _rng.exponential(1.0, 2_000_000), _rng.integers(0, len(QUERIES), 2_000_000)


def current_rate(t):
    """Arrival rate now: base rate shaped by the pattern, times the peak while the spike switch is on."""
    r, p, ph = state["rate"], state["peak"], t - epoch[0]
    if state["pattern"] == "wave":            # one "day" per minute: rate .. rate x peak
        r *= 1 + (p - 1) * (1 - math.cos(2 * math.pi * ph / 60)) / 2
    elif state["pattern"] == "bursts":        # a 3 s burst at rate x peak every 12 s
        r *= p if ph % 12 < 3 else 1
    if state["spike"]:
        r *= p
    return max(1.0, r)


def ndcg10(qid, ids):
    rel = QRELS.get(qid, {})
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ids[:10]) if rel.get(DOC_IDS[d], 0) > 0)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, sum(1 for v in rel.values() if v > 0))))
    return dcg / ideal if ideal else 0.0


def generator():
    k, nxt = 0, time.perf_counter()
    while True:
        now, wall = time.perf_counter(), time.time()
        warming = trial["id"] and not trial["pinned"] and wall < trial["start"]
        if reseed[0] and not warming:         # replay starts: drop warm-up leftovers, restart trace and pattern
            with cv:
                for item in queue:
                    item[2][1] = False
                queue.clear()
            k, nxt, reseed[0], epoch[0] = 0, now, False, wall
        rate = min(0.3 * current_rate(wall), 300.0) if warming else current_rate(wall)
        with cv:
            while nxt <= now:
                tid = trial["id"] if trial["start"] <= wall < trial["end"] else 0
                rec = [wall - (now - nxt), None, tid]
                arrivals.append(rec)
                qi, gap = int(QIDX[k % len(QIDX)]), float(GAPS[k % len(GAPS)])
                k += 1
                if len(queue) >= MAX_QUEUE:
                    dropped.append(wall)
                    rec[1] = False
                else:
                    queue.append((qi, nxt, rec))
                nxt += gap / rate
            cv.notify_all()
        time.sleep(max(0.0, min(0.005, nxt - time.perf_counter())))


_route_rng = random.Random(1)


def choose(wait_ms):
    """Returns (rung, encoder, ef, oversampling) for one request."""
    if state["route"] == "auto":
        now = time.time()
        rungs = ladder()
        for r, e, ef, os_ in rungs:
            if now - last_used[r] >= 2.0:     # stale: forget overload history, restart from quiet time
                service[r] = BASE[r]
            if wait_ms + service[r] <= HEADROOM * state["budget_ms"]:
                return r, e, ef, os_
        return rungs[-1]
    w = state["split"]
    tot = sum(w.values()) or 1.0
    x, acc = _route_rng.random() * tot, 0.0
    for e in ORDER:                            # weighted split: all encoders serve at once
        acc += w[e]
        if x < acc:
            return (e, e, *state["search"][e])
    return ("zero", "zero", *state["search"]["zero"])


def worker(idx):
    while True:
        with cv:
            while not queue or idx >= state["workers"]:
                cv.wait(0.05)
            i, t_arr, rec = queue.popleft()
        t0 = time.perf_counter()
        r, e, ef, os_ = choose((t0 - t_arr) * 1000)
        hits, enc_ms, srch_ms = serve(e, ef, os_, QUERIES[i])
        t1 = time.perf_counter()
        if r in service:
            service[r] = 0.9 * service[r] + 0.1 * (t1 - t0) * 1000
            last_used[r] = time.time()
        lat, nd = (t1 - t_arr) * 1000, ndcg10(QIDS[i], [h.id for h in hits])
        rec[1] = lat <= state["budget_ms"] and nd > 0
        done.append((time.time(), lat, e, nd, r, (t0 - t_arr) * 1000, enc_ms, srch_ms, rec[2]))


app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "studio.html")


@app.post("/set")
def set_state(body: dict):
    """Partial update: rate, pattern, peak, spike, route, split{enc: w}, search{enc: [ef, os]}, budget_ms, workers."""
    if "rate" in body:
        state["rate"] = max(1.0, min(5000.0, float(body["rate"])))
    if body.get("pattern") in ("steady", "wave", "bursts"):
        state["pattern"] = body["pattern"]
        epoch[0] = time.time()
    if "peak" in body:
        state["peak"] = max(1.0, min(100.0, float(body["peak"])))
    if "spike" in body:
        state["spike"] = bool(body["spike"])
    if body.get("route") in ("split", "auto"):
        state["route"] = body["route"]
    for e, v in (body.get("split") or {}).items():
        if e in ORDER:
            state["split"][e] = max(0.0, float(v))
    for e, v in (body.get("search") or {}).items():
        if e in ORDER:
            state["search"][e] = [max(4, min(4096, int(v[0]))), max(1.0, min(16.0, float(v[1])))]
    if "budget_ms" in body:
        state["budget_ms"] = max(1.0, float(body["budget_ms"]))
    if "workers" in body:
        state["workers"] = max(1, min(WORKERS, int(body["workers"])))
        with cv:
            cv.notify_all()
    return state


@app.post("/reset")
def reset():
    with cv:
        for item in queue:
            item[2][1] = False
        queue.clear()
    return {"queue": 0}


@app.post("/trial")
def start_trial():
    """Replay the same traffic for the current setup: warm up, then the fixed trace and pattern for TRIAL_S."""
    reset()
    start = time.time() + WARM_S
    trial.update(id=trial["id"] + 1, start=start, end=start + TRIAL_S, pinned=False,
                 config=json.loads(json.dumps({k: state[k] for k in ("rate", "pattern", "peak", "spike", "route", "split", "search", "budget_ms", "workers")})))
    reseed[0] = True
    return trial


@app.post("/unpin")
def unpin():
    pins.clear()
    return pins


def finish_trial():
    tid = trial["id"]
    arr = [a for a in list(arrivals) if a[2] == tid]
    comp = [d for d in list(done) if d[8] == tid]
    lat = [d[1] for d in comp]
    med = lambda j: float(np.median([d[j] for d in comp])) if comp else None
    per = {e: {"share": sum(1 for d in comp if d[2] == e) / len(comp) if comp else 0.0} for e in ORDER}
    row = {**trial["config"], "p95_ms": float(np.percentile(lat, 95)) if lat else None,
           "ndcg10": float(np.mean([d[3] for d in comp])) if comp else 0.0,
           "good_pct": 100.0 * sum(1 for a in arr if a[1]) / len(arr) if arr else 0.0,
           "served_per_s": len(comp) / TRIAL_S, "completed": len(comp), "offered": len(arr),
           "cores_busy": cpu["cores"], "mix": per,
           "split_ms": {"queue": med(5), "encode": med(6), "search": med(7)}}
    pins.append(row)
    with open(TRIAL_LOG, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "trial_s": TRIAL_S, **row}) + "\n")
    trial["pinned"] = True


def snapshot():
    now = time.time()
    tail = list(done)[-40000:]
    last = [d for d in tail if now - d[0] <= 2.0]
    sec = [d for d in last if now - d[0] <= 1.0]
    lat = [d[1] for d in last]
    pct = lambda xs, q: float(np.percentile(xs, q)) if xs else 0.0
    med = lambda j: float(np.median([d[j] for d in sec])) if sec else 0.0
    per = {}
    for e in ORDER:
        le = [d for d in last if d[2] == e]
        recent = [d[3] for d in tail[-6000:] if d[2] == e][-300:]
        per[e] = {"served_per_s": sum(1 for d in le if now - d[0] <= 1.0), "p95_ms": pct([d[1] for d in le], 95),
                  "ndcg10": float(np.mean(recent)) if recent else None, "share": (sum(1 for d in sec if d[2] == e) / len(sec)) if sec else 0.0}
    recent = tail[-600:]
    served = len(sec)
    due = [a for a in list(arrivals)[-8000:] if now - a[0] >= state["budget_ms"] / 1000][-500:]
    return {**state, "t": now, "arrival_rate": current_rate(now), "queue": len(queue),
            "served_per_s": served, "dropped_per_s": sum(1 for t in list(dropped)[-20000:] if now - t <= 1.0),
            "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95),
            "ndcg10": float(np.mean([d[3] for d in recent])) if recent else 0.0,
            "good_pct": 100.0 * sum(1 for a in due if a[1]) / len(due) if due else 0.0,
            "per": per, "split_ms": {"queue": med(5), "encode": med(6), "search": med(7)},
            "cores_busy": round(cpu["cores"], 1), "cores_total": os.cpu_count(),
            "cpu_s_per_1k": (cpu["cores"] / served * 1000) if served else None,
            "trial_left": max(0.0, trial["end"] - now) if not trial["pinned"] else 0.0,
            "warming": max(0.0, trial["start"] - now) if not trial["pinned"] else 0.0,
            "pins": pins, "n_docs": len(DOC_IDS), "max_workers": WORKERS}


def sampler():
    while True:
        if not trial["pinned"] and time.time() > trial["end"] + state["budget_ms"] / 1000 + 0.5:
            finish_trial()
        t, r = time.time(), resource.getrusage(resource.RUSAGE_SELF)
        used = r.ru_utime + r.ru_stime
        if cpu["last"] and t > cpu["last"][0]:
            cpu["cores"] = 0.7 * cpu["cores"] + 0.3 * max(0.0, (used - cpu["last"][1]) / (t - cpu["last"][0]))
        cpu["last"] = (t, used)
        HIST.append(snapshot())
        time.sleep(0.25)


@app.get("/stats")
def stats():
    s = HIST[-1] if HIST else snapshot()
    keep = ("t", "arrival_rate", "served_per_s", "p95_ms", "budget_ms", "ndcg10", "cores_busy")
    return {**s, "history": [{**{k: h[k] for k in keep}, "by": {e: h["per"][e]["served_per_s"] for e in ORDER}}
                             for h in list(HIST)[-240:]]}


if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=generator, daemon=True).start()
    threading.Thread(target=sampler, daemon=True).start()
    for i in range(WORKERS):
        threading.Thread(target=worker, args=(i,), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8767")), log_level="warning")
