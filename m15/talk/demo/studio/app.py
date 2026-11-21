"""Workload studio: live NQ traffic on one Qdrant Edge index, with Stella, Nano and Zero as query encoders.

All three encoders search the same Stella index (2.68M passages). Traffic is replayed from judged NQ questions,
so every answer is scored. Auto routing plans, four times a second, which share of traffic each encoder serves:
the best average quality at or above the floor that fits the machine's encode and search capacity, with headroom
tuned from measured p95 against the latency target (see allocator.py). Every control applies immediately.

  DATASET=nq .venv/bin/python m15/talk/demo/studio/app.py      # then open http://127.0.0.1:8767
Demo observations only, not registered results.
"""
import os

# One thread per library call: many workers each spawning BLAS or tokenizer pools would oversubscribe the CPU.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import collections  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).parent
REPO = HERE.parents[3]
sys.path[:0] = [str(HERE), str(REPO / "m15src"), str(REPO / "bench")]
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

import allocator  # noqa: E402
from common import load_public  # noqa: E402
from edge_search import EdgeSearch  # noqa: E402
from quality import QualityBench  # noqa: E402
from talk_throughput import make_encoder  # noqa: E402
from workload import ENCODERS, Workload  # noqa: E402

DATASET = os.environ.get("DATASET", "nq")
WORKERS = int(os.environ.get("WORKERS", "14"))

# ---- data, index, encoders --------------------------------------------------------------------------
data = load_public(DATASET, with_corpus=False)
DOC_IDS = json.loads((REPO / "work/m15/vecs/nq_stella/doc_ids.json").read_text())
search = EdgeSearch(REPO / "work/m15/edge" / DATASET)
encoders = {name: make_encoder(name) for name in ENCODERS}

# Defaults: each encoder's selected NQ setting (fastest within 1% of its own exact nDCG@10; Zero: widest tested).
state = {"rate": 40.0, "pattern": "steady", "peak": 15.0, "spike": False, "fanout": 1, "paused": False,
         "route": "auto", "split": {"stella": 34.0, "nano": 33.0, "zero": 33.0},
         "search": {"stella": [256, 4.0], "nano": [512, 4.0], "zero": [1024, 8.0]},
         "floor": 0.50, "target_ms": 150.0}

bench = QualityBench(search, encoders, data["q_texts"], data["q_ids"], data["qrels"], DOC_IDS, n=1000, rate=300)
work = Workload(search, encoders, data["q_texts"], data["q_ids"], data["qrels"], DOC_IDS, state, WORKERS)

# Until a new setting is measured, its quality and search time come from the offline NQ sweep (labelled "estimate").
SWEEP = {(e, s["ef"], s["oversampling"]): s for e in ENCODERS for s in
         json.loads((REPO / f"results/m15_talk_edge_{DATASET}.json").read_text())["results"][e]["settings"]}


def sweep_estimate(name, ef, oversampling):
    near = min((k for k in SWEEP if k[0] == name), key=lambda k: (abs(np.log(k[1] / ef)), abs(k[2] - oversampling)))
    return SWEEP[near]["ndcg10"], SWEEP[near]["search_ms_p50"]


# ---- calibration: what this machine sustains ---------------------------------------------------------
calib = {"encode_per_s": {}, "encode_ms": {}, "ready": False, "step": "loading"}


def _saturate(name, seconds):
    """Encodes per second for one encoder with every worker busy (encode only)."""
    stop, counts = time.time() + seconds, [0] * WORKERS

    def loop(i):
        q = data["q_texts"]
        n = i * 101
        while time.time() < stop:
            encoders[name].encode([q[n % len(q)]])
            counts[i] += 1
            n += 1
    threads = [threading.Thread(target=loop, args=(i,)) for i in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sum(counts) / seconds


def warm_index_pages():
    """Read the shard's vector files once so searches see warm pages; cold reads would skew every timing."""
    for f in sorted((REPO / "work/m15/edge" / DATASET / "segments").glob("*/vector_storage-*/**/*")):
        if f.is_file():
            with open(f, "rb") as fh:
                while fh.read(64 << 20):
                    pass


def calibrate():
    calib["step"] = "warming the index"
    warm_index_pages()
    calib["step"] = "warming up"
    for name in ENCODERS:                     # the first full-load burst after launch runs several times slower
        _saturate(name, 4 if name != "stella" else 3)
    calib["step"] = "measuring capacity"
    for name in ENCODERS:
        calib["encode_per_s"][name] = _saturate(name, 2.5)
        ts = []
        for q in data["q_texts"][:30]:
            t0 = time.perf_counter()
            encoders[name].encode([q])
            ts.append((time.perf_counter() - t0) * 1000)
        calib["encode_ms"][name] = float(np.median(ts))
    calib["step"] = "measuring quality"
    for name in ENCODERS:
        bench.want(name, *state["search"][name])
    while any(bench.status(n, *state["search"][n])["ndcg10"] is None for n in ENCODERS):
        time.sleep(0.2)
    work.start()
    calib["ready"], calib["step"] = True, "ready"
    print("studio ready:", {k: round(v) for k, v in calib["encode_per_s"].items()}, flush=True)


# ---- routing: plan four times a second ---------------------------------------------------------------
router = {"plan": None, "headroom": 0.8, "costs": {}, "over": 0}
service_p95 = {}                                         # encoder -> p95 of its own encode + search time, ms
config_since = {name: 0.0 for name in ENCODERS}          # when each encoder's search setting last changed


def live_search_ms(name, now):
    """Median shard time of this encoder's recent served searches at its current setting (None if too few)."""
    xs = [d[6] for d in work.recent(work.done, 3.0, now) if d[2] == name and d[0] >= config_since[name]]
    return float(np.median(xs)) if len(xs) >= 20 else None


def service_time(name, now, idle_ms):
    """p95 of this encoder's own processing time (encode + search, no queueing) under the current load. Sticky: with
    no recent traffic it relaxes toward the idle-machine time within a few seconds, so an encoder cannot flap."""
    xs = [d[5] + d[6] for d in work.recent(work.done, 3.0, now) if d[2] == name]
    prev = service_p95.get(name, idle_ms)
    service_p95[name] = float(np.percentile(xs, 95)) if len(xs) >= 10 else idle_ms + (prev - idle_ms) * 0.9
    return service_p95[name]


def costs_now():
    """Each encoder's quality and cost at its current search setting: measured when available, else estimated."""
    out, now = {}, time.time()
    for name in ENCODERS:
        ef, oversampling = state["search"][name]
        st = bench.status(name, ef, oversampling)
        est_q, est_ms = sweep_estimate(name, ef, oversampling)
        quality = st["ndcg10"] if st["ndcg10"] is not None else est_q
        # Search time: live from served searches; else the sweep's warm timing; else the bench (off-grid settings).
        live = live_search_ms(name, now)
        in_sweep = (name, int(ef), float(oversampling)) in SWEEP
        search_ms = live if live is not None else est_ms if in_sweep else st["search_ms"] if st["search_ms"] is not None else est_ms
        out[name] = {"quality": quality, "measured": st["ndcg10"] is not None, "done": st["done"],
                     "total": st["total"], "search_ms": search_ms, "ef": ef, "oversampling": oversampling,
                     "encode_per_s": calib["encode_per_s"].get(name, 1.0),
                     "service_p95_ms": service_time(name, now, calib["encode_ms"].get(name, 0.0) + search_ms)}
    return out


def plan_loop():
    while True:
        time.sleep(0.25)
        if not calib["ready"]:
            continue
        now = time.time()
        for name in ENCODERS:
            bench.want(name, *state["search"][name])
        costs = costs_now()
        router["costs"] = costs
        rate = work.offered_rate(now)
        if state["route"] == "split":
            work.shares, work.admit, router["plan"] = dict(state["split"]), 1.0, None
            continue
        # Headroom from measured p95 (per agent task when searches come in groups): back off fast, recover slowly.
        source = work.tasks if state["fanout"] > 1 else work.done
        lat = [r[1] for r in work.recent(source, 2.0, now)]
        p95 = float(np.percentile(lat, 95)) if len(lat) >= 20 else 0.0
        router["over"] = router["over"] + 1 if p95 > state["target_ms"] else 0
        h = router["headroom"]
        if router["over"] >= 2:                   # two readings in a row over target: back off, harder when far over
            h = max(0.4, h * (0.9 if p95 > 1.3 * state["target_ms"] else 0.97))
        elif p95 < 0.7 * state["target_ms"]:     # comfortably under: recover in a few seconds
            h = min(0.85, h + 0.02)
        router["headroom"] = h
        p = allocator.plan(max(rate, 1.0),
                           [allocator.EncoderCost(n, c["quality"], c["encode_per_s"], c["search_ms"], c["service_p95_ms"])
                            for n, c in costs.items()],
                           floor=state["floor"], target_ms=state["target_ms"], headroom=h)
        router["plan"] = p
        work.shares = dict(p.shares)
        work.admit = 1.0 if rate <= 0 else min(1.0, p.serve_rate / rate)


# ---- stats and the timeline --------------------------------------------------------------------------
HISTORY, EVENTS = collections.deque(maxlen=400), collections.deque(maxlen=40)
cpu = {"cores": 0.0, "last": None}


def snapshot():
    import resource
    now = time.time()
    r = resource.getrusage(resource.RUSAGE_SELF)
    used = r.ru_utime + r.ru_stime
    if cpu["last"] and now > cpu["last"][0]:
        cpu["cores"] = 0.7 * cpu["cores"] + 0.3 * max(0.0, (used - cpu["last"][1]) / (now - cpu["last"][0]))
    cpu["last"] = (now, used)
    last2 = work.recent(work.done, 2.0, now)
    last1 = [d for d in last2 if now - d[0] <= 1.0]
    lat = [d[1] for d in last2]
    tasks = [t[1] for t in work.recent(work.tasks, 2.0, now)]
    quality = [d[3] for d in work.recent(work.done, 3600, now, limit=600)]     # the last 600 served searches
    p = router["plan"]
    by = {n: sum(1 for d in last1 if d[2] == n) for n in ENCODERS}
    return {
        "t": now, "ready": calib["ready"], "step": calib["step"], **{k: state[k] for k in state},
        "offered": work.offered_rate(now), "arrived_per_s": len(work.recent(work.arrived, 1.0, now, key=lambda t: t)),
        "served_per_s": len(last1), "dropped_per_s": len(work.recent(work.dropped, 1.0, now, key=lambda t: t)),
        "queue": len(work.queue), "p95_ms": float(np.percentile(lat, 95)) if lat else 0.0,
        "task_p95_ms": float(np.percentile(tasks, 95)) if tasks else 0.0,
        "ndcg10": float(np.mean(quality)) if quality else 0.0, "n_quality": len(quality),
        "cores": round(cpu["cores"], 1), "cores_total": os.cpu_count(), "by": by,
        "actual_share": {n: (by[n] / len(last1)) if last1 else 0.0 for n in ENCODERS},
        "plan": None if p is None else {"shares": p.shares, "quality": p.quality, "encode_util": p.encode_util,
                                        "shard_util": p.shard_util, "serve_rate": p.serve_rate, "status": p.status},
        "headroom": router["headroom"], "costs": router["costs"], "n_docs": len(DOC_IDS),
        "events": [e for e in EVENTS if now - e[0] <= 60]}


def sample_loop():
    while True:
        HISTORY.append(snapshot())
        time.sleep(0.25)


# ---- HTTP ---------------------------------------------------------------------------------------------
app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/stats")
def stats():
    s = HISTORY[-1] if HISTORY else snapshot()
    keep = ("t", "offered", "served_per_s", "p95_ms", "task_p95_ms", "target_ms", "ndcg10", "floor", "by", "fanout")
    return {**s, "history": [{k: h[k] for k in keep} for h in list(HISTORY)[-240:]]}


@app.post("/set")
def update(body: dict):
    """Partial update of the controls; applies to the next request. Traffic never stops unless paused."""
    before = json.loads(json.dumps(state))
    for key in ("rate", "peak", "floor", "target_ms"):
        if key in body:
            state[key] = float(body[key])
    state["rate"] = min(max(state["rate"], 1.0), 5000.0)
    for key in ("spike", "paused"):
        if key in body:
            state[key] = bool(body[key])
    if "fanout" in body:
        state["fanout"] = min(max(int(body["fanout"]), 1), 64)
    if body.get("pattern") in ("steady", "wave", "bursts"):
        state["pattern"], work.epoch = body["pattern"], time.time()
    if body.get("route") in ("auto", "split"):
        state["route"] = body["route"]
    for name, w in (body.get("split") or {}).items():
        if name in ENCODERS:
            state["split"][name] = max(0.0, float(w))
    for name, (ef, oversampling) in (body.get("search") or {}).items():
        if name in ENCODERS:
            new = [min(max(int(ef), 4), 4096), min(max(float(oversampling), 1.0), 16.0)]
            if new != state["search"][name]:
                state["search"][name], config_since[name] = new, time.time()
    marker = _describe(before, state)
    if marker:
        EVENTS.append((time.time(), marker))
    return state


def _describe(a, b):
    """One short timeline marker for a control change, the most important first."""
    if a["paused"] != b["paused"]:
        return "paused" if b["paused"] else "resumed"
    if a["floor"] != b["floor"]:
        return f"floor {b['floor']:.2f}"
    if a["route"] != b["route"]:
        return b["route"]
    for name in ENCODERS:
        if a["search"][name] != b["search"][name]:
            return f"{name} ef {b['search'][name][0]} ×{b['search'][name][1]:g}"
    if a["rate"] != b["rate"]:
        return f"{b['rate']:,.0f}/s"
    if a["spike"] != b["spike"]:
        return f"spike ×{b['peak']:g}" if b["spike"] else "spike off"
    if a["fanout"] != b["fanout"]:
        return "agent tasks" if b["fanout"] > 1 else "single searches"
    if a["target_ms"] != b["target_ms"]:
        return f"target {b['target_ms']:.0f} ms"
    if a["pattern"] != b["pattern"]:
        return b["pattern"]
    return None


if __name__ == "__main__":
    import uvicorn
    for target in (calibrate, plan_loop, sample_loop):
        threading.Thread(target=target, daemon=True, name=target.__name__).start()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8767")), log_level="warning")
