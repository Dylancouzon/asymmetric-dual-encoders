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
    while any(bench.status(n, *state["search"][n])["ndcg10"] is None for n in ENCODERS):
        for name in ENCODERS:
            bench.want(name, *state["search"][name])
        time.sleep(0.2)
    work.start()
    calib["ready"], calib["step"] = True, "ready"
    print("studio ready:", {k: round(v) for k, v in calib["encode_per_s"].items()}, flush=True)


# ---- routing: plan four times a second ---------------------------------------------------------------
router = {"plan": None, "headroom": 0.8, "costs": {}, "adjusted": 0.0}
probes = collections.deque(maxlen=5000)                  # (t, encoder, encode_ms) timed under the live load
config_since = {name: 0.0 for name in ENCODERS}          # when each encoder's search setting last changed


def live_search_ms(name, recent):
    """Mean shard time of this encoder's recent served searches at its current setting (None if too few)."""
    xs = [d[6] for d in recent if d[2] == name and d[0] >= config_since[name]]
    return float(np.mean(xs)) if len(xs) >= 20 else None


def probe_loop():
    """Twice a second, time one encode per encoder under whatever load the machine carries, so Auto knows what an
    encoder would cost right now even while it serves no traffic."""
    i = 0
    while True:
        time.sleep(0.5)
        if not calib["ready"]:
            continue
        for name in ENCODERS:
            i += 1
            t0 = time.perf_counter()
            encoders[name].encode([data["q_texts"][i % len(data["q_texts"])]])
            probes.append((time.time(), name, (time.perf_counter() - t0) * 1000))


def service_time(name, recent, search_ms, now):
    """Tail of this encoder's own processing time (encode + search, no queueing) under the current load, from served
    searches and probes (10 s of probes: an encoder's time under load varies too much to judge from a few). A traffic
    change clears the probes, since they were timed under the old load. An agent task waits for the slowest of its searches, so the tail percentile rises with
    fan-out: P(max of k <= x) = F(x)^k >= 0.95 needs F(x) >= 0.95^(1/k)."""
    xs = [d[5] + d[6] for d in recent if d[2] == name]
    xs += [p[2] + search_ms for p in Workload.recent(probes, 10.0, now) if p[1] == name]
    q = 100 * 0.95 ** (1 / max(1, state["fanout"]))
    return float(np.percentile(xs, q)) if len(xs) >= 5 else calib["encode_ms"].get(name, 0.0) + search_ms


def costs_now():
    """Each encoder's quality and cost at its current search setting: measured when available, else estimated."""
    now = time.time()
    out, recent = {}, work.recent(work.done, 3.0, now)
    for name in ENCODERS:
        ef, oversampling = state["search"][name]
        st = bench.status(name, ef, oversampling)
        est_q, est_ms = sweep_estimate(name, ef, oversampling)
        quality = st["ndcg10"] if st["ndcg10"] is not None else est_q
        # Search time: live from served searches; else the sweep's warm timing; else the bench (off-grid settings).
        live = live_search_ms(name, recent)
        in_sweep = (name, int(ef), float(oversampling)) in SWEEP
        search_ms = live if live is not None else est_ms if in_sweep else st["search_ms"] if st["search_ms"] is not None else est_ms
        out[name] = {"quality": quality, "measured": st["ndcg10"] is not None, "done": st["done"],
                     "total": st["total"], "search_ms": search_ms, "ef": ef, "oversampling": oversampling,
                     "encode_per_s": calib["encode_per_s"].get(name, 1.0),
                     "service_ms": service_time(name, recent, search_ms, now)}
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
            # Manual split: no shedding, so the room sees what a fixed split does to latency.
            work.shares, work.admit, work.deadline_ms, router["plan"] = dict(state["split"]), 1.0, float("inf"), None
            continue
        if rate <= 0:                             # paused: keep the last plan while the queue drains
            continue
        # Headroom from measured p95 (per agent task when searches come in groups), once per fresh second of
        # completions: re-reading overlapping windows would punish the same slow requests several times.
        # Shedding by deadline keeps latency bounded at once; it also says the plan is too optimistic.
        target = state["target_ms"]
        work.deadline_ms = 0.8 * target
        if now - router["adjusted"] >= 1.0:
            router["adjusted"] = now
            source = work.tasks if state["fanout"] > 1 else work.done
            lat = [r[1] for r in work.recent(source, 1.0, now)]
            expired = len(work.recent(work.expired, 1.0, now, key=lambda t: t))
            if len(lat) >= 10:
                p95 = float(np.percentile(lat, 95))
                h = router["headroom"]
                if p95 > target or expired > 0.01 * rate:          # over target or shedding late: back off
                    h = max(0.3, h * (0.85 if p95 > 1.3 * target or expired > 0.1 * rate else 0.95))
                elif p95 < 0.7 * target and not expired:            # comfortably under: recover slowly
                    h = min(0.85, h + 0.01)
                router["headroom"] = h
        p = allocator.plan(rate,
                           [allocator.EncoderCost(n, c["quality"], c["encode_per_s"], c["search_ms"], c["service_ms"])
                            for n, c in costs.items()],
                           floor=state["floor"], target_ms=state["target_ms"], headroom=router["headroom"],
                           shard_reserved=bench.shard_busy)
        router["plan"] = p
        work.shares = dict(p.shares)
        work.admit = min(1.0, p.serve_rate / rate)


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
    # Served quality over the last 3 s, or the last 600 searches when traffic is light: fewer would be mostly noise.
    quality = [d[3] for d in work.recent(work.done, 3.0, now)]
    if len(quality) < 600:
        quality = [d[3] for d in work.recent(work.done, 3600, now, limit=600)]
    p = router["plan"]
    by = {n: sum(1 for d in last1 if d[2] == n) for n in ENCODERS}
    return {
        "t": now, "ready": calib["ready"], "step": calib["step"], **{k: state[k] for k in state},
        "offered": work.offered_rate(now), "arrived_per_s": len(work.recent(work.arrived, 1.0, now, key=lambda t: t)),
        "served_per_s": len(last1), "dropped_per_s": len(work.recent(work.dropped, 1.0, now, key=lambda t: t)) +
                                                    len(work.recent(work.expired, 1.0, now, key=lambda t: t)),
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
    keep = ("t", "arrived_per_s", "served_per_s", "p95_ms", "task_p95_ms", "target_ms", "ndcg10", "floor", "by", "fanout")
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
    if any(before[k] != state[k] for k in ("rate", "spike", "peak", "fanout", "paused", "pattern")):
        probes.clear()
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
    for target in (calibrate, plan_loop, probe_loop, sample_loop):
        threading.Thread(target=target, daemon=True, name=target.__name__).start()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8767")), log_level="warning")
