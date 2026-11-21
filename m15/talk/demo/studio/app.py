"""Workload studio: live NQ traffic on one Qdrant Edge index, with Stella, Nano and Zero as query encoders.

All three encoders search the same Stella index (2.68M passages). Traffic is replayed from judged NQ questions,
so every answer is scored. Auto routing plans, four times a second, which share of traffic each encoder serves
and how wide they search: the best average quality at or above the floor that fits the machine's encode and
search capacity, with headroom tuned from measured p95 against the latency target (see allocator.py). Every query
is served. Every control applies immediately.

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
from talk_throughput import make_encoder  # noqa: E402
from workload import ENCODERS, Workload  # noqa: E402

DATASET = os.environ.get("DATASET", "nq")
WORKERS = int(os.environ.get("WORKERS", "14"))

# ---- data, index, encoders --------------------------------------------------------------------------
data = load_public(DATASET, with_corpus=False)
DOC_IDS = json.loads((REPO / "work/m15/vecs/nq_stella/doc_ids.json").read_text())
search = EdgeSearch(REPO / "work/m15/edge" / DATASET)
encoders = {name: make_encoder(name) for name in ENCODERS}

# nDCG@10 and search time of every (encoder, ef, oversampling) on all judged NQ queries, from the offline sweep
# (receipt below). Search settings are limited to this grid, so every one has a measured quality.
SWEEP = {(e, s["ef"], s["oversampling"]): s for e in ENCODERS for s in
         json.loads((REPO / f"results/m15_talk_edge_{DATASET}.json").read_text())["results"][e]["settings"]}
EFS, OVERSAMPLING = sorted({k[1] for k in SWEEP}), sorted({k[2] for k in SWEEP})
PRESETS = [(128, 4.0), (256, 4.0), (512, 4.0), (1024, 8.0)]       # what Auto chooses from, narrow to wide

state = {"rate": 40.0, "pattern": "steady", "peak": 15.0, "spike": False, "fanout": 1, "paused": False,
         "route": "auto", "split": {"stella": 34.0, "nano": 33.0, "zero": 33.0},
         "search": {name: [1024, 8.0] for name in ENCODERS},
         "floor": 0.50, "target_ms": 150.0}

work = Workload(search, encoders, data["q_texts"], data["q_ids"], data["qrels"], DOC_IDS, state, WORKERS)


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
    work.start()
    calib["ready"], calib["step"] = True, "ready"
    print("studio ready:", {k: round(v) for k, v in calib["encode_per_s"].items()}, flush=True)


# ---- live costs ---------------------------------------------------------------------------------------
probes = collections.deque(maxlen=5000)                  # (t, encoder, encode_ms) timed under the live load


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


def encode_tail_ms(name, recent, now):
    """Tail of this encoder's encode time under the current load, from served searches and 10 s of probes (its time
    under load varies too much to judge from a few). A traffic change clears the probes, since they were timed under
    the old load, and served searches from before the change are ignored. An agent task waits for the slowest of its searches, so the percentile rises with fan-out:
    P(max of k <= x) = F(x)^k >= 0.95 needs F(x) >= 0.95^(1/k)."""
    xs = [d[5] for d in recent if d[2] == name and d[0] >= router["workload_t"]]
    xs += [p[2] for p in Workload.recent(probes, 10.0, now) if p[1] == name]
    q = 100 * 0.95 ** (1 / max(1, state["fanout"]))
    return float(np.percentile(xs, q)) if len(xs) >= 5 else calib["encode_ms"].get(name, 0.0)


def shard_slowdown(recent):
    """How much slower the shard searches right now than in the offline sweep (contention, cache), from recent
    searches against the sweep times of the same settings. One factor for every setting keeps the settings
    comparable: the one in use is measured live, the others are not."""
    if len(recent) < 50:
        return 1.0
    return float(np.mean([d[6] for d in recent]) / np.mean([SWEEP[(d[2], *d[7])]["search_ms_p50"] for d in recent]))


def cost(name, ef, oversampling, slowdown):
    """Quality from the sweep; shard time from the sweep scaled by the live slowdown."""
    s = SWEEP[(name, ef, oversampling)]
    return allocator.EncoderCost(name, s["ndcg10"], calib["encode_per_s"].get(name, 1.0), s["search_ms_p50"] * slowdown)


def slow_encoders(enc_ms, target, now):
    """Encoders whose own time does not fit the latency target. Once marked slow, an encoder must stay under the
    target for 10 s before it returns: its time under load varies, and one lucky reading should not readmit it.
    A traffic change clears the marks (see /set)."""
    for name in ENCODERS:
        if enc_ms[name] > target:
            router["slow"][name] = None               # slow now; restart the clock
        elif name in router["slow"]:
            since = router["slow"][name]
            if since is None:
                router["slow"][name] = now
            elif now - since >= 10:
                del router["slow"][name]
    return frozenset(router["slow"])


# ---- routing: plan four times a second ---------------------------------------------------------------
router = {"plan": None, "headroom": 0.8, "adjusted": 0.0, "costs": {}, "over_capacity_t": 0.0, "draining": False,
          "slow": {},                               # encoder -> when it last came under target (None: still over)
          "workload_t": 0.0}                        # when the traffic controls last changed


def backlog(now):
    """Searches waiting: in the queue, plus those the replay holds back while the queue is full."""
    return len(work.queue) + work.behind_s * work.offered_rate(now)


def plan_loop():
    while True:
        time.sleep(0.25)
        if not calib["ready"]:
            continue
        now = time.time()
        recent = work.recent(work.done, 3.0, now)
        enc_ms, slowdown = {n: encode_tail_ms(n, recent, now) for n in ENCODERS}, shard_slowdown(recent)
        router["costs"] = {n: cost(n, *state["search"][n], slowdown) for n in ENCODERS}
        rate = work.offered_rate(now)
        if state["route"] == "split":
            work.shares, router["plan"] = dict(state["split"]), None
            continue
        if rate <= 0:                             # paused: keep the last plan while the queue drains
            continue
        target = state["target_ms"]
        # Headroom from measured p95 (per agent task when searches come in groups), once per fresh second of
        # completions: re-reading overlapping windows would punish the same slow requests several times. Only
        # requests that arrived after the last overload or backlog count.
        if now - router["adjusted"] >= 1.0:
            router["adjusted"] = now
            lat = [r[1] for r in work.recent(work.tasks if state["fanout"] > 1 else work.done, 1.0, now)
                   if r[0] - r[1] / 1000 > router["over_capacity_t"]]
            if len(lat) >= 10:
                p95 = float(np.percentile(lat, 95))
                if p95 > target:                  # over target: back off, harder when far over
                    router["headroom"] = max(0.4, router["headroom"] * (0.85 if p95 > 1.3 * target else 0.95))
                elif p95 < 0.7 * target:          # comfortably under: recover in a few seconds
                    router["headroom"] = min(0.85, router["headroom"] + 0.02)
        # Plan for arrivals plus the backlog cleared within 2 s.
        presets = {pr: [cost(n, *pr, slowdown) for n in ENCODERS] for pr in PRESETS}
        args = dict(rate=rate + backlog(now) / 2, floor=state["floor"], headroom=router["headroom"],
                    slow=slow_encoders(enc_ms, target, now))
        p, prev = allocator.plan(presets=presets, **args), router["plan"]
        if prev and p.preset != prev.preset:
            # Narrow at once when the current width stops fitting; widen only with room to spare (20% more traffic),
            # so the search width does not flap at the edge of capacity.
            stay = allocator.plan(presets={prev.preset: presets[prev.preset]}, **args)
            wider = presets[p.preset][0].search_ms > presets[prev.preset][0].search_ms
            spare = allocator.plan(presets={p.preset: presets[p.preset]}, **{**args, "rate": 1.2 * args["rate"]})
            if stay.status == p.status == "ok" and (p.quality - stay.quality < 0.005 or (wider and spare.status != "ok")):
                p = stay
        if router["plan"] is None or p.preset != router["plan"].preset:
            EVENTS.append((now, f"auto ef {p.preset[0]} ×{p.preset[1]:g}"))
        router["plan"] = p
        # After an overload, the backlog drains; until it is short, its waits say nothing about this plan.
        router["draining"] = p.status == "over capacity at this floor" or (
            router["draining"] and backlog(now) > rate * 2 * target / 1000)
        if router["draining"]:
            router["over_capacity_t"] = now
        state["search"] = {n: list(p.preset) for n in ENCODERS}
        work.shares = dict(p.shares)


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
    queue = list(work.queue)[:1]
    oldest = max(work.behind_s, time.perf_counter() - queue[0][1] if queue else 0.0) * 1000
    p = router["plan"]
    by = {n: sum(1 for d in last1 if d[2] == n) for n in ENCODERS}
    return {
        "t": now, "ready": calib["ready"], "step": calib["step"], **{k: state[k] for k in state},
        "offered": work.offered_rate(now), "arrived_per_s": len(work.recent(work.arrived, 1.0, now, key=lambda t: t)),
        "served_per_s": len(last1), "waiting": round(backlog(now)), "oldest_ms": oldest,
        "p95_ms": float(np.percentile(lat, 95)) if lat else 0.0,
        "task_p95_ms": float(np.percentile(tasks, 95)) if tasks else 0.0,
        "ndcg10": float(np.mean(quality)) if quality else 0.0, "n_quality": len(quality),
        "cores": round(cpu["cores"], 1), "cores_total": os.cpu_count(), "by": by,
        "actual_share": {n: (by[n] / len(last1)) if last1 else 0.0 for n in ENCODERS},
        "plan": None if p is None else {"preset": p.preset, "shares": p.shares, "quality": p.quality,
                                        "encode_util": p.encode_util, "shard_util": p.shard_util, "status": p.status},
        "headroom": router["headroom"], "n_docs": len(DOC_IDS), "efs": EFS, "oversampling": OVERSAMPLING,
        "slow": sorted(router["slow"]),
        "costs": {n: {"quality": c.quality, "search_ms": c.search_ms, "encode_per_s": c.encode_per_s}
                  for n, c in router["costs"].items()},
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


def _nearest(values, x):
    return min(values, key=lambda v: abs(np.log(v / x)))


@app.post("/set")
def update(body: dict):
    """Partial update of the controls; applies to the next request. Traffic never stops unless paused."""
    before = json.loads(json.dumps(state))
    for key in ("rate", "peak", "floor", "target_ms"):
        if key in body:
            state[key] = float(body[key])
    state["rate"] = min(max(state["rate"], 1.0), 10000.0)
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
    if state["route"] == "split":            # Auto owns the search settings; in split they are the user's
        for name, (ef, oversampling) in (body.get("search") or {}).items():
            if name in ENCODERS:
                state["search"][name] = [_nearest(EFS, float(ef)), _nearest(OVERSAMPLING, float(oversampling))]
    if any(before[k] != state[k] for k in ("rate", "spike", "peak", "fanout", "paused", "pattern")):
        probes.clear()                        # timings, headroom and slow marks belong to the old workload
        router["headroom"], router["slow"], router["workload_t"] = 0.8, {}, time.time()
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
