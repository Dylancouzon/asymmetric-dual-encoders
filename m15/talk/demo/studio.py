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
import functools     # noqa: E402
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
HEADROOM = 0.6                    # auto plans to 60% of the latency target (tails run above the mean)
SLOW_CAP = 4                      # max concurrent requests on a setting that costs >20% of the latency plan
DECAY_S = 5.0                     # an unused setting's cost estimate halves its distance to quiet time every 5 s
ORDER = ("stella", "nano", "zero")
TRIAL_LOG = REPO / "work/m15/logs/studio_trials.jsonl"

data = load_public(DATASET, with_corpus=False)
QUERIES, QIDS, QRELS = data["q_texts"], data["q_ids"], data["qrels"]
DOC_IDS = json.loads((REPO / "work/m15/vecs/nq_stella/doc_ids.json").read_text())
shard = E.EdgeShard.load(str(REPO / "work/m15/edge" / DATASET))
ENC = {k: make_encoder(k) for k in ORDER}

# Defaults: each encoder's selected NQ setting (fastest within 1% of its own exact nDCG@10; Zero: widest tested).
state = {"rate": 40.0, "pattern": "steady", "peak": 15.0, "spike": False, "route": "auto",
         "split": {"stella": 1.0, "nano": 0.0, "zero": 0.0},
         "search": {"stella": [256, 4.0], "nano": [512, 4.0], "zero": [1024, 8.0]},
         "budget_ms": 150.0, "workers": WORKERS, "fanout": 1, "ndcg_floor": 0.50}
# Measured NQ nDCG@10 for every (encoder, ef, oversampling) in the sweep (exact-match validated runs).
_SW = json.loads((REPO / f"results/m15_talk_edge_{DATASET}.json").read_text())["results"]
SWEEP = {(e, x["ef"], x["oversampling"]): x["ndcg10"] for e in ORDER for x in _SW[e]["settings"]}


@functools.lru_cache(maxsize=None)
def quality(e, ef, os_):
    """Sweep nDCG@10 for this setting; nearest swept ef and oversampling when off the grid."""
    efs = sorted({k[1] for k in SWEEP if k[0] == e}); oss = sorted({k[2] for k in SWEEP if k[0] == e})
    ef = min(efs, key=lambda v: abs(math.log(v) - math.log(max(ef, 1)))); os_ = min(oss, key=lambda v: abs(v - os_))
    return SWEEP[(e, ef, os_)]
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


# Quiet-time cost of a setting = this machine's single-thread encode time (measured at startup) plus the
# median search time measured for that exact (encoder, ef, oversampling) in the NQ sweep.
ENC_MS = {}
for e in ORDER:
    for q in QUERIES[:5]:
        ENC[e].encode([q])
    ts = []
    for q in QUERIES[5:35]:
        t0 = time.perf_counter(); ENC[e].encode([q]); ts.append((time.perf_counter() - t0) * 1000)
    ENC_MS[e] = float(np.median(ts))
SEARCH_MS = {(e, x["ef"], x["oversampling"]): x["search_ms_p50"] for e in ORDER for x in _SW[e]["settings"]}


@functools.lru_cache(maxsize=None)
def search_ms(e, ef, os_):
    efs = sorted({k[1] for k in SEARCH_MS if k[0] == e}); oss = sorted({k[2] for k in SEARCH_MS if k[0] == e})
    return SEARCH_MS[(e, min(efs, key=lambda v: abs(math.log(v) - math.log(max(ef, 1)))), min(oss, key=lambda v: abs(v - os_)))]


# Throughput a setting can sustain here: encodes run about PAR at a time (measured: Nano ~1,400/s at 1.95 ms,
# Stella ~80/s at 28 ms with 14 threads), and searches on one Edge shard run one at a time.
PAR = 2.8


def capacity(e, ef, os_):
    return min(PAR * 1000 / max(ENC_MS[e], 1e-3), 1000 / max(search_ms(e, ef, os_), 1e-3))


def base_cost(e, ef, os_):
    efs = sorted({k[1] for k in SEARCH_MS if k[0] == e}); oss = sorted({k[2] for k in SEARCH_MS if k[0] == e})
    efn = min(efs, key=lambda v: abs(math.log(v) - math.log(max(ef, 1)))); osn = min(oss, key=lambda v: abs(v - os_))
    return ENC_MS[e] + SEARCH_MS[(e, efn, osn)]


BASE = {r: base_cost(e, ef, os_) for r, e, ef, os_ in ladder()}
service, last_used = dict(BASE), {r: 0.0 for r in BASE}
inflight = collections.Counter()              # picks not yet completed, per setting
warmed = [0]                                  # worker threads that finished warming up

queue = collections.deque()
cv = threading.Condition()
done = collections.deque(maxlen=400_000)      # (t, latency, enc, ndcg, rung, queue_ms, enc_ms, search_ms, trial)
arrivals = collections.deque(maxlen=400_000)  # [t_wall, ok, trial]
dropped = collections.deque(maxlen=400_000)
trial = {"id": 0, "start": 0.0, "end": 0.0, "pinned": True}
groups = {}                                   # agent task id -> [searches left, max latency so far]
tasks = collections.deque(maxlen=200_000)     # (t_done, task latency ms) for fan-out tasks
_gid = [0]
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
    interactive_qos()
    k, nxt = 0, time.perf_counter()
    while True:
        now, wall = time.perf_counter(), time.time()
        warming = trial["id"] and not trial["pinned"] and wall < trial["start"]
        if reseed[0] and not warming:         # replay starts: drop warm-up leftovers, restart trace and pattern
            with cv:
                for item in queue:
                    item[2][1] = False
                queue.clear()
                groups.clear()
            k, nxt, reseed[0], epoch[0] = 0, now, False, wall
        rate = min(0.3 * current_rate(wall), 300.0) if warming else current_rate(wall)
        f = max(1, int(state["fanout"]))
        with cv:
            while nxt <= now:
                # One arrival = one agent task issuing f searches together (f = 1: ordinary traffic).
                tid = trial["id"] if trial["start"] <= wall < trial["end"] else 0
                gap = float(GAPS[k % len(GAPS)])
                full = len(queue) + f > MAX_QUEUE
                gid = None
                if f > 1 and not full:
                    _gid[0] += 1
                    gid = _gid[0]
                    groups[gid] = [f, 0.0]
                for j in range(f):
                    rec = [wall - (now - nxt), None, tid]
                    arrivals.append(rec)
                    qi = int(QIDX[(k + j) % len(QIDX)])
                    if full:
                        dropped.append(wall)
                        rec[1] = False
                    else:
                        queue.append((qi, nxt, rec, gid))
                k += f
                nxt += gap * f / rate          # rate stays in searches per second
            cv.notify_all()
        time.sleep(max(0.0, min(0.005, nxt - time.perf_counter())))


_route_rng = random.Random(1)
smooth_wait = [0.0]                           # EWMA of queue wait over recent requests, ms


def overload_pick(allowed):
    """Overloaded: the allowed setting that sustains the most traffic; within 10% of that, the best quality."""
    top = max(capacity(*r[1:]) for r in allowed)
    return max((r for r in allowed if capacity(*r[1:]) >= 0.9 * top), key=lambda r: quality(*r[1:]))


def est_cost(r, now):
    """Live cost estimate of a setting; when unused it drifts back toward its quiet-time cost."""
    idle = max(0.0, now - last_used[r] - 1.0)
    return BASE[r] + (service[r] - BASE[r]) * 0.5 ** (idle / DECAY_S)


def plan_ms():
    return HEADROOM * state["budget_ms"]


def choose(wait_ms):
    """Returns (rung, encoder, ef, oversampling) for one request."""
    if state["route"] == "auto":
        # Quality first: never use a setting below the nDCG floor. Spare latency buys quality above it;
        # when nothing fits the latency target, fall back to the allowed setting with the lowest quiet-time cost.
        now = time.time()
        rungs = sorted(ladder(), key=lambda r: -quality(*r[1:]))
        ok = [r for r in rungs if quality(*r[1:]) >= state["ndcg_floor"]] or rungs[:1]
        # Plan with the larger of this request's wait and the recent average wait, so a sustained load keeps
        # the cheaper setting instead of oscillating each time the queue drains.
        wait_ms = max(wait_ms, smooth_wait[0])
        for r, e, ef, os_ in ok:
            c = est_cost(r, now)
            if wait_ms + c <= plan_ms():
                # A setting whose last measurement was over budget is recovering: one probe at a time.
                if service[r] + wait_ms > plan_ms() and inflight[r] > 0:
                    continue
                # An expensive setting may hold at most SLOW_CAP workers, so a sudden spike cannot tie up
                # every worker in slow requests before its cost estimate catches up.
                if c > 0.2 * plan_ms() and inflight[r] >= SLOW_CAP:
                    continue
                service[r], last_used[r] = c, now     # materialize the decay so this pick starts from it
                inflight[r] += 1
                return r, e, ef, os_
        pick = overload_pick(ok)
        service[pick[0]], last_used[pick[0]] = est_cost(pick[0], now), now
        inflight[pick[0]] += 1
        return pick
    w = state["split"]
    tot = sum(w.values()) or 1.0
    x, acc = _route_rng.random() * tot, 0.0
    for e in ORDER:                            # weighted split: all encoders serve at once
        acc += w[e]
        if x < acc:
            inflight[e] += 1
            return (e, e, *state["search"][e])
    inflight["zero"] += 1
    return ("zero", "zero", *state["search"]["zero"])


def auto_table(now, sec):
    """What auto sees right now, best quality first: why each setting is used or not."""
    rungs = sorted(ladder(), key=lambda r: -quality(*r[1:]))
    floor, wait = state["ndcg_floor"], smooth_wait[0]
    counts = collections.Counter(d[4] for d in sec)
    allowed = [r for r in rungs if quality(*r[1:]) >= floor]
    fits = [r for r in allowed if wait + est_cost(r[0], now) <= plan_ms()]
    chosen = fits[0][0] if fits else (overload_pick(allowed)[0] if allowed else None)
    rows = []
    for r, e, ef, os_ in rungs:
        q, c = quality(e, ef, os_), est_cost(r, now)
        why = ("below floor" if q < floor else "chosen" if r == chosen and fits
               else "chosen: overload" if r == chosen else "fits" if (r, e, ef, os_) in fits else "too slow")
        rows.append({"rung": r, "enc": e, "ef": ef, "os": os_, "quality": q, "est_ms": wait + c,
                     "share": counts.get(r, 0) / len(sec) if sec else 0.0, "why": why})
    return rows


def interactive_qos():
    """macOS: mark this thread user-interactive so the scheduler keeps it on performance cores. After a quiet
    period, background-class threads sit on efficiency cores and take seconds to ramp when a spike arrives."""
    if sys.platform == "darwin":
        try:
            import ctypes
            ctypes.CDLL("/usr/lib/libSystem.dylib").pthread_set_qos_class_self_np(0x21, 0)   # QOS_CLASS_USER_INTERACTIVE
        except Exception as err:              # not fatal: the demo still runs, just with slower ramp-up
            print("QoS not set:", err, flush=True)


def worker(idx):
    interactive_qos()
    # Warm every encoder and search setting on this thread first: a thread's first calls into a model are slow,
    # and a spike would otherwise hit cold threads.
    for r, e, ef, os_ in ladder():
        for q in QUERIES[idx * 7: idx * 7 + 4]:
            serve(e, ef, os_, q)
    warmed[0] += 1
    while True:
        with cv:
            while not queue or idx >= state["workers"]:
                cv.wait(0.05)
            i, t_arr, rec, gid = queue.popleft()
        t0 = time.perf_counter()
        w_ms = (t0 - t_arr) * 1000
        smooth_wait[0] = 0.95 * smooth_wait[0] + 0.05 * w_ms
        r, e, ef, os_ = choose(w_ms)
        hits, enc_ms, srch_ms = serve(e, ef, os_, QUERIES[i])
        t1 = time.perf_counter()
        inflight[r] -= 1
        if r in service:
            service[r] = 0.9 * service[r] + 0.1 * (t1 - t0) * 1000
            last_used[r] = time.time()
        lat, nd = (t1 - t_arr) * 1000, ndcg10(QIDS[i], [h.id for h in hits])
        rec[1] = lat <= state["budget_ms"] and nd > 0
        done.append((time.time(), lat, e, nd, r, (t0 - t_arr) * 1000, enc_ms, srch_ms, rec[2]))
        if gid is not None:                   # an agent task finishes when its last search does
            with cv:
                g = groups.get(gid)
                if g:
                    g[0] -= 1
                    g[1] = max(g[1], lat)
                    if g[0] == 0:
                        tasks.append((time.time(), g[1]))
                        del groups[gid]


app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "studio.html")


EVENTS = collections.deque(maxlen=50)         # (t, label) for chart markers


@app.post("/set")
def set_state(body: dict):
    """Partial update: rate, pattern, peak, spike, route, split{enc: w}, search{enc: [ef, os]}, budget_ms, workers,
    fanout, ndcg_floor. Changes apply to the next request; traffic never stops."""
    before = dict(state)
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
            r = e                              # the encoder's own rung follows its new search setting
            BASE[r] = service[r] = base_cost(e, *state["search"][e])
    if "ndcg_floor" in body:                  # a floor no setting can meet is refused: keep the best attainable
        best = max(quality(*r[1:]) for r in ladder())
        state["ndcg_floor"] = max(0.0, min(best, float(body["ndcg_floor"])))
    if "budget_ms" in body:
        state["budget_ms"] = max(1.0, float(body["budget_ms"]))
    if "fanout" in body:
        state["fanout"] = max(1, min(64, int(body["fanout"])))
    if "workers" in body:
        state["workers"] = max(1, min(WORKERS, int(body["workers"])))
        with cv:
            cv.notify_all()
    # One short marker per change, most important first.
    label = []
    if state["ndcg_floor"] != before["ndcg_floor"]:
        label.append(f"floor {state['ndcg_floor']:.2f}")
    elif state["route"] != before["route"]:
        label.append(state["route"])
    elif state["rate"] != before["rate"]:
        label.append(f"{state['rate']:,.0f}/s")
    elif state["spike"] != before["spike"]:
        label.append(f"spike ×{state['peak']:g}" if state["spike"] else "spike off")
    elif state["fanout"] != before["fanout"]:
        label.append("agent tasks" if state["fanout"] > 1 else "single searches")
    elif state["budget_ms"] != before["budget_ms"]:
        label.append(f"target {state['budget_ms']:.0f} ms")
    elif state["pattern"] != before["pattern"]:
        label.append(state["pattern"])
    if label:
        EVENTS.append((time.time(), " · ".join(label)))
    return state


@app.post("/reset")
def reset():
    with cv:
        for item in queue:
            item[2][1] = False
        queue.clear()
        groups.clear()
    return {"queue": 0}


@app.post("/trial")
def start_trial():
    """Replay the same traffic for the current setup: warm up, then the fixed trace and pattern for TRIAL_S."""
    reset()
    start = time.time() + WARM_S
    trial.update(id=trial["id"] + 1, start=start, end=start + TRIAL_S, pinned=False,
                 config=json.loads(json.dumps({k: state[k] for k in ("rate", "pattern", "peak", "spike", "route", "split", "search", "budget_ms", "workers", "fanout")})))
    reseed[0] = True
    return trial


@app.post("/unpin")
def unpin():
    pins.clear()
    return pins


def finish_trial():
    tid = trial["id"]
    arr = [a for a in list(arrivals) if a[2] == tid]
    cum = [h for h in list(HIST) if trial["start"] <= h["t"] <= trial["end"] + 0.5]
    cores = (cum[-1]["cpu_used"] - cum[0]["cpu_used"]) / (cum[-1]["t"] - cum[0]["t"]) if len(cum) > 1 else cpu["cores"]
    tl = [d[1] for d in list(tasks) if trial["start"] <= d[0] <= trial["end"] + 2]
    comp = [d for d in list(done) if d[8] == tid]
    lat = [d[1] for d in comp]
    med = lambda j: float(np.median([d[j] for d in comp])) if comp else None
    per = {e: {"share": sum(1 for d in comp if d[2] == e) / len(comp) if comp else 0.0} for e in ORDER}
    row = {**trial["config"], "p95_ms": float(np.percentile(lat, 95)) if lat else None,
           "ndcg10": float(np.mean([d[3] for d in comp])) if comp else 0.0,
           "good_pct": 100.0 * sum(1 for a in arr if a[1]) / len(arr) if arr else 0.0,
           "served_per_s": len(comp) / TRIAL_S, "completed": len(comp), "offered": len(arr),
           "dropped": sum(1 for t in list(dropped) if trial["start"] <= t <= trial["end"]), "outstanding": sum(1 for a in arr if a[1] is None),
           "task_p95_ms": float(np.percentile(tl, 95)) if tl else None,
           "cores_busy": cores, "cpu_s_per_1k": cores / (len(comp) / TRIAL_S) * 1000 if comp else None, "mix": per,
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
                  "work_ms": float(np.median([d[6] + d[7] for d in le])) if le else None,
                  "quality": quality(e, *state["search"][e]),
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
            "cores_busy": round(cpu["cores"], 1), "cores_total": os.cpu_count(), "cpu_used": cpu["last"][1] if cpu["last"] else 0.0,
            "tasks_per_s": sum(1 for d in list(tasks)[-5000:] if now - d[0] <= 1.0),
            "task_p95_ms": pct([d[1] for d in list(tasks)[-5000:] if now - d[0] <= 2.0], 95),
            "cpu_s_per_1k": (cpu["cores"] / served * 1000) if served else None,
            "trial_left": max(0.0, trial["end"] - now) if not trial["pinned"] else 0.0,
            "warming": max(0.0, trial["start"] - now) if not trial["pinned"] else 0.0,
            "auto": auto_table(now, sec), "plan_ms": plan_ms(), "events": [e for e in EVENTS if now - e[0] <= 60],
            "best_quality": max(quality(*r[1:]) for r in ladder()),
            "arrivals_per_s": sum(1 for a in list(arrivals)[-20000:] if now - a[0] <= 1.0),
            "pins": pins, "seg_age": now - seg["start"], "warmed": warmed[0], "ready": ready[0], "est_ms": {r: round(v, 2) for r, v in service.items()},
            "smooth_wait_ms": round(smooth_wait[0], 1),
            "rungs": dict(collections.Counter(d[4] for d in sec)), "n_docs": len(DOC_IDS), "max_workers": WORKERS}


SEG_MIN_S, SEG_SETTLE_S = 6.0, 2.0
seg = {"sig": None, "start": time.time(), "cfg": None}


def signature():
    tot = sum(state["split"].values()) or 1.0
    cfg = {k: state[k] for k in ("rate", "pattern", "peak", "spike", "route", "search", "budget_ms", "workers", "fanout", "ndcg_floor")}
    cfg["split"] = {e: round(state["split"][e] / tot, 2) for e in ORDER}
    return json.dumps(cfg, sort_keys=True), cfg


def record_segment(now):
    """Each setup that ran long enough becomes a 'recent setups' row: its steady state after settling."""
    t0 = seg["start"] + SEG_SETTLE_S
    comp = [d for d in list(done)[-200000:] if t0 <= d[0] <= now]
    if not comp:
        return
    hs = [h for h in list(HIST) if t0 <= h["t"] <= now]
    cores = (hs[-1]["cpu_used"] - hs[0]["cpu_used"]) / (hs[-1]["t"] - hs[0]["t"]) if len(hs) > 1 and hs[-1]["t"] > hs[0]["t"] else cpu["cores"]
    arr = [a for a in list(arrivals)[-400000:] if t0 <= a[0] <= now - state["budget_ms"] / 1000]
    tl = [d[1] for d in list(tasks)[-50000:] if t0 <= d[0] <= now]
    lat, span = [d[1] for d in comp], max(1e-6, now - t0)
    row = {**seg["cfg"], "auto_recorded": True, "seconds": round(now - seg["start"], 1),
           "p95_ms": float(np.percentile(lat, 95)), "task_p95_ms": float(np.percentile(tl, 95)) if tl else None,
           "ndcg10": float(np.mean([d[3] for d in comp])), "served_per_s": len(comp) / span,
           "arrival_rate": float(np.mean([h["arrival_rate"] for h in hs])) if hs else state["rate"],
           "completed": len(comp), "offered": len(arr) or len(comp),
           "dropped": sum(1 for t in list(dropped)[-200000:] if t0 <= t <= now),
           "cores_busy": cores, "cpu_s_per_1k": cores / (len(comp) / span) * 1000,
           "mix": {e: sum(1 for d in comp if d[2] == e) / len(comp) for e in ORDER}}
    pins.append(row)
    with open(TRIAL_LOG, "a") as f:
        f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}) + "\n")


def sampler():
    while True:
        now = time.time()
        sig, cfg = signature()
        if sig != seg["sig"]:
            if seg["sig"] and now - seg["start"] >= SEG_MIN_S and trial["pinned"]:
                record_segment(now)
            seg.update(sig=sig, start=now, cfg=cfg)
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
    keep = ("t", "arrival_rate", "served_per_s", "p95_ms", "budget_ms", "ndcg10", "ndcg_floor", "cores_busy", "queue", "task_p95_ms", "fanout")
    return {**s, "history": [{**{k: h[k] for k in keep}, "by": {e: h["per"][e]["served_per_s"] for e in ORDER}}
                             for h in list(HIST)[-240:]]}


ready = [False]


def burn_in():
    """Full-load warm-up before the demo: the first burst after start runs several times slower on this laptop
    (measured: Nano encode 28 ms falling to 5 ms over ~8 s; later bursts start at 4 ms). Then reset all stats."""
    while warmed[0] < WORKERS:
        time.sleep(0.2)
    defaults = json.loads(json.dumps(state))
    state.update(route="split", split={"stella": 0.0, "nano": 60.0, "zero": 40.0}, rate=900.0, spike=False, fanout=1)
    time.sleep(10)
    state.update(split={"stella": 100.0, "nano": 0.0, "zero": 0.0}, rate=60.0)
    time.sleep(4)
    state.clear(); state.update(defaults)
    reset()
    time.sleep(1.5)
    for d in (done, arrivals, dropped, tasks, HIST, EVENTS):
        d.clear()
    pins.clear()
    service.update(BASE)
    smooth_wait[0] = 0.0
    seg.update(sig=None, start=time.time())
    ready[0] = True
    print("studio ready", flush=True)


if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=burn_in, daemon=True).start()
    threading.Thread(target=generator, daemon=True).start()
    threading.Thread(target=sampler, daemon=True).start()
    for i in range(WORKERS):
        threading.Thread(target=worker, args=(i,), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8767")), log_level="warning")
