"""M10 step 8 generation: the registered diversity pilot, then the build for the approved forms.

Three subcommands, one process each:

  * `health`   -- re-runs step 0a's registered health assertion against the live server
                  (8 seeds x 8 forms, n=5, 64 in flight, 3 runs; gate >= 90% contract and
                  >= 700 aggregate output tok/s), and prints the two numbers.
  * `pilot`    -- 2,000 queries per form under the REAL contract from real, non-held-out build
                  seeds; rubric word-range filter (T2-10) and corpus-wide exact dedup; then the
                  amended A8 gate-1 near-duplicate rate at n = 200/500/1,000/2,000 plus opener
                  concentration. Writes `results/m10_diversity_pilot.json`.
  * `build`    -- the generation run itself, one form at a time, resumable BY SEED.

Nothing registered moves here. Quota 143,000 per generated form, 5 queries per seed, the 1.35x
seed margin and the 500-document FORMS-12 hold-out are `corpus10`'s and `build_form`'s registered
values, read from there rather than restated. The rubric word-range filter is T2-10.

**The protected-index and six's-documents screens are NOT applied here.** §Data executes them on
the build manifest at step 8 (`corpus10.screen_form`, which takes them as injected callables), and
the A8 gates run "on the immutable manifest before any arm". The driver therefore emits every
generated row plus the seed id needed to run them, and the manifest records that they are owed.
The two screens it DOES apply are the ones that are local to a row: the frozen rubric's word range
and the own-seed-passage word-5-gram copy check.
"""
import hashlib, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m10src"))
OUT = REPO / "work" / "m10gen"
RESULTS = REPO / "results"

import protected10                    # BEFORE anything imports m9base (CODEMAP pitfall 14)

import forms, gen, qfilter, corpus10
import seeds as seedmod

BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8001/v1")
QUOTA = 143_000                       # registered per-form quota for a generated form
N_PER_SEED = 5                        # registered
MARGIN = 1.35                         # corpus10.build_form's registered seed over-draw
HOLDOUT = corpus10.HOLDOUT_PER_FORM   # 500, FORMS-12
PILOT_N = 2_000
PILOT_SEED = 777                      # the pilot's own uniform draw from the build pool
WORKERS = 64                          # == the server's --max-num-seqs
CHUNK_SEEDS = 1_000                   # resume granularity

APPROVED = ["howto", "argument", "conversational", "finance", "comparison", "yesno"]
WIKIBODY_FORMS = ("health", "finance")
HOTPOT_FORMS = ["howto", "argument", "comparison", "yesno", "conversational"]
HOTPOT_PER_FORM = 39_200              # scripts/m10_seed_draw.py
SEEDS_DONE = OUT / "build_seeds.done"


# ---- identity ---------------------------------------------------------------------------------

def prompt_text(form):
    return forms.SYSTEM + "\x00" + forms.FORMS[form]


def prompt_sha256(form):
    return hashlib.sha256(prompt_text(form).encode()).hexdigest()


def prompt_hash8(form):
    """The registered approval hash (`m10src/smoke.prompt_hash`), so a row links to its approval."""
    return hashlib.blake2b(prompt_text(form).encode(), digest_size=8).hexdigest()


def doc_of(passage_id):
    """The SOURCE DOCUMENT of a seed passage. `wikipedia-body:8846237#28` -> `...:8846237`."""
    return passage_id.split("#", 1)[0]


def gate_ids():
    """Every seed passage a judge has seen -- never in a build corpus (T2-7 (8))."""
    ids = set()
    for name in ("gate/key.json", "gate/key-r1.json"):
        p = OUT / name
        if not p.exists():
            continue
        for form, rows in json.loads(p.read_text()).items():
            for r in rows.values():
                ids.add(r["passage_id"])
    return ids


# ---- seeds ------------------------------------------------------------------------------------

_WIKI, _HOTPOT = {}, {}


def seed_rows(form):
    """-> [(passage_id, passage_text)] for the form's FULL resolved seed store."""
    if form in WIKIBODY_FORMS:
        if not _WIKI:
            _WIKI.update(json.loads((OUT / "wikibody_gate_pool.json").read_text()))
        return [tuple(x) for x in _WIKI[form]]
    if not _HOTPOT:
        kept, _meta = seedmod.cached(HOTPOT_FORMS, per_form=HOTPOT_PER_FORM, pool_size=6_000_000,
                                     seed=0, store="hotpotqa-corpus", min_score=4, verbose=False)
        _HOTPOT.update(kept)           # a ~200 MB cache file; load it once per process
    return _HOTPOT[form]


def partition(form):
    rows = seed_rows(form)
    build, held, prep = corpus10.partition_seeds(rows, gate_ids=gate_ids(),
                                                 holdout_n=HOLDOUT, seed=0)
    return build, held, prep


# ---- the health assertion (step 0a) -------------------------------------------------------------

def run_health(runs=3):
    ids = gen.health(base=BASE)
    # the EXACT 40-seed smoke draw step 0a used, read from its cache rather than re-drawn: a
    # re-draw under the current screen version would be a different population and a 5.2M-doc
    # store load, and the assertion is about the SERVER, not the seeds.
    blob = json.loads((OUT / "seeds" / "seeds-be3fb88f10f18d8c.json").read_text())
    kept = {f: [tuple(x) for x in v] for f, v in blob["seeds"].items()}
    shape = [(f, kept[f][:8]) for f in ("howto", "argument", "finance", "comparison", "yesno",
                                        "conversational", "health")]
    shape.append(("factoid", kept["health"][:8]))          # 7 generated + factoid, as in step 0a
    out = {"when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "base": BASE, "served": ids,
           "model": gen.MODEL, "revision": gen.REVISION, "prompts": 64,
           "shape": "8 seeds x 8 forms (7 generated + factoid), n=5, all in flight", "runs": []}
    for r in range(runs):
        t0 = time.time()
        from concurrent.futures import ThreadPoolExecutor
        jobs = [(f, s) for f, ss in shape for s in ss]
        with ThreadPoolExecutor(64) as ex:
            rows = list(ex.map(lambda j: gen._one(j[0], j[1][1], j[1][0], 5, BASE, 600), jobs))
        dt = time.time() - t0
        ok = sum(1 for x in rows if x[1])
        tok = sum(x[3] for x in rows)
        out["runs"].append({"run": r, "wall_s": round(dt, 1), "contract_ok": ok,
                            "contract_rate": round(ok / len(rows), 4), "out_tok": tok,
                            "aggregate_out_tok_per_s": round(tok / dt, 1),
                            "finish_reasons": {k: sum(1 for x in rows if x[5] == k)
                                               for k in {x[5] for x in rows} if k}})
        print(json.dumps(out["runs"][-1]), flush=True)
    out["contract_rate"] = min(r["contract_rate"] for r in out["runs"])
    out["aggregate_out_tok_per_s"] = min(r["aggregate_out_tok_per_s"] for r in out["runs"])
    out["PASS_contract"] = out["contract_rate"] >= 0.90
    out["PASS_throughput"] = out["aggregate_out_tok_per_s"] >= 700
    out["_gate"] = ">= 90% contract and >= 700 aggregate output tok/s (results/m10_gen_health_box.json)"
    (RESULTS / "m10_gen_health_box_rerun.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1), flush=True)
    return out


# ---- the diversity pilot ------------------------------------------------------------------------

def a8_curve(texts, ns=(200, 500, 1000, 2000)):
    """The AMENDED (W10) A8 gate-1 rate at each prefix. The rate is monotone in n, so the CURVE
    is the quantity and every value is a floor for the build's 143,000."""
    out = {}
    for n in ns:
        if n > len(texts):
            continue
        _reps, _ded, rep = corpus10.near_dup_gate(texts[:n])
        out[str(n)] = {"near_dup_rate": rep["near_dup_rate"],
                       "near_dup_rate_raw": rep["near_dup_rate_raw"],
                       "near_duplicates": rep["near_duplicates"],
                       "post_exact_dedup": rep["post_exact_dedup"],
                       "caught_only_by_short_rule": rep["caught_only_by_short_rule"]}
    return out


def pilot_form(form, n_target=PILOT_N, seen=None):
    build, held, prep = partition(form)
    need = int(1.25 * n_target / N_PER_SEED)            # headroom for the range filter and dedup
    use = corpus10.draw_seeds(build, need, seed=PILOT_SEED)
    seen = set() if seen is None else seen
    t0 = time.time()
    g = gen.generate(form, use, n=N_PER_SEED, base=BASE, workers=WORKERS, label=form, seen=seen)
    kept, qrep = qfilter.filter_queries(g["queries"])
    seed_text = {p: t for p, t in use}
    n_copied = sum(1 for r in kept if corpus10.copied_span(r["query"], seed_text[r["seed_id"]]))
    texts = [r["query"] for r in kept][:n_target]
    curve = a8_curve(texts)
    top_share, distinct2 = corpus10.opener_concentration(texts)
    div = corpus10.diversity_report(texts)
    rate2000 = curve.get(str(min(n_target, len(texts))), list(curve.values())[-1])["near_dup_rate_raw"]
    r500 = curve.get("500", {}).get("near_dup_rate_raw")
    rising = (rate2000 - r500) if r500 is not None else None
    rep = {"form": form, "seeds_in_store": prep["n_input"], "gate_seeds_excluded": prep["gate_seeds_excluded"],
           "forms12_holdout": prep["forms12_holdout"], "build_seeds": prep["build_seeds"],
           "pilot_seeds": len(use), "pilot_draw_seed": PILOT_SEED,
           "prompt_sha256": prompt_sha256(form), "prompt_hash8": prompt_hash8(form),
           "generation": {k: v for k, v in g.items() if k != "queries"},
           "rubric_range_filter": qrep.get(form), "n_after_filter": len(kept),
           "own_seed_copied_span_hits": n_copied, "n_measured": len(texts),
           "a8_amended_curve": curve,
           "a8_rate_at_n": rate2000, "rise_500_to_2000": rising,
           "top10_opening_4gram_share": round(top_share, 4), "distinct_2": round(distinct2, 4),
           "unregistered_frac_diagnostics": div["curve"],
           "seconds": round(time.time() - t0, 1)}
    rep["above_25pct_cut"] = bool(rate2000 > corpus10.A8_MAX_NEAR_DUP_RATE)
    rep["rising_steeply"] = bool(rising is not None and rising > 0.03)
    rep["start_generation"] = not (rep["above_25pct_cut"] or rep["rising_steeply"])
    print(f"  {form:15s} A8@{len(texts)} {rate2000:.4f}  rise(500->) "
          f"{'n/a' if rising is None else f'{rising:+.4f}'}  top10-opener {top_share:.3f}  "
          f"start={rep['start_generation']}", flush=True)
    return rep, texts


def run_pilot(which):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "m10_diversity_pilot.json"
    blob = json.loads(path.read_text()) if path.exists() else {}
    blob.setdefault("_what", "the registered diversity pilot: 2,000 queries per form under the "
                             "real generation contract, from real non-held-out build seeds")
    blob.setdefault("_gate", "A8 gate 1 as amended by W10 (4-grams >= 50% below 39 words, the "
                             "registered 8-gram 16/32 rule at or above it); action above 0.25 is "
                             "cut-to-representatives, below 50,000 retained the form is dropped")
    blob.setdefault("_monotone", "the rate is non-decreasing in n, so every value is a FLOOR for "
                                 "the build's 143,000")
    blob.setdefault("model", gen.MODEL)
    blob.setdefault("revision", gen.REVISION)
    blob.setdefault("forms", {})
    for f in which:
        rep, _ = pilot_form(f)
        blob["forms"][f] = rep
        path.write_text(json.dumps(blob, indent=1))
    return blob


# ---- the build ----------------------------------------------------------------------------------

def build_paths(form):
    return (OUT / f"generated_{form}.jsonl", OUT / f"generated_{form}.seeds",
            OUT / f"manifest_{form}.json")


def run_build(form, quota=QUOTA):
    jl, sd, mf = build_paths(form)
    build, held, prep = partition(form)
    need = int(MARGIN * quota / N_PER_SEED)
    use = corpus10.draw_seeds(build, need, seed=0)
    if len(use) < need:
        print(f"  {form}: {len(use):,} build seeds for a {need:,} target -- SHORT", flush=True)
    done, seen, n_rows = set(), set(), 0
    if sd.exists():
        done = {l.strip() for l in sd.read_text().splitlines() if l.strip()}
    if jl.exists():
        with jl.open() as fh:
            for line in fh:
                r = json.loads(line)
                seen.add(" ".join(r["text"].split()).lower())
                n_rows += 1
    todo = [s for s in use if s[0] not in done]
    print(f"[{form}] {len(use):,} seeds drawn, {len(done):,} already done, {len(todo):,} to go, "
          f"{n_rows:,} rows on disk", flush=True)
    t0, tok = time.time(), 0
    chunks = [todo[i:i + CHUNK_SEEDS] for i in range(0, len(todo), CHUNK_SEEDS)]
    screens = {"out_of_rubric_range": 0, "copied_span": 0, "kept": 0, "n_input": 0}
    genstats = {"contract_ok": 0, "reached": 0, "n_seeds": 0, "retried": 0, "truncated": 0,
                "transport_failures": 0, "exact_dupes": 0}
    psha = prompt_sha256(form)
    for ci, chunk in enumerate(chunks):
        g = gen.generate(form, chunk, n=N_PER_SEED, base=BASE, workers=WORKERS, seen=seen)
        for k in genstats:
            genstats[k] += g.get(k, 0)
        tok += g["out_tok"]
        seed_text = {p: t for p, t in chunk}
        rows = []
        screens["n_input"] += len(g["queries"])
        for r in g["queries"]:
            q = r["query"]
            if not qfilter.in_range(form, q):
                screens["out_of_rubric_range"] += 1
                continue
            if corpus10.copied_span(q, seed_text[r["seed_id"]]):
                screens["copied_span"] += 1
                continue
            rows.append({"text": q, "form": form, "seed_id": r["seed_id"],
                         "doc": doc_of(r["seed_id"]), "prompt_sha256": psha})
        screens["kept"] += len(rows)
        with jl.open("a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        with sd.open("a") as fh:
            for p, _t in chunk:
                fh.write(p + "\n")
        n_rows += len(rows)
        el = time.time() - t0
        rate = (ci + 1) * len(chunk) / max(el, 1e-9)
        eta = (len(todo) - (ci + 1) * CHUNK_SEEDS) / max(rate, 1e-9)
        print(f"[{form}] chunk {ci + 1}/{len(chunks)}  rows {n_rows:,}  contract "
              f"{g['contract_rate']:.3f}  {g['out_tok_per_s']:.0f} out tok/s  "
              f"{rate:.1f} seeds/s  ETA {eta / 3600:.2f} h", flush=True)
        mf.write_text(json.dumps(manifest(form, prep, held, use, n_rows, screens, genstats,
                                          tok, time.time() - t0, complete=False), indent=1))
    mf.write_text(json.dumps(manifest(form, prep, held, use, n_rows, screens, genstats, tok,
                                      time.time() - t0, complete=True), indent=1))
    print(f"[{form}] COMPLETE {n_rows:,} rows in {(time.time() - t0) / 3600:.2f} h", flush=True)


def manifest(form, prep, held, use, n_rows, screens, genstats, tok, secs, complete):
    return {
        "form": form, "complete": complete, "quota": QUOTA, "n_per_seed": N_PER_SEED,
        "seed_margin": MARGIN, "seeds_drawn": len(use), "rows_written": n_rows,
        "generator": {"model": gen.MODEL, "revision": gen.REVISION,
                      "server": BASE, "vllm": "0.28.0",
                      "temperature": gen.TEMPERATURE, "top_p": gen.TOP_P,
                      "max_tokens_per_query": (gen.TOK_PER_QUERY_LONG if form in gen.LONG_FORMS
                                               else gen.TOK_PER_QUERY),
                      "max_tokens_per_request": gen.max_tokens_for(form, N_PER_SEED),
                      "seed_rule": "blake2b-64(seed_passage_id); retry seed adds b'#retry'",
                      "thinking": False, "retry": "one retry on a contract failure, then the seed is dropped"},
        "prompt_sha256": prompt_sha256(form), "prompt_hash8_registered": prompt_hash8(form),
        "seed_store": ("wikipedia-body (T2-5): wikimedia/wikipedia 20231101.en @ "
                       "b04c8d1ceb2f5cd4588862100d08de323dccfbaa, lead excluded, cap 3/article, "
                       "ROUTE min_score>=4" if form in WIKIBODY_FORMS else
                       "hotpotqa-corpus via m10src/seeds.draw, ROUTE (T2-3) min_score>=4, "
                       "40-220 words, pool 6,000,000, seed 0"),
        "screen_version": seedmod.SCREEN_VERSION,
        "seeds": prep, "forms12_holdout_seed_ids": sorted(p for p, _t in held),
        "forms12_holdout_docs": sorted({doc_of(p) for p, _t in held}),
        "screens_applied": screens,
        "screens_deferred": ("protected index and the six's documents -- run on the immutable "
                             "manifest at step 8 (corpus10.screen_form); A8 gate 1 and gate 2 "
                             "likewise run on the manifest before any arm"),
        "generation": genstats, "out_tokens": tok, "seconds": round(secs, 1),
        "out_tok_per_s": round(tok / max(secs, 1e-9), 1),
    }


def wait_for_seeds(timeout_s=7200):
    t0 = time.time()
    while not SEEDS_DONE.exists():
        if time.time() - t0 > timeout_s:
            raise SystemExit("build_seeds.done never appeared -- scripts/m10_seed_draw.py failed")
        print(f"[wait] seed draw still running ({(time.time() - t0) / 60:.1f} min)", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == "health":
        run_health()
    elif cmd == "pilot":
        run_pilot(args or APPROVED)
    elif cmd == "build":
        for f in args:
            if f in HOTPOT_FORMS:
                wait_for_seeds()
            run_build(f)
    elif cmd == "pilot_then_build":
        # one detached driver: pilot a form, and generate it only if the pilot clears the gate.
        blob_path = RESULTS / "m10_diversity_pilot.json"
        for f in args:
            if f in HOTPOT_FORMS:
                wait_for_seeds()
            blob = json.loads(blob_path.read_text()) if blob_path.exists() else {"forms": {}}
            rep = blob.get("forms", {}).get(f)
            if rep is None:
                run_pilot([f])
                rep = json.loads(blob_path.read_text())["forms"][f]
            if not rep["start_generation"]:
                print(f"[{f}] PILOT FLAG -- A8 {rep['a8_rate_at_n']:.4f}, rise "
                      f"{rep['rise_500_to_2000']}. NOT STARTED; reported to the lead.", flush=True)
                continue
            run_build(f)
    else:
        raise SystemExit(f"unknown command {cmd!r}")
