"""The M9.1 screen driver: a batch pilot, then seven sequential arms, then one decision pass.

Arms, order, dose, templates, surfaces and rules are read from `m9/registry.json`; this module
only assembles them. Every arm opens a `guard9` run token before touching the GPU and writes
through `guard9.write_result`, which voids the run if the lock, the code or the data moved while
it was in flight.

Usage:  python m9src/screen.py arm m9s1
        python m9src/screen.py decide
        python m9src/screen.py plan m9s6      (assemble and report, train nothing)
"""
import argparse
import json
import math
import time

import numpy as np
import torch

import m9base
from m9base import RESULTS, WORK

import data as m9data          # noqa: E402
import eval9                    # noqa: E402
import guard9                   # noqa: E402
import nano                     # noqa: E402
import screen_stats             # noqa: E402

RUNS = WORK / "m9runs"


def registry():
    return json.loads((m9base.M9 / "registry.json").read_text())


def constants():
    return json.loads((RESULTS / "m9_lock_constants.json").read_text())


def selected():
    p = RESULTS / "m9_screen_decisions.json"
    return json.loads(p.read_text())["selected"] if p.exists() else {}


def arm_spec(arm_id, sel=None):
    r = registry()
    row = dict(next(a for a in r["arms"] if a["id"] == arm_id))
    sel = sel or {}
    for k in ("teacher", "student", "prompt", "mix"):
        if row.get(k) == "selected":
            if k not in sel:
                raise SystemExit(f"{arm_id}: field {k!r} reads 'selected' but no earlier arm has "
                                 f"decided it -- run the screen in the registered order")
            row[k] = sel[k]
    return row


def arm_cfg(spec, sel=None):
    """-> the fully numeric training configuration for one arm."""
    r = registry()
    d = r["dose"]
    n = r["data"]["n_screen_queries"]
    bs = spec.get("batch_size") or int((sel or {}).get("batch_size") or d["batch_size"])
    if spec.get("epochs"):                                    # the batch pilot
        ex = spec["epochs"] * n
        steps = math.ceil(ex / bs)
    elif bs != d["batch_size"]:                               # registered batch-32 fallback dose
        fb = d["fallback_batch32"]
        assert bs == fb["batch_size"], f"no registered dose for batch size {bs}"
        ex, steps = fb["examples"], fb["steps"]
    else:
        ex, steps = d["examples"], d["steps"]
    warmup = max(1, int(round(0.03 * steps)))
    ck = [int(round(steps * k / 4)) for k in (1, 2, 3, 4)]
    if not spec.get("epochs") and bs == d["batch_size"]:
        ck, warmup = list(d["checkpoints"]), d["warmup_steps"]
    return {"batch_size": bs, "steps": steps, "examples": ex, "warmup_steps": warmup,
            "checkpoints": ck, "seed": int(spec.get("seed", d["seed"])),
            "lr_peak": d["lr_peak"], "lr_final": d["lr_final"],
            "epochs": spec.get("epochs") or (ex // n)}


def query_targets(teacher_key, texts, rows):
    if teacher_key == eval9.INCUMBENT:
        return np.asarray(m9data.stella_query_targets()[rows], dtype=np.float16)
    import teacher9
    v = teacher9.encode_cached(teacher_key, "m9screenq", texts, "query",
                               batch_size=64, max_length=512)
    out = np.asarray(v, dtype=np.float16)
    teacher9.release(teacher_key)
    return out


def doc_targets(teacher_key, rows, texts):
    if teacher_key == eval9.INCUMBENT:
        import pool as poolmod
        _i, vecs, _m = poolmod.build()
        return np.asarray(vecs[rows], dtype=np.float16)
    import teacher9
    v = teacher9.encode_cached(teacher_key, "m9screend", texts, "doc",
                               batch_size=32, max_length=512)
    out = np.asarray(v, dtype=np.float16)
    teacher9.release(teacher_key)
    return out


def build_plan(spec, cfg, tok):
    """-> (plan, meta). Token ids, teacher targets and the locked batch schedule."""
    r = registry()
    d, tpl = r["dose"], r["templates"]
    texts = json.loads((WORK / "m9_screen_queries.json").read_text())
    rows = np.load(WORK / "m9_screen_rows.npy")
    assert len(texts) == r["data"]["n_screen_queries"] == rows.size

    qprefix = tpl["query_policy_a_student"] if spec["prompt"] == "a" \
        else tpl["query_policy_b_student"]
    q_ids = nano.pretokenize(tok, [qprefix + t for t in texts], d["max_seq"], label="queries")
    q_tgt = query_targets(spec["teacher"], texts, rows)
    qlen = np.array([len(x) for x in q_ids], dtype=np.float64)
    meta = {"n_query_texts": len(texts), "student_query_prefix": qprefix,
            "query_tokens_per_epoch": int(qlen.sum())}

    if spec["mix"] == "query-only":
        order = nano.epoch_order(len(texts), cfg["epochs"], cfg["seed"])
        assert order.size == cfg["examples"], f"{order.size} vs {cfg['examples']}"
        flat, offs = nano.fixed_batches(order, cfg["batch_size"])
        ids, tgt = q_ids, q_tgt
    else:
        mx = d["mix_arm"]
        assert cfg["batch_size"] == d["batch_size"], "the mix arm is only defined at the main dose"
        cand, dmeta = m9data.doc_pool_rows(r["data"]["doc_candidates_n"],
                                           r["data"]["doc_candidates_seed"])
        assert dmeta["rows_sha256"] == r["data"]["doc_candidates_rows_sha256"], \
            "the doc candidate list does not match the M9.0 hash"
        n_docs = mx["n_docs_single_pass"]
        drows = cand[:n_docs]
        dtexts = m9data.row_texts(drows)
        d_ids = nano.pretokenize(tok, [tpl["doc_student"] + t for t in dtexts], d["max_seq"],
                                 label="docs")
        dlen = np.array([len(x) for x in d_ids], dtype=np.float64)
        assert set(map(tuple, q_ids[:2000])).isdisjoint(set(map(tuple, d_ids[:2000]))), \
            "query and document tokenized inputs overlap -- the role marker is not separating them"
        d_tgt = doc_targets(spec["teacher"], drows, dtexts)

        # query stream: consume the locked epoch order until the registered query-token target
        big = nano.epoch_order(len(texts), cfg["epochs"], cfg["seed"])
        cum = np.cumsum(qlen[big])
        nq = int(np.searchsorted(cum, mx["query_token_target"]) + 1)
        qstream = big[:nq]
        dstream = np.arange(n_docs, dtype=np.int64) + len(texts)
        alllen = np.concatenate([qlen, dlen])
        flat, offs, pos = nano.token_batches(
            [(qstream, alllen), (dstream, alllen)], cfg["steps"],
            d["per_step_token_budget"], [0.70, 0.30])
        ids, tgt = q_ids + d_ids, np.vstack([q_tgt, d_tgt])
        meta.update({"doc": dmeta, "n_doc_texts": n_docs, "student_doc_prefix": tpl["doc_student"],
                     "query_stream_len": int(nq), "doc_stream_len": int(dstream.size),
                     "consumed": {"query": pos[0], "doc": pos[1]},
                     "query_tokens": float(qlen[qstream].sum()),
                     "doc_tokens": float(dlen.sum())})

    meta["target_norms"] = {
        "min": round(float(np.linalg.norm(np.asarray(tgt[:256], np.float32), axis=1).min()), 6),
        "max": round(float(np.linalg.norm(np.asarray(tgt[:256], np.float32), axis=1).max()), 6)}
    meta["scheduled_examples"] = int(offs[-1])
    meta["scheduled_steps"] = int(len(offs) - 1)
    return {"ids": ids, "tgt": tgt, "flat": flat, "offs": offs}, meta


def run_arm(arm_id, smoke=0):
    r = registry()
    sel = selected()
    spec = arm_spec(arm_id, sel)
    cfg = arm_cfg(spec, sel)
    run_id = f"{arm_id}-smoke" if smoke else arm_id
    print(f"=== {run_id}: {json.dumps(spec)} | {json.dumps(cfg)}", flush=True)
    guard9.begin_run(run_id, extra={"spec": spec, "cfg": cfg})   # fail fast, before the GPU work

    stub = nano.Nano(spec["student"])
    tok = stub.tok
    del stub
    plan, meta = build_plan(spec, cfg, tok)
    if smoke:
        keep = int(np.searchsorted(plan["offs"], smoke)) + 1
        plan["offs"] = plan["offs"][:keep]
        cfg = {**cfg, "steps": keep - 1, "checkpoints": [keep - 1]}

    late = set(cfg["checkpoints"][-2:])

    def eval_fn(model, step):
        # SCREEN-3 every checkpoint; the three heavy DEV-6-only components at the last two only
        # (m9/registry.json dose.checkpoint_surfaces).
        comps = eval9.components("SCREEN3")
        if spec["teacher"] == eval9.INCUMBENT and step in late:
            comps = eval9.components("DEV6")
        per = eval9.eval_student(model, spec["teacher"], comps=comps)
        return {"macros": eval9.macros(per, spec["teacher"]), "per_component": per}

    rec, model = nano.train_arm(run_id, spec["student"], plan, cfg, eval_fn=eval_fn)
    rec.update({"spec": spec, "cfg": cfg, "plan": meta})
    rec["final"] = rec["history"][-1]["macros"]
    try:
        sym = eval9.cached_symmetric(spec["teacher"])
        rec["teacher_symmetric"] = eval9.macros(sym, spec["teacher"])
        rec["retention"] = {s: round(rec["final"][s]["macro"]
                                     / rec["teacher_symmetric"][s]["macro"], 4)
                            for s in rec["final"] if s in rec["teacher_symmetric"]}
    except Exception as e:                        # a ceiling row must never kill an arm
        rec["teacher_symmetric_error"] = repr(e)[:300]

    RUNS.mkdir(exist_ok=True)
    torch.save({"student": spec["student"], "state_dict": model.state_dict(), "spec": spec,
                "cfg": cfg}, RUNS / f"{run_id}.pt")
    guard9.write_result(RESULTS / f"m9_screen_{run_id}.json", rec, run_id)
    print(f"=== {run_id} DONE " + json.dumps({"final": rec["final"],
                                              "retention": rec.get("retention")}), flush=True)
    return rec


def _load():
    have = {}
    for a in registry()["arms"]:
        p = RESULTS / f"m9_screen_{a['id']}.json"
        if p.exists():
            blob = json.loads(p.read_text())
            if guard9.eligible(blob):
                have[a["id"]] = blob
            else:
                print(f"skipping {a['id']}: artifact is marked diagnostic, not decision-eligible")
    return have


def decide():
    """One function, fixed contrast orientation, shared resamples, registered outcome table."""
    have = _load()
    out, sel, notes = {}, {}, []

    def per(a):
        return have[a]["history"][-1]["per_component"]

    def hist(a):
        return have[a]["history"]

    # --- batch size -------------------------------------------------------------------
    if {"m9p-bs32", "m9p-bs128"} <= set(have):
        d = screen_stats.decide("batch_size", per("m9p-bs32"), per("m9p-bs128"))
        out["batch_size"] = d
        sel["batch_size"] = 32 if d["pass"] else 128

    # --- training-noise floor ---------------------------------------------------------
    F = None
    if {"m9s1", "m9s1b"} <= set(have):
        F = screen_stats.seed_floor(per("m9s1"), per("m9s1b"))
        out["seed_floor"] = {"F": F, "MDE": screen_stats.mde(F), "K": 2,
                             "limitation": registry()["rules"]["mde"]["limitation"]}

    # --- teacher (both challengers, shared resamples, then the outcome table) ----------
    if {"m9s1", "m9s2", "m9s3"} <= set(have):
        comps, _ = screen_stats.surface("SCREEN3")
        idx = screen_stats.indices(
            [(c, len(per("m9s1")[c])) for c in comps],
            registry()["statistic"]["B"], registry()["statistic"]["seed"])
        cand = {}
        for tag, arm in (("stella-1.5B-v5", "m9s2"), ("qwen3-embedding-0.6B", "m9s3")):
            d = screen_stats.decide("teacher_swap", per(arm), per("m9s1"), idx=idx)
            d["arm"] = arm
            cand[tag] = d
        out["teacher"] = cand
        win = [(t, d) for t, d in cand.items() if d["pass"]]
        if not win:
            sel["teacher"] = eval9.INCUMBENT
        else:
            best = max(win, key=lambda kv: kv[1]["point"])
            ties = [t for t, d in win if d["point"] == best[1]["point"]]
            sel["teacher"] = eval9.INCUMBENT if len(ties) > 1 else best[0]
            if sel["teacher"] != eval9.INCUMBENT:
                notes.append(
                    "TEACHER SWAP FIRED: student/prompt/mix cannot be decided on DEV-6 in a "
                    "challenger space. m9/LEDGER.md §0 requires stopping here and returning to "
                    "Dylan with the priced options. No downstream arm may run.")

    stop = sel.get("teacher", eval9.INCUMBENT) != eval9.INCUMBENT

    # --- student / prompt / mix -------------------------------------------------------
    if not stop and F is not None:
        anchor = "m9s1"
        if "m9s4" in have:
            d = screen_stats.decide("student", per("m9s4"), per(anchor), F=F,
                                    hist_a=hist("m9s4"), hist_b=hist(anchor))
            out["student"] = d
            sel["student"] = (have["m9s4"]["spec"]["student"] if d["pass"]
                              else have[anchor]["spec"]["student"])
            base = "m9s4" if d["pass"] else anchor
        else:
            base = anchor
        if "m9s5" in have:
            d = screen_stats.decide("prompt", per("m9s5"), per(base), F=F,
                                    hist_a=hist("m9s5"), hist_b=hist(base))
            out["prompt"] = {**d, "baseline_arm": base}
            sel["prompt"] = "a" if d["pass"] else "b"
            base = "m9s5" if d["pass"] else base
        if "m9s6" in have:
            d = screen_stats.decide("mix", per("m9s6"), per(base), F=F,
                                    hist_a=hist("m9s6"), hist_b=hist(base))
            out["mix"] = {**d, "baseline_arm": base}
            sel["mix"] = "70/30" if d["pass"] else "query-only"
    elif not stop:
        notes.append("student/prompt/mix withheld: the seed floor F needs arm m9s1b, which has "
                     "not run. The MDE is not defined without it.")

    payload = {
        "arms_present": sorted(have), "decisions": out, "selected": sel, "notes": notes,
        "final_macros": {k: v.get("final") for k, v in have.items()},
        "retention": {k: v.get("retention") for k, v in have.items()},
        "curves": {k: [{"step": h["step"], "examples": h["examples"],
                        "nonpad_tokens": h["nonpad_tokens"],
                        **{s: h["macros"][s]["macro"] for s in h["macros"]}}
                       for h in v["history"]] for k, v in have.items()},
        "scope": registry()["rules"]["scope"]}
    guard9.begin_run("m9-decisions")
    guard9.write_result(RESULTS / "m9_screen_decisions.json", payload, "m9-decisions")
    print(json.dumps({k: v for k, v in payload.items() if k != "decisions"}, indent=1)[:3000])
    for k, v in out.items():
        if isinstance(v, dict) and "point" in v:
            print(f"{k:14s} point {v['point']:+.5f} lower({v['quantile']}) "
                  f"{v['lower_bound']:+.5f} thr {v['threshold']:.5f} -> {v['action']}")
    for n in notes:
        print("NOTE: " + n)
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["arm", "decide", "plan"])
    ap.add_argument("arm_id", nargs="?")
    ap.add_argument("--smoke", type=int, default=0, help="examples; writes to <id>-smoke")
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd == "arm":
        run_arm(a.arm_id, smoke=a.smoke)
    elif a.cmd == "decide":
        decide()
    else:
        sel = selected()
        spec = arm_spec(a.arm_id, sel)
        print(json.dumps({"spec": spec, "cfg": arm_cfg(spec, sel)}, indent=1))
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
