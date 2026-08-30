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
    if not p.exists():
        return {}
    blob = json.loads(p.read_text())
    if not guard9.eligible(blob):
        raise SystemExit("results/m9_screen_decisions.json is not decision-eligible under the "
                         "current lock -- re-run `screen.py decide` before resolving any arm")
    return blob["selected"]


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
    """-> the fully numeric training configuration for one arm. One dose, one batch size; there
    are no pilot arms and no fallback regimes left to disagree with each other."""
    d = registry()["dose"]
    return {"batch_size": d["batch_size"], "steps": d["steps"], "examples": d["examples"],
            "warmup_steps": d["warmup_steps"], "checkpoints": list(d["checkpoints"]),
            "seed": int(spec.get("seed", d["seed"])),
            "lr_peak": d["lr_peak"], "lr_final": d["lr_final"],
            "epochs": d["epochs_query_only"],
            "token_matched": spec["prompt"] == "a" or spec["mix"] != "query-only"}


def require_predecessors(arm_id):
    """Refuse an arm whose registered predecessors have not produced eligible artifacts, and
    refuse every stage-B arm until the adequacy gate has passed (Codex pass 2, MAJOR-12)."""
    st = registry()["staging"]
    stage = "A" if arm_id in st["A"]["runs"] else "B"
    order = st["A"]["runs"] + st["B"]["runs"]
    assert arm_id in order, f"{arm_id} is not a registered run"
    for prev in order[:order.index(arm_id)]:
        if prev.startswith("m9-"):
            continue                      # the pilots are checked by their own gates
        p = RESULTS / f"m9_screen_{prev}.json"
        if not p.exists() or not guard9.eligible(json.loads(p.read_text())):
            raise SystemExit(f"{arm_id} may not run: predecessor {prev} has no eligible artifact. "
                             f"The registered order is {order}.")
    # A teacher swap STOPS the milestone; it does not merely annotate it (LEDGER §0).
    dec = RESULTS / "m9_screen_decisions.json"
    if dec.exists():
        blob = json.loads(dec.read_text())
        t = blob.get("selected", {}).get("teacher")
        if guard9.eligible(blob) and t and t != eval9.INCUMBENT:
            raise SystemExit(
                f"{arm_id} may not run: the teacher screen selected {t!r}. m9/LEDGER.md §0 "
                f"requires stopping and returning to Dylan, because student/prompt/mix cannot be "
                f"decided on DEV-6 in a challenger's space and M9 does not proceed on a proxy.")
    if stage == "B":
        g = RESULTS / "m9_adequacy.json"
        ok = False
        if g.exists():
            blob = json.loads(g.read_text())
            # recomputed, not trusted: a stored boolean cannot authorize six GPU-hours
            ok = guard9.eligible(blob) and adequacy_verdict()["pass"]
        if not ok:
            raise SystemExit(
                f"{arm_id} is a stage-B arm and the adequacy gate has not passed. Run "
                f"`python m9src/screen.py adequacy` after m9s1; if it fails, stage B does not run "
                f"and M9.1 reports the anchor curve instead (m9/registry.json adequacy_gate).")


def query_targets(teacher_key, texts, rows):
    import fp16_gate
    ok, why = fp16_gate.passed()
    if not ok:
        raise SystemExit(f"fp16 targets refused: {why}")
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

    if spec["mix"] == "query-only" and spec["prompt"] != "a":
        order = nano.epoch_order(len(texts), cfg["epochs"], cfg["seed"])
        assert order.size == cfg["examples"], f"{order.size} vs {cfg['examples']}"
        flat, offs = nano.fixed_batches(order, cfg["batch_size"])
        ids, tgt = q_ids, q_tgt
        meta["dose_form"] = "fixed 128-example batches over 16 full epochs"
    elif spec["mix"] == "query-only":
        # Prompt policy (a) prepends ~20 tokens to every query, so 16 epochs would be a LARGER
        # non-pad dose than the baseline and the contrast would confound prompt with dose
        # (Codex pass 2, BLOCKER-2). Match tokens and optimizer updates instead; the arm therefore
        # sees fewer presentations of the same pool, which is what a fixed compute budget means.
        big = nano.epoch_order(len(texts), cfg["epochs"], cfg["seed"])
        flat, offs, pos, real = nano.token_batches([(big, qlen)], cfg["steps"],
                                                   d["T_base_nonpad_tokens"], [1.0])
        ids, tgt = q_ids, q_tgt
        meta.update({"dose_form": "token-matched: same steps and same non-pad tokens as baseline",
                     "consumed": {"query": pos[0]}, "realized_tokens": real,
                     "presentations": int(offs[-1])})
    else:
        mx = d["mix_arm"]
        assert cfg["batch_size"] == d["batch_size"], "the mix arm is only defined at the main dose"
        cand, dmeta = m9data.doc_pool_rows(r["data"]["doc_candidates_n"],
                                           r["data"]["doc_candidates_seed"])
        assert dmeta["rows_sha256"] == r["data"]["doc_candidates_rows_sha256"], \
            "the doc candidate list does not match the M9.0 hash"
        n_docs = mx["n_doc_candidates"]
        drows = cand[:n_docs]
        dtexts = m9data.row_texts(drows)
        d_ids = nano.pretokenize(tok, [tpl["doc_student"] + t for t in dtexts], d["max_seq"],
                                 label="docs")
        dlen = np.array([len(x) for x in d_ids], dtype=np.float64)
        # the FULL pinned inputs, not a 2,000-row corner of them (Codex pass 2, MINOR role rule)
        assert set(map(tuple, q_ids)).isdisjoint(set(map(tuple, d_ids))), \
            "query and document tokenized inputs overlap -- the role marker is not separating them"
        d_tgt = doc_targets(spec["teacher"], drows, dtexts)

        # Both streams are long enough by construction (16 epochs of queries covers 100% of
        # T_base and the document candidate list covers well over 30%), so the batcher stops on
        # the step count and never on exhaustion.
        qstream = nano.epoch_order(len(texts), cfg["epochs"], cfg["seed"])
        dstream = np.arange(n_docs, dtype=np.int64) + len(texts)
        alllen = np.concatenate([qlen, dlen])
        flat, offs, pos, real = nano.token_batches(
            [(qstream, alllen), (dstream, alllen)], cfg["steps"],
            d["T_base_nonpad_tokens"], [0.70, 0.30])
        ids, tgt = q_ids + d_ids, np.vstack([q_tgt, d_tgt])
        meta.update({"dose_form": "token-matched 70/30 by TOKEN, same steps as baseline",
                     "doc": dmeta, "n_doc_candidates": n_docs,
                     "student_doc_prefix": tpl["doc_student"],
                     "consumed": {"query": pos[0], "doc": pos[1]},
                     "realized_tokens": real, "presentations": int(offs[-1]),
                     "doc_share_realized": round(real[1] / sum(real), 4)})

    meta["target_norms"] = {
        "min": round(float(np.linalg.norm(np.asarray(tgt[:256], np.float32), axis=1).min()), 6),
        "max": round(float(np.linalg.norm(np.asarray(tgt[:256], np.float32), axis=1).max()), 6)}
    meta["scheduled_examples"] = int(offs[-1])
    meta["scheduled_steps"] = int(len(offs) - 1)
    return {"ids": ids, "tgt": tgt, "flat": flat, "offs": offs}, meta


def run_arm(arm_id, smoke=0):
    r = registry()
    if not smoke:
        require_predecessors(arm_id)
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

    rec, model = nano.train_arm(run_id, spec["student"], plan, cfg, eval_fn=eval_fn,
                                warm_start=spec.get("warm_start", True),
                                warm_texts=json.loads((WORK / "m9_screen_queries.json").read_text()),
                                warm_prefix=meta["student_query_prefix"])
    rec.update({"spec": spec, "cfg": cfg, "plan": meta})
    rec["final"] = rec["history"][-1]["macros"]
    try:
        sym = eval9.cached_symmetric(spec["teacher"])
        rec["teacher_symmetric"] = eval9.macros(sym, spec["teacher"])
        rec["retention"] = {s: round(rec["final"][s]["macro"]
                                     / rec["teacher_symmetric"][s]["macro"], 4)
                            for s in rec["final"] if s in rec["teacher_symmetric"]}
    except Exception as e:
        # A missing ceiling is not cosmetic: it is the retention denominator and the adequacy
        # gate's input, so the arm records the failure AND refuses to look complete.
        rec["teacher_symmetric_error"] = repr(e)[:300]
        raise

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


def adequacy_verdict():
    """Recompute the gate from the artifacts. Pure; `adequacy()` is the writing wrapper."""
    import hashlib
    r = registry()
    g = r["adequacy_gate"]
    blob = json.loads((RESULTS / "m9_screen_m9s1.json").read_text())
    assert guard9.eligible(blob), "m9s1's artifact is not decision-eligible"
    cl = r["ceilings"]["stella-400M-v5"]
    got = hashlib.sha256((m9base.REPO / cl["artifact"]).read_bytes()).hexdigest()
    assert got == cl["sha256"], (
        f"the ceiling artifact {cl['artifact']} hashes {got[:12]}, the registry pins "
        f"{cl['sha256'][:12]} -- the retention denominator is not the registered one")
    ceil6 = cl["DEV6"]
    ck = r["dose"]["checkpoints"]
    m = {h["step"]: h["macros"]["DEV6"]["macro"] for h in blob["history"]
         if "DEV6" in h.get("macros", {})}
    missing = [c for c in ck[-2:] if c not in m]
    ret = m.get(ck[-1], 0.0) / ceil6
    slope = (m.get(ck[-1], 0.0) - m.get(ck[-2], 0.0)) if not missing else None
    out = {"anchor": "m9s1", "ceiling_dev6": ceil6, "final_macro": m.get(ck[-1]),
           "retention": round(ret, 4), "late_slope": slope,
           "conditions": g["conditions"], "missing_checkpoints": missing,
           "pass_retention": bool(ret >= g["conditions"]["retention_at_final"]["min"]),
           "pass_slope": bool(slope is not None
                              and slope <= g["conditions"]["late_slope"]["max"]),
           "curve": sorted(m.items())}
    out["pass"] = bool(not missing and out["pass_retention"] and out["pass_slope"])
    out["action"] = g["outcomes"]["pass" if out["pass"] else "fail"]
    out["kind"] = g["kind"]
    return out


def adequacy():
    """The registered budget trigger between stage A and stage B."""
    out = adequacy_verdict()
    guard9.begin_run("m9-adequacy")
    guard9.write_result(RESULTS / "m9_adequacy.json", out, "m9-adequacy")
    print(json.dumps({k: v for k, v in out.items() if k != "conditions"}, indent=1))
    return out


def decide():
    """One function, fixed contrast orientation, shared resamples, registered outcome table."""
    have = _load()
    out, sel, notes = {}, {}, []

    def per(a):
        return have[a]["history"][-1]["per_component"]

    def hist(a):
        return have[a]["history"]

    if {"m9s1", "m9s1b"} <= set(have):
        out["seed_sensitivity"] = screen_stats.seed_sensitivity(per("m9s1"), per("m9s1b"))
    if {"m9s1", "m9s1c"} <= set(have):
        a, b = have["m9s1"], have["m9s1c"]
        out["warm_start_value"] = {
            "delta_dev6": round(a["final"]["DEV6"]["macro"] - b["final"]["DEV6"]["macro"], 6),
            "status": "DIAGNOSTIC -- prices the closed-form head warm start, decides nothing"}

    if {"m9s1", "m9s2", "m9s3"} <= set(have):
        comps, _ = screen_stats.surface("SCREEN3")
        idx = screen_stats.indices([(c, len(per("m9s1")[c])) for c in comps],
                                   registry()["statistic"]["B"],
                                   registry()["statistic"]["seed"])
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
                    "TEACHER SWAP FIRED. m9/LEDGER.md §0 requires STOPPING here: student, prompt "
                    "and mix cannot be decided on DEV-6 in a challenger's space, and M9 does not "
                    "proceed on a proxy. No downstream arm may run until Dylan rules.")

    stop = sel.get("teacher", eval9.INCUMBENT) != eval9.INCUMBENT
    if not stop:
        base = "m9s1"
        if "m9s4" in have:
            d = screen_stats.decide("student", per("m9s4"), per(base),
                                    hist_a=hist("m9s4"), hist_b=hist(base))
            out["student"] = d
            sel["student"] = (have["m9s4"]["spec"]["student"] if d["pass"]
                              else have[base]["spec"]["student"])
            base = "m9s4" if d["pass"] else base
        if "m9s5" in have:
            d = screen_stats.decide("prompt", per("m9s5"), per(base),
                                    hist_a=hist("m9s5"), hist_b=hist(base))
            out["prompt"] = {**d, "baseline_arm": base}
            sel["prompt"] = "a" if d["pass"] else "b"
            base = "m9s5" if d["pass"] else base
        if "m9s6" in have:
            d = screen_stats.decide("mix", per("m9s6"), per(base),
                                    hist_a=hist("m9s6"), hist_b=hist(base))
            out["mix"] = {**d, "baseline_arm": base}
            sel["mix"] = "70/30" if d["pass"] else "query-only"

    payload = {
        "arms_present": sorted(have), "decisions": out, "selected": sel, "notes": notes,
        "final_macros": {k: v.get("final") for k, v in have.items()},
        "retention": {k: v.get("retention") for k, v in have.items()},
        "curves": {k: [{"step": h["step"], "examples": h["examples"],
                        "nonpad_tokens": h["nonpad_tokens"],
                        **{s_: h["macros"][s_]["macro"] for s_ in h["macros"]}}
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
    ap.add_argument("cmd", choices=["arm", "decide", "plan", "adequacy"])
    ap.add_argument("arm_id", nargs="?")
    ap.add_argument("--smoke", type=int, default=0, help="examples; writes to <id>-smoke")
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd == "arm":
        run_arm(a.arm_id, smoke=a.smoke)
    elif a.cmd == "decide":
        decide()
    elif a.cmd == "adequacy":
        adequacy()
    else:
        sel = selected()
        spec = arm_spec(a.arm_id, sel)
        print(json.dumps({"spec": spec, "cfg": arm_cfg(spec, sel)}, indent=1))
    print(f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
