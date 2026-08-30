"""B8 -- target design: does a DOCUMENT-CENTROID target beat the bare teacher-query target?

THE QUESTION, and why it is not idle. The table is fitted to reproduce the teacher's QUERY vector,
but nothing at retrieval time ever compares a query to a query: the score is `q . d` against
DOCUMENT vectors. If the teacher's query embedding sits somewhere the document manifold does not,
the table is being aimed at the wrong point, and the cheapest test of that is to aim it at the
positives' centroid instead and re-solve in closed form.

WHY IT RUNS NOW. Its bar read `TBD-noise-floor` for the whole milestone although the floor was
measured on 2026-08-29, so `probe_guard` refused it and it was deferred behind a multi-hour
tokenizer tournament -- the "false economy" the worklist named. Bar frozen 2026-08-29 (LEDGER §15).

THE THREE TARGETS, all solved with the identical X, W0 and lambda grid so only Y differs:
  bare      R0's own target: the teacher's prefixed query vector. THE COMPARATOR.
  centroid  the L2-normalized mean of the query's POSITIVE document vectors, from the frozen pool.
  mix50     the normalized half-and-half of the two normalized targets.

WHAT IT MAY DO. Nothing is adopted from a closed-form fit -- the same rule `D2-PRE` ran under. A
target that clears 0.0040 on the dev group vector buys a TRAINED confirmation at the chain bar
0.00519, never a place in R1 directly.
"""
import argparse
import json
import sys
import time

import numpy as np

import m8base
import blockcg
import probe_guard
from d2_pre import bag_matrix, counts_of, encode, ids_of, r0_denominator, tok_pre

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_b8_target.json"
OOD = ("cqadup-programmers", "cqadup-physics")
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
TARGETS = ("bare", "centroid", "mix50")
BAR = 0.0040


def _norm(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def build_targets(max_pairs=None):
    """(texts, {target: Y}) for decontaminated TRAIN pairs whose positives resolve in the pool."""
    import torch
    import mix
    import pool
    from teacher import QUERY_PREFIX, encode_cached

    tr, _ = mix.split_pairs()
    kept = json.loads((m8base.WORK / "decontam" / "kept.json").read_text())
    allow = {k: set(v) for k, v in kept.items()}
    tr = [p for p in tr if p[1] in allow.get(p[0], set())]
    print(f"{len(tr):,} decontaminated train pairs", flush=True)

    idx, vecs, meta = pool.build()
    stores = {s: mix.load_source(s)["docstore"] for s in {p[0] for p in tr}}
    texts, rows, dropped = [], [], 0
    for src, qid, q, pos, _hn in tr:
        st = stores[src]
        r = [idx.get(st, d) for d in pos]
        r = [x for x in r if x is not None]
        if not r:
            dropped += 1
            continue
        texts.append(q)
        rows.append(r)
        if max_pairs and len(texts) >= max_pairs:
            break
    print(f"{len(texts):,} usable ({dropped:,} dropped: no positive resolves in the pool)",
          flush=True)

    cent = np.empty((len(texts), vecs.shape[1]), dtype=np.float32)
    for i, r in enumerate(rows):
        cent[i] = np.asarray(vecs[r], dtype=np.float32).mean(0)
    cent = _norm(cent)
    bare = _norm(np.asarray(encode_cached(f"b8-trainq-{len(texts)}", texts, prefix=QUERY_PREFIX,
                                          dtype=torch.float16, verbose=False), dtype=np.float32))
    return texts, {"bare": bare, "centroid": cent, "mix50": _norm(0.5 * bare + 0.5 * cent)}, \
        {"n_pairs": len(texts), "dropped_no_positive": dropped,
         "mean_positives_per_query": float(np.mean([len(r) for r in rows]))}


def group_vector(per_q):
    """§4.7's registered group vector, via the same helper the floors use."""
    import noise_floor
    return noise_floor._group_vector(per_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    device = m8base.device()

    import dev_eval
    from init_table import get_init

    tok, pre = tok_pre()
    V = len(tok)
    texts, Y, prov = build_targets(max_pairs=a.max_pairs or (20_000 if a.smoke else None))
    ids = ids_of(tok, texts, pre)
    X = bag_matrix([counts_of(x) for x in ids], r0_denominator(ids), V)
    W0 = get_init("teacher", pre)
    print(f"X {X.shape} nnz={X.nnz:,}  W0 {W0.shape}  [{time.time()-t0:.0f}s]", flush=True)

    # dev query ids/texts for the CHEAP selection endpoint (the OOD pair)
    ood_ids = {}
    for c in OOD:
        _, _, _, qt, _, _ = dev_eval.doc_vecs(c)
        ood_ids[c] = (list(qt), ids_of(tok, list(qt), pre))

    lam_curve, best = {}, {}
    for t in TARGETS:
        lam_curve[t] = {}
        for lam in LAMBDAS:
            W, info = blockcg.block_cg_ridge(X, Y[t], W0, lam, device=device)
            W = W.astype(np.float32)
            m = float(np.mean([
                np.mean(list(dev_eval.eval_query_vecs(
                    c, encode([counts_of(x) for x in ood_ids[c][1]], W)).values()))
                for c in OOD]))
            lam_curve[t][lam] = m
            if best.get(t, (None, -1))[1] < m:
                best[t] = (lam, m, W)
            print(f"  [{t:8s}] lam={lam:<7g} ood={m:.6f} ({info['iterations']} its)", flush=True)
        lam, m, _ = best[t]
        print(f"  [{t:8s}] argmax lam={lam:g} ood={m:.6f}"
              f"{'  BOUNDARY' if lam in (min(LAMBDAS), max(LAMBDAS)) else '  interior'}", flush=True)

    # the registered endpoint: the dev GROUP VECTOR, for the selected table of each target only
    # COMPONENTS ARE THE OUTER LOOP. `dev_eval.doc_vecs` re-reads its corpus on every call and
    # HotpotQA's is 5.23M documents (~10.7 GB of fp16 vectors), so target-outer would load it three
    # times. Same reversal that turned a 3.6-hour job into 165 s in M7 (CODEMAP pitfall 6).
    import gc
    comps = list(OOD) if a.smoke else dev_eval.dev_components()
    per = {t: {} for t in TARGETS}
    for c in comps:
        tc = time.time()
        _, _, _, qt, _, _ = dev_eval.doc_vecs(c)
        f = [counts_of(x) for x in ids_of(tok, list(qt), pre)]
        for t in TARGETS:
            per[t][c] = dev_eval.eval_query_vecs(c, encode(f, best[t][2]))
        del f
        gc.collect()
        m8base.empty_cache()
        print(f"  {c}: 3 targets scored ({time.time()-tc:.0f}s)", flush=True)
    gv = {}
    for t in TARGETS:
        gv[t] = group_vector(per[t]) if not a.smoke else {
            "out_of_domain_macro": float(np.mean([np.mean(list(per[t][c].values()))
                                                  for c in OOD]))}
        print(f"  [{t:8s}] group vector "
              f"{json.dumps({k: v for k, v in gv[t].items() if k != 'component_means'})}",
              flush=True)

    key = "group_vector_median" if not a.smoke else "out_of_domain_macro"
    delta = {t: gv[t][key] - gv["bare"][key] for t in TARGETS if t != "bare"}
    winner = max(delta, key=lambda t: delta[t])
    out = {
        "_note": __doc__.strip().splitlines()[0],
        "targets": {t: {"argmax_lambda": best[t][0], "ood_macro_at_argmax": best[t][1],
                        "lambda_curve": lam_curve[t],
                        "argmax_at_grid_boundary": best[t][0] in (min(LAMBDAS), max(LAMBDAS)),
                        "group_vector": gv[t]} for t in TARGETS},
        "endpoint": key, "delta_vs_bare": delta, "bar": BAR,
        "verdict": (f"ROUTE {winner} to a trained confirmation" if delta[winner] >= BAR
                    else "NO SURVIVOR -- keep R0's target"),
        "cleared_bar": {t: bool(v >= BAR) for t, v in delta.items()},
        "provenance": prov, "components": comps, "smoke": bool(a.smoke),
        "adopts": ("nothing. A closed-form fit routes; it never ships. A target clearing 0.0040 "
                   "here buys a TRAINED confirmation at the chain bar 0.00519."),
        "lambda_rule": ("argmax on the OOD pair, the same instrument T1's teacher screen used; "
                        "boundary attainment is reported per target because an argmax at the grid "
                        "edge means the grid did not bracket the optimum."),
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps({"delta_vs_bare": delta, "verdict": out["verdict"]}, indent=2))
    dest = (RESULTS / "m8_b8_target.SMOKE.json") if a.smoke else OUT
    if a.smoke:
        dest.write_text(json.dumps(out, indent=2, default=str))
    else:
        probe_guard.write_result(dest, out, "B8")
    print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
