"""M12 Tier 1: reproduce M7's fusion grid, then score Qdrant's two shipping operators against it.

Registered in m12/LEDGER.md BEFORE this ran. Dev only, four text-backed components, depth 1000,
int8 release table. Descriptive: nothing here replaces the published 0.4911, and no number here is
about the six.

Order matters and is not negotiable. The reproduction gate runs FIRST and gates everything: M7's
grid was computed on a different torch/CUDA build, and GPU top-k tie order and fp16 accumulation
move the 4th-5th decimal. Comparing a new operator against the frozen literal would fold
run-reproduction noise into every delta and call it an operator effect.

  PYTHONPATH=m12src:m7src M7_ENCODER=stella-400M-v5 .venv/bin/python m12src/run_tier1.py
"""
import json
import sys
import time

import numpy as np

import dev_eval
import freeze
import fusion
import qfusion
from _paths import REPO, WORK
from evalkit import per_query_ndcg, run_from_arrays, topk_arrays
from hashing import sha_stream_list
from table import Preproc, ensure_release, load_table, read_meta

RUN_ID = "p35w-2m-s2500"
OUT = WORK / "m12"
GATE_TOL = 1e-4          # m7/STATUS.md:92-93, M7's own reproduction tolerance
BAR = 0.004              # results/m8_noise_floor_fused.json (a training-seed floor; see LEDGER)
K_GRID = [1, 2, 3, 4, 6, 11, 21, 31, 61, 101]        # Qdrant units. k_q = k_ours + 1.
DEPTHS = [10, 50, 100, 1000]
BOOT_B, BOOT_SEED = 10_000, 12


def dense_runs_cached(comps, model, pre, table_sha):
    """Dense top-1000 per component, persisted. `select_fusion.dense_run` returns a dict and writes
    nothing, so without this the depth curve and the bootstrap would each cost another GPU pass --
    the mandate's "one GPU pass total" is only true if the arrays land on disk."""
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    for c in comps:
        doc_ids, _, q_ids, q_texts, _, dv = dev_eval.doc_vecs(c)
        key = {"table_sha256": table_sha, "component": c, "depth": fusion.DEPTH,
               "n_docs": len(doc_ids), "n_queries": len(q_ids),
               "doc_ids_sha256": sha_stream_list(doc_ids), "q_ids_sha256": sha_stream_list(q_ids)}
        path = OUT / f"dense-{c}-d{fusion.DEPTH}.npz"
        bi = bs = None
        if path.exists():
            try:
                z = np.load(path, allow_pickle=False)
                if json.loads(bytes(z["key"]).decode()) == key:
                    bi, bs = z["bi"], z["bs"]
                    print(f"  {c}: dense run from cache", flush=True)
                else:
                    print(f"  {c}: dense cache REBUILD (key mismatch)", flush=True)
            except Exception as e:
                print(f"  {c}: dense cache REBUILD ({type(e).__name__})", flush=True)
        if bi is None:
            t = time.time()
            bi, bs = topk_arrays(model.encode(q_texts, pre), dv, k=fusion.DEPTH,
                                 chunk=dev_eval.CHUNK.get(c, 250_000))
            bi, bs = bi.astype(np.int32), bs.astype(np.float32)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, bi=bi, bs=bs,
                                key=np.frombuffer(json.dumps(key, sort_keys=True).encode(),
                                                  dtype=np.uint8))
            tmp.replace(path)
            print(f"  {c}: dense run built in {time.time() - t:.0f}s", flush=True)
        runs[c] = run_from_arrays(bi.astype(np.int64), bs, doc_ids, q_ids)
    return runs


def macro_and_per(fused, qrels, comps):
    per = {c: per_query_ndcg(fused[c], qrels[c]) for c in comps}
    return float(np.mean([np.mean(list(per[c].values())) for c in comps])), per


def bootstrap(per_a, per_b, comps, b=BOOT_B, seed=BOOT_SEED):
    """Paired bootstrap of (a - b) on the macro. Resampled WITHIN component, then macro-of-means:
    a pooled resample would weight hotpotqa's 7,405 queries 8x physics's 1,039 and is not the
    statistic. A CI on a FITTED winner is post-selection and optimistic -- say so when reporting."""
    rng = np.random.default_rng(seed)
    cols = []
    for c in comps:
        qs = sorted(per_a[c])
        cols.append((np.array([per_a[c][q] for q in qs]), np.array([per_b[c][q] for q in qs])))
    diffs = np.empty(b)
    for i in range(b):
        d = [(a[idx].mean() - bb[idx].mean())
             for a, bb in cols
             for idx in (rng.integers(0, len(a), len(a)),)]
        diffs[i] = np.mean(d)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    npz = ensure_release(WORK / "runs" / f"{RUN_ID}.npz")
    meta = read_meta(npz)
    assert meta.get("weights_folded"), f"{npz} is not a release-shape artifact"
    freeze.assert_encoder_matches_artifact(meta, "M12 TIER 1")
    spec = json.loads((REPO / "m7" / "FREEZE.json").read_text())
    table_sha = freeze.sha256_file(npz)
    if table_sha != spec["fusion"]["selected_on"]["table_sha256"]:
        raise SystemExit(f"M12 REFUSED: table sha {table_sha} != m7/FREEZE.json")
    pre = Preproc(**meta["preproc"])
    comps = [c for c in dev_eval.dev_components() if not c.startswith("heldout-")]
    print(f"M12 Tier 1 on {comps} at depth {fusion.DEPTH}\n")

    model = load_table(npz, variant="int8")
    dense = dense_runs_cached(comps, model, pre, table_sha)
    del model
    bm25, qrels, collisions = {}, {}, {}
    for c in comps:
        doc_ids, doc_texts, q_ids, q_texts, qr, _ = dev_eval.doc_vecs(c)
        bm25[c] = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts,
                                  cache_path=WORK / "fusionruns" / f"bm25-{c}-d{fusion.DEPTH}.npz")
        qrels[c] = qr
        # both runs drop self-hits AFTER retrieval; Qdrant does not, so a d=10 list may hold 9
        collisions[c] = len(set(q_ids) & set(doc_ids))
    print(f"  self-hit collisions per component: {collisions}\n")

    # ---- GATE: reproduce M7's 21 points before scoring anything new -------------------------
    frozen = {(g["family"], g["param"]): g["macro"] for g in spec["fusion"]["grid"]}
    repro, worst = {}, 0.0
    for (fam, p), want in frozen.items():
        fused = ({c: fusion.rrf([dense[c], bm25[c]], k=p) for c in comps} if fam == "rrf"
                 else {c: fusion.convex([dense[c], bm25[c]], w=p, floor_zero=(fam == "convex0"))
                       for c in comps})
        got, per = macro_and_per(fused, qrels, comps)
        repro[f"{fam}:{p}"] = {"m7": want, "m12": got, "delta": got - want}
        worst = max(worst, abs(got - want))
        if fam == "convex0" and p == 0.8:
            comparator, comparator_per = got, per
    print(f"  reproduction gate: max |delta| over 21 points = {worst:.2e} "
          f"(tolerance {GATE_TOL:g})")
    if worst > GATE_TOL:
        (REPO / "m12" / "tier1.json").write_text(json.dumps(
            {"gate": "FAILED", "tolerance": GATE_TOL, "max_abs_delta": worst,
             "reproduction": repro}, indent=2))
        raise SystemExit("M12 STOPPED: M7's grid does not reproduce within tolerance. That is the "
                         "finding; no new operator may be scored against a comparator we cannot "
                         "reproduce. See m12/tier1.json.")
    bar = comparator - BAR
    print(f"  comparator C (convex0 w=0.8, recomputed) = {comparator:.10f}")
    print(f"  frozen literal                           = {frozen[('convex0', 0.8)]:.10f}")
    print(f"  bar B = C - {BAR}                        = {bar:.10f}\n")

    # ---- TIER 1 -----------------------------------------------------------------------------
    rows, pers = [], {}

    def add(label, fused, budget, tier):
        m, per = macro_and_per(fused, qrels, comps)
        pers[label] = per
        rows.append({"label": label, "macro": m, "candidates": budget, "tier": tier,
                     "delta_vs_C": m - comparator, "passes": bool(m >= bar)})
        print(f"  {label:<34} {m:.4f}   ({m - comparator:+.4f})")
        return m

    print("  M7's row, the thing being corrected (our k units, 5 badly-placed points):")
    m7_best_k = max(fusion.RRF_K, key=lambda k: macro_and_per(
        {c: fusion.rrf([dense[c], bm25[c]], k=k) for c in comps}, qrels, comps)[0])
    add(f"M7 rrf (ours) k={m7_best_k}",
        {c: fusion.rrf([dense[c], bm25[c]], k=m7_best_k) for c in comps}, 5, 0)

    print("\n  Tier 1 — Qdrant's shipping operators:")
    add("DBSF", {c: qfusion.dbsf([dense[c], bm25[c]]) for c in comps}, 0, 1)
    k_scores = {}
    for k in K_GRID:
        k_scores[k] = macro_and_per({c: qfusion.rrf([dense[c], bm25[c]], k=k) for c in comps},
                                    qrels, comps)[0]
        print(f"    rrf k_q={k:<4} {k_scores[k]:.4f}")
    best_k = max(k_scores, key=k_scores.get)
    add(f"RRF over k (best k_q={best_k})",
        {c: qfusion.rrf([dense[c], bm25[c]], k=best_k) for c in comps}, len(K_GRID), 1)

    # ---- CIs and the depth curve, for the rows that matter ---------------------------------
    for r in rows:
        r["ci95_vs_C"] = bootstrap(pers[r["label"]], comparator_per, comps)
        print(f"  CI  {r['label']:<34} {r['delta_vs_C']:+.4f} "
              f"[{r['ci95_vs_C'][0]:+.4f}, {r['ci95_vs_C'][1]:+.4f}]")

    winners = ["DBSF", f"RRF over k (best k_q={best_k})"]
    depth = {}
    for d in DEPTHS:
        td = {c: qfusion.truncate(dense[c], d) for c in comps}
        tb = {c: qfusion.truncate(bm25[c], d) for c in comps}
        depth[d] = {
            "convex0_w0.8": macro_and_per(
                {c: fusion.convex([td[c], tb[c]], w=0.8, floor_zero=True) for c in comps},
                qrels, comps)[0],
            "DBSF": macro_and_per({c: qfusion.dbsf([td[c], tb[c]]) for c in comps}, qrels, comps)[0],
            f"RRF k_q={best_k}": macro_and_per(
                {c: qfusion.rrf([td[c], tb[c]], k=best_k) for c in comps}, qrels, comps)[0]}
        print(f"  depth {d:<5} " + "  ".join(f"{k}={v:.4f}" for k, v in depth[d].items()))

    passed = [r for r in rows if r["tier"] == 1 and r["passes"]]
    out = {"gate": "PASSED", "tolerance": GATE_TOL, "max_abs_delta": worst,
           "reproduction": repro, "components": comps, "depth": fusion.DEPTH,
           "table_sha256": table_sha, "self_hit_collisions": collisions,
           "comparator_recomputed": comparator, "comparator_frozen_literal": frozen[("convex0", 0.8)],
           "bar": bar, "bar_source": "max(0.0040, 2x floor), m8_noise_floor_fused.json (seed floor)",
           "k_grid_macros": {str(k): v for k, v in k_scores.items()},
           "rows": rows, "depth_curve": depth,
           "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "resample": "within component, macro of means"},
           "tier1_result": "MATCH" if passed else "NO MATCH",
           "tier2_required": not passed}
    (REPO / "m12" / "tier1.json").write_text(json.dumps(out, indent=2))
    print(f"\n  TIER 1: {out['tier1_result']}  ->  "
          + ("Tier 2 (weighted RRF) is NOT run" if passed
             else "Tier 2 (weighted RRF, split-half) is required"))
    print("  wrote m12/tier1.json")


if __name__ == "__main__":
    sys.exit(main())
