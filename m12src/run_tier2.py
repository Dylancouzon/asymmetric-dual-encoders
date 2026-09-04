"""M12 Tier 2: weighted RRF under the registered split-half. Runs only because Tier 1 failed.

Registered in m12/LEDGER.md before any M12 number. 24 candidates (4 k x 6 weight pairs) against a
bar that the two unfitted operators already missed by 0.0146-0.0192, so this row exists to be
falsified, not to rescue the milestone. It gets a split-half because it is the ONLY M12 row with a
fitting budget large enough that a dev-fitted winner would be reporting its own overfit: fit on the
even-hash half, score on the odd, then the reverse, and report the mean of the two HELD-OUT halves.

Runs are read from the caches Tier 1 persisted -- no GPU, no re-retrieval.

  PYTHONPATH=m12src:m7src M7_ENCODER=stella-400M-v5 .venv/bin/python m12src/run_tier2.py
"""
import hashlib
import json

import numpy as np

import dev_eval
import fusion
import qfusion
from _paths import REPO, WORK
from evalkit import per_query_ndcg, run_from_arrays

K_GRID = [2, 6, 11, 61]
W_GRID = [(1, 1), (2, 1), (3, 1), (4, 1), (1, 2), (1, 3)]     # (dense, bm25)
BAR_DELTA = 0.004


def half(qid):
    """Registered split: parity of the qid's sha256. Deterministic, corpus-independent, and not
    correlated with anything about the query -- an index-parity split would track whatever order
    the component was built in."""
    return int(hashlib.sha256(str(qid).encode()).hexdigest(), 16) % 2


def main():
    comps = [c for c in dev_eval.dev_components() if not c.startswith("heldout-")]
    dense, bm25, qrels = {}, {}, {}
    for c in comps:
        doc_ids, doc_texts, q_ids, q_texts, qr, _ = dev_eval.doc_vecs(c)
        z = np.load(WORK / "m12" / f"dense-{c}-d{fusion.DEPTH}.npz", allow_pickle=False)
        dense[c] = run_from_arrays(z["bi"].astype(np.int64), z["bs"], doc_ids, q_ids)
        bm25[c] = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts,
                                  cache_path=WORK / "fusionruns" / f"bm25-{c}-d{fusion.DEPTH}.npz")
        qrels[c] = qr
        print(f"  loaded {c}", flush=True)

    per_cache = {}

    def macros(fused):
        """(full macro, macro over the even-hash half, macro over the odd half). One per-query
        pass; the halves are subsets of it, so fitting and scoring cost the same as scoring."""
        per = {c: per_query_ndcg(fused[c], qrels[c]) for c in comps}
        per_cache["last"] = per
        full = [np.mean(list(per[c].values())) for c in comps]
        halves = []
        for h in (0, 1):
            halves.append([np.mean([v for q, v in per[c].items() if half(q) == h]) for c in comps])
        return float(np.mean(full)), float(np.mean(halves[0])), float(np.mean(halves[1]))

    c_full, c_even, c_odd = macros(
        {c: fusion.convex([dense[c], bm25[c]], w=0.8, floor_zero=True) for c in comps})
    convex_per = per_cache["last"]
    bars = {"full": c_full - BAR_DELTA, "even": c_even - BAR_DELTA, "odd": c_odd - BAR_DELTA}
    print(f"\n  convex0 w=0.8   full {c_full:.4f}   even {c_even:.4f}   odd {c_odd:.4f}")
    print(f"  bar (C-0.004)   full {bars['full']:.4f}   even {bars['even']:.4f}   odd {bars['odd']:.4f}\n")

    grid, per_by_cfg = [], {}
    for k in K_GRID:
        for w in W_GRID:
            f, e, o = macros({c: qfusion.rrf([dense[c], bm25[c]], k=k, weights=list(w))
                              for c in comps})
            grid.append({"k_q": k, "weights": list(w), "full": f, "even": e, "odd": o})
            per_by_cfg[(k, tuple(w))] = per_cache["last"]
            print(f"    k_q={k:<3} w={str(w):<7} full {f:.4f}  even {e:.4f}  odd {o:.4f}", flush=True)

    # Fit on one half, score on the other. The held-out score is the ONLY one the rule reads.
    fit_even = max(grid, key=lambda r: r["even"])
    fit_odd = max(grid, key=lambda r: r["odd"])
    held = (fit_even["odd"] + fit_odd["even"]) / 2
    bar_held = (bars["odd"] + bars["even"]) / 2
    best_full = max(grid, key=lambda r: r["full"])

    print(f"\n  fit on even -> k_q={fit_even['k_q']} w={fit_even['weights']}, held-out odd {fit_even['odd']:.4f}")
    print(f"  fit on odd  -> k_q={fit_odd['k_q']} w={fit_odd['weights']}, held-out even {fit_odd['even']:.4f}")
    print(f"  held-out macro {held:.4f}   vs bar {bar_held:.4f}   -> "
          f"{'PASS' if held >= bar_held else 'FAIL'}")
    print(f"  (dev-fitted best, for transparency only: k_q={best_full['k_q']} "
          f"w={best_full['weights']} full {best_full['full']:.4f})")

    # The registered Statistics section asks for a paired bootstrap on each operator-vs-convex0
    # difference; the first Tier-2 pass omitted it (Codex close-out review). Reported on the
    # DEV-FITTED best so it is comparable with the Tier-1 CIs, and flagged post-selection.
    from run_tier1 import bootstrap
    ci = bootstrap(per_by_cfg[(best_full["k_q"], tuple(best_full["weights"]))], convex_per, comps)
    print(f"  CI (dev-fitted best vs convex0, post-selection): "
          f"{best_full['full'] - c_full:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")

    t1 = json.loads((REPO / "m12" / "tier1.json").read_text())
    out = {"tier": 2, "why_run": "Tier 1 NO MATCH (registered rule)", "candidates": len(grid),
           "grid": grid, "convex0": {"full": c_full, "even": c_even, "odd": c_odd}, "bars": bars,
           "split": "sha256(qid) % 2", "fit_even": fit_even, "fit_odd": fit_odd,
           "held_out_macro": held, "held_out_bar": bar_held,
           "passes": bool(held >= bar_held),
           "dev_fitted_best": best_full,
           "held_out_comparator": (c_even + c_odd) / 2,
           "held_out_delta_vs_comparator": held - (c_even + c_odd) / 2,
           "ci95_dev_fitted_best_vs_convex0": list(ci),
           "_ci_note": "post-selection (24 candidates) and therefore optimistic; added after the "
                       "first pass, which omitted the registered bootstrap",
           "tier1_rows": {r["label"]: r["macro"] for r in t1["rows"]}}
    out["m12_result"] = "MATCH" if out["passes"] else "NO MATCH"
    (REPO / "m12" / "tier2.json").write_text(json.dumps(out, indent=2))
    print(f"\n  M12 RESULT: {out['m12_result']}\n  wrote m12/tier2.json")


if __name__ == "__main__":
    main()
