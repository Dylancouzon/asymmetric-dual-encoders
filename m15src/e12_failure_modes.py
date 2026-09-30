"""E12 (exploratory): two ways a lookup table loses, per query.

For every test query of the six public sets with gated Stella vectors: the Stella-minus-Zero and
Nano-minus-Zero nDCG@10 gaps (committed M20 rows), the query's fertility (subwords per word), and
its order dependence (Stella's nDCG@10 on the query minus on the same words shuffled, seed
20260930). Reports Spearman of each gap with each factor, and mean gaps by tercile of each factor.
"""
import numpy as np
from scipy.stats import spearmanr

from common import REPO, load_public, m20_rows, receipt, utc_now, write_result
import e6_router as R6
import encoders15 as E
import vectors15 as V
from e4_prefix import conditions

SETS = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")
OUT = REPO / "results" / "m15_e12_failure_modes.json"


def terciles(x, y):
    cut = np.quantile(x, [1 / 3, 2 / 3])
    groups = np.digitize(x, cut)
    # Ties (order dependence is often exactly 0) can empty a tercile; report it as None.
    return [float(y[groups == g].mean()) if (groups == g).any() else None for g in range(3)]


def main():
    started = utc_now()
    tok, _ = R6.zero_tokenizer()
    stella = E.make("stella-query")
    pooled = {k: [] for k in ("gap_stella", "gap_nano", "fert", "order", "ds", "stella_full")}
    per = {}
    for ds in SETS:
        data = load_public(ds)
        dv, _ = V.doc_vectors(ds, data)
        ids = data["q_ids"]
        rows = {s: m20_rows(ds, s)[0] for s in ("zero-dense", "nano-dense", "stella-query")}
        z, n, s = (np.array([rows[k][q] for q in ids]) for k in ("zero-dense", "nano-dense",
                                                                  "stella-query"))
        shuffled = conditions(data["q_texts"], np.random.default_rng(20260930))["shuffle"]
        # Full and shuffled scored by the same local encoder and vectors (Sol review): the
        # difference is word-order sensitivity, not reproduction noise against M20.
        full_l = V.ndcg10(V.exact_run(stella.encode(data["q_texts"]), dv, data["doc_ids"], ids),
                          data["qrels"], ids)
        shuf_l = V.ndcg10(V.exact_run(stella.encode(shuffled), dv, data["doc_ids"], ids),
                          data["qrels"], ids)
        order = np.array([full_l[q] - shuf_l[q] for q in ids])
        fert = R6.fertility(tok, data["q_texts"])
        per[ds] = {"n": len(ids), "mean_gap_stella_zero": float((s - z).mean()),
                   "mean_order_dependence": float(order.mean()), "mean_fertility": float(fert.mean()),
                   "rho_gap_fert": float(spearmanr(s - z, fert)[0]),
                   "rho_gap_order": float(spearmanr(s - z, order)[0])}
        for k, v in (("gap_stella", s - z), ("gap_nano", n - z), ("fert", fert), ("order", order),
                     ("ds", [ds] * len(ids)), ("stella_full", s)):
            pooled[k].extend(v)
        print(ds, per[ds], flush=True)
    g_s, g_n = np.array(pooled["gap_stella"]), np.array(pooled["gap_nano"])
    fert, order = np.array(pooled["fert"]), np.array(pooled["order"])
    summary = {"n_queries": int(len(g_s)),
               "rho": {"stella_gap_vs_fertility": float(spearmanr(g_s, fert)[0]),
                       "stella_gap_vs_order": float(spearmanr(g_s, order)[0]),
                       "nano_gap_vs_fertility": float(spearmanr(g_n, fert)[0]),
                       "nano_gap_vs_order": float(spearmanr(g_n, order)[0]),
                       "fertility_vs_order": float(spearmanr(fert, order)[0])},
               "stella_gap_by_fertility_tercile": terciles(fert, g_s),
               "nano_gap_by_fertility_tercile": terciles(fert, g_n),
               "stella_gap_by_order_tercile": terciles(order, g_s),
               "nano_gap_by_order_tercile": terciles(order, g_n),
               "share_order_dependent": float((order > 0).mean()),
               "stella_gap_order_dependent": float(g_s[order > 0].mean()),
                              "stella_gap_order_free": float(g_s[order <= 0].mean())}
    # Both the gap and order dependence grow with Stella's own full-query score (a query Stella
    # fails cannot lose to shuffling). Control for it: Spearman within quartile bins of that score
    # (queries with a Stella score of 0 form their own bin), averaged with query weights.
    sf = np.array(pooled["stella_full"])
    bins = np.where(sf == 0, -1, np.digitize(sf, np.quantile(sf[sf > 0], [.25, .5, .75])))
    strat = {}
    for name, x, y in (("stella_gap_vs_order", order, g_s), ("stella_gap_vs_fertility", fert, g_s),
                       ("nano_gap_vs_order", order, g_n), ("nano_gap_vs_fertility", fert, g_n)):
        vals, wts = [], []
        for b in np.unique(bins):
            m = bins == b
            if m.sum() > 20 and np.std(x[m]) > 0 and np.std(y[m]) > 0:
                vals.append(spearmanr(x[m], y[m])[0])
                wts.append(m.sum())
        strat[name] = float(np.average(vals, weights=wts))
    summary["rho_within_stella_score_bins"] = strat
    np.savez(REPO / "work" / "m15" / "e12_per_query.npz", gap_stella=g_s, gap_nano=g_n, fert=fert,
             order=order, stella_full=sf, ds=np.array(pooled["ds"]))
    write_result(OUT, {"status": "COMPLETE", "measurement": "E12 (exploratory)",
                       "scope": "exploratory; six public sets; per-query associations, not causes",
                       "summary": summary, "per_dataset": per,
                       "receipt": receipt(__file__, [], started, ("onnxruntime", "scipy"))})


if __name__ == "__main__":
    main()
