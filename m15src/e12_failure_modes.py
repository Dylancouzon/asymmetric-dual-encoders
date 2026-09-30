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
    return [float(y[groups == g].mean()) for g in range(3)]


def main():
    started = utc_now()
    tok, _ = R6.zero_tokenizer()
    stella = E.make("stella-query")
    pooled = {k: [] for k in ("gap_stella", "gap_nano", "fert", "order", "ds")}
    per = {}
    for ds in SETS:
        data = load_public(ds)
        dv, _ = V.doc_vectors(ds, data)
        ids = data["q_ids"]
        rows = {s: m20_rows(ds, s)[0] for s in ("zero-dense", "nano-dense", "stella-query")}
        z, n, s = (np.array([rows[k][q] for q in ids]) for k in ("zero-dense", "nano-dense",
                                                                  "stella-query"))
        shuffled = conditions(data["q_texts"], np.random.default_rng(20260930))["shuffle"]
        pq = V.ndcg10(V.exact_run(stella.encode(shuffled), dv, data["doc_ids"], ids),
                      data["qrels"], ids)
        order = s - np.array([pq[q] for q in ids])
        fert = R6.fertility(tok, data["q_texts"])
        per[ds] = {"n": len(ids), "mean_gap_stella_zero": float((s - z).mean()),
                   "mean_order_dependence": float(order.mean()), "mean_fertility": float(fert.mean()),
                   "rho_gap_fert": float(spearmanr(s - z, fert)[0]),
                   "rho_gap_order": float(spearmanr(s - z, order)[0])}
        for k, v in (("gap_stella", s - z), ("gap_nano", n - z), ("fert", fert), ("order", order),
                     ("ds", [ds] * len(ids))):
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
               "share_order_dependent": float((order > 0).mean()),
               "stella_gap_order_dependent": float(g_s[order > 0].mean()),
               "stella_gap_order_free": float(g_s[order <= 0].mean())}
    write_result(OUT, {"status": "COMPLETE", "measurement": "E12 (exploratory)",
                       "scope": "exploratory; six public sets; per-query associations, not causes",
                       "summary": summary, "per_dataset": per,
                       "receipt": receipt(__file__, [], started, ("onnxruntime", "scipy"))})


if __name__ == "__main__":
    main()
