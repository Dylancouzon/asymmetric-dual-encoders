"""E12b (exploratory): is E12's order dependence just fragility?

For each test query of the six sets: the Stella query vector q, its shuffled-words vector q_s, and
a random perturbation q_n with exactly the same cosine to q as q_s (seed 20260930). fragility =
Stella nDCG@10(q) - nDCG@10(q_n); order = nDCG@10(q) - nDCG@10(q_s). If the table's loss tracks
order dependence only because fragile queries lose under any perturbation, fragility should
predict the Stella-minus-Zero gap as well as order does.
"""
import numpy as np
from scipy.stats import spearmanr

from common import REPO, SEED, load_public, m20_rows, receipt, utc_now, write_result
import encoders15 as E
import vectors15 as V
from e4_prefix import conditions

SETS = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")
OUT = REPO / "results" / "m15_e12b_fragility.json"


def matched_noise(q, target_cos, rng):
    """Unit vectors at exactly target_cos to each row of q, in a random orthogonal direction."""
    r = rng.standard_normal(q.shape).astype(np.float32)
    r -= (r * q).sum(1, keepdims=True) * q
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    c = np.clip(target_cos, -1, 1)[:, None]
    return (c * q + np.sqrt(1 - c ** 2) * r).astype(np.float32)


def main():
    started = utc_now()
    stella = E.make("stella-query")
    rng = np.random.default_rng(SEED)
    pooled = {k: [] for k in ("gap", "order", "frag", "full")}
    for ds in SETS:
        data = load_public(ds)
        dv, _ = V.doc_vectors(ds, data)
        ids = data["q_ids"]
        z = m20_rows(ds, "zero-dense")[0]
        q = stella.encode(data["q_texts"])
        qs = stella.encode(conditions(data["q_texts"], np.random.default_rng(SEED))["shuffle"])
        qn = matched_noise(q, (q * qs).sum(1), rng)
        score = lambda v: V.ndcg10(V.exact_run(v, dv, data["doc_ids"], ids), data["qrels"], ids)
        full, shuf, noise = score(q), score(qs), score(qn)
        pooled["gap"] += [full[i] - z[i] for i in ids]
        pooled["order"] += [full[i] - shuf[i] for i in ids]
        pooled["frag"] += [full[i] - noise[i] for i in ids]
        pooled["full"] += [full[i] for i in ids]
        print(ds, "done", flush=True)
    g, o, f, s = (np.array(pooled[k]) for k in ("gap", "order", "frag", "full"))
    pos = s > 0
    # Partial association of gap with order, controlling for fragility: Spearman of the residual
    # ranks after regressing ranks of each on the ranks of fragility.
    from scipy.stats import rankdata
    rk = lambda x: rankdata(x) / len(x)
    def resid(y, x):
        b = np.polyfit(x, y, 1)
        return y - np.polyval(b, x)
    partial = float(np.corrcoef(resid(rk(g[pos]), rk(f[pos])), resid(rk(o[pos]), rk(f[pos])))[0, 1])
    summary = {"n_queries": int(len(g)), "n_stella_positive": int(pos.sum()),
               "rho_gap_order": float(spearmanr(g[pos], o[pos])[0]),
               "rho_gap_fragility": float(spearmanr(g[pos], f[pos])[0]),
               "rho_order_fragility": float(spearmanr(o[pos], f[pos])[0]),
               "partial_rho_gap_order_given_fragility": partial,
               "mean_order_drop": float(o[pos].mean()), "mean_noise_drop": float(f[pos].mean())}
    write_result(OUT, {"status": "COMPLETE", "measurement": "E12b (exploratory)",
                       "scope": "exploratory; queries where Stella scores above 0; associations",
                       "summary": summary,
                       "receipt": receipt(__file__, [], started, ("onnxruntime", "scipy"))})
    print(summary, flush=True)


if __name__ == "__main__":
    main()
