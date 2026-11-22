"""E4 (with E7 folded in): prefixes, incomplete queries and word order (m15/MEASUREMENTS.md, E4).

Exact search over Stella document vectors that passed the reproduction gate; BM25 is the
registered bm25s configuration. Queries run on ONNX Runtime CPU (Zero on NumPy).
"""
import math

import numpy as np

from common import REPO, SEED, load_public, receipt, utc_now, write_result
import encoders15 as E
import vectors15 as V

OUT = REPO / "results" / "m15_e4_prefix.json"
DATASETS = ("scifact", "nfcorpus", "fiqa")
DENSE = ("stella-query", "zero", "nano")
WORD_K = (1, 2, 3, 5)
FRACS = (25, 50, 75)


def conditions(texts, rng):
    """Condition name -> list of cut texts. Queries shorter than a cut keep their full text."""
    words = [t.split() for t in texts]
    out = {"full": list(texts), "shuffle": []}
    for w, t in zip(words, texts):
        perm = rng.permutation(len(w))
        out["shuffle"].append(" ".join(w[i] for i in perm) if len(w) > 1 else t)
    lengths = {f"prefix_k{k}": [min(k, len(w)) for w in words] for k in WORD_K}
    lengths.update({f"prefix_f{f}": [max(1, math.ceil(len(w) * f / 100)) for w in words]
                    for f in FRACS})
    for name, ns in lengths.items():
        out[name] = [" ".join(w[:n]) if n < len(w) else t for w, n, t in zip(words, ns, texts)]
        rand = []
        for w, n, t in zip(words, ns, texts):
            if n >= len(w):
                rand.append(t)
            else:
                keep = np.sort(rng.choice(len(w), n, replace=False))
                rand.append(" ".join(w[i] for i in keep))
        out[name.replace("prefix", "random")] = rand
    out["char_f50"] = [t[:max(1, math.ceil(len(t) / 2))] for t in texts]
    return out


def retention_ci(cond, full, draws=10_000, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(full), size=(draws, len(full)))
    ratio = cond[idx].mean(1) / np.maximum(full[idx].mean(1), 1e-12)
    return [float(np.quantile(ratio, .025)), float(np.quantile(ratio, .975))]


def main():
    started = utc_now()
    enc = {n: E.make(n) for n in DENSE}
    rng = np.random.default_rng(SEED)
    results, gates, inputs = {}, {}, []
    for ds in DATASETS:
        data = load_public(ds)
        dv, prov = V.doc_vectors(ds, data)
        ok, gates[ds] = V.gate(ds, data, dv, enc)
        if not ok:
            raise SystemExit(f"E4 STOP: reproduction gate failed on {ds}")
        inputs.append({ds: {"pins": data["pins"], "doc_vectors": prov}})
        conds = conditions(data["q_texts"], rng)
        per = {}
        for system in DENSE + ("bm25",):
            scores, full_vecs = {}, None
            for cname, texts in conds.items():
                if system == "bm25":
                    run = V.bm25(data, texts)
                else:
                    qv = enc[system].encode(texts)
                    if cname == "full":
                        full_vecs = qv
                    elif cname == "shuffle":
                        scores["_shuffle_cos"] = float((qv * full_vecs).sum(1).mean())
                    run = V.exact_run(qv, dv, data["doc_ids"], data["q_ids"])
                pq = V.ndcg10(run, data["qrels"], data["q_ids"])
                scores[cname] = np.array([pq[q] for q in data["q_ids"]])
            full = scores["full"]
            per[system] = {c: {"mean_ndcg10": float(v.mean()),
                               "retention": float(v.mean() / full.mean()),
                               "retention_ci95": retention_ci(v, full)}
                           for c, v in scores.items() if not c.startswith("_")}
            if "_shuffle_cos" in scores:
                per[system]["shuffle"]["mean_cos_to_full_query"] = scores["_shuffle_cos"]
            print(ds, system, {c: round(v["retention"], 3) for c, v in per[system].items()},
                  flush=True)
        words = np.array([len(t.split()) for t in data["q_texts"]])
        results[ds] = {"n_queries": len(data["q_ids"]), "mean_words": float(words.mean()),
                       "share_changed": {c: float(np.mean([a != b for a, b in
                                                          zip(conds[c], conds["full"])]))
                                         for c in conds if c != "full"},
                       "systems": per}
    write_result(OUT, {"status": "COMPLETE", "measurement": "E4 (E7 folded in)",
                       "scope": "descriptive; cut test queries, not real typed partial queries",
                       "conditions": list(conds), "reproduction_gate": gates,
                       "per_dataset": results,
                       "receipt": receipt(__file__, inputs, started,
                                          ("onnxruntime", "torch", "bm25s"))})


if __name__ == "__main__":
    main()
