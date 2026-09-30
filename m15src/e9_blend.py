"""E9: blended query vectors on one index (m15/MEASUREMENTS.md, E9). Exact search, gated vectors.

Also writes per-query Zero/Nano rows and Zero's top-1/top-10 cosines to work/m15/e9_cache/ for E10.
"""
import json

import numpy as np

from common import (REPO, SEED, load_public, m20_rows, receipt, sha_file, utc_now,
                    write_result)
import encoders15 as E
import vectors15 as V

FIT = ("cqadup-physics", "cqadup-programmers")
EVAL = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")
CLEAN4 = ("nfcorpus", "scidocs", "scifact", "trec-covid")
WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
OUT = REPO / "results" / "m15_e9_blend.json"
FROZEN = REPO / "results" / "m15_e9_frozen.json"
CACHE = REPO / "work" / "m15" / "e9_cache"


def blend(a, b, w):
    q = (1 - w) * a + w * b
    return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)


def dataset_scores(ds, enc):
    """Per-query nDCG@10 for every blend weight, plus Zero's top-k cosines, for one dataset."""
    data = load_public(ds)
    dv, prov = V.doc_vectors(ds, data)
    q = {n: enc[n].encode(data["q_texts"]) for n in ("nano", "zero", "stella-query")}
    ids, qrels = data["q_ids"], data["qrels"]

    def score(qv):
        pq = V.ndcg10(V.exact_run(qv, dv, data["doc_ids"], ids), qrels, ids)
        return np.array([pq[i] for i in ids])

    out = {"nano": {str(w): score(blend(q["nano"], q["zero"], w)) for w in WEIGHTS},
           "stella": {str(w): score(blend(q["stella-query"], q["zero"], w)) for w in WEIGHTS}}
    from evalkit import topk_arrays
    _, bs = topk_arrays(q["zero"], dv, k=10, chunk=250_000, device="cpu")
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"{ds}.npz", q_ids=np.array(ids), zero=score(q["zero"]),
             nano=out["nano"]["0.0"], top1=bs[:, 0], top10=bs[:, 9])
    return ids, out, prov


def paired_ci(diff, draws=10_000):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(draws, len(diff)))
    m = diff[idx].mean(1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def main():
    started = utc_now()
    enc = {n: E.make(n) for n in ("nano", "zero", "stella-query")}
    inputs = []
    # 1. Fit on the two dev forums and freeze.
    fit = {}
    for ds in FIT:
        _, fit[ds], prov = dataset_scores(ds, enc)
        inputs.append({ds: prov})
    frozen = {}
    for base in ("nano", "stella"):
        curve = {w: float(np.mean([fit[d][base][w].mean() for d in FIT])) for w in map(str, WEIGHTS)}
        best = max(curve.values())
        frozen[base] = {"weight": float(min(float(w) for w, v in curve.items() if v == best)),
                        "fit_curve": curve}
    write_result(FROZEN, {"status": "FROZEN", "measurement": "E9 weights", "frozen": frozen,
                          "fit": list(FIT), "receipt": receipt(__file__, inputs, started)})
    inputs.append({"frozen": str(FROZEN.relative_to(REPO)), "sha256": sha_file(FROZEN)})
    # 2. Evaluate on the six public sets.
    per, diffs = {}, {"nano": {}, "stella": {}}
    for ds in EVAL:
        ids, sc, prov = dataset_scores(ds, enc)
        inputs.append({ds: prov})
        fused, _ = m20_rows(ds, "nano+bm25 dbsf@100")
        per[ds] = {"nano+bm25_dbsf@100_m20": float(np.mean([fused[i] for i in ids]))}
        for base in ("nano", "stella"):
            w = str(frozen[base]["weight"])
            d = sc[base][w] - sc[base]["0.0"]
            diffs[base][ds] = d
            per[ds][base] = {"alone": float(sc[base]["0.0"].mean()),
                             "blend": float(sc[base][w].mean()),
                             "blend_minus_alone": float(d.mean()),
                             "ci95": paired_ci(d),
                             "posthoc_curve": {k: float(v.mean()) for k, v in sc[base].items()}}
    macro = {}
    for base in ("nano", "stella"):
        for name, keys in (("all6", EVAL), ("clean4", CLEAN4)):
            macro[f"{base}_{name}"] = {
                "alone": float(np.mean([per[d][base]["alone"] for d in keys])),
                "blend": float(np.mean([per[d][base]["blend"] for d in keys])),
                "blend_minus_alone": float(np.mean([per[d][base]["blend_minus_alone"]
                                                    for d in keys]))}
    write_result(OUT, {"status": "COMPLETE", "measurement": "E9",
                       "scope": "descriptive; exact search; weights frozen on the M7 dev forums",
                       "weights_grid": list(WEIGHTS), "frozen": frozen, "per_dataset": per,
                       "macro": macro, "receipt": receipt(__file__, inputs, started,
                                                          ("onnxruntime", "torch"))})


if __name__ == "__main__":
    main()
