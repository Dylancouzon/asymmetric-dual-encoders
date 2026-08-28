"""Which query-side transforms are ABSORBABLE into the lookup table? Settle it numerically.

research/m7-research-2026-08-26.md claims query-side centering / top-PC removal is "genuine new
capacity" because the centering offset "is not expressible as a mean of row vectors". That
argument assumes you may only edit the always-present [CLS]/[SEP] rows. You may edit EVERY row,
and for a constant mu, mean_t(W[t] - mu) = mean_t(W[t]) - mu exactly. So it is absorbable.

This checks each candidate transform against an explicitly reconstructed table, on ragged
token multisets with repeats, to machine precision. If a transform is absorbable, the lever is
not new capacity -- it is a re-parameterisation, and can only help by being a better prior or
initialisation. That is a much weaker claim than the research note makes, and it re-ranks the
plan: n-grams become the only structurally new lever on the list.

Writes results/m7_absorb_check.json. Pure algebra: no model, no data, no eval.
"""
import json

import numpy as np

from _paths import REPO

RNG = np.random.default_rng(0)
V, D, NQ = 500, 64, 200


def norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def encode(W, queries, weights=None):
    """The architecture: gather rows -> (weighted) mean over the token MULTISET -> L2 norm."""
    out = np.zeros((len(queries), W.shape[1]))
    for i, T in enumerate(queries):
        w = np.ones(len(T)) if weights is None else weights[T]
        out[i] = (w[:, None] * W[T]).sum(0) / w.sum()
    return norm(out)


def main():
    W = RNG.normal(size=(V, D))
    # ragged multisets WITH repeats and a shared always-present prefix token, like real queries
    queries = [np.concatenate([[0], RNG.integers(1, V, size=RNG.integers(2, 12))])
               for _ in range(NQ)]
    mu = RNG.normal(size=D)                     # a centering offset
    A = RNG.normal(size=(D, D))                 # a whitening / linear map
    v = norm(RNG.normal(size=D))                # a top principal direction
    P = np.eye(D) - np.outer(v, v)              # top-PC removal
    c = np.abs(RNG.normal(size=V)) + 0.1        # per-token scalar weights (IDF / SIF)

    checks = {}

    def rec(name, applied, table, weights=None, claim=None):
        got = encode(table, queries, weights)
        err = float(np.abs(applied - got).max())
        checks[name] = {"max_abs_diff": err, "absorbable": bool(err < 1e-10),
                        "reconstruction": claim}
        print(f"  {name:44s} max|diff| {err:.2e}  "
              f"{'ABSORBABLE' if err < 1e-10 else 'NOT absorbable'}   {claim}")

    print("transforms applied to the query vector, vs a table rebuilt to reproduce them:")
    # 1 centering
    base = np.stack([W[T].mean(0) for T in queries])
    rec("centering: normalize(mean - mu)", norm(base - mu), W - mu,
        claim="W' = W - mu (subtract mu from EVERY row)")
    # 2 pure linear map / whitening
    rec("whitening: normalize(A @ mean)", norm(base @ A.T), W @ A.T,
        claim="W' = W A^T")
    # 3 top-PC removal after centering (the full SIF/uSIF post-processing)
    rec("SIF post-proc: normalize(P @ (mean - mu))", norm((base - mu) @ P.T), (W - mu) @ P.T,
        claim="W' = (W - mu) P^T")
    # 4 per-token scalar weighting (IDF, SIF a/(a+p(w)))
    wmean = np.stack([(c[T][:, None] * W[T]).sum(0) / c[T].sum() for T in queries])
    rec("per-token weights: normalize(sum c_t W_t / sum c_t)", norm(wmean), c[:, None] * W,
        claim="W' = c_t * W_t (scale each row by its own weight)")
    # 5 the whole SIF recipe at once
    full = norm(np.stack([(c[T][:, None] * (W[T] - mu)).sum(0) / c[T].sum() for T in queries]) @ P.T)
    rec("full SIF (weights + centering + PC removal)", full, c[:, None] * (W - mu) @ P.T,
        claim="W' = c_t (W_t - mu) P^T")

    print("\ntransforms that depend on the query's COMPOSITION, not just token identity:")
    # 6 count saturation -- dedupe the multiset. Depends on multiplicity, not on row values.
    dedup = norm(np.stack([W[np.unique(T)].mean(0) for T in queries]))
    plain = norm(base)
    d = float(np.abs(dedup - plain).max())
    # try the best possible per-token rescaling: none can fix it, because the SAME token must
    # get the SAME row whether or not it repeats in a given query
    checks["count saturation (unique-tokens-only)"] = {
        "max_abs_diff_vs_plain_mean": d, "absorbable": False,
        "reconstruction": "IMPOSSIBLE: the correction depends on each query's multiplicity "
                          "vector, and a row is shared across all queries"}
    print(f"  {'count saturation (unique tokens only)':44s} differs from plain mean by "
          f"{d:.3e}  NOT absorbable (query-dependent)")
    # 7 length-dependent scaling: absorbable? normalize() kills any scalar function of |T|.
    lens = np.array([len(T) for T in queries])
    sqrtlen = norm(np.stack([W[T].sum(0) / np.sqrt(len(T)) for T in queries]))
    d2 = float(np.abs(sqrtlen - plain).max())
    checks["length scaling (1/sqrt|T| instead of 1/|T|)"] = {
        "max_abs_diff_vs_plain_mean": d2, "absorbable": True,
        "reconstruction": "VACUOUS: any positive scalar function of |T| is removed by the final "
                          "L2 normalization, so it is a no-op, not a lever"}
    print(f"  {'length scaling 1/sqrt|T|':44s} differs from plain mean by "
          f"{d2:.3e}  NO-OP (killed by L2 normalize)")
    # 8 n-gram rows: new features -> new capacity, trivially not expressible by unigram rows
    checks["n-gram / phrase rows"] = {
        "absorbable": False,
        "reconstruction": "NOT absorbable: two queries with the same token multiset in a "
                          "different order get IDENTICAL unigram-bag vectors, so no choice of "
                          "unigram rows can separate them. An n-gram row can."}
    print(f"  {'n-gram / phrase rows':44s} NOT absorbable (adds features unigrams cannot express)")

    # 9 DOC-side linear map. m7/LEDGER.md listed this as absorbable with no check behind it, and
    # the claim is only half true. Ranking by q.(M d) equals ranking by (M^T q).d, so absorbing M
    # into the rows is exact -- but ONLY if the mapped document is not renormalized. This system
    # retrieves on L2-normalized document vectors, so the served score is q.(M d / |M d|), and the
    # per-document factor 1/|M d| varies with d and cannot be moved to the query side at all.
    rng2 = np.random.default_rng(7)
    docs = norm(rng2.normal(size=(300, D)))
    M = rng2.normal(size=(D, D)) / np.sqrt(D)
    # docs are ROWS here, so a mapped document is `docs @ M.T` and the score is
    # plain @ (docs @ M.T).T = plain @ M @ docs.T -- the query-side equivalent is `plain @ M`,
    # NOT `plain @ M.T`. Writing the transpose the wrong way round made this check report rank
    # agreement 0.000 for a case the algebra says is exact, which is the whole reason it is a
    # numerical check and not a paragraph.
    absorbed = norm(plain @ M)                        # query side carries the map, docs untouched
    s_absorbed = absorbed @ docs.T
    s_unnorm = plain @ (docs @ M.T).T                 # doc side carries M, NO renormalization
    s_renorm = plain @ norm(docs @ M.T).T             # doc side carries M, WITH renormalization

    def rank_agree(a, b):
        return float(np.mean([np.array_equal(np.argsort(-x), np.argsort(-y))
                              for x, y in zip(a, b)]))

    checks["doc-side linear map, documents NOT renormalized"] = {
        "rank_agreement_with_query_side_absorption": rank_agree(s_unnorm, s_absorbed),
        "absorbable": rank_agree(s_unnorm, s_absorbed) == 1.0,
        "reconstruction": "W' = W M. q.(M d) = (M^T q).d exactly, and the query-side L2 "
                          "normalize is a per-query positive scalar that cannot change a ranking."}
    ag = rank_agree(s_renorm, s_absorbed)
    checks["doc-side linear map, documents RENORMALIZED (what this system does)"] = {
        "rank_agreement_with_query_side_absorption": ag,
        "absorbable": ag == 1.0,
        "reconstruction": "NOT absorbable: the served score is q.(M d / |M d|) and the factor "
                          "1/|M d| is per-DOCUMENT, so no change to the shared table reproduces "
                          "it. Retrieval here runs on L2-normalized document vectors, so this is "
                          "the case that applies. It is still not a lever we can pull -- changing "
                          "the document map means re-encoding the corpus with a different "
                          "teacher -- but 'absorbable' was the wrong reason to dismiss it."}
    print(f"  {'doc-side map, docs NOT renormalized':44s} rank agreement "
          f"{rank_agree(s_unnorm, s_absorbed):.3f}  absorbable")
    print(f"  {'doc-side map, docs RENORMALIZED':44s} rank agreement {ag:.3f}  NOT absorbable")

    out = {"_note": "Absorbability of query-side transforms into a freely-parameterised "
                    "vocab x dim table under normalize(weighted mean of rows). ABSORBABLE means "
                    "the transform adds NO capacity a trained table lacks -- it can still help "
                    "as a prior/initialisation, but it cannot raise the architecture's ceiling. "
                    "This CORRECTS research/m7-research-2026-08-26.md, which classed query-side "
                    "centering and top-PC removal as 'genuine new capacity'. They are not: "
                    "mean_t(W_t - mu) = mean_t(W_t) - mu exactly, for every token multiset.",
           "setup": {"vocab": V, "dim": D, "n_queries": NQ,
                     "queries": "ragged multisets with repeats and a shared prefix token"},
           "checks": checks,
           "consequence": "Of the levers in the plan, only n-gram/phrase rows and "
                          "multiplicity-dependent pooling (count saturation) add capacity. "
                          "Centering, whitening, top-PC removal, IDF and SIF weighting are all "
                          "re-parameterisations -- and per-token weighting is ALREADY in the "
                          "architecture (QueryTable.learned_weights), so p1-objB could in "
                          "principle have learned the SIF weighting itself."}
    (REPO / "results" / "m7_absorb_check.json").write_text(json.dumps(out, indent=1))
    print("\nwrote results/m7_absorb_check.json")


if __name__ == "__main__":
    main()
