"""E10: routers from signals Zero already has (m15/MEASUREMENTS.md, E10).

f1 fertility, f2 pooled-vector norm before normalization, f3 word count: 12 E5 datasets from
committed M20 rows. f4 Zero's top-1 minus top-10 cosine: the six E9 sets, from work/m15/e9_cache.
Direction and thresholds are fitted on the two M7 dev forums and frozen before evaluation.
"""
import json

import numpy as np
from scipy.stats import spearmanr

from common import (EVAL12, FIT_FORUMS, REPO, SEED, receipt, sha_file, utc_now, write_result)
import e6_router as R6
import encoders15 as E

BUDGETS = R6.BUDGETS
BUDGETS_STR = [str(b) for b in BUDGETS]
SIX = ("scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid")
CACHE = REPO / "work" / "m15" / "e9_cache"
OUT = REPO / "results" / "m15_e10_router.json"
FROZEN = REPO / "results" / "m15_e10_frozen.json"


def pooled_norm(zero, texts):
    """L2 norm of Zero's sqrt-weighted mean of rows, before normalization (zero_encoder.py)."""
    m = zero.model
    out = []
    for enc in m.tokenizer.encode_batch(list(texts)):
        uniq, counts = np.unique(np.asarray(enc.ids, dtype=np.int64), return_counts=True)
        w = np.sqrt(counts).astype(np.float32)
        out.append(float(np.linalg.norm((m.rows[uniq] * w[:, None]).sum(0) / max(w.sum(), 1e-6))))
    return np.array(out)


def text_features(tok, zero, texts):
    return {"f1_fertility": R6.fertility(tok, texts), "f2_pooled_norm": pooled_norm(zero, texts),
            "f3_words": np.array([len(t.split()) for t in texts], dtype=float)}


def margin(ds, qids=None, blend_weight=None):
    """Zero/Nano rows and f4 in sorted-qid order (R6.per_query's order); checked against qids.
    With blend_weight, also the E9 blend's per-query rows at that weight (escalation target)."""
    c = np.load(CACHE / f"{ds}.npz")
    order = np.argsort(c["q_ids"])
    if qids is not None and list(c["q_ids"][order]) != list(qids):
        raise SystemExit(f"{ds}: margin cache does not align with the M20 query ids")
    out = (c["zero"][order], c["nano"][order], {"f4_margin": (c["top1"] - c["top10"])[order],
                                                 "f5_agreement": c["agree"][order]})
    if blend_weight is not None:
        out += (c[f"blend_{blend_weight}"][order],)
    return out


def main():
    started = utc_now()
    tok, _ = R6.zero_tokenizer()
    zero = E.ZeroEncoder()
    # 1. Fit on the dev forums: direction by Spearman with (nano - zero), threshold by quantile.
    fz, fn, feats = [], [], {}
    for ds in FIT_FORUMS:
        texts, z, n, _ = R6.per_query(ds)
        f = text_features(tok, zero, texts)
        cz, cn, f4 = margin(ds, sorted(R6.m20_rows(ds, "zero-dense")[0]))
        if not (np.allclose(cz, z, atol=1e-6) and np.allclose(cn, n, atol=1e-6)):
            raise SystemExit(f"{ds}: local exact rows differ from the committed M20 rows")
        f.update(f4)
        fz.append(z)
        fn.append(n)
        for k, v in f.items():
            feats.setdefault(k, []).append(v)
    fz, fn = np.concatenate(fz), np.concatenate(fn)
    frozen = {}
    for k, parts in feats.items():
        v = np.concatenate(parts)
        rho = float(spearmanr(v, fn - fz)[0])
        sign = 1.0 if rho >= 0 else -1.0
        frozen[k] = {"spearman_with_gain": rho, "sign": sign,
                     "thresholds": {str(b): float(np.quantile(sign * v, 1 - b)) for b in BUDGETS}}
    write_result(FROZEN, {"status": "FROZEN", "measurement": "E10 routers", "features": frozen,
                          "fit_forums": list(FIT_FORUMS),
                          "receipt": receipt(__file__, [], started)})
    inputs = [{"frozen": str(FROZEN.relative_to(REPO)), "sha256": sha_file(FROZEN)}]
    rng = np.random.default_rng(SEED)
    # 2. Evaluate.
    table = {}
    for ds in EVAL12:
        texts, z, n, meta = R6.per_query(ds)
        inputs += meta
        f = text_features(tok, zero, texts)
        table[ds] = {k: {b: R6.evaluate(frozen[k]["sign"] * v, z, n, t, rng)
                         for b, t in frozen[k]["thresholds"].items()} for k, v in f.items()}
    # Escalation target: plain Nano, and (Astra 2026-09-30) the E9 blend at its frozen weight.
    a_star = str(json.loads((REPO / "results" / "m15_e9_frozen.json").read_text())
                 ["frozen"]["nano"]["weight"])
    for ds in SIX:
        z, n, f4, bl = margin(ds, blend_weight=a_star)
        for k in ("f4_margin", "f5_agreement"):
            table[ds][k] = {b: R6.evaluate(frozen[k]["sign"] * f4[k], z, n, t, rng)
                            for b, t in frozen[k]["thresholds"].items()}
        k = "f4_margin"
        table[ds]["f4_margin_to_blend"] = {b: R6.evaluate(frozen[k]["sign"] * f4[k], z, bl, t, rng)
                                           for b, t in frozen[k]["thresholds"].items()}
    macro = {}
    for k in list(frozen) + ["f4_margin_to_blend"]:
        keys = SIX if k.startswith(("f4", "f5")) else EVAL12
        macro[k] = {"datasets": list(keys), **{b: {m: float(np.mean([table[d][k][b][m] for d in keys]))
                    for m in ("nano_fraction", "router", "random_same_fraction",
                              "oracle_same_fraction", "router_minus_random")}
                    for b in BUDGETS_STR}}
    # System cost (registered): per-query encode from E1 (medium bucket p50) and search p50 from
    # E2 (msmarco1m, unquantized, hnsw_ef 128). An f4 router always pays Zero's encode and search,
    # plus Nano's encode and a second search on the routed fraction.
    e1 = json.loads((REPO / "results" / "m15_e1_latency.json").read_text())["summary"]
    enc_ms = {n: e1[n]["p50_ms_median_of_trials"]["medium"] for n in ("zero", "nano")}
    e2p = REPO / "results" / "m15_e2_ann_msmarco1m.json"
    search = None
    if e2p.exists():
        rows = json.loads(e2p.read_text())["rows"]
        search = {r["encoder"]: r["search_p50_ms"] for r in rows
                  if r["quant"] == "none" and r["hnsw_ef"] == 128}
    cost = {}
    for b in BUDGETS_STR:
        f = macro["f4_margin"][b]["nano_fraction"]
        cost[b] = {"nano_fraction": f,
                   "encode_ms": enc_ms["zero"] + f * enc_ms["nano"],
                   "always_nano_encode_ms": enc_ms["nano"]}
        if search:
            cost[b]["encode_plus_search_ms"] = (enc_ms["zero"] + search["zero"] +
                                                f * (enc_ms["nano"] + search["nano"]))
            cost[b]["always_nano_encode_plus_search_ms"] = enc_ms["nano"] + search["nano"]
    write_result(OUT, {"status": "COMPLETE", "measurement": "E10", "f4_router_cost": cost,
                       "scope": "descriptive; four features, all reported; fitted on dev forums",
                       "frozen": frozen, "macro": macro, "per_dataset": table,
                       "receipt": receipt(__file__, inputs, started, ("tokenizers", "scipy"))})


if __name__ == "__main__":
    main()
