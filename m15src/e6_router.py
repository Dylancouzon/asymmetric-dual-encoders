"""E6: a fertility router between Zero and Nano, fitted on the two M7 dev forums only.

Method: `m15/MEASUREMENTS.md`, E6. Thresholds are fixed from the fit forums before any
evaluation dataset is read; the evaluation reads committed M20 per-query rows and query texts.
"""
import json

import numpy as np

from common import (EVAL12, FIT_FORUMS, M20_SCORES, REPO, SEED, m20_rows, receipt,
                    refuse_reserved, utc_now, write_result)

OUT = REPO / "results" / "m15_e6_router.json"
BUDGETS = (0.10, 0.25, 0.50)


def zero_tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    import roster_ids as I
    path = hf_hub_download(I.ZERO_HUB_REPO, "tokenizer.json", revision=I.ZERO_HUB_REVISION)
    tok = Tokenizer.from_file(path)
    tok.no_padding()
    tok.no_truncation()
    return tok, path


def fertility(tok, texts):
    """Units per whitespace word, whole query, no special tokens (m8src/d2_pre.py:620)."""
    enc = tok.encode_batch(list(texts), add_special_tokens=False)
    words = np.array([max(1, len(t.split())) for t in texts], dtype=float)
    return np.array([len(e.ids) for e in enc], dtype=float) / words


def per_query(dataset):
    """(fertility-ready texts, zero, nano) aligned on the committed M20 query ids."""
    from datasets import load_dataset
    refuse_reserved(dataset)
    zero, zmeta = m20_rows(dataset, "zero-dense")
    nano, nmeta = m20_rows(dataset, "nano-dense")
    if set(zero) != set(nano):
        raise RuntimeError(f"{dataset}: query sets differ")
    row = json.loads((M20_SCORES / dataset / "zero-dense.json").read_text())
    queries = load_dataset(row["source"], "queries", revision=row["revision"])["queries"]
    text = {str(q): t for q, t in zip(queries["_id"], queries["text"]) if str(q) in zero}
    qids = sorted(zero)
    if len(text) != len(qids):
        raise RuntimeError(f"{dataset}: {len(qids) - len(text)} scored queries have no text")
    meta = [zmeta, nmeta, {"queries": row["source"], "revision": row["revision"]}]
    return ([text[q] for q in qids], np.array([zero[q] for q in qids]),
            np.array([nano[q] for q in qids]), meta)


def evaluate(fert, zero, nano, t, rng):
    route = fert > t
    f = float(route.mean())
    routed = np.where(route, nano, zero)
    random_mean = float((1 - f) * zero.mean() + f * nano.mean())
    k = int(round(f * len(zero)))
    oracle = float((zero.sum() + np.sort(nano - zero)[::-1][:k].sum()) / len(zero))
    headroom = oracle - random_mean
    # Router minus random at the same f, as a per-query paired quantity: random routing's expected
    # per-query score is (1 - f) zero + f nano.
    diff = routed - ((1 - f) * zero + f * nano)
    idx = rng.integers(0, len(diff), size=(10_000, len(diff)))
    boot = diff[idx].mean(axis=1)
    return {"nano_fraction": f, "router": float(routed.mean()), "always_zero": float(zero.mean()),
            "always_nano": float(nano.mean()), "random_same_fraction": random_mean,
            "oracle_same_fraction": oracle,
            "efficiency": float((routed.mean() - random_mean) / headroom) if headroom > 0 else None,
            "router_minus_random": float(diff.mean()),
            "router_minus_random_ci95": [float(np.quantile(boot, .025)),
                                         float(np.quantile(boot, .975))]}


def main():
    started = utc_now()
    tok, tok_path = zero_tokenizer()
    inputs = [{"zero_tokenizer": tok_path}]
    # 1. Fit on the two dev forums, then freeze the thresholds before touching EVAL12.
    fit = [per_query(d) for d in FIT_FORUMS]
    for _, _, _, m in fit:
        inputs += m
    fit_fert = np.concatenate([fertility(tok, texts) for texts, _, _, _ in fit])
    thresholds = {str(b): float(np.quantile(fit_fert, 1 - b)) for b in BUDGETS}
    print("frozen thresholds", thresholds, flush=True)
    rng = np.random.default_rng(SEED)
    fit_zero = np.concatenate([z for _, z, _, _ in fit])
    fit_nano = np.concatenate([n for _, _, n, _ in fit])
    in_sample = {b: evaluate(fit_fert, fit_zero, fit_nano, t, rng) for b, t in thresholds.items()}
    # 2. Evaluate on the 12 datasets.
    table = {}
    for dataset in EVAL12:
        texts, zero, nano, meta = per_query(dataset)
        inputs += meta
        fert = fertility(tok, texts)
        table[dataset] = {"mean_fertility": float(fert.mean()),
                          **{b: evaluate(fert, zero, nano, t, rng)
                             for b, t in thresholds.items()}}
    macro = {b: {k: float(np.mean([table[d][b][k] for d in EVAL12]))
                 for k in ("nano_fraction", "router", "always_zero", "always_nano",
                           "random_same_fraction", "oracle_same_fraction", "router_minus_random")}
             for b in thresholds}
    write_result(OUT, {"status": "COMPLETE", "measurement": "E6",
                       "scope": "descriptive; one feature, thresholds fitted on the M7 dev forums",
                       "feature": "Zero tokenizer units per whitespace word, no special tokens",
                       "rule": "route to Nano when fertility > t_B", "budgets": list(BUDGETS),
                       "thresholds": thresholds, "fit_forums": list(FIT_FORUMS),
                       "fit_in_sample": in_sample, "macro_over_12": macro,
                       "per_dataset": table,
                       "receipt": receipt(__file__, inputs, started,
                                          ("tokenizers", "datasets"))})


if __name__ == "__main__":
    main()
