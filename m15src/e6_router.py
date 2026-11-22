"""E6: a fertility router between Zero and Nano, fitted on the two M7 dev forums only.

Method: `m15/MEASUREMENTS.md`, E6. Thresholds are fixed from the fit forums before any
evaluation dataset is read; the evaluation reads committed M20 per-query rows and query texts.
"""
import numpy as np

from common import (EVAL12, FIT_FORUMS, REPO, SEED, load_public, m20_rows, receipt,
                    refuse_reserved, sha_file, utc_now, write_result)

OUT = REPO / "results" / "m15_e6_router.json"
FROZEN = REPO / "results" / "m15_e6_thresholds.json"
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
    """(query texts, zero, nano) aligned on the committed M20 query ids."""
    refuse_reserved(dataset)
    zero, zmeta = m20_rows(dataset, "zero-dense")
    nano, nmeta = m20_rows(dataset, "nano-dense")
    if set(zero) != set(nano):
        raise RuntimeError(f"{dataset}: query sets differ")
    data = load_public(dataset, with_corpus=False)   # pinned revisions, six-set access trail
    text = dict(zip(data["q_ids"], data["q_texts"]))
    qids = sorted(zero)
    return ([text[q] for q in qids], np.array([zero[q] for q in qids]),
            np.array([nano[q] for q in qids]), [zmeta, nmeta, data["pins"]])


def _contrasts(route, zero, nano):
    """Router mean, random-at-same-fraction mean and oracle-at-same-fraction mean."""
    f = route.mean()
    router = np.where(route, nano, zero).mean()
    random_mean = (1 - f) * zero.mean() + f * nano.mean()
    k = int(round(f * len(zero)))
    oracle = (zero.sum() + np.sort(nano - zero)[::-1][:k].sum()) / len(zero)
    return float(f), float(router), float(random_mean), float(oracle)


def evaluate(fert, zero, nano, t, rng, draws=10_000):
    route = fert > t
    f, router, random_mean, oracle = _contrasts(route, zero, nano)
    headroom = oracle - random_mean
    # Resample aligned (route, zero, nano) rows with t fixed, recomputing f in every draw, so the
    # interval is for routing advantage at equal traffic share (Astra 2026-09-30, finding 1).
    boot = np.empty(draws)
    for b in range(draws):
        i = rng.integers(0, len(zero), len(zero))
        _, r, m, _ = _contrasts(route[i], zero[i], nano[i])
        boot[b] = r - m
    return {"nano_fraction": f, "router": router, "always_zero": float(zero.mean()),
            "always_nano": float(nano.mean()), "random_same_fraction": random_mean,
            "oracle_same_fraction": oracle,
            "efficiency": float((router - random_mean) / headroom) if headroom > 0 else None,
            "router_minus_random": router - random_mean,
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
    # Persist the thresholds before any evaluation dataset is read (Astra 2026-09-30, finding 3).
    write_result(FROZEN, {"status": "FROZEN", "measurement": "E6 thresholds",
                          "budgets": list(BUDGETS), "thresholds": thresholds,
                          "fit_forums": list(FIT_FORUMS), "n_fit_queries": int(len(fit_fert)),
                          "receipt": receipt(__file__, list(inputs), started,
                                             ("tokenizers", "datasets"))})
    inputs.append({"frozen_thresholds": str(FROZEN.relative_to(REPO)),
                   "sha256": sha_file(FROZEN)})
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
