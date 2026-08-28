"""Paired full-suite comparison of two RELEASE-shape tables in one process.

The lever adoption rule (LEDGER 2026-08-27) judges a candidate against the winner on the full
pinned dev suite with signflip + paired CI, fp16 and int8 independently. This does that for any
two run ids, through ensure_release (never the raw training npz — review #2 BLOCKER 2) and the
verified matrix eval path (reproduces the gate winner macro exactly).

SUPERSEDED for the chain comparisons by `dev_audit.py`, which scores through the released
`QueryTable` path rather than the matrix shortcut, keeps unrounded per-query values, and reports
the dependence-preserving statistics the nested held-out components require. Kept for ad-hoc
two-way comparisons; anything decision-bearing should go through the audit.

Usage: compare_release.py <candidate_run_id> [baseline_run_id=s2w-1e3-s1000]
"""
import json
import sys

import numpy as np

import dev_eval
from _paths import REPO, WORK
from bigram_residual import eval_variants, macro, stats
from table import NO_PREFIX, WITH_PREFIX, Preproc, dequantize_int8, ensure_release, \
    get_tokenizer, load_table, quantize_int8, read_meta


def rows_of(run_id):
    rel = ensure_release(WORK / "runs" / f"{run_id}.npz")
    meta = read_meta(rel)
    assert meta["weights_folded"], meta
    m = load_table(rel, variant="fp16", device="cpu")
    return m.rows.detach().numpy().astype(np.float32), Preproc(**meta["preproc"])


def main(cand, base="s2w-1e3-s1000"):
    tok = get_tokenizer()
    Wc, pre_c = rows_of(cand)
    Wb, pre_b = rows_of(base)
    assert pre_c == pre_b, (pre_c, pre_b)
    V = tok.vocab_size
    comps = dev_eval.dev_components()
    per = eval_variants(
        {"cand": (Wc, {}), "base": (Wb, {}),
         "cand-int8": (dequantize_int8(*quantize_int8(Wc)), {}),
         "base-int8": (dequantize_int8(*quantize_int8(Wb)), {})},
        tok, pre_c, V, comps)
    import encoders
    sp = encoders.active()
    out = {"candidate": cand, "baseline": base, "components": comps,
           # M7_ENCODER defaults to bge-base when unset, so a comparison that does not record
           # which encoder produced it cannot prove it used the right caches (review #3 MINOR).
           "encoder": {"name": sp.name, "repo": sp.repo, "revision": sp.revision, "dim": sp.dim},
           "macro": {t: round(macro(p), 4) for t, p in per.items()},
           "per_component_macro": {
               c: {t: round(float(np.mean(list(per[t][c].values()))), 4) for t in per}
               for c in comps},
           "cand_vs_base": stats(per["cand"], per["base"]),
           "cand_vs_base_int8": stats(per["cand-int8"], per["base-int8"])}
    p = REPO / "results" / f"m7_compare_{cand}_vs_{base}.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "s2w-1e3-s1000")
