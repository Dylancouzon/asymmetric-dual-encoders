"""T1: the teacher screen, in the CG frame (LEDGER §10, registry probe `T1`).

WHAT B7 CHANGED. M7's screen shared ONE bag matrix across candidates because all ten registered
encoders ship a byte-identical 30,522-entry WordPiece vocabulary. Not one of T1's challengers
does, and the dense fp64 Gram at 50,368 rows is 20.3 GB against an 18 GB budget -- which is why
M7 closed granite-r2 and gte-modernbert "on arithmetic, not merit". `m8src/blockcg.py` never forms
the Gram, so the class is computable again (LEDGER §18). This module is the screen that uses it.

THE FRAME, and its honest name. "Fixed student frame" is fixed WITHIN a tokenizer family. A
cross-family challenger is screened in its own natural frame -- its tokenizer, its vocabulary --
because that is the frame it would actually ship in. The comparison is then a **teacher-plus-
tokenizer** comparison, not a teacher comparison, and the vocabulary size is reported at every row
so the confound is visible rather than hidden. Any swap argued on this screen must say which of
the two factors it is buying (LEDGER §10).

MECHANICS that are easy to get wrong and are therefore done explicitly:
  * ONE CANDIDATE PER SUBPROCESS. Every module that reads the encoder registry must be imported
    AFTER `M7_ENCODER` is set; M7's own script runs one candidate per process for exactly this
    reason.
  * `QueryTable`'s degenerate-empty-query fallback row defaults to `CLS_ID = 101`, which is BERT's.
    It is a constructor PARAMETER, so this passes `spec.cls_id` explicitly rather than editing
    frozen `m7src` (G3). A non-BERT tokenizer with the default would put a silently wrong vector
    behind every empty query.
  * Challenger `Spec`s are inserted into `encoders.REGISTRY` at RUNTIME by `m8src/challengers.py`.
    No file under `m7src/` is edited.
  * `validate_encoder.py` must pass for a Spec before it is screened -- skipping it is how a
    comparison silently runs the wrong model (m7/CODEMAP.md).

THE FIT LIST. `m8src/fitlist.py` regenerates it through the current protected-query filter. M7's
`work/trainq_texts.json` carries 4,582 R1 hits and may not back a quotable absolute number; the
screen refuses to use it unless `--allow-stale-fitlist` is passed, which marks the output.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import m8base

RESULTS = m8base.RESULTS
COMPONENTS = ("cqadup-programmers", "cqadup-physics")     # LEDGER §10: these two and no others
OFF_FAMILY = ("nq-250k", "hotpotqa")                      # the swap bar's condition 3
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)


def screen_one(name, fit_texts, lambdas=LAMBDAS, device=None):
    """Run ONE candidate in THIS process. The caller must have set M7_ENCODER before importing
    anything that reads the registry."""
    import torch
    import blockcg
    import dev_eval
    import encoders
    import stage0_ridge as sr
    from init_table import get_init
    from table import Preproc, QueryTable, get_tokenizer
    from teacher import QUERY_PREFIX, encode_cached

    device = device or m8base.device()
    spec = encoders.active()
    assert spec.name == name, f"active encoder is {spec.name!r}, expected {name!r}"
    pre = Preproc()
    tok = get_tokenizer()

    t0 = time.time()
    X = sr.bag_matrix(tok, fit_texts, pre, spec.vocab)
    Y = np.asarray(encode_cached(f"trainq-{len(fit_texts)}", fit_texts, prefix=QUERY_PREFIX,
                                 dtype=torch.float16, verbose=True), dtype=np.float32)
    W0 = get_init("teacher", pre)
    print(f"{name}: X {X.shape} ({X.nnz:,} nnz)  Y {Y.shape}  W0 {W0.shape} "
          f"({time.time()-t0:.0f}s)", flush=True)
    assert W0.shape[0] == spec.vocab, \
        f"{name}: init has {W0.shape[0]} rows but the Spec says vocab={spec.vocab}"

    out = {"encoder": {"name": spec.name, "repo": spec.repo, "revision": spec.revision,
                       "dim": spec.dim, "vocab": spec.vocab,
                       "tokenizer_id": spec.tokenizer_id, "cls_id": spec.cls_id},
           "int8_table_mb": round(spec.vocab * spec.dim / 1e6, 1),
           "n_fit_queries": len(fit_texts), "lambdas": {}}

    for lam in lambdas:
        t1 = time.time()
        W, info = blockcg.block_cg_ridge(X, Y, W0, lam, device=device, verbose=False)
        # fallback_id from the SPEC, not table.py's BERT default (see the module docstring).
        model = QueryTable(W, weight_init=None, learned_weights=False,
                           fallback_id=spec.cls_id).to(device)
        pq = dev_eval.eval_table(model, pre, components=COMPONENTS, tok=tok)
        macro = float(np.mean([np.mean(list(pq[c].values())) for c in COMPONENTS]))
        out["lambdas"][str(lam)] = {
            "dev_macro_2": macro,
            "per_component": {c: float(np.mean(list(pq[c].values()))) for c in COMPONENTS},
            "per_query": {c: {k: float(v) for k, v in pq[c].items()} for c in COMPONENTS},
            "cg": {k: info[k] for k in ("iterations", "worst_rel_residual", "converged",
                                        "seconds", "preconditioner")},
            "seconds": round(time.time() - t1, 1),
        }
        print(f"  {name} lam={lam:g}: dev_macro_2 {macro:.4f}  "
              f"({info['iterations']} cg its, {time.time()-t1:.0f}s)", flush=True)
        del model, W
        m8base.empty_cache()

    best = max(out["lambdas"], key=lambda k: out["lambdas"][k]["dev_macro_2"])
    out["best_lambda"] = best
    out["best_macro"] = out["lambdas"][best]["dev_macro_2"]
    # A best lambda at the grid EDGE means the optimum may lie outside it, and comparing a
    # candidate whose optimum is interior against one whose optimum is clipped is biased against
    # the second. M7 hit exactly this and widened the grid.
    out["best_lambda_at_grid_edge"] = best in (str(lambdas[0]), str(lambdas[-1]))
    return out


def load_fit_list(allow_stale=False):
    m8 = m8base.WORK / "m8_trainq_texts.json"
    if m8.exists():
        return json.loads(m8.read_text()), "m8_trainq_texts.json (filtered)"
    if not allow_stale:
        raise SystemExit(
            "m8src/fitlist.py has not been run: work/m8_trainq_texts.json is missing. M7's list "
            "carries 4,582 R1 hits (1.31%) against the current protected-query index and may not "
            "back a quotable absolute number (LEDGER §3.3). Run fitlist.py, or pass "
            "--allow-stale-fitlist to run a RANKING-ONLY screen whose output is marked stale.")
    return json.loads((m8base.WORK / "trainq_texts.json").read_text()), \
        "trainq_texts.json (STALE, 1.31% R1 hits -- ranking only, not quotable as clean)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", help="encoder Spec name, e.g. stella-400M-v5")
    ap.add_argument("--allow-stale-fitlist", action="store_true")
    ap.add_argument("--max-queries", type=int, default=None, help="smoke only")
    ap.add_argument("--lambdas", type=float, nargs="*", default=list(LAMBDAS))
    a = ap.parse_args()

    os.environ["M7_ENCODER"] = a.candidate
    import challengers                              # noqa: F401  (registers the Specs)

    texts, provenance = load_fit_list(a.allow_stale_fitlist)
    if a.max_queries:
        texts = texts[:a.max_queries]
    print(f"fit list: {len(texts):,} queries from {provenance}", flush=True)

    out = screen_one(a.candidate, texts, lambdas=a.lambdas)
    out["fit_list"] = provenance
    out["fit_list_is_clean"] = "STALE" not in provenance
    out["smoke"] = bool(a.max_queries)
    dest = RESULTS / (f"m8_t1_{a.candidate}.SMOKE.json" if a.max_queries
                      else f"m8_t1_{a.candidate}.json")
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n{a.candidate}: best lam={out['best_lambda']} macro={out['best_macro']:.4f} "
          f"table {out['int8_table_mb']} MB int8"
          f"{'  [BEST LAMBDA AT GRID EDGE]' if out['best_lambda_at_grid_edge'] else ''}")
    print(f"wrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
