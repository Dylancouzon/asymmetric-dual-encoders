"""B7's registered precondition: verify the Gram-free solver on the REAL system, not a synthetic
one, across the REAL lambda grid, and compare the two tables on DEV MACRO rather than on their
distance from each other.

Why this is separate from `blockcg.verify()` and why the ledger made it a precondition. The
synthetic check establishes that block CG solves the normal equations it claims to solve. It does
not establish that it does so accurately enough ON THIS PROBLEM, and there are two specific
reasons to doubt the generalisation:

  * the real bag matrix's spectrum is set by real token frequencies, not by a Zipf sampler;
  * `m7src/stage0_ridge.LAMBDAS` reaches down to **1e-4**, and the smaller lambda is, the worse
    conditioned `(X^T X + lam I)` becomes -- fp32 CG at 1e-4 on a real Gram is exactly the corner
    where "converged" and "correct" can part company. The synthetic check ran at 1e-3 only.

And a Frobenius distance is the wrong acceptance criterion regardless: what the project consumes
from this solver is a RANKING, and two tables can be close in norm while ranking differently, or
far apart in norm on directions no query ever touches. So the criterion here is the dev macro the
teacher-screen criterion actually reads -- computed from both tables, on the same components, and
required to agree to a tolerance far below the effects the screen resolves.

ONE DISCLOSURE, and it does not weaken the check. The fit list used here is M7's
`work/trainq_texts.json`, which is a stale superset carrying 4,582 R1 hits (1.31%) against the
current protected-query index (m7/LEDGER.md). That inflates the ABSOLUTE dev macro and the numbers
here may not be quoted as clean. It is irrelevant to what this measures: both solvers see the
IDENTICAL X, Y and W0, so their agreement is invariant to how the fit list was built. A clean list
is required before any teacher-screen number is adopted; it is not required to compare two
solvers on one system.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import m8base
import blockcg
import probe_guard

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_b7_realdata.json"
COMPONENTS = ("cqadup-programmers", "cqadup-physics")
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)
# The tolerance the ledger needs: far below the practical-equivalence band (0.0040) and below the
# smallest effect the teacher screen is expected to resolve.
MACRO_TOL = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="*", default=list(LAMBDAS))
    ap.add_argument("--max-queries", type=int, default=None,
                    help="subsample the fit list; for a smoke only")
    a = ap.parse_args()

    import torch
    import dev_eval
    import stage0_ridge as sr
    from init_table import get_init
    from table import Preproc, QueryTable, get_tokenizer
    from teacher import QUERY_PREFIX, encode_cached
    import encoders

    spec = encoders.active()
    print(f"encoder: {spec.name} (vocab {spec.vocab}, dim {spec.dim})", flush=True)

    texts = json.loads((m8base.WORK / "trainq_texts.json").read_text())
    if isinstance(texts, dict):
        texts = texts.get("texts") or list(texts.values())
    if a.max_queries:
        texts = texts[:a.max_queries]
    print(f"fit list: {len(texts):,} queries (STALE superset, 1.31% R1 hits -- disclosed)",
          flush=True)

    pre = Preproc()
    tok = get_tokenizer()
    t0 = time.time()
    X = sr.bag_matrix(tok, texts, pre, spec.vocab)
    print(f"bag matrix {X.shape}, {X.nnz:,} nnz ({time.time()-t0:.0f}s)", flush=True)
    Y = np.asarray(encode_cached(f"trainq-{len(texts)}", texts, prefix=QUERY_PREFIX,
                                 dtype=torch.float16, verbose=False), dtype=np.float32)
    W0 = get_init("teacher", pre)
    print(f"Y {Y.shape}  W0 {W0.shape}", flush=True)

    def macro(W):
        model = QueryTable(W, weight_init=None, learned_weights=False).to(m8base.device())
        pq = dev_eval.eval_table(model, pre, components=COMPONENTS, tok=tok)
        m = float(np.mean([np.mean(list(pq[c].values())) for c in COMPONENTS]))
        del model
        m8base.empty_cache()
        return m, {c: float(np.mean(list(pq[c].values()))) for c in COMPONENTS}

    rows = []
    for lam in a.lambdas:
        t0 = time.time()
        Wd, di = blockcg.direct_ridge(X, Y, W0, lam)
        t_direct = time.time() - t0
        Wc, ci = blockcg.block_cg_ridge(X, Y, W0, lam, device=m8base.device(), verbose=True)
        md, per_d = macro(Wd)
        mc, per_c = macro(Wc)
        rel = float(np.linalg.norm(Wc - Wd) / max(np.linalg.norm(Wd), 1e-30))
        row = {
            "lambda": lam,
            "direct": {"seconds": round(t_direct, 1), "dev_macro_2": md, "per_component": per_d},
            "block_cg": {"seconds": ci["seconds"], "iterations": ci["iterations"],
                         "worst_rel_residual": ci["worst_rel_residual"],
                         "converged": ci["converged"],
                         "dev_macro_2": mc, "per_component": per_c},
            "relative_error_fro": rel,
            "dev_macro_abs_difference": abs(mc - md),
            "within_tolerance": bool(abs(mc - md) <= MACRO_TOL),
            "speedup": round(t_direct / max(ci["seconds"], 1e-9), 1),
        }
        rows.append(row)
        print(f"  lam={lam:g}: direct {md:.6f} ({t_direct:.0f}s)  cg {mc:.6f} "
              f"({ci['seconds']:.1f}s, {ci['iterations']} its)  |d|={abs(mc-md):.2e}  "
              f"{'OK' if row['within_tolerance'] else 'OUT OF TOLERANCE'}", flush=True)
        del Wd, Wc
        m8base.empty_cache()

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "criterion": f"the two tables' dev macro over {COMPONENTS} must agree to <= {MACRO_TOL}. "
                     f"A Frobenius distance is not the acceptance criterion: what this solver "
                     f"feeds is a RANKING.",
        "fit_list_disclosure": (
            "work/trainq_texts.json is a STALE superset with 4,582 R1 hits (1.31%). The absolute "
            "dev macros here are inflated and may not be quoted as clean. Both solvers see the "
            "IDENTICAL X, Y, W0, so their AGREEMENT -- which is all this measures -- is invariant "
            "to that. A clean, regenerated list is required before any teacher-screen number is "
            "adopted."),
        "encoder": {"name": spec.name, "vocab": spec.vocab, "dim": spec.dim},
        "n_fit_queries": len(texts), "components": list(COMPONENTS),
        "lambda_grid": list(a.lambdas),
        "rows": rows,
        "pass": bool(rows) and all(r["within_tolerance"] for r in rows),
        "smoke": bool(a.max_queries),
    }
    if a.max_queries:
        (RESULTS / "m8_b7_realdata.SMOKE.json").write_text(json.dumps(out, indent=2, default=str))
    else:
        probe_guard.write_result(OUT, out, "B7")
    print(json.dumps({"pass": out["pass"],
                      "max_macro_difference": max((r["dev_macro_abs_difference"] for r in rows),
                                                  default=None)}, indent=2))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
