"""Evaluate the best closed-form ridge table on the FULL pinned dev suite.

stage0_ridge sweeps lambda on the fast three-component proxy so six 9.5-TFLOP Cholesky solves
stay affordable. The structural claim deserves a number on the same six components the gate uses,
so this re-evaluates the winning table (already on disk) against the whole suite. No solve.
"""
import json
import sys

import numpy as np
import torch

import boot
import dev_eval
from _paths import REPO, WORK
from table import NO_PREFIX, WITH_PREFIX, QueryTable

OUT = WORK / "stage0"
PRE = {"noprefix": NO_PREFIX, "prefix": WITH_PREFIX}


def main(init="teacher", pre_name="noprefix"):
    pre = PRE[pre_name]
    summary = json.loads((OUT / f"ridge-{init}-{pre.fingerprint()}.json").read_text())
    best = max(summary["results"], key=lambda k: summary["results"][k]["macro"])
    W = np.load(OUT / f"ridge-{init}-{pre.fingerprint()}-lam{best}.npy")
    print(f"best lambda on the proxy: {best} (proxy macro "
          f"{summary['results'][best]['macro']:.4f})")

    m = QueryTable(W, learned_weights=False).cuda().eval()
    comps = dev_eval.dev_components()
    per = dev_eval.eval_table(m, pre, components=comps)
    macro, means = dev_eval.report(per, "[ridge] closed-form flat table")

    refs = json.loads((dev_eval.DEVRES / "refs.json").read_text())
    text_backed = [c for c in comps if not c.startswith("heldout-")]
    out = {"_note": "Closed-form MSE-optimal flat-weight bag-of-tokens table: the global optimum "
                    "of flat distillation under squared loss, so no training run can beat it at "
                    "that objective. A diagnostic, not a gate input.",
           "init": init, "preproc": pre_name, "lambda": best,
           "train_cos": summary["results"][best]["train_cos"],
           "overlap_at_10": summary["results"][best]["overlap_at_10"],
           "vocab_coverage_train_queries": summary["vocab_coverage_train_queries"],
           "n_train_queries": summary["n_queries"],
           "dev_macro_full_suite": macro, "per_component": means,
           "components": comps, "text_backed_components": text_backed}

    for name in ("bm25", "potion-retrieval-32M", "bge-base-symmetric"):
        R = {c: refs[name][c] for c in refs[name] if c in comps}
        sub = [c for c in R]
        r = boot.paired({c: per[c] for c in sub}, R, alternative="greater")
        ref_macro = float(np.mean([np.mean(list(R[c].values())) for c in sub]))
        out[f"vs_{name}"] = {**r, "components": sub, "ref_macro": round(ref_macro, 4)}
        print(f"  vs {name:24s} ({len(sub)} comps, ref {ref_macro:.4f}): "
              f"d={r['delta']:+.4f} CI={r['ci95']} p={r['p_str']} "
              f"{'RESOLVED' if r['resolved'] else 'unresolved'}")
    ceil = out["vs_bge-base-symmetric"]["ref_macro"]
    out["retention_vs_teacher"] = round(macro / ceil, 4)
    print(f"  retention vs teacher: {out['retention_vs_teacher']:.3f}")
    (REPO / "results" / "m7_stage0_ridge.json").write_text(json.dumps(out, indent=1))
    print("wrote results/m7_stage0_ridge.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "teacher",
         sys.argv[2] if len(sys.argv) > 2 else "noprefix")
