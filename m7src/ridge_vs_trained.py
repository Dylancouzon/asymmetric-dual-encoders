"""Does the trained table actually beat the closed-form flat optimum? With a CI this time.

The session claimed +0.021 (gate fp16 0.5387 vs ridge 0.5181) and used it to refute "learned
weights buy nothing". Adversarial review was right that the claim had no CI and that the two
numbers came from separate runs. This scores both tables in one process, on the same components
and the same preprocessing, and paired-bootstraps the difference.

It stays a CONFOUNDED comparison and says so: flat+MSE+closed-form vs learned-weights+IDF+KL+SGD
differs in three ways at once, and the ridge lambda was selected on the 3-component proxy, so the
"flat optimum" here is proxy-optimal rather than suite-optimal. The controlled test is
program.phase4_mandatory's p4-weights. This only settles whether the gap is real, not what causes it.

Writes results/m7_ridge_vs_trained.json.
"""
import json
import sys

import numpy as np

import boot
import dev_eval
from _paths import REPO, WORK
from table import NO_PREFIX, WITH_PREFIX, Preproc, QueryTable, load_table, read_meta

OUT = WORK / "stage0"
PRE = {"noprefix": NO_PREFIX, "prefix": WITH_PREFIX}


def main(run_id="p1-objB", init="teacher", pre_name="noprefix"):
    pre = PRE[pre_name]
    summary = json.loads((OUT / f"ridge-{init}-{pre.fingerprint()}.json").read_text())
    best = max(summary["results"], key=lambda k: summary["results"][k]["macro"])
    W = np.load(OUT / f"ridge-{init}-{pre.fingerprint()}-lam{best}.npy")

    npz = WORK / "runs" / f"{run_id}.npz"
    tpre = Preproc(**read_meta(npz)["preproc"])
    if tpre != pre:
        raise AssertionError(f"preprocessing mismatch: ridge {pre} vs trained {tpre} -- the "
                             f"comparison would not be like-for-like")
    comps = dev_eval.dev_components()

    ridge = QueryTable(W, learned_weights=False).cuda().eval()
    per_r = dev_eval.eval_table(ridge, pre, components=comps)
    mr, _ = dev_eval.report(per_r, "[ridge] closed-form flat")
    del ridge

    for variant in ("fp16", "int8"):
        m = load_table(npz, variant=variant)
        per_t = dev_eval.eval_table(m, pre, components=comps)
        mt, _ = dev_eval.report(per_t, f"[{run_id}] trained ({variant})")
        del m
        r = boot.paired(per_t, per_r, alternative="greater")
        print(f"  trained({variant}) - ridge: d={r['delta']:+.4f} CI={r['ci95']} p={r['p_str']} "
              f"{'RESOLVED' if r['resolved'] else 'UNRESOLVED'}")
        if variant == "fp16":
            out = {"_note": "Trained table vs the closed-form MSE-optimal FLAT table, same "
                            "components, same preprocessing, one process, paired bootstrap. "
                            "CONFOUNDED BY DESIGN: flat+MSE+closed-form vs "
                            "learned-weights+IDF+KL+SGD differs in three ways at once, and the "
                            "ridge lambda was selected on the 3-component proxy so this 'flat "
                            "optimum' is proxy-optimal, not suite-optimal. Settles whether the gap "
                            "is real; does NOT attribute it to learned weights. The controlled "
                            "test is program.phase4_mandatory's p4-weights.",
                   "run_id": run_id, "preproc": pre_name, "ridge_lambda": best,
                   "components": comps, "ridge_macro": round(mr, 4)}
        out[f"trained_macro_{variant}"] = round(mt, 4)
        out[f"trained_minus_ridge_{variant}"] = r
    (REPO / "results" / "m7_ridge_vs_trained.json").write_text(json.dumps(out, indent=1))
    print("wrote results/m7_ridge_vs_trained.json")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["p1-objB"]))
