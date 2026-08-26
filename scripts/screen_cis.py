"""Put confidence intervals on the phase-2 screen, which reported point macros only.

The arms differ by ~0.01 dev proxy macro. That is small enough that "monotone in learning rate" and
"random negatives beat mined ones" are claims about noise until a paired bootstrap says otherwise,
and the screen's own output cannot supply one: it records final macros, not per-query scores.

This re-scores each saved arm table on the three proxy components (a table eval is a lookup and an
average -- no transformer, so it is cheap), then paired-bootstraps every arm against the fixed
starting checkpoint and against the best arm. Holm is applied over the family of arm-vs-start tests,
because six comparisons against one baseline is exactly where an uncorrected p misleads.

    ../.venv/bin/python scripts/screen_cis.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7src"))

import boot
import dev_eval
from _paths import REPO, WORK
from table import Preproc, QueryTable, get_tokenizer

# The default family is the screen plus its learning-rate extension, all measured against the same
# fixed starting checkpoint, so one bootstrap family covers the whole lr curve. Override on argv.
ARMS = ["p2s-start", "p2s-sane-1e5", "p2s-sane-5e5", "p2s-sane-1e4", "p2s-old-lr-3e3",
        "p2s-sane-randneg", "p2x-rn-3e4", "p2x-rn-1e3", "p2x-rn-3e3"]
COMPONENTS = ("nq-250k", "cqadup-programmers", "cqadup-physics")


def load_arm(rid):
    z = np.load(WORK / "runs" / f"{rid}.npz")
    meta = json.loads((WORK / "runs" / f"{rid}.meta.json").read_text())
    w = z["token_weights"]
    m = QueryTable(z["rows_fp16"].astype(np.float32),
                   weight_init=(None if w.size == 0 else w.astype(np.float32)),
                   learned_weights=bool(meta.get("learned_weights"))).cuda()
    return m, Preproc(**meta["preproc"])


def main(arms=None):
    arms = list(arms) if arms else [a for a in ARMS if (WORK / "runs" / f"{a}.npz").exists()]
    tok = get_tokenizer()
    per_arm = {}
    for rid in arms:
        model, pre = load_arm(rid)
        pq = dev_eval.eval_table(model, pre, components=COMPONENTS, tok=tok)
        per_arm[rid] = pq
        macro = float(np.mean([np.mean(list(pq[c].values())) for c in COMPONENTS]))
        print(f"  {rid:20s} macro {macro:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    ref = per_arm["p2s-start"]
    vs_start, pvals = {}, {}
    for rid in arms:
        if rid == "p2s-start":
            continue
        r = boot.paired(per_arm[rid], ref, alternative="greater")
        r["signflip"] = boot.signflip(per_arm[rid], ref, alternative="greater", strict=True)
        vs_start[rid] = r
        pvals[rid] = r["signflip"]["p"]
        print(f"  {rid:20s} vs start: d={r['delta']:+.4f} CI={r['ci95']} "
              f"p={r['signflip']['p_str']} (sign-flip) "
              f"{'RESOLVED' if r['resolved'] else 'UNRESOLVED'}", flush=True)

    holm = boot.holm(pvals, alpha=0.025) if hasattr(boot, "holm") else None
    best = max((k for k in arms if k != "p2s-start"),
               key=lambda k: np.mean([np.mean(list(per_arm[k][c].values())) for c in COMPONENTS]))
    vs_best = {}
    for rid in arms:
        if rid == best:
            continue
        r = boot.paired(per_arm[best], per_arm[rid], alternative="two-sided")
        vs_best[rid] = r
        print(f"  {best} vs {rid:20s} d={r['delta']:+.4f} CI={r['ci95']} boot-tail={r['boot_tail_str']} "
              f"{'RESOLVED' if r['resolved'] else 'UNRESOLVED'}", flush=True)

    out = {"_note": "Paired bootstrap over the phase-2 screen's arms, re-scored from the saved "
                    "tables on the three proxy dev components. The screen itself reported point "
                    "macros only, and the arms are ~0.01 apart. Holm is applied over the family of "
                    "arm-vs-start tests. Dev-only: not a gate input, the six are unread.",
           "components": list(COMPONENTS),
           "macros": {k: round(float(np.mean([np.mean(list(v[c].values())) for c in COMPONENTS])), 4)
                      for k, v in per_arm.items()},
           "vs_start": vs_start, "holm_over_vs_start": holm,
           "best_arm": best, "best_vs_others": vs_best}
    (REPO / "results" / "m7_phase2_screen_cis.json").write_text(json.dumps(out, indent=1))
    print("wrote results/m7_phase2_screen_cis.json")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
