"""What ships in the rows training never touched, and what they do to a query that hits one.

`table.apply_unseen_policy` exists, documents three policies, and is **never called on the save
path** -- so the released artifact serves every untouched row at its initialization, and no
committed number says what that means. This measures it instead of assuming it, because the
assumption is load-bearing twice over: the six are scientific, biomedical, financial and
argumentative while TRAIN is Wikipedia and e-commerce, so rare rows are exactly the rows the six
hit; and the recipe-simplification pre-registration argued for `input_emb` over `random` init on
precisely this ground.

Two distinctions the raw `updates` array does not make, and getting them wrong overstates the
problem by 2x:

  * `updates` is NOT restored from a `run:` init (`train.py` starts it at zero), so an A-only
    arm's `updates < 1` means "the A phase did not touch this row", not "training never did".
    The never-trained set is the INTERSECTION with the B checkpoint's.
  * a never-trained row that the tokenizer can never emit costs nothing. `[unusedN]` placeholders
    are counted separately from reachable pieces.

The quantity that matters at query time is not |row| but |w_i * row_i|, the row's contribution to
the bag before the final normalize -- a small row with a large learned weight is not small.

Usage: cold_rows.py <run_id> [<b_checkpoint_run_id>]   (the B id is read from the cfg if omitted)
"""
import json
import sys

import numpy as np

from _paths import REPO, WORK
from table import get_tokenizer


def main(run_id, b_id=None):
    cfg = json.loads((WORK / "runs" / f"{run_id}.json").read_text())["cfg"]
    if b_id is None:
        init = cfg.get("init", "")
        b_id = init.split(":", 1)[1] if init.startswith("run:") else None
    za = np.load(WORK / "runs" / f"{run_id}.npz")
    ua = za["updates"]
    ub = np.load(WORK / "runs" / f"{b_id}.npz")["updates"] if b_id else np.zeros_like(ua)
    rows = za["rows_fp16"].astype(np.float32)
    w = za["token_weights"]
    if w.size == 0:
        w = np.ones(len(rows), dtype=np.float32)

    never = (ua < 1) & (ub < 1)
    warm = ~never
    n = np.linalg.norm(rows, axis=1)
    eff = w * n                     # the contribution to the bag, before the final normalize

    tok = get_tokenizer()
    inv = {v: k for k, v in tok.get_vocab().items()}
    pieces = [inv.get(int(i), f"<{i}>") for i in np.where(never)[0]]
    unused = [p for p in pieces if p.startswith("[unused")]
    reachable = [p for p in pieces if not p.startswith("[unused")]

    def q(x, idx):
        return {k: float(v) for k, v in zip(("p5", "p50", "p95"),
                                            np.percentile(x[idx], [5, 50, 95]))}

    out = {
        "_what": "rows training never touched, in the artifact that would ship, and their "
                 "contribution to a query bag. `table.apply_unseen_policy` is never called on the "
                 "save path, so an untouched row ships at its initialization.",
        "run_id": run_id, "b_checkpoint": b_id, "init": cfg.get("init"),
        "vocab": int(len(rows)),
        "a_phase_untouched": int((ua < 1).sum()),
        "never_trained_either_phase": int(never.sum()),
        "never_trained_frac": float(never.mean()),
        "never_trained_unused_placeholders": len(unused),
        "never_trained_reachable_pieces": len(reachable),
        "reachable_sample": sorted(reachable)[:40],
        "row_norm_never": q(n, never), "row_norm_warm": q(n, warm),
        "bag_contribution_never": q(eff, never), "bag_contribution_warm": q(eff, warm),
        "contribution_ratio_never_over_warm": float(np.median(eff[never]) / np.median(eff[warm])),
        "_reading": "A ratio well BELOW 1 is the benign direction: an untrained token is nearly "
                    "ignored rather than injecting an arbitrary direction into the bag. A ratio "
                    "near or above 1 would mean the artifact lets untrained rows steer queries, "
                    "and would make the unseen-row policy a real decision rather than a "
                    "documented non-choice.",
    }
    (REPO / "results" / f"m7_cold_rows_{run_id}.json").write_text(json.dumps(out, indent=1))
    print(f"{run_id}: A-phase untouched {out['a_phase_untouched']}, never trained by EITHER phase "
          f"{out['never_trained_either_phase']} ({out['never_trained_frac']:.2%}) = "
          f"{len(unused)} [unused] placeholders + {len(reachable)} reachable pieces")
    print(f"  bag contribution |w*row|: never p50 {out['bag_contribution_never']['p50']:.3f} vs "
          f"warm p50 {out['bag_contribution_warm']['p50']:.3f}  ratio "
          f"{out['contribution_ratio_never_over_warm']:.3f}")
    print(f"  reachable sample: {out['reachable_sample'][:20]}")
    print(f"wrote results/m7_cold_rows_{run_id}.json")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
