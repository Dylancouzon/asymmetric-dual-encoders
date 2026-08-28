"""Capacity lever #5: pull rarely-updated rows back toward their B-phase values.

Protocol pre-registered in m7/LEDGER.md 2026-08-28 before any number. It is a SHRINKAGE
estimator, not new capacity: no extra rows, no extra query cost, and any row set is already
expressible by the table. The hypothesis is about generalization -- the A phase's update to a row
it saw a handful of times is dominated by cross-query interference, and the rare rows are exactly
the ones the six (scientific, biomedical, financial, argumentative) are most likely to hit, since
TRAIN is Wikipedia and e-commerce. `table.apply_unseen_policy` is the u=0 special case of this;
this is its continuous form.

  row_i = a_i * A_i + (1 - a_i) * B_i,   a_i = u_i / (u_i + tau)

with A the surviving candidate's rows, B the rows of the checkpoint it was initialized from, and
u_i the A-phase update count the candidate's npz already stores. tau in {1, 10, 100}; tau = 0 is
the baseline; it reconstructs the released artifact up to one fp16 rounding, which
is asserted and reported.

Usage: lever5_shrinkage.py [--smoke]
"""
import json
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

import boot
import dev_audit
import dev_eval
import encoders
import multieval
from _paths import REPO, WORK
from table import Preproc, QueryTable, dequantize_int8, ensure_release, get_tokenizer, \
    load_table, quantize_int8, read_meta

TAUS = (1.0, 10.0, 100.0)


def folded(rows, weights):
    """`save_release`'s fold: multiply each row by its token weight, so the artifact is
    self-contained and its int8 codes come from the folded rows.

    NOT bit-identical to the released file, and it cannot be: `save_release` computes
    fp16(w * rows_fp32) while the checkpoint on disk only keeps rows_fp16, so the best available
    reconstruction is w * fp16(rows). The difference is one fp16 rounding. This matters only for
    interpreting the tau=0 arm against the released macro; every comparison BELOW is tau=t against
    tau=0 computed the same way, so the rounding cancels out of the lever's own statistics."""
    return rows if weights is None or weights.size == 0 else weights[:, None] * rows


def load_pair(surv):
    a_cfg = json.loads((WORK / "runs" / f"{surv}.json").read_text())["cfg"]
    init = a_cfg.get("init", "")
    if not init.startswith("run:"):
        raise SystemExit(f"{surv} has init={init!r}: there is no B checkpoint to shrink toward")
    bid = init.split(":", 1)[1]
    za = np.load(WORK / "runs" / f"{surv}.npz")
    zb = np.load(WORK / "runs" / f"{bid}.npz")
    A = za["rows_fp16"].astype(np.float32)
    B = zb["rows_fp16"].astype(np.float32)
    w = za["token_weights"]
    u = za["updates"]
    if A.shape != B.shape:
        raise SystemExit(f"row shapes differ: {surv} {A.shape} vs {bid} {B.shape}")
    if u.size != A.shape[0]:
        raise SystemExit(f"{surv} stores {u.size} update counts for {A.shape[0]} rows")
    return bid, A, B, (w if w.size else None), u.astype(np.float64)


def main(smoke=False):
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer()
    spec = encoders.active()
    comps = dev_audit.SMOKE_COMPS if smoke else dev_eval.dev_components()
    pin = dev_audit.verify_pin(dev_eval.dev_components(), pool_bytes=not smoke)

    surv = json.loads((REPO / "results" / "m7_dev_audit_full.json").read_text())["surviving_candidate"]
    bid, A, B, w, u = load_pair(surv)
    rel = ensure_release(WORK / "runs" / f"{surv}.npz", device=dev)
    pre = Preproc(**read_meta(rel)["preproc"])
    print(f"lever 5: {surv} shrunk toward {bid}; {int((u == 0).sum()):,} of {len(u):,} rows were "
          f"never updated by the A phase, median update count {np.median(u):.0f}", flush=True)

    def table_for(rows, quant):
        r = folded(rows, w)
        if quant == "int8":
            r = dequantize_int8(*quantize_int8(r))
        return QueryTable(r, learned_weights=False).to(dev).eval()

    # tau = 0 must reconstruct the released artifact up to one fp16 rounding (see `folded`).
    # 5e-3 is save_release's own fixture tolerance; anything above that is a rule mismatch, not
    # rounding, and would mean the arms are being compared against the wrong baseline.
    base16 = table_for(A, "fp16")
    ref = load_table(rel, variant="fp16", device=dev)
    row_dev = float((base16.rows - ref.rows).abs().max())
    if row_dev > 5e-3:
        raise SystemExit(f"the tau=0 reconstruction deviates from the released table by {row_dev}; "
                         "the fold rule here does not match save_release")
    print(f"  tau=0 reconstruction matches the released rows to {row_dev:.2e}", flush=True)

    # tau=0 int8 is checked against the artifact's OWN stored codes, which is what ships. The
    # arms must all be built the same way (re-quantized from the blended rows), so the released
    # codes are used as a CHECK on the baseline rather than as the baseline itself.
    stored8 = np.load(rel)["rows_int8"]
    mine8, _ = quantize_int8(folded(A, w))
    code_dev = int(np.abs(stored8.astype(np.int16) - mine8.astype(np.int16)).max())
    print(f"  tau=0 int8 codes differ from the released artifact by at most {code_dev} step(s)",
          flush=True)
    models = {"tau0|fp16": base16, "tau0|int8": table_for(A, "int8")}
    frac = {}
    for tau in TAUS:
        a = (u / (u + tau)).astype(np.float32)
        rows = a[:, None] * A + (1.0 - a[:, None]) * B
        frac[tau] = {"mean_alpha": float(a.mean()), "rows_below_half": int((a < 0.5).sum())}
        for q in ("fp16", "int8"):
            models[f"tau{tau:g}|{q}"] = table_for(rows, q)
        del rows

    makers = {tag: (lambda m: (lambda c, texts: m.encode(texts, pre, tok=tok)))(m)
              for tag, m in models.items()}
    per = multieval.eval_makers(makers, components=comps,
                                max_docs=200_000 if smoke else None)

    arms = {f"{tau:g}": {q: boot.both_ways(per[f"tau{tau:g}|{q}"], per[f"tau0|{q}"])
                         for q in ("fp16", "int8")} for tau in TAUS}
    holm_by_q = {q: boot.holm({t: arms[t][q]["dependence_preserving"]["signflip"]["p"]
                               for t in arms}, alpha=0.05) for q in ("fp16", "int8")}
    passing = [t for t in arms
               if all(holm_by_q[q][t]["reject"]
                      and arms[t][q]["dependence_preserving"]["paired"]["ci95_raw"][0] > 0
                      for q in ("fp16", "int8"))]
    best = max(passing, key=lambda t: multieval.macro(per[f"tau{t}|fp16"])) if passing else None

    out = {"candidate": surv, "b_checkpoint": bid, "components": comps,
           "encoder": asdict(spec), "preproc": asdict(pre),
           "rows_never_updated": int((u == 0).sum()), "n_rows": int(len(u)),
           "tau0_row_dev_vs_released": row_dev, "tau0_int8_code_dev_vs_released": code_dev,
           "alpha_summary": {str(k): v for k, v in frac.items()},
           "baseline_macro_fp16": multieval.macro(per["tau0|fp16"]),
           "arms": {t: {"macro_fp16": multieval.macro(per[f"tau{t}|fp16"]),
                        "macro_int8": multieval.macro(per[f"tau{t}|int8"]),
                        "per_component": multieval.means(per[f"tau{t}|fp16"]),
                        "stats": arms[t]} for t in arms},
           "holm_alpha0.05_per_precision": holm_by_q, "passing": passing, "adopted": best,
           "pin_evidence": pin, "code_identity": dev_audit.code_identity(),
           "_protocol": "m7/LEDGER.md 'Capacity lever #5', pre-registered 2026-08-28 before any "
                        "number; shrinkage estimator, not a capacity claim. FOLDING CHOICE, "
                        "recorded before the numbers: the blend is computed on UNFOLDED rows and "
                        "then folded with the CANDIDATE's token weights, so a shrunk row is "
                        "served as w_A[i]*B_i, not as B's own served row w_B[i]*B_i. At u_i=0 the "
                        "two coincide (Adam never moves an untouched row's weight either), which "
                        "is the limit the rationale is about; for small u_i>0 it is a definitional "
                        "choice and this is where it is pinned.",
           "_status": "exploratory dev selection evidence (review #3 MAJOR 1)",
           "seconds": round(time.time() - t0, 1)}
    name = f"m7_lever5_shrinkage{'_smoke' if smoke else ''}.json"
    (REPO / "results" / name).write_text(json.dumps(out, indent=1))
    print(f"  baseline {out['baseline_macro_fp16']:.4f}  " +
          "  ".join(f"tau{t}={out['arms'][t]['macro_fp16']:.4f}" for t in arms) +
          f"  -> adopted={best}", flush=True)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
