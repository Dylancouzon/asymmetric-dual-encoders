"""Re-adjudicate capacity lever #4 (count saturation) on a NEW candidate artifact.

Lever #4 was adjudicated on `p35w-2m-s2500`. The negatives ablation then changed the candidate,
and `m7/LEDGER.md` says so explicitly in the negatives pre-registration: "A promoted winner
changes the candidate, which re-triggers fusion re-selection and re-adjudicates lever #4 on the
new artifact". This is that re-adjudication. `adopt_pool_mode.py` refuses until the committed
lever-4 artifact names the run id being adopted, which is exactly the interlock that forced this.

The rule is NOT re-derived here -- it is the one pre-registered before any pooling number existed
and is reproduced verbatim in `_protocol`: for each precision independently, Holm at alpha=0.05
over the three-arm family {binary, cap2, sqrt}, plus a raw paired CI lower bound above zero;
an arm must pass in BOTH fp16 and int8; among survivors, the largest fp16 dev macro. Statistics
are the dependence-preserving ones (`boot.both_ways` -> `dependence_preserving`), because two of
the six dev components share queries.

Two things this adds over `dev_audit.py`'s L4 block, neither of which changes the rule:

  * it takes the artifact by name instead of deriving it from the hard-coded lever CHAIN, so the
    adjudicated artifact is an argument rather than a consequence of a chain that no longer ends
    at the candidate;
  * it reports the OUT-OF-DOMAIN subset macro (cqadup-programmers + cqadup-physics) alongside the
    six-component macro, per the disclosure rule pre-registered in LEDGER.md's biased-estimator
    section. Disclosure only: the adoption bar is untouched.

The baseline is the artifact served at `mean`, whatever its metadata currently says -- an
already-adopted mode would otherwise silently become the thing the arms are measured against.

Usage: lever4_readjudicate.py <run_id> [--smoke]
"""
import gzip
import hashlib
import json
import sys
import time
from dataclasses import asdict

import torch

import boot
import dev_audit
import dev_eval
import encoders
import multieval
from _paths import REPO, WORK
from table import POOL_MODES, Preproc, encode_pooled, ensure_release, get_tokenizer, \
    load_table, read_meta

# The only dev components that are NOT in the TRAIN mix or its Wikipedia family. LEDGER.md's
# "THE DEV MACRO IS A BIASED ESTIMATOR" section requires every adoption to report them separately.
OOD = ("cqadup-programmers", "cqadup-physics")

PROTOCOL = (
    "m7/LEDGER.md 'Capacity lever #4', pre-registered 2026-08-27 before any number: adopt iff, "
    "under the dependence-preserving statistics, the arm clears Holm at alpha=0.05 within its "
    "precision's three-arm family AND its raw paired CI lower bound is > 0, in BOTH fp16 and "
    "int8; largest fp16 dev macro among those passing. Pooling counts post-truncation WordPiece "
    "occurrences INCLUDING specials. Re-run on a new candidate under the negatives "
    "pre-registration, which states that a promoted arm re-triggers this adjudication.")


def ood_macro(per_comp):
    """Macro over the out-of-domain components only. Same unweighted-component-mean shape as
    `multieval.macro`, so the two numbers are read the same way."""
    means = multieval.means(per_comp)
    present = [c for c in OOD if c in means]
    if not present:
        return None
    return sum(means[c] for c in present) / len(present)


def main(run_id, smoke=False):
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_tokenizer()
    spec = encoders.active()
    comps = dev_audit.SMOKE_COMPS if smoke else dev_eval.dev_components()
    pin = dev_audit.verify_pin(dev_eval.dev_components(), pool_bytes=not smoke)

    rel = ensure_release(WORK / "runs" / f"{run_id}.npz", device=device)
    meta = read_meta(rel)
    assert meta.get("weights_folded"), f"{run_id}: not a release-shape artifact"
    # Served at `mean` regardless of the artifact's current declaration: the baseline of this
    # family is the unsaturated rule, and re-running after an adoption must give the same answer.
    pre = Preproc(**{**meta["preproc"], "pool_mode": "mean"})
    models = {q: load_table(rel, variant=q, device=device) for q in ("fp16", "int8")}
    modes = [m for m in POOL_MODES if m != "mean"]
    print(f"lever4 re-adjudication on {run_id}: {pre}\n  arms {modes}  components {comps}",
          flush=True)

    def maker(model, mode):
        if mode == "mean":
            return lambda c, texts: model.encode(texts, pre, tok=tok)
        return lambda c, texts: encode_pooled(model, texts, pre, mode=mode, tok=tok)

    makers = {f"{q}|{m}": maker(models[q], m) for q in ("fp16", "int8")
              for m in ("mean", *modes)}
    per = multieval.eval_makers(makers, components=comps,
                                max_docs=200_000 if smoke else None)

    arms = {m: {q: boot.both_ways(per[f"{q}|{m}"], per[f"{q}|mean"]) for q in ("fp16", "int8")}
            for m in modes}
    holm_by_q = {q: boot.holm({m: arms[m][q]["dependence_preserving"]["signflip"]["p"]
                               for m in modes}, alpha=0.05) for q in ("fp16", "int8")}
    passing = [m for m in modes
               if all(holm_by_q[q][m]["reject"]
                      and arms[m][q]["dependence_preserving"]["paired"]["ci95_raw"][0] > 0
                      for q in ("fp16", "int8"))]
    best = max(passing, key=lambda m: multieval.macro(per[f"fp16|{m}"])) if passing else None

    out = {
        "adjudicated_on": run_id, "components": comps,
        "smoke": bool(smoke),
        "encoder": asdict(spec),
        "release": rel.name, "release_sha256": dev_audit.sha_file(rel),
        "baseline_preproc": asdict(pre),
        "baseline_macro_fp16": multieval.macro(per["fp16|mean"]),
        "baseline_macro_int8": multieval.macro(per["int8|mean"]),
        "baseline_ood_macro_fp16": ood_macro(per["fp16|mean"]),
        "arms": {m: {"macro_fp16": multieval.macro(per[f"fp16|{m}"]),
                     "macro_int8": multieval.macro(per[f"int8|{m}"]),
                     "ood_macro_fp16": ood_macro(per[f"fp16|{m}"]),
                     "ood_delta_fp16": (ood_macro(per[f"fp16|{m}"]) - ood_macro(per["fp16|mean"]))
                     if ood_macro(per[f"fp16|{m}"]) is not None else None,
                     "per_component": multieval.means(per[f"fp16|{m}"]),
                     "stats": arms[m]} for m in modes},
        "holm_alpha0.05_per_precision": holm_by_q, "passing": passing, "adopted": best,
        "_ood_disclosure": "`ood_macro_fp16` / `ood_delta_fp16` cover cqadup-programmers and "
                           "cqadup-physics, the only dev components outside the TRAIN mix and its "
                           "Wikipedia family. Mandatory disclosure per LEDGER.md; it does not "
                           "enter the adoption bar.",
        "_protocol": PROTOCOL,
        "_status": ("SMOKE: corpora truncated, every number here is MEANINGLESS" if smoke else
                    "exploratory dev selection evidence (review #3 MAJOR 1)"),
    }

    dump = {"components": comps, "encoder": asdict(spec), "run_id": run_id,
            "per_query": {t: {c: {qq: float(v) for qq, v in d.items()} for c, d in p.items()}
                          for t, p in per.items()}}
    tag = f"lever4_{run_id}" + ("_smoke" if smoke else "")
    dpath = REPO / "results" / f"m7_devperquery_{tag}.json.gz"
    raw = json.dumps(dump, sort_keys=True).encode()
    with gzip.GzipFile(filename=str(dpath), mode="wb", mtime=0) as f:
        f.write(raw)
    out["per_query_dump"] = {"path": dpath.name,
                             "payload_sha256": hashlib.sha256(raw).hexdigest(),
                             "file_sha256": dev_audit.sha_file(dpath)}
    out["pin_evidence"] = pin
    out["code_identity"] = dev_audit.code_identity()
    out["seconds"] = round(time.time() - t0, 1)

    # Named for the artifact it adjudicates, never "full". A fixed filename meaning "the current
    # candidate" silently re-points when the candidate changes: on 2026-08-28 the shipping
    # artifact's own metadata cited `m7_lever4_pooling_full.json` as the evidence for its adopted
    # rule, while that file had come to hold a DIFFERENT artifact's failed adjudication. A reader
    # following the pointer would have found evidence contradicting the adoption.
    name = f"m7_lever4_pooling_{run_id}{'_smoke' if smoke else ''}.json"
    (REPO / "results" / name).write_text(json.dumps(out, indent=1))

    print(f"\nbaseline (mean) {out['baseline_macro_fp16']:.4f} fp16 / "
          f"{out['baseline_macro_int8']:.4f} int8, ood {out['baseline_ood_macro_fp16']:.4f}")
    for m in modes:
        a, d16 = out["arms"][m], arms[m]["fp16"]["dependence_preserving"]
        d8 = arms[m]["int8"]["dependence_preserving"]
        print(f"  {m:>6}: {a['macro_fp16']:.4f} fp16 (ood {a['ood_macro_fp16']:.4f}, "
              f"{a['ood_delta_fp16']:+.4f})  dep fp16 {d16['paired']['ci95_raw']} "
              f"p={d16['signflip']['p']:.4f}  int8 p={d8['signflip']['p']:.4f}  "
              f"holm fp16 reject={holm_by_q['fp16'][m]['reject']} "
              f"int8 reject={holm_by_q['int8'][m]['reject']}")
    print(f"passing {passing} -> adopted {best}   [{out['seconds']}s] -> results/{name}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], smoke="--smoke" in sys.argv)
