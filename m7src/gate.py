"""The go/no-go gate (dev only), judged on a named checkpoint trained on TRAIN data only.

Four conditions from instructions-m7.md:
  G1  Stage-0 distilled table CI-resolved above the potion-retrieval-32M dev row
  G2  capacity probe passes its own bar (CI-resolved above the BM25 dev row) -- a diagnostic
      that is nonetheless a required gate condition
  G3  the candidate CI-resolved above BM25 on the dev macro
  G4  int8 equivalence: one-sided 97.5% upper bound of (fp16 - int8) below 0.005 on the dev macro

Pass -> full program. Fail -> negative-result report, stop. Report to Dylan either way.

WHAT THIS GATE IS, after Codex review #3. It cannot repair adaptive dev reuse -- the candidate was
selected on this same dev suite, so a gate run on it is not independent evidence. Its defensible
role is a MECHANICAL ELIGIBILITY AUDIT after all selection is finished: the exact frozen release
artifact, through the released `QueryTable` path, with the encoder fingerprint, table hashes and
all six pinned component hashes verified, aborting on any missing component or qid, dumping
unrounded per-query fp16 and int8 scores, and handling the nested held-out components with the
dependence-preserving statistics. Freeze immediately after; no recipe change once it has been seen.
"""
import json
import sys

import numpy as np

import boot
import dev_eval
from _paths import REPO, WORK
from table import NO_PREFIX, WITH_PREFIX, Preproc, ensure_release, load_table, read_meta

RUNS = WORK / "runs"
PRE = {"noprefix": NO_PREFIX, "prefix": WITH_PREFIX}


def refs():
    return json.loads(dev_eval.refs_path().read_text())


def restrict(per, comps):
    return {c: v for c, v in per.items() if c in comps}


def evaluate_checkpoint(run_id, components):
    """-> (per-component per-query dicts for fp16 and int8 variants, config)"""
    # Judge the RELEASE shape (weights folded into rows), not the training checkpoint: the
    # gate's claims are about the artifact that ships (review #2 BLOCKER 2). And read the query
    # rule from THAT artifact's metadata -- the previous name-keyed lookup reconstructed a Preproc
    # from the prefix alone, so it would have served a `pool_mode=sqrt` table under `mean`.
    src = ensure_release(RUNS / f"{run_id}.npz")
    meta = read_meta(src)
    pre = Preproc(**meta["preproc"])
    out = {}
    for variant in ("fp16", "int8"):
        m = load_table(src, variant=variant)
        out[variant] = dev_eval.eval_table(m, pre, components=list(components))
        mac, means = dev_eval.report(out[variant], f"  [{run_id}] {variant}")
        del m
    return out, meta


def run(run_id, stage0_id=None, components=None, probe_file=None, audit_vs=None,
        verify_pool_bytes=True):
    """components defaults to the PINNED dev suite. The gate's dev macro is "equal weight per
    component" over that suite -- silently dropping HotpotQA and the held-out slices (as an
    earlier default did) would have changed which side of the BM25 bar the candidate lands on,
    since BM25 is strongest on HotpotQA."""
    import dev_audit
    R = refs()
    comps = list(components) if components else dev_eval.dev_components()
    # The eligibility audit: pinned six, verified bytes, verified encoder. Aborts, never shrinks.
    pin_evidence = dev_audit.verify_pin(dev_eval.dev_components(), pool_bytes=verify_pool_bytes)
    if components is not None and list(components) != dev_eval.dev_components():
        print(f"[gate] WARNING: running on a SUBSET {comps}; this is not the pinned suite and the "
              "result is diagnostic only", flush=True)
    pot = restrict(R["potion-retrieval-32M"], comps)
    bm = restrict(R["bm25"], comps)
    # dev_eval.TEACHER_REF, not the literal: under a different teacher the key is that teacher's,
    # and a KeyError here would be the good outcome -- silently gating against the previous
    # teacher's rows would not be.
    teacher = restrict(R[dev_eval.TEACHER_REF], comps)
    # BM25 and potion have no row on the held-out slices (their corpora are pool row indices and
    # carry no document text). That restriction is disclosed, but it must be explicit here rather
    # than silently absorbed by boot._align's dataset intersection.
    text_backed = [c for c in comps if not c.startswith("heldout-")]
    for label, blob, need in (("potion", pot, text_backed), ("bm25", bm, text_backed),
                              ("teacher", teacher, comps)):
        missing = [c for c in need if c not in blob]
        if missing:
            raise SystemExit(f"GATE ABORTED: reference row '{label}' missing components {missing}; "
                             "run dev_eval.py first")
    print(f"[gate] dev components: {comps}")
    print(f"[gate] BM25/potion comparisons run on the text-backed subset: {text_backed}")

    per, meta = evaluate_checkpoint(run_id, comps)
    cand = per["fp16"]

    res = {"run_id": run_id, "components": comps, "conditions": {}}

    if stage0_id:
        s0, _ = evaluate_checkpoint(stage0_id, comps)
        g1 = boot.paired(restrict(s0["fp16"], text_backed), pot, alternative="greater",
                         strict=True)
    else:
        # G1 is defined on the Stage-0 distilled table; substituting the candidate is a weaker
        # test, so it is called out on the printed line, not just recorded in the JSON.
        g1 = boot.paired(restrict(cand, text_backed), pot, alternative="greater",
                         strict=True)
    res["conditions"]["G1_stage0_above_potion"] = {
        **g1, "pass": bool(g1["ci95_raw"][0] > 0), "checkpoint": stage0_id or run_id,
        "note": ("Stage-0 distilled table" if stage0_id else
                 "SUBSTITUTED the candidate for the Stage-0 table (weaker test)"),
        "components": text_backed}

    pf = probe_file or (REPO / "results" / "m7_capacity_probe_noprefix.json")
    if pf.exists():
        pr = json.loads(pf.read_text())
        import encoders
        probe_enc = pr.get("encoder")
        if probe_enc != encoders.active().name:
            # review #2 BLOCKER 3: a probe from another teacher (or a pre-tagging probe with no
            # encoder field) must not satisfy G2 for this one.
            res["conditions"]["G2_capacity_probe"] = {
                "pass": False, "note": f"probe encoder {probe_enc!r} != active "
                f"{encoders.active().name!r} -- re-run capacity_probe.py under this teacher"}
        else:
            res["conditions"]["G2_capacity_probe"] = {"pass": bool(pr["passed"]),
                                                     "macro": pr["macro"],
                                                     "vs_bm25": pr["vs_bm25_dev"],
                                                     "encoder": probe_enc}
    else:
        res["conditions"]["G2_capacity_probe"] = {"pass": False, "note": "probe not run"}

    # strict=True: a confirmatory-shaped comparison must abort on a missing qid rather than
    # silently score the intersection (Codex M-perquery; review #3's "abort on any missing qid").
    g3 = boot.paired(restrict(cand, text_backed), bm, alternative="greater", strict=True)
    g3_sf = boot.signflip(restrict(cand, text_backed), bm, alternative="greater", strict=True)
    res["conditions"]["G3_candidate_above_bm25"] = {
        **g3, "signflip_p": g3_sf["p"], "signflip_p_str": g3_sf["p_str"],
        "pass": bool(g3["ci95_raw"][0] > 0),
        "_note": "text-backed components only; BM25 has no row on the held-out slices"}

    # The held-out components are nested, so the equivalence bound gets the dependence-preserving
    # resampling; the dependence-blind one is kept beside it for comparison (review #3).
    g4 = boot.upper_bound_one_sided(per["fp16"], per["int8"], dep=True, strict=True)
    g4_blind = boot.upper_bound_one_sided(per["fp16"], per["int8"], strict=True)
    res["conditions"]["G4_int8_equivalence"] = {
        **g4, "bar": 0.005, "pass": bool(g4["upper_raw"] < 0.005),
        "dependence_blind": g4_blind}

    mean_over = lambda blob, cs: float(np.mean([np.mean(list(blob[c].values())) for c in cs]))
    macros = {k: mean_over(v, comps) for k, v in per.items()}
    macros_text_backed = {k: mean_over(v, text_backed) for k, v in per.items()}
    # BM25 and potion have no row on the held-out slices, so their macros are only defined on
    # the text-backed subset. Mixing the two component sets is how the first version of this
    # summary raised KeyError instead of quietly reporting an incomparable number.
    res["macros_all_components"] = {**macros,
                                    f"{dev_eval.TEACHER_REF} (teacher ceiling)": mean_over(teacher, comps)}
    res["macros_text_backed"] = {**macros_text_backed,
                                 "bm25": mean_over(bm, text_backed),
                                 "potion-retrieval-32M": mean_over(pot, text_backed),
                                 f"{dev_eval.TEACHER_REF} (teacher ceiling)": mean_over(teacher, text_backed)}
    res["macros"] = res["macros_text_backed"]   # the set every gate comparison uses
    res["retention_vs_teacher_text_backed"] = round(
        macros_text_backed["fp16"] / res["macros_text_backed"][f"{dev_eval.TEACHER_REF} (teacher ceiling)"], 4)
    res["retention_vs_teacher_all_components"] = round(
        macros["fp16"] / res["macros_all_components"][f"{dev_eval.TEACHER_REF} (teacher ceiling)"], 4)
    # Exploratory audit against the pre-lever winner: not a gate condition, and not confirmatory --
    # both artifacts were selected on this suite. It exists so the report can state the whole
    # selection chain's dev effect in one place, with the dependence handled (review #3).
    if audit_vs:
        base_per, _ = evaluate_checkpoint(audit_vs, comps)
        res["exploratory_audit_vs"] = {
            "baseline": audit_vs,
            "fp16": boot.both_ways(per["fp16"], base_per["fp16"]),
            "int8": boot.both_ways(per["int8"], base_per["int8"]),
            "_status": "EXPLORATORY: both sides were selected on this dev suite; this is not "
                       "confirmatory evidence (review #3 MAJOR 1)"}

    # Unrounded per-query scores for both precisions, so any later reader can redo every number
    # here without re-running the GPU pass (review #3's gate spec).
    import gzip
    dpath = REPO / "results" / f"m7_gate_perquery_{run_id}.json.gz"
    raw = json.dumps({"run_id": run_id, "components": comps,
                      "per_query": {v: {c: {q: float(x) for q, x in d.items()}
                                        for c, d in blob.items()}
                                    for v, blob in per.items()}}, sort_keys=True).encode()
    with gzip.GzipFile(filename=str(dpath), mode="wb", mtime=0) as f:
        f.write(raw)
    import hashlib
    res["per_query_dump"] = {"path": dpath.name,
                             "payload_sha256": hashlib.sha256(raw).hexdigest(),
                             "file_sha256": dev_audit.sha_file(dpath)}
    rel = ensure_release(RUNS / f"{run_id}.npz")
    res["artifact"] = {"release": rel.name, "sha256": dev_audit.sha_file(rel),
                       "meta_sha256": dev_audit.sha_file(rel.parent / (rel.stem + ".meta.json"))}
    res["pin_evidence"] = pin_evidence
    res["code_identity"] = dev_audit.code_identity()
    res["_role"] = ("mechanical eligibility audit after all selection; it cannot repair adaptive "
                    "dev reuse and is not evidence of generalization. Freeze immediately after.")
    res["PASS"] = all(v.get("pass") for v in res["conditions"].values())
    (REPO / "results" / f"m7_gate_{run_id}.json").write_text(json.dumps(res, indent=1))

    print()
    for k, v in res["conditions"].items():
        # G4 is an equivalence bound, not a paired CI, so it has no ci95 key: guard each field
        # independently rather than assuming the shapes match.
        bits = [f"d={v['delta']:+.4f}" if "delta" in v else "",
                f"CI={v['ci95']}" if "ci95" in v else "",
                f"upper={v['upper']} (bar {v.get('bar')})" if "upper" in v else "",
                f"boot-tail={v['boot_tail_str']}" if "boot_tail_str" in v else "",
                f"[{v['note']}]" if v.get("note") else ""]
        print(f"{'PASS' if v.get('pass') else 'FAIL'}  {k}  " + "  ".join(b for b in bits if b))
    print(f"\nmacros, text-backed ({len(text_backed)} comps): "
          f"{json.dumps({k: round(x,4) for k,x in res['macros_text_backed'].items()})}")
    print(f"macros, all ({len(comps)} comps): "
          f"{json.dumps({k: round(x,4) for k,x in res['macros_all_components'].items()})}")
    print(f"retention vs teacher: {res['retention_vs_teacher_text_backed']:.3f} (text-backed), "
          f"{res['retention_vs_teacher_all_components']:.3f} (all)")
    print(f"\nGO/NO-GO: {'GO' if res['PASS'] else 'NO-GO'}")
    return res


if __name__ == "__main__":
    # gate.py <run_id> [stage0_id] [--audit-vs <run_id>]
    #
    # The value FOLLOWING --audit-vs must be consumed with it. The previous parser stripped only
    # strings starting with "--", so `gate.py <run> --audit-vs s2w-1e3-s1000` silently passed
    # s2w-1e3-s1000 as positional arg 2 -- i.e. as `stage0_id` -- and G1 was then computed on that
    # run instead of the real Stage-0 checkpoint. Found by the pre-freeze Codex review.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("stage0_id", nargs="?", default=None)
    ap.add_argument("--audit-vs", dest="audit_vs", default=None)
    a = ap.parse_args()
    res = run(a.run_id, a.stage0_id, audit_vs=a.audit_vs)
    # A failing gate must STOP a driver. It printed NO-GO and returned 0, so `set -e` sailed past
    # it into the freeze -- the one place a silent pass is unrecoverable.
    sys.exit(0 if res.get("PASS") else 1)
