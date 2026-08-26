"""The go/no-go gate (dev only), judged on a named checkpoint trained on TRAIN data only.

Four conditions from instructions-m7.md:
  G1  Stage-0 distilled table CI-resolved above the potion-retrieval-32M dev row
  G2  capacity probe passes its own bar (CI-resolved above the BM25 dev row) -- a diagnostic
      that is nonetheless a required gate condition
  G3  the candidate CI-resolved above BM25 on the dev macro
  G4  int8 equivalence: one-sided 97.5% upper bound of (fp16 - int8) below 0.005 on the dev macro

Pass -> full program. Fail -> negative-result report, stop. Report to Dylan either way.
"""
import json
import sys

import numpy as np

import boot
import dev_eval
from _paths import REPO, WORK
from table import NO_PREFIX, WITH_PREFIX, load_table

RUNS = WORK / "runs"
PRE = {"noprefix": NO_PREFIX, "prefix": WITH_PREFIX}


def refs():
    return json.loads((dev_eval.DEVRES / "refs.json").read_text())


def restrict(per, comps):
    return {c: v for c, v in per.items() if c in comps}


def evaluate_checkpoint(run_id, components):
    """-> (per-component per-query dicts for fp16 and int8 variants, config)"""
    meta = json.loads((RUNS / f"{run_id}.meta.json").read_text())
    pre = PRE[meta["preproc"]["prefix"] and "prefix" or "noprefix"]
    out = {}
    for variant in ("fp16", "int8"):
        m = load_table(RUNS / f"{run_id}.npz", variant=variant)
        out[variant] = dev_eval.eval_table(m, pre, components=list(components))
        mac, means = dev_eval.report(out[variant], f"  [{run_id}] {variant}")
        del m
    return out, meta


def run(run_id, stage0_id=None, components=None, probe_file=None):
    """components defaults to the PINNED dev suite. The gate's dev macro is "equal weight per
    component" over that suite -- silently dropping HotpotQA and the held-out slices (as an
    earlier default did) would have changed which side of the BM25 bar the candidate lands on,
    since BM25 is strongest on HotpotQA."""
    R = refs()
    comps = list(components) if components else dev_eval.dev_components()
    pot = restrict(R["potion-retrieval-32M"], comps)
    bm = restrict(R["bm25"], comps)
    teacher = restrict(R["bge-base-symmetric"], comps)
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
        g1 = boot.paired(restrict(s0["fp16"], text_backed), pot, alternative="greater")
    else:
        # G1 is defined on the Stage-0 distilled table; substituting the candidate is a weaker
        # test, so it is called out on the printed line, not just recorded in the JSON.
        g1 = boot.paired(restrict(cand, text_backed), pot, alternative="greater")
    res["conditions"]["G1_stage0_above_potion"] = {
        **g1, "pass": bool(g1["ci95"][0] > 0), "checkpoint": stage0_id or run_id,
        "note": ("Stage-0 distilled table" if stage0_id else
                 "SUBSTITUTED the candidate for the Stage-0 table (weaker test)"),
        "components": text_backed}

    pf = probe_file or (REPO / "results" / "m7_capacity_probe_noprefix.json")
    if pf.exists():
        pr = json.loads(pf.read_text())
        res["conditions"]["G2_capacity_probe"] = {"pass": bool(pr["passed"]),
                                                 "macro": pr["macro"], "vs_bm25": pr["vs_bm25_dev"]}
    else:
        res["conditions"]["G2_capacity_probe"] = {"pass": False, "note": "probe not run"}

    g3 = boot.paired(restrict(cand, text_backed), bm, alternative="greater")
    res["conditions"]["G3_candidate_above_bm25"] = {**g3, "pass": bool(g3["ci95"][0] > 0)}

    g4 = boot.upper_bound_one_sided(per["fp16"], per["int8"])
    res["conditions"]["G4_int8_equivalence"] = {**g4, "bar": 0.005,
                                               "pass": bool(g4["upper"] < 0.005)}

    mean_over = lambda blob, cs: float(np.mean([np.mean(list(blob[c].values())) for c in cs]))
    macros = {k: mean_over(v, comps) for k, v in per.items()}
    macros_text_backed = {k: mean_over(v, text_backed) for k, v in per.items()}
    # BM25 and potion have no row on the held-out slices, so their macros are only defined on
    # the text-backed subset. Mixing the two component sets is how the first version of this
    # summary raised KeyError instead of quietly reporting an incomparable number.
    res["macros_all_components"] = {**macros,
                                    "bge-base-symmetric (teacher ceiling)": mean_over(teacher, comps)}
    res["macros_text_backed"] = {**macros_text_backed,
                                 "bm25": mean_over(bm, text_backed),
                                 "potion-retrieval-32M": mean_over(pot, text_backed),
                                 "bge-base-symmetric (teacher ceiling)": mean_over(teacher, text_backed)}
    res["macros"] = res["macros_text_backed"]   # the set every gate comparison uses
    res["retention_vs_teacher_text_backed"] = round(
        macros_text_backed["fp16"] / res["macros_text_backed"]["bge-base-symmetric (teacher ceiling)"], 4)
    res["retention_vs_teacher_all_components"] = round(
        macros["fp16"] / res["macros_all_components"]["bge-base-symmetric (teacher ceiling)"], 4)
    res["PASS"] = all(v.get("pass") for v in res["conditions"].values())
    (REPO / "results" / f"m7_gate_{run_id}.json").write_text(json.dumps(res, indent=1))

    print()
    for k, v in res["conditions"].items():
        # G4 is an equivalence bound, not a paired CI, so it has no ci95 key: guard each field
        # independently rather than assuming the shapes match.
        bits = [f"d={v['delta']:+.4f}" if "delta" in v else "",
                f"CI={v['ci95']}" if "ci95" in v else "",
                f"upper={v['upper']} (bar {v.get('bar')})" if "upper" in v else "",
                f"p={v['p_str']}" if "p_str" in v else "",
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
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
