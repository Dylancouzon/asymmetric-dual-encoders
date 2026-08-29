"""RESEARCH-ONLY clean-stack-tax arm (m7/LEDGER.md, 'the clean-stack tax').

One arm: the FROZEN recipe (p35b-2m -> p35w-2m-s2500), byte-faithful clones, with decontaminated
MS MARCO added as a sixth pair source (side-bank positives, pseudo-query pool untouched -- the
arm-shape pre-registration). Never released, never uploaded, never fused into the released system;
`freeze.assert_releasable` refuses it by name, and `score` asserts that refusal as a self-check.

Modes (one process per leg, per the memory lesson):
    smoke   tiny B->A chain through the side-bank path (no execution history) -- run FIRST
    b       tax-msmarco-b  (objective B, 16k steps)
    a       tax-msmarco-a  (objective A, 2500 steps, init from the b leg)
    score   full pinned dev suite + ONE non-confirmatory six-set read, both artifacts served
            under the FROZEN query rule; writes results/m7_cleanstack_tax.json

    M7_ENCODER=stella-400M-v5 PYTHONPATH=m7src .venv/bin/python m7src/cleanstack_tax.py <mode>
"""
import dataclasses
import json
import sys
from datetime import datetime, timezone

import numpy as np

from _paths import REPO, WORK

SOURCES = ("hotpotqa-train", "fever-train", "squad-train", "esci-us", "mrtydi-en", "msmarco-train")
BID, AID = "tax-msmarco-b", "tax-msmarco-a"
FROZEN_B, FROZEN_A = "p35b-2m", "p35w-2m-s2500"
OUT = REPO / "results" / "m7_cleanstack_tax.json"
SIX = ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid"]


def clone_cfg(rid, **over):
    from train import Cfg
    c = json.loads((WORK / "runs" / f"{rid}.json").read_text())["cfg"]
    known = {f.name for f in dataclasses.fields(Cfg)}
    c = {k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items() if k in known}
    c.update(over)
    return Cfg(**c)


TAX = dict(sources=SOURCES, side_pos_sources=("msmarco-train",))


def mode_smoke():
    import torch

    from train import run
    cb = clone_cfg(FROZEN_B, run_id="smoke-tax-b", steps_b=60, eval_every=10 ** 9, **TAX)
    print(f"SMOKE B {json.dumps(dataclasses.asdict(cb))}", flush=True)
    _, mb, _ = run(cb)
    del mb
    torch.cuda.empty_cache()
    ca = clone_cfg(FROZEN_A, run_id="smoke-tax-a", init="run:smoke-tax-b", steps_a=30,
                   eval_every=10 ** 9, **TAX)
    print(f"SMOKE A {json.dumps(dataclasses.asdict(ca))}", flush=True)
    _, ma, _ = run(ca)
    del ma
    print("SMOKE ok -- side-bank chain path executes; numbers meaningless at 90 steps", flush=True)


def mode_b():
    import sweep
    sweep.one(BID, base=clone_cfg(FROZEN_B, **TAX))


def mode_a():
    import sweep
    sweep.one(AID, base=clone_cfg(FROZEN_A, init=f"run:{BID}", **TAX))


def _ledger(line):
    with open(REPO / "m7" / "LEDGER.md", "a") as f:
        f.write(line.rstrip() + "\n")


def mode_score():
    import torch

    import boot
    import dev_eval
    import freeze
    import fusion
    from evalkit import per_query_ndcg, topk_ids_scores
    from table import Preproc, ensure_release, load_table
    from teacher import encode_cached

    # self-check: the release guard must REFUSE this lineage by name
    try:
        freeze.assert_releasable(AID)
        raise AssertionError("freeze.assert_releasable ACCEPTED the msmarco arm -- the release "
                             "guard is broken; stop and fix before anything else")
    except SystemExit as e:
        assert "msmarco" in str(e).lower(), f"refused for the wrong reason: {e}"
        print("[guard] assert_releasable refuses tax-msmarco-a, as it must", flush=True)

    fz = json.loads((REPO / "m7" / "FREEZE.json").read_text())
    pre = Preproc(**fz["preproc"])          # the FROZEN query rule (pool_mode=sqrt) for BOTH sides
    spec = fz["fusion"]
    tax_npz = ensure_release(WORK / "runs" / f"{AID}.npz")
    cand_npz = ensure_release(WORK / "runs" / f"{FROZEN_A}.npz")
    assert freeze.sha256_file(cand_npz) == fz["table_sha256"], \
        "the frozen candidate's release bytes changed; refusing to compare against them"

    res = {"_note": "RESEARCH-ONLY clean-stack-tax measurement (m7/LEDGER.md). NON-CONFIRMATORY: "
                    "supports one descriptive sentence, no tier claim, no selection, no change to "
                    "the released system. The msmarco arm is refused by freeze.assert_releasable "
                    "(asserted at the top of this run).",
           "arm": AID, "arm_table_sha256": freeze.sha256_file(tax_npz),
           "frozen_candidate": FROZEN_A, "preproc": fz["preproc"], "fusion_spec_param":
           {"family": spec["family"], "param": spec["param"], "depth": spec["depth"]}}

    # --- full pinned dev suite, both artifacts, matched code path -------------------------
    comps = dev_eval.dev_components()
    dev = {}
    for tag, npz in (("tax", tax_npz), ("frozen", cand_npz)):
        m = load_table(npz, variant="int8")
        dev[tag] = dev_eval.eval_table(m, pre, components=list(comps))
        del m
        torch.cuda.empty_cache()
    macro = {t: float(np.mean([np.mean(list(v.values())) for v in dev[t].values()])) for t in dev}
    r = boot.paired_dep(dev["tax"], dev["frozen"], alternative="two-sided") \
        if hasattr(boot, "paired_dep") else boot.paired(dev["tax"], dev["frozen"])
    res["dev"] = {"macro": {k: round(v, 4) for k, v in macro.items()},
                  "tax_minus_frozen": r}
    print(f"[dev] tax {macro['tax']:.4f} vs frozen {macro['frozen']:.4f} "
          f"delta {r['delta']:+.4f} CI={r['ci95']}", flush=True)

    # --- ONE non-confirmatory six-set read -------------------------------------------------
    with open(REPO / "m7" / "SIX_ACCESS.log", "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} cleanstack_tax.score: pre-registered "
                "NON-CONFIRMATORY six-set read (m7/LEDGER.md 'the clean-stack tax')\n")
    man = json.loads((REPO / "results" / "eval_manifest.json").read_text())["datasets"]
    from core import doc_text
    from datasets import load_dataset
    from hashing import sha_stream_list
    prior = json.loads((REPO / "results" / "m7_final_run.json").read_text())
    six = {}
    for ds in SIX:
        froz = json.loads((REPO / "results" / "frozen_eval" / f"{ds}.json").read_text())
        q_ids = sorted(froz["queries"])
        q_texts = [froz["queries"][q] for q in q_ids]
        corpus = load_dataset(f"BeIR/{ds}", "corpus")["corpus"]
        doc_ids = [str(x) for x in corpus["_id"]]
        doc_texts = [doc_text(r) for r in corpus]
        assert sha_stream_list(doc_ids) == man[ds]["corpus_ids_sha256"], f"{ds} corpus ids drifted"
        assert sha_stream_list(doc_texts) == man[ds]["corpus_text_sha256"], f"{ds} corpus drifted"
        dv = encode_cached(f"final-six-{ds}-docs", doc_texts, prefix="", dtype=torch.float32,
                           verify=True)
        runs = {}
        for variant in ("fp16", "int8"):
            m = load_table(tax_npz, variant=variant)
            runs[f"{variant}-table"] = topk_ids_scores(m.encode(q_texts, pre), dv, doc_ids,
                                                       k=fusion.DEPTH, qids=q_ids)
            del m
            torch.cuda.empty_cache()
        runs["bm25"] = fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts)
        runs["fusion"] = fusion.apply_frozen(spec, runs["int8-table"], runs["bm25"])
        six[ds] = {k: per_query_ndcg(r, froz["qrels"]) for k, r in runs.items()}
        print(f"  {ds}: " + " ".join(f"{k}={np.mean(list(v.values())):.4f}"
                                     for k, v in six[ds].items()), flush=True)

    by_sys = {s: {ds: six[ds][s] for ds in SIX} for s in ("fp16-table", "int8-table", "bm25",
                                                          "fusion")}
    res["six_macros"] = {s: round(float(np.mean([np.mean(list(v.values()))
                                                 for v in by_sys[s].values()])), 4)
                         for s in by_sys}
    # paired against the frozen candidate's stored final-run per-query values (descriptive)
    res["six_tax_minus_frozen"] = {}
    for s in ("int8-table", "fusion"):
        A, B = by_sys[s], prior["six"][s]
        rr = boot.paired(A, B)
        rr["signflip"] = boot.signflip(A, B)
        res["six_tax_minus_frozen"][s] = rr
        print(f"[six] tax-{s} vs frozen-{s}: {rr['delta']:+.4f} CI={rr['ci95']}", flush=True)
    # descriptive distance to the (spent) bars, for the report's one sentence
    pq = json.loads((REPO / "results" / "perquery.json").read_text())
    for name, sysname, b_name in (("vs_lr_dense_pertask", "int8-table", "lr-dense-pertask"),
                                  ("vs_opensearch", "fusion", "opensearch-doc-v3-gte")):
        B = boot.from_perquery_json(pq, b_name, set(SIX))
        rr = boot.paired(by_sys[sysname], B)
        res[name] = rr
        print(f"[six] tax-{sysname} vs {b_name}: {rr['delta']:+.4f} CI={rr['ci95']}", flush=True)

    res["six_per_query"] = {s: {ds: {q: float(x) for q, x in v.items()}
                                for ds, v in by_sys[s].items()} for s in by_sys}
    OUT.write_text(json.dumps(res, indent=1))
    _ledger(f"- {datetime.now(timezone.utc).isoformat()} — **CLEAN-STACK-TAX six-set read** "
            f"(pre-registered, NON-CONFIRMATORY): arm `{AID}`, results in "
            f"`results/m7_cleanstack_tax.json`. Supports one descriptive sentence; no tier claim.")
    print(f"\nwrote {OUT.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = {"smoke": mode_smoke, "b": mode_b, "a": mode_a, "score": mode_score}.get(mode)
    if fn is None:
        raise SystemExit("usage: cleanstack_tax.py smoke|b|a|score")
    fn()
