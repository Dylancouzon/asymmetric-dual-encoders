"""Where does the fusion gain come from? Per-component, with CIs.

`select_fusion.py` reports only the dev MACRO, and this ledger's most-repeated lesson is that a
macro without its per-component breakdown hides how narrow it is -- GO gate #1's +0.0270 was
carried entirely by nq-250k, with a CI-resolved LOSS to BM25 on HotpotQA. On the bge-era candidate
the fusion macro jumped 0.4795 -> 0.5520, far more than the +0.020..0.030 the M4 LightRetriever
hybrid precedent suggested, which is exactly the shape that has to be broken down before it is
believed.

This matters for transfer: the six-set contains no NQ-like component at all, so a gain
concentrated on nq-250k would NOT carry over, while a gain that is broad -- and especially one
that shows up on the two CQADupStack components, the nearest dev analogue to FiQA -- would.

Reuses the BM25 runs select_fusion.py cached; only the dense runs are recomputed.
Writes results/m7_fusion_report_<run_id>.json. Dev only; the six-set is never read.
"""
import json
import sys

import numpy as np

import boot
import dev_eval
import fusion
from _paths import REPO
from evalkit import per_query_ndcg
from select_fusion import bm25_run_and_key, dense_run
from table import Preproc, load_table, read_meta
from _paths import WORK


def main(run_id):
    spec = json.loads((REPO / "results" / f"m7_fusion_{run_id}.json").read_text())
    npz = WORK / "runs" / f"{run_id}.npz"
    pre = Preproc(**read_meta(npz)["preproc"])
    comps = spec["components"]
    model = load_table(npz, variant="int8")

    per = {}
    for c in comps:
        d = dense_run(c, model, pre)
        b, _ = bm25_run_and_key(c)
        qrels = dev_eval.doc_vecs(c)[4]
        f = fusion.apply_frozen(spec, d, b)
        per[c] = {"dense": per_query_ndcg(d, qrels), "bm25": per_query_ndcg(b, qrels),
                  "fused": per_query_ndcg(f, qrels)}
        m = {k: float(np.mean(list(v.values()))) for k, v in per[c].items()}
        print(f"  {c:22s} dense {m['dense']:.4f}  bm25 {m['bm25']:.4f}  "
              f"fused {m['fused']:.4f}  (fused-dense {m['fused']-m['dense']:+.4f})", flush=True)
    del model

    means = {c: {k: round(float(np.mean(list(v.values()))), 4) for k, v in per[c].items()}
             for c in comps}
    macro = {k: round(float(np.mean([means[c][k] for c in comps])), 4)
             for k in ("dense", "bm25", "fused")}
    out = {"_note": "Per-component decomposition of the dev fusion gain for the frozen spec in "
                    "results/m7_fusion_<run>.json. The macro alone cannot tell a broad gain from "
                    "a one-component gain, and the six-set has no NQ-like component, so transfer "
                    "depends on the breadth of this table -- especially the CQADupStack rows, the "
                    "nearest dev analogue to FiQA. Dev only.",
           "run_id": run_id, "spec": {k: spec[k] for k in ("family", "param", "depth")},
           "per_component": means, "macro": macro,
           "macro_gain_fused_over_dense": round(macro["fused"] - macro["dense"], 4)}

    for a, b in (("fused", "dense"), ("fused", "bm25"), ("dense", "bm25")):
        A = {c: per[c][a] for c in comps}
        B = {c: per[c][b] for c in comps}
        r = boot.paired(A, B, alternative="greater")
        out[f"{a}_vs_{b}"] = r
        print(f"  {a:6s} vs {b:6s}: d={r['delta']:+.4f} CI={r['ci95']} boot-tail={r['boot_tail_str']} "
              f"{'RESOLVED' if r['resolved'] else 'unresolved'}")
        for c, pd in r.get("per_dataset", {}).items():
            print(f"      {c:22s} {pd['delta']:+.4f} {pd['ci95']}")

    (REPO / "results" / f"m7_fusion_report_{run_id}.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote results/m7_fusion_report_{run_id}.json")


if __name__ == "__main__":
    # The run id is REQUIRED. It used to default to `p1-objB`, a retired bge-era checkpoint, so
    # running this bare would have reported a per-component breakdown of the wrong artifact under
    # the current candidate's name -- the same class of mistake as an evidence file named for
    # "the current candidate" (m7/FINDINGS.md, entry 18).
    if len(sys.argv) < 2:
        raise SystemExit("usage: fusion_report.py <run_id>   (no default: it would silently "
                         "report on a retired checkpoint)")
    main(sys.argv[1])
