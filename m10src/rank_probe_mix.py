"""Follow-up to rank_probe.py, from its caches: which 384-d subspace, fit on WHICH queries?

rank_probe.py fit the basis on NQ-open questions only. A student trained on a broad query mix
learns one compromise subspace for every distribution at once, so the bound that matters for M10
is a basis fit on a MIXTURE: does one 384-d affine subspace serve forum questions when it was fit
on NQ + the other forum component? Also the cross-component bound (fit on physics, score
programmers, and vice versa). No new encodes; reads work/m10_rank_probe/*.npy. Diagnostic.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7src"))
import os; os.environ.setdefault("M7_ENCODER", "stella-400M-v5")
import devsuite, evalkit  # noqa: E402
from _paths import REPO, WORK  # noqa: E402
from rank_probe import load_head, apply_head, pca_basis, project, HEADS, verify_manifest  # noqa: E402

CACHE = WORK / "m10_rank_probe"
OUT = REPO / "results" / "m10_rank_probe_mac.json"
COMPS = ["cqadup-programmers", "cqadup-physics"]
KS = [384, 512, 640]


def main():
    out = json.loads(OUT.read_text())
    head = load_head(1024)
    fitq = apply_head(np.load(CACHE / "fitq_20000.npy"), head)
    data = {}
    for name in COMPS:
        doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(name)
        verify_manifest(name, doc_ids, doc_texts, q_ids, qrels)
        D = apply_head(np.load(CACHE / f"{name}_docs.npy"), head).astype(np.float16)
        Q = apply_head(np.load(CACHE / f"{name}_queries.npy"), head)
        full = float(np.mean(list(evalkit.score(Q, q_ids, D, doc_ids, qrels).values())))
        data[name] = (D, Q, q_ids, doc_ids, qrels, full)
    rows = {}
    for target in COMPS:
        other = [c for c in COMPS if c != target][0]
        D, Q, q_ids, doc_ids, qrels, full = data[target]
        bases = {
            "nq_only": fitq,
            f"{other}_only": data[other][1],
            f"nq+{other}": np.vstack([fitq, data[other][1]]),
            f"nq+{other}+self(oracle)": np.vstack([fitq, data[other][1], Q]),
        }
        rows[target] = {"full_ndcg10": full, "bases": {}}
        for bname, X in bases.items():
            mu, vt, _ = pca_basis(X)
            r = {}
            for k in KS:
                Qk = project(Q, mu, vt, k)
                m = float(np.mean(list(evalkit.score(Qk, q_ids, D, doc_ids, qrels).values())))
                r[str(k)] = {"ndcg10": m, "retention": m / full}
                print(f"{target:20s} basis={bname:28s} k={k}: {m:.4f} ({m/full:.3f})", flush=True)
            rows[target]["bases"][bname] = {"n_fit": int(len(X)), "rank_k": r}
    out["mixture_bases_1024d"] = rows
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
