"""Rank teacher candidates by how LEARNABLE they are by a lookup table, not by their own ceiling.

The teacher probe (results/m7_teacher_probe.json) measures each candidate's SYMMETRIC retrieval
quality. That is the ceiling, not the product: the shipped system's score is the ceiling times how
well a bag-of-token-vectors can land where that teacher's query encoder lands, and a stronger tower
can be less additively decomposable. Codex's review made the same point (M-probe), and it is the
tie-break for a live disagreement: arctic-embed-l wins the measured probe and discloses no overlap
with our six, while the MTEB->six projection puts stella ~0.025 higher.

Method, per candidate:
  * X: row-normalized token-count matrix over the 349,934 TRAIN queries. Tokenizer-independent of
    the candidate (all five ship a byte-identical vocab.txt), so it is built once and shared.
  * Y: that candidate's own TRAIN query vectors (queries only -- no documents, no pool).
  * W0: that candidate's teacher-init rows, as the ridge's regularisation anchor.
  * solve the same closed form Stage 0 uses, for a small lambda grid.
  * score on DEV: retrieval nDCG@10 on the two CQADupStack components (THE criterion -- it is what
    ships) and cosine agreement to the teacher's own dev query vectors (diagnostic only).

    Cosine agreement was originally designated the primary metric, for signal-to-noise, and the runs
    refuted that: it is not monotone with retrieval in lambda (within a candidate, raising lambda
    raises cosine and lowers nDCG) and it mis-ranks candidates (e5-large-v2 has the HIGHEST cosine
    agreement at 0.90 and a mid-pack retrieval ratio of 0.63). Imitating a teacher's query vector in
    cosine is not the same as reproducing its ranking. That divergence is also independent evidence
    for the Codex finding that the closed-form ridge is not an upper bound on RETRIEVAL: its
    objective and the metric part company.

Fit on TRAIN, measured on dev, so no arm is scored on what it fitted. This is a CLOSED-FORM ranking:
phase 2 showed training moves a table, so a candidate that wins here is favoured, not crowned.

    ../.venv/bin/python scripts/teacher_learnability.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m7src"))

import boot
from _paths import DEVICE, REPO, WORK, empty_cache

CANDIDATES = ["arctic-embed-l", "stella-400M-v5"]
COMPONENTS = ("cqadup-programmers", "cqadup-physics")
# 1e-4 is here because the first run's best lambda was 1e-3, the grid's own lower EDGE: if one
# candidate's optimum sits inside the grid and another's sits below it, the comparison is biased
# against the second. Overridable on argv so a rerun can extend rather than redo (the per-lambda
# results are merged into the existing JSON).
LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1]
TRAINQ = WORK / "trainq_texts.json"


def run_candidate(name, X, q_texts, lambdas=None):
    """Imported inside the function: every module that reads the registry must be imported AFTER
    M7_ENCODER is set, and this script runs one candidate per subprocess for exactly that reason."""
    import dev_eval
    import stage0_ridge as sr
    from init_table import get_init
    from table import Preproc, QueryTable, get_tokenizer
    from teacher import QUERY_PREFIX, encode_cached

    pre = Preproc()
    tok = get_tokenizer()
    Y = np.asarray(encode_cached(f"trainq-{len(q_texts)}", q_texts, prefix=QUERY_PREFIX,
                                 dtype=torch.float16, verbose=False), dtype=np.float32)
    W0 = get_init("teacher", pre)
    out = {"dim": int(Y.shape[1]), "n_train_queries": int(Y.shape[0]), "lambdas": {}}
    for lam in (lambdas or LAMBDAS):
        t0 = time.time()
        W = sr.solve_ridge(X, Y, W0, lam)
        model = QueryTable(W, weight_init=None, learned_weights=False).to(DEVICE)
        pq = dev_eval.eval_table(model, pre, components=COMPONENTS, tok=tok)
        macro = float(np.mean([np.mean(list(pq[c].values())) for c in COMPONENTS]))
        # cosine agreement on DEV queries: the table's bag vector vs the teacher's own query vector
        cos = {}
        for c in COMPONENTS:
            _, _, _, qt, _, _ = dev_eval.doc_vecs(c)
            tq = np.asarray(encode_cached(f"dev-{c}-queries-pfx", qt, prefix=QUERY_PREFIX,
                                          dtype=torch.float16, verbose=False), dtype=np.float32)
            Xd = sr.bag_matrix(tok, qt, pre, W.shape[0])
            V = Xd @ W
            V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
            cos[c] = float(np.mean(np.sum(V * tq, axis=1)))
        out["lambdas"][str(lam)] = {
            "dev_macro_2": round(macro, 4),
            "dev_cosine_agreement": {k: round(v, 4) for k, v in cos.items()},
            "dev_cosine_mean": round(float(np.mean(list(cos.values()))), 4),
            "per_query": {c: {k: round(v, 6) for k, v in pq[c].items()} for c in COMPONENTS},
            "solve_seconds": round(time.time() - t0, 1)}
        print(f"  {name} lam={lam:g}: dev_macro_2 {macro:.4f}  cos {np.mean(list(cos.values())):.4f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        del model, W
        empty_cache()
    best = max(out["lambdas"], key=lambda k: out["lambdas"][k]["dev_macro_2"])
    out["best_lambda"] = best
    return out


def main():
    import encoders
    from table import Preproc, get_tokenizer
    import stage0_ridge as sr

    name = encoders.active().name
    lambdas = [float(a) for a in sys.argv[1:]] or LAMBDAS
    # verified against results/m7_trainq_manifest.json: a probe fitted on a silently
    # different TRAIN query set is not comparable to the committed incumbent row, and on a
    # second machine the list arrives by transfer rather than by re-derivation.
    import encode_trainq
    q_texts = encode_trainq.load_texts()
    print(f"{name}: building the shared bag matrix over {len(q_texts):,} TRAIN queries", flush=True)
    tok = get_tokenizer()
    X = sr.bag_matrix(tok, q_texts, Preproc(), tok.vocab_size)
    res = run_candidate(name, X, q_texts, lambdas)
    p = REPO / "results" / f"m7_learnability_{name}.json"
    if p.exists():          # keep lambdas already computed; an extension run must not redo them
        prior = json.loads(p.read_text()).get("lambdas", {})
        prior.update(res["lambdas"])
        res["lambdas"] = dict(sorted(prior.items(), key=lambda kv: float(kv[0])))
        res["best_lambda"] = max(res["lambdas"],
                                 key=lambda k: res["lambdas"][k]["dev_macro_2"])
    p.write_text(json.dumps({"_note": "Closed-form table learnability for one teacher candidate: "
                                      "ridge-fit on TRAIN query vectors, scored on dev. See the "
                                      "module docstring. Fit on TRAIN, measured on dev.",
                             "encoder": name, **res}, indent=1))
    print(f"wrote {p.name}")


if __name__ == "__main__":
    main()
