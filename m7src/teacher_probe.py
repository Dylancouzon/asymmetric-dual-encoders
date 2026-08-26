"""Rank candidate teachers by MEASURING them, on the two dev components that are clean for all.

Why this exists. The M7 teacher projection maps MTEB v1 English Retrieval onto our six-set.
results/m7_calibration.json shows that map is loose (ratio 0.926-1.001 across the nine models we
measured; affine resid sd 0.0102), so a 1-point MTEB gap between two candidates is inside the
calibration noise and cannot rank them. A direct measurement can.

Which components. NOT the six -- those are the final eval and must not inform a selection.
NOT nq-250k or hotpotqa either: MTEB's registry records NQ, HotpotQA, FEVER, MSMARCO, ArguAna and
FiQA2018 as in-domain training data for stella_en_400M_v5, so those components would flatter it
against bge-base. CQADupStack is on no candidate's disclosed list, is real qrels, and the ledger
already flags it as the nearest dev analogue to FiQA -- the six-set row most at risk. So the two
CQADupStack components are the honest common ground.

What it reports: each candidate's own SYMMETRIC retrieval score, i.e. the teacher ceiling our
table would be distilling toward, in our units and our harness. Not a table, not a gate input.

Cost: ~70K docs + ~1.9K queries per candidate.
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

import dev_eval
import encoders
from _paths import REPO, WORK
from evalkit import per_query_ndcg, topk_ids_scores

COMPONENTS = ("cqadup-programmers", "cqadup-physics")
OUT = WORK / "teacherprobe"
OUT.mkdir(parents=True, exist_ok=True)
# Candidates come from the shared registry in encoders.py -- pooling and prompt live in ONE place.
# An earlier version of this file kept its own copy of that table, which is how a comparison ends
# up silently running a mean-pooled model with CLS pooling.
CANDS = ("bge-base-en-v1.5", "bge-large-en-v1.5", "gte-large-en-v1.5", "stella-400M-v5",
         "arctic-embed-l")


# Pooling AND the post-pooling Dense both come from teacher.py, so the probe cannot drift from
# the encoder the rest of the pipeline uses.
from teacher import encode_batch, load_post_dense


@torch.no_grad()
def encode(spec, texts, prefix, tag):
    """Length-bucketed fp16 encode. Cached per (candidate, tag) so reruns are free."""
    maxlen = spec.max_length
    p = OUT / f"{spec.name}-{tag}.npy"
    if p.exists():
        return np.load(p)
    kw = {"trust_remote_code": True} if spec.trust_remote_code else {}
    tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision, **kw)
    model = AutoModel.from_pretrained(spec.repo, revision=spec.revision, dtype=torch.float16,
                                      **kw).cuda().eval()
    dense = load_post_dense(spec, "cuda")
    full = [prefix + t for t in texts]
    lens = [len(tok(t, add_special_tokens=True, truncation=True,
                    max_length=maxlen)["input_ids"]) for t in full]
    order = np.argsort(np.array(lens), kind="stable")
    dim = model.config.hidden_size
    out = np.empty((len(full), dim), dtype=np.float16)
    i, budget = 0, 24576  # tokens per batch; keeps a 435M model well inside 10 GB
    while i < len(order):
        n = max(1, min(256, budget // max(lens[order[i]], 1)))
        idx = order[i:i + n]
        out[idx] = encode_batch(tok, model, [full[j] for j in idx], maxlen, "cuda",
                                pooling=spec.pooling, dense=dense).astype(np.float16)
        i += n
    del model
    torch.cuda.empty_cache()
    np.save(p, out)
    return out


def main(names):
    res = {}
    for name in names:
        spec = encoders.get(name)
        per_comp = {}
        for c in COMPONENTS:
            doc_ids, doc_texts, q_ids, q_texts, qrels, _ = dev_eval.doc_vecs(c)
            dv = encode(spec, doc_texts, spec.doc_prefix, f"{c}-docs")
            qv = encode(spec, q_texts, spec.query_prefix, f"{c}-q")
            run = topk_ids_scores(torch.from_numpy(qv.astype(np.float32)), dv, doc_ids,
                                 k=10, chunk=250_000, qids=q_ids)
            pq = per_query_ndcg(run, qrels)
            per_comp[c] = pq
            print(f"  {name:20s} {c:20s} nDCG@10 {np.mean(list(pq.values())):.4f}", flush=True)
        macro = float(np.mean([np.mean(list(per_comp[c].values())) for c in COMPONENTS]))
        res[name] = {"repo": spec.repo, "pooling": spec.pooling,
                     "query_prefix": spec.query_prefix, "dim": int(dv.shape[1]),
                     "macro_cqadupstack": round(macro, 4),
                     "per_component": {c: round(float(np.mean(list(per_comp[c].values()))), 4)
                                       for c in COMPONENTS}}
        print(f"{name:20s} MACRO {macro:.4f}\n", flush=True)

    base = res.get("bge-base-en-v1.5", {}).get("macro_cqadupstack")
    if base:
        for k in res:
            res[k]["vs_current_teacher"] = round(res[k]["macro_cqadupstack"] - base, 4)
            res[k]["ratio_to_current_teacher"] = round(res[k]["macro_cqadupstack"] / base, 4)
    out = {"_note": "Candidate teacher ceilings measured on the two CQADupStack dev components "
                    "-- the only dev components on no candidate's disclosed training list, and "
                    "the nearest dev analogue to FiQA. Symmetric retrieval, each model's own "
                    "pooling and prompt, fp16, our harness. A SELECTION diagnostic on dev: not "
                    "a gate input, and the six-set is never read. Use this to rank teachers "
                    "instead of the loose MTEB->six map in results/m7_calibration.json.",
           "components": list(COMPONENTS), "candidates": res}
    (REPO / "results" / "m7_teacher_probe.json").write_text(json.dumps(out, indent=1))
    print("wrote results/m7_teacher_probe.json")


if __name__ == "__main__":
    main(sys.argv[1:] or list(CANDS))
