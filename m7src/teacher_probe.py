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
import devsuite
from _paths import REPO, WORK
from evalkit import per_query_ndcg, topk_ids_scores

COMPONENTS = ("cqadup-programmers", "cqadup-physics")
OUT = WORK / "teacherprobe"
OUT.mkdir(parents=True, exist_ok=True)

# name -> (repo, revision, pooling, query_prefix, doc_prefix, trust_remote_code, max_len)
# Pooling and prompt come from each model card; getting either wrong understates the model, so
# they are stated per candidate rather than inherited from the bge convention.
CANDS = {
    "bge-base-en-v1.5": (
        "BAAI/bge-base-en-v1.5", "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a", "cls",
        "Represent this sentence for searching relevant passages: ", "", False, 512),
    "bge-large-en-v1.5": (
        "BAAI/bge-large-en-v1.5", None, "cls",
        "Represent this sentence for searching relevant passages: ", "", False, 512),
    "gte-large-en-v1.5": (
        "Alibaba-NLP/gte-large-en-v1.5", None, "cls", "", "", True, 512),
    "stella_en_400M_v5": (
        "NovaSearch/stella_en_400M_v5", None, "mean",
        "Instruct: Given a web search query, retrieve relevant passages that answer the query.\n"
        "Query: ", "", True, 512),
    "arctic-embed-l": (
        "Snowflake/snowflake-arctic-embed-l", None, "cls",
        "Represent this sentence for searching relevant passages: ", "", False, 512),
}


def pool(h, mask, mode):
    if mode == "cls":
        return h[:, 0]
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)


@torch.no_grad()
def encode(name, texts, prefix, tag):
    """Length-bucketed fp16 encode. Cached per (candidate, tag) so reruns are free."""
    repo, rev, mode, _, _, trc, maxlen = CANDS[name]
    p = OUT / f"{name}-{tag}.npy"
    if p.exists():
        return np.load(p)
    kw = {"trust_remote_code": True} if trc else {}
    tok = AutoTokenizer.from_pretrained(repo, revision=rev, **kw)
    model = AutoModel.from_pretrained(repo, revision=rev, dtype=torch.float16,
                                      **kw).cuda().eval()
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
        b = tok([full[j] for j in idx], padding=True, truncation=True, max_length=maxlen,
                return_tensors="pt").to("cuda")
        h = model(**b).last_hidden_state
        v = F.normalize(pool(h, b["attention_mask"], mode).float(), dim=-1)
        out[idx] = v.cpu().numpy().astype(np.float16)
        i += n
    del model
    torch.cuda.empty_cache()
    np.save(p, out)
    return out


def main(names):
    res = {}
    for name in names:
        repo, rev, mode, qpfx, dpfx, trc, _ = CANDS[name]
        per_comp = {}
        for c in COMPONENTS:
            doc_ids, doc_texts, q_ids, q_texts, qrels, _ = dev_eval.doc_vecs(c)
            dv = encode(name, doc_texts, dpfx, f"{c}-docs")
            qv = encode(name, q_texts, qpfx, f"{c}-q")
            run = topk_ids_scores(torch.from_numpy(qv.astype(np.float32)), dv, doc_ids,
                                 k=10, chunk=250_000, qids=q_ids)
            pq = per_query_ndcg(run, qrels)
            per_comp[c] = pq
            print(f"  {name:20s} {c:20s} nDCG@10 {np.mean(list(pq.values())):.4f}", flush=True)
        macro = float(np.mean([np.mean(list(per_comp[c].values())) for c in COMPONENTS]))
        res[name] = {"repo": repo, "pooling": mode, "query_prefix": qpfx, "dim": int(dv.shape[1]),
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
