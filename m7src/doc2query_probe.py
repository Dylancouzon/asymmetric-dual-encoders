"""Capacity lever #3, cheapest decisive form: does doc2query expansion help THIS architecture?

EXPLORED.md demoted-not-closed row: Weller et al. find expansion gain anti-correlates with
retriever strength, but which "strength" governs here — the strong frozen doc tower (harmed
regime) or the BM25-class end-to-end system (helped regime) — is untested for a
frozen-strong-doc-tower + bag-of-tokens-query architecture. This is the pre-registered cheap
test: the two CQADupStack dev components, N generated queries appended to each doc (appended,
not prepended: a doc already at the encoder's max length truncates the expansion away instead
of losing content, so the truncation confound only shrinks the treatment), re-encoded with the
frozen teacher, scored with the SAME winner-table query vectors, paired per query.

DIAGNOSTIC ONLY. The available generators are MS MARCO-trained and MS MARCO is excluded from
the clean stack — nothing from this run may ship. If the effect resolves positive, the clean-
generator question goes to Dylan and a shippable run gets its own pre-registration.
"""
import json
import sys
import time

import numpy as np
import torch

import boot
import devsuite
from _paths import REPO, WORK
from evalkit import per_query_ndcg, topk_ids_scores
from table import NO_PREFIX, load_table
from teacher import encode_cached
from bigram_residual import WINNER

GEN_ID = "BeIR/query-gen-msmarco-t5-base-v1"
COMPONENTS = ("cqadup-programmers", "cqadup-physics")
N_Q = 5
D2Q = WORK / "d2q"


def generate(comp, doc_texts, batch=64, max_in=256, max_out=48):
    """Shard-resumable sampling generation; one JSONL line per doc, in corpus order."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    D2Q.mkdir(parents=True, exist_ok=True)
    p = D2Q / f"{comp}-n{N_Q}.jsonl"
    done = sum(1 for _ in open(p)) if p.exists() else 0
    if done >= len(doc_texts):
        return [json.loads(l) for l in open(p)]
    tok = AutoTokenizer.from_pretrained(GEN_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_ID, torch_dtype=torch.float16).cuda().eval()
    t0 = time.time()
    with open(p, "a") as f, torch.inference_mode():
        for lo in range(done, len(doc_texts), batch):
            enc = tok(doc_texts[lo:lo + batch], truncation=True, max_length=max_in,
                      padding=True, return_tensors="pt").to("cuda")
            torch.manual_seed(lo)  # deterministic under resume: seed keyed to corpus position
            out = model.generate(**enc, max_length=max_out, do_sample=True, top_k=10,
                                 num_return_sequences=N_Q)
            qs = tok.batch_decode(out, skip_special_tokens=True)
            for i in range(len(enc["input_ids"])):
                f.write(json.dumps(qs[i * N_Q:(i + 1) * N_Q]) + "\n")
            if (lo // batch) % 20 == 0:
                r = (lo + batch - done) / max(time.time() - t0, 1e-9)
                print(f"  {comp} d2q {lo+batch}/{len(doc_texts)} ({r:.0f} docs/s)", flush=True)
    del model
    torch.cuda.empty_cache()
    return [json.loads(l) for l in open(p)]


def main():
    t0 = time.time()
    m = load_table(WINNER, variant="fp16", device="cuda")
    out = {"generator": GEN_ID, "n_queries_per_doc": N_Q, "winner": WINNER.name,
           "per_component": {}, "_label": "DIAGNOSTIC ONLY — MS MARCO-trained generator, "
           "excluded from the clean stack; nothing here ships. Positive resolution escalates "
           "to Dylan for a clean-generator ruling, not to adoption."}
    per_orig, per_exp = {}, {}
    for comp in COMPONENTS:
        doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(comp)
        expansions = generate(comp, doc_texts)
        assert len(expansions) == len(doc_texts)
        exp_texts = [(t + " " + " ".join(e)).strip() for t, e in zip(doc_texts, expansions)]
        dv_o = encode_cached(f"dev-{comp}-docs", doc_texts, prefix="", dtype=torch.float16,
                             verbose=False)
        dv_e = encode_cached(f"dev-{comp}-docs-d2q{N_Q}", exp_texts, prefix="",
                             dtype=torch.float16)
        qv = m.encode(q_texts, NO_PREFIX)
        for tag, dv, per in (("orig", dv_o, per_orig), ("d2q", dv_e, per_exp)):
            run = topk_ids_scores(qv, dv, doc_ids, k=100, chunk=200_000, qids=q_ids)
            per[comp] = per_query_ndcg(run, qrels)
        out["per_component"][comp] = {
            "orig": round(float(np.mean(list(per_orig[comp].values()))), 4),
            "d2q": round(float(np.mean(list(per_exp[comp].values()))), 4)}
        print(f"  {comp}: {out['per_component'][comp]}", flush=True)
    r = boot.paired(per_exp, per_orig, alternative="two-sided")
    sf = boot.signflip(per_exp, per_orig, alternative="two-sided")
    out["d2q_minus_orig"] = {"delta": r["delta"], "ci95": r["ci95"], "resolved": r["resolved"],
                             "signflip_p": sf["p"], "signflip_p_str": sf["p_str"]}
    out["seconds"] = round(time.time() - t0, 1)
    (REPO / "results" / "m7_doc2query_probe.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
