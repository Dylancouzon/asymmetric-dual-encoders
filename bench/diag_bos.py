"""Diagnostic: does prepending the shipped tokenizer's <|bos|> to table rows fix the LR gap?
Partial tables over each dataset's query-token union only (exact for eval, ~50x cheaper)."""
import sys

import numpy as np
import torch

from core import evaluate, load_beir, load_vecs
from run_lightretriever import INSTRUCTIONS, SLUG, load_model, queries_from_table, query_token_ids

tok, model = load_model()
bos, eos = tok.bos_token_id, tok.eos_token_id
print("bos:", bos, "eos:", eos, flush=True)


@torch.no_grad()
def partial_table(instr, token_ids, with_bos, bs=500):
    prompt_ids = tok(f"Instruct: {instr}\nQuery: ", add_special_tokens=False)["input_ids"]
    head = ([bos] if with_bos else []) + prompt_ids
    rows = {}
    for s in range(0, len(token_ids), bs):
        chunk = token_ids[s : s + bs]
        batch = torch.tensor([head + [i, eos] for i in chunk], device="mps")
        h = model.model(input_ids=batch).last_hidden_state[:, -1].float().cpu().numpy()
        for j, i in enumerate(chunk):
            rows[i] = h[j]
        torch.mps.empty_cache()
    return rows


for ds in ["scifact", "arguana", "scidocs"]:
    doc_ids, doc_vecs = load_vecs(SLUG, ds, "doc")
    _, _, q_ids, q_texts, qrels = load_beir(ds)
    q_toks = query_token_ids(tok, q_texts)
    union = sorted({t for ids in q_toks for t in ids})
    for with_bos in [False, True]:
        rows = partial_table(INSTRUCTIONS[ds], union, with_bos)
        table = np.zeros((max(union) + 1, model.config.hidden_size), dtype=np.float32)
        for i, v in rows.items():
            table[i] = v
        qv = queries_from_table(table, q_toks)
        m = evaluate(doc_ids, doc_vecs, q_ids, qv, qrels)
        print(f"{ds:10s} bos={with_bos}: ndcg@10={m['ndcg@10']:.4f}", flush=True)
