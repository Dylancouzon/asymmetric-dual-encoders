"""LightRetriever (qwen2.5-1.5b) asymmetric inference on the BEIR subset.

Stages (run separately, each resumable via cached artifacts):
  python bench/run_lightretriever.py docs            # encode all corpora (dense + sparse)
  python bench/run_lightretriever.py tables          # build per-task + websearch lookup tables
  python bench/run_lightretriever.py eval            # dense / sparse / hybrid metrics

Conventions verified against github.com/caskcsg/lightretriever (see research/lightretriever.md):
- docs: no instruction, add_special_tokens=True, max_len 512, last-token pooling, L2 normalize
- table row i: forward [prompt_ids] + [i] + [eos] through base transformer, take EOS hidden state
- query dense: tokenize(add_special_tokens=False) -> table mean -> L2 normalize
- doc sparse: log1p(relu(amax_over_seq(logits))), padding masked out; query sparse: token counts
- hybrid: per-query min-max fusion, 0.7 dense + 0.3 sparse
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from core import ARTIFACTS, DATASETS, evaluate, fuse_linear, load_beir, load_vecs, record, save_vecs, score_run, topk_run

ADAPTER = "lightretriever/lightretriever-qwen2.5-1.5b"
BASE = "Qwen/Qwen2.5-1.5B"
SLUG = "lightretriever-qwen2.5-1.5b"
DEVICE = "mps"
INSTRUCTIONS = {
    "scifact": "Given a scientific claim, retrieve documents that support or refute the claim",
    "nfcorpus": "Given a question, retrieve relevant documents that best answer the question",
    "fiqa": "Given a financial question, retrieve user replies that best answer the question",
    "arguana": "Given a claim, find documents that refute the claim",
    "scidocs": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper",
    "trec-covid": "Given a query on COVID-19, retrieve documents that answer the query",
    "websearch": "Given a web search query, retrieve relevant passages that answer the query",
}
LR_DIR = ARTIFACTS / SLUG


def load_model():
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    tok.padding_side = "right"  # last-token pooling assumes right padding; adapter config leaves it null
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
    model.eval().to(DEVICE)
    return tok, model


def query_token_ids(tok, texts):
    return [tok(t, add_special_tokens=False, truncation=True, max_length=512)["input_ids"] for t in texts]


def sparse_query_union(tok, q_texts):
    """Union of one dataset's test-query token ids -> sorted column list."""
    union = set()
    for t in query_token_ids(tok, q_texts):
        union.update(t)
    return sorted(union)


@torch.no_grad()
def encode_docs(tok, model, texts, sparse_cols, bs_tokens=4096, max_bs=32):
    """Returns (dense [N,H] fp32 normalized, sparse [N,K] fp32 restricted to sparse_cols).

    Sparse logits are computed only for the query-token columns (hidden @ W_head[cols].T),
    which is exact for scoring and avoids materializing full-vocab logits.
    """
    enc = tok(texts, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
    order = np.argsort([-len(e) for e in enc])
    w_sub = model.lm_head.weight[torch.tensor(sparse_cols)].to(DEVICE)  # [K, H] bf16
    dense_out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    sparse_out = np.zeros((len(texts), len(sparse_cols)), dtype=np.float32)
    i, t0, done, nb = 0, time.time(), 0, 0
    while i < len(order):
        L = len(enc[order[i]])
        bs = min(max_bs, max(1, bs_tokens // max(L, 1)))
        idx = order[i : i + bs]
        batch = tok.pad({"input_ids": [enc[j] for j in idx]}, return_tensors="pt").to(DEVICE)
        hidden = model.model(**batch).last_hidden_state  # [B, L, H] bf16
        mask = batch["attention_mask"]
        last = mask.sum(1) - 1
        dvec = hidden[torch.arange(hidden.shape[0]), last]
        dvec = torch.nn.functional.normalize(dvec.float(), p=2, dim=-1)
        dense_out[idx] = dvec.cpu().numpy()
        lg = hidden @ w_sub.T  # [B, L, K] bf16
        lg.masked_fill_(~mask.bool().unsqueeze(-1), -torch.inf)
        sp = torch.log1p(torch.relu(lg.amax(1).float()))
        sparse_out[idx] = sp.cpu().numpy()
        i += bs
        done += len(idx)
        nb += 1
        if nb % 100 == 0:
            torch.mps.empty_cache()
        if done % 2000 < bs:
            print(f"  {done}/{len(texts)} docs, {done / (time.time() - t0):.1f} docs/s", flush=True)
    return dense_out, sparse_out


@torch.no_grad()
def build_table(tok, model, instruction, bs=1000):
    """Row i = hidden state at EOS of [bos] + prompt + [i] + [eos].

    The shipped adapter tokenizer bakes <|bos|>...<|endoftext|> into add_special_tokens=True,
    so the reference's bos check fires for this Qwen too. Building WITHOUT bos costs
    -2 to -12 nDCG per dataset (see diag_bos.py results in CLAUDE.md).
    """
    prompt_ids = tok(f"Instruct: {instruction}\nQuery: ", add_special_tokens=False)["input_ids"]
    head = [tok.bos_token_id] + prompt_ids
    eos = tok.eos_token_id
    V, H = len(tok), model.config.hidden_size
    table = np.zeros((V, H), dtype=np.float32)
    t0 = time.time()
    for start in range(0, V, bs):
        ids = list(range(start, min(start + bs, V)))
        batch = torch.tensor([head + [i, eos] for i in ids], device=DEVICE)
        hidden = model.model(input_ids=batch).last_hidden_state
        table[ids] = hidden[:, -1].float().cpu().numpy()
    print(f"  table built in {time.time() - t0:.0f}s ({V} rows)", flush=True)
    return table, time.time() - t0


def queries_from_table(table, tok_ids_list):
    out = np.zeros((len(tok_ids_list), table.shape[1]), dtype=np.float32)
    for i, ids in enumerate(tok_ids_list):
        v = table[ids].mean(0) if ids else np.zeros(table.shape[1], dtype=np.float32)
        out[i] = v / (np.linalg.norm(v) + 1e-12)
    return out


def stage_docs():
    tok, model = load_model()
    LR_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        if (LR_DIR / ds / "doc_vecs.npy").exists() and (LR_DIR / ds / "doc_sparse.npy").exists():
            print(f"skip {ds} (cached)", flush=True)
            continue
        doc_ids, doc_texts, q_ids, q_texts, _ = load_beir(ds)
        sparse_cols = sparse_query_union(tok, q_texts)
        (LR_DIR / ds).mkdir(parents=True, exist_ok=True)
        (LR_DIR / ds / "sparse_cols.json").write_text(json.dumps(sparse_cols))
        print(f"encoding {ds}: {len(doc_texts)} docs, {len(sparse_cols)} sparse cols", flush=True)
        dense, sparse = encode_docs(tok, model, doc_texts, sparse_cols)
        save_vecs(SLUG, ds, "doc", doc_ids, dense)
        np.save(LR_DIR / ds / "doc_sparse.npy", sparse.astype(np.float16))


def stage_tables():
    tok, model = load_model()
    timings = {}
    for name, instr in INSTRUCTIONS.items():
        p = LR_DIR / f"table_{name}.npy"
        if p.exists():
            print(f"skip table {name} (cached)", flush=True)
            continue
        print(f"building table: {name}", flush=True)
        table, secs = build_table(tok, model, instr)
        np.save(p, table.astype(np.float16))
        timings[name] = secs
    tpath = LR_DIR / "table_timings.json"
    old = json.loads(tpath.read_text()) if tpath.exists() else {}
    tpath.write_text(json.dumps({**old, **timings}, indent=1))


def stage_eval():
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    table_web = np.load(LR_DIR / "table_websearch.npy").astype(np.float32)
    for ds in DATASETS:
        doc_ids, _, q_ids, q_texts, qrels = load_beir(ds)
        sparse_cols = json.loads((LR_DIR / ds / "sparse_cols.json").read_text())
        col_pos = {c: i for i, c in enumerate(sparse_cols)}
        doc_ids, dense_docs = load_vecs(SLUG, ds, "doc")
        sparse_docs = np.load(LR_DIR / ds / "doc_sparse.npy").astype(np.float32)
        q_toks = query_token_ids(tok, q_texts)
        # dense, per-task table
        table = np.load(LR_DIR / f"table_{ds}.npy").astype(np.float32)
        qv_task = queries_from_table(table, q_toks)
        record(f"{SLUG}-dense", ds, evaluate(doc_ids, dense_docs, q_ids, qv_task, qrels))
        # int8-quantized per-task table (edge size lever): per-row absmax scale
        scale = np.abs(table).max(1, keepdims=True) / 127.0
        tq = np.round(table / np.maximum(scale, 1e-12)).clip(-127, 127) * scale
        record(f"{SLUG}-dense-int8table", ds, evaluate(doc_ids, dense_docs, q_ids, queries_from_table(tq, q_toks), qrels))
        # dense, single websearch table (deployment-realistic)
        qv_web = queries_from_table(table_web, q_toks)
        record(f"{SLUG}-dense-websearch", ds, evaluate(doc_ids, dense_docs, q_ids, qv_web, qrels))
        # sparse: counts @ restricted doc matrix
        qs = np.zeros((len(q_ids), len(sparse_cols)), dtype=np.float32)
        for i, ids in enumerate(q_toks):
            for t, c in Counter(ids).items():
                qs[i, col_pos[t]] = c
        sims_sp = qs @ sparse_docs.T
        run_sp = topk_run(doc_ids, sims_sp, q_ids)
        record(f"{SLUG}-sparse", ds, score_run(run_sp, qrels))
        # hybrid (per-task dense + sparse)
        sims_d = qv_task.astype(np.float32) @ np.asarray(dense_docs, dtype=np.float32).T
        run_d = topk_run(doc_ids, sims_d, q_ids)
        record(f"{SLUG}-hybrid", ds, score_run(fuse_linear(run_d, run_sp), qrels))
        # hybrid with websearch table
        sims_dw = qv_web @ np.asarray(dense_docs, dtype=np.float32).T
        record(f"{SLUG}-hybrid-websearch", ds, score_run(fuse_linear(topk_run(doc_ids, sims_dw, q_ids), run_sp), qrels))


if __name__ == "__main__":
    {"docs": stage_docs, "tables": stage_tables, "eval": stage_eval}[sys.argv[1]]()
