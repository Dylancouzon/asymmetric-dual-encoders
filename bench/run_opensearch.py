"""OpenSearch inference-free sparse doc encoder (doc-v3-gte) on the BEIR subset.

Doc side: MaskedLM logits -> max over non-pad positions -> log1p(relu) -> zero special tokens.
Query side: unique token ids weighted by the repo's shipped IDF table. No neural query compute.
Convention source: the model card's own example code (verify_card() prints it for cross-check).
"""
import json
import sys
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

from core import ARTIFACTS, DATASETS, load_beir, record, score_run, topk_run

MODEL_ID = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
SLUG = "opensearch-doc-v3-gte"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OS_DIR = ARTIFACTS / SLUG


def load_idf(tok):
    path = hf_hub_download(MODEL_ID, "idf.json")
    idf = json.load(open(path))
    if all(k.isdigit() for k in idf):  # keyed by token id
        out = {int(k): v for k, v in idf.items()}
    else:  # keyed by token string
        out = {}
        unk = tok.unk_token_id
        for k, v in idf.items():
            i = tok.convert_tokens_to_ids(k)
            if i != unk and i is not None:
                out[i] = v
    print(f"idf: {len(idf)} keys, {len(out)} resolved to token ids", flush=True)
    return out


@torch.no_grad()
def run():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    # MUST run under .venv-os (transformers 4.x): under transformers 5.x the remote code
    # leaves lm_head.decoder.weight untied/random and breaks rope position_ids.
    # eager attention: sdpa hits an MPS indexing bug in this remote code.
    import transformers
    assert transformers.__version__.startswith("4."), "run with .venv-os/bin/python (transformers 4.x)"
    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True,
        code_revision="40ced75c3017eb27626c9d4ea981bde21a2662f4",  # pinned in the model card
        attn_implementation="eager",
    )
    model.eval().to(DEVICE)
    idf = load_idf(tok)
    special = list(tok.all_special_ids)
    for ds in DATASETS:
        doc_ids, doc_texts, q_ids, q_texts, qrels = load_beir(ds)
        # per-dataset query token union -> restricted columns
        q_tok = [tok(t, add_special_tokens=False, truncation=True, max_length=512)["input_ids"] for t in q_texts]
        cols = sorted({t for ids in q_tok for t in ids} - set(special))
        col_pos = {c: i for i, c in enumerate(cols)}
        d = OS_DIR / ds
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "doc_sparse.npy").exists():
            enc = tok(doc_texts, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
            order = np.argsort([-len(e) for e in enc])
            out = np.zeros((len(doc_texts), len(cols)), dtype=np.float32)
            cols_t = torch.tensor(cols)
            i, t0, nb = 0, time.time(), 0
            while i < len(order):
                L = len(enc[order[i]])
                bs = min(32, max(1, 4096 // max(L, 1)))
                idx = order[i : i + bs]
                batch = tok.pad({"input_ids": [enc[j] for j in idx]}, return_tensors="pt").to(DEVICE)
                logits = model(**batch).logits  # [B, L, V]
                logits = logits * batch["attention_mask"].unsqueeze(-1)
                # v3 activation is double log saturation (per model card), not v1/v2's single log1p
                vals = torch.log1p(torch.log1p(torch.relu(logits.amax(1).float())))
                out[idx] = vals[:, cols_t].cpu().numpy()
                i += bs
                nb += 1
                if nb % 50 == 0:
                    torch.mps.empty_cache()
                if (i // 2000) != ((i - bs) // 2000):
                    print(f"  {i}/{len(order)} docs, {i / (time.time() - t0):.1f} docs/s", flush=True)
            np.save(d / "doc_sparse.npy", out.astype(np.float16))
            (d / "sparse_cols.json").write_text(json.dumps(cols))
        docs = np.load(d / "doc_sparse.npy").astype(np.float32)
        cached_cols = json.loads((d / "sparse_cols.json").read_text())
        assert cached_cols == cols, f"{ds}: cached sparse_cols differ from current query set — delete {d} and rerun"
        col_pos = {c: i for i, c in enumerate(cols)}
        missing = sum(1 for c in cols if c not in idf)
        print(f"{ds}: {missing}/{len(cols)} query tokens have no idf entry (weight 0, per model card)", flush=True)
        qv = np.zeros((len(q_ids), len(cols)), dtype=np.float32)
        for i, ids in enumerate(q_tok):
            for t in set(ids):
                if t in col_pos:
                    qv[i, col_pos[t]] = idf.get(t, 0.0)  # card default: absent from idf.json = 0
        record(SLUG, ds, score_run(topk_run(doc_ids, qv @ docs.T, q_ids), qrels))


def verify_card():
    path = hf_hub_download(MODEL_ID, "README.md")
    text = open(path).read()
    start = text.find("```python")
    print(text[start : start + 3000])


if __name__ == "__main__":
    {"run": run, "card": verify_card}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
