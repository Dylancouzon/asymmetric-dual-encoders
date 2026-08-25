"""Full-vocabulary sparse doc-side footprint for opensearch-doc-v3-gte (Codex major 7).

The quality eval used query-restricted columns (exact for scoring, wrong for index sizing).
This measures true nnz/doc over the full 30522 vocab on a 5,000-doc FiQA sample and
extrapolates postings size. Run with .venv-os (transformers 4.x).
"""
import json

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from core import REPO, load_beir

MID = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte"
N = 5000
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForMaskedLM.from_pretrained(
    MID, torch_dtype=torch.bfloat16, trust_remote_code=True,
    code_revision="40ced75c3017eb27626c9d4ea981bde21a2662f4", attn_implementation="eager",
).eval().to(DEV)
special = torch.tensor(list(tok.all_special_ids))

_, doc_texts, *_ = load_beir("fiqa")
rng = np.random.default_rng(0)
sample = [doc_texts[i] for i in rng.choice(len(doc_texts), N, replace=False)]
enc = tok(sample, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
order = np.argsort([-len(e) for e in enc])
nnz = np.zeros(N, dtype=np.int64)
i = 0
with torch.no_grad():
    while i < len(order):
        L = len(enc[order[i]])
        bs = min(16, max(1, 2048 // max(L, 1)))
        idx = order[i : i + bs]
        batch = tok.pad({"input_ids": [enc[j] for j in idx]}, return_tensors="pt").to(DEV)
        logits = model(**batch).logits * batch["attention_mask"].unsqueeze(-1)
        vals = torch.log1p(torch.log1p(torch.relu(logits.amax(1).float())))
        vals[:, special] = 0
        nnz[idx] = (vals > 0).sum(1).cpu().numpy()
        i += bs
        if i % 500 < bs:
            torch.mps.empty_cache()
            print(f"{i}/{N}", flush=True)

stats = {
    "n_docs_sampled": N,
    "nnz_mean": float(nnz.mean()),
    "nnz_p50": float(np.percentile(nnz, 50)),
    "nnz_p95": float(np.percentile(nnz, 95)),
    "bytes_per_doc_at_6B_per_posting": float(nnz.mean() * 6),
    "index_mb_extrapolated_fiqa_57638": round(nnz.mean() * 6 * 57638 / 1e6, 1),
    "index_mb_extrapolated_per_million_docs": round(nnz.mean() * 6 * 1e6 / 1e6, 1),
}
(REPO / "results" / "opensearch_index_cost.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
