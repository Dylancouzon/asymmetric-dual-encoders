"""M9 ablation: sparse doc rep with vs without masking pad positions before amax.
The paper's reference notebook doesn't mask; our main run does. Same forward pass, both variants."""
import json
from collections import Counter

import numpy as np
import torch

from core import evaluate, load_beir, score_run, topk_run
from run_lightretriever import LR_DIR, load_model, query_token_ids

tok, model = load_model()
ds = "scifact"
doc_ids, doc_texts, q_ids, q_texts, qrels = load_beir(ds)
cols = json.loads((LR_DIR / ds / "sparse_cols.json").read_text())
col_pos = {c: i for i, c in enumerate(cols)}
w_sub = model.lm_head.weight[torch.tensor(cols)].to("mps")

enc = tok(doc_texts, add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
order = np.argsort([-len(e) for e in enc])
masked = np.zeros((len(doc_texts), len(cols)), dtype=np.float32)
unmasked = np.zeros_like(masked)
i = 0
with torch.no_grad():
    while i < len(order):
        L = len(enc[order[i]])
        bs = min(32, max(1, 4096 // max(L, 1)))
        idx = order[i : i + bs]
        batch = tok.pad({"input_ids": [enc[j] for j in idx]}, return_tensors="pt").to("mps")
        hidden = model.model(**batch).last_hidden_state
        lg = hidden @ w_sub.T
        unmasked[idx] = torch.log1p(torch.relu(lg.amax(1).float())).cpu().numpy()
        lg = lg.masked_fill(~batch["attention_mask"].bool().unsqueeze(-1), -torch.inf)
        masked[idx] = torch.log1p(torch.relu(lg.amax(1).float())).cpu().numpy()
        i += bs
        if (i // 1000) != ((i - bs) // 1000):
            torch.mps.empty_cache()

q_toks = query_token_ids(tok, q_texts)
qs = np.zeros((len(q_ids), len(cols)), dtype=np.float32)
for qi, ids in enumerate(q_toks):
    for t, c in Counter(ids).items():
        if t in col_pos:
            qs[qi, col_pos[t]] = c
for name, mat in [("masked", masked), ("unmasked", unmasked)]:
    m = score_run(topk_run(doc_ids, qs @ mat.T, q_ids), qrels)
    print(f"sparse {ds} {name}: ndcg@10={m['ndcg@10']:.4f} (paper: 0.664)", flush=True)
