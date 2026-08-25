"""The three mandated table inits, plus the IDF init for the learned token weights.

  teacher    each vocab token forwarded through the frozen teacher in a query-shaped context
             ([CLS] <prefix tokens> <token> [SEP]), CLS-pooled and L2-normalized — the M1
             finding that LightRetriever's A2 ablation cost -11.2 BEIR when skipped
  input_emb  the teacher's own word-embedding matrix (control)
  random     N(0, 1/sqrt(dim)) (control)
"""
import json

import numpy as np
import torch

from _paths import WORK
from table import Preproc, get_tokenizer
from teacher import load_teacher

INIT = WORK / "init"
INIT.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def teacher_rows(pre: Preproc, batch=512, device="cuda"):
    tok = get_tokenizer()
    _, model = load_teacher(dtype=torch.float32, device=device)
    V = tok.vocab_size
    pre_ids = tok(pre.prefix, add_special_tokens=False)["input_ids"] if pre.prefix else []
    cls, sep = tok.cls_token_id, tok.sep_token_id
    out = np.empty((V, model.config.hidden_size), dtype=np.float32)
    for lo in range(0, V, batch):
        hi = min(lo + batch, V)
        seqs = [[cls] + pre_ids + [t] + [sep] for t in range(lo, hi)]
        ids = torch.tensor(seqs, dtype=torch.long, device=device)
        att = torch.ones_like(ids)
        h = model(input_ids=ids, attention_mask=att).last_hidden_state[:, 0]
        out[lo:hi] = torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()
    return out


def input_emb_rows(device="cuda"):
    _, model = load_teacher(dtype=torch.float32, device=device)
    return model.get_input_embeddings().weight.detach().float().cpu().numpy().copy()


def random_rows(vocab, dim, seed=0):
    return np.random.default_rng(seed).normal(0, 1 / np.sqrt(dim), (vocab, dim)).astype(np.float32)


def get_init(kind, pre: Preproc, vocab=None, dim=768):
    """Cached: work/init/<kind>[-<preproc fingerprint>].npy"""
    name = f"{kind}-{pre.fingerprint()}" if kind == "teacher" else kind
    p = INIT / f"{name}.npy"
    if p.exists():
        return np.load(p)
    if kind == "teacher":
        r = teacher_rows(pre)
    elif kind == "input_emb":
        r = input_emb_rows()
    elif kind == "random":
        r = random_rows(vocab or get_tokenizer().vocab_size, dim)
    else:
        raise KeyError(kind)
    np.save(p, r.astype(np.float32))
    return r


def idf_weights(doc_sample=400_000, seed=0, max_length=512):
    """Document-frequency IDF over a deterministic sample of the TRAIN doc stores."""
    p = INIT / f"idf-{doc_sample}-{seed}.npy"
    if p.exists():
        return np.load(p)
    import mix
    tok = get_tokenizer()
    texts = []
    for store in ("hotpotqa-corpus", "esci-prod", "squad-ctx", "mrtydi-docs", "fever-pos"):
        if not (WORK / "train" / "stores" / f"{store}.json").exists():
            continue
        _, t = mix.load_store(store)
        texts.append(t)
    rng = np.random.default_rng(seed)
    per = doc_sample // max(1, len(texts))
    pick = []
    for t in texts:
        idx = rng.choice(len(t), size=min(per, len(t)), replace=False)
        pick += [t[i] for i in idx]
    df = np.zeros(tok.vocab_size, dtype=np.int64)
    B = 2000
    for lo in range(0, len(pick), B):
        for ids in tok(pick[lo:lo + B], add_special_tokens=False, truncation=True,
                       max_length=max_length)["input_ids"]:
            df[list(set(ids))] += 1
    n = len(pick)
    idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)  # BM25-style, always > 0
    idf = np.clip(idf / max(1e-6, float(np.median(idf[df > 0]))), 0.05, 20.0)  # scale so median ~ 1
    np.save(p, idf)
    (INIT / "idf_meta.json").write_text(json.dumps({"n_docs_sampled": n, "seed": seed,
                                                    "formula": "log((N-df+0.5)/(df+0.5)+1), median-normalized, clipped [0.05,20]"}))
    return idf


if __name__ == "__main__":
    from table import NO_PREFIX, WITH_PREFIX
    for pre, lbl in [(NO_PREFIX, "no-prefix"), (WITH_PREFIX, "with-prefix")]:
        r = get_init("teacher", pre)
        print(f"teacher init {lbl}: {r.shape} norm[0]={np.linalg.norm(r[0]):.4f} "
              f"mean pairwise cos={float((r[:2000] @ r[:2000].T).mean()):.4f}", flush=True)
    for k in ("input_emb", "random"):
        r = get_init(k, NO_PREFIX)
        print(f"{k} init: {r.shape} row-norm mean={np.linalg.norm(r, axis=1).mean():.4f}", flush=True)
