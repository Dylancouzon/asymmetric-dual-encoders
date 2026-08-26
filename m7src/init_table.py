"""The three mandated table inits, plus the IDF init for the learned token weights.

  teacher    each vocab token forwarded through the frozen teacher in a query-shaped context
             ([CLS] <prefix tokens> <token> [SEP]), CLS-pooled and L2-normalized — the M1
             finding that LightRetriever's A2 ablation cost -11.2 BEIR when skipped
  input_emb  the teacher's own word-embedding matrix (control)
  random     N(0, 1/sqrt(dim)) (control)
"""
import json

import numpy as np

import encoders
import torch

from _paths import WORK
from table import Preproc, get_tokenizer
from teacher import load_post_dense, load_teacher, pool_project_normalize

INIT = WORK / "init"
INIT.mkdir(parents=True, exist_ok=True)


def spec_tag():
    """Encoder identity for a cache path. work/init/ was keyed on kind + preprocessing only, so a
    1024-d arctic run would have loaded bge-base's 768-d rows -- or, on a same-width swap, silently
    trained against another teacher's geometry with every shape check passing."""
    sp = encoders.active()
    return f"{sp.name}-{(sp.revision or 'norev')[:8]}"


@torch.no_grad()
def teacher_rows(pre: Preproc, batch=512, device="cuda"):
    """Each vocab token forwarded through the frozen teacher in a query-shaped context.

    Pooling and the post-pooling Dense come from the registry via pool_project_normalize. This
    function used to take last_hidden_state[:, 0] unconditionally and size itself from
    config.hidden_size, so for a mean-pooled teacher it built the table's rows with a DIFFERENT
    read-out from the one that encodes the documents those rows get scored against.
    """
    sp = encoders.active()
    tok = get_tokenizer()
    _, model = load_teacher(dtype=torch.float32, device=device)
    dense = load_post_dense(sp, device)
    V = tok.vocab_size
    pre_ids = tok(pre.prefix, add_special_tokens=False)["input_ids"] if pre.prefix else []
    cls, sep = tok.cls_token_id, tok.sep_token_id
    out = np.empty((V, sp.dim), dtype=np.float32)
    for lo in range(0, V, batch):
        hi = min(lo + batch, V)
        seqs = [[cls] + pre_ids + [t] + [sep] for t in range(lo, hi)]
        ids = torch.tensor(seqs, dtype=torch.long, device=device)
        att = torch.ones_like(ids)
        h = model(input_ids=ids, attention_mask=att).last_hidden_state
        v = pool_project_normalize(h, att, sp.pooling, dense)
        if v.shape[1] != sp.dim:
            raise AssertionError(f"{sp.name}: init rows are {v.shape[1]}-d but Spec.dim is {sp.dim}")
        out[lo:hi] = v.cpu().numpy()
    return out


def input_emb_rows(device="cuda"):
    _, model = load_teacher(dtype=torch.float32, device=device)
    return model.get_input_embeddings().weight.detach().float().cpu().numpy().copy()


def run_token_weights(rid):
    """The trained per-token weights saved alongside a run's rows, or None if it had none.

    A checkpoint init that restored rows but re-derived the weights from IDF would not be the
    system whose dev score is being used as the starting point -- the weights are part of the
    table, and p1-objB's learned ones are measurably IDF-LIKE but not IDF.
    """
    z = np.load(WORK / "runs" / f"{rid}.npz")
    w = z["token_weights"]
    return None if w.size == 0 else w.astype(np.float32)


def random_rows(vocab, dim, seed=0):
    return np.random.default_rng(seed).normal(0, 1 / np.sqrt(dim), (vocab, dim)).astype(np.float32)


def get_init(kind, pre: Preproc, vocab=None, dim=None):
    """Cached: work/init/<kind>[-<preproc fingerprint>].npy"""
    # Every init depends on the encoder, including `input_emb` (its embedding matrix) and `random`
    # (its width), so the tag is unconditional rather than "only for the teacher init".
    name = f"{spec_tag()}-{kind}-{pre.fingerprint()}" if kind == "teacher" \
        else f"{spec_tag()}-{kind}"
    p = INIT / f"{name}.npy"
    # "run:<run_id>" starts from a previously trained table instead of an init. It exists so the
    # contrastive phase can be tested from a FIXED starting table: comparing objective-C arms at a
    # matched step budget across a 60x learning-rate range compares tables that reached completely
    # different quality in the B phase (0.2731 at lr 5e-5 vs 0.4449 at 3e-3, both at 4k steps), so
    # any A-phase delta is confounded with how far B got. Not cached under work/init -- the table
    # it names is already on disk, and copying it would just risk the two diverging.
    if kind.startswith("run:"):
        rid = kind.split(":", 1)[1]
        npz = WORK / "runs" / f"{rid}.npz"
        if not npz.exists():
            raise FileNotFoundError(f"init 'run:{rid}' needs {npz}, which does not exist")
        z = np.load(npz)
        # rows_fp16 is what the artifact actually stores, so the A phase starts from the table that
        # would ship, not from an fp32 shadow of it. The difference is far below the dev resolution.
        rows = z["rows_fp16"].astype(np.float32)
        meta = json.loads((WORK / "runs" / f"{rid}.meta.json").read_text())
        if meta.get("preproc_fingerprint") not in (None, pre.fingerprint()):
            raise AssertionError(
                f"init 'run:{rid}' was trained under preprocessing "
                f"{meta['preproc_fingerprint']} but this run uses {pre.fingerprint()}")
        if meta.get("teacher") not in (None, encoders.active().repo):
            raise AssertionError(f"init 'run:{rid}' was trained against {meta['teacher']} but the "
                                 f"active encoder is {encoders.active().repo}")
        return rows
    if p.exists():
        return np.load(p)
    if kind == "teacher":
        r = teacher_rows(pre)
    elif kind == "input_emb":
        r = input_emb_rows()
    elif kind == "random":
        r = random_rows(vocab or get_tokenizer().vocab_size,
                        encoders.active().dim if dim is None else dim)
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
