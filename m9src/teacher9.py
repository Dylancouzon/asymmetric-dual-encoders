"""Challenger-teacher encoding for the M9 teacher screen.

The incumbent (stella-400M) keeps M7's frozen path (`m7src/teacher.py`) untouched — it produced
the document pool the whole project is built on. The two challengers need pooling M7's teacher
module does not implement (Qwen3-Embedding is last-token) and dense heads selected by their own
sentence-transformers configs, so they are encoded through `SentenceTransformer`, which honours
each repo's own `modules.json` / `1_Pooling` / prompt config rather than a re-implementation.

That choice is only safe if the ST path reproduces the frozen path where the two overlap, so
`parity_vs_frozen()` encodes the same texts with stella-400M both ways and reports min-cosine.
Run it before trusting any challenger number (m8/CODEMAP.md pitfall 19: a check whose input
cannot make it fail is not a check — this one can).
"""
import hashlib
import json

import numpy as np
import torch

import m9base
from m9base import WORK

ENC9 = WORK / "enc9"

# Pinned revisions: the local snapshot ids, recorded so a re-download cannot silently move.
TEACHERS = {
    "stella-400M-v5": {
        "repo": "NovaSearch/stella_en_400M_v5",
        "revision": "ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20",
        "dim": 1024, "frozen_path": True,
        "query_prompt": "Instruct: Given a web search query, retrieve relevant passages that "
                        "answer the query.\nQuery: ",
        "doc_prompt": "",
        # NewModel asserts on xformers unless both are off (m7src/encoders.py stella Spec)
        "config_kwargs": {"use_memory_efficient_attention": False, "unpad_inputs": False},
    },
    "stella-1.5B-v5": {
        "repo": "NovaSearch/stella_en_1.5B_v5",
        "revision": "7817065102fd9e1b031fe874e910c01f40b2f001",
        "dim": 1024, "frozen_path": False,
        # Its vendored modeling_qwen.py was written against an older transformers and calls
        # DynamicCache.get_usable_length, removed in 4.57. An embedding model has no use for a KV
        # cache, so turning it off avoids the legacy path entirely rather than patching a
        # third-party file in the HF module cache.
        "config_kwargs": {"use_cache": False},
        "query_prompt": "Instruct: Given a web search query, retrieve relevant passages that "
                        "answer the query.\nQuery: ",
        "doc_prompt": "",
    },
    "qwen3-embedding-0.6B": {
        "repo": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "dim": 1024, "frozen_path": False,
        # verbatim from the repo's config_sentence_transformers.json (no trailing space)
        "query_prompt": "Instruct: Given a web search query, retrieve relevant passages that "
                        "answer the query\nQuery:",
        "doc_prompt": "",
    },
}

_ST = {}


def _sha_texts(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()


def load_st(key):
    if key not in _ST:
        from sentence_transformers import SentenceTransformer
        spec = TEACHERS[key]
        kw = {}
        if spec.get("config_kwargs"):
            kw["config_kwargs"] = dict(spec["config_kwargs"])
        # bf16, not fp16. In fp16 the stella-1.5B forward produced NaN for 24 of 242,786 real
        # query texts (0.01%) -- an overflow in the attention path, invisible until a training
        # loss came back `nan`. bf16 has fp32's exponent range at the same memory cost.
        m = SentenceTransformer(spec["repo"], revision=spec["revision"],
                                trust_remote_code=True, device="cuda",
                                model_kwargs={"dtype": torch.bfloat16}, **kw)
        m.eval()
        got = m.get_sentence_embedding_dimension()
        assert got == spec["dim"], f"{key}: ST reports dim {got}, spec says {spec['dim']}"
        _ST[key] = m
    return _ST[key]


def release(key):
    m = _ST.pop(key, None)
    del m
    torch.cuda.empty_cache()


def encode(key, texts, role, batch_size=64, max_length=512, verbose=False):
    """-> (n, dim) fp32 unit-norm. `role` in {'query','doc'} picks the repo's own prompt."""
    spec = TEACHERS[key]
    m = load_st(key)
    m.max_seq_length = max_length
    prompt = spec["query_prompt"] if role == "query" else spec["doc_prompt"]
    v = m.encode(texts, prompt=prompt, batch_size=batch_size, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=verbose)
    v = np.asarray(v, dtype=np.float32)
    if not np.isfinite(v).all():
        n = int((~np.isfinite(v).all(axis=1)).sum())
        raise SystemExit(f"{key}: {n} of {len(texts)} encoded vectors are non-finite. A teacher "
                         f"target that is not a finite unit vector must never reach a trainer.")
    return v


def encode_cached(key, name, texts, role, batch_size=64, max_length=512, chunk=50_000,
                  verbose=True):
    """Chunk-resumable fp16 cache at work/enc9/<dir>/. Content-keyed on the exact text list, the
    repo, the revision, the role prompt and max_length, so any of them moving builds a new
    directory rather than reusing the wrong vectors."""
    import time

    spec = TEACHERS[key]
    blob = {"name": name, "repo": spec["repo"], "revision": spec["revision"], "role": role,
            "prompt": spec["query_prompt"] if role == "query" else spec["doc_prompt"],
            "max_length": max_length, "dim": spec["dim"], "store_dtype": "fp16",
            "corpus_sha256": _sha_texts(texts), "path": "sentence-transformers"}
    dkey = hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]
    d = ENC9 / f"{name}-{key}-{dkey}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(blob, indent=1))

    n, dim = len(texts), spec["dim"]
    nchunk = (n + chunk - 1) // chunk
    t0 = time.time()
    man_p = d / "chunks.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    for ci in range(nchunk):
        p = d / f"chunk_{ci:05d}.npy"
        cname = p.name          # NOT `key` -- that is the teacher id, and shadowing it made the
                                # next encode() call ask TEACHERS for 'chunk_00000.npy'
        if p.exists() and cname in man:
            # content hash, not shape: a truncated or half-written chunk has the right shape
            # (m8/CODEMAP.md's cache discipline; Codex pass 3, M15)
            if hashlib.sha256(p.read_bytes()).hexdigest() == man[cname]["sha256"]:
                continue
            print(f"  {cname}: content hash mismatch, re-encoding", flush=True)
        lo, hi = ci * chunk, min(n, (ci + 1) * chunk)
        v = encode(key, texts[lo:hi], role, batch_size=batch_size, max_length=max_length)
        np.save(p, v.astype(np.float16))
        man[p.name] = {"rows": int(hi - lo),
                       "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        man_p.write_text(json.dumps(man, indent=1))
        if verbose:
            done = hi
            el = time.time() - t0
            print(f"  {key}/{name}: {done:,}/{n:,} ({el:.0f}s, {done/max(el,1e-9):.0f}/s, "
                  f"eta {(n-done)/max(done/max(el,1e-9),1e-9)/60:.1f}m)", flush=True)

    comb = d / "combined.f16"
    if len(man) != nchunk:
        raise SystemExit(f"{d.name}: {len(man)} hashed chunks for {nchunk} expected -- refuse to "
                         f"stitch a cache whose parts are not all accounted for")
    if not comb.exists() or comb.stat().st_size != n * dim * 2:
        mm = np.memmap(comb, dtype=np.float16, mode="w+", shape=(n, dim))
        for ci in range(nchunk):
            lo, hi = ci * chunk, min(n, (ci + 1) * chunk)
            mm[lo:hi] = np.load(d / f"chunk_{ci:05d}.npy")
        mm.flush()
        del mm
    return np.memmap(comb, dtype=np.float16, mode="r", shape=(n, dim))


def parity_vs_frozen(n=64, seed=3):
    """Encode the same real texts with stella-400M through BOTH paths and report min-cosine.
    The ST path is only trustworthy for the challengers if it reproduces the frozen one here."""
    import teacher
    import data as m9data

    texts = json.loads((WORK / "m9_screen_queries.json").read_text())
    rng = np.random.default_rng(seed)
    sel = [texts[i] for i in rng.choice(len(texts), size=n, replace=False)]

    frozen = teacher.encode(sel, prefix=teacher.QUERY_PREFIX, max_length=512,
                            model_id=TEACHERS["stella-400M-v5"]["repo"],
                            revision=TEACHERS["stella-400M-v5"]["revision"],
                            dtype=torch.float32)
    frozen = frozen / np.linalg.norm(frozen, axis=1, keepdims=True)
    st = encode("stella-400M-v5", sel, "query")
    cos = (frozen * st).sum(1)
    out = {"n": n, "seed": seed, "min_cos": float(cos.min()), "mean_cos": float(cos.mean()),
           "max_abs": float(np.abs(frozen - st).max()),
           "_note": "frozen path = m7src/teacher.py fp32; ST path = SentenceTransformer fp16. "
                    "A gap here is dominated by dtype, not by pooling."}
    release("stella-400M-v5")
    del m9data
    return out


if __name__ == "__main__":
    print(json.dumps(parity_vs_frozen(), indent=1))
