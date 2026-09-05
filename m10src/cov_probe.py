"""COV probe encoding and per-query scoring — the surface's resolution number (§Surfaces).

Two models that are candidates in **no** M10 family (`intfloat/e5-small-v2`,
`thenlper/gte-small`) are encoded fresh over the admitted COV surface and scored symmetrically.
The mandate records only the DISTANCE between the point estimate and the one-sided 0.025/13
lower bound of their COV-macro difference — never which of them led — so this module returns
per-query nDCG@10 keyed by an opaque probe name and the caller never labels a direction.

Why not `m9src/teacher9`: it is a `guard9` "challenger"-scope file and may not be edited before
M9's close-out. The cache here is the same shape (content-keyed on the exact text list, repo,
revision, prompt and max_length) and deliberately smaller — these corpora fit memory.

Scoring UNITS, not components: BRIGHT's six slices each have their own corpus, so a slice is the
retrieval unit and the family macro averages slices within the family (§Surfaces).
"""
import hashlib, json, os, sys, time
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m10src"):
    sys.path.insert(0, str(REPO / p))
OUT = REPO / "work" / "m10cov"
CACHE = OUT / "probe"

import numpy as np

from cov_admit import COMPONENTS, BRIGHT_SLICES

# Pinned revisions. Both are 33M / 384d, both outside every M10 family (F is bge-small,
# MiniLM-L6, MiniLM-L12). Prompts are each repo's own documented protocol.
PROBES = {
    "P1": {"repo": "intfloat/e5-small-v2",
           "revision": "ffb93f3bd4047442299a41ebb6fa998a38507c52",
           "query_prompt": "query: ", "doc_prompt": "passage: ", "dim": 384},
    "P2": {"repo": "thenlper/gte-small",
           "revision": "17e1f347d17fe144873b1201da91788898c639cd",
           "query_prompt": "", "doc_prompt": "", "dim": 384},
}
MAX_LEN = 512
_ST = {}


def _sha_texts(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode()); h.update(b"\x00")
    return h.hexdigest()


def _model(key):
    if key not in _ST:
        from sentence_transformers import SentenceTransformer
        s = PROBES[key]
        m = SentenceTransformer(s["repo"], revision=s["revision"], device="cuda")
        m.eval()
        assert m.get_sentence_embedding_dimension() == s["dim"]
        _ST[key] = m
    return _ST[key]


def encode(key, name, texts, role, batch_size=128):
    """-> (n, dim) fp32 unit-norm, cached on disk under a content key."""
    s = PROBES[key]
    blob = {"repo": s["repo"], "revision": s["revision"], "role": role,
            "prompt": s[f"{role}_prompt"], "max_length": MAX_LEN,
            "corpus_sha256": _sha_texts(texts)}
    dk = hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}-{key}-{role}-{dk}.npy"
    if p.exists():
        return np.load(p).astype(np.float32)
    m = _model(key)
    m.max_seq_length = MAX_LEN
    t0 = time.time()
    v = m.encode([s[f"{role}_prompt"] + t for t in texts], batch_size=batch_size,
                 normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    v = np.asarray(v, dtype=np.float32)
    if not np.isfinite(v).all():
        raise SystemExit(f"{key}/{name}/{role}: non-finite vectors")
    np.save(p, v.astype(np.float16))
    print(f"    {key} {name} {role}: {len(texts):,} in {time.time()-t0:.0f}s", flush=True)
    return v


def units():
    """-> ordered [(unit_id, family, queries, qids, docs, doc_ids, qrels)].

    `qrels` is pytrec_eval's {qid: {docid: int}}. A unit is one (queries, corpus) retrieval
    problem; the family macro averages units within a family, families equally (§Surfaces).
    """
    from datasets import load_dataset
    out = []
    # consumer-health + legal: mteb layout (queries / corpus / default:test triples)
    for family, comps in COMPONENTS.items():
        for name, repo, rev in comps:
            if name in ("BRIGHT", "LEDGER"):
                continue
            q = load_dataset(repo, "queries", revision=rev, split="queries")
            c = load_dataset(repo, "corpus", revision=rev, split="corpus")
            t = load_dataset(repo, "default", revision=rev, split="test")
            dt = [(f"{a} {b}".strip() if a else b) for a, b in zip(c["title"], c["text"])] \
                if "title" in c.column_names else list(c["text"])
            qrels = {}
            for qi, ci, sc in zip(t["query-id"], t["corpus-id"], t["score"]):
                if int(sc) > 0:
                    qrels.setdefault(str(qi), {})[str(ci)] = int(sc)
            out.append((name, family, list(q["text"]), [str(x) for x in q["_id"]],
                        dt, [str(x) for x in c["_id"]], qrels))
    # BRIGHT: one unit per slice, its own corpus
    brev = dict((n, r) for _f, cs in COMPONENTS.items() for n, _rp, r in cs)["BRIGHT"]
    for sl in BRIGHT_SLICES:
        e = load_dataset("xlangai/BRIGHT", "examples", revision=brev, split=sl)
        d = load_dataset("xlangai/BRIGHT", "documents", revision=brev, split=sl)
        qrels = {}
        for qid, gold in zip(e["id"], e["gold_ids"]):
            qrels[str(qid)] = {str(g): 1 for g in gold}
        out.append((f"BRIGHT/{sl}", "BRIGHT", list(e["query"]), [str(x) for x in e["id"]],
                    list(d["content"]), [str(x) for x in d["id"]], qrels))
    # LEDGER: page-level, graded 0/1/2
    import cov_ledger
    qs, qrels_raw, ids, texts, _rep = cov_ledger.load(verbose=False)
    qids = [f"q{i}" for i in range(len(qs))]
    qrels = {}
    for qid, js in zip(qids, qrels_raw):
        g = {str(j["doc_id"]): int(j["relevance"]) for j in js if int(j["relevance"]) > 0}
        if g:
            qrels[qid] = g
    out.append(("LEDGER", "finance", qs, qids, texts, ids, qrels))
    return out


def score_units(keys=("P1", "P2"), k=200):
    """-> {probe: {unit: {qid: ndcg@10}}} plus a unit report."""
    import evalkit
    per, rep = {kk: {} for kk in keys}, {}
    for uid, family, qs, qids, ds, dids, qrels in units():
        # a query with no positive judgment cannot contribute; pytrec_eval drops it anyway
        keep = [i for i, q in enumerate(qids) if q in qrels]
        qs, qids = [qs[i] for i in keep], [qids[i] for i in keep]
        print(f"  {uid:26s} {len(qs):6,} q x {len(ds):7,} d", flush=True)
        for kk in keys:
            qv = encode(kk, uid.replace("/", "_"), qs, "query")
            dv = encode(kk, uid.replace("/", "_"), ds, "doc")
            s = evalkit.score(qv, qids, dv, dids, qrels, k=k)
            per[kk][uid] = s
        rep[uid] = dict(family=family, n_queries=len(qs), n_docs=len(ds),
                        n_scored={kk: len(per[kk][uid]) for kk in keys})
    return per, rep


if __name__ == "__main__":
    per, rep = score_units()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_scores.json").write_text(json.dumps(
        {"per_query": {k: {u: v for u, v in d.items()} for k, d in per.items()},
         "units": rep}, indent=1))
    print("wrote", OUT / "probe_scores.json")
