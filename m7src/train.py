"""Train the lookup table against the frozen teacher document tower.

Objectives (instructions-m7.md):
  A  contrastive InfoNCE against precomputed frozen doc vectors, with a VRAM-resident negative
     bank, dataset-aware batching, and false-negative filtering by teacher-score margin
  B  distillation to teacher query embeddings: normalized cosine + a ranking-preservation KL
     over top-k similarities against frozen doc vectors
  C  B-init, then A-finetune

Everything selected here is selected on the dev suite. TRAIN excludes the held-out slices and
everything the fingerprint decontamination dropped.
"""
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

import dev_eval
import mix
import pool as poolmod
from _paths import WORK
from decontam import OUT as DECON
import init_table
from init_table import get_init, idf_weights
from table import NO_PREFIX, WITH_PREFIX, Preproc, QueryTable, get_tokenizer, \
    occurrence_weights, ragged, tokenize
from teacher import QUERY_PREFIX, encode_cached

RUNS = WORK / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
PRE = {"noprefix": NO_PREFIX, "prefix": WITH_PREFIX}


@dataclass
class Cfg:
    run_id: str = "dev"
    objective: str = "C"                # A | B | C
    init: str = "teacher"               # teacher | input_emb | random
    preproc: str = "noprefix"           # noprefix | prefix -- RUNTIME query tokenization
    init_preproc: str = ""              # "" = same as preproc. The context the `teacher` init
                                        # forwards each vocab token in, separated from the runtime
                                        # rule because the mandate makes fixed runtime-prefix
                                        # variants MANDATORY and prefix-conditioned ROWS
                                        # exploratory (Codex review #3 BLOCKER 3): changing
                                        # `preproc` alone moved both at once, so that arm was not
                                        # the mandated ablation.
    learned_weights: bool = True
    idf_init_weights: bool = True
    lr: float = 3e-3                    # NOTE: 3e-3 is 10-300x above every published frozen-tower
    lr_weights: float = 1e-2            # recipe (NV-Retriever 1e-5, BGE/E5/GTE 1e-5..5e-5) and is
                                        # the surviving explanation for the phase-1 contrastive
                                        # collapse (results/m7_diag_scores.json + arXiv 2110.09348).
                                        # Kept as the default ONLY so phase-1 runs stay
                                        # reproducible; new arms must set it explicitly.
    warmup_steps: int = 0               # linear ramp, applied per phase (B and A each warm up)
    lr_schedule: str = "constant"       # constant | warmup_constant | warmup_linear
    steps_b: int = 4000
    steps_a: int = 8000
    batch: int = 512
    n_neg: int = 32768
    temp: float = 0.02
    hard_neg_k: int = 0                 # mined hard negatives per query (0 = random bank only)
    hard_neg_source: str = "teacher"    # teacher | bm25 | mixed -- the mandated negatives ablation
    use_provided_hardneg: bool = True   # ESCI Irrelevant + Mr.TyDi negatives
    fn_margin: float = 0.02             # false-negative filter: drop negs with teacher score > pos - margin
    kl_weight: float = 1.0
    kl_k: int = 32
    cos_weight: float = 1.0
    reg_init: float = 1e-3              # penalty toward init, scaled by 1/(1+row update count)
    bank_size: int = 2_000_000
    sources: tuple = ()                 # () = all available
    eval_every: int = 2000
    eval_components: tuple = ("nq-250k", "cqadup-programmers", "cqadup-physics")
    seed: int = 0
    b_query_sources_all: bool = True    # include nq-open/triviaqa query text in B
    b_pseudo_queries: int = 0           # vocabulary-coverage distillation (see m7src/pseudoq.py)
    b_pseudo_frac: float = 0.5          # share of each B batch drawn from pseudo-queries
    b_pseudo_kind: str = "short"        # pseudoq.KINDS. "mixed" is capacity lever #7: half the
                                        # usual <=32-word spans, half 64-320-word ones. "short"
                                        # reproduces every prior run.
    pool_mode: str = "mean"             # count saturation, applied in the TRAINING forward too
                                        # (table.POOL_MODES). "mean" reproduces every prior run.


# ---- data ----------------------------------------------------------------------------

def kept_pairs(sources=None):
    """TRAIN pairs surviving decontamination, held-out slices already excluded by mix."""
    keep = json.loads((DECON / "kept.json").read_text())
    tr, _ = mix.split_pairs(sources)
    allow = {s: set(v) for s, v in keep.items()}
    return [p for p in tr if p[1] in allow.get(p[0], set())]


def build_arrays(cfg, index):
    pairs = kept_pairs(list(cfg.sources) or None)
    store_of = {s: mix.load_source(s)["docstore"] for s in {p[0] for p in pairs}}
    q_texts, pos_idx, hn_idx, src_id = [], [], [], []
    srcs = sorted({p[0] for p in pairs})
    sid = {s: i for i, s in enumerate(srcs)}
    # grouped by source so at most one store's id map is resident at a time
    by_src = {}
    for p in pairs:
        by_src.setdefault(p[0], []).append(p)
    _, _bset, _ = banned_rows()
    n_banned_pos = 0
    for src in srcs:
        st = store_of[src]
        for _, qid, query, posd, hneg in by_src[src]:
            ps = [j for j in (index.get(st, d) for d in posd) if j is not None]
            # review #2 MAJOR 10: a banned row must not enter the loss as a POSITIVE either --
            # dropping the row (and the pair, if nothing remains) extends R2 to the new classes.
            kept_ps = [j for j in ps if j not in _bset]
            if len(kept_ps) < len(ps):
                n_banned_pos += len(ps) - len(kept_ps)
            ps = kept_ps
            if not ps:
                continue
            q_texts.append(query)
            pos_idx.append(ps)
            hn_idx.append([j for j in (index.get(st, d) for d in hneg)
                           if j is not None and j not in _bset]
                          if cfg.use_provided_hardneg else [])
            src_id.append(sid[src])
        index.drop(st)
    if n_banned_pos:
        print(f"  build_arrays: dropped {n_banned_pos} banned positives (B2 mask)", flush=True)
    return q_texts, pos_idx, hn_idx, np.array(src_id, dtype=np.int32), srcs


_BANNED = None


def banned_rows():
    """(sorted int64 array, python set, digest) of pool rows banned as negatives -- the Codex B2
    mask, produced by decontam_pool.py. Loaded once. REFUSES to run if the artifact is missing:
    an empty mask must be the measured result of the pool pass, never a missing file."""
    global _BANNED
    if _BANNED is None:
        import hashlib as _h
        p = WORK / "decontam" / "banned_pool_rows.npy"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- run m7src/decontam_pool.py first (Codex B2)")
        mp = p.with_suffix(".meta.json")
        if not mp.exists():
            raise FileNotFoundError(f"{mp} missing -- the mask must carry the pool identity it "
                                    "was computed against (review #2 MAJOR 11)")
        import json as _json
        pool_meta = _json.loads((WORK / "pool" / "meta.json").read_text())
        mask_meta = _json.loads(mp.read_text())
        if mask_meta["pool_id_sha256"] != pool_meta["id_sha256"]:
            raise AssertionError("banned_pool_rows was computed against a different pool "
                                 "(id_sha256 mismatch) -- re-run decontam_pool.py")
        arr = np.load(p)
        _BANNED = (arr, set(arr.tolist()), _h.sha256(arr.tobytes()).hexdigest()[:8])
    return _BANNED


def clean_fallback_row(bset, start=0, also_avoid=()):
    """First row from `start` that is neither banned nor in `also_avoid` (a query's own
    positives -- review #2 MAJOR 12: a fallback that only dodges the ban can hand a query its
    positive back as a hard negative)."""
    r = start
    while r in bset or r in also_avoid:
        r += 1
    return r


def mine_bm25_negatives(name, q_texts, src_of, store_of, index, k, exclude, banned=None):
    """BM25-mined hard negatives, the lexical arm of the mandated negatives ablation.

    Mined WITHIN each query's own doc store rather than across the whole pool: a negative is only
    informative if it was a plausible candidate, and an ESCI product retrieved for a Wikipedia
    question is not. Each store is indexed once (hotpotqa-corpus is 5.23M documents, so this is
    the expensive part) and the result is cached.
    """
    import hashlib as _h
    _, _bset, _bdig = banned_rows() if banned is None else banned
    ex_sig = _h.sha256(b"".join(np.asarray(sorted(e), dtype=np.int64).tobytes()
                                for e in exclude)).hexdigest()[:8]
    sig = _h.sha256(("|".join(q_texts)).encode()).hexdigest()[:12] + "-x" + ex_sig + "-b" + _bdig
    p = WORK / "runs" / f"hardneg-bm25-{name}-k{k}-{sig}.npy"
    if p.exists():
        return np.load(p)
    import Stemmer
    import bm25s
    out = np.zeros((len(q_texts), k), dtype=np.int64)
    by_store = {}
    for i, src in enumerate(src_of):
        by_store.setdefault(store_of[src], []).append(i)
    st = Stemmer.Stemmer("english")
    for store, rows in by_store.items():
        ids, texts = mix.load_store(store)
        print(f"  bm25 mining {store}: {len(texts):,} docs, {len(rows):,} queries, "
              f"rss {int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0])/1e6:.1f} GB",
              flush=True)
        # Tokenize, then FREE THE TEXTS, then index. The original order passed the tokenizer's
        # output straight into index(), so 5.23M HotpotQA document strings (~2 GB), their
        # tokenized form, and the index under construction were all alive at once -- inside a
        # process already holding the pseudo-query targets and the pool's faulted pages. Same
        # class of bug as the negative-bank gather (see the 2026-08-28 incident).
        tok_corpus = bm25s.tokenize(texts, stopwords="en", stemmer=st, show_progress=False)
        del texts
        r = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
        r.index(tok_corpus, show_progress=False)
        del tok_corpus
        over = k + max((len(exclude[i]) for i in rows), default=0) + 4
        qt = bm25s.tokenize([q_texts[i] for i in rows], stopwords="en", stemmer=st,
                            show_progress=False)
        got, _ = r.retrieve(qt, k=min(over, len(ids)), show_progress=False)
        del r
        base = index.spans[store][0]
        for j, i in enumerate(rows):
            ex = set(exclude[i])
            picked = [base + int(d) for d in got[j]
                      if base + int(d) not in ex and base + int(d) not in _bset][:k]
            while len(picked) < k:
                picked.append(picked[-1] if picked else clean_fallback_row(_bset, base, ex))
            out[i] = picked
        index.drop(store)
    np.save(p, out)
    return out


@torch.no_grad()
def mine_hard_negatives(name, q_vecs, pool_vecs, k, exclude, q_chunk=2048, d_chunk=500_000,
                        banned=None):
    """Teacher-mined negatives: top-k pool docs per query by the teacher's own query vector,
    minus that query's positives.

    LOOP ORDER IS THE WHOLE COST. The first version put queries outside and the pool inside, so
    the 9.5 GB pool was read from the memmap and uploaded to the GPU once PER QUERY CHUNK: 171
    chunks x 9.5 GB = 1.6 TB of traffic, measured at 76 s per chunk = 3.6 hours, for a function
    whose docstring promised a few minutes. The arithmetic is identical either way, so the pool
    goes on the outside and is touched exactly once. All query vectors fit in VRAM (349,934 x 768
    fp16 = 537 MB), which is what makes the inversion possible.

    The cache key hashes the query vectors and the pool shape, not just a name and a length: a
    name-and-count key silently reuses a stale mining result when the mix changes but the count
    does not. It also carries the B2 banned-mask digest, so pre-mask caches can never be reused.

    `banned` is (sorted_array, set, digest) in the SAME row space as pool_vecs. Default None loads
    the real pool mask -- only correct when pool_vecs is the real pool; a synthetic pool (e.g.
    scripts/check_mining.py) must inject its own, typically the empty mask."""
    import hashlib as _h
    _, _bset, _bdig = banned_rows() if banned is None else banned
    ex_sig = _h.sha256(b"".join(np.asarray(sorted(e), dtype=np.int64).tobytes()
                                for e in exclude)).hexdigest()[:8]
    sig = _h.sha256(np.ascontiguousarray(q_vecs).tobytes()
                    + str((len(q_vecs), pool_vecs.shape, k)).encode()).hexdigest()[:12] \
        + "-x" + ex_sig + "-b" + _bdig
    p = WORK / "runs" / f"hardneg-{name}-k{k}-{sig}.npy"
    if p.exists():
        return np.load(p)
    n_q = len(q_vecs)
    over = k + max((len(e) for e in exclude), default=0) + 4
    qg = torch.from_numpy(np.ascontiguousarray(q_vecs)).cuda().half()
    best_s = torch.full((n_q, over), float("-inf"), device="cuda")
    best_i = torch.zeros((n_q, over), dtype=torch.int64, device="cuda")
    t0 = time.time()
    for dlo in range(0, len(pool_vecs), d_chunk):
        d = torch.from_numpy(np.ascontiguousarray(pool_vecs[dlo:dlo + d_chunk])).cuda()
        for qlo in range(0, n_q, q_chunk):
            qhi = min(qlo + q_chunk, n_q)
            s = (qg[qlo:qhi] @ d.T).float()          # q_chunk x d_chunk, the peak allocation
            kk = min(over, s.shape[1])
            cs, ci = torch.topk(s, kk, dim=1)
            del s
            cat_s = torch.cat([best_s[qlo:qhi], cs], 1)
            cat_i = torch.cat([best_i[qlo:qhi], ci + dlo], 1)
            ts, o = torch.topk(cat_s, over, dim=1)
            best_s[qlo:qhi], best_i[qlo:qhi] = ts, torch.gather(cat_i, 1, o)
        del d
        torch.cuda.empty_cache()
        print(f"    mine pool {min(dlo + d_chunk, len(pool_vecs))}/{len(pool_vecs)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    cand = best_i.cpu().numpy()
    del qg, best_s, best_i
    torch.cuda.empty_cache()
    out = np.zeros((n_q, k), dtype=np.int64)
    for r in range(n_q):
        ex = set(exclude[r])
        picked = [c for c in cand[r] if c not in ex and c not in _bset][:k]
        while len(picked) < k:
            picked.append(picked[-1] if picked else clean_fallback_row(_bset, 0, ex))
        out[r] = picked
    np.save(p, out)
    return out


@torch.no_grad()
def _mine_hard_negatives_qouter(name, q_vecs, pool_vecs, k, exclude, q_chunk=2048,
                                d_chunk=500_000, cache=False):
    """The original query-outer implementation, kept ONLY as the reference for the equivalence
    test in scripts/check_mining.py. Do not call it in a run: it re-reads the whole pool per query
    chunk. It is here because "I made it 30x faster" is a claim that needs a witness."""
    import hashlib as _h
    sig = _h.sha256(np.ascontiguousarray(q_vecs[::997]).tobytes()
                    + str((len(q_vecs), pool_vecs.shape, k)).encode()).hexdigest()[:12]
    p = WORK / "runs" / f"hardneg-qouter-{name}-k{k}-{sig}.npy"
    if cache and p.exists():
        return np.load(p)
    out = np.zeros((len(q_vecs), k), dtype=np.int64)
    over = k + max((len(e) for e in exclude), default=0) + 4
    for lo in range(0, len(q_vecs), q_chunk):
        hi = min(lo + q_chunk, len(q_vecs))
        q = torch.from_numpy(np.ascontiguousarray(q_vecs[lo:hi])).cuda().half()
        bs, bi = None, None
        for dlo in range(0, len(pool_vecs), d_chunk):
            d = torch.from_numpy(np.ascontiguousarray(pool_vecs[dlo:dlo + d_chunk])).cuda()
            s = q @ d.T
            del d
            kk = min(over, s.shape[1])
            cs, ci = torch.topk(s.float(), kk, dim=1)
            del s
            if bs is None:
                bs, bi = cs, ci + dlo
            else:
                cat_s, cat_i = torch.cat([bs, cs], 1), torch.cat([bi, ci + dlo], 1)
                bs, o = torch.topk(cat_s, min(over, cat_s.shape[1]), dim=1)
                bi = torch.gather(cat_i, 1, o)
        cand = bi.cpu().numpy()
        for r in range(hi - lo):
            ex = set(exclude[lo + r])
            picked = [c for c in cand[r] if c not in ex][:k]
            while len(picked) < k:
                picked.append(picked[-1] if picked else 0)
            out[lo + r] = picked
    if cache:
        np.save(p, out)
    return out


# ---- losses --------------------------------------------------------------------------

def infonce(qv, pos_v, neg_v, temp, teacher_q=None, teacher_pos=None, fn_margin=0.0,
            neg_pool_idx=None, pos_pool_idx=None, all_pos_idx=None, stats=None):
    """qv (B,d) student; pos_v (B,d); neg_v (N,d) shared bank sample (+ optional per-query hard
    negatives folded in by the caller). False negatives are masked by teacher-score margin.

    The query's own positive can be drawn into the shared bank sample; that is a bug, not the
    false-negative phenomenon the fn_margin ablation studies, so it is masked unconditionally by
    pool index whatever fn_margin is set to."""
    s_pos = (qv * pos_v).sum(1, keepdim=True) / temp
    s_neg = (qv @ neg_v.T) / temp
    if neg_pool_idx is not None and pos_pool_idx is not None:
        same = neg_pool_idx.unsqueeze(0) == pos_pool_idx.unsqueeze(1)
        if all_pos_idx is not None:
            # every positive of this query, not only the sampled one. ESCI averages ~13.5
            # positives per query, so the siblings were reaching the denominator guarded only by
            # the fn_margin filter -- which the fn_margin=0 ablation arm switches off, confounding
            # its own reading.
            # all_pos_idx is (B, max_pos), -1 padded; pool indices are >= 0 so padding never
            # matches. (B, max_pos, 1) == (1, 1, N) -> (B, max_pos, N), any over the positives.
            same = same | (all_pos_idx.unsqueeze(2) == neg_pool_idx.view(1, 1, -1)).any(1)
        s_neg = s_neg.masked_fill(same, float("-inf"))
    if fn_margin > 0 and teacher_q is not None:
        with torch.no_grad():
            t_neg = teacher_q @ neg_v.T
            t_pos = (teacher_q * teacher_pos).sum(1, keepdim=True)
            mask = t_neg > (t_pos - fn_margin)
        if stats is not None:
            # MAJOR-3: the post-mask negative count was unobserved for the whole phase-1 grid,
            # and the filter is the leading suspect for the contrastive collapse.
            stats["fn_masked_frac"] = round(float(mask.float().mean()), 4)
        s_neg = s_neg.masked_fill(mask, float("-inf"))
    logits = torch.cat([s_pos, s_neg], 1)
    return F.cross_entropy(logits, torch.zeros(len(qv), dtype=torch.long, device=qv.device))


def distill(qv, teacher_q, cand_v, temp, cos_w, kl_w):
    """Normalized cosine to the teacher query vector + KL over top-k similarities."""
    cos = (1.0 - (qv * teacher_q).sum(1)).mean()
    kl = torch.zeros((), device=qv.device)
    if kl_w > 0 and cand_v is not None:
        with torch.no_grad():
            pt = F.softmax(torch.einsum("bd,bkd->bk", teacher_q, cand_v) / temp, dim=1)
        ls = F.log_softmax(torch.einsum("bd,bkd->bk", qv, cand_v) / temp, dim=1)
        kl = F.kl_div(ls, pt, reduction="batchmean")
    return cos_w * cos + kl_w * kl, float(cos), float(kl)


# ---- training ------------------------------------------------------------------------

def run(cfg: Cfg, log=print):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    from dataclasses import replace as _replace
    # The pooling rule is part of the frozen preprocessing, so it must reach BOTH the training
    # forward and every eval in this run -- a table trained under `mean` and served under `sqrt`
    # is the eval-only probe (lever #4), not the trained system (lever #6).
    pre = _replace(PRE[cfg.preproc], pool_mode=cfg.pool_mode)
    init_pre = _replace(PRE[cfg.init_preproc or cfg.preproc], pool_mode=cfg.pool_mode)
    tok = get_tokenizer()
    V = tok.vocab_size

    log(f"[{cfg.run_id}] {json.dumps(asdict(cfg))}")
    index, pool_vecs, pmeta = poolmod.build()
    log(f"  pool {pool_vecs.shape} ({pool_vecs.nbytes/1e9:.2f} GB fp16)")
    q_texts, pos_idx, hn_idx, src_id, srcs = build_arrays(cfg, index)
    log(f"  train pairs {len(q_texts):,} over sources {srcs}")

    tq = np.asarray(encode_cached(f"trainq-{len(q_texts)}", q_texts, prefix=QUERY_PREFIX,
                                  dtype=torch.float16, verbose=True), dtype=np.float32)
    ids_all = tokenize(tok, q_texts, pre)

    # objective-B-only extras: query text with no positives (nq-open, TriviaQA) and
    # pseudo-queries for vocabulary coverage. Both feed the cosine term, never the KL term.
    b_texts, b_tq = [], None
    if cfg.objective in ("B", "C"):
        extra = []
        if cfg.b_query_sources_all:
            import json as _json

            from mix import QUERYTEXT_SOURCES
            kq_p = WORK / "decontam" / "kept_querytext.json"
            if not kq_p.exists():
                raise RuntimeError(f"{kq_p} missing: run decontam_querytext.py. Silently "
                                   "training without the query-text sources is not a fallback.")
            kq = _json.loads(kq_p.read_text())
            for src in QUERYTEXT_SOURCES:
                fp = WORK / "train" / "querytext" / f"{src}.json"
                if not fp.exists():
                    continue
                if src not in kq:
                    raise RuntimeError(f"kept_querytext.json has no entry for {src}: re-run "
                                       "decontam_querytext.py")
                qs = _json.loads(fp.read_text())
                extra += [qs[i] for i in kq[src]]
        if cfg.b_pseudo_queries:
            import pseudoq
            extra += pseudoq.build_decontaminated(cfg.b_pseudo_queries, kind=cfg.b_pseudo_kind)
        if extra:
            b_texts = extra
            # `encode_cached`'s key carries sha_texts(texts), so a different pseudo-query kind of
            # the same size cannot collide with a cached encode even though the label matches.
            b_tq = np.asarray(encode_cached(f"bextra-{len(b_texts)}", b_texts, prefix=QUERY_PREFIX,
                                            dtype=torch.float16, verbose=True), dtype=np.float32)
            log(f"  objective-B extra query text: {len(b_texts):,} "
                f"(query-text-only sources + {cfg.b_pseudo_queries:,} "
                f"{cfg.b_pseudo_kind} pseudo-queries)")
    b_ids_all = tokenize(tok, b_texts, pre) if b_texts else []

    hard = None
    if cfg.hard_neg_k:
        k = cfg.hard_neg_k
        tag = f"{cfg.preproc}-{len(q_texts)}"
        store_of = {s: mix.load_source(s)["docstore"] for s in srcs}
        src_of = [srcs[i] for i in src_id]
        parts = []
        if cfg.hard_neg_source in ("teacher", "mixed"):
            kk = k // 2 if cfg.hard_neg_source == "mixed" else k
            parts.append(mine_hard_negatives(tag, tq, pool_vecs, kk, pos_idx)[:, :kk])
        if cfg.hard_neg_source in ("bm25", "mixed"):
            kk = k - (k // 2) if cfg.hard_neg_source == "mixed" else k
            parts.append(mine_bm25_negatives(tag, q_texts, src_of, store_of, index, kk, pos_idx))
        hard = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
        log(f"  {cfg.hard_neg_source}-mined hard negatives {hard.shape}")

    # negative bank: fixed deterministic sample, VRAM-resident fp16
    nb = min(cfg.bank_size, len(pool_vecs))
    bank_ids = np.sort(rng.choice(len(pool_vecs), size=nb, replace=False))
    _barr, _, _ = banned_rows()
    if _barr.size:
        _i = np.minimum(np.searchsorted(_barr, bank_ids), _barr.size - 1)
        _hit = _barr[_i] == bank_ids
        bank_ids = bank_ids[~_hit]
        log(f"  bank: dropped {int(_hit.sum())} banned pool rows (Codex B2 mask)")
    nb = len(bank_ids)   # review #2 BLOCKER 1: every sampler draws [0, nb); a stale nb after the
                                                                                # filter is OOB
    # Gathered in CHUNKS straight into the destination GPU tensor. The one-liner this replaces --
    # `torch.from_numpy(np.ascontiguousarray(pool_vecs[bank_ids])).cuda()` -- materialized the
    # whole 2M x 1024 fp16 bank (4.1 GB) on the HOST first, on top of the pseudo-query targets
    # (another 4.1 GB at the 2M mix) and the pool pages the gather faults in. That combination
    # reached 24.7 GB RSS on a 25 GB box and thrashed with the GPU idle (2026-08-28). Identical
    # result, ~4 GB lower peak: bank_ids is sorted, so the chunks are sequential in the memmap.
    bank = torch.empty((nb, pool_vecs.shape[1]), dtype=torch.float16, device="cuda")
    for lo in range(0, nb, 250_000):
        hi = min(lo + 250_000, nb)
        bank[lo:hi] = torch.from_numpy(np.ascontiguousarray(pool_vecs[bank_ids[lo:hi]])).cuda()
    log(f"  negative bank {tuple(bank.shape)} ({bank.numel()*2/1e9:.2f} GB VRAM)")

    W0 = torch.from_numpy(get_init(cfg.init, init_pre, vocab=V, runtime_pre=pre)).cuda()
    if cfg.init_preproc and cfg.init_preproc != cfg.preproc:
        log(f"  init rows built under preprocessing {cfg.init_preproc!r}; runtime tokenization "
            f"uses {cfg.preproc!r} (mandated runtime-prefix ablation)")
    # A "run:<id>" init restores the trained TOKEN WEIGHTS too. Restoring rows but re-deriving
    # weights from IDF would start from a system that is not the one whose dev score is being used
    # as the baseline -- p1-objB's learned weights are IDF-LIKE (spearman -0.44 vs update count)
    # but they are not IDF, and the weights are as much part of the table as the rows.
    if cfg.init.startswith("run:") and cfg.learned_weights:
        w0w = init_table.run_token_weights(cfg.init.split(":", 1)[1])
        if w0w is None:
            w0w = idf_weights() if cfg.idf_init_weights else None
            log(f"  init {cfg.init} carries no token weights; falling back to "
                f"{'IDF' if cfg.idf_init_weights else 'uniform'}")
        else:
            log(f"  init {cfg.init}: restored rows AND {len(w0w)} trained token weights")
    else:
        w0w = idf_weights() if (cfg.learned_weights and cfg.idf_init_weights) else None
    model = QueryTable(W0.cpu().numpy(), weight_init=w0w, learned_weights=cfg.learned_weights).cuda()
    updates = torch.zeros(V, device="cuda")
    opt = torch.optim.Adam([
        {"params": [model.rows], "lr": cfg.lr},
        *([{"params": [model.w_raw], "lr": cfg.lr_weights}] if cfg.learned_weights else []),
    ])

    hist = []
    # A fixed query sample for the collapse diagnostics, drawn once so the numbers are comparable
    # across steps and across runs.
    diag_idx = np.random.default_rng(12345).choice(len(ids_all),
                                                   size=min(2048, len(ids_all)), replace=False)

    @torch.no_grad()
    def collapse_stats():
        """Is the representation degenerating? Loss can fall while this goes bad -- that is the
        documented failure mode of a too-high lr (arXiv 2110.09348), so it must be observed and
        not inferred from the dev curve."""
        model.eval()
        q = _fwd([ids_all[i] for i in diag_idx]).float()
        model.train()
        g = q @ q.T
        n = g.shape[0]
        off = (g.sum() - g.diagonal().sum()) / (n * (n - 1))
        sv = torch.linalg.svdvals(q)
        # participation ratio of the singular values: how many directions the batch really uses
        eff_rank = (sv.sum() ** 2) / (sv ** 2).sum()
        # Wang & Isola uniformity, on the unit sphere; more negative = better spread
        sq = torch.cdist(q, q).pow(2)
        iu = torch.triu_indices(n, n, offset=1, device=q.device)
        unif = torch.log(torch.exp(-2.0 * sq[iu[0], iu[1]]).mean())
        return {"mean_pairwise_cos": round(float(off), 4),      # -> 1.0 means collapse
                "effective_rank": round(float(eff_rank), 2),    # -> 1.0 means one direction
                "dim": int(q.shape[1]),
                "uniformity": round(float(unif), 4)}

    def dev(tag, step):
        model.eval()
        per = dev_eval.eval_table(model, pre, components=list(cfg.eval_components), tok=tok)
        m, means = dev_eval.report(per, f"  [{cfg.run_id}] {tag} step {step}")
        cs = collapse_stats()
        log(f"  [{cfg.run_id}] {tag} step {step} collapse: {json.dumps(cs)}")
        hist.append({"step": step, "phase": tag, "macro": m, "per_component": means,
                     "collapse": cs})
        model.train()
        return m

    def _fwd(id_lists):
        """One place that turns token-id lists into query vectors during training, so the pooling
        rule cannot reach two of the three call sites and miss the third."""
        f, o, l = ragged(id_lists, "cuda")
        psw = None if cfg.pool_mode == "mean" else occurrence_weights(id_lists, cfg.pool_mode,
                                                                     device="cuda")
        return model(f, o, l, extra_psw=psw)

    def batch_of(idx):
        return _fwd([ids_all[i] for i in idx])

    def step_b(idx):
        # optional pseudo-query / query-text-only part: cosine only, no positive to rank against
        n_ps = int(len(idx) * cfg.b_pseudo_frac) if b_texts else 0
        extra_loss = torch.zeros((), device="cuda")
        touched = []
        if n_ps:
            j = rng.integers(0, len(b_texts), n_ps)
            qv2 = _fwd([b_ids_all[k] for k in j])
            extra_loss = (1.0 - (qv2 * torch.from_numpy(b_tq[j]).cuda()).sum(1)).mean()
            idx = idx[:max(1, len(idx) - n_ps)]
            touched = [b_ids_all[k] for k in j]
        qv = batch_of(idx)
        t = torch.from_numpy(tq[idx]).cuda()
        cand = None
        if cfg.kl_weight > 0:
            k = cfg.kl_k
            # the candidate set is the query's own positive plus k-1 distractors. Distractors come
            # from the VRAM bank (or the teacher-mined hard negatives), never from random memmap
            # reads: 512 x 31 random reads per step would make the pool the bottleneck.
            p_i = np.array([pos_idx[i][0] for i in idx])
            pos_v = torch.from_numpy(np.ascontiguousarray(pool_vecs[p_i])).cuda().float()
            # hard_neg_k need not equal kl_k - 1. It usually does not: the screen mines 16 while
            # kl_k is 32, and `hard[idx][:, :k-1]` then yielded 16 columns reshaped as 31, which
            # crashed every hard-negative arm of the phase-2 screen -- i.e. every arm that could
            # satisfy the kill criterion. The candidate-set SIZE is held at kl_k for all arms and
            # only its COMPOSITION varies with hard_neg_k; sizing it from whatever mining returned
            # would confound "hard negatives help" with "the KL set got smaller".
            def _bank_rows(n_per_q):
                sel = torch.from_numpy(rng.integers(0, nb, len(idx) * n_per_q)).cuda()
                return bank.index_select(0, sel).float().view(len(idx), n_per_q, model.dim)

            if hard is not None:
                hk = min(k - 1, hard.shape[1])
                d = torch.from_numpy(np.ascontiguousarray(pool_vecs[hard[idx][:, :hk].ravel()]))
                dist = d.cuda().float().view(len(idx), hk, model.dim)
                if hk < k - 1:
                    dist = torch.cat([dist, _bank_rows(k - 1 - hk)], 1)
            else:
                dist = _bank_rows(k - 1)
            cand = torch.cat([pos_v.unsqueeze(1), dist], 1)
        loss, cos, kl = distill(qv, t, cand, cfg.temp, cfg.cos_weight, cfg.kl_weight)
        if n_ps:
            loss = loss + cfg.cos_weight * extra_loss
        return loss, {"cos": cos, "kl": kl, "cos_extra": round(float(extra_loss), 4)}, \
            [ids_all[i] for i in idx] + touched

    def step_a(idx):
        qv = batch_of(idx)
        p_i = np.array([pos_idx[i][rng.integers(len(pos_idx[i]))] for i in idx])
        pos_v = torch.from_numpy(np.ascontiguousarray(pool_vecs[p_i])).cuda().float()
        sel_np = rng.integers(0, nb, cfg.n_neg)
        sel = torch.from_numpy(sel_np).cuda()
        neg = bank.index_select(0, sel).float()
        neg_pool = bank_ids[sel_np]
        extra = []
        if hard is not None:
            extra.append(hard[idx].ravel())
        hn = [h for i in idx for h in hn_idx[i][:4]]
        if hn:
            extra.append(np.array(hn))
        if extra:
            e = np.unique(np.concatenate(extra))
            neg = torch.cat([neg, torch.from_numpy(np.ascontiguousarray(pool_vecs[e])).cuda().float()])
            neg_pool = np.concatenate([neg_pool, e])
        t = torch.from_numpy(tq[idx]).cuda()
        tp = torch.from_numpy(np.ascontiguousarray(pool_vecs[p_i])).cuda().float()
        mx = max(len(pos_idx[i]) for i in idx)
        allpos = np.full((len(idx), mx), -1, dtype=np.int64)
        for r, i in enumerate(idx):
            allpos[r, :len(pos_idx[i])] = pos_idx[i]
        st = {}
        loss = infonce(qv, pos_v, neg, cfg.temp, t, tp, cfg.fn_margin,
                       neg_pool_idx=torch.from_numpy(neg_pool).cuda(),
                       pos_pool_idx=torch.from_numpy(p_i).cuda(),
                       all_pos_idx=torch.from_numpy(allpos).cuda(),
                       stats=st)
        return loss, {"n_neg": neg.shape[0], **st}, [ids_all[i] for i in idx]

    base_lrs = [g["lr"] for g in opt.param_groups]

    def set_lr(step, steps):
        """Linear warmup (then constant or linear decay), applied per phase: objective C's A-phase
        needs its own ramp, since that is where the collapse happened."""
        if cfg.lr_schedule == "constant":
            return
        w = max(1, cfg.warmup_steps)
        f = min(1.0, step / w)
        if cfg.lr_schedule == "warmup_linear" and step > w:
            f = max(0.1, 1.0 - (step - w) / max(1, steps - w))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f

    def train_phase(tag, steps, stepfn):
        if steps <= 0:
            return
        t0 = time.time()
        if cfg.lr_schedule != "constant":
            log(f"  [{cfg.run_id}] {tag} lr {cfg.lr:g} schedule {cfg.lr_schedule} "
                f"warmup {cfg.warmup_steps}")
        for s in range(1, steps + 1):
            set_lr(s, steps)
            idx = rng.integers(0, len(q_texts), cfg.batch)
            loss, extra, used = stepfn(idx)
            # the rows actually touched this step, pseudo-queries included: `updates` is a
            # reported coverage metric and the scale of the pull-toward-init penalty, so it must
            # not be derived from the pre-truncation batch
            rows = np.unique(np.concatenate([np.asarray(u, dtype=np.int64) for u in used]))
            ri = torch.from_numpy(rows).cuda()   # rows touched by the pair part of the batch
            if cfg.reg_init > 0:
                # low-update rows stay pulled toward init: the penalty is scaled by 1/(1+updates)
                scale = 1.0 / (1.0 + updates[ri])
                loss = loss + cfg.reg_init * (scale * (model.rows[ri] - W0[ri]).pow(2).sum(1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                updates[ri] += 1
            if s % 500 == 0:
                log(f"  [{cfg.run_id}] {tag} {s}/{steps} loss {float(loss):.4f} "
                    f"{extra} {(time.time()-t0)/s*1000:.0f} ms/step")
            if cfg.eval_every and s % cfg.eval_every == 0:
                dev(tag, s)

    if cfg.objective in ("B", "C"):
        train_phase("B", cfg.steps_b, step_b)
    if cfg.objective in ("A", "C"):
        if cfg.objective == "C" and cfg.reg_init > 0:
            # re-anchor: during C's A-phase the penalty must pull toward the B checkpoint, not
            # toward the teacher init -- which the ridge sweep showed is itself a poor table
            # (~0.20 at lambda=10), so anchoring there actively drags C away from its own gains.
            W0 = model.rows.detach().clone()
            log("  [C] re-anchored the reg_init penalty to the B-phase checkpoint")
        train_phase("A", cfg.steps_a, step_a)

    cov = {"rows_updated": int((updates > 0).sum()), "vocab": V,
           "rows_updated_frac": float((updates > 0).float().mean()),
           "median_updates_of_touched": float(updates[updates > 0].median()) if (updates > 0).any() else 0.0}
    final = dev("final", cfg.steps_b + cfg.steps_a)
    from table import save_table
    upd = updates.detach().cpu().numpy()
    save_table(RUNS / f"{cfg.run_id}.npz", model, pre, updates=upd,
               meta={"cfg": asdict(cfg), "dev_macro": final, "coverage": cov})
    np.save(RUNS / f"{cfg.run_id}.init.npy", W0.detach().cpu().numpy().astype(np.float16))
    (RUNS / f"{cfg.run_id}.json").write_text(json.dumps(
        {"cfg": asdict(cfg), "history": hist, "final_macro": final, "coverage": cov,
         "n_train_pairs": len(q_texts)}, indent=1))
    # work/ is gitignored, so the collapse evidence the phase-2 screen exists to produce would not
    # survive a box wipe. Commit the small part: the per-eval curve and its collapse diagnostics.
    from _paths import REPO
    (REPO / "results" / f"m7_run_{cfg.run_id}.json").write_text(json.dumps(
        {"run_id": cfg.run_id, "cfg": asdict(cfg), "final_macro": final, "coverage": cov,
         "history": [{k: h[k] for k in ("step", "phase", "macro", "collapse") if k in h}
                     for h in hist]}, indent=1))
    log(f"  [{cfg.run_id}] coverage {json.dumps(cov)}")
    return final, model, hist
