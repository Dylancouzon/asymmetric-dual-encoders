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
from init_table import get_init, idf_weights
from table import NO_PREFIX, WITH_PREFIX, Preproc, QueryTable, get_tokenizer, ragged, tokenize
from teacher import QUERY_PREFIX, encode_cached

RUNS = WORK / "runs"
RUNS.mkdir(parents=True, exist_ok=True)
PRE = {"noprefix": NO_PREFIX, "prefix": WITH_PREFIX}


@dataclass
class Cfg:
    run_id: str = "dev"
    objective: str = "C"                # A | B | C
    init: str = "teacher"               # teacher | input_emb | random
    preproc: str = "noprefix"           # noprefix | prefix
    learned_weights: bool = True
    idf_init_weights: bool = True
    lr: float = 3e-3
    lr_weights: float = 1e-2
    steps_b: int = 4000
    steps_a: int = 8000
    batch: int = 512
    n_neg: int = 32768
    temp: float = 0.02
    hard_neg_k: int = 0                 # teacher-mined hard negatives per query (0 = bank only)
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
    for src in srcs:
        st = store_of[src]
        for _, qid, query, posd, hneg in by_src[src]:
            ps = [j for j in (index.get(st, d) for d in posd) if j is not None]
            if not ps:
                continue
            q_texts.append(query)
            pos_idx.append(ps)
            hn_idx.append([j for j in (index.get(st, d) for d in hneg) if j is not None]
                          if cfg.use_provided_hardneg else [])
            src_id.append(sid[src])
        index.drop(st)
    return q_texts, pos_idx, hn_idx, np.array(src_id, dtype=np.int32), srcs


@torch.no_grad()
def mine_hard_negatives(name, q_vecs, pool_vecs, k, exclude, q_chunk=2048, d_chunk=500_000):
    """Teacher-mined negatives: top-k pool docs per query by the teacher's own query vector,
    minus that query's positives. Frozen doc vectors make this a few minutes for the whole set.

    The cache key hashes the query vectors and the pool shape, not just a name and a length: a
    name-and-count key silently reuses a stale mining result when the mix changes but the count
    does not."""
    import hashlib as _h
    sig = _h.sha256(np.ascontiguousarray(q_vecs[::997]).tobytes()
                    + str((len(q_vecs), pool_vecs.shape, k)).encode()).hexdigest()[:12]
    p = WORK / "runs" / f"hardneg-{name}-k{k}-{sig}.npy"
    if p.exists():
        return np.load(p)
    out = np.zeros((len(q_vecs), k), dtype=np.int64)
    over = k + max((len(e) for e in exclude), default=0) + 4
    t0 = time.time()
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
        if (lo // q_chunk) % 20 == 0:
            print(f"    mine {hi}/{len(q_vecs)} ({time.time()-t0:.0f}s)", flush=True)
    np.save(p, out)
    return out


# ---- losses --------------------------------------------------------------------------

def infonce(qv, pos_v, neg_v, temp, teacher_q=None, teacher_pos=None, fn_margin=0.0,
            neg_pool_idx=None, pos_pool_idx=None):
    """qv (B,d) student; pos_v (B,d); neg_v (N,d) shared bank sample (+ optional per-query hard
    negatives folded in by the caller). False negatives are masked by teacher-score margin.

    The query's own positive can be drawn into the shared bank sample; that is a bug, not the
    false-negative phenomenon the fn_margin ablation studies, so it is masked unconditionally by
    pool index whatever fn_margin is set to."""
    s_pos = (qv * pos_v).sum(1, keepdim=True) / temp
    s_neg = (qv @ neg_v.T) / temp
    if neg_pool_idx is not None and pos_pool_idx is not None:
        s_neg = s_neg.masked_fill(neg_pool_idx.unsqueeze(0) == pos_pool_idx.unsqueeze(1),
                                  float("-inf"))
    if fn_margin > 0 and teacher_q is not None:
        with torch.no_grad():
            t_neg = teacher_q @ neg_v.T
            t_pos = (teacher_q * teacher_pos).sum(1, keepdim=True)
            mask = t_neg > (t_pos - fn_margin)
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
    pre = PRE[cfg.preproc]
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
            from mix import QUERYTEXT_SOURCES
            import json as _json
            kq_p = WORK / "decontam" / "kept_querytext.json"
            kq = _json.loads(kq_p.read_text()) if kq_p.exists() else {}
            for src in QUERYTEXT_SOURCES:
                fp = WORK / "train" / "querytext" / f"{src}.json"
                if fp.exists() and src in kq:
                    qs = _json.loads(fp.read_text())
                    extra += [qs[i] for i in kq[src]]
        if cfg.b_pseudo_queries:
            import pseudoq
            extra += pseudoq.build_decontaminated(cfg.b_pseudo_queries)
        if extra:
            b_texts = extra
            b_tq = np.asarray(encode_cached(f"bextra-{len(b_texts)}", b_texts, prefix=QUERY_PREFIX,
                                            dtype=torch.float16, verbose=True), dtype=np.float32)
            log(f"  objective-B extra query text: {len(b_texts):,} "
                f"(query-text-only sources + {cfg.b_pseudo_queries:,} pseudo-queries)")
    b_ids_all = tokenize(tok, b_texts, pre) if b_texts else []

    hard = None
    if cfg.hard_neg_k:
        hard = mine_hard_negatives(f"{cfg.preproc}-{len(q_texts)}", tq, pool_vecs,
                                   cfg.hard_neg_k, pos_idx)
        log(f"  teacher-mined hard negatives {hard.shape}")

    # negative bank: fixed deterministic sample, VRAM-resident fp16
    nb = min(cfg.bank_size, len(pool_vecs))
    bank_ids = np.sort(rng.choice(len(pool_vecs), size=nb, replace=False))
    bank = torch.from_numpy(np.ascontiguousarray(pool_vecs[bank_ids])).cuda()
    log(f"  negative bank {tuple(bank.shape)} ({bank.numel()*2/1e9:.2f} GB VRAM)")

    W0 = torch.from_numpy(get_init(cfg.init, pre, vocab=V)).cuda()
    w0w = idf_weights() if (cfg.learned_weights and cfg.idf_init_weights) else None
    model = QueryTable(W0.cpu().numpy(), weight_init=w0w, learned_weights=cfg.learned_weights).cuda()
    updates = torch.zeros(V, device="cuda")
    opt = torch.optim.Adam([
        {"params": [model.rows], "lr": cfg.lr},
        *([{"params": [model.w_raw], "lr": cfg.lr_weights}] if cfg.learned_weights else []),
    ])

    hist = []

    def dev(tag, step):
        model.eval()
        per = dev_eval.eval_table(model, pre, components=list(cfg.eval_components), tok=tok)
        m, means = dev_eval.report(per, f"  [{cfg.run_id}] {tag} step {step}")
        hist.append({"step": step, "phase": tag, "macro": m, "per_component": means})
        model.train()
        return m

    def batch_of(idx):
        f, o, l = ragged([ids_all[i] for i in idx], "cuda")
        return model(f, o, l)

    def step_b(idx):
        # optional pseudo-query / query-text-only part: cosine only, no positive to rank against
        n_ps = int(len(idx) * cfg.b_pseudo_frac) if b_texts else 0
        extra_loss = torch.zeros((), device="cuda")
        if n_ps:
            j = rng.integers(0, len(b_texts), n_ps)
            f2, o2, l2 = ragged([b_ids_all[k] for k in j], "cuda")
            qv2 = model(f2, o2, l2)
            extra_loss = (1.0 - (qv2 * torch.from_numpy(b_tq[j]).cuda()).sum(1)).mean()
            idx = idx[:max(1, len(idx) - n_ps)]
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
            if hard is not None:
                d = torch.from_numpy(np.ascontiguousarray(pool_vecs[hard[idx][:, :k - 1].ravel()]))
                dist = d.cuda().float().view(len(idx), k - 1, model.dim)
            else:
                sel = torch.from_numpy(rng.integers(0, nb, len(idx) * (k - 1))).cuda()
                dist = bank.index_select(0, sel).float().view(len(idx), k - 1, model.dim)
            cand = torch.cat([pos_v.unsqueeze(1), dist], 1)
        loss, cos, kl = distill(qv, t, cand, cfg.temp, cfg.cos_weight, cfg.kl_weight)
        if n_ps:
            loss = loss + cfg.cos_weight * extra_loss
        return loss, {"cos": cos, "kl": kl, "cos_extra": round(float(extra_loss), 4)}

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
        loss = infonce(qv, pos_v, neg, cfg.temp, t, tp, cfg.fn_margin,
                       neg_pool_idx=torch.from_numpy(neg_pool).cuda(),
                       pos_pool_idx=torch.from_numpy(p_i).cuda())
        return loss, {"n_neg": neg.shape[0]}

    def train_phase(tag, steps, stepfn):
        if steps <= 0:
            return
        t0 = time.time()
        for s in range(1, steps + 1):
            idx = rng.integers(0, len(q_texts), cfg.batch)
            loss, extra = stepfn(idx)
            rows = np.unique(np.concatenate([ids_all[i] for i in idx]))
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
    log(f"  [{cfg.run_id}] coverage {json.dumps(cov)}")
    return final, model, hist
