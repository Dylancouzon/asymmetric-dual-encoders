"""D2-PRE -- the closed-form preflight that routes (or cancels) D2's five training chains.

THE SHAPE, AND WHY IT IS FORCED. `m8/registry.json` probe `D2-PRE` requires "closed-form ridge
residual solves on FROZEN INCUMBENT ROWS", a "SUM-INITIALIZED, zero-residual compile" that
reproduces R0, and four arms "at EQUAL ADDED-ROW BUDGET". Those are only mutually satisfiable one
way: every arm keeps R0's 30,522 incumbent rows frozen and ADDS exactly K = 35,014 rows
(30,522 + 35,014 = 65,536, the vocabulary D2 itself names). A wholesale replacement has no
incumbent rows to freeze and no constituents to sum-initialize from. D2 PROPER is still a
replacement; this is its preflight analogue.

THE FOUR ARMS, all at K added rows:
  seg       (a) BPE learning K adjacent-token merges over the INCUMBENT WordPiece stream, merges
                free to cross the whitespace boundary, applied at inference by LEARNED MERGE RANK.
                A fired unit REMOVES its constituents. Row init = SUM of constituent rows.
  add_word  (b) additive overlapping WORD n-grams (n in {2,3}). Init = ZERO.
  add_char  (c) additive overlapping CHARACTER n-grams (n in {4,5}). Init = ZERO.
  seg_cold  (d) `seg` with rows under 20 fit activations forced to zero residual.

WHY THE TWO INITS DIFFER, and why that is forced. Under the released rule
`q = normalize( sum_types sqrt(c_t) w_t / sum_types sqrt(c_t) )`, replacing constituents a,b
(count 1 each) by `w_p = w_a + w_b` leaves the NUMERATOR identical and changes only the
denominator, a positive per-query scalar the final L2 normalize cancels -- the served vector is
EXACTLY unchanged. An ADDITIVE row fires alongside its constituents, so a sum init would
double-count them, while a zero row contributes nothing to the numerator. Same principle, opposite
init, both exact. `self_test` checks this ALGEBRAICALLY on real rows: it is the registry's pooling
canary, and it must carry CONTEXT TOKENS -- a phrase that is a query's whole content normalizes sum
and mean to the same vector, so a two-token fixture cannot fail (CODEMAP pitfall 19).

WHAT IT MAY AND MAY NOT DO. Nothing ships from a closed-form fit. This is DIAGNOSTIC AND ROUTING.
`route()` is a pure function of the measured numbers so the router cannot be re-read in this
session's favour, and the amended row is the authority for every constant in it.
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp

import m8base
import blockcg
import probe_guard

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_d2_pre.json"
CKPT = m8base.WORK / "d2pre"

# R0: section 23's diagonal cell (b0,a0), which is M7's shipped candidate's frame.
R0_RUN = "m8nf-seed0"
OOD = ("cqadup-programmers", "cqadup-physics")
FUSED_COMPONENTS = ("nq-250k", "hotpotqa", "cqadup-programmers", "cqadup-physics")

# Private Use Area-B: each incumbent row id becomes one codepoint, so a query is a STRING over an
# alphabet of incumbent rows and an ordinary BPE trainer learns merges over it. Nothing else in
# this file depends on the choice.
PUA = 0x100000

# ---- every constant the registry freezes (registry `frozen_definitions`) -------------------
K_ADDED = 35_014                 # 30,522 + 35,014 = 65,536
WORD_NGRAM_N = (2, 3)
CHAR_NGRAM_N = (4, 5)
COLD_THRESHOLD = 20              # arm (d): fewer than this many fit activations -> zero residual
N_FOLDS = 5
FOLD_SEED = 8                    # deterministic fold assignment; fixed before any solve
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
INNER_VAL_FRACTION = 0.10
BAR = 0.00519                    # the chain bar, unrounded
FOLD_SIGN_MIN = 4
COLD_MASS_MAX = 0.20
FUSED_MIN = -0.0020
REVERSAL_MARGIN = 0.0020
COMPILE_TOL = 0.001

ARMS = ("seg", "add_word", "add_char", "seg_cold")
SEG_ARMS = ("seg", "seg_cold")
ADDITIVE_ARMS = ("add_word", "add_char")
CLS_ID = 101


# ---- deterministic partitioning ------------------------------------------------------------

def fold_of(keys, n_folds=N_FOLDS, seed=FOLD_SEED):
    """Deterministic balanced folds over a list of GROUP keys.

    Grouping by key is what stops an exact-duplicate query text from straddling folds -- the
    duplicate would otherwise put (nearly) the same row in a fold's fit and another fold's fit,
    which is the leak the cross-fitting exists to prevent.
    """
    groups = sorted(set(keys))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(groups))
    gf = np.empty(len(groups), dtype=np.int64)
    gf[perm] = np.arange(len(groups)) % n_folds
    idx = {g: int(gf[i]) for i, g in enumerate(groups)}
    return np.array([idx[k] for k in keys], dtype=np.int64)


# ---- incumbent tokenization ------------------------------------------------------------------

def tok_pre():
    from table import Preproc, get_tokenizer
    return get_tokenizer(), Preproc(prefix="", add_special_tokens=True, max_length=512,
                                    pool_mode="sqrt")


def ids_of(tok, texts, pre, chunk=50_000):
    from table import tokenize
    out = []
    for lo in range(0, len(texts), chunk):
        out.extend(tokenize(tok, texts[lo:lo + chunk], pre))
    return out


def word_spans(tok, texts):
    """(word -> id span) per query, by tokenizing each whitespace word separately.

    BERT's basic tokenizer splits on whitespace and punctuation and WordPiece has no cross-word
    context, so concatenating per-word ids reproduces whole-text tokenization. `self_test` ASSERTS
    that -- it is the premise the word-n-gram arm rests on, and nothing else would notice a break.
    """
    spans, flat = [], []
    for t in texts:
        words = t.split()
        enc = tok(words, add_special_tokens=False)["input_ids"] if words else []
        s, pos = [], 0
        for e in enc:
            s.append((pos, pos + len(e)))
            pos += len(e)
        spans.append(s)
        flat.append([i for e in enc for i in e])
    return spans, flat


def counts_of(rows):
    u, c = np.unique(np.asarray(rows, dtype=np.int64), return_counts=True)
    return u, c.astype(np.float32)


def r0_denominator(ids):
    """R0's own per-query sqrt-count denominator, used by EVERY arm's ridge (registry MAJOR 7).

    It cancels at serve time under L2 normalization but NOT inside the least-squares objective, so
    letting each arm use its own would silently change that arm's residual target, its per-query
    weighting and the effective strength of lambda -- a degree of freedom that picks winners.
    """
    return np.array([max(float(np.sqrt(counts_of(x)[1]).sum()), 1e-12) for x in ids],
                    dtype=np.float64)


# ---- candidate mining (fold-honest: FIT text only) -------------------------------------------

def train_seg_units(fit_ids, k_added, v0):
    """Deterministic BPE over the incumbent id alphabet.

    Returns (units, apply_fn, sha). `units[j]` is the tuple of incumbent ids that added row v0+j
    stands for -- BPE units over this alphabet are BY CONSTRUCTION fully expanded into incumbent
    rows, so the sum init is always defined and needs no recursive walk. Special tokens never enter
    the mining stream, so no merge can contain one.
    """
    from tokenizers import Tokenizer, models, trainers
    lines = ["".join(chr(PUA + i) for i in x) for x in fit_ids]
    t = Tokenizer(models.BPE(unk_token=None, continuing_subword_prefix="", end_of_word_suffix=""))
    tr = trainers.BpeTrainer(vocab_size=v0 + k_added, min_frequency=1, show_progress=False,
                             initial_alphabet=[chr(PUA + i) for i in range(v0)], special_tokens=[])
    t.train_from_iterator(lines, tr)
    vocab = t.get_vocab()
    multi = sorted((s for s in vocab if len(s) > 1), key=lambda s: vocab[s])
    if len(multi) != k_added:
        raise SystemExit(f"BPE produced {len(multi):,} multi-token units, not the frozen "
                         f"K={k_added:,}. The budget is frozen and an arm may not run short "
                         f"(registry frozen_definitions).")
    units = [tuple(ord(c) - PUA for c in s) for s in multi]
    remap = np.zeros(max(vocab.values()) + 1, dtype=np.int64)
    mi = {s: j for j, s in enumerate(multi)}
    for s, tid in vocab.items():
        remap[tid] = (ord(s) - PUA) if len(s) == 1 else (v0 + mi[s])
    tstr = t.to_str()
    sha = hashlib.sha256(tstr.encode()).hexdigest()

    def apply_fn(id_lists, batch=20_000):
        out = []
        for lo in range(0, len(id_lists), batch):
            chunk = id_lists[lo:lo + batch]
            enc = t.encode_batch(["".join(chr(PUA + i) for i in x) for x in chunk])
            out.extend(remap[np.asarray(e.ids, dtype=np.int64)] for e in enc)
        return out

    return units, apply_fn, sha, tstr


def seg_units_from_tokenizer(tokenizer_str, v0):
    """The (units, remap) a saved seg tokenizer implies. Recomputed from the tokenizer's own
    vocabulary rather than stored beside it, so a saved dictionary cannot drift from the segmenter
    that produced it."""
    from tokenizers import Tokenizer
    t = Tokenizer.from_str(tokenizer_str)
    vocab = t.get_vocab()
    multi = sorted((s for s in vocab if len(s) > 1), key=lambda s: vocab[s])
    remap = np.zeros(max(vocab.values()) + 1, dtype=np.int64)
    mi = {s: j for j, s in enumerate(multi)}
    for s, tid in vocab.items():
        remap[tid] = (ord(s) - PUA) if len(s) == 1 else (v0 + mi[s])
    units = [tuple(ord(c) - PUA for c in s) for s in multi]
    return t, units, remap


def _top_k(cnt, k, what):
    if len(cnt) < k:
        raise SystemExit(f"only {len(cnt):,} distinct {what} candidates, fewer than the frozen "
                         f"K={k:,}. The budget is frozen and an arm may not run short.")
    return [key for key, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]


def mine_word_ngrams(tok, fit_texts, k_added):
    spans, flat = word_spans(tok, fit_texts)
    cnt = Counter()
    for s, f in zip(spans, flat):
        for n in WORD_NGRAM_N:
            for i in range(len(s) - n + 1):
                cnt[tuple(f[s[i][0]:s[i + n - 1][1]])] += 1
    return _top_k(cnt, k_added, "word n-gram")


def mine_char_ngrams(fit_texts, k_added):
    cnt = Counter()
    for t in fit_texts:
        for w in t.lower().split():
            for n in CHAR_NGRAM_N:
                for i in range(len(w) - n + 1):
                    cnt[w[i:i + n]] += 1
    return _top_k(cnt, k_added, "character n-gram")


# ---- featurizers: query -> (row ids, occurrence counts) in the extended row space -------------

def feats_plain(ids):
    return [counts_of(x) for x in ids]


def feats_add_seq(ids, seqs, v0):
    index = {s: v0 + j for j, s in enumerate(seqs)}
    lens = sorted({len(s) for s in seqs})
    out = []
    for x in ids:
        rows = list(x)
        for n in lens:
            for i in range(len(x) - n + 1):
                r = index.get(tuple(x[i:i + n]))
                if r is not None:
                    rows.append(r)
        out.append(counts_of(rows))
    return out


def feats_add_char(texts, ids, grams, v0):
    index = {g: v0 + j for j, g in enumerate(grams)}
    lens = sorted({len(g) for g in grams})
    out = []
    for x, t in zip(ids, texts):
        rows = list(x)
        for w in t.lower().split():
            for n in lens:
                for i in range(len(w) - n + 1):
                    r = index.get(w[i:i + n])
                    if r is not None:
                        rows.append(r)
        out.append(counts_of(rows))
    return out


# ---- the released pooling rule, as a matrix and as an encoder --------------------------------

def bag_matrix(feats, denom, V):
    """Row i is `sqrt(c_t) / denom_i`. `denom` is R0's, for every arm (see `r0_denominator`)."""
    indptr, indices, data = [0], [], []
    for (u, c), d in zip(feats, denom):
        indices.append(u)
        data.append((np.sqrt(c) / d).astype(np.float64))
        indptr.append(indptr[-1] + len(u))
    return sp.csr_matrix((np.concatenate(data) if data else np.zeros(0),
                          np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64),
                          np.array(indptr)), shape=(len(feats), V))


def encode(feats, W):
    """THE released query path for an arbitrary row space. The per-query denominator is omitted
    because the final L2 normalize cancels it exactly; `self_test` checks this against
    `QueryTable.encode(pool_mode='sqrt')` on real dev queries and real int8 rows."""
    out = np.empty((len(feats), W.shape[1]), dtype=np.float32)
    fb = W[min(CLS_ID, W.shape[0] - 1)]
    fbn = float(np.linalg.norm(fb))
    fb = (fb / fbn) if fbn > 1e-6 else np.eye(1, W.shape[1], 0, dtype=np.float32)[0]
    for i, (u, c) in enumerate(feats):
        w = np.sqrt(c)
        v = (W[u] * w[:, None]).sum(0)
        n = float(np.linalg.norm(v))
        out[i] = (v / n) if n > 1e-6 else fb
    return out


def serve_int8(W_inc_dequant, W_new):
    """The int8 serving frame. INCUMBENT CODES ARE PRESERVED: `W_inc_dequant` is already
    `q * scale`, and re-quantizing it is the identity (absmax = 127*scale recovers scale exactly),
    which `self_test` asserts rather than assumes. Only the NEW block is quantized."""
    from table import dequantize_int8, quantize_int8
    q, s = quantize_int8(W_new)
    return np.concatenate([W_inc_dequant, dequantize_int8(q, s)]).astype(np.float32)


# ---- scoring ---------------------------------------------------------------------------------

class DevSet:
    """Query texts, ids and OOF query folds for the components a stage reads. Corpora stay in
    `dev_eval`'s own cache; only the light side is held here."""

    def __init__(self, tok, pre, components):
        import dev_eval
        self.dev_eval = dev_eval
        self.components = tuple(components)
        self.qids, self.texts, self.ids, self.folds = {}, {}, {}, {}
        for c in self.components:
            _, _, q_ids, q_texts, _, _ = dev_eval.doc_vecs(c)
            self.qids[c] = list(q_ids)
            self.texts[c] = list(q_texts)
            self.ids[c] = ids_of(tok, self.texts[c], pre)
            self.folds[c] = fold_of(self.qids[c])

    def denom(self, c):
        return r0_denominator(self.ids[c])


def per_query(dev, c, feats, W):
    return dev.dev_eval.eval_query_vecs(c, encode(feats, W))


def macro_over(dev, per_q, mask=None):
    """Equal weight per component, over the selected queries of each."""
    means = {}
    for c in dev.components:
        sel = [v for q, v in per_q[c].items()
               if mask is None or mask[c].get(q, False)]
        means[c] = float(np.mean(sel)) if sel else float("nan")
    return float(np.mean(list(means.values()))), means


# ---- the solve --------------------------------------------------------------------------------

def init_new_rows(kind, payload, W_inc, k):
    """SUM of constituents for a segmentation arm, ZERO for an additive one. The registry's
    corrected `compositional_init_floor`: the MEAN is not a floor, because (w_a+w_b)/2 is not a
    positive scalar multiple of w_a+w_b and normalization therefore does not restore it."""
    W = np.zeros((k, W_inc.shape[1]), dtype=np.float32)
    if kind == "sum":
        for j, unit in enumerate(payload):
            W[j] = W_inc[list(unit)].sum(0)
    return W


def solve(Xn, Rt, lam, device, maxiter=3000):
    D, info = blockcg.block_cg_ridge(Xn, Rt, np.zeros((Xn.shape[1], Rt.shape[1]), np.float32),
                                     lam, device=device, maxiter=maxiter)
    return D.astype(np.float32), info


def residual_target(X, Y, W_inc, W_init, v0):
    Xi, Xn = X[:, :v0].tocsr(), X[:, v0:].tocsr()
    Rt = (Y - (Xi @ W_inc) - (Xn @ W_init)).astype(np.float32)
    return Xn, Rt


def pick_lambda(Xn_in, Rt_in, Xn_val, Y_val, X_val_base, lam_grid, device):
    """The nested, RETRIEVAL-BLIND lambda rule. Selection reads the cosine of the normalized
    predicted vector to the teacher target on inner-validation queries and nothing else, so the
    scored endpoint cannot select its own regularization. Ties break to the LARGER lambda.

    The grid is DESCENDED and the descent stops after three consecutive decreases (registry
    `frozen_definitions`, amendment 3). The grid itself is unchanged; only the order and the stop
    are. This exists because the additive character arm's system is badly conditioned -- one
    measured solve at lambda=1e-2 ran 1,051 CG iterations in 403 s -- so evaluating the two
    smallest lambdas on every arm and fold would cost 7-9 hours AND would likely return
    non-converged solves, which is a wrong number rather than a slow one.
    """
    grid, conv, skipped, drops = {}, {}, [], 0
    for lam in sorted(lam_grid, reverse=True):
        if drops >= 3:
            skipped.append(lam)
            continue
        D, info = solve(Xn_in, Rt_in, lam, device)
        pred = (X_val_base + Xn_val @ D).astype(np.float32)
        pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
        c = float((pred * Y_val).sum(1).mean())
        prev = list(grid.values())[-1] if grid else None
        drops = (drops + 1) if (prev is not None and c < prev) else 0
        grid[lam] = c
        conv[lam] = {"iterations": info["iterations"], "converged": info["converged"],
                     "seconds": info["seconds"], "worst_rel_residual": info["worst_rel_residual"]}
        del D
        m8base.empty_cache()
    best = max(grid, key=lambda l: (grid[l], l))
    return best, {"cosine_grid": grid, "cg": conv, "skipped_lambdas": skipped,
                  "argmax_at_grid_boundary": best in (max(lam_grid), min(lam_grid))}


# ---- the router (a pure function of the measured numbers) ------------------------------------

def route(m):
    """The frozen router. Every threshold is a literal from the registry row; nothing here reads a
    file, so it cannot be argued with after a number exists."""
    g = m["oof_gain"]
    g_d2 = max(g["seg"], g["seg_cold"])
    g_add = max(g["add_word"], g["add_char"])
    winner = max(g, key=lambda a: (g[a], a))
    conds = {
        "1_g_best_at_or_above_bar": g[winner] >= BAR,
        "2_positive_in_4_of_5_folds": sum(1 for v in m["fold_gain"][winner] if v > 0)
                                      >= FOLD_SIGN_MIN,
        # An UNMEASURED mass is a FAILED condition, never a passed one: a gate that passes when its
        # input is missing is not a gate (CODEMAP pitfall 17).
        "3_zero_update_occurrence_mass_at_most_20pct":
            m["zero_update_mass"][winner] is not None
            and m["zero_update_mass"][winner] <= COLD_MASS_MAX,
        "4_fused_gain_at_or_above_-0.0020": (m.get("fused_gain") or {}).get(winner) is not None
                                            and m["fused_gain"][winner] >= FUSED_MIN,
        "5_compile_reproduces_R0_and_canary_passes":
            abs(m["compile_delta_vs_r0"]) <= COMPILE_TOL and m["pooling_canary_pass"],
    }
    reversal = (g_add - g_d2) >= REVERSAL_MARGIN
    # The class the chains would go to: additive on a >= 0.0020 lead, and on any tie between 0 and
    # 0.0020, because an additive row with zero residual recovers R0 exactly and a segmentation
    # change does not.
    klass = "additive" if (reversal or g_add >= g_d2) else "D2"
    return {
        "g_by_arm": g, "g_D2": g_d2, "g_additive": g_add, "winning_arm": winner,
        "conditions": conds, "authorised": all(conds.values()),
        "reversal_margin_met": reversal, "class_the_chains_would_take": klass,
        "verdict": ("AUTHORISE " + klass) if all(conds.values()) else "DO NOT AUTHORISE",
        "_thresholds": {"bar": BAR, "fold_sign_min": FOLD_SIGN_MIN,
                        "cold_mass_max": COLD_MASS_MAX, "fused_min": FUSED_MIN,
                        "reversal_margin": REVERSAL_MARGIN, "compile_tol": COMPILE_TOL},
    }


# ---- conformance -------------------------------------------------------------------------------

def self_test(n=512):
    """Four properties every number in this probe rests on, each checked against REAL data rather
    than a fixture built out of the claim (CODEMAP pitfalls 15, 17, 19)."""
    from table import Preproc, QueryTable, dequantize_int8, ensure_release, load_table, quantize_int8, tokenize
    import dev_eval

    tok, pre = tok_pre()
    v0 = len(tok)
    rel = ensure_release(m8base.WORK / "runs" / f"{R0_RUN}.npz", device=m8base.device())
    W_inc = load_table(rel, variant="int8", device="cpu").rows.detach().numpy().astype(np.float32)

    _, _, _, q_texts, _, _ = dev_eval.doc_vecs(OOD[0])
    texts = list(q_texts)[:n]
    ids = ids_of(tok, texts, pre)

    ref = QueryTable(W_inc, learned_weights=False).to(m8base.device()).eval().encode(
        texts, pre, tok=tok)
    d1 = float(np.abs(ref - encode(feats_plain(ids), W_inc)).max())

    _, flat = word_spans(tok, texts)
    bare = Preproc(prefix="", add_special_tokens=False, max_length=512, pool_mode="sqrt")
    d2 = sum(1 for a, b in zip(flat, tokenize(tok, texts, bare)) if list(a) != list(b))

    # THE POOLING CANARY, algebraic and with CONTEXT TOKENS. A phrase that is a query's only
    # content normalizes sum and mean to the SAME vector, so a two-token fixture scores 0.0 for
    # both and can never fail -- the first version of this test did exactly that.
    rng = np.random.default_rng(0)
    quads = [tuple(int(v) for v in row) for row in rng.integers(1000, v0, size=(64, 4))]
    Ws = np.concatenate([W_inc, np.zeros((len(quads), W_inc.shape[1]), np.float32)])
    Wm = Ws.copy()
    for j, (a, b, _c, _d) in enumerate(quads):
        Ws[v0 + j] = W_inc[a] + W_inc[b]
        Wm[v0 + j] = 0.5 * (W_inc[a] + W_inc[b])
    base = [(np.array([a, b, c, d]), np.ones(4, np.float32)) for a, b, c, d in quads]
    merged = [(np.array([v0 + j, q[2], q[3]]), np.ones(3, np.float32)) for j, q in enumerate(quads)]
    d3_sum = float(np.abs(encode(base, W_inc) - encode(merged, Ws)).max())
    d3_mean = float(np.abs(encode(base, W_inc) - encode(merged, Wm)).max())

    # re-quantizing an already-dequantized int8 block is the identity, so preserving incumbent
    # codes and re-quantizing the whole table are the same thing -- asserted, not assumed.
    q, s = quantize_int8(W_inc)
    d4 = float(np.abs(dequantize_int8(q, s) - W_inc).max())

    ok = d1 < 5e-7 and d2 == 0 and d3_sum < 1e-6 and d3_mean > 1e-3 and d4 == 0.0
    out = {"encode_matches_released_path_max_abs": d1,
           "per_word_tokenization_mismatches": d2,
           "canary_sum_init_max_abs": d3_sum, "canary_mean_init_max_abs": d3_mean,
           "canary_pass": bool(d3_sum < 1e-6 and d3_mean > 1e-3),
           "int8_requantize_is_identity_max_abs": d4, "pass": bool(ok)}
    print(json.dumps(out, indent=2))
    return out


# ---- inputs -------------------------------------------------------------------------------

def load_incumbent():
    """R0's DEQUANTIZED shipped int8 rows (the int8 endpoint's own frame) and its training
    `updates` array -- which the release export does not carry, so it is read from the checkpoint."""
    from table import ensure_release, load_table
    rel = ensure_release(m8base.WORK / "runs" / f"{R0_RUN}.npz", device=m8base.device())
    W = load_table(rel, variant="int8", device="cpu").rows.detach().numpy().astype(np.float32)
    updates = np.load(m8base.WORK / "runs" / f"{R0_RUN}.npz")["updates"]
    return W, updates


def teacher_targets(texts):
    import torch
    from teacher import QUERY_PREFIX, encode_cached
    return np.asarray(encode_cached(f"trainq-{len(texts)}", texts, prefix=QUERY_PREFIX,
                                    dtype=torch.float16, verbose=False), dtype=np.float32)


def dev_query_texts(tok, pre):
    """Query texts + ids for the FULL PINNED DEV SUITE, cached. Condition (3)'s denominator is the
    whole suite, and HotpotQA's component cache is 1.6 GB, so this is parsed once and kept."""
    import dev_eval
    CKPT.mkdir(parents=True, exist_ok=True)
    p = CKPT / "dev_queries.json"
    if p.exists():
        blob = json.loads(p.read_text())
    else:
        # NOT `dev_eval.doc_vecs`: that also materializes the component's DOCUMENT VECTORS, and
        # HotpotQA's are 5.23M x 1024 fp16 = 10.7 GB, for query text this never needs.
        import devsuite
        import heldout
        blob = {}
        for c in dev_eval.dev_components():
            if c.startswith("heldout-"):
                _, _, q_ids, q_texts, _, _ = heldout.load(c)
            else:
                _, _, q_ids, q_texts, _ = devsuite.load(c)
            blob[c] = {"q_ids": list(q_ids), "q_texts": list(q_texts)}
            print(f"  dev queries {c}: {len(q_ids):,}", flush=True)
            gc.collect()
        p.write_text(json.dumps(blob))
    return {c: (v["q_ids"], v["q_texts"], ids_of(tok, v["q_texts"], pre))
            for c, v in blob.items()}


# ---- one arm on one fold ----------------------------------------------------------------------

def build_dictionary(arm, tok, inner_texts, inner_ids, v0, k):
    """Mine an arm's K added rows on the INNER 90% of this fold's fit half, then hold it fixed for
    the outer refit (registry `frozen_definitions`). Returns (kind, payload, featurize, sha)."""
    if arm in SEG_ARMS:
        units, apply_fn, sha = train_seg_units(inner_ids, k, v0)
        return "sum", units, (lambda texts, ids: feats_plain(apply_fn(ids))), sha
    if arm == "add_word":
        seqs = mine_word_ngrams(tok, inner_texts, k)
        return ("zero", seqs, (lambda texts, ids: feats_add_seq(ids, seqs, v0)),
                hashlib.sha256(repr(seqs).encode()).hexdigest())
    if arm == "add_char":
        grams = mine_char_ngrams(inner_texts, k)
        return ("zero", grams, (lambda texts, ids: feats_add_char(texts, ids, grams, v0)),
                hashlib.sha256(repr(grams).encode()).hexdigest())
    raise KeyError(arm)




def rebuild_featurizer(arm, payload, v0):
    """Reconstruct an arm's featurizer from a SAVED dictionary, so the fused read does not redo the
    solves. Deterministic and dictionary-identical by construction: the seg remap is recomputed
    from the tokenizer's own vocabulary, not stored beside it and trusted."""
    if arm in SEG_ARMS:
        t, _units, remap = seg_units_from_tokenizer(payload, v0)

        def featurize(texts, ids, batch=20_000):
            out = []
            for lo in range(0, len(ids), batch):
                enc = t.encode_batch(["".join(chr(PUA + i) for i in x)
                                      for x in ids[lo:lo + batch]])
                out.extend(counts_of(remap[np.asarray(e.ids, dtype=np.int64)]) for e in enc)
            return out
        return featurize
    if arm == "add_word":
        seqs = [tuple(s) for s in payload]
        return lambda texts, ids: feats_add_seq(ids, seqs, v0)
    if arm == "add_char":
        return lambda texts, ids: feats_add_char(texts, ids, list(payload), v0)
    raise KeyError(arm)


def _fold_path(arm, j, tag):
    CKPT.mkdir(parents=True, exist_ok=True)
    return CKPT / f"{tag}{arm}-f{j}"


def save_fold(arm, j, payload_json, W_new, tag=""):
    from table import quantize_int8
    q, s = quantize_int8(W_new)
    np.savez(str(_fold_path(arm, j, tag)) + ".npz", rows_int8=q, int8_scale=s)
    (_fold_path(arm, j, tag).with_suffix(".dict.json")).write_text(payload_json)


def load_fold(arm, j, tag=""):
    from table import dequantize_int8
    z = np.load(str(_fold_path(arm, j, tag)) + ".npz")
    payload = json.loads((_fold_path(arm, j, tag).with_suffix(".dict.json")).read_text())
    return dequantize_int8(z["rows_int8"], z["int8_scale"]).astype(np.float32), payload


def _dict_json(arm, payload, tokenizer_str):
    if arm in SEG_ARMS:
        return json.dumps(tokenizer_str)
    return json.dumps([list(x) for x in payload] if arm == "add_word" else list(payload))


# ---- stage 1: opportunity (DESCRIPTIVE; it may not stop the probe) -----------------------------

def stage1(tok, pre, apply_fn):
    """Fertility (units per whitespace word) under the incumbent tokenizer and under arm (a).

    REPORTED, NEVER GATED ON (registry `staged_checks`, amended). The '~0.104' threshold the first
    registration named is 0.00519/0.050, which reads §17b's association as a one-directional
    bound -- exactly what §17b forbids -- and it is a fertility test that could have killed the
    additive arms, whose mechanisms need not reduce fertility at all. `retention_decomp`'s
    definition: whole-query tokenization over whitespace words.
    """
    import dev_eval
    bare = type(pre)(prefix="", add_special_tokens=False, max_length=512, pool_mode="sqrt")
    sets = {}
    for c in OOD:
        _, _, _, qt, _, _ = dev_eval.doc_vecs(c)
        sets[c] = list(qt)
    for ds in m8base.SIX:
        sets[ds] = list(json.loads((RESULTS / "frozen_eval" / f"{ds}.json").read_text())
                        ["queries"].values())
    out = {}
    for name, texts in sets.items():
        ids = ids_of(tok, texts, bare)
        nw = np.array([max(1, len(t.split())) for t in texts], dtype=float)
        inc = np.array([len(x) for x in ids], dtype=float) / nw
        new = np.array([len(x) for x in apply_fn(ids)], dtype=float) / nw
        out[name] = {"incumbent_fertility": float(inc.mean()),
                     "d2_fertility": float(new.mean()),
                     "reduction": float((inc - new).mean())}
    return out


# ---- the main run ------------------------------------------------------------------------------

def run(smoke=False, device=None):
    t_start = time.time()
    device = device or m8base.device()
    tok, pre = tok_pre()
    v0 = len(tok)
    k = 512 if smoke else K_ADDED
    n_folds = 2 if smoke else N_FOLDS
    V = v0 + k
    tag = "smoke-" if smoke else ""

    st = self_test()
    if not st["pass"]:
        raise SystemExit("conformance failed; nothing downstream is interpretable")

    W_inc, updates = load_incumbent()
    fit_texts = json.loads((m8base.WORK / "m8_trainq_texts.json").read_text())
    Y_all = teacher_targets(fit_texts)
    if smoke:
        # A RANDOM subsample, not a prefix. The fit list is ordered by source, so `[:20_000]` is
        # pure e-commerce text; the first smoke's segmentation compile came out 0.011 below R0 for
        # that reason alone, which looks exactly like a broken compile. A smoke may be meaningless,
        # but it must not be meaningless in a way that mimics a real failure.
        sel = np.random.default_rng(0).choice(len(fit_texts), 20_000, replace=False)
        fit_texts, Y_all = [fit_texts[i] for i in sel], Y_all[sel]
    fit_ids = ids_of(tok, fit_texts, pre)
    fit_denom = r0_denominator(fit_ids)
    folds = fold_of(fit_texts, n_folds=n_folds)
    print(f"fit {len(fit_texts):,} queries ({len(set(fit_texts)):,} unique), {n_folds} folds, "
          f"K={k:,}, V={V:,}  [{time.time()-t_start:.0f}s]", flush=True)

    dev = DevSet(tok, pre, OOD)
    fitset = set(fit_texts)
    overlap = {c: sum(1 for t in dev.texts[c] if t in fitset) for c in OOD}
    print(f"OOD query texts also verbatim in the fit list: {overlap}", flush=True)
    full_dev = {} if smoke else dev_query_texts(tok, pre)
    full_dev_folds = {c: fold_of(v[0], n_folds=n_folds) for c, v in full_dev.items()}
    fold_qids = {c: {j: {q for q, f in zip(dev.qids[c], dev.folds[c]) if f == j}
                     for j in range(n_folds)} for c in OOD}

    r0_pq = {c: per_query(dev, c, feats_plain(dev.ids[c]), W_inc) for c in OOD}
    r0_macro, r0_means = macro_over(dev, r0_pq)
    print(f"R0 OOD macro {r0_macro:.10f} {r0_means}", flush=True)

    oof = {a: {c: {} for c in OOD} for a in ARMS}
    oof_c = {c: {} for c in OOD}
    fold_g, fold_c = {a: [] for a in ARMS}, []
    lam_log, sha_log, support = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    occ = {a: {"total": 0.0, "zero": 0.0} for a in ARMS}
    stage1_out = None

    for j in range(n_folds):
        fit_i = np.flatnonzero(folds != j)
        rng = np.random.default_rng(FOLD_SEED * 100 + j)
        perm = rng.permutation(len(fit_i))
        n_val = max(1, int(round(INNER_VAL_FRACTION * len(fit_i))))
        val_i, inner_i = fit_i[perm[:n_val]], fit_i[perm[n_val:]]
        inner_texts = [fit_texts[i] for i in inner_i]
        inner_ids = [fit_ids[i] for i in inner_i]
        f_texts = [fit_texts[i] for i in fit_i]
        f_ids = [fit_ids[i] for i in fit_i]
        pos = {v: i for i, v in enumerate(fit_i)}
        inner_r = np.array([pos[v] for v in inner_i])
        val_r = np.array([pos[v] for v in val_i])
        print(f"\n=== fold {j}: fit {len(fit_i):,} (inner {len(inner_i):,} / val {len(val_i):,})"
              f"  [{time.time()-t_start:.0f}s]", flush=True)
        seg_dict = None

        for arm in ARMS:
            t0 = time.time()
            if arm == "seg_cold":
                kind, payload, featurize, sha, tstr = seg_dict
            else:
                if arm in SEG_ARMS:
                    units, apply_fn, sha, tstr = train_seg_units(inner_ids, k, v0)
                    kind, payload = "sum", units
                    featurize = (lambda texts, ids, _a=apply_fn: feats_plain(_a(ids)))
                    if j == 0 and stage1_out is None and not smoke:
                        stage1_out = stage1(tok, pre, apply_fn)
                        print(f"  stage1 (descriptive) OOD reduction: "
                              f"{ {c: round(stage1_out[c]['reduction'], 4) for c in OOD} }",
                              flush=True)
                    seg_dict = (kind, payload, featurize, sha, tstr)
                elif arm == "add_word":
                    payload = mine_word_ngrams(tok, inner_texts, k)
                    kind, tstr = "zero", None
                    sha = hashlib.sha256(repr(payload).encode()).hexdigest()
                    featurize = (lambda texts, ids, _p=payload: feats_add_seq(ids, _p, v0))
                else:
                    payload = mine_char_ngrams(inner_texts, k)
                    kind, tstr = "zero", None
                    sha = hashlib.sha256(repr(payload).encode()).hexdigest()
                    featurize = (lambda texts, ids, _p=payload: feats_add_char(texts, ids, _p, v0))
            sha_log[arm][j] = sha
            t_dict = time.time() - t0

            X = bag_matrix(featurize(f_texts, f_ids), fit_denom[fit_i], V)
            W_init = init_new_rows(kind, payload, W_inc, k)
            Xn, Rt = residual_target(X, Y_all[fit_i], W_inc, W_init, v0)
            base = (X[:, :v0] @ W_inc + Xn @ W_init).astype(np.float32)
            sup = np.asarray((Xn != 0).sum(axis=0)).ravel()
            warm = (sup >= COLD_THRESHOLD) if arm == "seg_cold" else np.ones(k, dtype=bool)
            support[arm][j] = {"rows_with_0_activations": int((sup == 0).sum()),
                               "rows_under_5": int((sup < 5).sum()),
                               "rows_under_20": int((sup < 20).sum()),
                               "rows_forced_to_zero_residual": int((~warm).sum())}
            Xw = Xn[:, warm].tocsr()
            lam, grid = pick_lambda(Xw[inner_r], Rt[inner_r], Xw[val_r], Y_all[val_i],
                                    base[val_r], LAMBDA_GRID, device)
            lam_log[arm][j] = {"lambda": lam, **grid}
            D, info = solve(Xw, Rt, lam, device)
            lam_log[arm][j]["outer_cg"] = {"iterations": info["iterations"],
                                           "converged": info["converged"],
                                           "seconds": info["seconds"]}
            W_new = W_init.copy()
            W_new[warm] += D
            del X, Xn, Rt, base, Xw, D
            gc.collect()
            m8base.empty_cache()

            save_fold(arm, j, _dict_json(arm, payload, tstr), W_new, tag)
            W_arm = serve_int8(W_inc, W_new)
            W_cmp = serve_int8(W_inc, W_init) if arm == "seg" else None
            for c in OOD:
                F = featurize(dev.texts[c], dev.ids[c])
                keep = fold_qids[c][j]
                oof[arm][c].update({q: v for q, v in per_query(dev, c, F, W_arm).items()
                                    if q in keep})
                if W_cmp is not None:
                    oof_c[c].update({q: v for q, v in per_query(dev, c, F, W_cmp).items()
                                     if q in keep})
                del F
            for c, (qids, texts_c, ids_c) in full_dev.items():
                sel = np.flatnonzero(full_dev_folds[c] == j)
                if not len(sel):
                    continue
                for u, cnt in featurize([texts_c[i] for i in sel], [ids_c[i] for i in sel]):
                    inc = u < v0
                    occ[arm]["total"] += float(cnt.sum())
                    occ[arm]["zero"] += float(cnt[inc][updates[u[inc]] == 0].sum())
                    occ[arm]["zero"] += float(cnt[~inc][sup[u[~inc] - v0] == 0].sum())
            print(f"  [{arm:9s}] dict {t_dict:5.0f}s  lam={lam:<7g} cg {info['iterations']:3d} its "
                  f"{info['seconds']:5.0f}s  arm {time.time()-t0:5.0f}s", flush=True)
            del W_arm, W_cmp, W_new, W_init
            gc.collect()

        def _fold_macro(pq):
            return float(np.mean([np.mean([pq[c][q] for q in fold_qids[c][j]]) for c in OOD]))
        fold_c.append(_fold_macro(oof_c))
        for arm in ARMS:
            fold_g[arm].append(_fold_macro(oof[arm]) - fold_c[-1])
        print(f"  fold {j} gains vs C: { {a: round(fold_g[a][-1], 5) for a in ARMS} }", flush=True)

    c_macro, c_means = macro_over(dev, oof_c)
    res = {"oof_gain": {}, "oof_macro": {}, "fold_gain": dict(fold_g),
           "fold_comparator_macro": fold_c,
           "zero_update_mass": {a: (occ[a]["zero"] / occ[a]["total"]) if occ[a]["total"] else None
                                for a in ARMS},
           "compile_delta_vs_r0": c_macro - r0_macro,
           "pooling_canary_pass": bool(st["canary_pass"])}
    for a in ARMS:
        m, _ = macro_over(dev, oof[a])
        res["oof_macro"][a] = m
        res["oof_gain"][a] = m - c_macro
    res.update({"stage1_opportunity_DESCRIPTIVE": stage1_out,
                "r0_ood_macro": r0_macro, "r0_per_component": r0_means,
                "compile_oof_macro": c_macro, "compile_per_component": c_means,
                "gain_vs_R0_secondary": {a: res["oof_macro"][a] - r0_macro for a in ARMS},
                "lambda": {a: dict(lam_log[a]) for a in ARMS},
                "support": {a: dict(support[a]) for a in ARMS},
                "dictionary_sha256": {a: dict(sha_log[a]) for a in ARMS},
                "ood_query_text_overlap_with_fit": overlap,
                "conformance": st, "seconds": round(time.time() - t_start, 1),
                "smoke": bool(smoke), "K": k, "n_folds": n_folds, "n_fit": len(fit_texts),
                "self_test": st})
    return res


# ---- the fused read (condition 4) --------------------------------------------------------------

def fused(arm, n_folds=N_FOLDS, tag="", device=None):
    """Condition (4), out-of-fold, for the DENSE WINNER and the shared comparator C only.

    Read as a GAIN against C, not a raw macro: the raw fused macro is ~0.57, against which
    `>= -0.0020` would be vacuous. Components are the outer loop -- HotpotQA's corpus is 5.23M
    documents and `dev_eval.doc_vecs` re-parses it on every call (CODEMAP pitfall 6).
    """
    import dev_eval
    import fusion
    import select_fusion
    from evalkit import per_query_ndcg

    device = device or m8base.device()
    tok, pre = tok_pre()
    v0 = len(tok)
    spec = json.loads((RESULTS / "m7_fusion_p35w-2m-s2500.json").read_text())
    spec = {"family": spec["family"], "param": spec["param"]}
    W_inc, _ = load_incumbent()

    tables, feats = {}, {}
    for j in range(n_folds):
        W_new, payload = load_fold(arm, j, tag)
        f = rebuild_featurizer(arm, payload, v0)
        tables[("arm", j)] = serve_int8(W_inc, W_new)
        feats[("arm", j)] = f
        # C is the sum-init ZERO-RESIDUAL compile, rebuilt from the saved DICTIONARY, never from
        # the saved (solved) rows.
        _, payload_c = load_fold("seg", j, tag)
        _t, units, _r = seg_units_from_tokenizer(payload_c, v0)
        tables[("C", j)] = serve_int8(W_inc, init_new_rows("sum", units, W_inc, len(units)))
        feats[("C", j)] = rebuild_featurizer("seg", payload_c, v0)
        del W_new

    per_q = {"arm": {}, "C": {}}
    for comp in FUSED_COMPONENTS:
        t0 = time.time()
        b_run, _ = select_fusion.bm25_run_and_key(comp)
        doc_ids, doc_texts, q_ids, q_texts, qrels, dv = dev_eval.doc_vecs(comp)
        qfold = fold_of(list(q_ids), n_folds=n_folds)
        ids = ids_of(tok, list(q_texts), pre)
        from evalkit import topk_ids_scores
        for who in ("arm", "C"):
            acc = {}
            for j in range(n_folds):
                sel = np.flatnonzero(qfold == j)
                if not len(sel):
                    continue
                qt = [q_texts[i] for i in sel]
                qv = encode(feats[(who, j)](qt, [ids[i] for i in sel]), tables[(who, j)])
                d_run = topk_ids_scores(qv, dv, doc_ids, k=fusion.DEPTH,
                                        qids=[q_ids[i] for i in sel])
                acc.update({q: float(v) for q, v in
                            per_query_ndcg(fusion.apply_frozen(spec, d_run, b_run), qrels).items()})
            per_q[who][comp] = acc
        del dv, doc_ids, doc_texts, b_run, q_texts
        gc.collect()
        m8base.empty_cache()
        print(f"  fused {comp}: {time.time()-t0:.0f}s", flush=True)

    macros = {w: {c: float(np.mean(list(v.values()))) for c, v in per_q[w].items()}
              for w in per_q}
    m = {w: float(np.mean(list(macros[w].values()))) for w in macros}
    return {"arm": arm, "fused_macro": m, "per_component": macros,
            "fused_gain": m["arm"] - m["C"],
            "operator": spec, "components": list(FUSED_COMPONENTS)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["self-test", "run", "fused"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", default=None, help="fused: the dense winner")
    a = ap.parse_args()
    if a.step == "self-test":
        sys.exit(0 if self_test()["pass"] else 1)
    if a.step == "fused":
        r = fused(a.arm, n_folds=2 if a.smoke else N_FOLDS, tag="smoke-" if a.smoke else "")
        print(json.dumps(r, indent=2))
        (CKPT / f"fused-{a.arm}{'-smoke' if a.smoke else ''}.json").write_text(
            json.dumps(r, indent=2))
        sys.exit(0)

    r = run(smoke=a.smoke)
    # Condition (4) is evaluated only when 1 and 2 already hold: the router is a CONJUNCTION, so a
    # failure elsewhere fixes the verdict regardless, and the fused pass costs a HotpotQA corpus
    # load. If 1 and 2 hold, it MUST run before anything is called authorised.
    prelim = route({**r, "fused_gain": None})
    if prelim["conditions"]["1_g_best_at_or_above_bar"] and \
       prelim["conditions"]["2_positive_in_4_of_5_folds"]:
        fr = fused(prelim["winning_arm"], n_folds=r["n_folds"], tag="smoke-" if a.smoke else "")
        r["fused"] = fr
        r["fused_gain"] = {prelim["winning_arm"]: fr["fused_gain"]}
    else:
        r["fused"] = ("NOT RUN: conditions 1 and 2 do not both hold, and the router is a "
                      "conjunction -- the verdict is fixed without it. It is mandatory before any "
                      "authorisation and would have been run had they held.")
        r["fused_gain"] = None
    r["route"] = route(r)
    print(json.dumps({"oof_gain": r["oof_gain"], "fold_gain": r["fold_gain"],
                      "zero_update_mass": r["zero_update_mass"],
                      "compile_delta_vs_r0": r["compile_delta_vs_r0"],
                      "route": r["route"], "seconds": r["seconds"]}, indent=2))
    dest = (RESULTS / "m8_d2_pre.SMOKE.json") if a.smoke else OUT
    if a.smoke:
        dest.write_text(json.dumps(r, indent=2, default=str))
    else:
        probe_guard.write_result(dest, r, "D2-PRE")
    print(f"wrote {dest}", file=sys.stderr)
