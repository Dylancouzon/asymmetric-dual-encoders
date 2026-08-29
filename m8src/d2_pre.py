"""D2-PRE -- the closed-form preflight that routes (or cancels) D2's five training chains.

WHAT THE REGISTRY ASKS FOR, and the one reading it forces. `m8/registry.json` probe `D2-PRE` says
"one deterministic 64K tokenizer, then closed-form ridge residual solves on FROZEN INCUMBENT ROWS
... four arms at EQUAL ADDED-ROW BUDGET", and its staged check (2) requires a SUM-INITIALIZED,
zero-residual compile to reproduce R0. Those clauses are only mutually satisfiable one way: every
arm keeps R0's 30,522 incumbent rows frozen and ADDS exactly K rows. A wholesale 64K vocabulary
replacement has no incumbent rows to freeze and no constituents to sum-initialize from. So the
self-trained tokenizer here is the SOURCE OF THE ADDED UNITS, not a replacement vocabulary, and it
is trained as BPE over the INCUMBENT WORDPIECE ID SEQUENCE -- which makes every learned unit a
sequence of incumbent rows (so `sum of constituents` is always defined) and lets merges cross the
whitespace boundary, which is the multi-word property D2 exists to test.

THE FOUR ARMS, at equal added-row budget K:
  seg          (a) non-overlapping BPE segmentation; a fired phrase REMOVES its constituents.
                   Row init = SUM of constituent rows (registry `compositional_init_floor`).
  add_word     (b) additive overlapping WORD n-grams (n in {2,3}); incumbent segmentation
                   untouched, phrase rows fire ON TOP. Row init = ZERO.
  add_char     (c) additive overlapping CHARACTER n-grams (n in {4,5}) within whitespace words.
                   Row init = ZERO. Reaches rare strings a frequency tokenizer never allocates.
  seg_cold     (d) `seg` with a zero-residual fallback: a row with fewer than MIN_UPDATES fit
                   activations keeps its sum init exactly.

WHY THE TWO INITS DIFFER, and why that is forced rather than chosen. Under the released serving
rule `q = normalize( sum_types sqrt(c_t) w_t / sum_types sqrt(c_t) )`, replacing constituents a,b
(each count 1) by a phrase row `w_p = w_a + w_b` leaves the NUMERATOR identical and changes only the
denominator, which is a positive per-query scalar that the final L2 normalize cancels -- so the
served vector is EXACTLY unchanged. That is a true floor, and it is why the segmentation arms are
sum-initialized. An ADDITIVE row fires alongside its constituents, so a sum init would double-count
them; a zero row contributes nothing to the numerator and only inflates the (cancelling)
denominator, so an additive arm at zero residual reproduces R0 exactly. Same principle, opposite
init, both exact.

WHAT IT MAY AND MAY NOT DO. Nothing ships from a closed-form fit. This is DIAGNOSTIC AND ROUTING:
it authorises or cancels D2's chains against the frozen numeric router in the registry row, and it
can REVERSE the plan onto the additive class. The router is read by `route()`, which is a pure
function of the measured numbers so that it cannot be re-read in this session's favour.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter

import numpy as np
import scipy.sparse as sp

import m8base
import blockcg
import probe_guard

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_d2_pre.json"

# The incumbent chain D2 is measured against: section 23's diagonal cell (b0,a0), which is also
# M7's shipped candidate's frame (LEDGER 4.7: it reproduces the released artifact on every
# rank-based lens).
R0_RUN = "m8nf-seed0"
OOD = ("cqadup-programmers", "cqadup-physics")

# Private Use Area-B: 30,522 incumbent ids map to 30,522 distinct codepoints, so a query becomes a
# STRING over an alphabet of incumbent rows and an ordinary BPE trainer learns merges over it.
PUA = 0x100000

# Added-row budget, FROZEN before any solve. 30,522 + 34,014 = 64,536 ~ the 64K vocabulary D2's own
# row names, so the segmentation arm carries the row count D2 would actually ship, and the additive
# arms are compared at that same count.
K_ADDED = 34_014
WORD_NGRAM_N = (2, 3)
CHAR_NGRAM_N = (4, 5)
# Arm (d)'s cold-row threshold: a row activated by fewer than this many FIT queries keeps its init.
MIN_UPDATES = 5
N_FOLDS = 5
# lambda: B7's real-data run selected 1e-2 as the argmax over m7src/stage0_ridge.LAMBDAS on this
# exact system (LEDGER 18: "argmax lambda=1e-2 at 0.343924, reproducing M7's 0.3439 for stella").
# Reusing that PRE-EXISTING measurement rather than selecting per arm removes the single largest
# degree of freedom in this probe: a lambda chosen per arm on the scored endpoint would let the
# router pick its own answer. The sensitivity curve is reported descriptively (--lam-curve).
LAMBDA = 1e-2
LAMBDA_GRID = (1e-3, 1e-2, 1e-1)

ARMS = ("seg", "add_word", "add_char", "seg_cold")


# ---- feature extraction ------------------------------------------------------------------
#
# Every arm produces, per query, a mapping row_id -> occurrence count in the EXTENDED row space
# [0, V0 + K). `bag_matrix` then applies the released pooling rule to it. Keeping the arms behind
# one contract is what makes "equal added-row budget" a checkable statement rather than a claim.

def _tok_pre():
    from table import Preproc, get_tokenizer
    return get_tokenizer(), Preproc(prefix="", add_special_tokens=True, max_length=512,
                                    pool_mode="sqrt")


def _ids(tok, texts, pre, chunk=50_000):
    """Incumbent WordPiece ids, exactly what the released path tokenizes."""
    from table import tokenize
    out = []
    for lo in range(0, len(texts), chunk):
        out.extend(tokenize(tok, texts[lo:lo + chunk], pre))
    return out


def _word_spans(tok, texts):
    """(word -> id span) for each query, by tokenizing each whitespace word separately.

    BERT's basic tokenizer splits on whitespace and punctuation and its WordPiece step has no
    cross-word context, so concatenating per-word ids reproduces the whole-text tokenization. That
    is ASSERTED in `self_test`, not assumed -- it is the premise the word-n-gram arm rests on.
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


def _counts(rows):
    u, c = np.unique(np.asarray(rows, dtype=np.int64), return_counts=True)
    return u, c.astype(np.float32)


# --- arm (a)/(d): BPE over incumbent id sequences ---

def train_seg_units(train_ids, k_added, v0):
    """Deterministic BPE over the incumbent id alphabet. Returns (units, apply_fn, sha).

    `units[j]` is the tuple of incumbent ids the added row V0+j stands for. `apply_fn(id_list)`
    returns the NON-OVERLAPPING segmentation of one query in extended row space.
    """
    from tokenizers import Tokenizer, models, trainers
    lines = ["".join(chr(PUA + i) for i in x) for x in train_ids]
    t = Tokenizer(models.BPE(unk_token=None, continuing_subword_prefix="", end_of_word_suffix=""))
    tr = trainers.BpeTrainer(vocab_size=v0 + k_added, min_frequency=2, show_progress=False,
                             initial_alphabet=[chr(PUA + i) for i in range(v0)], special_tokens=[])
    t.train_from_iterator(lines, tr)
    vocab = t.get_vocab()
    multi = sorted((s for s in vocab if len(s) > 1), key=lambda s: vocab[s])
    units = [tuple(ord(c) - PUA for c in s) for s in multi]
    # tokenizer id -> extended row id. Single-char units keep their incumbent row; multi-char
    # units take V0 + their rank in merge order.
    remap = np.zeros(max(vocab.values()) + 1, dtype=np.int64)
    mi = {s: j for j, s in enumerate(multi)}
    for s, tid in vocab.items():
        remap[tid] = (ord(s) - PUA) if len(s) == 1 else (v0 + mi[s])
    sha = hashlib.sha256(t.to_str().encode()).hexdigest()

    def apply_fn(id_lists, batch=20_000):
        out = []
        for lo in range(0, len(id_lists), batch):
            chunk = id_lists[lo:lo + batch]
            enc = t.encode_batch(["".join(chr(PUA + i) for i in x) for x in chunk])
            out.extend(remap[np.asarray(e.ids, dtype=np.int64)] for e in enc)
        return out

    return units, apply_fn, sha


# --- arm (b): additive overlapping word n-grams ---

def mine_word_ngrams(tok, train_texts, k_added):
    """Top-k most frequent WORD n-grams, as tuples of incumbent ids."""
    spans, flat = _word_spans(tok, train_texts)
    cnt = Counter()
    for s, f in zip(spans, flat):
        for n in WORD_NGRAM_N:
            for i in range(len(s) - n + 1):
                cnt[tuple(f[s[i][0]:s[i + n - 1][1]])] += 1
    # ties broken by the tuple itself so the selection is deterministic, not dict-order dependent
    ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:k_added]]


# --- arm (c): additive overlapping character n-grams ---

def mine_char_ngrams(train_texts, k_added):
    cnt = Counter()
    for t in train_texts:
        for w in t.lower().split():
            for n in CHAR_NGRAM_N:
                for i in range(len(w) - n + 1):
                    cnt[w[i:i + n]] += 1
    ranked = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:k_added]]


# --- featurizers ---

def feats_incumbent(ids):
    return [_counts(x) for x in ids]


def feats_seg(seg_ids):
    return [_counts(x) for x in seg_ids]


def feats_add_seq(tok, texts, ids, seqs, v0):
    """Incumbent ids PLUS every occurrence of a selected id-subsequence (overlapping)."""
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
        out.append(_counts(rows))
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
        out.append(_counts(rows))
    return out


# ---- the released pooling rule, as a matrix and as an encoder ----------------------------

def bag_matrix(feats, V):
    """Row i is the released `sqrt` pooling operator for query i: sqrt(c_t) / sum_t sqrt(c_t).

    The denominator cancels under the final L2 normalize at SERVE time, but it is part of the
    linear operator the ridge fits, so it belongs here.
    """
    indptr, indices, data = [0], [], []
    for u, c in feats:
        w = np.sqrt(c)
        indices.append(u)
        data.append((w / max(w.sum(), 1e-12)).astype(np.float64))
        indptr.append(indptr[-1] + len(u))
    return sp.csr_matrix((np.concatenate(data) if data else np.zeros(0),
                          np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64),
                          np.array(indptr)), shape=(len(feats), V))


CLS_ID = 101


def encode(feats, W):
    """The released query path for an arbitrary row space. Verified against
    `QueryTable.encode(..., pool_mode='sqrt')` in `self_test` to 5e-8."""
    out = np.empty((len(feats), W.shape[1]), dtype=np.float32)
    fb = W[min(CLS_ID, W.shape[0] - 1)]
    fbn = np.linalg.norm(fb)
    fb = fb / fbn if fbn > 1e-6 else np.eye(1, W.shape[1], 0, dtype=np.float32)[0]
    for i, (u, c) in enumerate(feats):
        w = np.sqrt(c)
        v = (W[u] * w[:, None]).sum(0) / max(w.sum(), 1e-6)
        n = np.linalg.norm(v)
        out[i] = (v / n) if n > 1e-6 else fb
    return out


def int8_roundtrip(W):
    from table import dequantize_int8, quantize_int8
    q, s = quantize_int8(W)
    return dequantize_int8(q, s)


# ---- scoring -----------------------------------------------------------------------------

class Scorer:
    """The dense out-of-domain macro at int8/sqrt, and the fused read, on the SAME tables.

    Dev query texts and their corpora are loaded once; scoring a table is then ~1 s, which is what
    makes a 4-arm x 5-fold design affordable inside the row's one-hour target.
    """

    def __init__(self, tok, pre, components=OOD):
        import dev_eval
        self.dev_eval = dev_eval
        self.components = tuple(components)
        self.texts = {}
        for c in self.components:
            _, _, _, q_texts, _, _ = dev_eval.doc_vecs(c)
            self.texts[c] = q_texts
        self.ids = {c: _ids(tok, v, pre) for c, v in self.texts.items()}

    def macro(self, feats_by_comp, W, precision="int8"):
        Wq = int8_roundtrip(W) if precision == "int8" else W.astype(np.float32)
        per = {}
        for c in self.components:
            qv = encode(feats_by_comp[c], Wq)
            pq = self.dev_eval.eval_query_vecs(c, qv)
            per[c] = float(np.mean(list(pq.values())))
        return float(np.mean(list(per.values()))), per


# ---- the arms ----------------------------------------------------------------------------

def build_arm(name, tok, pre, fit_texts, fit_ids, score_texts, score_ids, v0, k_added):
    """Mine an arm's added rows on FIT text only, and return everything the solve and the score
    need. `score_texts`/`score_ids` are the dev components' queries -- they are NEVER mined from.

    Returns dict with: units_init (K x d builder), feats_fit, feats_score, k, sha.
    """
    t0 = time.time()
    if name in ("seg", "seg_cold"):
        units, apply_fn, sha = train_seg_units(fit_ids, k_added, v0)
        k = len(units)
        feats_fit = feats_seg(apply_fn(fit_ids))
        feats_score = {c: feats_seg(apply_fn(v)) for c, v in score_ids.items()}
        init = ("sum", units)
    elif name == "add_word":
        seqs = mine_word_ngrams(tok, fit_texts, k_added)
        k, sha = len(seqs), hashlib.sha256(repr(seqs).encode()).hexdigest()
        feats_fit = feats_add_seq(tok, fit_texts, fit_ids, seqs, v0)
        feats_score = {c: feats_add_seq(tok, score_texts[c], score_ids[c], seqs, v0)
                       for c in score_ids}
        init = ("zero", seqs)
    elif name == "add_char":
        grams = mine_char_ngrams(fit_texts, k_added)
        k, sha = len(grams), hashlib.sha256(repr(grams).encode()).hexdigest()
        feats_fit = feats_add_char(fit_texts, fit_ids, grams, v0)
        feats_score = {c: feats_add_char(score_texts[c], score_ids[c], grams, v0)
                       for c in score_ids}
        init = ("zero", grams)
    else:
        raise KeyError(name)
    return {"arm": name, "k": k, "sha256": sha, "init": init,
            "feats_fit": feats_fit, "feats_score": feats_score,
            "build_seconds": round(time.time() - t0, 1)}


def init_rows(arm, W_inc, k):
    """W_init for the added rows: SUM of constituents for a segmentation arm, ZERO for an
    additive one. The registry's `compositional_init_floor` is the sum, and the correction it
    records is load-bearing: the MEAN is not a floor, because (w_a+w_b)/2 is not a positive scalar
    multiple of w_a+w_b and normalization therefore does not restore it."""
    kind, payload = arm["init"]
    W = np.zeros((k, W_inc.shape[1]), dtype=np.float32)
    if kind == "sum":
        for j, unit in enumerate(payload):
            W[j] = W_inc[list(unit)].sum(0)
    return W


def solve_residual(X, Y, W_inc, W_init, v0, lam, device):
    """Solve only the ADDED rows, against the residual the frozen incumbent rows leave behind."""
    Xi, Xn = X[:, :v0].tocsr(), X[:, v0:].tocsr()
    Rt = Y - (Xi @ W_inc).astype(np.float32) - (Xn @ W_init).astype(np.float32)
    D, info = blockcg.block_cg_ridge(Xn, Rt, np.zeros_like(W_init), lam, device=device)
    return D.astype(np.float32), info


# ---- conformance ---------------------------------------------------------------------------

def self_test(n=512):
    """Three properties this probe's every number rests on, each checked against REAL data rather
    than a fixture built out of the claim (m8/CODEMAP.md pitfalls 15, 17, 19).

    1. `encode` IS the released path. Checked against `QueryTable.encode(pool_mode='sqrt')` on real
       dev queries with R0's real int8 rows -- not on random unit vectors, which is the mistake
       `e14_head.py`'s first self-test made.
    2. Per-word tokenization concatenates to whole-text tokenization. The word-n-gram arm's spans
       are meaningless otherwise, and nothing else would notice.
    3. The SUM init is exact and the MEAN init is not. This is the registry's corrected
       `compositional_init_floor`, and it is checked as an ALGEBRAIC identity on real rows, so a
       sign error in it fails here rather than in a verdict.
    """
    import torch
    from table import Preproc, QueryTable, ensure_release, load_table
    import dev_eval

    tok, pre = _tok_pre()
    v0 = len(tok)
    rel = ensure_release(m8base.WORK / "runs" / f"{R0_RUN}.npz", device=m8base.device())
    W_inc = load_table(rel, variant="int8", device="cpu").rows.detach().numpy().astype(np.float32)

    _, _, _, q_texts, _, _ = dev_eval.doc_vecs(OOD[0])
    texts = list(q_texts)[:n]
    ids = _ids(tok, texts, pre)

    ref = QueryTable(W_inc, learned_weights=False).to(m8base.device()).eval().encode(
        texts, pre, tok=tok)
    mine = encode(feats_incumbent(ids), W_inc)
    d1 = float(np.abs(ref - mine).max())

    spans, flat = _word_spans(tok, texts)
    from table import tokenize
    bare = Preproc(prefix="", add_special_tokens=False, max_length=512, pool_mode="sqrt")
    d2 = sum(1 for a, b in zip(flat, tokenize(tok, texts, bare)) if list(a) != list(b))

    # Sum vs mean init, through the real serving rule. THE FIXTURE MUST CARRY CONTEXT TOKENS.
    # The first version merged a two-token query into a one-row query, where sum and mean differ
    # only by a positive scalar and the final L2 normalize erases the difference -- both scored
    # 0.0 deviation and the negative control could never have fired. That is exactly pitfall 19's
    # failure (a test built out of the claim), and it is why the registry's own wording says the
    # phrase is downweighted "against every other token": the other tokens are the mechanism.
    rng = np.random.default_rng(0)
    quads = [tuple(int(v) for v in row) for row in rng.integers(1000, v0, size=(64, 4))]
    W2 = np.concatenate([W_inc, np.zeros((len(quads), W_inc.shape[1]), np.float32)])
    W2m = W2.copy()
    for j, (a, b, _c, _d) in enumerate(quads):
        W2[v0 + j] = W_inc[a] + W_inc[b]
        W2m[v0 + j] = 0.5 * (W_inc[a] + W_inc[b])
    base = [(np.array([a, b, c, d]), np.ones(4, np.float32)) for a, b, c, d in quads]
    merged = [(np.array([v0 + j, q[2], q[3]]), np.ones(3, np.float32))
              for j, q in enumerate(quads)]
    d3_sum = float(np.abs(encode(base, W_inc) - encode(merged, W2)).max())
    d3_mean = float(np.abs(encode(base, W_inc) - encode(merged, W2m)).max())

    ok = d1 < 5e-7 and d2 == 0 and d3_sum < 1e-6
    print(json.dumps({
        "encode_matches_released_path_max_abs": d1,
        "per_word_tokenization_mismatches": d2,
        "sum_init_max_abs_deviation": d3_sum,
        "mean_init_max_abs_deviation": d3_mean,
        "mean_init_is_worse_as_registered": bool(d3_mean > d3_sum),
        "pass": bool(ok),
    }, indent=2))
    del torch
    return 0 if ok else 1
