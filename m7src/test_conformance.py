"""Mandatory conformance suite for the query-preprocessing rule (instructions-m7.md).

Covers: special tokens, padding, repeated tokens, truncation, max length, empty queries,
near-zero-norm sums, prefix handling (byte-for-byte, double application forbidden),
determinism, batch invariance, and int8 round-trip.

Run: python test_conformance.py   (exits nonzero on any failure)
"""
import sys

import numpy as np
import torch

from table import (EPS, NO_PREFIX, WITH_PREFIX, Preproc, QueryTable, dequantize_int8,
                   get_tokenizer, quantize_int8, ragged, tokenize)
from teacher import QUERY_PREFIX

FAILS = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        FAILS.append(name)


tok = get_tokenizer()
V, D = tok.vocab_size, 32
rng = np.random.default_rng(0)
rows = rng.normal(0, 1 / np.sqrt(D), (V, D)).astype(np.float32)
m = QueryTable(rows, weight_init=np.abs(rng.normal(1.0, 0.2, V)).astype(np.float32) + 0.1)
m_flat = QueryTable(rows, learned_weights=False)

# --- tokenizer / prefix conformance ---------------------------------------------------
t = "what is a vector database"
check("special tokens present when configured",
      tokenize(tok, [t], Preproc(add_special_tokens=True))[0][0] == tok.cls_token_id
      and tokenize(tok, [t], Preproc(add_special_tokens=True))[0][-1] == tok.sep_token_id)
ids_ns = tokenize(tok, [t], Preproc(add_special_tokens=False))[0]
check("special tokens absent when configured",
      tok.cls_token_id not in ids_ns and tok.sep_token_id not in ids_ns)
check("prefix applied byte-for-byte",
      tokenize(tok, [t], WITH_PREFIX)[0] == tok(QUERY_PREFIX + t)["input_ids"])
check("no-prefix variant is bare text", tokenize(tok, [t], NO_PREFIX)[0] == tok(t)["input_ids"])
try:
    tokenize(tok, [QUERY_PREFIX + t], WITH_PREFIX)
    check("double prefix application raises", False)
except ValueError:
    check("double prefix application raises", True)
check("prefix changes the token bag",
      tokenize(tok, [t], WITH_PREFIX)[0] != tokenize(tok, [t], NO_PREFIX)[0])

# --- truncation / max length ----------------------------------------------------------
long_q = " ".join(["retrieval"] * 5000)
for L in (512, 128):
    ids = tokenize(tok, [long_q], Preproc(max_length=L))[0]
    check(f"truncation at max_length={L}", len(ids) == L, f"got {len(ids)}")
check("truncation counts the prefix",
      len(tokenize(tok, [long_q], Preproc(prefix=QUERY_PREFIX, max_length=512))[0]) == 512)

# --- repeated tokens: multiplicity is kept -------------------------------------------
a = m.encode(["cat"], NO_PREFIX, tok)[0]
b = m.encode(["cat cat cat"], NO_PREFIX, tok)[0]
check("repeated tokens change the vector (multiplicity kept)", not np.allclose(a, b, atol=1e-6),
      f"cos={float(a @ b):.6f}")
ids1 = tokenize(tok, ["cat cat"], NO_PREFIX)[0]
check("repeated token appears twice in the bag", sum(1 for i in ids1 if i == tok("cat", add_special_tokens=False)["input_ids"][0]) == 2)

# --- padding never contributes: ragged bags, and batch invariance --------------------
texts = ["a", "quantum field theory", "how do i reset my router", "x " * 40]
one_at_a_time = np.stack([m.encode([t_], NO_PREFIX, tok)[0] for t_ in texts])
batched = m.encode(texts, NO_PREFIX, tok)
check("batch invariance (no padding leakage)", np.allclose(one_at_a_time, batched, atol=1e-6),
      f"max|d|={np.abs(one_at_a_time-batched).max():.2e}")
split = np.concatenate([m.encode(texts[:1], NO_PREFIX, tok), m.encode(texts[1:], NO_PREFIX, tok)])
check("batch-grouping invariance", np.allclose(split, batched, atol=1e-6))

# --- empty and degenerate queries ----------------------------------------------------
for label, q, pre in [("empty query, specials on", "", NO_PREFIX),
                      ("empty query, specials off", "", Preproc(add_special_tokens=False)),
                      ("whitespace-only query", "   ", NO_PREFIX)]:
    v = m.encode([q], pre, tok)[0]
    check(label + " -> unit norm, finite", np.isfinite(v).all() and abs(np.linalg.norm(v) - 1) < 1e-5,
          f"norm={np.linalg.norm(v):.6f}")
fb = m.fallback_vector().detach().cpu().numpy()
v_empty = m.encode([""], Preproc(add_special_tokens=False), tok)[0]
check("empty bag returns the documented fallback (normalized fallback row)",
      np.allclose(v_empty, fb, atol=1e-6))
# non-circular: the fallback must be the actual [CLS] row of the actual tokenizer, not row 0
cls_row = rows[tok.cls_token_id] / np.linalg.norm(rows[tok.cls_token_id])
check("fallback row IS the tokenizer's [CLS] row, not row 0",
      np.allclose(fb, cls_row, atol=1e-6) and tok.cls_token_id != 0,
      f"cls_token_id={tok.cls_token_id}, row0 is {tok.convert_ids_to_tokens([0])[0]}")

# --- near-zero-norm sums --------------------------------------------------------------
z = np.zeros((V, D), dtype=np.float32)
one = tok("apple", add_special_tokens=False)["input_ids"][0]
two = tok("banana", add_special_tokens=False)["input_ids"][0]
z[one] = 1.0
z[two] = -1.0
z[0, 0] = 1.0  # a usable [CLS] row so the fallback is well defined
mz = QueryTable(z, learned_weights=False)
v = mz.encode(["apple banana"], Preproc(add_special_tokens=False), tok)[0]
check("cancelling bag (near-zero-norm sum) -> fallback, unit norm",
      abs(np.linalg.norm(v) - 1) < 1e-5 and np.allclose(v, mz.fallback_vector().detach().cpu().numpy(), atol=1e-6))

# --- count saturation (capacity lever #4): the pooling rule lives in Preproc ----------
from table import POOL_MODES, encode_pooled, occurrence_weights

rep = "apple apple apple banana"          # counts 3 and 1, before specials
ids_rep = tokenize(tok, [rep], Preproc(add_special_tokens=False))[0]
for mode, want in (("mean", [3.0, 1.0]), ("binary", [1.0, 1.0]),
                   ("cap2", [2.0, 1.0]), ("sqrt", [3.0 ** 0.5, 1.0])):
    psw = occurrence_weights([ids_rep], mode).numpy()
    tot = {t: float(psw[[i for i, x in enumerate(ids_rep) if x == t]].sum())
           for t in dict.fromkeys(ids_rep)}
    check(f"pool mode {mode}: a token seen c times contributes f(c)",
          np.allclose(sorted(tot.values(), reverse=True), want, atol=1e-6),
          f"{tot} want {want}")

check("pool_mode='mean' reproduces the original forward bit-for-bit",
      m.encode(texts, NO_PREFIX, tok).tobytes()
      == encode_pooled(m, texts, NO_PREFIX, mode="mean", tok=tok).tobytes())
sq = Preproc(pool_mode="sqrt")
check("Preproc routes encode through the declared pool mode",
      np.allclose(m.encode(texts, sq, tok), encode_pooled(m, texts, sq, mode="sqrt", tok=tok)))
check("a non-default pool mode changes the query vectors",
      not np.allclose(m.encode(texts, NO_PREFIX, tok), m.encode(texts, sq, tok), atol=1e-6))
# A LITERAL, not a self-comparison: every preproc_fingerprint already written to disk is this
# value, so the check has to fail if the hash drifts for any reason, including both sides drifting
# together (Codex code-review #2, under-tested item 3).
check("pool_mode='mean' keeps every pre-existing preproc fingerprint byte-identical",
      Preproc().fingerprint() == "4f7978fa7f69b559"
      and Preproc(pool_mode="mean").fingerprint() == "4f7978fa7f69b559"
      and Preproc(**{"prefix": "", "add_special_tokens": True, "max_length": 512}).fingerprint()
      == "4f7978fa7f69b559"
      and Preproc().fingerprint() != sq.fingerprint(),
      f"{Preproc().fingerprint()} / {sq.fingerprint()}")

# The TRAINING forward must agree with the serving path under every pooling rule -- capacity
# lever #6 trains through `forward(extra_psw=...)` while everything else serves via encode_pooled.
from table import ragged as _ragged
for md in POOL_MODES:
    _ids = tokenize(tok, texts, Preproc(pool_mode=md))
    _f, _o, _l = _ragged(_ids, "cpu")
    _psw = occurrence_weights(_ids, md)
    _train = m(_f, _o, _l, extra_psw=(None if md == "mean" else _psw)).detach().numpy()
    check(f"training forward == serving path under pool mode {md}",
          np.allclose(_train, encode_pooled(m, texts, Preproc(pool_mode=md), mode=md, tok=tok),
                      atol=1e-6))
check("every declared pool mode is implemented and unit-norm",
      all(abs(np.linalg.norm(m.encode(["apple apple banana"], Preproc(pool_mode=md), tok)[0]) - 1)
          < 1e-5 for md in POOL_MODES))
# a query with no repeated token must be identical under every mode -- the rule may only act on
# multiplicity, which is what makes it non-absorbable in the first place
uniq = ["quantum chromodynamics lattice"]
check("no repeated token -> every pool mode agrees",
      all(np.allclose(m.encode(uniq, Preproc(pool_mode=md), tok)[0],
                      m.encode(uniq, NO_PREFIX, tok)[0], atol=1e-6) for md in POOL_MODES))

# --- determinism ----------------------------------------------------------------------
v1 = m.encode(texts, WITH_PREFIX, tok)
v2 = m.encode(texts, WITH_PREFIX, tok)
check("bit-identical across repeated calls", v1.tobytes() == v2.tobytes())

# --- flat vs learned weights are actually different ----------------------------------
check("learned token weights change the output",
      not np.allclose(m.encode(texts, NO_PREFIX, tok), m_flat.encode(texts, NO_PREFIX, tok), atol=1e-6))
w = m.token_weights()
check("token weights are strictly positive", bool((w > 0).all()))

# --- flat-weight math is exactly the unweighted mean ---------------------------------
ids = tokenize(tok, [texts[1]], NO_PREFIX)[0]
manual = rows[ids].mean(0)
manual /= np.linalg.norm(manual)
check("flat weights == unweighted mean of rows",
      np.allclose(m_flat.encode([texts[1]], NO_PREFIX, tok)[0], manual, atol=1e-5))

# --- int8 round-trip ------------------------------------------------------------------
q8, sc = quantize_int8(rows)
deq = dequantize_int8(q8, sc)
cos = (rows * deq).sum(1) / (np.linalg.norm(rows, axis=1) * np.linalg.norm(deq, axis=1) + 1e-12)
check("int8 per-row absmax round-trip cos > 0.9999 (Gaussian rows; the REAL artifact is "
      "gated by G4 on dev, not by this)", float(cos.min()) > 0.9999, f"min={cos.min():.6f}")
check("int8 dtype and shape", q8.dtype == np.int8 and q8.shape == rows.shape and sc.shape == (V,))

# --- unseen / out-of-range token behavior is deterministic ---------------------------
unk = tok.unk_token_id
v_unk = m.encode([tok.unk_token], Preproc(add_special_tokens=False), tok)[0]
r = rows[unk] / np.linalg.norm(rows[unk])
check("single-UNK query is the normalized UNK row", np.allclose(v_unk, r, atol=1e-5))

# --- the released path: save -> load(fp16/int8) -> encode ----------------------------
import tempfile
from pathlib import Path as _P

from table import load_table, read_meta, save_table

with tempfile.TemporaryDirectory() as td:
    pth = _P(td) / "roundtrip.npz"
    upd = rng.integers(0, 50, V)
    save_table(pth, m, WITH_PREFIX, meta={"probe": 1}, updates=upd)
    meta = read_meta(pth)
    check("saved metadata carries the preprocessing rule and its fingerprint",
          meta["preproc"]["prefix"] == QUERY_PREFIX
          and meta["preproc_fingerprint"] == WITH_PREFIX.fingerprint())
    ref = m.encode(texts, WITH_PREFIX, tok)
    m16 = load_table(pth, variant="fp16", device="cpu")
    m8 = load_table(pth, variant="int8", device="cpu")
    v16 = m16.encode(texts, WITH_PREFIX, tok)
    v8 = m8.encode(texts, WITH_PREFIX, tok)
    check("fp16 round-trip reproduces the in-memory encode", np.abs(v16 - ref).max() < 2e-3,
          f"max|d|={np.abs(v16-ref).max():.2e}")
    check("int8 round-trip stays close to the in-memory encode", np.abs(v8 - ref).max() < 2e-2,
          f"max|d|={np.abs(v8-ref).max():.2e}")
    check("both variants share ONE preprocessing rule (identical token bags)",
          tokenize(tok, texts, WITH_PREFIX) == tokenize(tok, texts, WITH_PREFIX))
    check("learned token weights survive the round-trip",
          np.allclose(m16.token_weights().detach().numpy(),
                      m.token_weights().detach().numpy(), atol=1e-5))
    check("per-row update counts are persisted for the unseen-row policy",
          np.array_equal(np.load(pth)["updates"], upd))

print()
if FAILS:
    print(f"CONFORMANCE FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("OK: query preprocessing conforms on all checks.")
