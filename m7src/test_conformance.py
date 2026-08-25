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
check("empty bag returns the documented fallback ([CLS] row, normalized)",
      np.allclose(v_empty, fb, atol=1e-6))

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
check("int8 per-row absmax round-trip cos > 0.9999", float(cos.min()) > 0.9999, f"min={cos.min():.6f}")
check("int8 dtype and shape", q8.dtype == np.int8 and q8.shape == rows.shape and sc.shape == (V,))

# --- unseen / out-of-range token behavior is deterministic ---------------------------
unk = tok.unk_token_id
v_unk = m.encode([tok.unk_token], Preproc(add_special_tokens=False), tok)[0]
r = rows[unk] / np.linalg.norm(rows[unk])
check("single-UNK query is the normalized UNK row", np.allclose(v_unk, r, atol=1e-5))

print()
if FAILS:
    print(f"CONFORMANCE FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("OK: query preprocessing conforms on all checks.")
