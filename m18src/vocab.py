"""Candidate term discovery, ranking, selection and count-weighted row initialization.

Forked from ``m17src/vocab.py`` at commit 4213920. The M18 registry owns the support minima and
one resident-row cap. M17's domain quotas, abbreviation double-minimum and unsupported owner-pin
exemptions are intentionally removed. This module
is the deterministic procedure:

  1. **Discover** candidate lexical units from the admitted TRAINING partition only. A term is a
     candidate if the frozen tokenizer splits it into more than one piece (or into `[UNK]`);
     already-single-token terms are recorded as a diagnostic, never as a row to add.
  2. **Rank** by `residual * log(1 + distinct_source_documents)`, where residual is the mean of
     `1 - dot(q_teacher, q_v1)` over the deduplicated training queries containing the term.
     Ties: distinct documents descending, then lexical ascending.
  3. **Select** under Qdrant training-only support minima and reject near-unique event strings.
  4. **Initialize** each new row as `sum_j sqrt(c_j) * R_eff_j` over the term's own constituent
     pieces as the ORIGINAL tokenizer produced them in isolation, excluding [CLS]/[SEP], with
     `c_j` their counts in that isolated tokenization; the new learned scalar is 1.

Two hazards this module must not paper over:

* **P0b cross-term sharing drift.** Sum-initialization is output-preserving only for a query
  whose other tokens do not share the new term's constituent pieces. Under sqrt-count pooling a
  shared piece changes its own weight elsewhere in the query, so the extended table is NOT
  equivalent to the old one on such queries. `drift_report` measures it; V0 scores it.
* **Never fold twice.** The warm start is an UNFOLDED checkpoint whose learned scalars are still
  separate; the released table has those scalars already folded into its rows. Effective rows
  must be computed once, from the unfolded checkpoint, and old-vocabulary parity against the v1
  release must be verified BEFORE any new row is constructed.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from common import sha_text

# A lexical unit: letters/digits with internal . _ - + # / :, e.g. s3, k8s, kubectl,
# c++, .net, s3://bucket is split on the // by design (the released tokenizer's own
# pre-tokenizer decides what a row can ever match).
TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+#/:-]*[A-Za-z0-9+#]|[A-Za-z0-9]")
VOWELS = set("aeiouy")


@dataclass
class TermStat:
    term: str
    docs: set = field(default_factory=set)        # distinct source document ids
    contexts: set = field(default_factory=set)    # distinct deduplicated query ids
    # Domain counts are PER DISTINCT SUPPORTING DOCUMENT, not per query occurrence: a document
    # that fifty queries ask about votes once, exactly as it contributes one to `docs`. The domain
    # of a document is `support_manifest.document_domain` (ruling A3) and depends on the document
    # text alone, never on the query.
    domains: "defaultdict" = field(default_factory=lambda: defaultdict(int))
    residual_sum: float = 0.0
    residual_n: int = 0

    @property
    def n_docs(self):
        return len(self.docs)

    @property
    def n_contexts(self):
        return len(self.contexts)

    @property
    def residual(self):
        return self.residual_sum / self.residual_n if self.residual_n else 0.0


def is_abbreviation(term: str) -> bool:
    """FALLBACK heuristic, used only when the caller supplies no abbreviation inventory: short,
    and either carrying a digit or vowel-free. `k8s`, `s3`, `tls` match; `kubernetes`, `kubectl`
    do not — and neither does a vowel-carrying abbreviation such as `iam`, which is exactly why
    a supplied inventory is the authority in `select`."""
    t = term.lower()
    if len(t) > 6:
        return False
    return any(c.isdigit() for c in t) or not (set(t) & VOWELS)


def discover(queries, residuals, tokenizer, min_len=2, abbreviations=None):
    """Count support and accumulate teacher-vs-v1 residual per candidate term.

    `queries` are `cache.QuerySpec`-shaped records carrying `text`, `domain` and a
    `source_doc` attribute or key (the document the query came from); `residuals[i]` is
    `1 - dot(q_teacher, q_v1)` for `queries[i]`, already deduplicated by the caller.

    **The `domain` field is the domain of the query's SOURCE DOCUMENT**, as
    `support_manifest.document_domain(source, document_text)` computes it under ruling A3: the
    source-to-domain map, except that a 'general'-mapped source's documents are classified one at
    a time by the panel builder's keyword classifier on the document text alone. It is not a
    property of the query text, and a caller must not derive it from the query.

    An example whose `source_doc` is explicitly `None` ships no document and casts NO document
    vote (`data.documentless_sources_vote == "none"`, ruling A4); it still contributes contexts
    and residuals. A MISSING `source_doc` key remains an error.

    Domains are therefore counted once per DISTINCT supporting document, not once per query
    occurrence: `st.domains[domain]` is incremented only the first time a given `source_doc` is
    added to `st.docs`, so `sum(st.domains.values()) == st.n_docs` and `domain_of`'s majority is a
    majority of documents. Counting occurrences instead would let one heavily queried document
    outvote many documents from another domain.
    """
    stats: dict[str, TermStat] = {}
    seen_text = set()
    single_token = set()
    doc_domain: dict[str, str] = {}
    unk = _unk_id(tokenizer)
    for i, q in enumerate(queries):
        text = _get(q, "text")
        if text in seen_text:            # deduplicated training queries only
            continue
        seen_text.add(text)
        qid = _get(q, "qid", str(i))
        doc = _get(q, "source_doc", _MISSING)
        if doc is _MISSING:
            raise ValueError(
                f"query {qid!r} carries no `source_doc`. Ruling A3 counts one vote per distinct "
                "supporting DOCUMENT and takes the domain from that document, so a record "
                "without its document identity cannot be counted at all. Pass `source_doc=None` "
                "explicitly for an example that ships no document.")
        domain = _get(q, "domain", "general")
        if doc is not None and doc_domain.setdefault(doc, domain) != domain:
            raise ValueError(
                f"document {doc!r} is labelled {doc_domain[doc]!r} on one query and {domain!r} "
                "on another. A document's domain comes from its own text "
                "(support_manifest.document_domain), so two labels mean the caller derived one "
                "from the query.")
        for term in {m.group(0).lower() for m in TERM_RE.finditer(text)}:
            if len(term) < min_len:
                continue
            pieces = tokenize_term(tokenizer, term)
            # A single `[UNK]` is not an existing vocabulary token: the base tokenizer cannot
            # represent the term at all, which makes it a candidate row, not a diagnostic.
            if len(pieces) <= 1 and not (unk is not None and pieces == [unk]):
                single_token.add(term)
                continue
            st = stats.setdefault(term, TermStat(term))
            # `source_doc=None` means NO DOCUMENT VOTE (`data.documentless_sources_vote ==
            # "none"`, ruling A4): nqopen and triviaqa ship no document, so they still supply
            # contexts and residuals but cannot push a term toward the 20-document minimum.
            if doc is not None and doc not in st.docs:   # one vote per distinct document
                st.docs.add(doc)
                st.domains[domain] += 1
            st.contexts.add(qid)
            st.residual_sum += float(residuals[i])
            st.residual_n += 1
    return stats, single_token


def _unk_id(tokenizer):
    """The base tokenizer's `[UNK]` id, or None if it has no unknown token."""
    try:
        return tokenizer.token_to_id("[UNK]")
    except AttributeError:                            # pragma: no cover - exotic tokenizers
        return None


_MISSING = object()


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def tokenize_term(tokenizer, term):
    """The term's own pieces, in isolation, WITHOUT [CLS]/[SEP].

    `add_special_tokens=False` is the whole point: the initialization sums the constituent
    rows, and including the sentence markers would pull two unrelated rows into every new row.
    """
    enc = tokenizer.encode(term, add_special_tokens=False)
    return list(enc.ids)


def score(stat: TermStat) -> float:
    """`vocabulary_ranking.score`: residual x log support weight."""
    return stat.residual * math.log(1.0 + stat.n_docs)


def domain_of(stat: TermStat) -> str:
    """Majority domain of the supporting source documents; no majority -> 'general'."""
    if not stat.domains:
        return "general"
    top = max(stat.domains.values())
    winners = sorted(d for d, c in stat.domains.items() if c == top)
    total = sum(stat.domains.values())
    if len(winners) > 1 or top * 2 <= total:
        return "general"
    return winners[0]


def rank(stats):
    """Deterministic full ranking: score desc, distinct documents desc, lexical asc."""
    return sorted(stats.values(), key=lambda s: (-score(s), -s.n_docs, s.term))


UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)
LONG_HEX_RE = re.compile(r"^(?:0x)?[0-9a-f]{16,}$", re.I)
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[tT].*)?$")
RANDOM_SUFFIX_RE = re.compile(r".*-(?=[a-z0-9]{8,16}$)(?=[a-z0-9]*[a-z])(?=[a-z0-9]*\d)[a-z0-9]+$", re.I)


def is_near_unique(term: str) -> bool:
    """Strings that identify an event instance rather than a reusable project concept."""
    t = term.strip()
    return bool(UUID_RE.fullmatch(t) or LONG_HEX_RE.fullmatch(t)
                or TIMESTAMP_RE.fullmatch(t) or t.count("/") >= 2
                or RANDOM_SUFFIX_RE.fullmatch(t))


def select(stats, reg, expansions=None, abbreviations=None, single_token=(), **_ignored):
    """Apply M18's project-only support minima and one explicit resident-row ceiling.

    ``expansions`` and ``abbreviations`` remain accepted so the copied caller surface is stable,
    but do not change eligibility: aliases require the same evidence as every other exact row.
    """
    policy = reg["vocabulary"]
    minima_docs = int(policy["min_distinct_documents"])
    minima_ctx = int(policy["min_training_contexts"])
    cap_total = int(policy["max_added_rows"])
    chosen = []
    dropped = {"below_minima": [], "near_unique": [], "total_cap": [],
               "already_single_token": sorted(set(single_token))}
    for st in rank(stats):
        if is_near_unique(st.term):
            dropped["near_unique"].append(st.term)
            continue
        if st.n_docs < minima_docs or st.n_contexts < minima_ctx:
            dropped["below_minima"].append(st.term)
            continue
        if len(chosen) >= cap_total:
            dropped["total_cap"].append(st.term)
            continue
        chosen.append({"term": st.term, "reason": "ranked_supported",
                       "residual": st.residual, "score": score(st),
                       "distinct_source_documents": st.n_docs,
                       "distinct_training_contexts": st.n_contexts,
                       "abbreviation": bool(is_abbreviation(st.term))})
    supports = [x["distinct_source_documents"] for x in chosen]
    return {"terms": chosen,
            "dropped": dropped,
            "minima": {"distinct_source_documents": minima_docs,
                       "distinct_training_contexts": minima_ctx},
            "caps": {"total": cap_total},
            "support_distribution": ({"min": min(supports), "median": float(np.median(supports)),
                                      "max": max(supports)} if supports else {}),
            "vocabulary_sha256": sha_text("\n".join(c["term"] for c in chosen))}


# ---- rows -------------------------------------------------------------------------------

def effective_rows(rows, token_weights=None):
    """Fold the learned per-token scalars into the rows ONCE. `None` means already effective."""
    r = np.asarray(rows, dtype=np.float32)
    if token_weights is None:
        return r
    w = np.asarray(token_weights, dtype=np.float32)
    if w.shape[0] != r.shape[0]:
        raise ValueError(f"token weights {w.shape} do not match rows {r.shape}")
    return w[:, None] * r


def verify_old_vocab_parity(eff_rows, released_rows, tol=5e-3):
    """Effective rows of the UNFOLDED warm start must reproduce the v1 release's rows.

    This is the "never fold twice" check: if the checkpoint's scalars had already been folded,
    folding again scales every row by w twice and this comparison fails loudly instead of
    silently shipping a differently scaled table.
    """
    a = np.asarray(eff_rows, dtype=np.float32)
    b = np.asarray(released_rows, dtype=np.float32)
    if a.shape != b.shape:
        raise AssertionError(f"old-vocabulary parity: shapes differ {a.shape} vs {b.shape}; "
                             "new rows must be appended AFTER this check, not before")
    dev = float(np.abs(a - b).max())
    if dev > tol:
        raise AssertionError(
            f"old-vocabulary parity FAILED: max abs {dev:.3e} > {tol:.1e}. Either the warm start "
            "is not the checkpoint the release was folded from, or its learned scalars were "
            "folded twice. Do not construct new rows from these rows.")
    return dev


def init_new_rows(terms, tokenizer, eff_rows):
    """`new_row = sum_j sqrt(c_j) * R_eff_j` over the term's isolated constituent pieces.

    Plain summation over occurrences would initialize `C++` as `R_c + 2 R_+` where the original
    sqrt-count pool computed `R_c + sqrt(2) R_+` (Astra P2), so the counts enter under a square
    root exactly as they do at query time.
    """
    eff = np.asarray(eff_rows, dtype=np.float32)
    out = np.zeros((len(terms), eff.shape[1]), dtype=np.float32)
    pieces_used = []
    for i, term in enumerate(terms):
        ids = tokenize_term(tokenizer, term)
        uniq, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
        w = np.sqrt(counts.astype(np.float32))
        out[i] = (eff[uniq] * w[:, None]).sum(0)
        pieces_used.append({"term": term, "piece_ids": [int(x) for x in uniq],
                            "counts": [int(c) for c in counts]})
    return out, pieces_used


def extend_tokenizer(tokenizer, terms):
    """Add `terms` as `single_word=True` added tokens and return (tokenizer, hash, n_added).

    `single_word=True` means the row can only match at word boundaries, so `xk8sy` keeps its
    ordinary WordPiece path. Padding stays disabled; truncation is set by the loader.
    """
    from tokenizers import AddedToken
    before = tokenizer.get_vocab_size(with_added_tokens=True)
    added = tokenizer.add_tokens([AddedToken(t, single_word=True, normalized=True) for t in terms])
    after = tokenizer.get_vocab_size(with_added_tokens=True)
    if after - before != added:
        raise AssertionError(f"tokenizer grew by {after - before} for {added} added tokens")
    return tokenizer, tokenizer_hash(tokenizer), added


def tokenizer_hash(tokenizer) -> str:
    """Hash the SERIALIZED tokenizer: the bytes that ship, not the object's repr."""
    return sha_text(tokenizer.to_str())


def new_row_ids(tokenizer, terms):
    """Ids the extended tokenizer assigned, in `terms` order. New ids belong to the student."""
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    return [int(vocab[t]) for t in terms]


def drift_report(tokenizer_old, tokenizer_new, eff_rows, new_rows, new_ids, probes,
                 pool_sqrt=True):
    """P0b: how far sum-initialization moves a query's vector, per probe query.

    Zero for a query whose remaining tokens share no piece with the new term; nonzero exactly
    when a constituent piece also occurs elsewhere, because its sqrt-count weight changes.
    Recorded before training so V0's read has a predicted mechanism, not a surprise.
    """
    ext = np.concatenate([np.asarray(eff_rows, dtype=np.float32),
                          np.zeros((max(new_ids) + 1 - len(eff_rows), eff_rows.shape[1]),
                                   dtype=np.float32)], axis=0) if new_ids else eff_rows
    for row, i in zip(new_rows, new_ids):
        ext[i] = row
    out = []
    for text in probes:
        a = _pool(np.asarray(tokenizer_old.encode(text).ids), eff_rows, pool_sqrt)
        b = _pool(np.asarray(tokenizer_new.encode(text).ids), ext, pool_sqrt)
        out.append({"query": text, "cosine": float(a @ b),
                    "max_abs": float(np.abs(a - b).max())})
    return out


def _pool(ids, rows, pool_sqrt=True):
    if ids.size == 0:
        return np.zeros(rows.shape[1], dtype=np.float32)
    uniq, counts = np.unique(ids, return_counts=True)
    w = np.sqrt(counts.astype(np.float32)) if pool_sqrt else counts.astype(np.float32)
    v = (rows[uniq] * w[:, None]).sum(0) / max(float(w.sum()), 1e-6)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else v
