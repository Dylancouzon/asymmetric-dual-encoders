"""Vocabulary discovery, ranking, selection and count-weighted initialization.

The two checks that matter most are the last two: sum-initialization must reproduce the old
pool on an ISOLATED term (so the warm start is not a random jump) and must NOT be assumed to
reproduce it when a constituent piece is shared with another token in the query — the P0b
hazard the registry sends V0 to measure.
"""
from __future__ import annotations

import numpy as np
import pytest

import vocab as V


def _stats(**kw):
    s = V.TermStat(kw.pop("term"))
    s.docs.update(kw.pop("docs", []))
    s.contexts.update(kw.pop("contexts", []))
    for d, c in kw.pop("domains", {}).items():
        s.domains[d] = c
    s.residual_sum, s.residual_n = kw.pop("residual", 0.0), 1
    return s


def test_discovery_skips_already_single_tokens(tok_and_vocab):
    tok, _ = tok_and_vocab
    qs = [{"text": "storage bucket k8s", "qid": "q1", "source_doc": "d1", "domain": "general"}]
    stats, single = V.discover(qs, [0.5], tok)
    assert "k8s" in stats and "storage" in single and "storage" not in stats


def test_discovery_deduplicates_repeated_query_text(tok_and_vocab):
    tok, _ = tok_and_vocab
    qs = [{"text": f"k8s ingress {w}", "qid": f"q{i}", "source_doc": f"d{i}",
           "domain": "general"}
          for i, w in enumerate(["a", "the", "for"])]
    qs.append({"text": qs[0]["text"], "qid": "q3", "source_doc": "d3", "domain": "general"})
    stats, _ = V.discover(qs, [0.5] * 4, tok)
    assert stats["k8s"].n_docs == 3, "the repeated query text must not add support"


def test_ranking_order_is_score_then_support_then_lexical():
    a = _stats(term="bbb", docs=range(10), contexts=range(50), residual=0.5)
    b = _stats(term="aaa", docs=range(10), contexts=range(50), residual=0.5)
    c = _stats(term="ccc", docs=range(100), contexts=range(50), residual=0.5)
    order = [s.term for s in V.rank({s.term: s for s in (a, b, c)})]
    assert order == ["ccc", "aaa", "bbb"]


def test_minima_caps_and_pinned_term(reg):
    stats = {}
    for i in range(40):
        stats[f"term{i:03d}"] = _stats(term=f"term{i:03d}", docs=range(9), contexts=range(9),
                                       domains={"cloud-software": 5}, residual=0.5)
    stats["thin"] = _stats(term="thin", docs=range(1), contexts=range(1), residual=0.9)
    small = {**reg, "added_rows_max": 5, "per_domain_added_rows_max": 3,
             "technical_priority_slots_max": 3}
    sel = V.select(stats, small, single_token=())
    assert len(sel["terms"]) <= 5
    assert sel["per_domain"].get("cloud-software", 0) <= 3
    assert "thin" in sel["dropped"]["below_minima"]
    assert sel["terms"][0]["term"] == "k8s" and sel["terms"][0]["reason"] == "owner_pinned"
    assert sel["breadth"] == "narrow"


def test_abbreviation_policy_expands_or_drops(reg):
    stats = {
        "k9s": _stats(term="k9s", docs=range(3), contexts=range(5), residual=0.9),
        "kubernetes": _stats(term="kubernetes", docs=range(30), contexts=range(80),
                             residual=0.5),
        "t2x": _stats(term="t2x", docs=range(3), contexts=range(5), residual=0.9),
    }
    small = {**reg, "data": {**reg["data"], "new_term_min_distinct_source_documents": 2,
                             "new_term_min_distinct_training_contexts": 3}}
    sel = V.select(stats, small, expansions={"k9s": "kubernetes"}, single_token=())
    assert sel["dropped"]["abbreviation_expanded"] == [{"abbrev": "k9s",
                                                        "expanded": "kubernetes"}]
    assert "t2x" in sel["dropped"]["abbreviation_dropped"]
    assert "kubernetes" in [t["term"] for t in sel["terms"]]


def test_domain_of_needs_a_majority():
    s = _stats(term="x", domains={"a": 3, "b": 3})
    assert V.domain_of(s) == "general"
    assert V.domain_of(_stats(term="x", domains={"a": 5, "b": 1})) == "a"


def test_fold_twice_is_refused():
    rows = np.ones((5, 4), dtype=np.float32)
    w = np.full(5, 2.0, dtype=np.float32)
    eff = V.effective_rows(rows, w)
    V.verify_old_vocab_parity(eff, eff)
    with pytest.raises(AssertionError, match="parity FAILED"):
        V.verify_old_vocab_parity(V.effective_rows(eff, w), eff)


def test_count_weighted_init_reproduces_the_isolated_pool(tok_and_vocab):
    """`c++` must initialize to R_c + sqrt(2) R_+, not R_c + 2 R_+ (Astra P2)."""
    tok, n = tok_and_vocab
    rng = np.random.default_rng(0)
    eff = rng.normal(size=(n, 8)).astype(np.float32)
    rows, pieces = V.init_new_rows(["c++"], tok, eff)
    ids = V.tokenize_term(tok, "c++")
    uniq, counts = np.unique(np.asarray(ids), return_counts=True)
    want = (eff[uniq] * np.sqrt(counts)[:, None]).sum(0)
    assert np.allclose(rows[0], want, atol=1e-5)
    assert counts.max() == 2, "the fixture must actually contain a repeated piece"
    plain = eff[np.asarray(ids)].sum(0)
    assert not np.allclose(rows[0], plain, atol=1e-4)


def test_p0b_shared_piece_drift_is_visible(tok_and_vocab):
    """Zero drift when nothing is shared, nonzero when a constituent piece recurs."""
    tok, n = tok_and_vocab
    from tokenizers import Tokenizer
    rng = np.random.default_rng(1)
    eff = rng.normal(size=(n, 8)).astype(np.float32)
    ext = Tokenizer.from_str(tok.to_str())
    rows, _ = V.init_new_rows(["s3"], tok, eff)
    ext, _, _ = V.extend_tokenizer(ext, ["s3"])
    ids = V.new_row_ids(ext, ["s3"])
    clean, shared = V.drift_report(tok, ext, eff, rows, ids,
                                   ["s3 bucket policy", "s3 storage s cluster"])
    assert clean["cosine"] > 1 - 1e-5
    assert shared["cosine"] < clean["cosine"]


def test_single_word_added_token_does_not_match_inside_a_word(tok_and_vocab):
    tok, _ = tok_and_vocab
    from tokenizers import Tokenizer
    ext, _, added = V.extend_tokenizer(Tokenizer.from_str(tok.to_str()), ["s3"])
    assert added == 1
    assert "s3" in ext.encode("s3 bucket").tokens
    assert "s3" not in ext.encode("xs3y").tokens


def test_tokenizer_hash_changes_with_the_vocabulary(tok_and_vocab):
    tok, _ = tok_and_vocab
    from tokenizers import Tokenizer
    a = V.tokenizer_hash(tok)
    ext, b, _ = V.extend_tokenizer(Tokenizer.from_str(tok.to_str()), ["s3"])
    assert a != b
