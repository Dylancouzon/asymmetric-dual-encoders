"""The four registered extraction rules, each tested against the way it could over- or under-fire."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harvest as H


def test_claim_requires_all_four_registered_conditions():
    ok = "Bethlem myopathy is a slowly progressive muscle disease characterized by contractures."
    assert H.is_claim(ok)
    assert not H.is_claim("Too short entirely."), "under the 8-word floor"
    assert not H.is_claim(" ".join(["word"] * 41) + "."), "over the 40-word ceiling"
    assert not H.is_claim("We show that our method is better than the previous state of the art."), \
        "first person"
    assert not H.is_claim("What is the treatment for Bethlem myopathy in older adults today?"), \
        "a question is not a declarative claim"
    assert not H.is_claim("A very long noun phrase of exactly nine plain nouns here"), \
        "a bare noun phrase is not a declarative claim, and its plural must not read as a verb"
    assert not H.is_claim("Several distinct approaches to the problem of automated theorem proving"), \
        "no finite verb"
    assert H.is_claim("The protocol consists of three phases that are executed in a fixed order.")


def test_claims_come_from_the_LEAD_not_from_anywhere():
    text = ("Short. " + "The compound is a white crystalline solid used widely in industry. "
            + "A later sentence that is also a perfectly good declarative claim about things.")
    got = H.claims_from(text, max_per_doc=1)
    assert len(got) == 1 and got[0].startswith("The compound is a white")


def test_asks_keep_the_preceding_sentence_and_only_questions():
    text = "Background about a thing. What causes it in adults? It is treated with rest."
    got = H.asks_from(text)
    assert len(got) == 1
    q, body = got[0]
    assert q == "What causes it in adults?" and body == "Background about a thing."
    assert H.asks_from("No questions at all here. None whatsoever.") == []


def test_the_first_question_in_a_document_has_no_body():
    q, body = H.asks_from("Why does this happen so often in practice? Because of X.")[0]
    assert body is None


def test_titles_route_by_the_frozen_rubric_range():
    assert H.route_by_length("Gene therapy") == "keyword"          # 2-4 words
    assert H.route_by_length("A general theory of relativity and its modern applications") == "title"
    assert H.route_by_length("Rome") is None, "1 word is under keyword's floor"
    assert H.route_by_length(" ".join(["w"] * 40)) is None, "over title's ceiling"


def test_headings_drop_the_apparatus_sections():
    lines = ["Lead text here.", "", "History", "", "Body.", "", "References", "", "[1]"]
    got = H.headings_from("\n".join(lines))
    assert "History" in got and "References" not in got


def test_sentence_split_does_not_break_on_abbreviations_mid_number():
    ss = H.sentences("Version 3.5 was released in 2020. It replaced the earlier build.")
    assert len(ss) == 2, ss


if __name__ == "__main__":
    for k, v in sorted(globals().items()):
        if k.startswith("test_"):
            v(); print("PASS", k)


# ---- draw(): the pass-1 rules, with the screens stubbed out -----------------------------------
#
# `draw()` had no test at all, which is how the missing `factoid`/`product` quota row survived.
# These exercise the four rules pass 1 is responsible for -- no-quota accounting, the frozen-rubric
# range, dedup, and uniformity of the truncation -- with the protected/document screens stubbed,
# since those are covered by their own modules' tests and need multi-GB corpora.

import json
import types


def _fixture(tmp_path, rows):
    p = tmp_path / "h.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "h.jsonl.report.json").write_text(json.dumps({"complete": True}))
    return p


def _row(form, text, src="wiki", rule="title"):
    return {"rule": rule, "form": form, "text": text, "body": "", "src": src, "doc": "d1"}


def _stub_screens(monkeypatch):
    """No-op query/document screens, so what is measured is pass 1 and the truncation."""
    monkeypatch.setitem(sys.modules, "protected10",
                        types.SimpleNamespace(build=lambda verbose=True: None,
                                              hits=lambda t, idx: False))
    monkeypatch.setitem(sys.modules, "decontam", types.SimpleNamespace(
        Inverted=lambda grams, ex: types.SimpleNamespace(
            match=lambda d, s: (__import__("numpy").array([], dtype=int),
                                __import__("numpy").array([], dtype=int))),
        query_grams=lambda t: (), exact_u64=lambda t: 0, stream_six_docs=lambda: iter(())))
    monkeypatch.setitem(sys.modules, "cov_screen",
                        types.SimpleNamespace(MIN_SHARE=0.25,
                                             load_component=lambda n, r, v: ((), ())))
    monkeypatch.setitem(sys.modules, "devsuite", types.SimpleNamespace(COMPONENTS=()))
    monkeypatch.setitem(sys.modules, "cov_admit", types.SimpleNamespace(COMPONENTS={}))


def test_draw_counts_but_never_draws_a_form_with_no_quota_row(tmp_path, monkeypatch):
    _stub_screens(monkeypatch)
    monkeypatch.setattr(H, "OUT", tmp_path)
    rows = [_row("title", f"a perfectly ordinary harvested article title number {i}")
            for i in range(10)]
    rows += [_row("factoid", "who wants to be a millionaire?", src="hotpotqa-corpus", rule="ask")]
    rows += [_row("product", "what are you waiting for?", src="esci-prod", rule="ask")] * 3
    out, rep = H.draw(quota={"title": 5}, margin=1.0,
                      paths=[_fixture(tmp_path, rows)], verbose=False)
    assert set(out) == {"title"}, "a form with no quota row is never drawn"
    assert rep["skipped_no_quota"] == {"factoid": 1, "product": 3}, \
        "and its rows are COUNTED, so the exclusion is visible in the artifact"


def test_draw_enforces_the_frozen_rubric_range_not_the_extraction_window(tmp_path, monkeypatch):
    _stub_screens(monkeypatch)
    monkeypatch.setattr(H, "OUT", tmp_path)
    import qfilter
    lo, hi = qfilter.RANGES["claim"]
    assert (lo, hi) == (8, 25), "the frozen rubric's claim range"
    in_range = "The compound is a white crystalline solid used widely in modern industry today."
    assert lo <= len(in_range.split()) <= hi
    over = "The compound " + " ".join(["extremely"] * 30) + " is a solid."   # inside 8-40, over 25
    assert hi < len(over.split()) <= H.CLAIM_MAX, "the case the extraction window admits"
    rows = [_row("claim", in_range, rule="claim")] + [_row("claim", over, rule="claim")]
    out, rep = H.draw(quota={"claim": 1}, margin=1.0,
                      paths=[_fixture(tmp_path, rows)], verbose=False)
    assert [r["text"] for r in out["claim"]] == [in_range]
    assert rep["off_rubric_range"]["claim"] == 1
    assert rep["rubric_ranges"]["claim"] == [8, 25]


def test_draw_dedups_on_normalized_text_and_reports_it(tmp_path, monkeypatch):
    _stub_screens(monkeypatch)
    monkeypatch.setattr(H, "OUT", tmp_path)
    rows = [_row("title", "The Battle of Hastings and its aftermath"),
            _row("title", "the   BATTLE of hastings and its aftermath"),   # same, normalized
            _row("title", "A different harvested title about something else")]
    out, rep = H.draw(quota={"title": 2}, margin=1.0,
                      paths=[_fixture(tmp_path, rows)], verbose=False)
    assert len(out["title"]) == 2 and rep["exact_dups"]["title"] == 1


def test_draw_refuses_an_incomplete_or_missing_pass(tmp_path, monkeypatch):
    import pytest
    _stub_screens(monkeypatch)
    monkeypatch.setattr(H, "OUT", tmp_path)
    with pytest.raises(SystemExit, match="missing"):
        H.draw(quota={"title": 1}, paths=[tmp_path / "nope.jsonl"], verbose=False)
    p = _fixture(tmp_path, [_row("title", "a title that is long enough to be in range")])
    (tmp_path / "h.jsonl.report.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(SystemExit, match="complete"):
        H.draw(quota={"title": 1}, paths=[p], verbose=False)


def test_draw_truncation_is_uniform_over_the_stream_not_biased_to_its_prefix(tmp_path,
                                                                            monkeypatch):
    """The regression that motivated the shuffle: first-n-of-reservoir keeps the stream prefix."""
    _stub_screens(monkeypatch)
    monkeypatch.setattr(H, "OUT", tmp_path)
    N, n, margin = 4000, 1000, 1.5
    want = int(margin * n)                                            # 1500
    rows = [_row("title", f"harvested article title number {i} of the stream") for i in range(N)]
    path = _fixture(tmp_path, rows)
    out, _ = H.draw(quota={"title": n}, margin=margin, paths=[path], verbose=False)
    pos = [int(r["text"].split()[4]) for r in out["title"]]
    assert len(pos) == n
    # The EXACT signature of the bug: a stream item at position in [n, want) can only ever occupy
    # reservoir slot `pos` (Algorithm R overwrites slot j with LATER items only), and those slots
    # are precisely the ones first-n truncation discards -- so the biased draw takes ZERO of them,
    # against n*(want-n)/N = 125 expected under a uniform draw.
    band = sum(1 for p in pos if n <= p < want)
    assert band > 60, f"positions [{n},{want}) drawn {band} times; 0 means truncation is biased"
    # and the prefix is not inflated by the margin (uniform expects n*n/N = 250, biased 1.5x that)
    prefix = sum(1 for p in pos if p < n)
    assert prefix < 1.3 * n * n / N, f"prefix over-represented: {prefix} vs 250 expected"
