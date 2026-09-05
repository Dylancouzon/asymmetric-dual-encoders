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
