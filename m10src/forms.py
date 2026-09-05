"""The 12 synthetic query forms of instructions-m10.md, as generator prompts.

Each form is a fixed instruction the generator (Qwen3-8B, pinned at M10.1) receives with one seed
passage. The output contract is one JSON list of strings so the pipeline can parse, count and
fingerprint without heuristics. Quotas (per form, from the manifest; ≈143K for each generated form) and seeds are set in the M10.1 manifest;
this file only fixes the wording, which the 200-query smoke per form reads before any scaling.
Seven of the twelve are GENERATED under decision 14 (howto, argument, finance, comparison, yesno,
conversational, health); the other five are harvested real text and their prompts here are unused.

Rules baked into every prompt: write from the passage's topic, never copy a span of five or more
consecutive words from it (the decontamination screen drops copies anyway), no preamble, plain
text, English. One template per form, no prompt-tuning machinery; the smoke decides
whether a form's wording needs a second draft.
"""
import json

SYSTEM = ("You write search queries for a retrieval benchmark. Follow the form exactly. Use the "
          "passage only for its topic and facts; never copy five or more consecutive words from it. "
          "Answer with one JSON list of strings and nothing else.")

FORMS = {
    "factoid": ("{n} short factual questions a person might type into a search engine about the "
                "passage's topic, each answerable by a document like it. 5 to 15 words."),
    # Revision 1 (2026-09-04, smoke): the original wording made the generator emit the title and
    # the body as two SEPARATE list items, so the list held 2n strings and the strict parser
    # rejected 55% of replies. Only the output shape is restated; the form itself is unchanged.
    "howto": ("{n} troubleshooting or how-to questions in the style of a technical forum post. "
              "Each list item is ONE string built as: a one-line title, then a newline "
              "character, then a body of EXACTLY ONE OR TWO SENTENCES describing the situation "
              "and what was tried. Never three sentences. Never a title alone. Never join the "
              "title to the body with a colon or a dash — the separator is the newline. The "
              "whole item is 25 to 60 words. Never split the title and the body into separate "
              "list items: the list has exactly {n} strings."),
    "claim": ("{n} scientific or factual claims stated as declarative sentences (not questions), "
              "of the kind a fact-checker would verify against a document like the passage. Some "
              "true, some plausible but false. 8 to 25 words."),
    # Revision 1 (2026-09-04, smoke): 38 of 50 came in under the 120-word floor. The register
    # and the one-sidedness were already right; only the length floor is restated, as a hard
    # requirement with a self-check the model can act on.
    "argument": ("{n} argumentative paragraphs, each taking ONE side of a debate the passage's "
                 "topic could be part of, written as a forum debater would, so that a document "
                 "arguing the opposing view would be the best match. **Each paragraph must be "
                 "160 to 210 words.** Reaching that length is the hard part, so build every "
                 "paragraph from EIGHT OR MORE full sentences, in this order: state the claim; "
                 "give three separate reasons, each its own sentence and each with a concrete "
                 "specific (a number, a case, a mechanism, a consequence); state the strongest "
                 "objection from the other side; rebut it; close with what follows if the claim "
                 "holds. A paragraph of four or five sentences is too short and is wrong."),
    "finance": ("{n} personal-finance or economics questions a member of the public might ask "
                "about the passage's topic (money, prices, taxes, investing, markets, policy). "
                "8 to 30 words."),
    "title": ("{n} phrases that read like the title of a research paper or report on the passage's "
              "topic: noun phrases, no question mark, 6 to 16 words."),
    "keyword": ("{n} keyword queries of 2 to 4 words, the way someone types into a search box, "
                "covering different aspects of the passage."),
    # Revision 1 (2026-09-04, smoke): 17 of 50 came in under the 8-word floor (bare "What is X
    # used for?" forms) and 2 were not patient-facing. Length floor and the patient framing are
    # restated; the topic set is unchanged.
    "health": ("{n} consumer-health questions a patient or caregiver would ask about the "
               "passage's topic — symptoms, treatments, risks, side effects, what a term means, "
               "what to do next — in plain non-clinical language, each phrased from the "
               "patient's or carer's point of view. **Each question must be at least 8 words "
               "and at most 30 words** — a hard requirement: \"What is X used for?\" is too "
               "short, so add the detail a real patient would give. Never ask about a person's "
               "biography or about laboratory animals; the question must be about human health."),
    "product": ("{n} product-search queries a shopper would type to find an item related to the "
                "passage's topic: brand-agnostic, with attributes such as size, material, use or "
                "price range. 3 to 12 words."),
    "comparison": ("{n} questions comparing two or more things related to the passage's topic "
                   "(X vs Y, which is better for Z, difference between). 8 to 25 words."),
    "yesno": ("{n} yes/no verification questions about the passage's topic, phrased so a "
              "document like it could answer them. 6 to 20 words."),
    # Revision 1 (2026-09-04, smoke): 39 of 50 came in under the 30-word floor; the register was
    # already right. Only the length floor is restated, with the structure that reaches it.
    "conversational": ("{n} multi-sentence requests written the way a person talks to an "
                       "assistant. Build each one as FOUR sentences in this exact order. "
                       "Sentence 1: who the person is or what they are doing, in the first "
                       "person. Sentence 2: a further detail of their situation — what they "
                       "have already tried, a constraint they are under, or why it matters to "
                       "them. Sentence 3: one more such detail, different from sentence 2. "
                       "Sentence 4, and ONLY here: what they want to find. Do not state the "
                       "want before sentence 4, and do not restate it afterwards. Each request "
                       "is 30 to 80 words."),
}
assert len(FORMS) == 12

# The REGISTERED rubric — frozen at commit 7fff677, before the 2026-09-04 smoke.
#
# `FORMS` above is the generator PROMPT and is revisable (<=2 revisions per form, LEDGER §1).
# `RUBRIC` is what the independent on-form judge scores against, and it never moves. Keeping
# one dict for both let a prompt revision silently move the bar the gate measures: after the
# `argument` r2 prompt was retuned to ask for 160-210 words (inside the registered 120-220, to
# pull a short generator up), scoring against the PROMPT would have read 8% on-form where the
# REGISTERED description reads 67%. A gate whose standard follows the prompt is not a gate.
RUBRIC = {
    "factoid": "{n} short factual questions a person might type into a search engine about the passage's topic, each answerable by a document like it. 5 to 15 words.",
    "howto": "{n} troubleshooting or how-to questions in the style of a technical forum post: a one-line title, then one or two sentences of body describing the situation and what was tried. 25 to 60 words each, title and body separated by a newline.",
    "claim": "{n} scientific or factual claims stated as declarative sentences (not questions), of the kind a fact-checker would verify against a document like the passage. Some true, some plausible but false. 8 to 25 words.",
    "argument": "{n} argumentative paragraphs of 120 to 220 words, each taking one side of a debate the passage's topic could be part of, written as a forum debater would, so that a document with the opposing view would be the best match.",
    "finance": "{n} personal-finance or economics questions a member of the public might ask about the passage's topic (money, prices, taxes, investing, markets, policy). 8 to 30 words.",
    "title": "{n} phrases that read like the title of a research paper or report on the passage's topic: noun phrases, no question mark, 6 to 16 words.",
    "keyword": "{n} keyword queries of 2 to 4 words, the way someone types into a search box, covering different aspects of the passage.",
    "health": "{n} consumer-health questions a patient or caregiver might ask about the passage's topic (symptoms, treatments, risks, what a term means), in plain language. 8 to 30 words.",
    "product": "{n} product-search queries a shopper would type to find an item related to the passage's topic: brand-agnostic, with attributes such as size, material, use or price range. 3 to 12 words.",
    "comparison": "{n} questions comparing two or more things related to the passage's topic (X vs Y, which is better for Z, difference between). 8 to 25 words.",
    "yesno": "{n} yes/no verification questions about the passage's topic, phrased so a document like it could answer them. 6 to 20 words.",
    "conversational": "{n} multi-sentence requests written the way a person talks to an assistant: some context about their situation, then what they want to find. 30 to 80 words.",
}
assert set(RUBRIC) == set(FORMS)



def prompt(form, passage, n=5):
    """The user turn for one seed passage."""
    return (f"Passage:\n\"\"\"\n{passage.strip()}\n\"\"\"\n\nWrite {FORMS[form].format(n=n)}\n"
            f"Output: a JSON list of exactly {n} strings.")


def parse(text, n):
    """The generator's reply -> list of n strings, or None if it broke the contract.

    Strict: the whole reply must be the JSON list. Preamble is a contract failure (one retry, then
    the seed is dropped), not something to salvage -- salvaging would hide a prompt that drifts.
    """
    try:
        out = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    if not (isinstance(out, list) and len(out) == n and all(isinstance(x, str) and x.strip() for x in out)):
        return None
    return [x.strip() for x in out]


if __name__ == "__main__":
    demo = ("The Hubble Space Telescope was launched into low Earth orbit in 1990 and remains in "
            "operation. Its four main instruments observe in the ultraviolet, visible and near-infrared.")
    for f in FORMS:
        print(f"=== {f}\n{prompt(f, demo, 3)}\n")
    assert parse('["a", "b", "c"]', 3) == ["a", "b", "c"]
    assert parse('Sure! ["a", "b", "c"]', 3) is None      # preamble breaks the contract
    assert parse('["a", "b"]', 3) is None
    print("parse self-check ok")
