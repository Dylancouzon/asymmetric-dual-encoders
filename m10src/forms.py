"""The 12 synthetic query forms of instructions-m10.md, as generator prompts.

Each form is a fixed instruction the generator (Qwen3-8B, pinned at M10.1) receives with one seed
passage. The output contract is one JSON list of strings so the pipeline can parse, count and
fingerprint without heuristics. Quotas (per form, from the manifest; ≈143K for each generated form) and seeds are set in the M10.1 manifest;
this file only fixes the wording, which the 200-query smoke per form reads before any scaling.
Seven of the twelve are GENERATED under decision 14 (howto, argument, finance, comparison, yesno,
conversational, health); the other five are harvested real text and their prompts here are unused.

Rules baked into every prompt: write from the passage's topic, never copy a span of five or more
consecutive words from it (the decontamination screen drops copies anyway), no preamble, plain
text, English. `ponytail:` one template per form, no prompt-tuning machinery; the smoke decides
whether a form's wording needs a second draft.
"""
import json

SYSTEM = ("You write search queries for a retrieval benchmark. Follow the form exactly. Use the "
          "passage only for its topic and facts; never copy five or more consecutive words from it. "
          "Answer with one JSON list of strings and nothing else.")

FORMS = {
    "factoid": ("{n} short factual questions a person might type into a search engine about the "
                "passage's topic, each answerable by a document like it. 5 to 15 words."),
    "howto": ("{n} troubleshooting or how-to questions in the style of a technical forum post: a "
              "one-line title, then one or two sentences of body describing the situation and "
              "what was tried. 25 to 60 words each, title and body separated by a newline."),
    "claim": ("{n} scientific or factual claims stated as declarative sentences (not questions), "
              "of the kind a fact-checker would verify against a document like the passage. Some "
              "true, some plausible but false. 8 to 25 words."),
    "argument": ("{n} argumentative paragraphs of 120 to 220 words, each taking one side of a "
                 "debate the passage's topic could be part of, written as a forum debater would, "
                 "so that a document with the opposing view would be the best match."),
    "finance": ("{n} personal-finance or economics questions a member of the public might ask "
                "about the passage's topic (money, prices, taxes, investing, markets, policy). "
                "8 to 30 words."),
    "title": ("{n} phrases that read like the title of a research paper or report on the passage's "
              "topic: noun phrases, no question mark, 6 to 16 words."),
    "keyword": ("{n} keyword queries of 2 to 4 words, the way someone types into a search box, "
                "covering different aspects of the passage."),
    "health": ("{n} consumer-health questions a patient or caregiver might ask about the passage's "
               "topic (symptoms, treatments, risks, what a term means), in plain language. "
               "8 to 30 words."),
    "product": ("{n} product-search queries a shopper would type to find an item related to the "
                "passage's topic: brand-agnostic, with attributes such as size, material, use or "
                "price range. 3 to 12 words."),
    "comparison": ("{n} questions comparing two or more things related to the passage's topic "
                   "(X vs Y, which is better for Z, difference between). 8 to 25 words."),
    "yesno": ("{n} yes/no verification questions about the passage's topic, phrased so a "
              "document like it could answer them. 6 to 20 words."),
    "conversational": ("{n} multi-sentence requests written the way a person talks to an "
                       "assistant: some context about their situation, then what they want to "
                       "find. 30 to 80 words."),
}
assert len(FORMS) == 12


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
