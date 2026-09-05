"""`wikipedia-body` — the seed store that closes the health/finance supply shortfall (W3).

**Why this store and not another keyword lever.** `ROUTE_WIDE` failed its judged-precision gate
(28% / 38% on-topic vs a >= 80% bar) because the router selects on the PRESENCE of medical or
financial words, not on subject. Widening further is the same error at larger scale. The supply
shortfall is instead closed by corpus size at an unchanged router: `hotpotqa-corpus` is Wikipedia
INTRO paragraphs only, so 5.23M articles yield one candidate chunk each; the same articles' BODY
text yields ~13. A 20,000-article pilot at the registered `ROUTE` and `min_score >= 4` gave
health 1,840 / finance 3,532 seeds, i.e. ~92,000 health per 1M articles against a ~33K need.

Wikipedia is a **registered** seed source, not a new admission: `instructions-m10.md` §Data
("Seeds for the generated half: Wikipedia stratified by top-level category (CC BY-SA) and the
approved pool corpora") and `research/m7-data-licensing.md:42,61` (CC BY-SA family position,
confirmed by Dylan 2026-08-25, model-card attribution required). This is why the FineWeb closure
(`m10/EXPLORED.md`) does not reach it: FineWeb was an unregistered source whose overlap with
reserved text is unbounded, and this store's overlap is bounded by construction (below).

**The lead section is excluded.** Two reserved corpora are Wikipedia lead text — DBpedia-entity
(abstracts) and FEVER (introductory sections) — and reserved-set DOCUMENT fingerprints do not
exist and cannot be built without opening the reserved corpora (W4). Body-only text is disjoint
from lead text by construction, so this store is strictly cleaner on that axis than the intro
store already in use. It is also near-disjoint from `hotpotqa-corpus` for the same reason.

**Divergence disclosed, not fixed:** the mandate says "stratified by top-level category"; this
routes by keyword, as `seeds.ROUTE` already does for the existing stores (T2-3). Category-
membership routing over the `categorylinks` dump is the registered next lever if the gate fails —
it is the subject-level filter the judge asked for, and it is not a relaxed floor.
"""
import json, os, re, sys
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m10src"):
    sys.path.insert(0, str(REPO / p))
OUT = REPO / "work" / "m10gen"

REPO_ID = "wikimedia/wikipedia"
CONFIG = "20231101.en"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
# CC BY-SA 4.0 (the 2023-11 dump post-dates Wikimedia's June 2023 move to 4.0; the HF card still
# tags 3.0 — both recorded). Dual-licensed GFDL; CC BY-SA is the licence chosen for this use.
LICENCE = "CC BY-SA 4.0 (HF card tags 3.0; dual GFDL, CC BY-SA chosen)"

# Registered BEFORE the scan, and not tuned to any count it produces.
FORMS = ("health", "finance")        # `howto` clears on the existing stores and is not touched
PER_ARTICLE_CAP = 3                  # per form: one article must not become 3 x 5 near-identical
                                     # questions x 30 chunks (the A8 near-duplicate trap)
MIN_SCORE = 4                        # unchanged
_TERMINAL = ('.', '!', '?', '"', "'", ')', ']', '”', '’')


def is_heading(line):
    """A section heading in this dump's `text` field: a short, unterminated, capitalised line.

    The dump renders sections as a bare title line between blank lines — there is no `==` marker
    to key on — so the rule is structural. It is used ONLY to find where the lead ends, so its
    failure mode is asymmetric by design: a missed heading keeps lead text (refused as a seed,
    costing supply), a false heading admits body text one paragraph early (the thing to avoid),
    which is why the test is conservative on every axis.
    """
    s = line.strip()
    if not s or len(s.split()) > 12:
        return False
    if s.endswith(_TERMINAL) or ". " in s or "; " in s:
        return False
    return s[0].isupper() or s[0].isdigit()


def body(text):
    """-> the article text AFTER the first heading line, or '' if the article has no heading."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if is_heading(ln):
            return "\n".join(lines[i + 1:])
    return ""


def chunks(text, min_words, max_words):
    """Paragraph chunks in the registered length window; adjacent short paragraphs merged."""
    out, buf = [], []
    for para in text.split("\n"):
        w = para.split()
        if not w:
            continue
        buf += w
        if len(buf) >= min_words:
            out.append(" ".join(buf[:max_words]))
            buf = []
    if len(buf) >= min_words:
        out.append(" ".join(buf[:max_words]))
    return out


def stream(limit=None):
    from datasets import load_dataset
    ds = load_dataset(REPO_ID, CONFIG, split="train", streaming=True, revision=REVISION)
    for i, r in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield r


def verify_lead_rule(n=60, out=None):
    """Condition 2 of the Fable review: the lead rule is hand-verifiable before the scan.

    Dumps, for `n` articles, the last lead line and the detected heading, so the boundary can be
    eyeballed; and reports the share of articles with no heading at all (they contribute nothing).
    """
    rows, no_head = [], 0
    for r in stream(n):
        lines = r["text"].split("\n")
        idx = next((i for i, ln in enumerate(lines) if is_heading(ln)), None)
        if idx is None:
            no_head += 1
            rows.append(dict(title=r["title"], heading=None, last_lead_line=None))
            continue
        prev = next((lines[j].strip() for j in range(idx - 1, -1, -1) if lines[j].strip()), "")
        rows.append(dict(title=r["title"], heading=lines[idx].strip(),
                         last_lead_line=prev[-140:],
                         body_words=len(body(r["text"]).split())))
    rep = dict(n=n, no_heading=no_head, no_heading_frac=round(no_head / n, 4), rows=rows)
    if out:
        Path(out).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    r = verify_lead_rule(int(sys.argv[1]) if len(sys.argv) > 1 else 60,
                         out=OUT / "wikibody_lead_check.json")
    print(f"no heading: {r['no_heading']}/{r['n']} ({r['no_heading_frac']:.1%})")
    for row in r["rows"][:40]:
        print(f"\n{row['title']}\n  ...lead ends: {row['last_lead_line']}\n"
              f"  heading -> {row['heading']!r}  body_words={row.get('body_words')}")
