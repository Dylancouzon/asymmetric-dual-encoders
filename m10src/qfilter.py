"""Hold generated queries to the word range their own FROZEN rubric already specifies.

Every form's registered description in `m10src/forms.RUBRIC` ends with a word range — health "8 to
30 words", yesno "6 to 20", argument "120 to 220". `forms.parse` validates the output SHAPE (n
items, the list format) but not the length, so out-of-range strings reach the corpus: in the T2-7
diagnostic, **being under the 8-word floor was the single largest on-form failure**, about 10% of
health output, larger than every topical failure combined.

This enforces the range that is already registered. It is **not a new standard** — the ranges are
parsed out of the frozen `RUBRIC` at import, so they cannot drift from the text the judges score
against, and a rubric edit moves the filter with it rather than leaving the two to disagree.

Supply pays for it and there is supply: the store carries 51,633 health seeds against a 33,000
need. Per-form drop counts are reported into the manifest, never silently absorbed.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m10src"))

import forms

_RANGE = re.compile(r"(\d+)\s+to\s+(\d+)\s+words")


def word_range(form):
    """-> (lo, hi) parsed from the FROZEN rubric, so the filter cannot disagree with the judge."""
    m = _RANGE.search(forms.RUBRIC[form])
    if not m:
        raise KeyError(f"{form}: no word range in the frozen rubric")
    return int(m.group(1)), int(m.group(2))


RANGES = {f: word_range(f) for f in forms.RUBRIC}


def n_words(q):
    return len(q.split())


def in_range(form, q):
    lo, hi = RANGES[form]
    return lo <= n_words(q) <= hi


def filter_queries(rows, key="query", form_key="form"):
    """rows: [{form, query, ...}] -> (kept, report). Report carries the per-form drop split so a
    form whose prompt is systematically short is visible rather than quietly thinned."""
    kept, rep = [], {}
    for r in rows:
        f = r[form_key]
        c = rep.setdefault(f, {"seen": 0, "kept": 0, "short": 0, "long": 0,
                               "range": list(RANGES[f])})
        c["seen"] += 1
        w = n_words(r[key])
        lo, hi = RANGES[f]
        if w < lo:
            c["short"] += 1
        elif w > hi:
            c["long"] += 1
        else:
            c["kept"] += 1
            kept.append(r)
    for c in rep.values():
        c["drop_rate"] = round(1 - c["kept"] / max(c["seen"], 1), 4)
    return kept, rep
