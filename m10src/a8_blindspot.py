"""W10's evidence: the A8 gate beside a sensitive diagnostic, on real data.

One-shot. It exists so W10 was decided on numbers rather than in the abstract, and its output is
frozen at `results/m10_a8_blindspot.json` and tabulated in `m10/LEDGER.md` §W10. Kept only so the
artifact is reproducible; all the measurement lives in `corpus10`.
"""
import collections, glob, json, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import corpus10 as C
import qfilter


def _rows(form):
    smoke = REPO / "work" / "m10gen" / "smoke"
    for p in sorted(glob.glob(str(smoke / f"{form}.json"))) + \
             sorted(glob.glob(str(smoke / f"round2_{form}.json"))):
        got = json.load(open(p))
        got = got.get("queries") or got
        if isinstance(got, list):
            return [x["query"] if isinstance(x, dict) else x for x in got]
    return []


def measure(texts, form):
    _r, _d, a8 = C.near_dup_gate(texts)
    lo, hi = qfilter.RANGES[form]
    return {"n": len(texts), "rubric_range": [lo, hi],
            "a8_rate": a8["near_dup_rate_raw"],
            "caught_only_by_short_rule": a8["caught_only_by_short_rule"],
            "diagnostic_4gram_rate": round(C.prop_near_dup_rate(texts)[0], 4),
            "diagnostic_rate_vs_n": {str(k): round(C.prop_near_dup_rate(texts[:k])[0], 4)
                                     for k in (50, 100, 150, 200) if k <= len(texts)}}


def main():
    out = {"_what": "W10 evidence: the A8 gate beside a sensitive 4-gram diagnostic",
           "_diagnostic": "word-4-grams, near-dup at >= 50% of the smaller set. NOT REGISTERED.",
           "_monotone": "the rate is non-decreasing in n, so every value is a FLOOR for 143,000",
           "generated_smoke": {}, "harvested_corpus": {}}
    for f in ("howto", "argument", "finance", "comparison", "yesno", "conversational", "health"):
        qs = _rows(f)
        if qs:
            out["generated_smoke"][f] = measure(qs, f)
    rows = collections.defaultdict(list)
    with (REPO / "work" / "m10harvest" / "harvest_train.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            rows[r["form"]].append(r["text"])
    random.seed(0)
    for f in ("keyword", "title", "claim"):
        out["harvested_corpus"][f] = measure(random.sample(rows[f], 60_000), f)
    (REPO / "results" / "m10_a8_blindspot.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
