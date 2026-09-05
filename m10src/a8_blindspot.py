"""The W10 evidence: what the registered A8 gate reads vs a sensitive diagnostic, on real data.

Not a registered instrument. It exists so W10 is decided on numbers rather than in the abstract.
Writes results/m10_a8_blindspot.json.
"""
import sys, json, glob, collections
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))
import decontam, qfilter, corpus10 as C


def kgrams(words, k):
    if len(words) <= k:
        return {tuple(words)}
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def prop_rate(texts, k=4, frac=0.5):
    """Near-dup if >= `frac` of the SMALLER gram set is shared with an earlier text. Chosen
    because it degrades gracefully on short strings, which is exactly what 16/32 cannot do."""
    index, grams, n = collections.defaultdict(list), [], 0
    for t in texts:
        g = kgrams(decontam.norm_words(t), k)
        cnt = collections.Counter()
        for x in g:
            for j in index[x]:
                cnt[j] += 1
        dup = any(c >= frac * min(len(g), len(grams[j])) for j, c in cnt.items())
        grams.append(g)
        if dup:
            n += 1
        else:
            for x in g:
                index[x].append(len(grams) - 1)
    return n / max(len(texts), 1), n


out = {"_what": "W10 evidence: registered A8 gate vs a sensitive 4-gram diagnostic",
       "_diagnostic": "word-4-grams, near-dup if >= 50% of the smaller gram set is shared with an "
                      "earlier query. NOT REGISTERED -- it establishes direction, not a verdict.",
       "_a8_registered": "word-8-gram bottom-32 sketch, >= 16/32 with an earlier query; action "
                         "cuts to representatives above a 25% rate",
       "_floor": "an N-word query has N-7 word-8-grams, so 16/32 is unreachable below N = 23",
       "generated_smoke": {}, "harvested_corpus": {}}

for f in ("howto", "argument", "finance", "comparison", "yesno", "conversational", "health"):
    qs = []
    smoke = REPO / "work" / "m10gen" / "smoke"
    for p in sorted(glob.glob(str(smoke / f"{f}.json"))) + \
             sorted(glob.glob(str(smoke / f"round2_{f}.json"))):
        d = json.load(open(p))
        got = d.get("queries") or d
        if isinstance(got, list):
            qs = [x["query"] if isinstance(x, dict) else x for x in got]
    if not qs:
        continue
    _r, _d, a8 = C.near_dup_gate(qs)
    r4, n4 = prop_rate(qs)
    lo, hi = qfilter.RANGES[f]
    out["generated_smoke"][f] = {
        "n": len(qs), "rubric_range": [lo, hi],
        "gate_can_fire": "always" if lo >= 23 else ("never" if hi < 23 else "partly"),
        "a8_registered_rate": a8["near_dup_rate"], "diagnostic_4gram_rate": round(r4, 4),
        "diagnostic_near_dups": n4}

import random
rows = collections.defaultdict(list)
with (REPO / "work" / "m10harvest" / "harvest_train.jsonl").open() as fh:
    for l in fh:
        r = json.loads(l)
        rows[r["form"]].append(r["text"])
random.seed(0)
for f in ("keyword", "title", "claim"):
    s = random.sample(rows[f], 60000)
    _r, _d, a8 = C.near_dup_gate(s)
    r4, n4 = prop_rate(s)
    lo, hi = qfilter.RANGES[f]
    out["harvested_corpus"][f] = {
        "n_sampled": len(s), "rubric_range": [lo, hi],
        "gate_can_fire": "always" if lo >= 23 else ("never" if hi < 23 else "partly"),
        "a8_registered_rate": a8["near_dup_rate"], "diagnostic_4gram_rate": round(r4, 4),
        "diagnostic_near_dups": n4}

(REPO / "results" / "m10_a8_blindspot.json").write_text(
    json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("generated_smoke", "harvested_corpus")}, indent=1))
