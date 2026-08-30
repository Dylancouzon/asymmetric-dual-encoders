"""Which WORDS carry the fragmentation cost? A follow-on to `retention_decomp.py`, same class:
descriptive, zero new access, adopts nothing.

`results/m8_retention_decomposition.json` measured that the table falls further behind its teacher
as subword fragmentation rises, and that the mechanism is the TEACHER pulling ahead on fragmented
queries rather than the table collapsing. That is a channel D2's self-trained tokenizer can reach.

But "train a 64-128K tokenizer" is not yet a decision until you know WHAT it has to cover. Two
possibilities with very different consequences:

  * the cost is carried by ordinary English that WordPiece-30522 happens to split badly. Then a
    tokenizer trained on general text fixes it, and the pool composition barely matters.
  * the cost is carried by DOMAIN vocabulary the 2018 WordPiece vocabulary never saw -- "covid",
    "sars", chemical and financial terms, code identifiers. Then the tokenizer must be trained on
    text that CONTAINS those terms, which makes the genre bundle (LEDGER §3) a prerequisite for D2
    rather than an independent lever, and it makes D2's coverage spec load-bearing.

This measures which. It reports, within each dataset so the dataset fixed effect drops out:
  * the gap for queries containing a badly-fragmented word (>= 3 subwords) versus those without;
  * the specific words that fragment most, weighted by how often they appear;
  * whether those words are domain terms or ordinary English, by dataset.

TWO CORRECTIONS FROM THE 2026-08-29 REVIEW, both of which changed numbers rather than wording:
  1. A dataset where nearly every query contains a fragmented word has NO CONTRAST to measure and
     is excluded from the sign test. ArguAna is the case: 174-word queries, ~99% of them contain
     one, contrast +0.0008 (z = 0.02). Counting its coin-flip sign in a "6/6 consistent" tally
     while the same paragraph called it uninformative was having it both ways.
  2. The word ranking's baseline was the cross-dataset mean of WITH-arm gaps, so a dataset's
     overall difficulty leaked into every one of its words -- `covid-19` scored high partly
     because trec-covid is hard. Each word is now scored against ITS OWN dataset's without-arm
     mean.

STATUS: descriptive / exploratory. Adopts nothing and may not be cited for an adoption
(LEDGER 4.6). It sharpens D2's design; it does not license D2.
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

import m8base

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_fragmentation_attribution.json"
SIX = list(m8base.SIX)
BAD_FRAG = 3          # a word split into >= 3 subwords is "badly fragmented"
PUNCT = ".,;:!?\"'()[]{}<>"


def _clean(w):
    """Strip punctuation BEFORE tokenizing. The first version tokenized the raw whitespace token
    with its punctuation attached and stripped only for the ranking key, so a comma or bracket
    could push an ordinary word over the >= 3-subword threshold and into the 'badly fragmented'
    arm. Same string in both places now."""
    return w.strip(PUNCT)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "NovaSearch/stella_en_400M_v5",
        revision="ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20")

    fr = json.loads((RESULTS / "m7_final_run.json").read_text())["six"]
    table, teacher = fr["int8-table"], fr["teacher-symmetric"]

    # PASS 1: per-query gaps, fragmented-word membership, and each dataset's own without-arm mean.
    scanned = {}
    for ds in SIX:
        payload = json.loads((RESULTS / "frozen_eval" / f"{ds}.json").read_text())["queries"]
        qids = sorted(set(table[ds]) & set(teacher[ds]) & set(payload))
        recs = []
        for q in qids:
            words = [w for w in (_clean(x) for x in payload[q].split()) if w]
            if not words:
                continue
            pieces = [len(tok(w, add_special_tokens=False)["input_ids"]) for w in words]
            bad = [w.lower() for w, pc in zip(words, pieces) if pc >= BAD_FRAG]
            recs.append({"gap": float(teacher[ds][q]) - float(table[ds][q]),
                         "bad": bad, "n_words": len(words),
                         "pieces": {w.lower(): pc for w, pc in zip(words, pieces)
                                    if pc >= BAD_FRAG}})
        scanned[ds] = recs

    per_ds, base_without = {}, {}
    for ds, recs in scanned.items():
        g = np.array([r["gap"] for r in recs])
        hb = np.array([bool(r["bad"]) for r in recs])
        if len(g) < 20 or hb.sum() < 5 or (~hb).sum() < 5:
            per_ds[ds] = {"n": len(g), "note": "too few queries in one arm to compare"}
            continue
        base_without[ds] = float(g[~hb].mean())
        se = float(np.sqrt(g[hb].var(ddof=1) / hb.sum() + g[~hb].var(ddof=1) / (~hb).sum()))
        share = float(hb.mean())
        per_ds[ds] = {
            "n": len(g), "share_of_queries_with_a_badly_fragmented_word": share,
            "mean_gap_with": float(g[hb].mean()), "mean_gap_without": base_without[ds],
            "difference": float(g[hb].mean() - base_without[ds]),
            "se_of_difference": se,
            "z": float((g[hb].mean() - base_without[ds]) / se) if se else None,
            "mean_fraction_of_words_badly_fragmented":
                float(np.mean([len(r["bad"]) / r["n_words"] for r in recs])),
            # A dataset where almost every query is in the WITH arm has no contrast to measure.
            "informative": bool(0.05 <= share <= 0.95),
        }

    # PASS 2: word attribution, each word against ITS OWN dataset's without-arm mean.
    word_cost = defaultdict(lambda: {"n": 0, "excess": 0.0, "pieces": 0, "ds": Counter()})
    for ds, recs in scanned.items():
        if ds not in base_without:
            continue
        for r in recs:
            if not r["bad"]:
                continue
            ex = r["gap"] - base_without[ds]
            for w in set(r["bad"]):
                e = word_cost[w]
                e["n"] += 1
                e["excess"] += ex
                e["pieces"] = max(e["pieces"], r["pieces"][w])
                e["ds"][ds] += 1

    ranked = []
    for w, e in word_cost.items():
        if e["n"] < 5 or not w.isascii() or len(w) < 3:
            continue
        ranked.append({"word": w, "count": e["n"], "subwords": e["pieces"],
                       "mean_excess_vs_own_dataset": e["excess"] / e["n"],
                       "total_excess": e["excess"], "datasets": dict(e["ds"])})
    ranked.sort(key=lambda r: -r["total_excess"])

    # The sign test, over INFORMATIVE datasets only.
    inf = {d: v for d, v in per_ds.items() if v.get("informative")}
    pos = sum(1 for v in inf.values() if v["difference"] > 0)
    k, n = pos, len(inf)
    from math import comb
    p_one_sided = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n) if n else None

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "status": "DESCRIPTIVE / EXPLORATORY. Adopts nothing; may not be cited for an adoption "
                  "(LEDGER 4.6). No new evaluation access.",
        "definition": f"a word is 'badly fragmented' when WordPiece-30522 splits the "
                      f"punctuation-stripped token into >= {BAD_FRAG} subwords",
        "per_dataset": per_ds,
        "sign_test_informative_only": {
            "informative_datasets": sorted(inf), "n": n, "positive": k,
            "p_one_sided": p_one_sided,
            "_caveat": "the one-sided direction was chosen after seeing section 17's slope, so "
                       "this p is descriptive dressing, not an inferential claim. Excluded as "
                       "uninformative (no contrast to measure): "
                       + ", ".join(sorted(set(per_ds) - set(inf))),
        },
        "top_words_by_total_excess_gap": ranked[:60],
        "n_distinct_badly_fragmented_words": len(word_cost),
        "caveats": [
            "The comparison within a dataset is between DIFFERENT queries, not a paired one: "
            "queries containing a rare technical term may be harder for reasons other than "
            "tokenization. This bounds the channel; it does not isolate it.",
            "ArguAna and FiQA are on the teacher's disclosed training list; their gaps are "
            "deflated and they carry the caveat at the row.",
            "Word-level attribution counts a query once per DISTINCT fragmented word it contains, "
            "so a query with three of them contributes to three words.",
            "Excess is now measured against each word's OWN dataset's without-arm mean, so a "
            "hard dataset no longer inflates every word that appears in it.",
        ],
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    for ds, v in per_ds.items():
        if "difference" in v:
            print(f"{ds:12s} n={v['n']:5d} share={v['share_of_queries_with_a_badly_fragmented_word']:.2f} "
                  f"diff={v['difference']:+.4f} z={v['z']:+.2f} "
                  f"{'informative' if v['informative'] else 'NO CONTRAST'}")
    print(f"\nsign test over informative datasets: {k}/{n} positive, one-sided p={p_one_sided}")
    print("\nTop 20 words by total excess (vs own dataset's without-arm mean):")
    for r in ranked[:20]:
        print(f"  {r['word'][:22]:22s} n={r['count']:4d} pieces={r['subwords']} "
              f"mean_excess={r['mean_excess_vs_own_dataset']:+.4f} "
              f"total={r['total_excess']:+7.2f} {list(r['datasets'])}")
    print(f"\nwrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
