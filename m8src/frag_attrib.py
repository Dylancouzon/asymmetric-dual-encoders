"""Which WORDS carry the fragmentation cost? A follow-on to `retention_decomp.py`, same class:
descriptive, zero new access, adopts nothing.

`results/m8_retention_decomposition.json` established that the table falls 0.050 nDCG further
behind its teacher per +1.0 subwords-per-word (t = 4.6), independent of query length (r = 0.006),
and that the mechanism is the TEACHER pulling ahead on fragmented queries rather than the table
collapsing. That is a channel D2's self-trained tokenizer can reach.

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


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "NovaSearch/stella_en_400M_v5",
        revision="ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20")

    fr = json.loads((RESULTS / "m7_final_run.json").read_text())["six"]
    table, teacher = fr["int8-table"], fr["teacher-symmetric"]

    per_ds, word_cost = {}, defaultdict(lambda: {"n": 0, "gap": 0.0, "pieces": 0, "ds": Counter()})
    for ds in SIX:
        payload = json.loads((RESULTS / "frozen_eval" / f"{ds}.json").read_text())["queries"]
        qids = sorted(set(table[ds]) & set(teacher[ds]) & set(payload))
        gaps, has_bad, frac_bad = [], [], []
        for q in qids:
            words = payload[q].split()
            if not words:
                continue
            pieces = [len(tok(w, add_special_tokens=False)["input_ids"]) for w in words]
            gap = float(teacher[ds][q]) - float(table[ds][q])
            bad = [w for w, p in zip(words, pieces) if p >= BAD_FRAG]
            gaps.append(gap)
            has_bad.append(bool(bad))
            frac_bad.append(len(bad) / len(words))
            for w, p in zip(words, pieces):
                if p >= BAD_FRAG:
                    e = word_cost[w.lower().strip(".,;:!?\"'()[]")]
                    e["n"] += 1
                    e["gap"] += gap
                    e["pieces"] = max(e["pieces"], p)
                    e["ds"][ds] += 1
        g = np.array(gaps)
        hb = np.array(has_bad)
        if len(g) < 20 or hb.sum() < 5 or (~hb).sum() < 5:
            per_ds[ds] = {"n": len(g), "note": "too few queries in one arm to compare"}
            continue
        # A crude but honest uncertainty: the SE of a difference of two independent means.
        se = float(np.sqrt(g[hb].var(ddof=1) / hb.sum() + g[~hb].var(ddof=1) / (~hb).sum()))
        per_ds[ds] = {
            "n": len(g),
            "share_of_queries_with_a_badly_fragmented_word": float(hb.mean()),
            "mean_gap_with": float(g[hb].mean()), "mean_gap_without": float(g[~hb].mean()),
            "difference": float(g[hb].mean() - g[~hb].mean()),
            "se_of_difference": se,
            "z": float((g[hb].mean() - g[~hb].mean()) / se) if se else None,
            "mean_fraction_of_words_badly_fragmented": float(np.mean(frac_bad)),
        }

    # The words themselves, ranked by TOTAL excess gap they sit on: frequency x mean gap. A word
    # that appears twice with a huge gap is a coincidence; one that appears 200 times is a channel.
    overall = float(np.mean([v["mean_gap_with"] for v in per_ds.values()
                             if "mean_gap_with" in v] or [0]))
    ranked = []
    for w, e in word_cost.items():
        if e["n"] < 5 or not w.isascii() or len(w) < 3:
            continue
        mean_gap = e["gap"] / e["n"]
        ranked.append({"word": w, "count": e["n"], "subwords": e["pieces"],
                       "mean_gap": mean_gap, "excess_vs_all": mean_gap - overall,
                       "total_excess": (mean_gap - overall) * e["n"],
                       "datasets": dict(e["ds"])})
    ranked.sort(key=lambda r: -r["total_excess"])

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "status": "DESCRIPTIVE / EXPLORATORY. Adopts nothing; may not be cited for an adoption "
                  "(LEDGER 4.6). No new evaluation access.",
        "definition": f"a word is 'badly fragmented' when WordPiece-30522 splits it into "
                      f">= {BAD_FRAG} subwords",
        "per_dataset": per_ds,
        "top_words_by_total_excess_gap": ranked[:60],
        "bottom_words_by_total_excess_gap": ranked[-20:],
        "n_distinct_badly_fragmented_words": len(word_cost),
        "caveats": [
            "The comparison within a dataset is between DIFFERENT queries, not a paired one: "
            "queries containing a rare technical term may be harder for reasons other than "
            "tokenization. This bounds the channel, it does not isolate it.",
            "ArguAna and FiQA are on the teacher's disclosed training list; their gaps are "
            "deflated and they carry the caveat at the row.",
            "Word-level attribution double-counts a query across all its fragmented words.",
        ],
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({"per_dataset": per_ds}, indent=2, default=str))
    print("\nTop 25 words by total excess gap:")
    for r in ranked[:25]:
        print(f"  {r['word'][:24]:24s} n={r['count']:4d} pieces={r['subwords']} "
              f"excess={r['excess_vs_all']:+.4f} total={r['total_excess']:+7.2f} "
              f"{list(r['datasets'])}")
    print(f"\nwrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
