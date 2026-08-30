"""Shadow candidates, measured — so that if E10 reopens, the owner's decision comes with numbers.

LEDGER 2.3 registers LoTTE as the mandatory shadow, and S0 is the screen that decides whether it
can serve. If S0 rejects it, the pipeline loses a STOP gate and E10 goes back to Dylan. That is
his decision, not the session's — but handing him a decision with no alternatives attached is a
worse handoff than handing him one with them measured.

THE ALTERNATIVE THE PLAN DID NOT NAME. CQADupStack has **twelve** subforums. This project uses
four: `programmers` and `physics` as dev components, `android` and `english` as two of the four
reserved confirmatory sets. **Eight are entirely unused**: gaming, gis, mathematica, stats, tex,
unix, webmasters, wordpress. They are CC BY-SA 3.0 under a licence this project has already
verified from the primary source (the ADCS 2015 paper, a 2014 StackExchange dump predating the
2024 clickwrap), they already have a loader in the harness, and they have never been read.

Its properties, stated honestly against LoTTE's:

  FOR  already licence-cleared, with no split like LoTTE's non-commercial `search` queries; zero
       acquisition cost; a loader that already exists; and no contamination against the reserved
       android/english, which are DIFFERENT subforums.
  AGAINST  it is the same benchmark family as two of the four reserved sets, so it is a weaker
       independence check than a genuinely separate corpus would be. That objection is real —
       but it applies to LoTTE too, which is also StackExchange, and the shadow is a
       non-regression gate crossed ONCE after the manifest is immutable, not a selection
       instrument, so the tuning channel it opens is narrow.

This script only COUNTS. It reads corpus and query sizes and writes them to
`results/m8_shadow_alternatives.json`. It scores nothing, ranks nothing, and caches no qrels into
`work/dev/` — that pattern is exactly what produced the 2026-08-29 near-miss (LEDGER §15).
"""
import json
import sys

import m8base

RESULTS = m8base.RESULTS
OUT = RESULTS / "m8_shadow_alternatives.json"

USED = {"programmers": "dev component", "physics": "dev component",
        "android": "RESERVED confirmatory", "english": "RESERVED confirmatory"}
UNUSED = ("gaming", "gis", "mathematica", "stats", "tex", "unix", "webmasters", "wordpress")


def count(sub):
    from datasets import load_dataset
    import paths_guard
    paths_guard.ensure_loader_guard()          # the reserved subforums must still be refused
    corpus = load_dataset(f"mteb/cqadupstack-{sub}", "corpus")["corpus"]
    queries = load_dataset(f"mteb/cqadupstack-{sub}", "queries")["queries"]
    qrel_rows = load_dataset(f"mteb/cqadupstack-{sub}", "default", split="test")
    qids = {str(r["query-id"]) for r in qrel_rows}
    return {"n_docs": len(corpus), "n_queries_total": len(queries),
            "n_queries_with_qrels": len(qids), "n_qrel_rows": len(qrel_rows)}


def main():
    rows, total_d, total_q = {}, 0, 0
    for sub in UNUSED:
        try:
            r = count(sub)
        except Exception as e:                                  # noqa: BLE001
            rows[sub] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  {sub:12s} ERROR {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        rows[sub] = r
        total_d += r["n_docs"]
        total_q += r["n_queries_with_qrels"]
        print(f"  {sub:12s} {r['n_docs']:>7,} docs  {r['n_queries_with_qrels']:>6,} queries",
              flush=True)

    out = {
        "_note": __doc__.strip().splitlines()[0],
        "status": "COUNTS ONLY. Nothing scored, nothing ranked, no qrels cached. Whether a shadow "
                  "is substituted is E10, which is Dylan's ruling (LEDGER 2.3 / G9).",
        "cqadupstack_used_by_this_project": USED,
        "cqadupstack_unused": rows,
        "totals_unused": {"n_docs": total_d, "n_queries": total_q, "n_subforums": len(UNUSED)},
        "for": ["licence already verified from the primary source (ADCS 2015, a 2014 dump "
                "predating the 2024 StackExchange clickwrap) -- and with no non-commercial "
                "split like LoTTE's `search` queries",
                "zero acquisition cost; the loader already exists in devsuite",
                "no contamination against the reserved android/english: different subforums"],
        "against": ["same benchmark family as two of the four reserved sets, so a weaker "
                    "independence check than a separate corpus -- though LoTTE is also "
                    "StackExchange, and the shadow is a once-crossed non-regression gate, not a "
                    "selection instrument"],
        "what_it_does_not_settle": "whether a same-family shadow is acceptable as the STOP gate "
                                   "E10 mandates. That is the ruling, and it is Dylan's.",
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n{len(UNUSED)} unused subforums: {total_d:,} docs, {total_q:,} queries")
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
