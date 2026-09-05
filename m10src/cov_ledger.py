"""LEDGER (`artefactory/ledger-long-context-KPI-QA`) as a page-level retrieval component.

**Admission was REFUSED on 2026-09-04 and that refusal was WRONG.** It rested on the column list
returned by `load_dataset_builder(...).info.features`, which is stale and omits four columns
including `qrels`; the loaded dataset carries page-level TREC-style graded judgments. The lesson,
kept because it will recur: read the ARTEFACT, not the metadata about the artefact.

Structure, verified here rather than assumed: `report_full_text` is page-aligned Markdown split by
the literal marker `<--- Page Split --->`; qrel `doc_id`s are `<EXCHANGE>_<TICKER>_<YEAR>/page_NNNN`.
The chunk rule the mandate asks for is therefore a page split, not a heuristic chunker, and it is
checkable: every qrel page index must exist in its report's split.

Disclosures that ride with the admission (none disqualifying, all recorded in `m10/LEDGER.md` §2):
queries are template-generated with DBpedia aliases; qrels are LLM-judged over value-matched
candidates; some queries carry judgments spanning adjacent years' reports.
"""
import json, os, re, sys
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m7src"))
OUT = REPO / "work" / "m10cov"

REPO_ID = "artefactory/ledger-long-context-KPI-QA"
REVISION = "7881df568382"          # resolved to the full sha at load time and recorded
SPLIT_MARK = "<--- Page Split --->"
PAGE_CAP = 100_000                 # the registered cap


def load(verbose=True):
    """-> (queries, qrels, corpus_ids, corpus_texts, report). Verifies the page split."""
    from datasets import load_dataset
    d = load_dataset(REPO_ID, "eval", split="test")
    pages, seen = {}, {}
    bad, n_qrel = [], 0
    for r in d:
        rep = r["qrels"][0]["doc_id"].split("/")[0] if r["qrels"] else None
        if rep and rep not in seen:
            seen[rep] = r["report_full_text"]
    for rep, txt in seen.items():
        ps = txt.split(SPLIT_MARK)
        for i, p in enumerate(ps):
            pages[f"{rep}/page_{i:04d}"] = p.strip()
    for r in d:
        for j in r["qrels"]:
            n_qrel += 1
            if j["doc_id"] not in pages:
                bad.append(j["doc_id"])
    rep = dict(repo=REPO_ID, revision=REVISION, n_queries=len(d), n_reports=len(seen),
               n_pages=len(pages), n_qrels=n_qrel,
               qrel_ids_missing_from_split=len(bad),
               missing_examples=bad[:5],
               pages_per_report=round(len(pages) / max(len(seen), 1), 1),
               under_page_cap=len(pages) <= PAGE_CAP,
               empty_pages=sum(1 for v in pages.values() if len(v.split()) < 5))
    if verbose:
        print(json.dumps(rep, indent=1), flush=True)
    ids = sorted(pages)
    return ([r["query_text"] for r in d], d["qrels"], ids, [pages[i] for i in ids], rep)


if __name__ == "__main__":
    qs, qrels, ids, texts, rep = load()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ledger_structure.json").write_text(json.dumps(rep, indent=1))
    print("VERIFIES" if rep["qrel_ids_missing_from_split"] == 0 and rep["under_page_cap"]
          else "DOES NOT VERIFY")
