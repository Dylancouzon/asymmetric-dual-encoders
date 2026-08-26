"""The objective-by-dataset field table the mandate requires BEFORE training: for each source,
the fields used, the rights relied on, how the positive is constructed, and the usable pair
count after held-out slicing and fingerprint decontamination.

Counts are read from the built mix and the decontamination summary, never hand-copied.
"""
import json

from _paths import REPO, WORK
import mix
from trainmix import heldout

SPEC = {
    "hotpotqa-train": {
        "fields": "BeIR/hotpotqa queries.text; BeIR/hotpotqa-qrels train split (score>0); "
                  "BeIR/hotpotqa corpus title+text",
        "rights": "CC BY-SA 4.0, dataset and underlying Wikipedia corpus, hotpotqa.github.io",
        "positive": "every train-qrels document with score>0 (the supporting paragraphs)",
        "objectives": "A and B",
    },
    "fever-train": {
        "fields": "BeIR/fever queries.text (claims); BeIR/fever-qrels train split; corpus title+text",
        "rights": "CC BY-SA, fever.ai/download/fever/license.html (per-article Wikipedia terms, "
                  "3.0 fallback)",
        "positive": "every train-qrels evidence document with score>0",
        "objectives": "A and B. Kept or dropped by the phase-5 dev ablation: keeping it makes "
                      "BEIR FEVER an in-domain untouched-final row, which the report labels.",
    },
    "squad-train": {
        "fields": "rajpurkar/squad train: question, context",
        "rights": "CC BY-SA 4.0, canonical HF repo (rajpurkar/squad)",
        "positive": "the question's own context paragraph; contexts deduplicated into the store",
        "objectives": "A and B",
    },
    "esci-us": {
        "fields": "tasksource/esci, product_locale=='us': query, query_id, product_id, "
                  "esci_label, product_text",
        "rights": "Apache 2.0 at amazon-science/esci-data repo root (the repo is the dataset). "
                  "Caveat: unanswered issue #21 asks whether Apache-2.0 covers the data.",
        "positive": "esci_label=='Exact'. esci_label=='Irrelevant' is kept as a provided hard "
                    "negative -- a graded judgment, not a mined guess.",
        "objectives": "A and B",
    },
    "mrtydi-en": {
        "fields": "castorini/mr-tydi english train (parquet mirror): query, positive_passages, "
                  "negative_passages",
        "rights": "Apache 2.0, castorini/mr.tydi LICENSE",
        "positive": "positive_passages (title + text); negative_passages kept as provided hard negatives",
        "objectives": "A and B",
    },
}
QUERYTEXT = {
    "nqopen": {
        "fields": "google-research-datasets/nq_open train: question only",
        "rights": "CC BY-SA 3.0, declared by Google's maintainers in merged PR #11 "
                  "(2019-06-10, commit c307fa7030); dropped from the live README in Aug 2019, "
                  "so cite the commit. The repo LICENSE (Apache-2.0) covers code only.",
        "positive": "none -- query text only",
        "objectives": "B only",
    },
    "triviaqa": {
        "fields": "mandarjoshi/trivia_qa rc.nocontext train: question only",
        "rights": "Apache 2.0 for the QA pairs (mandarjoshi/triviaqa). Evidence documents keep "
                  "their own copyright, so they are never used -- this is why TriviaQA is B-only.",
        "positive": "none -- query text only",
        "objectives": "B only",
    },
}


def build():
    kept = json.loads((WORK / "decontam" / "kept.json").read_text())
    kq_p = WORK / "decontam" / "kept_querytext.json"
    kq = json.loads(kq_p.read_text()) if kq_p.exists() else {}
    rows = []
    for src, spec in SPEC.items():
        if not (WORK / "train" / "sources" / f"{src}.json").exists():
            continue
        blob = mix.load_source(src)
        pairs = blob["pairs"]
        n_all = len(pairs)
        n_ho = sum(1 for p in pairs if heldout(src, p["qid"]))
        n_train = n_all - n_ho
        n_kept = len(kept.get(src, []))
        rows.append({"source": src, **spec, "docstore": blob["docstore"],
                     "queries_total": n_all, "queries_heldout_dev": n_ho,
                     "queries_train": n_train, "queries_after_decontam": n_kept,
                     "positives_total": sum(len(p["pos"]) for p in pairs),
                     "provided_hard_negatives": sum(len(p.get("hardneg", [])) for p in pairs)})
    for src, spec in QUERYTEXT.items():
        p = WORK / "train" / "querytext" / f"{src}.json"
        if not p.exists():
            continue
        qs = json.loads(p.read_text())
        n_ho = sum(1 for i in range(len(qs)) if heldout(src, str(i)))
        rows.append({"source": src, **spec, "docstore": "-",
                     "queries_total": len(qs), "queries_heldout_dev": n_ho,
                     "queries_train": len(qs) - n_ho,
                     "queries_after_decontam": len(kq.get(src, [])) or None,
                     "positives_total": 0, "provided_hard_negatives": 0})
    return rows


def main():
    rows = build()
    (REPO / "results" / "m7_field_table.json").write_text(json.dumps(rows, indent=1))
    L = ["# M7 objective-by-dataset field table",
         "",
         "Written before the first training run, as `instructions-m7.md` requires. Counts are read",
         "from the built mix and `results/m7_decontam.json`, never hand-copied. Rights are",
         "source-level (see `m7/LEDGER.md` for the primary URLs and caveats).",
         "",
         "| source | objectives | queries (total) | held-out dev | TRAIN | after decontam | positives | provided hard negs |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['source']} | {r['objectives'].split('.')[0]} | {r['queries_total']:,} | "
                 f"{r['queries_heldout_dev']:,} | {r['queries_train']:,} | "
                 f"{('%s' % format(r['queries_after_decontam'], ',')) if r['queries_after_decontam'] else '—'} | "
                 f"{r['positives_total']:,} | {r['provided_hard_negatives']:,} |")
    L += ["", "## Fields, rights, and positive construction", ""]
    for r in rows:
        L += [f"### {r['source']}", f"- **fields**: {r['fields']}", f"- **rights**: {r['rights']}",
              f"- **positive**: {r['positive']}", f"- **objectives**: {r['objectives']}",
              f"- **doc store**: `{r['docstore']}`", ""]
    (REPO / "results" / "m7_field_table.md").write_text("\n".join(L))
    print("\n".join(L[:12]))
    print(f"\nwrote results/m7_field_table.md and .json ({len(rows)} sources)")


if __name__ == "__main__":
    main()
