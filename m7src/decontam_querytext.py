"""Rule R1 applied to the query-text-only TRAIN sources (nq-open, TriviaQA) and to the
pseudo-query pools.

decontam.py scans pairs; these sources have no positives and feed objective B only, so they get
their own pass against the same protected-query index (six + dev + untouched-final) through the
shared `decontam.query_hits`. Pseudo-queries are spans of TRAIN documents and are TRAIN queries
under the mandate's "all partitions" wording, so they are filtered here too.

`mix.query_texts` and `pseudoq.build_decontaminated` both RAISE if the outputs of this script are
missing -- an earlier version silently fell back to unfiltered text, which would have trained
objective B on 213 nq-open and 155 TriviaQA queries that overlap protected queries.
"""
import json

from _paths import REPO, WORK
from decontam import OUT, protected_query_index, query_hits
from trainmix import heldout

TRAIN = WORK / "train"
SOURCES = ["nqopen", "triviaqa"]


def main():
    q_ex, q_gram, q_whole, counts = protected_query_index()
    print(f"protected queries: {counts}; index {len(q_ex):,} exact, "
          f"{q_gram.size:,} 8-grams", flush=True)

    summary, kept = {}, {}
    for s in SOURCES:
        p = TRAIN / "querytext" / f"{s}.json"
        if not p.exists():
            continue
        qs = json.loads(p.read_text())
        train_idx = [i for i in range(len(qs)) if not heldout(s, str(i))]
        keep, drop = [], {"exact": 0, "near": 0}
        for i in train_idx:
            h = query_hits(qs[i], q_ex, q_gram, q_whole)
            if h:
                drop[h] += 1
            else:
                keep.append(i)
        kept[s] = keep
        summary[s] = {"n_total": len(qs), "n_heldout": len(qs) - len(train_idx),
                      "n_train": len(train_idx), "n_kept": len(keep), "dropped": drop}
        print(f"{s}: {len(train_idx):,} train -> {len(keep):,} kept  dropped {drop}", flush=True)
    (OUT / "kept_querytext.json").write_text(json.dumps(kept))

    ps_dir = WORK / "pseudoq"
    ps_summary = {}
    if ps_dir.exists():
        for f in sorted(ps_dir.glob("pseudoq-*.json")):
            qs = json.loads(f.read_text())
            keep, drop = [], {"exact": 0, "near": 0}
            for i, q in enumerate(qs):
                h = query_hits(q, q_ex, q_gram, q_whole)
                if h:
                    drop[h] += 1
                else:
                    keep.append(i)
            (ps_dir / f"kept-{f.stem}.json").write_text(json.dumps(keep))
            ps_summary[f.name] = {"n": len(qs), "n_kept": len(keep), "dropped": drop}
            print(f"{f.name}: {len(qs):,} -> {len(keep):,} kept  dropped {drop}", flush=True)

    out = {"querytext_sources": summary, "pseudo_query_pools": ps_summary,
           "protected_query_counts": counts}
    (REPO / "results" / "m7_decontam_querytext.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
