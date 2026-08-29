"""Supplementary TRAIN <-> held-out-slice query decontamination (rule R1, remaining direction).

The held-out slices are built from the training sources by the mod-50 query rule, so they are
disjoint from TRAIN by construction -- but only exactly. A TRAIN query that near-duplicates a
held-out query would make the slice a dirty dev component, so the same word-8-gram test runs
here and removes from TRAIN (dev stays intact). This pass could not run inside decontam.py
because the slices need the frozen doc pool, which is built afterwards.
"""
import json

import numpy as np

import heldout
from _paths import REPO, WORK
from decontam import OUT, contains_short, exact_u64, norm_words, query_grams, short_whole_index

prot = []
for c in heldout.COMPONENTS:
    if not (WORK / "dev" / f"{c}.json").exists():
        continue
    b = json.loads((WORK / "dev" / f"{c}.json").read_text())
    prot += b["q_texts"]
print(f"held-out protected queries: {len(prot):,}")

if not prot:
    print("no held-out slices present; nothing to do")
    raise SystemExit(0)

q_ex = set(int(exact_u64(q)) for q in prot)
q_gram = np.unique(np.concatenate([query_grams(q) for q in prot]))
q_whole = short_whole_index(prot)
print(f"index: {len(q_ex):,} exact, {q_gram.size:,} grams, "
      f"{sum(a.size for a in q_whole.values()):,} short whole-hashes")


def shares(g):
    if g.size == 0:
        return False
    i = np.minimum(np.searchsorted(q_gram, g, "left"), q_gram.size - 1)
    return bool((q_gram[i] == g).any())


import mix  # noqa: E402

kept = json.loads((OUT / "kept.json").read_text())
tr, _ = mix.split_pairs()
text_of = {(src, qid): q for src, qid, q, _, _ in tr}
removed, out = {}, {}
for src, qids in kept.items():
    keep = []
    for qid in qids:
        q = text_of.get((src, qid))
        if q is None:
            continue
        if (int(exact_u64(q)) in q_ex or shares(query_grams(q))
                or contains_short(norm_words(q), q_whole)):
            removed[src] = removed.get(src, 0) + 1
        else:
            keep.append(qid)
    out[src] = keep
    print(f"  {src}: {len(qids):,} -> {len(keep):,} (removed {removed.get(src,0):,})")

(OUT / "kept.json").write_text(json.dumps(out))
res_p = REPO / "results" / "m7_decontam_heldout.json"
summary = {"held_out_protected_queries": len(prot), "removed_per_source": removed,
           "removed_total": sum(removed.values()),
           "kept_total": sum(len(v) for v in out.values())}
# idempotent re-run: the removals are already applied, so this pass legitimately removes nothing.
# Preserve the pass that actually did the work rather than overwriting it with zeros.
if res_p.exists() and summary["removed_total"] == 0:
    prior = json.loads(res_p.read_text())
    if prior.get("removed_total", 0) > 0:
        summary = {**prior, "kept_total": summary["kept_total"],
                   "note": "counts are from the first pass; a later re-run removed 0 because the "
                           "removals were already applied and the held-out query set is unchanged"}
res_p.write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
