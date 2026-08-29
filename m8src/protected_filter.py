"""LEDGER G2 class (b) -- the one module that may read protected query text and protected
CQADupStack document text, in order to protect against both. Three jobs, all prerequisites of the
binding pipeline (LEDGER 6 steps 1-2):

  S0      the LoTTE overlap screen: which shadow slices are clean enough to use (LEDGER 2.3).
  REMEDY  per-item removal, re-screen and remediated-file write for the seven LoTTE shadow slices
          S0 keeps (LEDGER `E10-REMEDY`, `m8/LEDGER.md` 2026-08-29 entry) -- see the REMEDY
          section below for the full contamination-mechanics rationale.
  FILTER  the protected-query fingerprint inventory covering six + reserved-4 + surviving LoTTE
          + M9-reserve, which every downstream decontamination pass reads.

REMEDY moved into this module rather than staying in its own file (`m8src/freeze_lotte.py`, which
now holds only the PIN step) for one reason: `paths_guard.claim()` permits exactly one entry per
process, and this module already claims the allowlist entry that covers every kind REMEDY needs --
`untouched_labels` (the protected query index) AND `lotte` (the CQADupStack corpora and the raw
LoTTE slices). `freeze_lotte.py`'s own entry only ever covered `lotte`, so it could never legally
read the protected query index, and it could not import this module to borrow the capability
either -- one process, one claim, and there is no ordering of two `claim()` calls that avoids the
collision. Keeping REMEDY's two copied instruments (`_rate`, the CQADupStack index builder) in a
second file was the symptom; moving REMEDY to where the capability already lives is the fix. The
instruments live where the capability lives.

It emits a **query-only hash inventory** -- fingerprints, never labels, never text -- so that no
downstream process needs the label-bearing capability this one has. That separation is the point
of the allowlist: exactly one module holds the key, and what it hands on cannot unlock anything.

Method is M7's, unchanged and reused rather than reimplemented (`m7src/decontam.py`): blake2b-64
word hashes, polynomial rolling word-8-grams, bottom-32 sketch, >= 8/32 shared (est. Jaccard
>= 0.25), plus word-4-grams for 4-7-word queries on query paths. The index is built over the
SMALL side and the large corpus is streamed against it -- 134K CQADupStack documents indexed,
5.2M LoTTE documents streamed -- because the other loop order is what turned a 165-second job
into 3.6 hours in M7 (m7/CODEMAP.md pitfall 7).

S0's bar, from `m8/registry.json` and not from this file: drop a slice on any community-name
intersection, a document near-duplicate rate > 0.5%, or ANY query-leakage hit (zero tolerance --
a leaked query is a scored query). Reopen E10 with Dylan if the surviving near-duplicate rate
exceeds 2% or fewer than two topics survive in the crossing split.

THE STANDING USE LIMIT on REMEDY's output (LEDGER 2026-08-29, step 5, restated because nothing
mechanical enforces it): the shadow is a CHECK, never a selection surface. It may not be used to
choose between candidates, and the moment it is optimised against it becomes a second dev set and
stops doing its job. Nothing in this file can stop a future session from reading
`m8_lotte_remedy.json` and tuning against it -- the prohibition lives here the way a CLAUDE.md
directive binds: by being read, not by code.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import m8base
import paths_guard

paths_guard.claim("m8src.protected_filter",
                  note="S0 overlap screen + E10-REMEDY per-item removal/re-screen + the "
                       "protected-query fingerprint inventory")

import probe_guard                                    # noqa: E402
import decontam                                       # noqa: E402  (m7src, the one implementation)

REPO = m8base.REPO
LOTTE = REPO / "work" / "lotte" / "lotte"
OUT = REPO / "work" / "decontam"
OUT.mkdir(parents=True, exist_ok=True)
TOPICS = ("writing", "recreation", "science", "technology", "lifestyle")
SPLITS = ("dev", "test")
# `pooled` is the UNION of the five named topics, so screening the five covers it; screening it
# again would double-count every near-duplicate.
RESERVED_CQA = ("cqadup-android", "cqadup-english")
DEV_CQA = ("cqadup-programmers", "cqadup-physics")
# The StackExchange sites behind the four CQADupStack components we must stay clear of.
PROTECTED_COMMUNITIES = {
    "cqadup-android": {"android.stackexchange.com"},
    "cqadup-english": {"english.stackexchange.com"},
    "cqadup-programmers": {"softwareengineering.stackexchange.com",
                           "programmers.stackexchange.com"},
    "cqadup-physics": {"physics.stackexchange.com"},
}


def _rate(done, total, t0, what):
    """Progress with an HONEST rate. The first version passed a per-slice counter against a global
    start time, so it printed a cumulative average as if it were the current slice's rate -- which
    is exactly the reading the long-run discipline depends on being right (CLAUDE.md: "sanity-check
    the RATE against an estimate before walking away"). `t0` must be this slice's own start."""
    el = max(time.time() - t0, 1e-9)
    r = done / el
    eta = (total - done) / max(r, 1e-9)
    return f"{what} {done:,}/{total:,} ({el:.0f}s, {r:,.0f}/s, eta {eta/60:.1f}m)"


# ---------------------------------------------------------------- S0 -------------------------

def lotte_communities(topic, split):
    """-> set of StackExchange sites behind this slice, from metadata.jsonl's `dataset` field."""
    p = LOTTE / topic / split / "metadata.jsonl"
    if not p.exists():
        return set()
    out = set()
    with open(p) as fh:
        for line in fh:
            try:
                out.add(json.loads(line)["dataset"])
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def lotte_docs(topic, split, limit=None):
    p = LOTTE / topic / split / "collection.tsv"
    with open(p) as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                return
            t = line.rstrip("\n").split("\t", 1)
            if len(t) == 2:
                yield t[1]


def lotte_forum_queries(topic, split):
    """FORUM queries only. LoTTE's `search` queries are non-commercial-research-only (GooAQ
    licence, quoted in LEDGER 2.3); they are not read, not encoded and not scored."""
    p = LOTTE / topic / split / "questions.forum.tsv"
    if not p.exists():
        return []
    out = []
    with open(p) as fh:
        for line in fh:
            t = line.rstrip("\n").split("\t", 1)
            if len(t) == 2:
                out.append(t[1])
    return out


def _cqa_index():
    """Inverted index over the four protected CQADupStack corpora -- the SMALL side."""
    import devsuite
    per_item, exact, owner = [], [], []
    for comp in RESERVED_CQA + DEV_CQA:
        _, doc_texts, *_ = devsuite.load(comp)
        for t in doc_texts:
            per_item.append(decontam.sketch(t))
            exact.append(decontam.exact_u64(t))
            owner.append(comp)
    return decontam.Inverted(per_item, exact), np.array(owner)


def s0(limit=None, smoke=False):
    """The LoTTE overlap screen. Returns the per-slice verdicts the shadow pin consumes."""
    stamp = probe_guard.stamp("S0", strict_commit=not smoke)
    t0 = time.time()
    print("building the protected CQADupStack document index (the small side)...", flush=True)
    idx, owner = _cqa_index()
    print(f"  indexed {idx.n:,} documents, {idx.nbytes/1e6:.0f} MB, {time.time()-t0:.0f}s",
          flush=True)

    print("building the protected-query index (R1)...", flush=True)
    q_ex, q_gram, q_whole, q_counts = decontam.protected_query_index()
    print(f"  {sum(q_counts.values()):,} protected queries {q_counts}", flush=True)

    slices, t_all = {}, time.time()
    for topic in TOPICS:
        for split in SPLITS:
            t1 = time.time()                      # per-slice, so the printed rate is this slice's
            d = LOTTE / topic / split
            if not d.exists():
                continue
            key = f"{topic}/{split}"
            comm = lotte_communities(topic, split)
            comm_hits = sorted({c for s in PROTECTED_COMMUNITIES.values() for c in comm & s})

            n_docs = n_near = n_exact = n_dup = 0
            hit_owners = {}
            for text in lotte_docs(topic, split, limit=limit):
                n_docs += 1
                ex, near = idx.match(text, decontam.DUP_SHARE)
                if ex.size:
                    n_exact += 1
                # A document that is BOTH an exact and a near hit is ONE duplicate. Summing the
                # two counters double-counted it against the registered 0.5% / 2% bars.
                if ex.size or near.size:
                    n_dup += 1
                if near.size:
                    n_near += 1
                    for o in np.unique(owner[near]):
                        hit_owners[str(o)] = hit_owners.get(str(o), 0) + 1
                if n_docs % 100_000 == 0:
                    print("  " + _rate(n_docs, n_docs, t1, f"{key} docs"), flush=True)
                    # total is unknown until the file ends, so eta is 0 by construction here;
                    # the RATE is the number to read.

            qs = lotte_forum_queries(topic, split)
            q_hits = [decontam.query_hits(q, q_ex, q_gram, q_whole) for q in qs]
            n_qhit = sum(1 for h in q_hits if h)

            dup_rate = n_dup / max(n_docs, 1)
            drop_reasons = []
            if comm_hits:
                drop_reasons.append(f"community intersection {comm_hits}")
            if dup_rate > 0.005:
                drop_reasons.append(f"document near-duplicate rate {dup_rate:.4%} > 0.5%")
            if n_qhit:
                drop_reasons.append(f"{n_qhit} query-leakage hits (zero tolerance)")

            # DESCRIPTIVE ONLY, and explicitly NOT the registered rule. The registered bar drops a
            # whole slice on any query-leakage hit. The remedy this project uses everywhere else
            # for a contaminated item is to remove the ITEM (R1 removes pairs, not sources), and
            # the smoke showed the slice-level rule dropping every slice on hit rates under 1%.
            # Changing the bar now would be tuning it after watching it bite, which is forbidden
            # (m7/LEDGER.md, lever #7: "loosening a margin after measuring that it might bite is
            # tuning"). So the alternative is COMPUTED and REPORTED for Dylan's decision, and the
            # verdict below is still the registered one.
            per_query_alt = {
                "_status": "DESCRIPTIVE ALTERNATIVE, NOT ADOPTED, NOT the registered bar",
                "queries_after_dropping_leaked": len(qs) - n_qhit,
                "query_leak_rate": n_qhit / max(len(qs), 1),
                "would_keep": bool(not comm_hits and dup_rate <= 0.005
                                   and (len(qs) - n_qhit) >= 500),
            }
            slices[key] = {
                "topic": topic, "split": split,
                "per_query_remedy_alternative": per_query_alt,
                "n_docs_screened": n_docs, "n_forum_queries": len(qs),
                "communities": sorted(comm),
                "community_intersection": comm_hits,
                "doc_exact_hits": n_exact, "doc_near_hits": n_near,
                "doc_duplicate_docs": n_dup, "doc_dup_rate": dup_rate,
                "doc_hit_owners": hit_owners,
                "query_leak_hits": n_qhit,
                "query_leak_kinds": {k: q_hits.count(k) for k in ("exact", "near", "contains")},
                "verdict": "DROP" if drop_reasons else "KEEP",
                "drop_reasons": drop_reasons,
            }
            print(f"  {key:24s} {slices[key]['verdict']:5s} docs={n_docs:>9,} "
                  f"dup={dup_rate:.4%} qleak={n_qhit} {drop_reasons or ''}", flush=True)

    kept = {k: v for k, v in slices.items() if v["verdict"] == "KEEP"}
    surviving_rate = (sum(v["doc_duplicate_docs"] for v in kept.values())
                      / max(sum(v["n_docs_screened"] for v in kept.values()), 1))
    topics_by_split = {s: sorted({v["topic"] for v in kept.values() if v["split"] == s})
                       for s in SPLITS}
    reopen = []
    if surviving_rate > 0.02:
        reopen.append(f"surviving near-duplicate rate {surviving_rate:.4%} > 2%")
    for s in SPLITS:
        if len(topics_by_split[s]) < 2:
            reopen.append(f"only {len(topics_by_split[s])} topic(s) survive in {s}")

    out = {
        "_note": "LEDGER 2.3 / registry probe S0. Forum queries only; the search split is "
                 "non-commercial-research-only per the GooAQ licence and is never read.",
        "smoke": bool(smoke or limit),
        "doc_limit_per_slice": limit,
        "slices": slices,
        "kept": sorted(kept),
        "dropped": sorted(k for k in slices if k not in kept),
        "surviving_dup_rate": surviving_rate,
        "surviving_topics_by_split": topics_by_split,
        "reopen_E10": reopen,
        "verdict": ("E10 REOPENS WITH DYLAN" if reopen
                    else "PROCEED with the kept slices" if kept else "NO SLICE SURVIVES"),
        "seconds": round(time.time() - t0, 1),
        "screen_seconds": round(time.time() - t_all, 1),
    }
    dest = REPO / "results" / ("m8_lotte_overlap.SMOKE.json" if (smoke or limit)
                               else "m8_lotte_overlap.json")
    if smoke or limit:
        dest.write_text(json.dumps({**out, "_registration": stamp}, indent=2, default=str))
    else:
        probe_guard.write_result(dest, out, "S0")
    print(f"\n{out['verdict']}  ->  {dest}", flush=True)
    return out


# ---------------------------------------------------------------- REMEDY ----------------------
# LEDGER `E10-REMEDY` (`m8/LEDGER.md`, 2026-08-29 entry): per-item removal, re-screen and
# remediated-file write for the seven LoTTE shadow slices that survive S0's community-overlap bar.
#
# WHAT SURVIVES AND WHAT DOES NOT, and why REMEDY only ever remedies the former. S0 above dropped
# three slices outright -- `writing/test`, `science/test`, `technology/test` -- because their
# StackExchange COMMUNITY intersects a protected CQADupStack component (english / physics /
# android+programmers). A community-overlap drop is not a contamination RATE problem that per-item
# removal can fix; the protected set and the slice's population are the same population, so there
# is no clean remainder to keep. The seven survivors -- `writing/dev`, `recreation/dev`,
# `recreation/test`, `science/dev`, `technology/dev`, `lifestyle/dev`, `lifestyle/test` -- have no
# such community overlap; S0 measured their contamination as a low RATE among otherwise-independent
# items (leaked queries 0.10-0.75%, near-duplicate documents 0.001-0.010%), which is exactly the
# situation R1/R2-style per-item removal is for. `_assert_survivor` below refuses any DEAD slice
# name defensively, so a future caller cannot point this at a community-overlap drop by accident.
#
# THE STEP THE LEDGER SPEC DOES NOT WRITE DOWN, and why it has to be here anyway. The ledger's
# remedy is "drop the leaked queries AND the near-duplicate documents, then re-screen." But a LoTTE
# qrels entry is a (query, positive document) PAIR, and dropping a document does not know it owes
# anything to the queries that pointed at it. Two things follow, neither optional:
#   (a) every qrels pair whose positive document was dropped must itself be dropped -- otherwise
#       the remediated qas file cites a document that is no longer in the remediated collection,
#       which is not a smaller clean slice, it is a corrupt one;
#   (b) a query that loses EVERY positive this way is left with zero relevant documents. Scoring it
#       would silently produce nDCG=0 for a reason that has nothing to do with retrieval quality --
#       a data-integrity artifact wearing a hard-query costume -- so it must be dropped too.
# Both counts are kept separate from the query-leak and document-dup counts, so a reader can tell
# "removed because it leaked" from "removed because remediation orphaned it."
#
# Hashing is deliberately NOT this section's job. `remedy()` below screens, removes, re-screens and
# writes the remediated files; `freeze_lotte.pin()` is the sole hash authority, reading those files
# back fresh from disk rather than trusting a number computed inline during screening -- see
# `freeze_lotte.py`'s module docstring. So there is no `_hash_slice` here and no "hashes" field in
# `_remediate_one`'s return value.
#
# Forum queries only, always. LoTTE's `search` queries (`questions.search.tsv`, `qas.search.jsonl`)
# are GooAQ-licensed non-commercial-research-only and this section never opens them -- the same
# rule the rest of this module and the LEDGER already state.

REMEDIATED = REPO / "work" / "lotte" / "remediated"

# The seven surviving slices and the three dead ones, per `m8/LEDGER.md` 2026-08-29 ("E10's
# remedy SPECIFIED, with the exact slices"). Hardcoded because this IS the registration; a
# CLI flag that let a caller name arbitrary slices would let a future session remedy a
# community-overlap drop by typo.
SURVIVING = ("writing/dev", "recreation/dev", "recreation/test", "science/dev",
            "technology/dev", "lifestyle/dev", "lifestyle/test")
DEAD = ("writing/test", "science/test", "technology/test")
assert not (set(SURVIVING) & set(DEAD)), "a slice cannot be both surviving and dead"


def _assert_survivor(key):
    if key in DEAD:
        raise SystemExit(
            f"E10-REMEDY: {key!r} is one of the three DEAD slices (community overlap with a "
            f"protected StackExchange site) -- no remedy applies to a community-overlap drop; "
            f"per-item removal is for a contamination RATE among independent items, not for a "
            f"slice whose population IS the protected one.")
    if key not in SURVIVING:
        raise SystemExit(
            f"E10-REMEDY: {key!r} is not one of the seven surviving slices named in "
            f"m8/LEDGER.md's 2026-08-29 E10-REMEDY entry. Known survivors: {SURVIVING}.")


def _read_collection(path):
    doc_ids, doc_texts = [], []
    with open(path) as fh:
        for line in fh:
            t = line.rstrip("\n").split("\t", 1)
            if len(t) == 2:
                doc_ids.append(t[0])
                doc_texts.append(t[1])
    return doc_ids, doc_texts


def _read_questions(path):
    q_ids, q_texts = [], []
    with open(path) as fh:
        for line in fh:
            t = line.rstrip("\n").split("\t", 1)
            if len(t) == 2:
                q_ids.append(t[0])
                q_texts.append(t[1])
    return q_ids, q_texts


def _read_qas(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _require(path, slice_key):
    if not path.exists():
        raise SystemExit(
            f"E10-REMEDY: {path} is missing for surviving slice {slice_key!r}. Refusing to "
            f"silently skip a slice -- fix the input or drop the slice by name, not by omission.")


def _screen(doc_texts, qas_rows, idx, cqa_owner, q_ex, q_gram, q_whole, t0, label):
    """One pass of the two instruments over one slice's current (pre- or post-remedy) content.
    -> (dup_doc_row_indices, doc_hit_owners, leaked_qid_kind). Used both for the FIRST screen
    (to find what to drop) and the RE-screen (to confirm zero hits survive removal) -- same
    function, same instruments, so "re-screen" cannot mean something subtly different from
    "screen"."""
    n_docs = len(doc_texts)
    dup_idx, hit_owners, n_exact, n_near = set(), {}, 0, 0
    for i, text in enumerate(doc_texts):
        ex, near = idx.match(text, decontam.DUP_SHARE)
        if ex.size:
            n_exact += 1
        if ex.size or near.size:
            dup_idx.add(i)
        if near.size:
            n_near += 1
            for o in np.unique(cqa_owner[near]):
                hit_owners[str(o)] = hit_owners.get(str(o), 0) + 1
        if (i + 1) % 100_000 == 0:
            print("  " + _rate(i + 1, n_docs, t0, f"{label} docs"), flush=True)

    leaked_qid_kind, kinds = {}, {"exact": 0, "near": 0, "contains": 0}
    for j, row in enumerate(qas_rows):
        k = decontam.query_hits(row["query"], q_ex, q_gram, q_whole)
        if k:
            # str-keyed, like every other qid/doc-id comparison in this module. `qas.forum.jsonl`
            # carries whatever JSON type the source encoded, so keying on the raw value works only
            # while both sides happen to read the SAME object -- which is true today and is the
            # one place here that relied on type identity instead of normalizing. A leaked query
            # that silently failed to match its own drop test would put a known-contaminated item
            # into a partition whose entire purpose is being clean.
            leaked_qid_kind[str(row["qid"])] = k
            kinds[k] += 1
        if (j + 1) % 2_000 == 0:
            print("  " + _rate(j + 1, len(qas_rows), t0, f"{label} queries"), flush=True)

    return {"dup_idx": dup_idx, "n_exact": n_exact, "n_near": n_near, "hit_owners": hit_owners,
            "leaked_qid_kind": leaked_qid_kind, "leak_kinds": kinds}


def _remediate_one(key, idx, cqa_owner, q_ex, q_gram, q_whole, limit=None):
    _assert_survivor(key)
    topic, split = key.split("/")
    src = LOTTE / topic / split
    coll_p, quest_p, qas_p = (src / "collection.tsv", src / "questions.forum.tsv",
                              src / "qas.forum.jsonl")
    for p in (coll_p, quest_p, qas_p):
        _require(p, key)

    doc_ids, doc_texts = _read_collection(coll_p)
    if limit:
        doc_ids, doc_texts = doc_ids[:limit], doc_texts[:limit]
    q_ids, q_texts = _read_questions(quest_p)
    qas_rows = _read_qas(qas_p)

    # The remediated questions.forum.tsv is derived FROM qas_rows below (so its qid set can never
    # drift from the qrels that justify it) -- but that is only sound if the two source files
    # agree on text in the first place. A silent disagreement here would ship a remediated file
    # that disagrees with its own frozen qrels, which is worse than refusing outright.
    q_text_by_id = dict(zip(q_ids, q_texts))
    for row in qas_rows:
        qid = str(row["qid"])
        if q_text_by_id.get(qid) != row["query"]:
            raise SystemExit(
                f"E10-REMEDY: {key} qid {qid} disagrees between questions.forum.tsv and "
                f"qas.forum.jsonl -- refusing to remediate a slice whose own source files "
                f"disagree with each other.")

    n_docs_before, n_queries_before = len(doc_texts), len(qas_rows)
    n_qrels_before = sum(len(r["answer_pids"]) for r in qas_rows)

    t1 = time.time()
    print(f"[{key}] screening {n_docs_before:,} docs / {n_queries_before:,} queries...", flush=True)
    first = _screen(doc_texts, qas_rows, idx, cqa_owner, q_ex, q_gram, q_whole, t1, f"{key} (1st)")

    dup_doc_ids = {doc_ids[i] for i in first["dup_idx"]}
    leaked_qid_kind = first["leaked_qid_kind"]

    # --- drop, in the ledger's order, plus the orphan cascade the ledger spec omits -----------
    kept_doc_ids = [d for i, d in enumerate(doc_ids) if i not in first["dup_idx"]]
    kept_doc_texts = [t for i, t in enumerate(doc_texts) if i not in first["dup_idx"]]

    orphaned_qrels_pairs, zero_positive_queries, kept_rows = 0, 0, []
    for row in qas_rows:
        if str(row["qid"]) in leaked_qid_kind:      # str-keyed, see _screen
            continue                                          # leaked-query drop
        surv_pids = [p for p in row["answer_pids"] if str(p) not in dup_doc_ids]
        orphaned_qrels_pairs += len(row["answer_pids"]) - len(surv_pids)
        if not surv_pids:
            zero_positive_queries += 1                        # orphan cascade, step (b)
            continue
        new_row = dict(row)
        new_row["answer_pids"] = surv_pids
        kept_rows.append(new_row)

    keep_qid_set = {str(r["qid"]) for r in kept_rows}
    kept_q_ids = [q for q in q_ids if q in keep_qid_set]
    kept_q_texts = [q_text_by_id[q] for q in kept_q_ids]

    # --- write the remediated slice, inside work/lotte (the guard's protected "lotte" root) ---
    dest = REMEDIATED / topic / split
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "collection.tsv").write_text(
        "".join(f"{i}\t{t}\n" for i, t in zip(kept_doc_ids, kept_doc_texts)))
    (dest / "questions.forum.tsv").write_text(
        "".join(f"{i}\t{t}\n" for i, t in zip(kept_q_ids, kept_q_texts)))
    (dest / "qas.forum.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in kept_rows))

    # --- re-screen: removal is not assumed to have worked because it was performed -----------
    t2 = time.time()
    second = _screen(kept_doc_texts, kept_rows, idx, cqa_owner, q_ex, q_gram, q_whole, t2,
                     f"{key} (re-screen)")
    rescreen_hits = len(second["dup_idx"]) + len(second["leaked_qid_kind"])
    passed = rescreen_hits == 0
    disposition = "SURVIVES" if passed else "DROPPED_AT_RESCREEN"

    print(f"  [{key}] {disposition}  docs {n_docs_before:,}->{len(kept_doc_ids):,}  "
          f"queries {n_queries_before:,}->{len(kept_rows):,}  "
          f"re-screen hits doc={len(second['dup_idx'])} query={len(second['leaked_qid_kind'])}",
          flush=True)

    return {
        "topic": topic, "split": split,
        "before": {"docs": n_docs_before, "queries": n_queries_before,
                  "qrels_pairs": n_qrels_before},
        "after": {"docs": len(kept_doc_ids), "queries": len(kept_rows),
                 "qrels_pairs": sum(len(r["answer_pids"]) for r in kept_rows)},
        "dropped": {
            "leaked_query": len(leaked_qid_kind),
            "leaked_query_kinds": first["leak_kinds"],
            "dup_doc": len(first["dup_idx"]),
            "dup_doc_owners": first["hit_owners"],
            "orphaned_qrels_pairs": orphaned_qrels_pairs,
            "zero_positive_query": zero_positive_queries,
        },
        "first_screen": {"n_exact": first["n_exact"], "n_near": first["n_near"]},
        "rescreen": {"doc_hits": len(second["dup_idx"]), "query_hits": len(second["leaked_qid_kind"]),
                    "passed": passed},
        "final_disposition": disposition,
        "written_relpath": str(dest.relative_to(REPO)),
        "seconds": round(time.time() - t1, 1),
    }


def remedy(limit=None, smoke=False):
    stamp = probe_guard.stamp("E10-REMEDY", strict_commit=not smoke)
    t0 = time.time()
    print("building the protected CQADupStack document index (the small side)...", flush=True)
    idx, cqa_owner = _cqa_index()
    print(f"  indexed {idx.n:,} documents, {idx.nbytes/1e6:.0f} MB, {time.time()-t0:.0f}s",
          flush=True)

    print("building the protected-query index (six + dev + untouched-final)...", flush=True)
    q_ex, q_gram, q_whole, q_counts = decontam.protected_query_index()
    print(f"  {sum(q_counts.values()):,} protected queries {q_counts}", flush=True)

    slices = {}
    for key in SURVIVING:
        slices[key] = _remediate_one(key, idx, cqa_owner, q_ex, q_gram, q_whole, limit=limit)

    surviving_slices = sorted(k for k, v in slices.items() if v["final_disposition"] == "SURVIVES")
    dropped_at_rescreen = sorted(k for k, v in slices.items()
                                 if v["final_disposition"] != "SURVIVES")
    total_surviving_queries = sum(slices[k]["after"]["queries"] for k in surviving_slices)

    if not surviving_slices:
        verdict = "NO SLICE SURVIVES remedy + re-screen -- E10 REOPENS WITH DYLAN"
    elif dropped_at_rescreen:
        verdict = (f"PROCEED with {len(surviving_slices)}/{len(SURVIVING)} slices "
                  f"({total_surviving_queries:,} queries); re-screen DROPPED "
                  f"{dropped_at_rescreen} despite the first-pass remedy")
    else:
        verdict = (f"PROCEED: all {len(surviving_slices)} surviving slices pass remedy + "
                  f"re-screen, {total_surviving_queries:,} queries total")

    out = {
        "_note": "LEDGER 2026-08-29 E10-REMEDY. Forum queries only; the search split is "
                 "non-commercial-research-only per the GooAQ licence and is never read. "
                 "The shadow is a CHECK, never a selection surface -- see the module docstring. "
                 "Hashing is NOT done here -- freeze_lotte.pin() is the sole hash authority, "
                 "reading these remediated files back fresh from disk.",
        "smoke": bool(smoke or limit),
        "limit": limit,
        "dead_slices_no_remedy_applies": list(DEAD),
        "slices": slices,
        "surviving_slices": surviving_slices,
        "dropped_at_rescreen": dropped_at_rescreen,
        "total_surviving_queries": total_surviving_queries,
        "verdict": verdict,
        "seconds": round(time.time() - t0, 1),
    }
    dest = REPO / "results" / ("m8_lotte_remedy.SMOKE.json" if (smoke or limit)
                               else "m8_lotte_remedy.json")
    if smoke or limit:
        dest.write_text(json.dumps({**out, "_registration": stamp}, indent=2, default=str))
    else:
        probe_guard.write_result(dest, out, "E10-REMEDY")
    print(f"\n{verdict}\n-> {dest}", flush=True)
    return out


# ---------------------------------------------------------------- FILTER ----------------------

def protected_query_groups(lotte_slices=None):
    """EVERY protected query string, by partition group: six + dev + reserved-4 (from M7's own
    `decontam.protected_queries`) PLUS the surviving LoTTE forum queries and the M9-reserve query
    text, which M7's function knows nothing about.

    This exists as ONE function because it was briefly TWO. `build_filter` extended the set and
    `build_fitlist` called `decontam.protected_query_index()` directly, so the fit list was
    screened against 25,834 queries while its own manifest claimed 80,954 -- a false statement in
    an artifact, which is the failure class this project cares most about. One source now."""
    pq = decontam.protected_queries()                       # six + dev + untouched-final
    groups = {k: list(v) for k, v in pq.items()}

    if lotte_slices:
        lot = []
        for key in lotte_slices:
            topic, split = key.split("/")
            lot += lotte_forum_queries(topic, split)
        groups["lotte-shadow"] = lot

    # M9-RESERVE. The first version of this looked for a bare list of strings under `queries` and
    # silently found none, because both inventories store LISTS OF RECORDS. A filter that quietly
    # covers three partitions where the ledger says four is worse than one that fails loudly, so
    # the field names are explicit and a miss is an error, not an empty list (G7).
    M9_FIELDS = {
        "eurlex_inventory.json": [("queries", "query_text")],
        # USPTO's registered construction is query = abstract, gold doc = the application body.
        # The abstract is the query text, so that is what the protected-query index must hold.
        "uspto_inventory.json": [("records", "abstract_text"), ("records", "title")],
    }
    m9, m9_detail = [], {}
    for f, specs in M9_FIELDS.items():
        p = REPO / "work" / "m9reserve" / f
        if not p.exists():
            raise SystemExit(f"G7: {f} is missing; the filter must cover the M9 reserve before "
                             f"any teacher download or probe")
        d = json.loads(p.read_text())
        for listkey, textkey in specs:
            rows = d.get(listkey) or []
            got = [str(r[textkey]) for r in rows
                   if isinstance(r, dict) and r.get(textkey)]
            if not got:
                raise SystemExit(f"G7: {f}[{listkey}][{textkey}] yielded no query text. The "
                                 f"filter must not silently cover fewer partitions than the "
                                 f"ledger claims -- fix the field name or the inventory.")
            m9 += got
            m9_detail[f"{f}:{listkey}.{textkey}"] = len(got)
    groups["m9-reserve"] = m9

    return groups, m9_detail


def extended_index(lotte_slices=None):
    """-> (exact-key set, sorted gram array, short-whole index, per-group counts) over EVERY
    protected partition. The same three structures `decontam.query_hits` consumes."""
    groups, m9_detail = protected_query_groups(lotte_slices)
    prot = [q for v in groups.values() for q in v]
    q_ex = set(int(decontam.exact_u64(q)) for q in prot)
    q_gram = np.unique(np.concatenate([decontam.query_grams(q) for q in prot]))
    q_whole = decontam.short_whole_index(prot)
    return q_ex, q_gram, q_whole, {k: len(v) for k, v in groups.items()}, m9_detail


def build_filter(lotte_slices=None):
    """The protected-query fingerprint inventory. Emits HASHES ONLY."""
    t0 = time.time()
    groups, m9_detail = protected_query_groups(lotte_slices)
    prot = [q for v in groups.values() for q in v]
    q_ex = np.unique(np.array([decontam.exact_u64(q) for q in prot], dtype=np.uint64))
    q_gram = np.unique(np.concatenate([decontam.query_grams(q) for q in prot]))
    short = [q for q in prot if 4 <= len(decontam.norm_words(q)) <= 7]

    dest = OUT / "m8_protected_query_index.npz"
    np.savez_compressed(dest, exact=q_ex, grams=q_gram)
    import hashlib
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    meta = {
        "_note": "QUERY-ONLY fingerprint inventory. Hashes, never text, never labels: downstream "
                 "decontamination reads this and therefore never needs the label-bearing "
                 "capability that built it (LEDGER G2 class b).",
        "groups": {k: len(v) for k, v in groups.items()},
        "m9_reserve_fields": m9_detail,
        "n_protected_queries": len(prot),
        "n_exact_keys": int(q_ex.size), "n_gram_keys": int(q_gram.size),
        "n_short_queries_4to7_words": len(short),
        "method": {"ngram": decontam.NGRAM, "sketch": decontam.SKETCH,
                   "dup_share": decontam.DUP_SHARE, "short_ngram": decontam.SHORT_NGRAM},
        "npz_relpath": str(dest.relative_to(REPO)), "npz_sha256": sha,
        "lotte_slices": sorted(lotte_slices) if lotte_slices else [],
        "seconds": round(time.time() - t0, 1),
    }
    (REPO / "results" / "m8_protected_filter.json").write_text(
        json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, indent=2))
    return meta


# ---------------------------------------------------------------- FIT LIST --------------------

def build_fitlist(limit=None):
    """Regenerate the closed-form fit list THROUGH the current filter (LEDGER §3.3).

    This lives HERE, not in its own module, and the guard is why. A separate `m8src/fitlist.py`
    tried to `claim("m8src.protected_filter")` and was refused: an entry may only claim itself.
    That refusal was correct and it pointed at better architecture -- screening TRAIN queries
    against protected query TEXT *is* this module's contact class (G2 class b), so the work
    belongs in the one module that already holds the capability rather than in a second one that
    borrows it. Everything downstream still reads only the query-only hash inventory.

    M7's `work/trainq_texts.json` carries 4,582 R1 hits (1.31%) and M8's filter covers partitions
    M7's never screened -- the reserved four, the shadow, and 55,120 M9-reserve queries -- so
    "the same contaminated list for everyone" no longer covers it. M7's own pins are NOT touched
    (G3); the two lists coexist and their difference is the measurement.
    """
    import hashlib
    import pool as poolmod
    import train
    from train import Cfg

    t0 = time.time()
    index, _vecs, _meta = poolmod.build()
    q_texts, *_ = train.build_arrays(Cfg(), index)
    q_texts = list(q_texts)
    if limit:
        q_texts = q_texts[:limit]
    print(f"derived {len(q_texts):,} TRAIN query texts ({time.time()-t0:.0f}s)", flush=True)

    lot = None
    lp = REPO / "results" / "m8_lotte_overlap.json"
    if lp.exists():
        lot = json.loads(lp.read_text())["kept"] or None
    q_ex, q_gram, q_whole, counts, m9_detail = extended_index(lot)
    print(f"protected-query index: {sum(counts.values()):,} queries {counts}", flush=True)

    t1 = time.time()
    kinds, hits = {"exact": 0, "near": 0, "contains": 0}, set()
    for i, q in enumerate(q_texts):
        k = decontam.query_hits(q, q_ex, q_gram, q_whole)
        if k:
            kinds[k] += 1
            hits.add(i)
        if (i + 1) % 100_000 == 0:
            print("  " + _rate(i + 1, len(q_texts), t1, "screened"), flush=True)
    keep = [q for i, q in enumerate(q_texts) if i not in hits]

    payload = json.dumps(keep)
    dest = REPO / "work" / "m8_trainq_texts.json"
    if not limit:
        dest.write_text(payload)
    meta = {
        "_what": "the M8 closed-form fit list: TRAIN query texts screened through the CURRENT "
                 "protected-query filter. `screened_against` records exactly which partitions "
                 "that was -- an earlier version claimed M9-reserve coverage it did not have.",
        "_why_not_in_place": "work/trainq_texts.json and results/m7_trainq_manifest.json are M7's "
                             "provenance pins for a frozen, released system; G3 forbids M8 from "
                             "editing M7's record.",
        "screened_against": counts, "n_protected_queries": sum(counts.values()),
        "m9_reserve_fields": m9_detail,
        "n_derived": len(q_texts), "n_kept": len(keep), "n_removed": len(hits),
        "removal_rate": len(hits) / max(len(q_texts), 1), "hits_by_kind": kinds,
        "m7_list_known_r1_hits": 4582, "m7_list_known_r1_rate": 0.0131,
        "relpath": str(dest.relative_to(REPO)),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(), "bytes": len(payload),
        "produced_by": "m8src/protected_filter.py fitlist", "smoke": bool(limit),
        "seconds": round(time.time() - t0, 1),
    }
    if not limit:
        (REPO / "results" / "m8_trainq_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["s0", "filter", "fitlist", "remedy", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="documents per slice; use for the smoke, never for the real run")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.step == "fitlist":
        build_fitlist(limit=a.limit)
        return 0
    if a.step == "remedy":
        remedy(limit=a.limit, smoke=a.smoke)
        return 0
    kept = None
    if a.step in ("s0", "all"):
        r = s0(limit=a.limit, smoke=a.smoke)
        kept = r["kept"]
    if a.step in ("filter", "all"):
        if kept is None:
            p = REPO / "results" / "m8_lotte_overlap.json"
            kept = json.loads(p.read_text())["kept"] if p.exists() else None
        build_filter(kept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
