"""LEDGER G2 class (b) -- the one module that may read protected query text, in order to protect
against it. Two jobs, both prerequisites of the binding pipeline (LEDGER 6 steps 1-2):

  S0  the LoTTE overlap screen: which shadow slices are clean enough to use (LEDGER 2.3).
  FILTER  the protected-query fingerprint inventory covering six + reserved-4 + surviving LoTTE
          + M9-reserve, which every downstream decontamination pass reads.

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
                  note="S0 overlap screen + the protected-query fingerprint inventory")

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
    ap.add_argument("step", choices=["s0", "filter", "fitlist", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="documents per slice; use for the smoke, never for the real run")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.step == "fitlist":
        build_fitlist(limit=a.limit)
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
