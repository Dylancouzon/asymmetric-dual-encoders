"""Regenerate the closed-form fit list THROUGH the M8 protected-query filter (LEDGER §3.3).

The teacher screen and every closed-form table are fitted on a list of TRAIN query texts. M7's
list, `work/trainq_texts.json`, was dumped on 2026-08-26 before a later decontamination pass and
carries **4,582 R1 hits (1.31%)** against the current protected-query index. M7 disclosed this and
did not repair it, on the correct reasoning that its probes' RANKING was unaffected because every
candidate shared the identical fit set. M8 does not inherit that excuse: its filter covers
partitions M7's never screened against -- the reserved four, the shadow, and 55,120 M9-reserve
queries -- so "the same contaminated list for everyone" is no longer the whole story, and any
absolute number quoted from a fit on the old list is not clean.

This derives the list from the CURRENT kept pairs, screens it against the CURRENT filter, and
writes an M8-owned artifact. It deliberately does **not** overwrite `work/trainq_texts.json` or
`results/m7_trainq_manifest.json`: those are M7's provenance pins for a frozen, released system,
and G3 forbids M8 from editing M7's record. The two lists coexist and their difference is the
measurement.

What it reports, because a filter without its counts is unauditable (§3.4): the survivor count,
the hits by rule and by protected partition group, and a sha256 pin so a later probe can verify it
is fitting on the list this file produced.
"""
import argparse
import hashlib
import json
import sys
import time

import numpy as np

import m8base
import paths_guard

paths_guard.claim("m8src.protected_filter",
                  note="fit-list regeneration screens TRAIN queries against protected query text")

import decontam                                       # noqa: E402

REPO = m8base.REPO
OUT = m8base.WORK / "m8_trainq_texts.json"
MANIFEST = m8base.RESULTS / "m8_trainq_manifest.json"


def derive():
    """The TRAIN query text list, from the current kept pairs. Same derivation M7 used, so the two
    lists are comparable; the pool index is needed because a pair's query text is resolved through
    it."""
    import pool as poolmod
    import train
    from train import Cfg
    t0 = time.time()
    index, _, _ = poolmod.build()
    q_texts, *_ = train.build_arrays(Cfg(), index)
    print(f"derived {len(q_texts):,} TRAIN query texts ({time.time()-t0:.0f}s)", flush=True)
    return list(q_texts)


def screen(q_texts):
    """R1 against the CURRENT protected-query index -- the one covering six + dev + reserved-4 +
    M9-reserve."""
    q_ex, q_gram, q_whole, counts = decontam.protected_query_index()
    print(f"protected-query index: {sum(counts.values()):,} queries {counts}", flush=True)
    t0 = time.time()
    kinds, hits = {"exact": 0, "near": 0, "contains": 0}, []
    for i, q in enumerate(q_texts):
        k = decontam.query_hits(q, q_ex, q_gram, q_whole)
        if k:
            kinds[k] += 1
            hits.append(i)
        if (i + 1) % 100_000 == 0:
            el = time.time() - t0
            print(f"  screened {i+1:,}/{len(q_texts):,} ({el:.0f}s, {(i+1)/el:,.0f}/s)", flush=True)
    return kinds, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="smoke only")
    a = ap.parse_args()

    q_texts = derive()
    if a.limit:
        q_texts = q_texts[:a.limit]
    kinds, hits = screen(q_texts)
    keep = [q for i, q in enumerate(q_texts) if i not in set(hits)]

    old_n, old_hits = None, None
    old = m8base.WORK / "trainq_texts.json"
    if old.exists():
        try:
            old_list = json.loads(old.read_text())
            old_n = len(old_list) if isinstance(old_list, list) else None
        except (json.JSONDecodeError, MemoryError):
            pass

    payload = json.dumps(keep)
    if not a.limit:
        OUT.write_text(payload)
    sha = hashlib.sha256(payload.encode()).hexdigest()

    meta = {
        "_what": "the M8 closed-form fit list: TRAIN query texts, screened through the CURRENT "
                 "protected-query filter (six + dev + reserved-4 + M9-reserve).",
        "_why_not_in_place": "work/trainq_texts.json and results/m7_trainq_manifest.json are M7's "
                             "provenance pins for a frozen, released system. G3 forbids M8 from "
                             "editing M7's record, so the two lists coexist and their difference "
                             "is the measurement.",
        "n_derived": len(q_texts), "n_kept": len(keep), "n_removed": len(hits),
        "removal_rate": len(hits) / max(len(q_texts), 1),
        "hits_by_kind": kinds,
        "m7_list_n": old_n,
        "m7_list_known_r1_hits": 4582,
        "m7_list_known_r1_rate": 0.0131,
        "relpath": str(OUT.relative_to(REPO)), "sha256": sha, "bytes": len(payload),
        "produced_by": "m8src/fitlist.py",
        "smoke": bool(a.limit),
    }
    if not a.limit:
        MANIFEST.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"\n{'(smoke, nothing written)' if a.limit else 'wrote ' + str(OUT)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
