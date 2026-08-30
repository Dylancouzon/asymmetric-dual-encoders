"""Screen M9's candidate query-text sources through M8's EXTENDED protected-query filter.

Why this module exists. Three sources of query-shaped text were built and decontaminated during
M7 — 923,590 pseudo-queries, 85,871 `nqopen` questions, 134,761 `triviaqa` questions — but only
against M7's protected set. M8 extended that set with the reserved four, the LoTTE shadow and the
M9-reserve, and the extended screen needs a structure (`decontam.short_whole_index`) that the
persisted fingerprint npz does not carry. So M9.0 excluded all three, and the 7-day build would
have trained on 242,786 query texts seen ~870 times each.

What it does, and why it needs NO new capability. Dylan authorised a G2 allowlist entry for this
module; it turned out to be unnecessary, and the guard is what showed that. `m8src/protected_filter`
claims its entry **at import time**, so importing it both grants and performs the one protected
read — building the extended protected-query index — inside the module that already holds that
contact class. This module then handles only what that returns: an exact-hash set, a gram array and
a short-whole index. Every candidate source it screens (`work/pseudoq/`, `work/train/querytext/`)
is ordinary training text on no protected path.

So the protection boundary is **unchanged from M8**, the allowlist addition was reverted, and
`m8src/paths_guard.py` is byte-identical to M8's. The output is hashes and kept-INDEX lists only —
never a label, a qrel, a document, or a line of protected text. Nothing leaves the process but
counts and hashes, which is why this is done here rather than by an external agent whose context
would hold the text.

    python m9src/extended_screen.py            # screen every registered candidate source
"""
import hashlib
import json
import time

import numpy as np

import m9base
from m9base import REPO, WORK, RESULTS

# Import order is the whole design: `protected_filter` claims its own G2 entry at import time and
# is the only module that opens a protected path. One entry per process, so this module must NOT
# claim anything itself -- doing so made the very next import fail, correctly.
import protected_filter    # noqa: E402
import decontam            # noqa: E402

OUT = WORK / "decontam"
ARTIFACT = RESULTS / "m9_extended_screen.json"

# Every candidate, and how to get its texts. Each is already decontaminated against M7's set;
# this pass re-screens against M8's extended one.
SOURCES = {
    # NB what these actually are: M7 built them as short spans drawn out of the TRAIN document
    # stores (product titles, Wikipedia fragments, capped at 32 words) and labelled them a
    # VOCABULARY mitigation supplying "neither in-domain documents nor relevance structure".
    # They are therefore more DOCUMENT text, not query-shaped text, and M9 weights them as such.
    "pseudoq": {"path": "work/pseudoq/pseudoq-2000000-0.json",
                "kept": "work/pseudoq/kept-pseudoq-2000000-0.json",
                "what": "M7 vocabulary-coverage spans drawn from the decontaminated document "
                        "stores, <=32 words. NOT questions; treated as short document text"},
    "nqopen": {"path": "work/train/querytext/nqopen.json",
               "kept": "work/decontam/kept_querytext.json", "kept_key": "nqopen",
               "what": "google-research-datasets/nq_open train questions -- REAL queries, the "
                       "scarce resource for a query tower"},
    "triviaqa": {"path": "work/train/querytext/triviaqa.json",
                 "kept": "work/decontam/kept_querytext.json", "kept_key": "triviaqa",
                 "what": "mandarjoshi/trivia_qa rc.nocontext train questions -- REAL queries"},
}


def _load_source(name):
    """-> (texts already surviving M7's screen, their indices into the raw file)."""
    spec = SOURCES[name]
    raw = json.loads((REPO / spec["path"]).read_text())
    if isinstance(raw, dict):
        raw = raw.get("texts") or raw.get("queries") or list(raw.values())[0]
    keep = json.loads((REPO / spec["kept"]).read_text())
    idx = keep[spec["kept_key"]] if "kept_key" in spec else keep
    if isinstance(idx, dict):
        idx = idx.get(name) or list(idx.values())[0]
    idx = [int(i) for i in idx]
    return [str(raw[i]) for i in idx], idx


def _sha_texts(texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode())
        h.update(b"\x00")
    return h.hexdigest()


def main():
    t0 = time.time()
    # THE LoTTE SLICES COME FROM THE REMEDY RECORD, NOT THE FIRST SCREEN.
    # `m8_lotte_overlap.json` records the FIRST screen, whose verdict was "E10 REOPENS WITH DYLAN"
    # and whose `kept` list is EMPTY -- no slice passed before remediation. The slices that exist
    # are in `m8_lotte_remedy.json`: "PROCEED: all 7 surviving slices pass remedy + re-screen,
    # 14,034 queries". Reading the first file yields an empty LoTTE group and silently screens
    # nothing against M9's own fresh confirmation surface, which is precisely the surface new
    # training text must not leak into.
    rem = json.loads((RESULTS / "m8_lotte_remedy.json").read_text())
    lot = list(rem["surviving_slices"])
    assert lot and rem["verdict"].startswith("PROCEED"), rem["verdict"]
    print(f"LoTTE shadow: {len(lot)} surviving slices, {rem['total_surviving_queries']:,} queries "
          f"(dead: {rem['dead_slices_no_remedy_applies']})", flush=True)

    # The one protected read. Everything after this line is hashes.
    q_ex, q_gram, q_whole, counts, m9_detail = protected_filter.extended_index(lot)
    print(f"extended protected-query index: {sum(counts.values()):,} queries {counts} "
          f"({time.time()-t0:.0f}s)", flush=True)

    out = {"_what": "M9 candidate query-text sources re-screened through M8's EXTENDED "
                    "protected-query filter. Hashes and kept-index lists only; no text, no "
                    "labels, no documents leave this module.",
           "capability": "none added. m8src/protected_filter holds the only claim and performs the only "
                  "protected read, exactly as in M8; the G2 allowlist Dylan authorised was "
                  "reverted unused and m8src/paths_guard.py is byte-identical to M8's.",
           "screened_against": counts, "m9_reserve_fields": m9_detail,
           "lotte_source": "results/m8_lotte_remedy.json surviving_slices -- NOT "
                           "m8_lotte_overlap.json, whose `kept` is empty because its verdict was "
                           "'E10 REOPENS WITH DYLAN' before remediation",
           "n_protected_queries": sum(counts.values()),
           "method": {"ngram": decontam.NGRAM, "sketch": decontam.SKETCH,
                      "dup_share": decontam.DUP_SHARE, "short_ngram": decontam.SHORT_NGRAM,
                      "hash": "blake2b-64 word hashes, polynomial rolling n-gram, bottom-k sketch"},
           "sources": {}}

    for name in SOURCES:
        t1 = time.time()
        texts, raw_idx = _load_source(name)
        kinds, keep = {"exact": 0, "near": 0, "contains": 0}, []
        for i, q in enumerate(texts):
            k = decontam.query_hits(q, q_ex, q_gram, q_whole)
            if k:
                kinds[k] += 1
            else:
                keep.append(i)
            if (i + 1) % 200_000 == 0:
                el = time.time() - t1
                print(f"  {name} {i+1:,}/{len(texts):,} ({el:.0f}s, {(i+1)/el:.0f}/s)", flush=True)
        kept_rows = [raw_idx[i] for i in keep]
        dest = OUT / f"m9_kept_{name}.json"
        dest.write_text(json.dumps(kept_rows))
        out["sources"][name] = {
            **{k: v for k, v in SOURCES[name].items() if k in ("path", "what")},
            "n_in_after_m7_screen": len(texts), "n_kept": len(keep),
            "n_removed": len(texts) - len(keep), "hits_by_kind": kinds,
            "removal_rate": round((len(texts) - len(keep)) / max(len(texts), 1), 5),
            "kept_index_relpath": str(dest.relative_to(REPO)),
            "kept_index_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "kept_text_sha256": _sha_texts([texts[i] for i in keep]),
            "seconds": round(time.time() - t1, 1)}
        print(f"  {name}: {len(texts):,} in -> {len(keep):,} kept "
              f"({out['sources'][name]['removal_rate']:.4%} removed) {kinds}", flush=True)

    out["total_kept"] = sum(v["n_kept"] for v in out["sources"].values())
    out["seconds"] = round(time.time() - t0, 1)
    ARTIFACT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("sources", "m9_reserve_fields", "method")}, indent=1))
    print(f"TOTAL KEPT: {out['total_kept']:,} query-shaped texts")
    return out


if __name__ == "__main__":
    main()
