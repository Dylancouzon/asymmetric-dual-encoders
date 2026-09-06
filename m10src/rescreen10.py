"""Re-screen the M9 pools against the M10 protected index.

`instructions-m10.md`:462 — "The M9 real-query pool and the document pool are re-screened against
the COV additions (R1 removes matching queries; matching pool documents are removed too)."

The M9 pools were screened against M7's protected index. `protected10` adds the admitted COV
queries AND documents and the `arxiv-title` draw, so a text that was clean under M7 can be a COV
match now. This module applies R1 (`protected10.hits`: exact `blake2b-64`, any shared word-8-gram,
or a 4-7-word protected query contained verbatim) to both pools and caches the KEEP MASK.

The cache identity names `protected10._ident()` and the pool's own identity, so admitting a COV
component invalidates every mask rather than serving a stale one, and `corpus_loader` binds that
identity into the corpus manifest and the token-cache key — an unscreened pool cannot load.

Measured on this box (single core, 2026-09-05): 16,900 queries/s and 2,900 documents/s.

CLI:
    .venv/bin/python m10src/rescreen10.py --queries          # ~30 s
    .venv/bin/python m10src/rescreen10.py --documents        # ~45 min over 6.17M pool documents
"""
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import numpy as np

CACHE = REPO / "work" / "m10cov" / "rescreen10"
_IDX = {}


def protected_ident():
    import protected10
    return protected10._ident()


def _h(obj):
    return hashlib.blake2b(json.dumps(obj, sort_keys=True, default=str).encode(),
                           digest_size=10).hexdigest()


def texts_ident(texts):
    """A pool's identity: its length and the hash of its own strings."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return {"n": len(texts), "sha256": h.hexdigest()}


def index(verbose=False):
    if "idx" not in _IDX:
        import protected10
        _IDX["idx"] = protected10.build(verbose=verbose)
    return _IDX["idx"]


def _screen(texts, verbose=True, label="", log_every=100_000):
    """-> (keep mask, {hit kind: count}). R1, one text at a time, exactly `harvest.draw`'s
    query-side screen."""
    import protected10
    idx = index(verbose=verbose)
    keep = np.ones(len(texts), dtype=bool)
    kinds, t0 = {}, time.time()
    for i, t in enumerate(texts):
        h = protected10.hits(t, idx)
        if h:
            keep[i] = False
            kinds[h] = kinds.get(h, 0) + 1
        if verbose and log_every and i and i % log_every == 0:
            el = time.time() - t0
            print(f"  [{label}] {i:,}/{len(texts):,} ({i / max(el, 1e-9):,.0f}/s, "
                  f"{int((~keep[:i]).sum()):,} dropped)", flush=True)
    return keep, kinds


# ------------------------------------------------------------------------------- the queries ----

def query_keep_mask(texts, name, verbose=True, compute=True):
    """-> (bool mask, report). Cached on (protected10 identity, this pool's identity)."""
    ident = {"role": "query", "name": name, "protected10": protected_ident(),
             "pool": texts_ident(texts)}
    p = CACHE / f"q-{name}-{_h(ident)}.npz"
    if p.exists():
        z = np.load(p, allow_pickle=False)
        rep = json.loads(p.with_suffix(".json").read_text())
        return z["keep"].astype(bool), rep
    if not compute:
        raise SystemExit(
            f"the M9 query pool {name!r} has no M10 re-screen mask ({p}).\n"
            f"Run: .venv/bin/python m10src/rescreen10.py --queries")
    t0 = time.time()
    keep, kinds = _screen(texts, verbose=verbose, label=f"q:{name}")
    rep = {"ident": ident, "n": len(texts), "kept": int(keep.sum()),
           "removed": int((~keep).sum()), "by_hit": kinds,
           "seconds": round(time.time() - t0, 1)}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, keep=keep)
    p.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    return keep, rep


# ----------------------------------------------------------------------------- the documents ----

def doc_pool_ident():
    """The frozen stella document pool, named by what determines which rows exist.

    `m9base` FIRST: it pins `M7_ENCODER=stella-400M-v5`, and `pool.build()` returns a different
    meta (and a different pool) without it."""
    import m9base                                   # noqa: F401 -- imported for the pin
    import pool as poolmod
    _i, _v, pmeta = poolmod.build()
    banned = np.load(REPO / "work" / "decontam" / "banned_pool_rows.npy")
    return {"encoder": pmeta.get("encoder"), "n": int(pmeta["n"]),
            "id_sha256": dict(sorted(pmeta.get("id_sha256", {}).items())),
            "spans": {k: [int(a), int(b)] for k, (a, b) in sorted(pmeta["spans"].items())},
            "banned_sha256": hashlib.sha256(np.ascontiguousarray(banned).tobytes()).hexdigest()}


def _doc_path():
    ident = {"role": "document", "protected10": protected_ident(), "pool": doc_pool_ident()}
    return CACHE / f"d-{_h(ident)}.npz", ident


def doc_banned_rows(verbose=True, compute=False, limit_stores=None):
    """-> (sorted global pool row ids removed by the M10 re-screen, report).

    `compute=False` refuses rather than screening 6.17M documents by accident: the pass is ~45
    minutes and belongs to the CLI, not to a training launch.
    """
    p, ident = _doc_path()
    if p.exists():
        z = np.load(p, allow_pickle=False)
        return z["rows"].astype(np.int64), json.loads(p.with_suffix(".json").read_text())
    if not compute:
        raise SystemExit(
            f"the M9 document pool has no M10 re-screen mask ({p}).\n"
            f"Run: .venv/bin/python m10src/rescreen10.py --documents")
    import mix
    import pool as poolmod
    _i, _v, pmeta = poolmod.build()
    import data as m9data
    banned = set(np.load(REPO / "work" / "decontam" / "banned_pool_rows.npy").tolist())
    import protected10
    idx = index(verbose=verbose)
    dropped, per_store, t0 = [], {}, time.time()
    stores = limit_stores or sorted(pmeta["spans"])
    for store in stores:
        lo, hi = pmeta["spans"][store]
        if store == m9data.FEVER_STORE:
            per_store[store] = {"skipped": "fever-pos, excluded from the M9 doc pool"}
            continue
        t1 = time.time()
        _ids, texts = mix.load_store(store)
        n_elig, n_drop = 0, 0
        for j, t in enumerate(texts):
            row = lo + j
            if row in banned:
                continue
            n_elig += 1
            if protected10.hits(t, idx):
                dropped.append(row)
                n_drop += 1
        per_store[store] = {"eligible": n_elig, "dropped": n_drop,
                            "seconds": round(time.time() - t1, 1)}
        del texts
        if verbose:
            print(f"  {store}: {n_drop:,}/{n_elig:,} dropped "
                  f"({per_store[store]['seconds']:.0f}s, {time.time() - t0:.0f}s total)",
                  flush=True)
    rows = np.array(sorted(dropped), dtype=np.int64)
    rep = {"ident": ident, "n_dropped": int(rows.size), "per_store": per_store,
           "complete": limit_stores is None, "seconds": round(time.time() - t0, 1)}
    if limit_stores is not None:
        return rows, rep                       # a smoke never writes the real mask
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, rows=rows)
    p.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    return rows, rep


def doc_screen_ident():
    """What the manifest and the token-cache key bind so an unscreened pool cannot load."""
    p, ident = _doc_path()
    return {"protected10": _h(protected_ident()), "mask": p.name, "built": p.exists()}


REPORT_PATH = REPO / "results" / "m10_rescreen10.json"
# the segment names `corpus_loader._m9_segments` produces, in the order the query pass reports them
QUERY_SEGMENTS = ("m9-queries_pair", "m9-nqopen", "m9-triviaqa")


def merge_report(path, sections):
    """Load the existing report at `path` (or {} if it does not exist yet) and update it BY
    SECTION, then write it back. Re-running one pass alone (`--queries` XOR `--documents`) used to
    start from `out = {}` and overwrite the other pass's section entirely -- `main()` ran once for
    each and the second write erased the first (Codex 2026-09-05 finding: the 709-query section
    was gone from the shipped report). Returns the merged dict.
    """
    p = Path(path)
    out = json.loads(p.read_text()) if p.exists() else {}
    out.update(sections)
    p.write_text(json.dumps(out, indent=1, default=str))
    return out


def load_report(path=None):
    p = Path(path) if path else REPORT_PATH
    if not p.exists():
        raise SystemExit(f"{p}: no M10 re-screen report -- run "
                         f"rescreen10.py --queries --documents")
    return json.loads(p.read_text())


def validate(report, masks):
    """-> None, or SystemExit. Called by `corpus_loader.assemble_arm` before a training stream is
    handed to a launcher, so an incomplete or stale report (the merge bug above, or a re-screen
    run against a different protected10 identity) is caught here rather than trusted silently.

    `masks` is {segment name: the keep mask actually used, "documents": the banned-row array
    actually used, "protected10": the LIVE protected10 identity} -- both query and document
    sections must be present in `report`, each mask's length (or, for documents, its ban-row
    count) must match what the report recorded, and the report's own `protected10` identity must
    match the live one.
    """
    missing = [s for s in QUERY_SEGMENTS if s not in report]
    if "documents" not in report:
        missing.append("documents")
    if missing:
        raise SystemExit(f"the M10 re-screen report is missing sections: {missing} -- re-run "
                         f"rescreen10.py --queries --documents")
    live = masks.get("protected10")
    if live is not None and _h(report.get("protected10")) != _h(live):
        raise SystemExit("the M10 re-screen report's protected10 identity does not match the "
                         "live one -- re-run rescreen10.py --queries --documents")
    for seg in QUERY_SEGMENTS:
        n = report[seg]["n"]
        m = masks.get(seg)
        if m is None or len(m) != n:
            raise SystemExit(f"{seg}: mask length {0 if m is None else len(m)} != the report's "
                             f"pool size {n}")
    dm = masks.get("documents")
    n_dropped = report["documents"].get("n_dropped")
    if dm is not None and n_dropped is not None and len(dm) != n_dropped:
        raise SystemExit(f"documents: mask length {len(dm)} != the report's n_dropped "
                         f"{n_dropped}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", action="store_true")
    ap.add_argument("--documents", action="store_true")
    ap.add_argument("--stores", nargs="*", default=None, help="smoke: screen these stores only")
    a = ap.parse_args()
    # BEFORE anything imports `m9base`: `protected10.build` reads the reserved QUERY text through
    # `decontam.protected_query_index`, which `paths_guard` refuses once its guard is installed.
    # Same order `harvest.draw` relies on; the training paths never build the index at all
    # (`compute=False` reads the cached mask).
    index(verbose=True)
    sections = {}
    if a.queries:
        import corpus_loader as CL
        for seg in CL._m9_segments(screen=False):
            _keep, rep = query_keep_mask(seg.texts, seg.name)
            sections[seg.name] = {k: rep[k] for k in ("n", "kept", "removed", "by_hit", "seconds")}
            print(json.dumps({seg.name: sections[seg.name]}, indent=1), flush=True)
    if a.documents:
        _rows, rep = doc_banned_rows(compute=True, limit_stores=a.stores)
        sections["documents"] = {k: rep[k] for k in
                                 ("n_dropped", "per_store", "complete", "seconds")}
        print(json.dumps(sections["documents"], indent=1), flush=True)
    if a.queries or a.documents:
        sections["_what"] = ("the M9 pools re-screened against the M10 protected index "
                             "(instructions-m10.md:462)")
        sections["protected10"] = protected_ident()
        merge_report(REPORT_PATH, sections)


if __name__ == "__main__":
    main()
