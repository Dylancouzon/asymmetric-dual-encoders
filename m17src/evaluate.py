"""M17 evaluation: exact dense retrieval, per-domain nDCG@10/Recall@10, paired family
bootstrap, DBSF@100 through `m12src/qfusion.py`, and the held-out alias test.

Quality numbers come from EXACT dense retrieval over a declared corpus. ANN measurements
establish deployment behavior and must never be mixed into these numbers (CLAUDE.md).

Uncertainty is a paired bootstrap over QUERY FAMILIES, not over queries: two views of the same
intent and near-duplicate variants move together, so resampling them independently would
understate the interval.

The alias test reports top-10 overlap and rank correlation between the two views of each
held-out judged pair, plus each view's nDCG@10 where judged. It is descriptive: it appears in no
selection predicate (`registry.training.decision_protocol.panel_and_alias_test_role`).

**Surfaces are gated.** The default surface is `synthetic`. The pinned development suite and the
M17 panel each require their explicit flag AND an executable registry, and neither is to be run
during implementation. Nothing here can reach `results/frozen_eval/untouched-*`, the reserved
qrels caches, `work/m9reserve` or any six-set/LoTTE payload: those paths are refused by name.

The dev-suite reader below is wired (`--surface dev-suite --allow-dev-suite --out`, or
`allow_dev_suite=True` in-process). It serves the pinned M7/M8 suite EXACTLY as
`results/m7_dev_manifest.json:_pinned.components` lists it, through the M7 loaders, and aborts on
a missing or hash-mismatched component: the suite may never silently shrink. It needs the
EXECUTED lock half and an unread `v0_export`, refuses a dirty tree, verifies the bundle against
`lock.executed.v0_export`, binds the query pairs and the document-vector bytes, preflights every
component before the first score, writes only to the registry's canonical read path, and claims a
durable receipt there atomically. The production entry point (`dev_suite_read`) takes no manifest,
subset, loader or depth override: those live in the test-only `_dev_suite_read_fixture`.
`screen_read` is the same read for ONE step-6d screen arm (`--screen`): same surface, gate,
clean-tree rule and receipt discipline, but it binds the arm's own TRAINED bundle (refusing the
locked V0 export) and writes `work/m17/runs/<run_id>/screen.json`, never the V0 read's paths.
The panel reader is still unwritten.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from common import (REPO, RESULTS, WORK, admit_read, admit_write, registry, require_executable,
                    sha_file, sha_json, write_json)

FORBIDDEN = ("frozen_eval/untouched-", "m9reserve", "reserved_qrels", "lotte")


def _check_path(p):
    """Spelling check plus `common.admit_read`, which resolves symlinks before refusing."""
    s = str(p).replace("\\", "/").lower()
    for bad in FORBIDDEN:
        if bad in s:
            raise SystemExit(f"M17 EVAL REFUSED: {p} names a protected surface ({bad!r}). "
                             "M17 has no protected access (registry.protected_access).")
    return admit_read(p)


# ---- the pinned development suite -----------------------------------------------------------

DEV_MANIFEST = REPO / "results" / "m7_dev_manifest.json"
DEV_SUITE_REFUSAL = ("M17 EVAL REFUSED: surface 'dev-suite' needs its explicit --allow-dev-suite "
                     "flag (allow_dev_suite=True in-process); it is a registered read, not a "
                     "development convenience.")


def dev_manifest(path=None):
    """The pinned M7/M8 dev manifest. `path` is for fixtures only."""
    return json.loads(_check_path(Path(path) if path else DEV_MANIFEST).read_text())


def dev_components(manifest=None, path=None):
    """`_pinned.components`, AUTHORITATIVE and in manifest order (6 components).

    `dev_eval.dev_components()`'s rule: the suite may never shrink silently, so an unpinned
    manifest is an error here rather than a smaller suite."""
    man = dev_manifest(path) if manifest is None else manifest
    names = (man.get("_pinned") or {}).get("components")
    if not names:
        raise SystemExit(f"M17 EVAL REFUSED: {DEV_MANIFEST} has no _pinned.components; the "
                         "development suite is defined by that pinned list only.")
    return list(names)


V0_EXPORT_DIGESTS = ("model_npz_sha256", "tokenizer_sha256", "config_sha256")
LOCKED = "LOCKED_EXECUTABLE"


def _executed_v0_export(reg):
    """`lock.executed.v0_export` with its three digests present, or a refusal."""
    v0 = (((reg.get("lock") or {}).get("executed") or {}).get("v0_export")) or {}
    missing = [k for k in V0_EXPORT_DIGESTS if not v0.get(k)]
    if missing:
        raise SystemExit(f"M17 EVAL REFUSED: lock.executed.v0_export is missing {missing}. A "
                         "dev-suite read needs the EXECUTED lock half (m17src/lock.py --phase "
                         "executed): the identities that were exported are what the read binds.")
    return v0


def _require_dev_suite(allow_dev_suite, reg=None):
    """The gate: an explicit opt-in AND the EXECUTED lock half. No rehearsal bypass — a
    rehearsal is synthetic and may never be pointed at a development component (common.py).

    `EXECUTABLE` (the pre-clock half) admits preparation only; a development read is downstream
    of the V0 export, so it needs `LOCKED_EXECUTABLE` and the executed identities themselves
    (Astra dev-reader review P1: `{"status": "EXECUTABLE"}` used to pass).
    """
    if not allow_dev_suite:
        raise SystemExit(DEV_SUITE_REFUSAL)
    reg = reg or registry()
    status = require_executable(reg, rehearsal=False, what="a dev-suite read")
    if status != LOCKED:
        raise SystemExit(f"M17 EVAL REFUSED: registry status is {status!r}; a dev-suite read "
                         f"needs {LOCKED!r}, the executed lock half. {status!r} admits the "
                         "on-clock preparation only.")
    _executed_v0_export(reg)
    return status


def _verify_v0_bundle(bundle_dir, reg):
    """The bundle being read must BE the locked V0 export.

    `export.gate_artifact` recomputes the three digests with the exporter's own conventions
    (file bytes for model.npz/tokenizer.json, `sha_json` of the parsed config) and checks them
    against the bundle's provenance; this compares the same digests against
    `lock.executed.v0_export`, so a replaced bundle, tokenizer or config cannot be read.
    """
    import export
    want = _executed_v0_export(reg)
    got = export.gate_artifact(_check_path(bundle_dir))
    bad = [f"{k}: bundle {str(got.get(k))[:12]}, lock {str(want.get(k))[:12]}"
           for k in V0_EXPORT_DIGESTS if got.get(k) != want.get(k)]
    if bad:
        raise SystemExit(f"M17 EVAL REFUSED: {bundle_dir} is not the locked V0 export: "
                         + "; ".join(bad) + ". The registered read is of the artifact the lock "
                         "names, not of a bundle that happens to sit at that path.")
    return {k: got[k] for k in V0_EXPORT_DIGESTS}


def _check_identity(name, entry, got):
    """Every pinned field the loaded component can reproduce must match, or refuse by field."""
    bad = [f"{k}: loaded {v!r}, manifest {entry[k]!r}"
           for k, v in got.items() if k in entry and entry[k] != v]
    if bad:
        raise SystemExit(f"M17 EVAL REFUSED: pinned dev component {name} does not match "
                         f"{DEV_MANIFEST}: " + "; ".join(bad) + ". A dev component may not "
                         "change under a selection; restore it or re-pin deliberately.")


# Pinned fields a production read REQUIRES. `_check_identity` only compares the fields a
# manifest happens to carry, so a missing field used to disable its own check silently
# (Astra dev-reader review P2).
REQUIRED_FIELDS = {
    "text": ("n_docs", "n_queries", "corpus_ids_sha256", "corpus_text_sha256", "qids_sha256",
             "qrels_sha256"),
    "full-pool": ("json_sha256", "n_docs", "n_queries", "qids_ordered_sha256", "qids_sha256",
                  "qtexts_ordered_sha256", "qrels_sha256"),
}


def _require_pinned_fields(name, entry):
    kind = "full-pool" if entry.get("corpus") == "full-pool" else "text"
    missing = [f for f in REQUIRED_FIELDS[kind] if f not in entry]
    if missing:
        raise SystemExit(f"M17 EVAL REFUSED: pinned dev component {name} has no {missing} in "
                         f"{DEV_MANIFEST}; a missing pinned field is an unverifiable component, "
                         "not a skipped check.")


def load_dev_component(name, *, allow_dev_suite=False, reg=None, manifest=None,
                       manifest_path=None, with_doc_vecs=False, verify=True, strict=False):
    """One pinned component -> {doc_ids, doc_texts, q_ids, q_texts, qrels, doc_vecs, ...}.

    The four text-backed components come from `m7src/devsuite.load` (its `work/dev` cache) and
    their documents from the teacher encode cache. The two held-out slices are read from their
    pinned JSON directly and their corpus IS the frozen pool memmap, read through
    `prepare_data.PoolReader` — never `m7src/pool.build()`, which REBUILDS a 12.6 GiB artifact
    (m17/CODEMAP.md). `doc_texts` is None for them; they carry pool row indices, not text.
    """
    man = dev_manifest(manifest_path) if manifest is None else manifest
    names = dev_components(man)
    if name not in names:
        raise SystemExit(f"M17 EVAL REFUSED: {name!r} is not a pinned dev component {names}.")
    _require_dev_suite(allow_dev_suite, reg)
    entry = man[name]
    if strict:
        _require_pinned_fields(name, entry)
    comp = (_load_heldout if entry.get("corpus") == "full-pool" else _load_text_component)(
        name, entry, verify)
    comp["name"] = name
    comp["n_docs"] = len(comp["doc_ids"])
    comp["n_queries"] = len(comp["q_ids"])
    if set(comp["qrels"]) - set(comp["q_ids"]):
        raise SystemExit(f"M17 EVAL REFUSED: {name} has qrels for queries it does not serve.")
    if len(comp["q_ids"]) != len(comp["q_texts"]):
        raise SystemExit(f"M17 EVAL REFUSED: {name} has {len(comp['q_ids'])} qids and "
                         f"{len(comp['q_texts'])} query texts.")
    comp["queries"] = _query_identity(name, entry, comp)
    comp["doc_vecs"], comp["doc_vec_identity"] = (_dev_doc_vecs(comp, man) if with_doc_vecs
                                                  else (None, None))
    return comp


def _load_text_component(name, entry, verify):
    import devsuite                     # the M7 builder/cache: nq-250k, hotpotqa, the two cqadup
    from hashing import sha, sha_stream_list
    p = _check_path(devsuite.CACHE / f"{name}.json")
    if not p.exists():
        raise SystemExit(f"M17 EVAL REFUSED: pinned dev component {name} is missing ({p}). "
                         "Rebuild it with m7src/devsuite.py; the suite may not shrink silently.")
    doc_ids, doc_texts, q_ids, q_texts, qrels = devsuite.load(name)
    if verify:
        _check_identity(name, entry, {
            "n_docs": len(doc_ids), "n_queries": len(q_ids),
            "corpus_ids_sha256": sha_stream_list(doc_ids),
            "corpus_text_sha256": sha_stream_list(doc_texts),
            "qids_sha256": sha(sorted(q_ids)), "qrels_sha256": sha(qrels)})
    return {"doc_ids": doc_ids, "doc_texts": doc_texts, "q_ids": q_ids, "q_texts": q_texts,
            "qrels": qrels, "corpus": "text"}


def _load_heldout(name, entry, verify):
    # `heldout` is imported for its pinned path and the ONE shared pool-id list (both held-out
    # components address the same 6.17M rows; a per-component copy is ~400 MB of strings and
    # makes two callers unable to prove they share a corpus). `heldout.load` is NOT called: it
    # rebuilds from the training mix and verifies the pool through `pool.build()`.
    import heldout
    from hashing import sha, sha_stream_list
    p = _check_path(heldout.HELD / f"{name}.json")
    if not p.exists():
        raise SystemExit(f"M17 EVAL REFUSED: pinned dev component {name} is missing ({p}). "
                         "Rebuild it with m7src/heldout.py; the suite may not shrink silently.")
    b = json.loads(p.read_text())
    if verify:
        _check_identity(name, entry, {
            "json_sha256": sha_file(p), "n_docs": int(b["n_docs"]),
            "n_queries": len(b["q_ids"]), "qids_ordered_sha256": sha_stream_list(b["q_ids"]),
            "qids_sha256": sha(sorted(b["q_ids"])),
            "qtexts_ordered_sha256": sha_stream_list(b["q_texts"]),
            "qrels_sha256": sha(b["qrels"])})
    return {"doc_ids": heldout.pool_doc_ids(int(b["n_docs"])), "doc_texts": None,
            "q_ids": b["q_ids"], "q_texts": b["q_texts"], "qrels": b["qrels"],
            "corpus": "full-pool"}


# Ruling 2 (Dylan, 2026-09-12, m17/LEDGER.md): the four text-backed components' ordered query
# texts stay RECORDED-ONLY — the manifest is not amended to pin a digest to itself.
QUERY_TEXT_RECORDED_ONLY = "recorded_only (owner ruling 2026-09-12, LEDGER)"


def _query_identity(name, entry, comp):
    """The ordered (qid, text) pairs this read actually scores.

    Replacing or permuting `q_texts` while keeping the ids, counts, documents and qrels used to
    pass every check (Astra dev-reader review P1). The two held-out components pin the ordered
    qids AND ordered query texts, so they are VERIFIED here; the four text-backed components pin
    only `qids_sha256` over SORTED ids, so their pair digest is recorded in the receipt and the
    result, and the manifest is not edited to add one.
    """
    from hashing import sha_stream_list
    got = {"qids_ordered_sha256": sha_stream_list(comp["q_ids"]),
           "qtexts_ordered_sha256": sha_stream_list(comp["q_texts"])}
    _check_identity(name, entry, got)          # verifies whichever of the two the manifest pins
    verified = sorted(k for k in got if k in entry)
    return {**got,
            "query_pairs_sha256": sha_stream_list(f"{q}\x00{t}" for q, t
                                                  in zip(comp["q_ids"], comp["q_texts"])),
            "pinned_fields_verified": verified,
            "query_text_binding": ("verified" if "qtexts_ordered_sha256" in entry
                                   else QUERY_TEXT_RECORDED_ONLY),
            "_note": ("ordered query identity verified against the manifest" if verified else
                      "the manifest pins no ordered query identity for this component; the "
                      "digests above are RECORDED, not verified")}


_POOL_VERIFIED = {}


def _reset_pool_memo():
    """Every read ATTEMPT re-hashes the pool. The digest is shared by the two held-out
    components of one attempt only: an in-process retry after a same-size pool replacement used
    to be answered from the previous attempt's digest (Astra dev-reader re-check P1)."""
    _POOL_VERIFIED.clear()


def _pool_identity(man, n_docs):
    """`m7src/heldout._verify_pool`'s pinned-pool verification, without `pool.build()`.

    The held-out corpora ARE `_pinned.pool`'s vector file plus the store layout that indexes it,
    so the pool's own `meta.json` is compared field by field against the pinned block and the
    12.6 GiB file is checked by SIZE and by SHA-256 — "same size, different content" is exactly
    the failure the row-count check could not see. `pool.build()` is never called: it rebuilds
    the artifact (m17/CODEMAP.md). The digest is computed once per read and shared by both
    held-out components, which address the same rows.
    """
    import encoders
    import prepare_data
    pin = ((man.get("_pinned") or {}).get("pool")) or {}
    if not pin.get("vectors_sha256"):
        raise SystemExit(f"M17 EVAL REFUSED: {DEV_MANIFEST} pins no pool vector identity "
                         "(_pinned.pool.vectors_sha256); the held-out corpora cannot be bound.")
    active = encoders.active()                # M7's own check (`heldout._verify_pool`)
    if pin.get("encoder") != active.name:
        raise SystemExit(f"M17 EVAL REFUSED: the pinned pool was built with encoder "
                         f"{pin.get('encoder')!r} but the active encoder is {active.name!r}.")
    pool = prepare_data.PoolReader(pin["encoder"])
    for pin_key, meta_key in (("n", "n"), ("dim", "dim"), ("encoder", "encoder"),
                              ("encoder_revision", "encoder_revision"), ("stores", "stores"),
                              ("spans", "spans"), ("counts", "counts"),
                              ("store_id_sha256", "id_sha256")):
        if pin_key in pin and pool.meta.get(meta_key) != pin[pin_key]:
            raise SystemExit(f"M17 EVAL REFUSED: pinned pool identity changed: {meta_key} is "
                             f"{pool.meta.get(meta_key)!r}, {DEV_MANIFEST} says {pin[pin_key]!r}.")
    p = _check_path(prepare_data.POOL_DIR / pin["encoder"] / "vecs.f16")
    if p.stat().st_size != pin.get("vectors_bytes"):
        raise SystemExit(f"M17 EVAL REFUSED: {p} is {p.stat().st_size} bytes, the manifest pins "
                         f"{pin.get('vectors_bytes')}.")
    key = str(p)
    if key not in _POOL_VERIFIED:
        _POOL_VERIFIED[key] = sha_file(p)
    if _POOL_VERIFIED[key] != pin["vectors_sha256"]:
        raise SystemExit("M17 EVAL REFUSED: the pinned pool vectors changed: same size, "
                         "different content. Every held-out dev number is scored against these "
                         "vectors.")
    if len(pool.vecs) != n_docs:
        raise SystemExit(f"M17 EVAL REFUSED: the pool holds {len(pool.vecs)} rows but the "
                         f"component pins {n_docs}.")
    # The file that was HASHED must be the file that is SCORED, not a second path spelled the
    # same way (Sol dev-reader-fix review P2).
    scored = Path(getattr(pool.vecs, "filename", "") or "").resolve()
    if scored != p:
        raise SystemExit(f"M17 EVAL REFUSED: the held-out corpus actually memmapped is {scored}, "
                         f"not the hashed {p}.")
    return pool.vecs, {"source": "frozen pool memmap", "path": str(p),
                       "vectors_sha256": _POOL_VERIFIED[key],
                       "vectors_bytes": int(pin["vectors_bytes"]),
                       "verified_against": "results/m7_dev_manifest.json:_pinned.pool"}


# ---- Ruling 1: the disclosed trust-on-first-use teacher caches ------------------------------

TOFU_DISCLOSURE = REPO / "m17" / "tofu_disclosure.json"
# The two refusals `m7src/teacher.py` raises for trust-on-first-use bytes (shards, then the
# stitched combined.f16). Nothing else is accepted here: a changed shard, a bad stitch, a shard
# layout change or a missing shard is still a refusal.
TOFU_MARKS = ("predate hash recording", "predates hash recording")


def _is_tofu_refusal(exc):
    msg = str(getattr(exc, "code", exc) or "")
    return msg.startswith("ENCODE CACHE REFUSED:") and any(m in msg for m in TOFU_MARKS)


def _disclosed_tofu(exc, d, n_shards):
    """The dated owner disclosure for cache dir `d`, or a re-raise of `exc`.

    Ruling 1 (Dylan, 2026-09-12, m17/LEDGER.md) accepts the four stella dev caches whose shard
    and stitch digests were adopted from their own bytes, keyed to the spot-check recorded in
    `m17/tofu_disclosure.json`. The disclosure is an ANCHOR, not a new "trusted" digest: it
    carries the digests as `shards.json` recorded them, every shard of the cache must still
    equal its disclosed digest, and the bytes actually scored are re-hashed against the
    disclosure by the caller. An undisclosed cache, a disclosure that no longer matches the
    cache's own manifest, or any other refusal reason still refuses.
    """
    if not _is_tofu_refusal(exc) or not TOFU_DISCLOSURE.exists():
        raise exc
    disc = json.loads(_check_path(TOFU_DISCLOSURE).read_text())
    entry = (disc.get("caches") or {}).get(d.name)
    if not entry:
        raise exc
    man = json.loads((d / "shards.json").read_text())
    now = {f"shard_{sid}.npy": rec.get("sha256") for sid, rec in (man.get("shards") or {}).items()
           if int(sid) < n_shards}
    if entry.get("shards") != now:
        raise SystemExit(f"M17 EVAL REFUSED: {d} is disclosed in {TOFU_DISCLOSURE} but its shard "
                         "digests are no longer the disclosed ones. The disclosure covers the "
                         "bytes the spot-check examined, not whatever the cache holds now.")
    expected = (entry.get("combined") if n_shards > 1
                else entry["shards"]["shard_00000.npy"])
    if not expected:
        raise SystemExit(f"M17 EVAL REFUSED: {TOFU_DISCLOSURE} discloses no digest for the bytes "
                         f"{d} would actually score ({n_shards} shards).")
    return {"disclosure": str(TOFU_DISCLOSURE),
            "disclosure_sha256": sha_file(TOFU_DISCLOSURE),
            "ruling": disc.get("ruling"), "expected_sha256": expected,
            "spotcheck": disc.get("spotcheck")}


def _teacher_doc_vecs(comp, man=None):
    """The M7 teacher encode cache for a text-backed component: a cache HIT, never an encode.

    The bytes are hashed and compared against the cache's own recorded identity
    (`shards.json`, surfaced as `teacher.PROVENANCE[name]`), so a cache file that changed under
    us cannot be accepted on its row count alone.

    `verify=True` (Sol dev-reader-fix review P1): `encode_cached` REFUSES a shard or a stitched
    `combined.f16` that predates hash recording (trust-on-first-use) and re-hashes every
    pre-existing shard, so the vectors behind a registered number cannot be bytes that nothing
    ever checked. The ONE exception is a cache named in `m17/tofu_disclosure.json` under Ruling 1
    (Dylan, 2026-09-12): its disclosed digests must still equal the cache's own `shards.json`,
    and the bytes scored are re-hashed against the disclosed digest here.
    The teacher identity is pinned too: the cache key already binds model,
    revision, pooling, prefix and max_length, and the manifest's `_pinned.active_encoder` repo,
    revision and dimension are compared here.
    """
    import torch
    import teacher
    pin = (((man or {}).get("_pinned") or {}).get("active_encoder")) or {}
    if pin and (teacher.TEACHER, teacher.TEACHER_REV) != (pin.get("repo"), pin.get("revision")):
        raise SystemExit(f"M17 EVAL REFUSED: the teacher is {teacher.TEACHER}@"
                         f"{teacher.TEACHER_REV}, the manifest pins {pin.get('repo')}@"
                         f"{pin.get('revision')} (_pinned.active_encoder).")
    name, texts = f"dev-{comp['name']}-docs", comp["doc_texts"]
    key, _ = teacher.cache_key(name, "", 512, teacher.TEACHER, teacher.TEACHER_REV,
                               teacher.sha_texts(texts), torch.float16)
    d = _check_path(teacher.ENC / key)
    n_shards = (len(texts) + teacher.SHARD - 1) // teacher.SHARD
    absent = [s for s in range(n_shards) if not (d / f"shard_{s:05d}.npy").exists()]
    if absent:
        raise SystemExit(f"M17 EVAL REFUSED: the teacher encode cache {d} is missing "
                         f"{len(absent)} of {n_shards} shards for {comp['name']}. The registered "
                         "read consumes cached document vectors; it does not encode them.")
    try:
        vecs = teacher.encode_cached(name, texts, prefix="", dtype=torch.float16, verbose=False,
                                     verify=True)
        disclosed = None
    except SystemExit as exc:
        # ONLY a disclosed trust-on-first-use cache may continue; `_disclosed_tofu` re-raises
        # every other refusal, and the disclosed digests are checked against the bytes below.
        disclosed = _disclosed_tofu(exc, d, n_shards)
        vecs = teacher.encode_cached(name, texts, prefix="", dtype=torch.float16, verbose=False,
                                     verify=False)
    if int(vecs.shape[0]) != len(comp["doc_ids"]):
        raise SystemExit(f"M17 EVAL REFUSED: {comp['name']} has {len(comp['doc_ids'])} documents "
                         f"but its encode cache holds {vecs.shape[0]} rows.")
    if pin.get("dim") and int(vecs.shape[1]) != int(pin["dim"]):
        raise SystemExit(f"M17 EVAL REFUSED: {comp['name']}'s cached document vectors are "
                         f"{vecs.shape[1]}-dimensional, the manifest pins {pin['dim']}.")
    prov = teacher.PROVENANCE.get(name) or {}
    shards = prov.get("shard_sha256") or {}
    recorded = prov.get("combined_sha256") or (shards.get("00000") if n_shards == 1 else None)
    path = getattr(vecs, "filename", None)
    if not path:
        raise SystemExit(f"M17 EVAL REFUSED: {comp['name']}'s document vectors are not backed by "
                         "a file, so the bytes actually scored cannot be hashed.")
    digest = sha_file(path)
    if recorded and digest != recorded:
        raise SystemExit(f"M17 EVAL REFUSED: {path} hashes {digest[:12]} but the encode cache "
                         f"records {str(recorded)[:12]}. The cache is mutable and gitignored; a "
                         "registered number may not rest on bytes that changed under it.")
    if disclosed and digest != disclosed["expected_sha256"]:
        raise SystemExit(f"M17 EVAL REFUSED: {path} hashes {digest[:12]} but "
                         f"{TOFU_DISCLOSURE} discloses "
                         f"{str(disclosed['expected_sha256'])[:12]}. The owner ruling accepts "
                         "the bytes the spot-check examined, not bytes that changed since.")
    ident = {"source": "m7 teacher encode cache", "path": str(path),
             "vectors_sha256": digest, "cache_key": prov.get("cache_key"),
             "n_rows": int(vecs.shape[0]),
             "verified_against": ("teacher shards.json" if recorded else None),
             "_note": ("cache identity verified" if recorded else
                       "the cache records no digest for these bytes; RECORDED, not verified")}
    if disclosed:
        ident.update(tofu_disclosed=True, tofu_disclosure=disclosed["disclosure"],
                     tofu_disclosure_sha256=disclosed["disclosure_sha256"],
                     tofu_ruling=disclosed["ruling"], tofu_spotcheck=disclosed["spotcheck"],
                     verified_against=f"{disclosed['disclosure']} (owner ruling)",
                     _note="trust-on-first-use cache accepted under the dated owner disclosure; "
                           "the scored bytes were re-hashed against the disclosed digest")
    else:
        ident["tofu_disclosed"] = False
    return vecs, ident


def _dev_doc_vecs(comp, man):
    """Frozen stella document vectors plus the identity of the bytes actually scored."""
    if comp["corpus"] == "full-pool":
        return _pool_identity(man, len(comp["doc_ids"]))
    return _teacher_doc_vecs(comp, man)


# The registered destination of the ONE V0 read, and the production surface it must use.
V0_READ_PATH = RESULTS / "m17_v0_read.json"
PROD_LOADER = {"variant": "int8", "mode": "resident_int8"}


def _v0_read_path(reg):
    """The canonical result path of the ONE V0 read: whatever the registry names, else the
    constant. A production read may write nowhere else (Sol dev-reader-fix review P1)."""
    named = (((reg.get("training") or {}).get("untrained_vocab_export_v0")) or {}).get("read_path")
    if not named:
        named = (_executed_v0_export(reg)).get("read_path")
    if not named:
        return V0_READ_PATH
    p = Path(named)
    return p if p.is_absolute() else REPO / p


def _require_unread(reg):
    """`lock.executed.v0_export.read` must still be `false`; the read is registered once."""
    v0 = _executed_v0_export(reg)
    if v0.get("read") is not False:
        raise SystemExit(f"M17 EVAL REFUSED: lock.executed.v0_export.read is {v0.get('read')!r}, "
                         "not false. The V0 read is registered once "
                         "(training.untrained_vocab_export_v0.reads = 1).")
    return v0


def _git_sha():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):        # pragma: no cover - no git
        return None


def _git_porcelain():
    """Tracked-file changes, or a refusal if git cannot answer."""
    try:
        r = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - no git
        raise SystemExit(f"M17 EVAL REFUSED: git could not report the working tree ({exc}); the "
                         "read records the code and registry it ran against.")
    if r.returncode != 0:                                 # pragma: no cover - no git
        raise SystemExit(f"M17 EVAL REFUSED: git status failed ({r.stderr.strip()[:200]}).")
    return r.stdout.strip()


def _require_clean_tree():
    """A dirty tree makes `git_sha` a lie: uncommitted code, registry or manifest changes would
    shape the surface while the receipt named a clean commit (Sol dev-reader-fix review P1)."""
    dirty = _git_porcelain()
    if dirty:
        lines = dirty.splitlines()
        raise SystemExit("M17 EVAL REFUSED: the working tree has uncommitted tracked changes, so "
                         "the recorded git sha would not describe what ran: "
                         + "; ".join(lines[:5]) + (f" (+{len(lines) - 5} more)"
                                                   if len(lines) > 5 else "")
                         + ". Commit or stash them and read again.")
    return dirty


def _utc():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _receipt_path(out):
    return out.with_name(out.stem + ".receipt.json")


def _claim_receipt(receipt_p):
    """Claim the one read ATOMICALLY. `O_CREAT|O_EXCL` cannot be raced the way an existence
    test can: exactly one process creates the receipt (Sol dev-reader-fix review P1)."""
    receipt_p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (0, 1):
        try:
            fd = os.open(receipt_p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            # A ZERO-BYTE receipt is an abandoned claim: a refused or killed preflight created it
            # and never persisted a receipt, so it records no read and blocks every later attempt
            # (Astra dev-reader re-check P1). `O_EXCL` cannot truncate-and-continue, so the empty
            # file is removed and the claim retried exactly once.
            if attempt == 0 and receipt_p.exists() and receipt_p.stat().st_size == 0:
                receipt_p.unlink(missing_ok=True)
                continue
            return False
        os.close(fd)
        return True
    return False


# Everything a continuation must reproduce EXACTLY. Anything else is a different read.
IDENTITY_FIELDS = ("registry_status", "bundle", "bundle_digests", "loader", "retrieval_depth",
                   "dev_manifest_sha256", "components", "component_identities", "git_sha",
                   "git_porcelain", "fixture", "out")


def _prior_receipt(receipt_p):
    """The ONE recovery path: a receipt an interrupted attempt left in `started` or `failed`.

    Anything else — a complete receipt, an empty claim, unparseable JSON — is refused. The
    identities are compared after the preflight (`_check_resumable`), so a continuation scores
    the remaining components of the SAME read, never a second one.
    """
    try:
        prior = json.loads(receipt_p.read_text())
    except (OSError, ValueError):
        prior = None
    state = (prior or {}).get("state") if isinstance(prior, dict) else None
    if state not in ("started", "failed"):
        raise SystemExit(f"M17 EVAL REFUSED: {receipt_p} already exists (state {state!r}). The V0 "
                         "read is registered once (training.untrained_vocab_export_v0.reads = 1); "
                         "it is not overwritten, and there is no --force. Only a receipt left in "
                         "'started' or 'failed' by an interrupted attempt may be continued.")
    return prior


def _check_resumable(prior, receipt, receipt_p):
    """Continue only if every recorded identity matches this preflight; else refuse by field."""
    bad = [k for k in IDENTITY_FIELDS if prior.get(k) != receipt[k]]
    if bad:
        raise SystemExit(f"M17 EVAL REFUSED: {receipt_p} records an interrupted read whose "
                         f"{bad} differ from this preflight. A continuation must be the same "
                         "read: restore the identities it names, or the attempt is a new read "
                         "and needs its own registration.")
    return {k: v for k, v in (prior.get("metrics") or {}).items() if k in receipt["components"]}


def _ndcg_and_recall(run, qrels, cut=10):
    """nDCG@10 and Recall@10 from ONE `pytrec_eval` evaluator over ONE run.

    Both metrics must resolve a tie at the rank-`cut` boundary the same way; a Python stable sort
    beside pytrec_eval's own ordering could count a tied document for one metric and not the
    other (Sol dev-reader-fix review P2). The qrels are passed exactly as `m7src/evalkit`
    passes them, so this suite keeps M7's relevance threshold.
    """
    import pytrec_eval
    ev = pytrec_eval.RelevanceEvaluator({q: v for q, v in qrels.items() if q in run},
                                        {f"ndcg_cut.{cut}", f"recall.{cut}"})
    res = ev.evaluate(run)
    return ({q: s[f"ndcg_cut_{cut}"] for q, s in res.items()},
            {q: s[f"recall_{cut}"] for q, s in res.items()})


def _enforce_production_surface(reg, names, man, manifest, manifest_path, variant, mode, k):
    """The registered surface, not the caller's arguments (Astra dev-reader review P2).

    `training.decision_protocol.screen_routing_surface` is the COMPLETE pinned component list
    read as int8 folded artifacts through the released QueryTable path; the retrieval depth is
    the registered `serving.prefetch`. A subset, fp16/eager loading, another depth or another
    manifest are reachable only from `fixture=True`, which no production caller sets.
    """
    if manifest is not None or manifest_path is not None:
        raise SystemExit("M17 EVAL REFUSED: a production dev-suite read uses "
                         f"{DEV_MANIFEST}; an alternative manifest is a fixture-only argument.")
    pinned = dev_components(man)
    if list(names) != list(pinned):
        raise SystemExit(f"M17 EVAL REFUSED: the registered surface is the complete pinned list "
                         f"{pinned}; {list(names)} is a different surface, not a selection.")
    if {"variant": variant, "mode": mode} != PROD_LOADER:
        raise SystemExit(f"M17 EVAL REFUSED: the registered read is {PROD_LOADER}, not "
                         f"{ {'variant': variant, 'mode': mode} }.")
    depth = int(reg["serving"]["prefetch"])
    if k is not None and int(k) != depth:
        raise SystemExit(f"M17 EVAL REFUSED: the registered retrieval depth is {depth} "
                         f"(serving.prefetch), not {k}.")
    return depth


def dev_suite_read(bundle_dir, out=None, *, allow_dev_suite=False):
    """THE production entry point for the ONE registered V0 dev-suite read.

    It takes no manifest, subset, loader, depth or fixture override: the registry is loaded from
    disk here, the destination is the registry's own canonical path, and the surface is the
    registered one. Everything else lives in `_dev_suite_read_fixture`, which the CLI cannot
    reach (Sol dev-reader-fix review P1).
    """
    if not allow_dev_suite:
        raise SystemExit(DEV_SUITE_REFUSAL)
    reg = registry()
    dest = _v0_read_path(reg)
    if not out:
        raise SystemExit("M17 EVAL REFUSED: a dev-suite read needs its output path (the "
                         f"registered destination is {dest}); a read whose numbers are "
                         "not written is a spent read with no evidence.")
    if Path(out).expanduser().absolute().resolve() != dest.resolve():
        raise SystemExit(f"M17 EVAL REFUSED: the registered destination of the V0 read is {dest}; "
                         f"{out} is another path, and a fresh path is another read.")
    _require_unread(reg)
    _require_clean_tree()
    return _dev_suite_read(bundle_dir, dest, reg=reg, fixture=False)


SCREEN_READ_NAME = "screen.json"
SCREEN_RUNS_DIR = WORK / "runs"


def _verify_screen_bundle(bundle_dir, reg):
    """A screen arm reads a TRAINED bundle, so it cannot be bound to `lock.executed.v0_export`.

    The digests are still recomputed with the exporter's own conventions and recorded, and the
    bundle is refused if it IS the locked V0 export: V0 has its own single registered read
    (`training.untrained_vocab_export_v0.reads = 1`) and must not be re-read as a screen arm.
    """
    import export
    got = export.gate_artifact(_check_path(bundle_dir))
    want = _executed_v0_export(reg)
    if all(got.get(k) == want.get(k) for k in V0_EXPORT_DIGESTS):
        raise SystemExit(f"M17 EVAL REFUSED: {bundle_dir} IS the locked V0 export, not a trained "
                         "screen arm. The V0 read is registered once and already has its own "
                         "entry point; a screen read is of an arm's own exported table.")
    return {k: got[k] for k in V0_EXPORT_DIGESTS}


def _screen_read_dest(out):
    """`work/m17/runs/<run>/screen.json` — the arm's own run directory, never the V0 read's.

    The screen read is the registered ONE quality read of a screen arm
    (`training.quality_reads_per_screen_arm = 1`); like the V0 read it refuses to overwrite its
    result or its receipt, so each arm's read lives at its own registered path.
    """
    if not out:
        raise SystemExit("M17 EVAL REFUSED: a screen read needs its output path "
                         f"({SCREEN_RUNS_DIR}/<run_id>/{SCREEN_READ_NAME}); a read whose numbers "
                         "are not written is a spent read with no evidence.")
    p = Path(out).expanduser().absolute()
    ok = (p.name == SCREEN_READ_NAME and p.parent.parent == SCREEN_RUNS_DIR
          and p.parent.name not in ("", ".", ".."))
    if not ok:
        raise SystemExit(f"M17 EVAL REFUSED: a screen read writes to "
                         f"{SCREEN_RUNS_DIR}/<run_id>/{SCREEN_READ_NAME}; {out} is another "
                         "destination. The screen arms never write the V0 read's paths.")
    return p


def screen_read(bundle_dir, out=None, *, allow_dev_suite=False):
    """THE production entry point for ONE screen arm's registered dev-suite read.

    Same surface, same gate and the same provenance/receipt discipline as `dev_suite_read`: the
    registry is loaded here, the complete pinned `screen_routing_surface` is read as int8 folded
    rows through the released QueryTable path, the tree must be clean and the receipt is claimed
    atomically. It differs in exactly two ways, both forced by what it reads: the bundle is the
    arm's own trained export rather than the locked V0 one, and the destination is that arm's run
    directory rather than the V0 read's canonical path. `v0_export.read` is untouched.
    """
    if not allow_dev_suite:
        raise SystemExit(DEV_SUITE_REFUSAL)
    reg = registry()
    dest = _screen_read_dest(out)
    if dest.resolve() == _v0_read_path(reg).resolve():       # defensive; the shape already differs
        raise SystemExit("M17 EVAL REFUSED: a screen read may not write the V0 read's path.")
    _require_clean_tree()
    return _dev_suite_read(bundle_dir, dest, reg=reg, fixture=False,
                           verify_bundle=_verify_screen_bundle)


def _dev_suite_read_fixture(bundle_dir, out=None, *, allow_dev_suite=False, reg=None, names=None,
                            manifest=None, manifest_path=None, variant="int8",
                            mode="resident_int8", k=None, verify_bundle=None):
    """TEST-ONLY: the same reader with the surface overridable. Never called in production."""
    return _dev_suite_read(bundle_dir, out, allow_dev_suite=allow_dev_suite, reg=reg, names=names,
                           manifest=manifest, manifest_path=manifest_path, variant=variant,
                           mode=mode, k=k, fixture=True, verify_bundle=verify_bundle)


def _dev_suite_read(bundle_dir, out=None, *, allow_dev_suite=True, reg=None, names=None,
                    manifest=None, manifest_path=None, variant="int8", mode="resident_int8",
                    k=None, fixture=False, verify_bundle=None):
    """ONE registered dev-suite read of ONE exported bundle: per-component nDCG@10 and
    Recall@10 and the equal-weight component macro of each (`_pinned.macro`).

    Queries go through the released QueryTable path as int8 folded rows (`loader_np`), the
    registered `screen_routing_surface`. Retrieval is exact dense over each component's declared
    corpus, through the same M7 scorer (`evalkit`) every pinned dev number was computed with;
    both metrics come from that ONE retrieval run.

    The read is single-use and provenanced. `out` is mandatory; the read refuses if `out` exists,
    and the receipt beside it is claimed with `O_CREAT|O_EXCL` so two processes cannot both start
    it. Every identity — registry status, the locked V0 digests, the per-component manifest
    hashes, the ordered query digests and the document-vector digests — is established in a
    preflight over all components before any scoring begins, and each component's per-query
    metrics are persisted into the receipt (tmp + `os.replace`) as it completes. A receipt left
    in `started` or `failed` by an interrupted attempt is CONTINUED — the remaining components
    only — when every recorded identity matches this preflight, and refused otherwise.
    """
    _reset_pool_memo()       # a pool digest is shared WITHIN one attempt only (re-check P1)
    status = _require_dev_suite(allow_dev_suite, reg)
    reg = reg or registry()
    man = dev_manifest(manifest_path) if manifest is None else manifest
    names = list(names or dev_components(man))
    if fixture:
        depth = int(k if k is not None else reg["serving"]["prefetch"])
    else:
        depth = _enforce_production_surface(reg, names, man, manifest, manifest_path,
                                            variant, mode, k)
    if not out:
        raise SystemExit("M17 EVAL REFUSED: a dev-suite read needs its output path (the "
                         f"registered destination is {V0_READ_PATH}); a read whose numbers are "
                         "not written is a spent read with no evidence.")
    out = Path(admit_write(out))
    receipt_p = _receipt_path(out)
    if out.exists():
        raise SystemExit(f"M17 EVAL REFUSED: {out} already exists. The V0 read is registered "
                         "once (training.untrained_vocab_export_v0.reads = 1); it is not "
                         "overwritten, and there is no --force.")
    claimed = _claim_receipt(receipt_p)
    prior = None if claimed else _prior_receipt(receipt_p)

    # ---- preflight: everything that can refuse, before the first score ----
    try:
        from evalkit import macro, topk_ids_scores
        from loader_np import M17QueryEncoder
        bundle_digests = (verify_bundle or _verify_v0_bundle)(bundle_dir, reg)
        enc = M17QueryEncoder(_check_path(bundle_dir), variant=variant, mode=mode)
        comps = []
        for name in names:
            c = load_dev_component(name, allow_dev_suite=True, reg=reg, manifest=man,
                                   with_doc_vecs=True, strict=not fixture)
            c["doc_texts"] = None        # the vectors are loaded; the texts are not scored
            comps.append(c)
        components = {c["name"]: {"manifest_entry_sha256": sha_json(man[c["name"]]),
                                  "manifest_hashes": {kk: vv for kk, vv in man[c["name"]].items()
                                                      if kk.endswith("_sha256")},
                                  "n_docs": c["n_docs"], "n_queries": c["n_queries"],
                                  "queries": c["queries"],
                                  "document_vectors": c["doc_vec_identity"]} for c in comps}
        receipt = {"_schema": "m17-dev-suite-read-receipt-v1", "state": "started", "reads": 1,
                   "surface": "m7/m8 pinned development suite", "components": names,
                   "registry_status": status, "bundle": str(bundle_dir),
                   "bundle_digests": bundle_digests,
                   "loader": {"variant": variant, "mode": mode}, "retrieval_depth": depth,
                   "dev_manifest_sha256": (None if fixture else sha_file(DEV_MANIFEST)),
                   "component_identities": components, "fixture": bool(fixture),
                   "git_sha": _git_sha(), "git_porcelain": ("" if fixture else _git_porcelain()),
                   "started_utc": _utc(), "out": str(out),
                   "completed_components": [], "metrics": {}}
        if prior is not None:
            done = _check_resumable(prior, receipt, receipt_p)
            receipt.update(metrics=done, completed_components=[n for n in names if n in done],
                           started_utc=prior.get("started_utc") or receipt["started_utc"],
                           resumed_utc=_utc(), resumed_from=prior.get("state"))
        write_json(receipt_p, receipt)          # the first persistence; the claim is now a read
    except BaseException:
        # An empty claim is not a spent read: release it so the refusal can be fixed and the
        # read attempted again. EVERY refusal between the claim and that first persistence is
        # covered — the git-status and resume checks included (Astra dev-reader re-check P1).
        if claimed and receipt_p.exists() and receipt_p.stat().st_size == 0:
            receipt_p.unlink(missing_ok=True)
        raise

    # ---- scoring: one retrieval run per component, both metrics from it ----
    try:
        for c in comps:
            if c["name"] in receipt["metrics"]:
                c["doc_vecs"] = None                 # already scored by the interrupted attempt
                continue
            run = topk_ids_scores(enc.encode(c["q_texts"]), c["doc_vecs"], c["doc_ids"],
                                  k=depth, qids=c["q_ids"])
            nd, rc = _ndcg_and_recall(run, c["qrels"], cut=10)
            c["doc_vecs"] = None
            receipt["metrics"][c["name"]] = {"ndcg@10": nd, "recall@10": rc}
            receipt["completed_components"].append(c["name"])
            write_json(receipt_p, receipt)           # persisted before the next component
    except BaseException as exc:                     # a spent attempt still leaves its receipt
        receipt.update(state="failed", failed_utc=_utc(), error=f"{type(exc).__name__}: {exc}")
        write_json(receipt_p, receipt)
        raise
    nd_all = {n: receipt["metrics"][n]["ndcg@10"] for n in names}
    rc_all = {n: receipt["metrics"][n]["recall@10"] for n in names}
    nd_macro, nd_means = macro(nd_all)
    rc_macro, rc_means = macro(rc_all)
    rep = {"surface": "m7/m8 pinned development suite",
           "components": names, "registry_status": status,
           "bundle": str(bundle_dir), "bundle_digests": bundle_digests,
           "loader": {"variant": variant, "mode": mode}, "retrieval_depth": depth,
           "ndcg@10": {"macro": nd_macro, "per_component": nd_means,
                       "n_queries": {n: len(v) for n, v in nd_all.items()}},
           "recall@10": {"macro": rc_macro, "per_component": rc_means},
           "per_query_ndcg@10": nd_all,
           "reads": 1, "git_sha": receipt["git_sha"],
           "git_porcelain": receipt["git_porcelain"], "read_utc": receipt["started_utc"],
           "dev_manifest_sha256": receipt["dev_manifest_sha256"],
           "component_identities": components, "receipt": str(receipt_p),
           "_macro": (man.get("_pinned") or {}).get("macro"),
           "_note": "exact dense retrieval; ANN measurements are never mixed in"}
    write_json(out, rep)
    receipt.update(state="complete", completed_utc=_utc(),
                   ndcg_at_10={"macro": nd_macro, "per_component": nd_means},
                   recall_at_10={"macro": rc_macro, "per_component": rc_means})
    write_json(receipt_p, receipt)
    return rep


# ---- exact dense retrieval -----------------------------------------------------------------

DOC_BLOCK = 65536       # 4096 queries x 65536 documents of float32 is 1 GiB per score block


def search(query_vecs, doc_vecs, k=10, doc_ids=None, block=4096, doc_block=DOC_BLOCK,
           query_ids=None):
    """Exact inner product over L2-normalized vectors. Returns a run dict {qkey: {docid: score}}.

    Blocked over queries AND over documents: a 4096 x 5.2M score matrix is 85 GB, so the corpus
    is walked in `doc_block` slices and each slice's top-k is merged. The merge is exact — each
    block keeps its WHOLE cutoff tie (every document scoring at least the block's k-th largest
    score) and then selects by (-score, ascending document id), the same rule the global merge
    and the cache builder use. Taking only `argpartition`'s arbitrary k would discard smaller
    document ids from a tie wider than k and make the answer depend on the block size.
    Document ids must therefore be unique.

    `query_ids` keys the run by the caller's own query ids instead of positional integers; the
    panel's qrels/domains/families are keyed that way (`results/m17_panel_manifest.json`
    reader_contract).
    """
    q = np.asarray(query_vecs, dtype=np.float32)
    d = np.asarray(doc_vecs, dtype=np.float32)
    ids = list(doc_ids) if doc_ids is not None else list(range(d.shape[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("doc_ids contains duplicates; the ascending-id tie rule needs one key "
                         "per document, and a duplicate key would silently drop a document")
    if query_ids is not None and len(query_ids) != q.shape[0]:
        raise ValueError(f"{len(query_ids)} query_ids for {q.shape[0]} query vectors")
    kk = min(k, d.shape[0])
    run = {}
    for lo in range(0, q.shape[0], block):
        qb = q[lo:lo + block]
        cand = [[] for _ in range(qb.shape[0])]
        for dlo in range(0, d.shape[0], doc_block):
            s = qb @ d[dlo:dlo + doc_block].T
            m = min(kk, s.shape[1])
            if m <= 0:
                continue
            top = np.argpartition(-s, m - 1, axis=1)[:, :m]
            for r in range(s.shape[0]):
                cut = float(s[r, top[r]].min())          # the block's k-th largest score
                tied = np.flatnonzero(s[r] >= cut)       # the whole cutoff tie, not a subset
                block = sorted(((float(s[r, j]), ids[dlo + int(j)]) for j in tied),
                               key=lambda t: (-t[0], t[1]))
                cand[r].extend(block[:m])
        for r in range(qb.shape[0]):
            key = query_ids[lo + r] if query_ids is not None else lo + r
            best = sorted(cand[r], key=lambda t: (-t[0], t[1]))[:kk]
            run[key] = {doc: sc for sc, doc in best}
    return run


def ndcg_at_k(run, qrels, k=10):
    """Binary-or-graded nDCG@10 with the standard log2 discount, per query."""
    out = {}
    for qi, docs in run.items():
        rel = qrels.get(qi, {})
        ranked = sorted(docs.items(), key=lambda kv: -kv[1])[:k]
        dcg = sum(float(rel.get(d, 0)) / np.log2(i + 2) for i, (d, _) in enumerate(ranked))
        ideal = sorted((float(v) for v in rel.values()), reverse=True)[:k]
        idcg = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))
        out[qi] = float(dcg / idcg) if idcg > 0 else 0.0
    return out


def recall_at_k(run, qrels, k=10):
    out = {}
    for qi, docs in run.items():
        rel = {d for d, v in qrels.get(qi, {}).items() if float(v) > 0}
        if not rel:
            out[qi] = 0.0
            continue
        ranked = [d for d, _ in sorted(docs.items(), key=lambda kv: -kv[1])[:k]]
        out[qi] = len(rel & set(ranked)) / len(rel)
    return out


def per_domain(values, domains):
    """Mean per domain plus the macro over domains. `domains` maps query key -> domain."""
    by = {}
    for qi, v in values.items():
        by.setdefault(domains.get(qi, "general"), []).append(v)
    means = {d: float(np.mean(v)) for d, v in sorted(by.items())}
    return {"per_domain": means, "macro": float(np.mean(list(means.values()))) if means else 0.0,
            "n_per_domain": {d: len(v) for d, v in sorted(by.items())}}


# ---- paired family bootstrap ------------------------------------------------------------

def paired_family_bootstrap(a, b, families, replicates=10000, confidence=0.95, seed=0,
                            domains=None):
    """Interval on the reported statistic's difference, resampling FAMILIES with replacement.

    With `domains`, the statistic is the EQUAL-WEIGHT DOMAIN MACRO — the same quantity
    `per_domain()` reports — recomputed inside every replicate, not the query mean. The two
    differ badly whenever domains have unequal sizes: 100 queries improving by 1.0 in one domain
    and one query worsening by 1.0 in another is a macro difference of 0 and a query mean of
    0.98. The query-weighted estimate is kept beside it, labelled.

    `domains` also stratifies the resample, so a domain does not vanish from a replicate.
    A wide interval here is a statement about this panel's size, not evidence of equivalence.
    """
    if set(a) != set(b):
        miss, extra = sorted(set(a) - set(b))[:3], sorted(set(b) - set(a))[:3]
        raise ValueError(f"baseline keys do not match the candidate's: missing {miss}, "
                         f"unexpected {extra}. Intersecting would silently drop queries from "
                         "one side and report the paired interval as if they were compared.")
    keys = sorted(a)
    if not keys:
        return {"delta": 0.0, "ci": [None, None], "n_families": 0}
    delta = np.asarray([a[k] - b[k] for k in keys], dtype=np.float64)
    if domains:
        dom_names = sorted({domains.get(k, "general") for k in keys})
        dom_ix = {d: i for i, d in enumerate(dom_names)}
        dom_of = np.asarray([dom_ix[domains.get(k, "general")] for k in keys], dtype=np.int64)

        def statistic(idx):
            sums = np.bincount(dom_of[idx], weights=delta[idx], minlength=len(dom_names))
            cnts = np.bincount(dom_of[idx], minlength=len(dom_names))
            hit = cnts > 0
            return float((sums[hit] / cnts[hit]).mean())
    else:
        def statistic(idx):
            return float(delta[idx].mean())
    fam = np.asarray([families.get(k, k) for k in keys], dtype=object)
    groups = {}
    for i, f in enumerate(fam):
        groups.setdefault(f, []).append(i)
    strata = {}
    for f, idx in groups.items():
        s = (domains or {}).get(keys[idx[0]], "_all") if domains else "_all"
        strata.setdefault(s, []).append(np.asarray(idx))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        picked = []
        for fams in strata.values():
            sel = rng.integers(0, len(fams), size=len(fams))
            picked.extend(fams[i] for i in sel)
        draws[r] = statistic(np.concatenate(picked))
    lo = float(np.percentile(draws, 100 * (1 - confidence) / 2))
    hi = float(np.percentile(draws, 100 * (1 + confidence) / 2))
    all_idx = np.arange(len(keys))
    return {"delta": statistic(all_idx),
            "statistic": "equal-weight domain macro" if domains else "query mean",
            "delta_query_mean": float(delta.mean()),
            "ci": [lo, hi], "confidence": confidence,
            "replicates": replicates, "n_families": len(groups), "n_queries": len(keys),
            "_note": "paired over query families; excludes training-seed variation"}


# ---- fusion -------------------------------------------------------------------------------

def dbsf_at(dense_run, bm25_run, prefetch=100):
    """Fixed DBSF over the two prefetches, using Qdrant's operator from `m12src/qfusion.py`."""
    import sys
    from common import REPO
    if str(REPO / "m12src") not in sys.path:
        sys.path.insert(0, str(REPO / "m12src"))
    import qfusion
    return qfusion.dbsf([qfusion.truncate(dense_run, prefetch),
                         qfusion.truncate(bm25_run, prefetch)])


# ---- alias test ---------------------------------------------------------------------------

def alias_test(run_a, run_b, k=10):
    """Top-10 overlap and rank correlation between the two views of each held-out pair."""
    if set(run_a) != set(run_b):
        miss, extra = sorted(set(run_a) - set(run_b))[:3], sorted(set(run_b) - set(run_a))[:3]
        raise ValueError(f"the two alias views do not cover the same pairs: missing {miss}, "
                         f"unexpected {extra}. Every declared pair needs one result per view.")
    overlaps, rhos = [], []
    for key in sorted(run_a):
        ra = [d for d, _ in sorted(run_a[key].items(), key=lambda kv: -kv[1])[:k]]
        rb = [d for d, _ in sorted(run_b[key].items(), key=lambda kv: -kv[1])[:k]]
        overlaps.append(len(set(ra) & set(rb)) / max(1, k))
        shared = [d for d in ra if d in rb]
        if len(shared) >= 2:
            x = np.asarray([ra.index(d) for d in shared], dtype=np.float64)
            y = np.asarray([rb.index(d) for d in shared], dtype=np.float64)
            rhos.append(_spearman(x, y))
    return {"n_pairs": len(overlaps),
            "top10_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
            "rank_correlation_mean": float(np.mean(rhos)) if rhos else None,
            "n_pairs_with_rank_correlation": len(rhos),
            "_role": "descriptive; appears in no selection predicate"}


def _spearman(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else 0.0


def _rankdata(a):
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(1, len(a) + 1)
    return r


# ---- one evaluation ------------------------------------------------------------------------

def evaluate(query_vecs, doc_vecs, qrels, domains, families=None, doc_ids=None, bm25_run=None,
             baseline_ndcg=None, reg=None, k=10, seed=0, query_ids=None):
    """One artifact, one declared corpus. Returns the full descriptive block.

    `query_ids` (the panel's own `query_id` strings) keys the run, so qrels/domains/families
    keyed that way line up. A key-set mismatch raises here rather than silently scoring every
    query as unjudged and every domain as 'general'.
    """
    reg = reg or registry()
    prefetch = int(reg["serving"]["prefetch"])
    if query_ids is not None:
        keys = list(query_ids)
        if len(set(keys)) != len(keys):
            raise ValueError("query_ids contains duplicates; each query needs one key")
        supplied = [("qrels", qrels), ("domains", domains)]
        if families:
            supplied.append(("families", families))
        for name, m in supplied:
            if set(m) != set(keys):
                miss, extra = sorted(set(keys) - set(m))[:3], sorted(set(m) - set(keys))[:3]
                raise ValueError(f"{name} keys do not match query_ids: missing {miss}, "
                                 f"unexpected {extra}")
    run = search(query_vecs, doc_vecs, k=max(k, prefetch), doc_ids=doc_ids, query_ids=query_ids)
    nd = ndcg_at_k(run, qrels, k)
    rc = recall_at_k(run, qrels, k)
    out = {"n_queries": len(run), "n_docs": int(np.asarray(doc_vecs).shape[0]),
           "ndcg@10": per_domain(nd, domains), "recall@10": per_domain(rc, domains),
           "per_query_ndcg@10": nd}
    if bm25_run is not None:
        if set(bm25_run) != set(run):
            miss = sorted(set(run) - set(bm25_run), key=str)[:3]
            extra = sorted(set(bm25_run) - set(run), key=str)[:3]
            raise ValueError(f"bm25_run keys do not match the dense run: missing {miss}, "
                             f"unexpected {extra}. Positional keys against string query ids "
                             "would fuse unrelated queries and report them as fused.")
        fused = dbsf_at(run, bm25_run, prefetch)
        fnd = ndcg_at_k(fused, qrels, k)
        out["fused_dbsf@100"] = {**per_domain(fnd, domains), "prefetch": prefetch}
    if baseline_ndcg is not None:
        prefs = reg["screening_preferences_not_release_bars"]
        out["vs_baseline"] = paired_family_bootstrap(
            nd, baseline_ndcg, families or {}, replicates=int(prefs["audit_bootstrap_replicates"]),
            confidence=float(prefs["audit_interval_confidence"]), seed=seed, domains=domains)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixtures", default=None,
                    help="directory of synthetic .npy/.json fixtures to evaluate")
    ap.add_argument("--surface", choices=("synthetic", "dev-suite", "panel"), default="synthetic")
    ap.add_argument("--allow-dev-suite", action="store_true")
    ap.add_argument("--allow-panel", action="store_true")
    ap.add_argument("--bundle", default=None, help="exported bundle to read the dev suite with")
    ap.add_argument("--screen", action="store_true",
                    help="read a trained SCREEN ARM's bundle into its own run directory "
                         "(work/m17/runs/<run_id>/screen.json) instead of the V0 read")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    reg = registry()
    if args.surface != "synthetic":
        flag = {"dev-suite": args.allow_dev_suite, "panel": args.allow_panel}[args.surface]
        if not flag:
            raise SystemExit(f"M17 EVAL REFUSED: surface {args.surface!r} needs its explicit "
                             f"--allow-{args.surface} flag; it is a registered read, not a "
                             "development convenience.")
        require_executable(reg, rehearsal=False, what=f"a {args.surface} read")
        if args.surface == "panel":
            raise SystemExit("M17 EVAL REFUSED: the panel reader is not wired up in this "
                             "pre-clock implementation. It is registered work for the execution "
                             "session (m17/STATUS.md steps 5-6), after the lock.")
        if not args.bundle:
            raise SystemExit("M17 EVAL REFUSED: --surface dev-suite needs --bundle, the exported "
                             "artifact whose registered read this is.")
        if not args.out:
            raise SystemExit("M17 EVAL REFUSED: --surface dev-suite needs --out, the registered "
                             f"destination of the read (it is {V0_READ_PATH} for the V0 read). "
                             "The reader writes the result and its receipt itself and refuses to "
                             "overwrite either.")
        rep = (screen_read if args.screen else dev_suite_read)(args.bundle, args.out,
                                                               allow_dev_suite=True)
        rep.pop("per_query_ndcg@10", None)
        rep.pop("component_identities", None)
        print(json.dumps(rep, indent=1, sort_keys=True))
        return 0
    if not args.fixtures:
        raise SystemExit("--fixtures is required for the synthetic surface")
    d = _check_path(Path(args.fixtures))
    fx = json.loads((d / "fixtures.json").read_text())
    rep = evaluate(np.load(d / "query_vecs.npy"), np.load(d / "doc_vecs.npy"),
                   {int(k): v for k, v in fx["qrels"].items()},
                   {int(k): v for k, v in fx["domains"].items()},
                   {int(k): v for k, v in fx.get("families", {}).items()},
                   doc_ids=fx.get("doc_ids"), reg=reg)
    rep.pop("per_query_ndcg@10", None)
    print(json.dumps(rep, indent=1, sort_keys=True))
    if args.out:
        write_json(args.out, rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
