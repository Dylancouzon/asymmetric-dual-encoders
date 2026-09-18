"""M13-prepared helpers for the execution of the triggered descriptive reserved batch.

The protected-path capability is claimed by ``m13src.score13`` before any function here is
called.  This module never claims or weakens that boundary.  It authenticates the pre-encoded
document shards, opens the frozen query/qrel payload once per system, performs exact retrieval,
and derives the zero-alpha NDO-3 report registered in ``m10/final_run_registry.json``.

**M20 extension (owner rulings R20, R22; registered pre-observation in ``m20/REGISTRATION.md``
and ``m20/beir15_registry.json``).**  The roster grows from three systems to eight.  What did not
change: the four datasets, the R1/R2 estimands, the NDO-3 weights, B, the seed, the interval
method and ``alpha = 0``.  R1 is still nano minus bge-small and R2 is still nano minus LEAF,
computed from exactly the same two pairs of systems.  The five added rows are descriptive and
enter no contrast.

The three systems on the Stella tower -- Nano, Zero and the Stella query tower -- share ONE set of
document shards.  ``m20src/roster.py`` holds the roster, the query towers and the fusion operator
so the protected transaction and the unprotected BEIR-15 pass cannot drift apart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in ("m7src", "m10src", "m12src", "m13src", "m20src"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

import roster as R20  # noqa: E402

DATASETS = ("fever", "dbpedia-entity", "cqadup-android", "cqadup-english")
SYSTEMS = R20.SYSTEMS
DENSE_SYSTEMS = R20.DENSE_SYSTEMS
DERIVED_SYSTEMS = R20.DERIVED_SYSTEMS
RUN_PRODUCERS = R20.RUN_PRODUCERS
ENC_ROOT = REPO / "work" / "m13-reserved-enc"
RUN_ROOT = REPO / "work" / "m20-runs"
RUN_STAGE = "reserved"
LEAF_QUERY_REVISION = R20.LEAF_QUERY_REVISION
BGE_REVISION = R20.BGE_REVISION
ARCTIC_REVISION = R20.ARCTIC_REVISION
STELLA_REVISION = R20.STELLA_REVISION
BGE_PREFIX = R20.BGE_PREFIX
# Per system, the document side its rows were produced against.  `bm25` has none; the two DBSF
# rows inherit their inputs' towers and are recorded as derived.
DOCUMENT_ENCODERS = {
    system: {"model": R20.TOWERS[R20.DOC_TOWER[system]]["model"],
             "revision": R20.TOWERS[R20.DOC_TOWER[system]]["revision"]}
    for system in DENSE_SYSTEMS
}
DOCUMENT_ENCODERS["bm25"] = {"model": "bm25s", "revision": "corpus text; no neural document tower"}
for _derived, (_dense, _lex) in DERIVED_SYSTEMS.items():
    DOCUMENT_ENCODERS[_derived] = {"model": "derived", "revision": f"{_dense} + {_lex} dbsf@100"}


def sha_file(path, block=8 << 20):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def nano_dependency_identity(model):
    """Re-derive the M13 dependency identity from the exact constructed Nano10 objects."""
    return R20.nano_dependency_identity(model)


def registered_nano_dependency(repo=REPO):
    return R20.registered_nano_dependency(repo)


class ShardedVectors:
    """Read-only array façade over authenticated ``.npy`` shards."""

    def __init__(self, manifest_path, verify=True):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("status") != "COMPLETE":
            raise ValueError(f"{self.manifest_path}: pre-encode is not complete")
        self.n = int(self.manifest["n_docs"])
        self.dim = int(self.manifest["dim"])
        self.shard_rows = int(self.manifest["shard_rows"])
        count = (self.n + self.shard_rows - 1) // self.shard_rows
        self.paths = []
        for shard in range(count):
            sid = f"{shard:05d}"
            record = self.manifest.get("shards", {}).get(sid)
            path = self.manifest_path.parent / f"shard_{sid}.npy"
            if record is None or not path.exists():
                raise ValueError(f"{self.manifest_path}: missing recorded shard {sid}")
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"{path}: byte size changed")
            if verify and sha_file(path) != record["sha256"]:
                raise ValueError(f"{path}: sha256 changed")
            self.paths.append(path)
        self._arrays = [None] * len(self.paths)
        self.shape = (self.n, self.dim)

    def __len__(self):
        return self.n

    def _array(self, shard):
        if self._arrays[shard] is None:
            value = np.load(self.paths[shard], mmap_mode="r", allow_pickle=False)
            expected_rows = min(self.shard_rows, self.n - shard * self.shard_rows)
            if value.shape != (expected_rows, self.dim) or value.dtype != np.float16:
                raise ValueError(f"{self.paths[shard]}: shape/dtype {value.shape}/{value.dtype}")
            self._arrays[shard] = value
        return self._arrays[shard]

    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key, key + 1, 1)
        if not isinstance(key, slice):
            raise TypeError("ShardedVectors accepts integer or slice indexing only")
        start, stop, step = key.indices(self.n)
        if step != 1:
            raise ValueError("ShardedVectors does not support strided slices")
        if start >= stop:
            return np.empty((0, self.dim), dtype=np.float16)
        pieces = []
        pos = start
        while pos < stop:
            shard = pos // self.shard_rows
            local = pos - shard * self.shard_rows
            take = min(stop - pos, len(self._array(shard)) - local)
            pieces.append(self._array(shard)[local:local + take])
            pos += take
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)


def cache_dir_for(system):
    """The shard directory a system reads.  Three systems share the Stella tower's directory."""
    return R20.TOWER_DIR[R20.DOC_TOWER[system]]


def cache_for(system, dataset, repo=REPO, verify=True):
    root = Path(repo) / "work" / "m13-reserved-enc"
    vectors = ShardedVectors(root / cache_dir_for(system) / dataset / "manifest.json",
                             verify=verify)
    ids_path = Path(repo) / vectors.manifest["doc_ids_path"]
    if sha_file(ids_path) != vectors.manifest["doc_ids_file_sha256"]:
        raise ValueError(f"{ids_path}: document-id file changed")
    doc_ids = [str(value) for value in json.loads(ids_path.read_text())]
    if len(doc_ids) != len(vectors) or len(set(doc_ids)) != len(doc_ids):
        raise ValueError(f"{dataset}: document ids are missing or duplicated")
    return doc_ids, vectors


def _doc_text(row):
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def corpus_for(dataset, repo=REPO):
    """Document ids and texts for the lexical system, authenticated against the frozen manifest.

    This is the public corpus, the same bytes the pre-encode already hashed pre-tag.  It is read
    here because BM25 has no pre-encoded vectors to stand in for it, and the ids are checked
    against the pre-encode's shared id file so BM25 and every dense system rank the same document
    list in the same order.

    It deliberately does NOT import `m8src/pre_encode.py`.  That module CLAIMS the corpus-only
    allowlist entry at import time, and `paths_guard.claim` refuses a second, different claim in
    one process -- so importing it here would raise inside the tagged transaction, after the
    reserved access had already been spent.  The loading is therefore inlined under the
    `m13src.score13` capability this process already holds.
    """
    from datasets import load_dataset

    from hashing import sha_stream_list

    registry = json.loads((Path(repo) / "m20" / "beir15_registry.json").read_text())
    forums = registry["cqadupstack"]["forums"]
    if dataset.startswith("cqadup-"):
        row = forums[dataset.split("-", 1)[1]]
        source, revision = row["source"], row["revision"]
    else:
        row = next(r for r in registry["datasets"] if r["key"] == dataset)
        source, revision = row["source"], row["revision"]
    corpus = load_dataset(source, "corpus", revision=revision)["corpus"]
    expected = json.loads((Path(repo) / "results" / "eval_manifest.json").read_text())[
        "m7_untouched_final"][dataset]
    identity = {
        "n_docs": len(corpus),
        "corpus_ids_sha256": sha_stream_list(str(value) for value in corpus["_id"]),
        "corpus_text_sha256": sha_stream_list(_doc_text(item) for item in corpus),
    }
    bad = [key for key in identity if identity[key] != expected.get(key)]
    if bad:
        raise ValueError(f"{dataset}: public corpus differs from the frozen manifest on {bad}")
    doc_ids = [str(value) for value in corpus["_id"]]
    doc_texts = [_doc_text(item) for item in corpus]
    ids_path = Path(repo) / "work" / "m13-reserved-enc" / "corpora" / dataset / "doc_ids.json"
    if not ids_path.exists():
        raise ValueError(f"{dataset}: the pre-encode's shared document-id file is missing")
    if [str(value) for value in json.loads(ids_path.read_text())] != doc_ids:
        raise ValueError(f"{dataset}: BM25 document order differs from the pre-encoded order")
    return doc_ids, doc_texts, {"source": source, "revision": revision, **identity}


_PAYLOAD_CACHE = {}


def load_payload(cfg, dataset):
    """Open and authenticate one frozen reserved query/qrel payload.

    Memoized for the life of the process. The payload is immutable and is hash-authenticated on
    first load, so a second read buys no assurance and costs another protected open. The archive
    exporter below consumes these cached objects instead of reopening the files, which is what
    makes the registered "no additional protected read" archive contract true rather than merely
    claimed (Astra review, 2026-09-18, P1).
    """
    import access13 as A

    key = (str(cfg.frozen_eval_dir), dataset)
    if key in _PAYLOAD_CACHE:
        return _PAYLOAD_CACHE[key]
    payload = json.loads((Path(cfg.frozen_eval_dir) / f"untouched-{dataset}.json").read_text())
    expected = json.loads(Path(cfg.manifest_path).read_text())["m7_untouched_final"][dataset]
    qids = sorted(str(q) for q in payload["queries"])
    qtexts = [payload["queries"][q] for q in qids]
    got = {
        "qids_sha256": A.sha_json(qids),
        "qtexts_sha256": A.sha_json(qtexts),
        "qrels_sha256": A.sha_json(payload["qrels"]),
    }
    bad = [key for key, value in got.items() if expected.get(key) != value]
    if bad:
        raise ValueError(f"{dataset}: protected payload hash mismatch: {bad}")
    if set(qids) != set(payload["qrels"]):
        raise ValueError(f"{dataset}: query/qrel id sets differ")
    _PAYLOAD_CACHE[key] = (qids, qtexts, payload["qrels"], got)
    return _PAYLOAD_CACHE[key]


class QueryEncoder(R20.QueryEncoder):
    """The registered query towers, under the transaction's device setting."""

    def __init__(self, system, cfg):
        super().__init__(system, cfg=cfg,
                         device=str(cfg.extra.get("reserved_device", "cuda")), repo=cfg.repo)


def preflight_models(cfg=None, device="cuda"):
    """Load and exercise every query tower without contacting a protected payload."""
    import access13 as A

    cfg = cfg or A.production()
    cfg.extra["reserved_device"] = device
    R20.assert_registered_identities(cfg.repo)
    rows = {}
    for system in DENSE_SYSTEMS:
        encoder = QueryEncoder(system, cfg)
        value = encoder.encode(["M13 reserved query-tower preflight; no benchmark text."])
        rows[system] = {"shape": list(value.shape), "identity": encoder.identity,
                        "finite": bool(np.isfinite(value).all())}
        encoder.release()
        del encoder
    print(json.dumps({"status": "PASSED", "device": device, "systems": rows}, default=str))
    return rows


# ------------------------------------------------------------------ scoring one system

def _run_path(cfg, system, dataset):
    return R20.run_path(Path(cfg.repo) / "work" / "m20-runs", RUN_STAGE, system, dataset)


def _check_scores(system, dataset, scores, qids):
    if set(scores) != set(qids):
        raise ValueError(f"{system}/{dataset}: scorer omitted "
                         f"{len(set(qids) - set(scores))} queries")
    values = np.asarray(list(scores.values()), dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"{system}/{dataset}: nDCG falls outside [0, 1]")


def _score_dense(cfg, system, encoder):
    from evalkit import topk_ids_scores

    datasets = {}
    for dataset in DATASETS:
        doc_ids, doc_vectors = cache_for(system, dataset, repo=cfg.repo, verify=True)
        qids, qtexts, qrels, payload_hashes = load_payload(cfg, dataset)
        qvectors = encoder.encode(qtexts)
        if qvectors.shape[1] != doc_vectors.shape[1]:
            raise ValueError(f"{system}/{dataset}: query/doc dimensions differ")
        run = topk_ids_scores(qvectors, doc_vectors, doc_ids, k=R20.DENSE_DEPTH,
                              chunk=int(cfg.extra.get("reserved_chunk", 50_000)),
                              device=str(cfg.extra.get("reserved_device", "cuda")), qids=qids)
        scores = R20.per_query_ndcg10(run, qrels)
        _check_scores(system, dataset, scores, qids)
        row = {
            "scores": scores,
            "mean_ndcg10": float(np.mean(list(scores.values()))),
            "n_queries": len(scores),
            "payload_hashes": payload_hashes,
            "document_cache_manifest": str(doc_vectors.manifest_path.relative_to(cfg.repo)),
            "document_cache_manifest_sha256": sha_file(doc_vectors.manifest_path),
        }
        if system in RUN_PRODUCERS:
            path = _run_path(cfg, system, dataset)
            row["run_path"] = str(path.relative_to(cfg.repo))
            row["run_sha256"] = R20.save_run(path, R20.truncate(run), qids)
        datasets[dataset] = row
        print(f"[reserved13] {system}: {dataset} complete ({len(scores):,} queries)", flush=True)
    return datasets


def _score_bm25(cfg):
    datasets = {}
    for dataset in DATASETS:
        doc_ids, doc_texts, corpus_identity = corpus_for(dataset, repo=cfg.repo)
        qids, qtexts, qrels, payload_hashes = load_payload(cfg, dataset)
        run = R20.bm25_run(doc_ids, doc_texts, qids, qtexts)
        scores = R20.per_query_ndcg10(run, qrels)
        # BM25 can return nothing for a query whose terms are all stopwords or all self-hits.
        # pytrec_eval scores only the queries present in the run, so restore the missing ones at
        # 0.0 rather than silently reporting a mean over a different query set.
        for qid in qids:
            scores.setdefault(qid, 0.0)
        _check_scores("bm25", dataset, scores, qids)
        path = _run_path(cfg, "bm25", dataset)
        datasets[dataset] = {
            "scores": scores,
            "mean_ndcg10": float(np.mean([scores[q] for q in qids])),
            "n_queries": len(scores),
            "payload_hashes": payload_hashes,
            "corpus_identity": corpus_identity,
            "bm25_package_versions": R20.bm25_versions(),
            "empty_runs": int(sum(1 for q in qids if not run.get(q))),
            "run_path": str(path.relative_to(cfg.repo)),
            "run_sha256": R20.save_run(path, R20.truncate(run), qids),
        }
        # Measured: FEVER's BM25 peaks at 20.1 GB and MS MARCO's at 20.9 GB on a 26.7 GB box,
        # and two large corpora in ONE process reached 23.7 GB. Dropping this dataset's text and
        # collecting before the next corpus loads is what keeps the peak at one corpus rather than
        # two. Datasets are ordered largest-first, so the worst case happens on the cleanest heap.
        import gc

        del doc_texts, run
        gc.collect()
        print(f"[reserved13] bm25: {dataset} complete ({len(scores):,} queries)", flush=True)
    return datasets


def _score_derived(cfg, system, outputs):
    """Fuse two persisted top-100 runs.  Nothing is encoded and nothing is re-retrieved."""
    dense_key, lexical_key = DERIVED_SYSTEMS[system]
    for key in (dense_key, lexical_key):
        if key not in outputs:
            raise ValueError(f"{system}: input system {key!r} has not completed")
    datasets = {}
    for dataset in DATASETS:
        rows = {key: outputs[key]["datasets"][dataset] for key in (dense_key, lexical_key)}
        runs = {}
        for key, row in rows.items():
            run, _qids = R20.load_run(Path(cfg.repo) / row["run_path"], row["run_sha256"])
            runs[key] = run
        qids, _qtexts, qrels, payload_hashes = load_payload(cfg, dataset)
        if rows[dense_key]["payload_hashes"] != rows[lexical_key]["payload_hashes"] \
                or rows[dense_key]["payload_hashes"] != payload_hashes:
            raise ValueError(f"{system}/{dataset}: inputs scored a different payload")
        fused = R20.dbsf_at_depth(runs[dense_key], runs[lexical_key])
        scores = R20.per_query_ndcg10(fused, qrels)
        for qid in qids:
            scores.setdefault(qid, 0.0)
        _check_scores(system, dataset, scores, qids)
        datasets[dataset] = {
            "scores": scores,
            "mean_ndcg10": float(np.mean([scores[q] for q in qids])),
            "n_queries": len(scores),
            "payload_hashes": payload_hashes,
            "inputs": {key: {"run_path": rows[key]["run_path"],
                             "run_sha256": rows[key]["run_sha256"]} for key in rows},
            "operator": "m12src/qfusion.py:dbsf over both prefetches truncated to 100 first",
        }
        print(f"[reserved13] {system}: {dataset} complete ({len(scores):,} queries)", flush=True)
    return datasets


def score_system(cfg, conf, system, _rows_candidate=None, outputs=None):
    """Score all four datasets for one system; caller writes the atomic system output."""
    if system not in SYSTEMS or list(conf["reserved"]["datasets"]) != [
            "FEVER", "dbpedia-entity", "cqadup-android", "cqadup-english"]:
        raise ValueError("reserved systems or datasets differ from the registered batch")
    R20.assert_registered_identities(cfg.repo)
    started = time.time()
    encoder_identity = {"query_model": system}
    if system in DENSE_SYSTEMS:
        encoder = QueryEncoder(system, cfg)
        encoder_identity = encoder.identity
        datasets = _score_dense(cfg, system, encoder)
        encoder.release()
    elif system in DERIVED_SYSTEMS:
        datasets = _score_derived(cfg, system, outputs or {})
        encoder_identity = {"query_model": "derived",
                            "inputs": list(DERIVED_SYSTEMS[system]),
                            "compute_dtype": "n/a"}
    else:
        datasets = _score_bm25(cfg)
        import fusion

        encoder_identity = {"query_model": "bm25s", "compute_dtype": "n/a",
                            "config": fusion.BM25_CONFIG, "depth": fusion.DEPTH,
                            "package_versions": R20.bm25_versions()}
    return {
        "status": "COMPLETE",
        "system": system,
        "datasets": datasets,
        "query_encoder": encoder_identity,
        "document_encoder": DOCUMENT_ENCODERS[system],
        "document_compute_dtype": (R20.DOC_COMPUTE_NOTE if system in DENSE_SYSTEMS
                                   else "n/a; no neural document tower"),
        "query_compute_dtype": encoder_identity.get("compute_dtype", "fp32"),
        "elapsed_seconds": time.time() - started,
        "transaction": dict(cfg.extra.get("reserved_identity") or {}),
    }


def validate_output(record, cfg, system):
    """Authenticate an atomic per-system output before accepting it on continuation.

    A completed file is the only state the reserved transaction resumes past (registry
    ``reserved.crash``, confirmed by owner ruling R23).  Presence alone is therefore insufficient:
    bind the file to this transaction, the frozen payload hashes, and -- for a dense system -- the
    exact document-cache manifest that the scorer verified.
    """
    import access13 as A

    if system not in SYSTEMS or record.get("status") != "COMPLETE" \
            or record.get("system") != system:
        raise ValueError(f"{system}: saved output has the wrong system or status")
    identity = dict(cfg.extra.get("reserved_identity") or {})
    if not identity or record.get("transaction") != identity:
        raise ValueError(f"{system}: saved output belongs to a different transaction")
    if record.get("document_encoder") != DOCUMENT_ENCODERS[system]:
        raise ValueError(f"{system}: saved document identity changed")
    if system in DENSE_SYSTEMS:
        if record.get("document_compute_dtype") != R20.DOC_COMPUTE_NOTE \
                or record.get("query_compute_dtype") != "fp32":
            raise ValueError(f"{system}: saved encoder dtype changed")
    datasets = record.get("datasets") or {}
    if set(datasets) != set(DATASETS):
        raise ValueError(f"{system}: saved output has incomplete datasets")
    frozen = json.loads(Path(cfg.manifest_path).read_text())["m7_untouched_final"]
    for dataset in DATASETS:
        row = datasets[dataset]
        scores = row.get("scores")
        if not isinstance(scores, dict) or not scores or any(not isinstance(q, str) for q in scores):
            raise ValueError(f"{system}/{dataset}: invalid saved per-query scores")
        values = np.asarray([scores[q] for q in sorted(scores)], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{system}/{dataset}: saved nDCG falls outside [0, 1]")
        if row.get("n_queries") != len(scores) \
                or abs(float(row.get("mean_ndcg10", np.nan)) - float(values.mean())) > 1e-12:
            raise ValueError(f"{system}/{dataset}: saved count or mean does not match scores")
        expected_hashes = {key: frozen[dataset][key]
                           for key in ("qids_sha256", "qtexts_sha256", "qrels_sha256")}
        if row.get("payload_hashes") != expected_hashes \
                or A.sha_json(sorted(scores)) != expected_hashes["qids_sha256"]:
            raise ValueError(f"{system}/{dataset}: saved query/payload identity changed")
        if system in DENSE_SYSTEMS:
            expected_cache = Path("work") / "m13-reserved-enc" / cache_dir_for(system) / dataset \
                / "manifest.json"
            if row.get("document_cache_manifest") != str(expected_cache):
                raise ValueError(f"{system}/{dataset}: saved document-cache path changed")
            cache = Path(cfg.repo) / expected_cache
            if not cache.is_file() or sha_file(cache) != row.get("document_cache_manifest_sha256"):
                raise ValueError(f"{system}/{dataset}: saved document-cache manifest changed")
        if system in RUN_PRODUCERS:
            run_path = Path(cfg.repo) / str(row.get("run_path", "missing"))
            if not run_path.is_file() or sha_file(run_path) != row.get("run_sha256"):
                raise ValueError(f"{system}/{dataset}: persisted top-100 run is missing or changed")
    return record


def _archive_intact(root, entry):
    """Is a previously exported dataset still on disk, byte for byte, as its record describes?"""
    for name in ("queries.jsonl.gz", "qrels.jsonl.gz"):
        recorded = entry.get(name)
        if not recorded:
            return False
        path = Path(root) / recorded["path"]
        if not path.exists() or path.stat().st_size != recorded["bytes"]:
            return False
        if sha_file(path) != recorded["sha256"]:
            return False
    return True


def export_reserved_payload_archive(cfg, root):
    """Write the reserved four's queries and qrels into the archive, INSIDE the transaction.

    R22 requires raw corpora, queries and qrels for every evaluated dataset at both archive
    targets. The reserved labels can only be read under the capability this transaction holds, so
    they are exported here, from payloads already opened and authenticated for scoring, rather
    than by a later archiving pass reopening protected data. This adds no protected read: every
    dataset's payload has already been loaded by the systems above.

    The format matches `m20src/archive.py`: gzipped JSON lines, mtime zeroed so the bytes and the
    manifest hash are reproducible.

    It exports ONLY from payloads this process already holds, and each dataset's record is
    persisted as it is written. A continuation in which every system output already exists loads no
    payload, so nothing is cached; it then reuses the archive the earlier attempt wrote, after
    re-hashing it, and refuses outright rather than reopening protected data to rebuild it. Before
    the memoization and this check the exporter reopened all four payloads on every run, including
    such a continuation, which contradicted the contract the call site claimed (Astra review,
    2026-09-18, P1).
    """
    import gzip

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    record_path = root / "payload_archive.json"
    written = json.loads(record_path.read_text()) if record_path.exists() else {}
    for dataset in DATASETS:
        key = (str(cfg.frozen_eval_dir), dataset)
        if key not in _PAYLOAD_CACHE:
            entry = written.get(dataset)
            if entry and _archive_intact(root, entry):
                print(f"[reserved13] {dataset} queries and qrels already archived and verified; "
                      "no protected read", flush=True)
                continue
            raise ValueError(
                f"{dataset}: its payload is not open in this process and no verified archive of it "
                "exists; refusing to reopen protected data to build the archive")
        qids, qtexts, qrels, payload_hashes = _PAYLOAD_CACHE[key]
        out = root / "datasets" / dataset
        out.mkdir(parents=True, exist_ok=True)
        rows = {
            "queries.jsonl.gz": [{"_id": qid, "text": text} for qid, text in zip(qids, qtexts)],
            "qrels.jsonl.gz": [{"query-id": qid, "corpus-id": did, "score": int(score)}
                               for qid in qids for did, score in sorted(qrels[qid].items())],
        }
        entry = {"payload_hashes": payload_hashes}
        for name, items in rows.items():
            path = out / name
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                for item in items:
                    stream.write((json.dumps(item, sort_keys=True) + "\n").encode())
            tmp.replace(path)
            entry[name] = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                           "sha256": sha_file(path), "rows": len(items)}
        written[dataset] = entry
        # Persist as each dataset lands, so a crash after the last system completes cannot leave
        # the archive unbuildable without another protected read.
        tmp = record_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(written, indent=2, sort_keys=True) + "\n")
        tmp.replace(record_path)
        print(f"[reserved13] archived {dataset} queries and qrels", flush=True)
    return written


def _plan(qids_by_dataset, B, seed):
    rng = np.random.default_rng(int(seed))
    plan, digest = {}, hashlib.sha256(f"B={B};seed={seed}".encode())
    for dataset in sorted(qids_by_dataset):
        n = len(qids_by_dataset[dataset])
        idx = rng.integers(0, n, size=(int(B), n), dtype=np.int64)
        plan[dataset] = idx
        digest.update(dataset.encode())
        digest.update(idx.tobytes())
    return plan, digest.hexdigest()


def summarize(outputs, conf):
    """Derive the registered zero-alpha report from every complete atomic output.

    The registered contrasts are unchanged by the M20 roster extension: R1 is nano minus
    bge-small and R2 is nano minus LEAF, over the same datasets, weights, B and seed.  The five
    added systems contribute descriptive per-dataset means and nothing else.
    """
    if set(outputs) != set(SYSTEMS):
        raise ValueError(f"reserved outputs are incomplete: {sorted(outputs)}")
    for system, record in outputs.items():
        if record.get("status") != "COMPLETE" or set(record.get("datasets", {})) != set(DATASETS):
            raise ValueError(f"{system}: incomplete reserved output")
    qids = {}
    for dataset in DATASETS:
        sets = [set(outputs[system]["datasets"][dataset]["scores"]) for system in SYSTEMS]
        if any(value != sets[0] for value in sets[1:]) or not sets[0]:
            raise ValueError(f"{dataset}: systems scored different query ids")
        for system in SYSTEMS:
            values = np.asarray(list(outputs[system]["datasets"][dataset]["scores"].values()),
                                dtype=np.float64)
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError(f"{system}/{dataset}: nDCG scores fall outside [0, 1]")
        qids[dataset] = sorted(sets[0])

    registered = conf["reserved"]
    B, seed = int(registered["B"]), int(registered["seed"])
    plan, plan_sha = _plan(qids, B, seed)
    weights = {"dbpedia-entity": float(registered["ndo3_weights"]["dbpedia"]),
               "cqadup-android": float(registered["ndo3_weights"]["cqadup-android"]),
               "cqadup-english": float(registered["ndo3_weights"]["cqadup-english"])}
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise ValueError("registered NDO-3 weights do not sum to one")
    contrasts = {
        "R1": ("nano-dense", "bge-small-en-v1.5"),
        "R2": ("nano-dense", "leaf-ir-asym"),
    }
    result = {}
    for name, (a, b) in contrasts.items():
        diffs, draws = {}, {}
        for dataset in DATASETS:
            av = outputs[a]["datasets"][dataset]["scores"]
            bv = outputs[b]["datasets"][dataset]["scores"]
            d = np.asarray([av[q] - bv[q] for q in qids[dataset]], dtype=np.float64)
            diffs[dataset] = d
            draws[dataset] = d[plan[dataset]].mean(axis=1)

        def estimate(use_weights):
            total = sum(use_weights.values())
            normalized = {key: value / total for key, value in use_weights.items()}
            sampled = sum(normalized[key] * draws[key] for key in normalized)
            point = sum(normalized[key] * float(diffs[key].mean()) for key in normalized)
            return {"delta_raw": point,
                    "ci95_raw": np.quantile(sampled, [0.025, 0.975],
                                              method="inverted_cdf").tolist(),
                    "weights": normalized}

        ndo3 = estimate(weights)
        equal = estimate({dataset: 1.0 for dataset in weights})
        leave_one_out = {omitted: estimate({key: value for key, value in weights.items()
                                            if key != omitted}) for omitted in weights}
        pooled = np.concatenate([diffs[key] for key in weights])
        pooled_weights = {key: len(diffs[key]) / len(pooled) for key in weights}
        pooled_draws = sum(pooled_weights[key] * draws[key] for key in weights)
        result[name] = {
            "a": a, "b": b,
            "ndo3": ndo3,
            "equal_weight_macro": equal,
            "query_pooled": {
                "delta_raw": float(pooled.mean()),
                "ci95_raw": np.quantile(pooled_draws, [0.025, 0.975],
                                          method="inverted_cdf").tolist(),
                "weights": pooled_weights,
            },
            "leave_one_out": leave_one_out,
            "per_dataset": {
                dataset: {"delta_raw": float(diffs[dataset].mean()),
                          "ci95_raw": np.quantile(draws[dataset], [0.025, 0.975],
                                                    method="inverted_cdf").tolist(),
                          "n": len(diffs[dataset]),
                          "classification": ("double-contaminated sensitivity; zero alpha"
                                             if dataset == "fever" else "descriptive; zero alpha")}
                for dataset in DATASETS},
        }
    return {
        "status": "complete",
        "scope": "Registered descriptive reserved batch; zero alpha; no gate or release claim.",
        "roster": list(SYSTEMS),
        "roster_authority": "R20 and R22; m10/final_run_registry.json reserved._amended_2026_09_16",
        "B": B, "seed": seed, "quantile_method": "inverted_cdf",
        "draw_plan_sha256": plan_sha,
        "systems": {system: {dataset: outputs[system]["datasets"][dataset]["mean_ndcg10"]
                             for dataset in DATASETS} for system in SYSTEMS},
        "contrasts": result,
        "contrast_note": "R1 and R2 are the only registered contrasts and are unchanged by the "
                         "roster extension; the five added systems enter neither.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-models", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.preflight_models:
        parser.error("--preflight-models is required")
    preflight_models(device=args.device)
