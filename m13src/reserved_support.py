"""M13-prepared helpers for M14's execution of the triggered descriptive reserved batch.

The protected-path capability is claimed by ``m13src.score13`` before any function here is
called.  This module never claims or weakens that boundary.  It authenticates the pre-encoded
document shards, opens the frozen query/qrel payload once per system, performs exact retrieval,
and derives the zero-alpha NDO-3 report registered in ``m10/final_run_registry.json``.
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
for _p in ("m7src", "m10src", "m13src"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

DATASETS = ("fever", "dbpedia-entity", "cqadup-android", "cqadup-english")
SYSTEMS = ("nano-dense", "bge-small-en-v1.5", "leaf-ir-asym")
ENC_ROOT = REPO / "work" / "m13-reserved-enc"
LEAF_QUERY_REVISION = "4262131b32c3182bd06e67e92ae69d7bd66e0c5c"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
ARCTIC_REVISION = "e58a8f756156a1293d763f17e3aae643474e9b8a"
STELLA_REVISION = "ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
DOCUMENT_ENCODERS = {
    "nano-dense": {"model": "NovaSearch/stella_en_400M_v5", "revision": STELLA_REVISION},
    "bge-small-en-v1.5": {"model": "BAAI/bge-small-en-v1.5", "revision": BGE_REVISION},
    "leaf-ir-asym": {"model": "Snowflake/snowflake-arctic-embed-m-v1.5",
                     "revision": ARCTIC_REVISION},
}


def sha_file(path, block=8 << 20):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def nano_dependency_identity(model):
    """Re-derive the M13 dependency identity from the exact constructed Nano10 objects."""
    import nano10 as N

    repo = N.REPOS[model.key]
    h = hashlib.sha256()
    h.update(repo.encode())
    h.update(model.tok.backend_tokenizer.to_str().encode())
    h.update(model.backbone.config.to_json_string().encode())
    return {"repo": repo, "sha256": h.hexdigest()}


def registered_nano_dependency(repo=REPO):
    manifest = json.loads((Path(repo) / "m13" / "LOTTE_GATE_MANIFEST.json").read_text())
    dependency = (manifest.get("candidate") or {}).get("dependencies")
    if not isinstance(dependency, dict) or set(dependency) != {"repo", "sha256"}:
        raise ValueError("LoTTE manifest does not carry the frozen Nano dependency identity")
    return dependency


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


def cache_for(system, dataset, repo=REPO, verify=True):
    root = Path(repo) / "work" / "m13-reserved-enc"
    vectors = ShardedVectors(root / system / dataset / "manifest.json", verify=verify)
    ids_path = Path(repo) / vectors.manifest["doc_ids_path"]
    if sha_file(ids_path) != vectors.manifest["doc_ids_file_sha256"]:
        raise ValueError(f"{ids_path}: document-id file changed")
    doc_ids = [str(value) for value in json.loads(ids_path.read_text())]
    if len(doc_ids) != len(vectors) or len(set(doc_ids)) != len(doc_ids):
        raise ValueError(f"{dataset}: document ids are missing or duplicated")
    return doc_ids, vectors


def load_payload(cfg, dataset):
    """Open and authenticate one frozen reserved query/qrel payload."""
    import access13 as A

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
    return qids, qtexts, payload["qrels"], got


class QueryEncoder:
    def __init__(self, system, cfg):
        self.system = system
        search_device = str(cfg.extra.get("reserved_device", "cuda"))
        # Nano10's serving method deliberately enters bf16 autocast on CUDA.  The inherited M8
        # confirmatory contract is fp32 compute, so keep that existing method on CPU rather than
        # forking its math here.  The two SentenceTransformer query towers use fp32 on the GPU.
        self.device = "cpu" if system == "nano-dense" else search_device
        self.model = None
        if system == "nano-dense":
            import score13 as S

            freeze = S._freeze_blob(cfg)
            self.model = S.Nano10Student(freeze, device=self.device, repo=cfg.repo)
            dependency = nano_dependency_identity(self.model.model)
            if dependency != registered_nano_dependency(cfg.repo):
                raise ValueError("constructed Nano tokenizer/backbone dependency changed")
            self.dim = 1024
            self.identity = {"query_model": "M13 frozen nano", "revision": freeze["_sha256"],
                             "device": self.device, "compute_dtype": "fp32",
                             "dependency": dependency}
        else:
            import torch
            from sentence_transformers import SentenceTransformer

            if system == "bge-small-en-v1.5":
                repo, revision, dim = "BAAI/bge-small-en-v1.5", BGE_REVISION, 384
            elif system == "leaf-ir-asym":
                repo, revision, dim = "MongoDB/mdbr-leaf-ir", LEAF_QUERY_REVISION, 768
            else:
                raise ValueError(f"unregistered reserved system {system!r}")
            if self.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
            self.model = SentenceTransformer(repo, revision=revision, device=self.device,
                                             model_kwargs={"dtype": torch.float32})
            self.model.max_seq_length = 512
            self.dim = dim
            self.identity = {"query_model": repo, "revision": revision,
                             "device": self.device, "compute_dtype": "fp32"}

    def encode(self, texts):
        values = list(texts)
        if self.system == "nano-dense":
            out = self.model.encode(values)
        elif self.system == "leaf-ir-asym":
            # The pinned model card defines the asymmetric query route through its named
            # ``query`` prompt.  Use that interface directly so the model's own pinned config,
            # rather than a duplicated free-form string, remains part of the execution path.
            out = self.model.encode(values, prompt_name="query", batch_size=256,
                                    normalize_embeddings=True, show_progress_bar=False,
                                    convert_to_numpy=True)
        else:
            out = self.model.encode([BGE_PREFIX + value for value in values], batch_size=256,
                                    normalize_embeddings=True, show_progress_bar=False,
                                    convert_to_numpy=True)
        out = np.asarray(out, dtype=np.float32)
        if out.shape != (len(values), self.dim) or not np.isfinite(out).all():
            raise ValueError(f"{self.system}: invalid query vectors {out.shape}")
        return out


def preflight_models(cfg=None, device="cuda"):
    """Load and exercise every query tower without contacting a protected payload."""
    import access13 as A

    cfg = cfg or A.production()
    cfg.extra["reserved_device"] = device
    rows = {}
    for system in SYSTEMS:
        encoder = QueryEncoder(system, cfg)
        value = encoder.encode(["M13 reserved query-tower preflight; no benchmark text."])
        rows[system] = {"shape": list(value.shape), "identity": encoder.identity,
                        "finite": bool(np.isfinite(value).all())}
        del encoder
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    print(json.dumps({"status": "PASSED", "device": device, "systems": rows}))
    return rows


def score_system(cfg, conf, system, _rows_candidate=None):
    """Score all four datasets for one system; caller writes the atomic system output."""
    from evalkit import per_query_ndcg, topk_ids_scores

    if system not in SYSTEMS or list(conf["reserved"]["datasets"]) != [
            "FEVER", "dbpedia-entity", "cqadup-android", "cqadup-english"]:
        raise ValueError("reserved systems or datasets differ from the registered batch")
    started = time.time()
    encoder = QueryEncoder(system, cfg)
    datasets = {}
    for dataset in DATASETS:
        doc_ids, doc_vectors = cache_for(system, dataset, repo=cfg.repo, verify=True)
        qids, qtexts, qrels, payload_hashes = load_payload(cfg, dataset)
        qvectors = encoder.encode(qtexts)
        if qvectors.shape[1] != doc_vectors.shape[1]:
            raise ValueError(f"{system}/{dataset}: query/doc dimensions differ")
        run = topk_ids_scores(qvectors, doc_vectors, doc_ids, k=cfg.topk,
                              chunk=int(cfg.extra.get("reserved_chunk", 50_000)),
                              device=str(cfg.extra.get("reserved_device", "cuda")), qids=qids)
        scores = {str(q): float(v) for q, v in per_query_ndcg(run, qrels).items()}
        if set(scores) != set(qids):
            raise ValueError(f"{system}/{dataset}: scorer omitted {len(set(qids) - set(scores))} queries")
        datasets[dataset] = {
            "scores": scores,
            "mean_ndcg10": float(np.mean(list(scores.values()))),
            "n_queries": len(scores),
            "payload_hashes": payload_hashes,
            "document_cache_manifest": str(doc_vectors.manifest_path.relative_to(cfg.repo)),
            "document_cache_manifest_sha256": sha_file(doc_vectors.manifest_path),
        }
        print(f"[reserved13] {system}: {dataset} complete ({len(scores):,} queries)", flush=True)
    return {
        "status": "COMPLETE",
        "system": system,
        "datasets": datasets,
        "query_encoder": encoder.identity,
        "document_encoder": DOCUMENT_ENCODERS[system],
        "document_compute_dtype": "fp32 on CUDA; normalized vectors stored fp16",
        "query_compute_dtype": encoder.identity["compute_dtype"],
        "elapsed_seconds": time.time() - started,
        "transaction": dict(cfg.extra.get("reserved_identity") or {}),
    }


def validate_output(record, cfg, system):
    """Authenticate an atomic per-system output before accepting it on continuation.

    A completed file is the only state the reserved transaction resumes past.  Presence alone is
    therefore insufficient: bind the file to this transaction, the frozen payload hashes and the
    exact document-cache manifest that the scorer verified.
    """
    import access13 as A

    if system not in SYSTEMS or record.get("status") != "COMPLETE" \
            or record.get("system") != system:
        raise ValueError(f"{system}: saved output has the wrong system or status")
    identity = dict(cfg.extra.get("reserved_identity") or {})
    if not identity or record.get("transaction") != identity:
        raise ValueError(f"{system}: saved output belongs to a different transaction")
    if record.get("document_encoder") != DOCUMENT_ENCODERS[system] \
            or record.get("document_compute_dtype") != \
            "fp32 on CUDA; normalized vectors stored fp16" \
            or record.get("query_compute_dtype") != "fp32":
        raise ValueError(f"{system}: saved encoder identity or dtype changed")
    datasets = record.get("datasets") or {}
    if set(datasets) != set(DATASETS):
        raise ValueError(f"{system}: saved output has incomplete datasets")
    frozen = json.loads(Path(cfg.manifest_path).read_text())["m7_untouched_final"]
    for dataset in DATASETS:
        row = datasets[dataset]
        scores = row.get("scores")
        if not isinstance(scores, dict) or not scores or any(not isinstance(q, str) for q in scores):
            raise ValueError(f"{system}/{dataset}: invalid saved per-query scores")
        values = np.asarray(list(scores.values()), dtype=np.float64)
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
        expected_cache = Path("work") / "m13-reserved-enc" / system / dataset / "manifest.json"
        if row.get("document_cache_manifest") != str(expected_cache):
            raise ValueError(f"{system}/{dataset}: saved document-cache path changed")
        cache = Path(cfg.repo) / expected_cache
        if not cache.is_file() or sha_file(cache) != row.get("document_cache_manifest_sha256"):
            raise ValueError(f"{system}/{dataset}: saved document-cache manifest changed")
    return record


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
    """Derive the registered zero-alpha report from three complete atomic outputs."""
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
        "B": B, "seed": seed, "quantile_method": "inverted_cdf",
        "draw_plan_sha256": plan_sha,
        "systems": {system: {dataset: outputs[system]["datasets"][dataset]["mean_ndcg10"]
                             for dataset in DATASETS} for system in SYSTEMS},
        "contrasts": result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-models", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.preflight_models:
        parser.error("--preflight-models is required")
    preflight_models(device=args.device)
