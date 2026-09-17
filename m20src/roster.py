"""M20's eight-system roster: query towers, BM25, DBSF fusion, run persistence.

Registered in `m20/REGISTRATION.md` and `m20/beir15_registry.json` under owner rulings R20 and
R22, pre-observation.  This module is shared by the protected reserved transaction
(`m13src/reserved_support.py`) and the unprotected BEIR-15 pass (`m20src/beir15.py`) so the two
cannot drift into scoring the same system two different ways.

**It contains no path to a protected payload and must never grow one.**  Protected loading stays
in `m13src/reserved_support.py`, which is the module that holds the allowlist capability.  This
module only ever sees texts, ids and qrels that its caller has already authenticated.

Three of the eight systems share ONE Stella document index -- Nano, Zero and the Stella query
tower -- which is the whole premise of the project, so `DOC_TOWER` maps system to tower and the
pre-encode is per TOWER, never per system.  Two systems are derived: they re-read persisted
top-100 runs rather than re-encoding anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in ("m7src", "m10src", "m12src", "m13src", "m11/release"):
    if str(REPO / _p) not in sys.path:
        sys.path.insert(0, str(REPO / _p))

# ------------------------------------------------------------------ the registered roster

SYSTEMS = (
    "nano-dense",
    "zero-dense",
    "stella-query",
    "bge-small-en-v1.5",
    "leaf-ir-asym",
    "bm25",
    "zero+bm25 dbsf@100",
    "nano+bm25 dbsf@100",
)
DENSE_SYSTEMS = ("nano-dense", "zero-dense", "stella-query", "bge-small-en-v1.5", "leaf-ir-asym")
LEXICAL_SYSTEMS = ("bm25",)
# Derived system -> (dense input, lexical input).  Nothing is encoded for these.
DERIVED_SYSTEMS = {
    "zero+bm25 dbsf@100": ("zero-dense", "bm25"),
    "nano+bm25 dbsf@100": ("nano-dense", "bm25"),
}
# The three systems whose top-100 runs a derived system consumes, and which therefore persist one.
RUN_PRODUCERS = ("nano-dense", "zero-dense", "bm25")

STELLA_REVISION = "ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
ARCTIC_REVISION = "e58a8f756156a1293d763f17e3aae643474e9b8a"
LEAF_QUERY_REVISION = "4262131b32c3182bd06e67e92ae69d7bd66e0c5c"
ZERO_HUB_REPO = "DylanCouzon/constella-zero"
ZERO_HUB_REVISION = "ebee6ea94999f26182505548ffc5df035e727351"
ZERO_VARIANT = "int8"
BGE_PREFIX = "Represent this sentence for searching relevant passages: "
# The literal registered Stella s2p prompt.  Kept here as the one executable copy; the registration
# pins the same bytes and `assert_registered_identities` proves the two agree before a run.
STELLA_QUERY_PROMPT = ("Instruct: Given a web search query, retrieve relevant passages that "
                       "answer the query.\nQuery: ")

# Document towers.  Systems, not towers, are what the registry lists; the pre-encode is per tower.
TOWERS = {
    "stella-400M-v5": {"model": "NovaSearch/stella_en_400M_v5", "revision": STELLA_REVISION,
                       "backend": "stella", "dim": 1024, "doc_prefix": ""},
    "bge-small-en-v1.5": {"model": "BAAI/bge-small-en-v1.5", "revision": BGE_REVISION,
                          "backend": "sentence-transformers", "dim": 384, "doc_prefix": ""},
    "arctic-m-v1.5": {"model": "Snowflake/snowflake-arctic-embed-m-v1.5",
                      "revision": ARCTIC_REVISION, "backend": "sentence-transformers",
                      "dim": 768, "doc_prefix": ""},
}
DOC_TOWER = {
    "nano-dense": "stella-400M-v5",
    "zero-dense": "stella-400M-v5",
    "stella-query": "stella-400M-v5",
    "bge-small-en-v1.5": "bge-small-en-v1.5",
    "leaf-ir-asym": "arctic-m-v1.5",
}
# The cache directory each tower's shards live under.  `stella-400M-v5` keeps the historical
# `nano-dense` directory name so an existing or resumed M13 pre-encode is not orphaned by a rename.
TOWER_DIR = {"stella-400M-v5": "nano-dense", "bge-small-en-v1.5": "bge-small-en-v1.5",
             "arctic-m-v1.5": "leaf-ir-asym"}

DENSE_DEPTH = 100           # registered dense retrieval depth; see the registry search note
FUSION_DEPTH = 100          # m12src registered prefetch depth
DOC_COMPUTE_NOTE = "fp32 on CUDA; normalized vectors stored fp16"


def sha_file(path, block=8 << 20):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def registration(repo=REPO):
    return json.loads((Path(repo) / "m20" / "beir15_registry.json").read_text())


def assert_registered_identities(repo=REPO):
    """Executable code and the pushed registration must name the same models and strings.

    A registration that pins one prompt while the executor sends another is worse than no
    registration at all, so this runs before every stage rather than being trusted by reading.
    """
    reg = registration(repo)
    problems = []
    if [s["key"] for s in reg["systems"]] != list(SYSTEMS):
        problems.append("registered system order differs from the executor")
    by_key = {s["key"]: s for s in reg["systems"]}
    if by_key["stella-query"]["query_prompt"] != STELLA_QUERY_PROMPT:
        problems.append("registered Stella query prompt differs from the executor")
    if by_key["bge-small-en-v1.5"]["query_prompt"] != BGE_PREFIX:
        problems.append("registered bge query prompt differs from the executor")
    if by_key["zero-dense"]["query_hub_revision"] != ZERO_HUB_REVISION \
            or by_key["zero-dense"]["query_variant"] != ZERO_VARIANT:
        problems.append("registered constella-zero revision or variant differs from the executor")
    for key, tower in DOC_TOWER.items():
        if by_key[key]["document_tower"] != tower:
            problems.append(f"{key}: registered document tower differs from the executor")
    for name, tower in TOWERS.items():
        row = reg["document_towers"][name]
        if row["repo"] != tower["model"] or row["revision"] != tower["revision"] \
                or row["dim"] != tower["dim"]:
            problems.append(f"{name}: registered tower identity differs from the executor")
    bm = by_key["bm25"]
    bm_versions = bm25_versions
    import fusion

    if (bm["k1"], bm["b"], bm["method"], bm["retrieval_depth"]) != (
            fusion.BM25_CONFIG["k1"], fusion.BM25_CONFIG["b"], fusion.BM25_CONFIG["method"],
            fusion.DEPTH):
        problems.append("registered BM25 parameters differ from m7src/fusion.py")
    if bm_versions() != bm["package_versions_pinned_at_registration"]:
        problems.append(f"installed BM25 stack {bm_versions()} differs from the registered "
                        f"{bm['package_versions_pinned_at_registration']}")
    for key in DERIVED_SYSTEMS:
        if by_key[key]["prefetch_depth"] != FUSION_DEPTH:
            problems.append(f"{key}: registered prefetch depth differs from the executor")
    if reg["search"]["dense_depth"] != DENSE_DEPTH:
        problems.append("registered dense retrieval depth differs from the executor")
    if problems:
        raise ValueError("registration and executor disagree: " + "; ".join(problems))
    return True


# ------------------------------------------------------------------ query towers

class QueryEncoder:
    """One query tower, at its registered prompt, precision and device.

    Nano stays on CPU: its serving method deliberately enters bf16 autocast on CUDA and the
    inherited M8 confirmatory contract is fp32 compute, so the existing method runs on CPU rather
    than this module forking its math.  Zero is numpy and has no device.  The three
    SentenceTransformer/Transformers towers run fp32 on the GPU with TF32 matmul disabled.
    """

    def __init__(self, system, cfg=None, device="cuda", repo=REPO):
        if system not in DENSE_SYSTEMS:
            raise ValueError(f"{system!r} is not a dense system")
        self.system = system
        self.repo = Path(repo)
        self.device = "cpu" if system == "nano-dense" else device
        self.model = None
        getattr(self, f"_init_{system.replace('-', '_').replace('.', '_')}")(cfg)

    # -- Nano ---------------------------------------------------------------------------
    def _init_nano_dense(self, cfg):
        import score13 as S

        if cfg is None:
            raise ValueError("nano-dense needs the transaction config to locate its freeze")
        freeze = S._freeze_blob(cfg)
        self.model = S.Nano10Student(freeze, device=self.device, repo=cfg.repo)
        dependency = nano_dependency_identity(self.model.model)
        if dependency != registered_nano_dependency(cfg.repo):
            raise ValueError("constructed Nano tokenizer/backbone dependency changed")
        self.dim = 1024
        self.identity = {"query_model": "M13 frozen nano", "revision": freeze["_sha256"],
                         "device": self.device, "compute_dtype": "fp32",
                         "dependency": dependency}

    # -- Zero ---------------------------------------------------------------------------
    def _init_zero_dense(self, cfg):
        from huggingface_hub import snapshot_download
        from zero_encoder import ZeroQueryEncoder

        reg = {s["key"]: s for s in registration(self.repo)["systems"]}["zero-dense"]
        local = Path(snapshot_download(ZERO_HUB_REPO, revision=ZERO_HUB_REVISION,
                                       allow_patterns=list(reg["required_files"])))
        got = {name: sha_file(local / name) for name in reg["required_files"]}
        bad = [name for name in got if got[name] != reg["query_file_sha256"][name]]
        if bad:
            raise ValueError(f"constella-zero file hashes changed at the pinned revision: {bad}")
        if got["model.npz"] != reg["frozen_table_sha256_must_equal"]:
            raise ValueError("published constella-zero table is not the frozen M7 table")
        self.model = ZeroQueryEncoder(local, variant=ZERO_VARIANT)
        self.dim = self.model.dim
        self.identity = {"query_model": ZERO_HUB_REPO, "revision": ZERO_HUB_REVISION,
                         "variant": ZERO_VARIANT, "device": "cpu-numpy",
                         "compute_dtype": "fp32", "file_sha256": got}

    # -- the symmetric teacher -----------------------------------------------------------
    def _init_stella_query(self, cfg):
        import torch

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        self.dim = 1024
        self.identity = {"query_model": TOWERS["stella-400M-v5"]["model"],
                         "revision": STELLA_REVISION, "device": self.device,
                         "compute_dtype": "fp32", "prompt": STELLA_QUERY_PROMPT,
                         "max_length": 512, "batch_tokens": 32768}

    # -- the two SentenceTransformer comparators -----------------------------------------
    def _init_bge_small_en_v1_5(self, cfg):
        self._init_sentence_transformer("BAAI/bge-small-en-v1.5", BGE_REVISION, 384)

    def _init_leaf_ir_asym(self, cfg):
        self._init_sentence_transformer("MongoDB/mdbr-leaf-ir", LEAF_QUERY_REVISION, 768)

    def _init_sentence_transformer(self, repo, revision, dim):
        import torch
        from sentence_transformers import SentenceTransformer

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        self.model = SentenceTransformer(repo, revision=revision, device=self.device,
                                         model_kwargs={"dtype": torch.float32})
        self.model.max_seq_length = 512
        self.dim = dim
        self.identity = {"query_model": repo, "revision": revision, "device": self.device,
                         "compute_dtype": "fp32"}

    # -- encode --------------------------------------------------------------------------
    def encode(self, texts):
        values = list(texts)
        if self.system == "nano-dense":
            out = self.model.encode(values)
        elif self.system == "zero-dense":
            out = self.model.encode(values)
        elif self.system == "stella-query":
            import torch
            import teacher

            out = teacher.encode(values, prefix=STELLA_QUERY_PROMPT, max_length=512,
                                 batch_tokens=32768, model_id=TOWERS["stella-400M-v5"]["model"],
                                 revision=STELLA_REVISION, dtype=torch.float32,
                                 device=self.device, verbose=False)
        elif self.system == "leaf-ir-asym":
            # The pinned card defines the asymmetric query route through its named `query` prompt.
            # Use that interface so the model's own pinned config, not a duplicated free-form
            # string, stays part of the execution path.
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
        norms = np.linalg.norm(out, axis=1)
        if len(values) and np.max(np.abs(norms - 1.0)) > 2e-3:
            raise ValueError(f"{self.system}: query vectors are not unit-normalized")
        return out

    def release(self):
        import gc

        self.model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


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


# ------------------------------------------------------------------ BM25 and DBSF

def bm25_versions():
    """The installed versions of the two packages that DEFINE the lexical function.

    `m7src/fusion.py` puts these in its cache key, which is how a version change is normally made
    visible -- but only when a cache path is supplied, and M20 supplies none.  So on M20's paths
    nothing would have recorded that `bm25s` or `PyStemmer` had changed, and a different stemmer
    or scoring build would have moved every lexical and fused number silently.  Recording them in
    each BM25 row closes that.
    """
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for package in ("bm25s", "PyStemmer"):
        try:
            out[package] = version(package)
        except PackageNotFoundError as error:
            raise RuntimeError(f"BM25 REFUSED: no installed distribution metadata for "
                               f"{package!r}, so the lexical function cannot be pinned into the "
                               f"result ({error})")
    return out


def assert_registered_bm25_versions(repo=REPO):
    """Refuse if the installed lexical stack differs from the registered one.

    Registered at `m20/beir15_registry.json` systems[bm25].package_versions_pinned_at_registration.
    A change here is disclosed, never silently inherited.
    """
    registered = {s["key"]: s for s in registration(repo)["systems"]}["bm25"][
        "package_versions_pinned_at_registration"]
    got = bm25_versions()
    if got != registered:
        raise ValueError(f"installed BM25 stack {got} differs from the registered {registered}; "
                         f"this changes the lexical function and must be re-registered, not "
                         f"silently inherited")
    return got


def bm25_run(doc_ids, doc_texts, q_ids, q_texts, cache_path=None):
    """M7's frozen lexical function, unchanged: bm25s lucene k1=1.2 b=0.75, depth 1000,
    zero-score rows and self-hits dropped.  Those two filters are part of the frozen function."""
    import fusion

    assert_registered_bm25_versions()
    return fusion.bm25_run(doc_ids, doc_texts, q_ids, q_texts, cache_path=cache_path)


def dbsf_at_depth(dense, lexical, depth=FUSION_DEPTH):
    """Qdrant DBSF over two prefetches, each truncated to `depth` FIRST.

    Truncation order is not cosmetic: DBSF normalizes by the mean and sample sd of whatever the
    prefetch returned, so fusing first and cutting after is a different function.
    """
    import qfusion

    return qfusion.dbsf([qfusion.truncate(dense, depth), qfusion.truncate(lexical, depth)])


def truncate(run, depth=FUSION_DEPTH):
    import qfusion

    return qfusion.truncate(run, depth)


def per_query_ndcg10(run, qrels):
    from evalkit import per_query_ndcg

    return {str(q): float(v) for q, v in per_query_ndcg(run, qrels).items()}


# ------------------------------------------------------------------ persisted top-100 runs

def slug(system):
    """Filesystem-safe name for a system. Two of the eight carry spaces and a `+`."""
    return system.replace("/", "_").replace(" ", "_")


def run_path(root, stage, system, dataset):
    return Path(root) / stage / slug(system) / f"{dataset}.npz"


def save_run(path, run, qids):
    """Persist one top-`FUSION_DEPTH` run as a hash-recorded npz, and return its sha256.

    Runs live under `work/`, not in git: they are large, they are reproducible from the same
    shards, and the document vectors they depend on live there too.  The hash goes into the
    system's atomic output, so a derived system can prove it fused the run its input actually
    produced rather than whatever is on disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    order = [str(q) for q in qids]
    flat_ids, flat_scores, counts = [], [], np.zeros(len(order), dtype=np.int64)
    for i, q in enumerate(order):
        pairs = sorted(run.get(q, {}).items(), key=lambda kv: -kv[1])
        if len(pairs) > FUSION_DEPTH:
            raise ValueError(f"{path}: run for {q!r} is deeper than {FUSION_DEPTH}; truncate first")
        counts[i] = len(pairs)
        for d, s in pairs:
            flat_ids.append(str(d))
            flat_scores.append(s)
    tmp = path.with_suffix(".tmp.npz")
    # Fixed-width unicode, not object arrays: no pickle on the persistence path.
    np.savez_compressed(tmp,
                        qids=np.asarray(order, dtype=np.str_),
                        doc_ids=np.asarray(flat_ids, dtype=np.str_),
                        scores=np.asarray(flat_scores, dtype=np.float32),
                        counts=counts, depth=np.asarray(FUSION_DEPTH, dtype=np.int64))
    tmp.replace(path)
    return sha_file(path)


def load_run(path, expect_sha256=None):
    path = Path(path)
    if expect_sha256 is not None and sha_file(path) != expect_sha256:
        raise ValueError(f"{path}: persisted run hash changed")
    with np.load(path, allow_pickle=False) as blob:
        qids = [str(q) for q in blob["qids"]]
        ids, scores, counts = blob["doc_ids"], blob["scores"], blob["counts"]
        if int(blob["depth"]) != FUSION_DEPTH:
            raise ValueError(f"{path}: persisted run depth {int(blob['depth'])} != {FUSION_DEPTH}")
    if int(counts.sum()) != len(ids) or len(ids) != len(scores):
        raise ValueError(f"{path}: persisted run arrays disagree")
    run, pos = {}, 0
    for i, q in enumerate(qids):
        n = int(counts[i])
        run[q] = {str(ids[j]): float(scores[j]) for j in range(pos, pos + n)}
        pos += n
    return run, qids
