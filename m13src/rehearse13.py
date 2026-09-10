"""M13 — the synthetic rehearsal for the six-set scoring transaction.

Builds a complete fake world under `work/m13rehearse/` and runs the PRODUCTION executor against it,
unmodified: six datasets named exactly as `partitions.all6`, a frozen-eval payload and an
`eval_manifest.json` in the real schema, a comparator in `results/perquery.json`'s schema, a real
`teacher.encode_cached` document cache (written with `teacher.cache_key`, so `verify=True`
authenticates it exactly as it will on the real run), a `hf-internal-testing/tiny-random-BertModel`
backed `Nano10` student, and a bare `git init` origin so BEGIN, the annotated tag and both pushes
are real git operations — against the fixture's own repository, never this one.

Everything the executor touches comes from one `access13.Config`. Nothing here writes to
`results/`, to the real registry, or to the real origin.

Two fixture properties worth stating, because they are constructions and not measurements:

* **The qrels are generated from the student's own ranking** (top-1 plus one random document), so
  the tiny random student scores near 1.0 and the tiny random anchor scores near chance. That is
  what makes the sequence reach C2b and the reserved branch fire — the point is to exercise the
  machinery, not to produce a quality number.
* **The frozen bge-small comparator row IS the fixture anchor's own scores**, so the bridge's
  dataset-mean delta is ~0 and the gate exercises its arithmetic rather than a random offset. The
  tests perturb that row to drive the 0.002 / 0.004 cases.

  .venv/bin/python m13src/rehearse13.py --run          # build and run end to end
  .venv/bin/python m13src/rehearse13.py --recover       # decisions from the persisted scores
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# OFFLINE, before `transformers` is imported anywhere below (review B12). The rehearsal must not
# depend on the network — not for the tiny backbone, not for a tokenizer, not for a dataset — and
# a fixture that quietly downloaded something would be rehearsing a different code path than the
# cloud box will run. Every `from_pretrained` below also passes `local_files_only=True`.
for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_k, "1")

import numpy as np

import access13
import score13
from access13 import Config, sha_json, sha256_file

REPO = access13.REPO
DEFAULT_ROOT = REPO / "work" / "m13rehearse"
TINY_HUB = "hf-internal-testing/tiny-random-BertModel"
TINY_KEY = "tiny-random"
# The hub cache on this box holds only tiny-random-BertModel's config.json — no weights and no
# tokenizer — and the harness runs offline. So the fixture backbone is that EXACT config,
# instantiated locally with a seeded initialisation and a small WordPiece vocab written beside it.
# Same architecture, same size, no network.
VOCAB = (["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
         + ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
            "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma"]
         + ["document", "query", "represent", "this", "sentence", "for", "searching", "relevant",
            "passages", "scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec", "covid", "-"]
         + [str(i) for i in range(10)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
         + ["##" + chr(c) for c in range(ord("a"), ord("z") + 1)]
         + ["##" + str(i) for i in range(10)])


def materialize_tiny(root, seed=0):
    """-> a local model directory with tiny-random-BertModel's config, seeded weights and a
    tokenizer. Idempotent; rebuilt if absent so `--recover` on an existing fixture still works."""
    import torch
    from transformers import AutoConfig, BertModel
    d = Path(root) / "repo" / "work" / "tiny-bert"
    if (d / "config.json").exists() and (d / "vocab.txt").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    conf = AutoConfig.from_pretrained(TINY_HUB, local_files_only=True)   # cached; no network
    conf.vocab_size = len(VOCAB)
    torch.manual_seed(seed)
    BertModel(conf).save_pretrained(d)
    (d / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
    (d / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "BertTokenizer", "do_lower_case": True, "model_max_length": 512,
         "unk_token": "[UNK]", "sep_token": "[SEP]", "pad_token": "[PAD]", "cls_token": "[CLS]",
         "mask_token": "[MASK]"}, indent=1))
    return d


# ------------------------------------------------------------------ the tiny models

def register_tiny_student(model_dir):
    """Teach `nano10` about the tiny backbone. Module globals, not a source edit: `m10src` is
    owned elsewhere and imports from it must keep working unchanged."""
    import nano10
    nano10.REPOS[TINY_KEY] = str(model_dir)
    nano10.LAYERS.setdefault(TINY_KEY, {1: (5,), 3: (5, 3, 1), 4: (5, 4, 3, 1)})
    return nano10


class TinyAnchor:
    """The bridge anchor's `encode(texts) -> ndarray` contract over the tiny backbone. Same shape
    as `score13.BgeSmallAnchor`, small enough to run six corpora on a CPU in seconds."""

    def __init__(self, model_dir):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.m = AutoModel.from_pretrained(str(model_dir), local_files_only=True).eval()

    def encode(self, texts, batch_size=128):
        torch = self.torch
        texts = list(texts)
        out = np.empty((len(texts), self.m.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                b = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                             max_length=64, return_tensors="pt")
                h = self.m(**b).last_hidden_state[:, 0]
                out[i:i + batch_size] = torch.nn.functional.normalize(
                    h.float(), dim=-1).cpu().numpy()
        return out


def fixture_reserved_encoder(cfg, conf, system, rows_candidate):
    """The reserved stage's injected encoder, SYNTHETIC and fixture-only.

    Production has none: the reserved four's stella vectors do not exist, so `score13.reserved_batch`
    refuses and the run ends INCOMPLETE_RESERVED (review B13). This stand-in exists so the rehearsal
    exercises the per-system atomic write and the COMPLETE end status too; it opens no payload,
    reads no reserved dataset and produces no number that means anything. A test drops it to get
    the production refusal back.
    """
    return {"system": system, "datasets": list(conf["reserved"]["datasets"]),
            "n_candidate_datasets": len(rows_candidate), "written": access13.utcnow(),
            "_fixture": "SYNTHETIC rehearsal record. No reserved payload was opened and no "
                        "reserved score was computed; this is machinery, not a measurement."}


def _pin_fixture_teacher(cfg):
    """The fixture's document cache is written with whatever `M7_ENCODER` selects on this box, so
    the executor's pinned stella identity (review A2) is overridden to match it — and only here."""
    import teacher
    cfg.teacher_model_id, cfg.teacher_revision = teacher.TEACHER, teacher.TEACHER_REV
    cfg.teacher_dtype = "fp32"
    return cfg


# ------------------------------------------------------------------ the fixture

def _texts(rng, n, kind, ds):
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota",
             "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho", "sigma"]
    return [f"{ds} {kind} {i} " + " ".join(rng.choice(vocab, size=8)) for i in range(n)]


def _write_doc_cache(cfg, ds, doc_texts, vecs):
    """Write the encode cache `teacher.encode_cached(verify=True)` will authenticate, at the key
    `teacher.cache_key` derives — the real layout, so nothing about the cache path is stubbed."""
    import torch
    import teacher
    teacher.ENC = Path(cfg.enc_root)
    key, blob = teacher.cache_key(cfg.doc_cache_fmt.format(ds=ds), "", 512, teacher.TEACHER,
                                 teacher.TEACHER_REV, teacher.sha_texts(doc_texts), torch.float32)
    d = Path(cfg.enc_root) / key
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(blob)
    p = d / "shard_00000.npy"
    np.save(p, vecs.astype(np.float16))
    (d / "shards.json").write_text(json.dumps({"shards": {"00000": {
        "bytes": p.stat().st_size, "sha256": teacher.sha_file(p), "rows": int(len(vecs)),
        "shard_size": teacher.SHARD, "trusted_on_first_use": False}}}, indent=1, sort_keys=True))
    return d


def _git(root, *a, check=True):
    r = subprocess.run(("git",) + a, cwd=root, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"fixture git failed: {' '.join(a)}\n{r.stderr}")
    return r.stdout.strip()


def build(root=DEFAULT_ROOT, n_docs=200, n_queries=30, seed=0, dim=1024, clean=True) -> Config:
    root = Path(root)
    if clean and root.exists():
        shutil.rmtree(root)
    repo, origin = root / "repo", root / "origin.git"
    for d in ("m10", "results/frozen_eval", "work/enc", "work/corpus"):
        (repo / d).mkdir(parents=True, exist_ok=True)
    (repo / ".gitignore").write_text("work/\n")

    real_registry = json.loads((REPO / "m10" / "final_run_registry.json").read_text())
    conf = json.loads(json.dumps(real_registry))          # a COPY; the real file is never touched
    conf["_fixture"] = ("SYNTHETIC REHEARSAL COPY built by m13src/rehearse13.py. Partitions, "
                        "conjuncts, bars, sequence, bootstrap and sign-flip constants are the "
                        "registered ones, byte-identical. Only ratified_by_owner, origin_url and "
                        "comparator_source.sha256 point at the fixture.")
    conf["ratified_by_owner"] = True
    conf["origin_url"] = str(origin)
    datasets = list(conf["partitions"]["all6"])

    cfg = Config(
        repo=repo,
        registry_path=repo / "m10" / "final_run_registry.json",
        result_path=repo / "results" / "m10_final_run.json",
        scores_dir=repo / "results" / "m10_final_scores",
        ledger_path=repo / "m10" / "LEDGER.md",
        freeze_path=repo / "m10" / "FREEZE.json",
        perquery_path=repo / "results" / "perquery.json",
        manifest_path=repo / "results" / "eval_manifest.json",
        frozen_eval_dir=repo / "results" / "frozen_eval",
        enc_root=repo / "work" / "enc",
        lock_path=repo / "work" / "m10final.lock",
        origin_url=str(origin),
        min_free_gb=0.05,
        serving_parity_artifacts=("results/m10_student_parity_box.json",),
        corpus_reader=None,          # set below, once the corpus exists
        load_anchor=None,            # set below, once the tiny backbone exists
        reserved_encoder=fixture_reserved_encoder,
        topk=50, chunk=10_000,
    )
    _pin_fixture_teacher(cfg)
    (repo / "results" / "m10_student_parity_box.json").write_text(
        json.dumps({"_fixture": "serving-parity stand-in"}, indent=1))
    (repo / "m10" / "LEDGER.md").write_text("# Fixture ledger\n")

    # --- the student, frozen ----------------------------------------------------------------
    import torch
    tiny_dir = materialize_tiny(root, seed=seed)
    nano10 = register_tiny_student(tiny_dir)
    torch.manual_seed(seed)
    model = nano10.Nano10(TINY_KEY, n_layers=3, head="linear", out_dim=dim, max_seq=64)
    ckpt = repo / "work" / "student.pt"
    torch.save(model.state_dict(), ckpt)
    freeze = {"checkpoint": "work/student.pt", "checkpoint_sha256": sha256_file(ckpt),
              "student_key": TINY_KEY, "n_layers": 3, "head": "linear", "max_seq": 64,
              "out_dim": dim, "query_prefix": "", "system": "nano-dense"}
    (repo / "m10" / "FREEZE.json").write_text(json.dumps(freeze, indent=1))
    cfg.load_student = lambda fz: score13.Nano10Student(fz, repo=repo)
    cfg.load_anchor = lambda _d=tiny_dir: TinyAnchor(_d)

    # --- corpora, queries, generated qrels ---------------------------------------------------
    rng = np.random.default_rng(seed)
    student = cfg.load_student(freeze)
    anchor = cfg.load_anchor()
    manifest, per_ds_ref, per_ds_anchor = {"datasets": {}}, {}, {}
    for ds in datasets:
        doc_ids = [f"d{i}" for i in range(n_docs)]
        doc_texts = _texts(rng, n_docs, "document", ds)
        q_ids = [f"q{i}" for i in range(n_queries)]
        q_texts = _texts(rng, n_queries, "query", ds)
        (repo / "work" / "corpus" / f"{ds}.json").write_text(
            json.dumps({"doc_ids": doc_ids, "doc_texts": doc_texts}))

        dv = rng.normal(size=(n_docs, dim)).astype(np.float32)
        dv /= np.linalg.norm(dv, axis=1, keepdims=True)
        _write_doc_cache(cfg, ds, doc_texts, dv)

        qv = student.encode(q_texts)
        sims = qv @ dv.T
        qrels = {}
        for i, q in enumerate(q_ids):
            top = int(np.argmax(sims[i]))
            other = int(rng.integers(n_docs))
            qrels[q] = {doc_ids[top]: 2}
            if other != top:
                qrels[q][doc_ids[other]] = 1
        (repo / "results" / "frozen_eval" / f"{ds}.json").write_text(
            json.dumps({"queries": dict(zip(q_ids, q_texts)), "qrels": qrels}))
        manifest["datasets"][ds] = {
            "n_docs": n_docs, "n_queries": n_queries,
            "corpus_ids_sha256": sha_json(doc_ids), "corpus_text_sha256": sha_json(doc_texts),
            "qids_sha256": sha_json(sorted(q_ids)),
            "qtexts_sha256": sha_json([dict(zip(q_ids, q_texts))[q] for q in sorted(q_ids)]),
            "qrels_sha256": sha_json(qrels)}

        # the reference rows, scored exactly the way the executor will score them
        sorted_q = sorted(q_ids)
        order = [q_ids.index(q) for q in sorted_q]
        per_ds_ref[ds] = score13.exact_scores(cfg, qv[order], dv, doc_ids, sorted_q, qrels)
        adv, aqv = anchor.encode(doc_texts), anchor.encode(
            [score13.ANCHOR_PREFIX + q_texts[i] for i in order])
        per_ds_anchor[ds] = score13.exact_scores(cfg, aqv, adv, doc_ids, sorted_q, qrels)
    (repo / "results" / "eval_manifest.json").write_text(json.dumps(manifest, indent=1))

    # --- the comparator, in perquery.json's schema -------------------------------------------
    pq = {"_note": "SYNTHETIC fixture comparator in results/perquery.json's schema. Not a "
                   "measurement of anything.", "datasets": {}}
    for ds in datasets:
        qids = sorted(per_ds_ref[ds])
        nano = np.array([per_ds_ref[ds][q] for q in qids])
        pq["datasets"][ds] = {"qids": qids, "systems": {
            # the bridge anchor: the fixture anchor's OWN scores, so the bridge delta is ~0
            "bge-small-en-v1.5": [per_ds_anchor[ds][q] for q in qids],
            # an M9/leaf-style row a hair below the candidate, and a zero-style lookup row
            "leaf-ir-asym": list(np.clip(nano - 0.02, 0.0, 1.0)),
            "lr-dense-websearch": list(np.clip(nano * 0.5, 0.0, 1.0))}}
    cfg.perquery_path.write_text(json.dumps(pq, indent=1))

    conf["comparator_source"]["sha256"] = sha256_file(cfg.perquery_path)
    cfg.registry_path.write_text(json.dumps(conf, indent=1))
    cfg.corpus_reader = lambda ds: (
        lambda b: (b["doc_ids"], b["doc_texts"]))(
        json.loads((repo / "work" / "corpus" / f"{ds}.json").read_text()))

    # --- a real bare origin, a real commit, a real upstream -----------------------------------
    origin.mkdir(parents=True, exist_ok=True)
    _git(origin, "init", "--bare", "-q")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "rehearsal@example.invalid")
    _git(repo, "config", "user.name", "M13 rehearsal")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture: the synthetic six")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "-u", "origin", "main")
    print(f"[rehearse13] fixture built at {root} ({len(datasets)} datasets, {n_docs} docs / "
          f"{n_queries} queries each); origin {origin}")
    return cfg


def main(argv=None):
    """`--run` (the default) rebuilds the fixture from scratch and runs the WHOLE transaction, so
    it is repeatable: one command, any number of times, on a world that never touches `results/`.
    Ruling R14 removed post-tag continuation, which makes this rehearsal the reliability story."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--run", action="store_true",
                    help="build a fresh fixture and run the whole transaction (the default)")
    ap.add_argument("--recover", action="store_true", help="build nothing; --recover an existing "
                                                           "fixture run")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--build-only", action="store_true", help="build the fixture and stop")
    a = ap.parse_args(argv)
    if a.recover:
        cfg = build(a.root, clean=False) if not (Path(a.root) / "repo").exists() else _reopen(a.root)
        code = score13.run(cfg, recover=True)
    else:
        cfg = build(a.root)
        if a.build_only:
            return 0
        code = score13.run(cfg, preflight_only=a.preflight_only)
    print(f"[rehearse13] exit {code} "
          f"({'OK' if code == 0 else 'nonzero — see the report above'})")
    return code


def _reopen(root):
    """Re-derive the Config for an existing fixture (used by `--recover` after a simulated crash)."""
    root = Path(root)
    repo = root / "repo"
    freeze = json.loads((repo / "m10" / "FREEZE.json").read_text())
    tiny_dir = materialize_tiny(root)
    register_tiny_student(tiny_dir)
    cfg = Config(
        repo=repo, registry_path=repo / "m10" / "final_run_registry.json",
        result_path=repo / "results" / "m10_final_run.json",
        scores_dir=repo / "results" / "m10_final_scores",
        ledger_path=repo / "m10" / "LEDGER.md", freeze_path=repo / "m10" / "FREEZE.json",
        perquery_path=repo / "results" / "perquery.json",
        manifest_path=repo / "results" / "eval_manifest.json",
        frozen_eval_dir=repo / "results" / "frozen_eval",
        enc_root=repo / "work" / "enc", lock_path=repo / "work" / "m10final.lock",
        origin_url=str(root / "origin.git"), min_free_gb=0.05,
        load_anchor=lambda _d=tiny_dir: TinyAnchor(_d), topk=50, chunk=10_000,
        reserved_encoder=fixture_reserved_encoder,
        corpus_reader=lambda ds: (lambda b: (b["doc_ids"], b["doc_texts"]))(
            json.loads((repo / "work" / "corpus" / f"{ds}.json").read_text())),
    )
    _pin_fixture_teacher(cfg)
    cfg.load_student = lambda fz, _r=repo: score13.Nano10Student(fz, repo=_r)
    _ = freeze
    return cfg


if __name__ == "__main__":
    sys.exit(main())
