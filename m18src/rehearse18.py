"""One-command synthetic end-to-end rehearsal of the M18 production modules.

Creates a disposable pinned git/API snapshot, parses it, seals five-stratum splits, builds
BM25/dense fixtures and a shared candidate cache, pauses/resumes the fixed long-schedule trainer,
proves resumed/uninterrupted equivalence and inherited-row immutability, exports an int8 bundle,
scores dense+DBSF, and proves a second confirmation claim is refused. No real evaluation data is
read and all fixture quality numbers are meaningless by construction.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

import cache
import corpus
import evaluate
import export
import preprocess
import protocol
import train
import vocab
from common import RESULTS, WORK, atomic_write_bytes, registry, sha_array, sha_json, write_json

DIM = 16
MARKER = ".m18_rehearsal"


def build_tokenizer():
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    words = ["how", "does", "what", "mean", "in", "search", "request", "with", "timeout",
             "error", "configure", "cluster", "replica", "setting", "difference", "between",
             "and", "status", "vector", "select", "neighbors", "maintainer", "answer", "use",
             "because", "this", "the", "solution", "works", "project", "memory"]
    pieces = ["q", "##dr", "##ant", "h", "##ns", "##w", "v", "##1", ".", "2", "3"]
    mapping = {x: i for i, x in enumerate(specials + words + pieces)}
    tok = Tokenizer(models.WordPiece(mapping, unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", mapping["[CLS]"]),
                                                  ("[SEP]", mapping["[SEP]"])])
    tok.no_padding()
    return tok


def unit(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def rehearsal_registry(commit):
    reg = copy.deepcopy(registry())
    reg["source"]["commit"] = commit
    reg["source"]["github_cutoff_utc"] = "2026-01-01T00:00:00Z"
    reg["corpus"]["chunk_tokens"] = 64
    reg["corpus"]["chunk_overlap_tokens"] = 8
    reg["split"].update({"development_per_stratum": 2, "confirmation_per_stratum": 1,
                         "minimum_development_per_available_stratum": 1,
                         "minimum_confirmation_per_available_stratum": 1})
    reg["vocabulary"].update({"min_distinct_documents": 1, "min_training_contexts": 1,
                              "max_added_rows": 32})
    reg["training"].update({"batch": 2, "schedule_steps": 4, "diagnostic_stop_after": 2,
                            "warmup_steps": 1, "checkpoint_steps": [0, 2, 4], "candidate_k": 4,
                            "candidate_mix_labeled": {"known_positive": 1, "teacher_top": 1,
                                                       "zero_v1_top": 1, "uniform": 1},
                            "candidate_mix_query_only": {"known_positive": 0, "teacher_top": 2,
                                                          "zero_v1_top": 1, "uniform": 1}})
    return reg


def _write_fixture_source(root):
    source = root / "source"
    repo = source / "repository"
    repo.mkdir(parents=True)
    (repo / "README.md").write_text(
        "# Qdrant project memory\n\nQdrant uses HNSW (Hierarchical Navigable Small World) "
        "for vector neighbors.\n\n# Configure qdrant\n\nUse cluster replica settings because the "
        "configuration controls durability.\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "M18 fixture"], cwd=repo, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            text=True, capture_output=True).stdout.strip()
    title_templates = {
        "concept_howto": "How does {topic} behavior select neighbors {i}",
        "error_troubleshooting": "Why does {topic} request fail with error {i}",
        "config_api_ops": "How configure {topic} operational setting number {i}",
        "exact_version_numeric": "Difference between {topic} v1.2 and v1.3 status {i}",
        "alias_jargon": "What does {topic} mean in qdrant project {i}",
    }
    issues, comments = [], []
    number = 1
    topics = ("vector", "payload", "replication", "quantization")
    for _stratum, template in title_templates.items():
        for i in range(4):
            created = f"2025-01-{number:02d}T00:00:00Z"
            issues.append({"id": 1000 + number, "node_id": f"ISSUE{number}", "number": number,
                           "title": template.format(i=i, topic=topics[i]), "body": "A distinct user question.",
                           "created_at": created, "updated_at": created,
                           "html_url": f"https://example.invalid/issues/{number}",
                           "user": {"login": "user", "type": "User"},
                           "author_association": "NONE", "labels": []})
            answered = created.replace("T00:", "T01:")
            comments.append({"id": 2000 + number, "node_id": f"COMMENT{number}",
                             "body": (f"Use the documented qdrant solution {number} because this "
                                      "maintainer answer explains the behavior, resolves the distinct "
                                      "case, and works with the registered project configuration."),
                             "created_at": answered, "updated_at": answered,
                             "issue_url": f"https://api.github.invalid/repos/qdrant/qdrant/issues/{number}",
                             "html_url": f"https://example.invalid/issues/{number}#comment",
                             "user": {"login": "maintainer", "type": "User"},
                             "author_association": "MEMBER"})
            number += 1
    for endpoint, rows in (("issues", issues), ("issue_comments", comments),
                           ("review_comments", []), ("issue_events", [])):
        p = source / "github" / endpoint / "pages" / "000001.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(rows))
    return source, commit


def _clear(root):
    if not root.exists():
        return
    if not root.is_dir() or (any(root.iterdir()) and not (root / MARKER).exists()):
        raise SystemExit(f"REFUSED: {root} is not a marked disposable M18 rehearsal directory")
    shutil.rmtree(root)


def build(root, seed=0, device="cpu", log=print):
    root = Path(root)
    _clear(root)
    root.mkdir(parents=True)
    (root / MARKER).write_text("Disposable output owned by m18src/rehearse18.py\n")
    t0 = time.time()
    source, commit = _write_fixture_source(root)
    reg = rehearsal_registry(commit)
    derived = root / "derived"
    cm = corpus.build(source, derived, registry_data=reg, fixture=True)
    pm = protocol.build(derived / "corpus.jsonl", derived / "protocol", registry_data=reg)
    docs = list(protocol._read_jsonl(derived / "corpus.jsonl"))
    doc_ids = [d["doc_id"] for d in docs]
    doc_texts = [d["text"] for d in docs]
    rng = np.random.default_rng(seed)
    doc_vecs = unit(rng.normal(size=(len(docs), DIM)))
    bm25 = evaluate.BM25Index(doc_ids, doc_texts)
    bm25.save(root / "bm25")

    train_rows = list(protocol._read_jsonl(derived / "protocol" / "training_queries.jsonl"))
    tok = build_tokenizer()
    base_n = tok.get_vocab_size(with_added_tokens=True)
    inherited = unit(rng.normal(size=(base_n, DIM))) * 0.4
    base_q = unit(np.stack([vocab._pool(np.asarray(tok.encode(q["text"]).ids), inherited)
                            for q in train_rows]))
    teacher_q_all = unit(0.8 * base_q + 0.2 * rng.normal(size=base_q.shape))
    residual = 1.0 - np.sum(base_q * teacher_q_all, axis=1)
    stats, single = vocab.discover(
        [{"text": q["text"], "qid": q["query_id"], "source_doc": q["source_doc"],
          "domain": q["stratum"]} for q in train_rows], residual, tok)
    selection = vocab.select(stats, reg, single_token=single)
    terms = [x["term"] for x in selection["terms"]]
    new_rows, _pieces = vocab.init_new_rows(terms, tok, inherited)
    tok, tok_sha, _ = vocab.extend_tokenizer(tok, terms)
    eligible = [i for i, q in enumerate(train_rows)
                if any(t >= base_n for t in preprocess.active_ids(q["text"], tok, "T0"))]
    if len(eligible) < reg["training"]["batch"]:
        raise AssertionError(f"fixture produced only {len(eligible)} trainable queries")
    train_rows = [train_rows[i] for i in eligible]
    teacher_q = teacher_q_all[eligible]
    v1_q = base_q[eligible]
    specs = [cache.QuerySpec(qid=q["query_id"], text=q["text"], source="synthetic-qdrant",
                             domain=q["stratum"], bucket="coverage", family=q["family"],
                             positive_ids=((q["target_doc"],) if q.get("target_doc") else ()),
                             alias_pair_id=q.get("alias_pair_id", ""),
                             alias_view=q.get("alias_view", "")) for q in train_rows]
    bank = cache.Bank(doc_ids, doc_vecs, [d["kind"] for d in docs], seed=seed)
    arrays, sidecar = cache.build(specs, bank, teacher_q, v1_q, reg, cache_seed=seed,
        manifests={"v1_artifact": {"fixture": True},
                   "teacher_query_preprocessing": {"raw": True, "prefix": "fixture"}})
    sidecar = cache.save(root / "cache", arrays, sidecar,
                         artifact_inputs={"doc_vectors_sha256": sha_array(doc_vecs)})
    ids = [preprocess.active_ids(q["text"], tok, "T0") for q in train_rows]
    lineage = {"inherited_rows_sha256": sha_array(inherited),
               "inherited_scalars_sha256": sha_array(np.ones(base_n, np.float32))}

    def data():
        value = {"model": train.build_model(inherited, new_rows, fallback_id=0, device=device),
                "lineage": lineage, "ids": ids, "teacher_q": teacher_q,
                "bank": doc_vecs, "candidate_ids": arrays["candidate_ids"],
                "teacher_scores": arrays["teacher_scores"],
                "alias_pair_ids": arrays["alias_pair_ids"],
                "query_ids": [q["query_id"] for q in train_rows]}
        value["data_identity"] = train.training_data_identity(value)
        return value

    exemplar = data()
    cfg = train.RunCfg.from_registry(reg, "T0", seed=seed, device=device, rehearsal=True,
                                    tokenizer_sha256=tok_sha,
                                    preprocessing_sha256=preprocess.implementation_hash(),
                                    vocabulary_sha256=selection["vocabulary_sha256"],
                                    cache_artifact_sha256=sidecar["artifact_sha256"],
                                    prepared_data_sha256=exemplar["data_identity"]["sha256"],
                                    run_id="m18-rehearsal-t0")
    paused = train.run(cfg, data(), root / "run-resumed", resume=True, log=log)
    cfg2 = copy.copy(cfg); cfg2.stop_after = reg["training"]["schedule_steps"]
    resumed = train.run(cfg2, data(), root / "run-resumed", resume=True, log=log)
    uninterrupted = train.run(cfg2, data(), root / "run-uninterrupted", resume=False, log=log)
    resumed_rows, meta = train.load_snapshot(root / "run-resumed/snapshots/step_000004")
    uninterrupted_rows, _ = train.load_snapshot(root / "run-uninterrupted/snapshots/step_000004")
    resume_max_abs = float(np.abs(resumed_rows - uninterrupted_rows).max())
    if resume_max_abs != 0:
        raise AssertionError(f"resume trajectory mismatch: {resume_max_abs}")
    bundle = export.build_bundle(root / "bundle", resumed_rows, tok,
        {"fixture": True, "training_snapshot": meta, "source": "synthetic"}, variant="T0",
        preprocessing_sha256=preprocess.implementation_hash(), fixture=True)

    dev, qrels = protocol.load_surface("development", derived / "protocol")
    from loader_np import M18QueryEncoder
    enc = M18QueryEncoder(bundle, variant="T0")
    qv = enc.encode([q["text"] for q in dev])
    dense = evaluate.search(qv, doc_vecs, 100, doc_ids, [q["query_id"] for q in dev])
    bmrun = bm25.run([q["query_id"] for q in dev], [q["text"] for q in dev], 100)
    ev = evaluate.compare_models({"zero_v1": dense, "trained": dense}, bmrun, qrels, dev,
                                 prefetch=100, replicates=100, seed=seed)
    confirmation_identity = {"fixture_bundle": sha_json(meta)}
    protocol.lock_confirmation(confirmation_identity, derived / "protocol")
    protocol.run_confirmation(confirmation_identity,
                              lambda queries, qr: {"queries": len(queries), "qrels": len(qr)},
                              root / "confirmation-result.json", derived / "protocol")
    refused_twice = False
    try:
        protocol.run_confirmation(confirmation_identity,
                                  lambda queries, qr: {"queries": len(queries), "qrels": len(qr)},
                                  root / "confirmation-result-2.json", derived / "protocol")
    except SystemExit:
        refused_twice = True
    result = {"_schema": "m18-rehearsal-v1", "synthetic": True,
              "corpus": cm, "protocol": pm, "vocabulary_terms": terms,
              "cache_artifact_sha256": sidecar["artifact_sha256"],
              "paused_step": paused["step"], "resumed_step": resumed["step"],
              "uninterrupted_step": uninterrupted["step"],
              "resume_vs_uninterrupted_max_abs": resume_max_abs,
              "inherited_rows_unchanged": meta["inherited_rows_sha256"] == sha_array(inherited),
              "new_rows_changed": meta["new_rows_changed"],
              "confirmation_second_claim_refused": refused_twice,
              "evaluation_identity": ev["identity"], "bundle": str(bundle),
              "wall_clock_seconds": round(time.time() - t0, 3)}
    if not all((result["inherited_rows_unchanged"], result["new_rows_changed"], refused_twice)):
        raise AssertionError("rehearsal invariant failed")
    write_json(root / "rehearsal.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(WORK / "rehearsal"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--result", default=None)
    args = ap.parse_args(argv)
    result = build(args.out, args.seed, args.device)
    dest = Path(args.result) if args.result else (
        RESULTS / "m18_rehearsal.json" if Path(args.out).resolve() == (WORK / "rehearsal").resolve()
        else Path(args.out) / "rehearsal.json")
    write_json(dest, result)
    print(f"rehearsal OK -> {dest}")


if __name__ == "__main__":
    main()
