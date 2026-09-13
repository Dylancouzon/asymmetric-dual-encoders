"""Build M18 vocabulary, V0-Q variants, shared teacher/candidate cache and arm data."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tokenizers import AddedToken, Tokenizer

import cache
import export
import preprocess
import protocol
import system
import train
import vocab
from common import (REPO, WORK, admit_read, atomic_write_bytes, registry, sha_array, sha_file,
                    sha_json, write_json)


def _zero_encoder():
    import sys
    p = str(REPO / "m11" / "release")
    if p not in sys.path:
        sys.path.insert(0, p)
    from zero_encoder import ZeroQueryEncoder
    reg = registry()
    return ZeroQueryEncoder(reg["models"]["zero_v1"]["source_path"], variant="int8")


def _save_tokenizer(path, tokenizer):
    path = Path(path)
    payload = tokenizer.to_str().encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise SystemExit(f"M18 PREPARE REFUSED: tokenizer identity changed at {path}")
    else:
        atomic_write_bytes(path, payload)
    return sha_file(path)


def _add_placeholders(tokenizer, names):
    before = tokenizer.get_vocab_size(with_added_tokens=True)
    literals = [preprocess.PLACEHOLDERS[n] for n in names]
    n = tokenizer.add_tokens([AddedToken(x, single_word=False, normalized=True) for x in literals])
    if n != len(literals) or tokenizer.get_vocab_size(with_added_tokens=True) != before + n:
        raise SystemExit("M18 PREPARE REFUSED: typed placeholders did not add one stable row each")
    return literals


def _specs(rows):
    return [cache.QuerySpec(qid=q["query_id"], text=q["text"], source="qdrant-project",
                            domain=q["stratum"], bucket="coverage", family=q["family"],
                            positive_ids=((q["target_doc"],) if q.get("target_doc") else ()),
                            alias_pair_id=q.get("alias_pair_id", ""),
                            alias_view=q.get("alias_view", "")) for q in rows]


def _atomic_torch(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def build(index_root=None, protocol_root=None, out_root=None, device="cuda"):
    reg = registry()
    index_root = Path(index_root or WORK / "derived" / "index")
    protocol_root = Path(protocol_root or WORK / "derived" / "protocol")
    out = Path(out_root or WORK / "prepared")
    out.mkdir(parents=True, exist_ok=True)
    train_queries = list(protocol._read_jsonl(protocol_root / "training_queries.jsonl"))
    raw_texts = [q["text"] for q in train_queries]
    doc_ids = json.loads(admit_read(index_root / "doc_ids.json").read_text())
    doc_vecs = np.load(admit_read(index_root / "documents/vectors.f16.npy"), mmap_mode="r")
    if len(doc_ids) != len(doc_vecs):
        raise SystemExit("M18 PREPARE REFUSED: index ids/vectors disagree")

    # One raw teacher cache is shared by all variants, as registered.
    teacher_all, teacher_manifest = system.encode_teacher_queries(
        raw_texts, out / "teacher", name="training_raw", device=device)
    v1_encoder = _zero_encoder()
    v1_all = v1_encoder.encode(raw_texts)
    residual = 1.0 - np.sum(np.asarray(teacher_all, np.float32) * v1_all, axis=1)
    base_tok = Tokenizer.from_file(str(admit_read(
        Path(reg["models"]["zero_v1"]["source_path"]) / "tokenizer.json")))
    inherited, inherited_scalars, lineage = train.load_warm_start()
    stats, already_single = vocab.discover(
        [{"text": q["text"], "qid": q["query_id"], "source_doc": q["source_doc"],
          "domain": q["stratum"]} for q in train_queries], residual, base_tok)
    selection = vocab.select(stats, reg, single_token=already_single)
    terms = [x["term"] for x in selection["terms"]]
    new_rows, compositions = vocab.init_new_rows(terms, base_tok, inherited)
    t0 = Tokenizer.from_str(base_tok.to_str())
    t0, t0_sha_mem, n_added = vocab.extend_tokenizer(t0, terms)
    t0_rows = np.concatenate([inherited, new_rows], axis=0)
    tokenization = [{"term": term, "before_ids": vocab.tokenize_term(base_tok, term),
                     "after_ids": vocab.tokenize_term(t0, term)} for term in terms]
    vocab_dir = out / "vocabulary"; vocab_dir.mkdir(exist_ok=True)
    t0_path = vocab_dir / "tokenizer_t0.json"
    t0_sha = _save_tokenizer(t0_path, t0)
    if t0_sha != t0_sha_mem:
        raise SystemExit("M18 PREPARE REFUSED: serialized T0 tokenizer hash drifted")

    collision = preprocess.collision_audit([
        {"text": q["text"], "relevance_group": q.get("target_doc") or q["family_group"]}
        for q in train_queries])
    write_json(out / "t1_collision_audit.json", collision)
    if out_root is None:
        write_json(REPO / "results/m18_t1_collision_audit.json", collision)
    pattern_names = [name for name, count in collision["pattern_counts"].items() if count > 0]
    variants = {"T0": {"tokenizer": t0, "rows": t0_rows, "tokenizer_path": t0_path},
                "T2": {"tokenizer": Tokenizer.from_str(t0.to_str()), "rows": t0_rows,
                       "tokenizer_path": t0_path}}
    placeholder_report = []
    if collision["pass"] and pattern_names:
        t1 = Tokenizer.from_str(t0.to_str())
        literals = _add_placeholders(t1, pattern_names)
        placeholder_rows, placeholder_parts = vocab.init_new_rows(literals, t0, t0_rows)
        t1_rows = np.concatenate([t0_rows, placeholder_rows], 0)
        t1_path = vocab_dir / "tokenizer_t1.json"
        _save_tokenizer(t1_path, t1)
        variants["T1"] = {"tokenizer": t1, "rows": t1_rows, "tokenizer_path": t1_path}
        placeholder_report = [{"pattern": n, "token": lit, "composition": part}
                              for n, lit, part in zip(pattern_names, literals, placeholder_parts)]

    vocab_report = {"_schema": "m18-vocabulary-manifest-v1", "training_queries": len(train_queries),
                    "base_vocab": len(inherited), "added_rows_t0": n_added, "terms": selection,
                    "compositions": compositions, "tokenization_before_after": tokenization,
                    "placeholder_rows_t1": placeholder_report,
                    "table_bytes": {name: int(v["rows"].shape[0] * v["rows"].shape[1] +
                                              v["rows"].shape[0] * 4)
                                    for name, v in variants.items()},
                    "tokenizers": {name: {"path": str(v["tokenizer_path"]),
                                            "sha256": sha_file(v["tokenizer_path"]),
                                            "vocab": v["tokenizer"].get_vocab_size(True)}
                                   for name, v in variants.items()},
                    "lineage": lineage}
    vocab_report["sha256"] = sha_json(vocab_report)
    write_json(out / "vocabulary_manifest.json", vocab_report)
    if out_root is None:
        write_json(REPO / "results/m18_vocabulary_manifest.json", vocab_report)

    # First-class step-0 exports exist before any optimizer step.
    bundles = {}
    for name, info in variants.items():
        dest = out / "bundles" / f"V0-{name}"
        table_identity = {"variant": name, "tokenizer_sha256": sha_file(info["tokenizer_path"]),
                          "preprocessing_sha256": preprocess.implementation_hash(),
                          "vocabulary_sha256": vocab_report["sha256"],
                          "rows_sha256": sha_array(info["rows"])}
        if dest.exists():
            export.gate_bundle(dest, float_rows=info["rows"])
            stored = json.loads(admit_read(dest / "provenance.json").read_text()).get("table_identity")
            if stored != table_identity:
                raise SystemExit(f"M18 PREPARE REFUSED: stale V0-{name} bundle identity")
        else:
            export.build_bundle(dest, info["rows"], info["tokenizer"], {
                "candidate": f"V0-{name}", "step": 0, "vocabulary_manifest": vocab_report["sha256"],
                "source_snapshot": json.loads(admit_read(REPO / "results/m18_source_manifest.json").read_text()),
                "table_identity": table_identity, "fixture": False}, variant=name,
                preprocessing_sha256=preprocess.implementation_hash())
        bundles[name] = str(dest)

    # The shared cache covers the union of queries that can update at least one arm.
    active_by_variant = {}
    for name, info in variants.items():
        ids = [preprocess.active_ids(t, info["tokenizer"], name) for t in raw_texts]
        start = len(inherited)
        active_by_variant[name] = {i for i, row in enumerate(ids) if any(x >= start for x in row)}
    union = sorted(set().union(*active_by_variant.values()))
    if not union:
        raise SystemExit("M18 PREPARE REFUSED: no training query contains a selected exact row")
    queries = [train_queries[i] for i in union]
    specs = _specs(queries)
    teacher_q = np.asarray(teacher_all[union], np.float32)
    v1_q = v1_all[union]
    bank = cache.Bank(doc_ids, doc_vecs, ["qdrant"] * len(doc_ids), seed=reg["training"]["seed_primary"])
    cache_dir = out / "shared_cache"
    cache_manifests = {"v1_artifact": reg["models"]["zero_v1"],
                       "teacher_query_preprocessing": reg["models"]["teacher"],
                       "source_split": {"protocol_sha256": sha_file(protocol_root / "protocol_manifest.json")}}
    cache_inputs = {"teacher_vectors_sha256": sha_array(teacher_q),
                    "v1_vectors_sha256": sha_array(v1_q),
                    "document_vectors_sha256": sha_array(doc_vecs)}
    if cache_dir.exists():
        arrays, sidecar = cache.load(cache_dir)
        expected = cache.identity(specs, bank, reg, reg["training"]["seed_primary"], cache_manifests)
        if sidecar.get("identity") != expected or sidecar.get("artifact_inputs") != cache_inputs:
            raise SystemExit("M18 PREPARE REFUSED: existing shared cache has stale inputs")
        if arrays["qids"].tolist() != [q.qid for q in specs]:
            raise SystemExit("M18 PREPARE REFUSED: existing shared cache query order changed")
    else:
        arrays, sidecar = cache.build(specs, bank, teacher_q, v1_q, reg,
                                      cache_seed=reg["training"]["seed_primary"],
                                      manifests=cache_manifests)
        sidecar = cache.save(cache_dir, arrays, sidecar, artifact_inputs=cache_inputs)

    prepared = {}
    union_pos = {old: new for new, old in enumerate(union)}
    for name, info in variants.items():
        chosen_old = sorted(active_by_variant[name])
        positions = np.asarray([union_pos[i] for i in chosen_old], dtype=np.int64)
        chosen_queries = [train_queries[i] for i in chosen_old]
        ids = [preprocess.active_ids(q["text"], info["tokenizer"], name) for q in chosen_queries]
        data = {"model": train.build_model(inherited, info["rows"][len(inherited):],
                                             fallback_id=101, device="cpu"),
                "lineage": lineage, "ids": ids, "teacher_q": teacher_q[positions],
                "bank": np.asarray(doc_vecs), "candidate_ids": arrays["candidate_ids"][positions],
                "teacher_scores": arrays["teacher_scores"][positions],
                "alias_pair_ids": arrays["alias_pair_ids"][positions],
                "query_ids": [q["query_id"] for q in chosen_queries],
                "run_identities": {
                    "tokenizer_sha256": sha_file(info["tokenizer_path"]),
                    "preprocessing_sha256": preprocess.implementation_hash(),
                    "vocabulary_sha256": vocab_report["sha256"],
                    "cache_artifact_sha256": sidecar["artifact_sha256"]}}
        data["data_identity"] = train.training_data_identity(data)
        path = out / f"prepared_{name}.pt"
        _atomic_torch(path, data)
        prepared[name] = {"path": str(path), "sha256": sha_file(path),
                          "eligible_queries": len(ids), "shared_positions_sha256": sha_array(positions)}

    result = {"_schema": "m18-prepare-manifest-v1", "raw_training_queries": len(train_queries),
              "shared_cache_queries": len(union), "variants": prepared, "bundles": bundles,
              "teacher_cache": teacher_manifest, "candidate_cache_identity": sidecar["identity"],
              "candidate_cache_artifact_sha256": sidecar["artifact_sha256"],
              "vocabulary_manifest_sha256": vocab_report["sha256"],
              "collision_audit_pass": collision["pass"]}
    result["sha256"] = sha_json(result)
    write_json(out / "prepare_manifest.json", result)
    if out_root is None:
        write_json(REPO / "results/m18_prepare_manifest.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", default=None); ap.add_argument("--protocol", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    print(json.dumps(build(args.index, args.protocol, args.out, args.device), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
