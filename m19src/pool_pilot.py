"""Build the frozen ten-query seven-route pool and report judgment cost only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from m19src import judgments, retrieval, zero
from m19src.benchmark_serving import (_collapse_top, _exact_top_indices, _load_context,
                                     BUILD_RECEIPT, T0_BUNDLE)
from m19src.build_candidate import _encode_teachers, _json_bytes, _publish
from m19src.common import M19, RESULTS, WORK, admit_read, load_json, sha_file, sha_json

PILOT_LOCK = M19 / "pool-pilot-lock-v1.json"
V0_BUNDLE = WORK / "bundles" / "v0-compose"
POOL_OUT = WORK / "pilot-v2" / "pool-manifest.json"
PACKET_OUT = WORK / "pilot-v2" / "evidence-packet.json"
RESULT_OUT = RESULTS / "m19_pool_pilot_v2.json"


def _aligned_index(inheritance):
    index = inheritance["inherited_data"]
    documents = np.load(admit_read(index["document_vectors"]["path"]), mmap_mode="c")
    passage_ids = json.loads(admit_read(index["doc_ids"]["path"]).read_text())
    positions = {passage_id: position for position, passage_id in enumerate(passage_ids)}
    passages = [None] * len(passage_ids)
    artifact_ids = [None] * len(passage_ids)
    metadata_candidates = {}
    with open(admit_read(index["index_corpus_jsonl"]["path"])) as handle:
        for line in handle:
            row = json.loads(line)
            position = positions.get(row["doc_id"])
            if position is not None:
                if passages[position] is not None:
                    raise SystemExit("M19 PILOT REFUSED: duplicate indexed passage ID")
                passages[position] = row
                artifact_ids[position] = str(row["artifact_id"])
                artifact_id = str(row["artifact_id"])
                candidate = {
                    "title": str(row.get("title") or row.get("path") or "").strip(),
                    "url_or_path": str(row.get("source_url") or row.get("path") or "").strip(),
                    "kind": str(row.get("kind") or "").strip(),
                    "path": str(row.get("path") or "").strip(),
                }
                score = (bool(candidate["title"]),
                         candidate["kind"] not in {"issue_comment", "review_comment"},
                         bool(candidate["url_or_path"]))
                if artifact_id not in metadata_candidates or score > metadata_candidates[
                        artifact_id][0]:
                    metadata_candidates[artifact_id] = (score, candidate)
    if (documents.shape[0] != len(passage_ids) or any(row is None for row in passages) or
            any(value is None for value in artifact_ids)):
        raise SystemExit("M19 PILOT REFUSED: inherited index rows are misaligned")
    artifact_metadata = {}
    for artifact_id, (_, candidate) in metadata_candidates.items():
        parent = {"artifact_id": artifact_id, "title": candidate["title"],
                  "url_or_path": candidate["url_or_path"], "kind": candidate["kind"]}
        if candidate["path"]:
            parent["path"] = candidate["path"]
        artifact_metadata[artifact_id] = {**candidate, "parent_metadata": parent}
    return index, documents, passage_ids, passages, artifact_ids, artifact_metadata


def _route(scores, passage_ids, passages, artifact_ids, exclusions):
    mask = np.fromiter((artifact_id in exclusions for artifact_id in artifact_ids),
                       dtype=bool, count=len(artifact_ids))
    indices = _exact_top_indices(scores, passage_ids, mask, depth=500)
    return _collapse_top(indices, np.asarray(scores)[indices], artifact_ids, passage_ids,
                         passages, excluded_artifacts=exclusions)


def run():
    import bm25s
    import Stemmer
    import torch

    lock = load_json(PILOT_LOCK)
    (inheritance, roster, base_config, verification, v1, _compact_v1, t0,
     development, benchmark_inputs) = _load_context()
    if (lock.get("query_count") != 10 or
            lock.get("development_queries_sha256") !=
            benchmark_inputs["development_queries_sha256"] or
            lock.get("query_seal_sha256") != benchmark_inputs["query_seal_sha256"] or
            lock.get("roster_identity_sha256") != roster["identity_sha256"]):
        raise SystemExit("M19 PILOT REFUSED: pilot lock differs from authenticated inputs")
    by_id = {row["query_id"]: row for row in development}
    if set(lock["pilot_query_ids"]) - set(by_id):
        raise SystemExit("M19 PILOT REFUSED: pilot query is absent from sealed development")
    queries = [by_id[query_id] for query_id in lock["pilot_query_ids"]]
    if (len({row["term"] for row in queries}) != 10 or
            any(row["primary_class"] != "short_context" for row in queries)):
        raise SystemExit("M19 PILOT REFUSED: pilot is not ten distinct short-context terms")

    build = load_json(BUILD_RECEIPT)
    v0_identity = build["bundles"]["V0-compose"]["identity_sha256"]
    v0 = zero.M19QueryEncoder.from_bundle(
        V0_BUNDLE, verification=verification, expected_identity=v0_identity)
    teacher_receipt = load_json(RESULTS / "m19_teacher_receipt.json")
    teacher_path = teacher_receipt["runtime"]["snapshot_path"]
    stella_vectors, runtime = _encode_teachers(
        teacher_path, [row["text"] for row in queries],
        load_json(M19 / "registry.json")["candidate"]["query_prefix"],
        base_config["document_encoder"]["config_kwargs"], "cuda")
    if runtime != teacher_receipt["runtime"]:
        raise SystemExit("M19 PILOT REFUSED: Stella runtime differs from teacher receipt")

    (index, documents, passage_ids, passages, artifact_ids,
     artifact_metadata) = _aligned_index(inheritance)
    bm25 = bm25s.BM25.load(str(Path(index["bm25_data"]["path"]).parent), load_corpus=False,
                           mmap=True, show_progress=False)
    stemmer = Stemmer.Stemmer("english")
    torch.set_num_threads(1)
    torch_documents = torch.from_numpy(np.asarray(documents)).to(
        device="cuda", dtype=torch.float32)

    encoders = {"v1_dense": v1, "v0_compose_dense": v0, "t0_teacher_dense": t0}
    routes = {name: {} for name in judgments.ROUTES}
    for query_index, row in enumerate(queries):
        query_id, text = row["query_id"], row["text"]
        exclusions = {
            *([str(row["source_artifact_id"])] if row.get("source_artifact_id") else []),
            *(str(value) for value in row.get("source_equivalent_artifact_ids") or []),
        }
        tokens = bm25s.tokenize([text], stemmer=stemmer, stopwords="en", return_ids=False,
                                show_progress=False)
        lexical = _route(
            bm25.get_scores(tokens[0]), passage_ids, passages, artifact_ids, exclusions)
        dense = {}
        for route_name, encoder in encoders.items():
            vector = encoder.encode(text)[0]
            scores = torch.mv(
                torch_documents, torch.from_numpy(vector).to(device="cuda", dtype=torch.float32)
            ).cpu().numpy()
            dense[route_name] = _route(
                scores, passage_ids, passages, artifact_ids, exclusions)
        stella_scores = torch.mv(
            torch_documents,
            torch.from_numpy(stella_vectors[query_index]).to(device="cuda", dtype=torch.float32),
        ).cpu().numpy()
        stella = _route(stella_scores, passage_ids, passages, artifact_ids, exclusions)
        routes["bm25"][query_id] = lexical["artifacts"][:10]
        for route_name in encoders:
            routes[route_name][query_id] = dense[route_name]["artifacts"][:10]
        routes["stella_dense"][query_id] = stella["artifacts"][:10]
        routes["v1_dbsf"][query_id] = retrieval.dbsf_artifacts(
            {"lexical": lexical, "dense": dense["v1_dense"]}, limit=10)
        routes["t0_teacher_dbsf"][query_id] = retrieval.dbsf_artifacts(
            {"lexical": lexical, "dense": dense["t0_teacher_dense"]}, limit=10)

    specs = [{**row, "source_exclusion_identity": sha_json(sorted({
        *([str(row["source_artifact_id"])] if row.get("source_artifact_id") else []),
        *(str(value) for value in row.get("source_equivalent_artifact_ids") or []),
    }))} for row in queries]
    registry = load_json(M19 / "registry.json")
    pool, packet = judgments.build_pool(
        routes, specs, artifact_metadata,
        seed=registry["judgments"]["randomization_seed"],
        cap=registry["judgments"]["development_cap"])
    judgments.assert_blinded(packet)
    pool_payload, packet_payload = _json_bytes(pool), _json_bytes(packet)
    _publish(POOL_OUT, pool_payload)
    _publish(PACKET_OUT, packet_payload)

    packet_words = sum(len(str(value).split()) for item in packet["items"]
                       for value in (item["title"], item["url_or_path"],
                                     *(passage["text"] for passage in item["passages"])))
    reviewer_minutes = packet_words / 200.0 + len(packet["items"]) * 0.25
    projected_items = int(np.ceil(len(packet["items"]) / len(queries) * len(development)))
    result = {
        "_schema": "m19-pool-cost-pilot-v2",
        "state": "complete",
        "inputs": {**benchmark_inputs, "pilot_lock_sha256": sha_file(PILOT_LOCK)},
        "query_count": len(queries),
        "terms": len({row["term"] for row in queries}),
        "routes": list(judgments.ROUTES),
        "unique_query_artifact_items": len(packet["items"]),
        "unique_artifacts_global": len({item["artifact_id"] for item in packet["items"]}),
        "packet_bytes": len(packet_payload),
        "reviewer_minutes_estimate": reviewer_minutes,
        "reviewer_minutes_assumption": "200 evidence words/minute plus 0.25 minute per label",
        "development_projection": {
            "queries": len(development), "unique_query_artifact_items": projected_items,
            "packet_bytes": int(np.ceil(len(packet_payload) / len(queries) * len(development))),
            "reviewer_minutes_estimate": reviewer_minutes / len(queries) * len(development),
            "cap": registry["judgments"]["development_cap"],
            "within_cap": projected_items <= registry["judgments"]["development_cap"],
        },
        "outputs": {
            "pool_manifest_sha256": sha_file(POOL_OUT),
            "evidence_packet_sha256": sha_file(PACKET_OUT),
        },
        "quality_or_judgments_read": False,
    }
    _publish(RESULT_OUT, _json_bytes(result))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["development_projection"]["within_cap"]:
        raise SystemExit("M19 PILOT STOP: projected development pool exceeds fixed cap")
    return result


if __name__ == "__main__":
    run()
