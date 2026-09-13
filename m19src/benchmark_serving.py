"""Measure M19 v1/T0 encoder and artifact-collapsed hybrid serving latency."""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
import tracemalloc
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from m11.release.zero_encoder import ZeroQueryEncoder
from m19src import inherit, retrieval, zero
from m19src.build_candidate import _json_bytes, _publish
from m19src.common import (M19, RESULTS, WORK, admit_read, load_json, sha_file,
                           sha_json, sha_texts)

T0_BUNDLE = WORK / "bundles" / "t0-teacher"
DEV_QUERIES = WORK / "queries" / "development.jsonl"
BUILD_RECEIPT = RESULTS / "m19_candidate_build.json"
QUERY_SEAL = M19 / "query-split-seal-v1.json"
SERVING_RECEIPT = RESULTS / "m19_serving_receipt_t0_v2.json"


def _percentiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {"median_ms": float(np.median(array)), "p95_ms": float(np.percentile(array, 95))}


def _collapse_top(indices, scores, artifact_ids, passage_ids, passages, *, excluded_artifacts=()):
    excluded = set(map(str, excluded_artifacts))
    ordered = sorted(
        ((float(score), str(passage_ids[int(index)]), int(index))
         for index, score in zip(indices, scores)
         if str(artifact_ids[int(index)]) not in excluded),
        key=lambda row: (-row[0], row[1]),
    )
    rows = []
    seen = set()
    for score, passage_id, index in ordered[:500]:
        artifact_id = str(artifact_ids[index])
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        rows.append({"artifact_id": artifact_id, "score": float(score),
                     "passage_id": passage_id, "passage": passages[index]})
        if len(rows) == 100:
            break
    return {"artifacts": rows}


def _exact_top_indices(scores, passage_ids, excluded, depth=500):
    """Select an exact top depth without depending on an unstable top-k tie order."""
    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (len(passage_ids),) or not np.isfinite(values).all():
        raise SystemExit("M19 SERVING REFUSED: route scores are malformed")
    eligible = np.flatnonzero(~np.asarray(excluded, dtype=bool))
    if len(eligible) > int(depth):
        eligible_values = values[eligible]
        cutoff = np.partition(eligible_values, len(eligible_values) - int(depth))[
            len(eligible_values) - int(depth)]
        above = eligible[eligible_values > cutoff]
        tied = eligible[eligible_values == cutoff]
        need = int(depth) - len(above)
        tied = np.asarray(sorted(tied, key=lambda index: str(passage_ids[index]))[:need])
        eligible = np.concatenate([above, tied])
    return np.asarray(sorted(
        eligible, key=lambda index: (-float(values[index]), str(passage_ids[index]))),
        dtype=np.int64,
    )


def _load_context():
    inheritance = inherit.verify()
    registry = load_json(M19 / "registry.json")
    roster = load_json(M19 / "term-roster-lock-v3.json")
    seal = load_json(QUERY_SEAL)
    build = load_json(BUILD_RECEIPT)
    if (seal.get("registry_sha256") != sha_json(registry) or
            seal.get("roster_sha256") != sha_json(roster) or
            seal.get("splits", {}).get("development", {}).get("sha256") != sha_file(DEV_QUERIES)):
        raise SystemExit("M19 SERVING REFUSED: development query seal differs")
    if (build.get("_schema") != "m19-candidate-build-v1" or
            build.get("state") != "complete" or
            build.get("inheritance_identity") != inheritance["identity_sha256"] or
            build.get("roster_identity") != roster["identity_sha256"] or
            set(build.get("bundles", {})) != {"V0-compose", "T0-teacher"}):
        raise SystemExit("M19 SERVING REFUSED: candidate build receipt differs")
    base_config = load_json(inheritance["inherited_data"]["zero_v1_config"]["path"])
    tokenizer_payload = admit_read(
        inheritance["inherited_data"]["zero_v1_tokenizer"]["path"]).read_bytes()
    codes, scales = zero.load_base(inheritance["inherited_data"]["zero_v1_model"]["path"])
    verification = {
        "base_codes": codes, "base_scales": scales, "base_tokenizer_payload": tokenizer_payload,
        "roster": roster, "inheritance_identity": inheritance["identity_sha256"],
        "pooling_identity_sha256": inheritance["identities"]["released_effective_table"][
            "pooling_sha256"],
    }
    compact_v1 = zero.M19QueryEncoder(codes, scales, Tokenizer.from_str(tokenizer_payload.decode()),
                                      base_config)
    v1 = ZeroQueryEncoder(Path(inheritance["inherited_data"]["zero_v1_model"]["path"]).parent)
    expected_bundle_identity = build["bundles"]["T0-teacher"]["identity_sha256"]
    t0 = zero.M19QueryEncoder.from_bundle(
        T0_BUNDLE, verification=verification, expected_identity=expected_bundle_identity)
    query_rows = [json.loads(line) for line in admit_read(DEV_QUERIES).read_text().splitlines()]
    if [row["query_id"] for row in query_rows] != seal["splits"]["development"]["query_ids"]:
        raise SystemExit("M19 SERVING REFUSED: development query order differs from seal")
    observed_bundle = zero.verify_bundle(
        T0_BUNDLE, verification=verification, expected_identity=expected_bundle_identity)
    inputs = {
        "registry_sha256": sha_file(M19 / "registry.json"),
        "inheritance_identity_sha256": inheritance["identity_sha256"],
        "roster_identity_sha256": roster["identity_sha256"],
        "query_seal_sha256": sha_file(QUERY_SEAL),
        "development_queries_sha256": sha_file(DEV_QUERIES),
        "candidate_build_sha256": sha_file(BUILD_RECEIPT),
        "bundle_identity_sha256": observed_bundle["identity_sha256"],
        "index_inputs_sha256": {
            role: inheritance["inherited_data"][role]["sha256"]
            for role in ("index_corpus_jsonl", "doc_ids", "document_vectors", "bm25_data",
                         "bm25_indices", "bm25_indptr", "bm25_vocab", "bm25_params")
        },
    }
    return (inheritance, roster, base_config, verification, v1, compact_v1, t0,
            query_rows, inputs)


def _parity(verification, base_config, query_rows, released_v1, compact_v1, t0, documents):
    teachers = np.load(admit_read(WORK / "candidate" / "teacher-vectors.npy"))
    terms = [row["term"] for row in verification["roster"]["terms"]]
    built = zero.construct_added_rows(
        verification["base_codes"], verification["base_scales"],
        verification["base_tokenizer_payload"], verification["roster"],
        {term: teachers[index] for index, term in enumerate(terms)})
    codes, scales = zero.compact_table(
        verification["base_codes"], verification["base_scales"], built["T0-teacher"])
    direct = zero.M19QueryEncoder(
        codes, scales, Tokenizer.from_str(built["tokenizer"].to_str()), base_config)
    texts = [row["text"] for row in query_rows]
    loader_max_abs = float(np.max(np.abs(direct.encode(texts) - t0.encode(texts))))
    released_max_abs = float(np.max(np.abs(released_v1.encode(texts) - compact_v1.encode(texts))))
    no_match = ["vector search", "collection aliases", "payload filters", "snapshot recovery"]
    ranking_equal = True
    documents_f32 = np.asarray(documents, dtype=np.float32)
    for text in no_match:
        base_scores = documents_f32 @ released_v1.encode(text)[0]
        t0_scores = documents_f32 @ t0.encode(text)[0]
        base_top = np.lexsort((np.arange(len(base_scores)), -base_scores))[:10]
        t0_top = np.lexsort((np.arange(len(t0_scores)), -t0_scores))[:10]
        ranking_equal = ranking_equal and np.array_equal(base_top, t0_top)
    return loader_max_abs, released_max_abs, ranking_equal


def benchmark(query_count, warmup):
    import bm25s
    import Stemmer
    import torch

    registry = load_json(M19 / "registry.json")
    (inheritance, roster, base_config, verification, v1, compact_v1, t0,
     query_rows, inputs) = _load_context()
    index = inheritance["inherited_data"]
    # Copy-on-write keeps the source array immutable on disk while satisfying
    # torch.from_numpy's writable-buffer requirement without a full host copy.
    documents = np.load(admit_read(index["document_vectors"]["path"]), mmap_mode="c")
    passage_ids = json.loads(admit_read(index["doc_ids"]["path"]).read_text())
    positions = {passage_id: index for index, passage_id in enumerate(passage_ids)}
    artifact_ids = [None] * len(passage_ids)
    passages = [None] * len(passage_ids)
    with open(admit_read(index["index_corpus_jsonl"]["path"])) as handle:
        for line in handle:
            row = json.loads(line)
            position = positions.get(row["doc_id"])
            if position is not None:
                if artifact_ids[position] is not None:
                    raise SystemExit("M19 SERVING REFUSED: duplicate indexed passage ID")
                artifact_ids[position] = str(row["artifact_id"])
                passages[position] = row
    if (documents.shape[0] != len(artifact_ids) or
            any(artifact_id is None for artifact_id in artifact_ids) or
            any(passage is None for passage in passages)):
        raise SystemExit("M19 SERVING REFUSED: inherited index rows are misaligned")
    loader_max_abs, released_max_abs, ranking_equal = _parity(
        verification, base_config, query_rows, v1, compact_v1, t0, documents)

    bm25 = bm25s.BM25.load(str(Path(index["bm25_data"]["path"]).parent), load_corpus=False,
                           mmap=True, show_progress=False)
    stemmer = Stemmer.Stemmer("english")
    torch.set_num_threads(1)
    torch_documents = torch.from_numpy(np.asarray(documents)).to(device="cuda", dtype=torch.float32)
    sequence = [query_rows[index % len(query_rows)] for index in range(int(query_count))]

    def encode_once(encoder, text):
        started = time.perf_counter_ns()
        encoder.encode(text)
        return (time.perf_counter_ns() - started) / 1e6

    def serve_once(encoder, query_row):
        text = query_row["text"]
        excluded = {
            *([str(query_row["source_artifact_id"])] if query_row.get("source_artifact_id") else []),
            *(str(value) for value in query_row.get("source_equivalent_artifact_ids") or []),
        }
        excluded_mask = np.fromiter(
            (artifact_id in excluded for artifact_id in artifact_ids), dtype=bool,
            count=len(artifact_ids))
        started = time.perf_counter_ns()
        vector = encoder.encode(text)[0]
        encoded = time.perf_counter_ns()
        tokens = bm25s.tokenize([text], stemmer=stemmer, stopwords="en", return_ids=False,
                                show_progress=False)
        lexical_all_scores = bm25.get_scores(tokens[0])
        lexical_indices = _exact_top_indices(
            lexical_all_scores, passage_ids, excluded_mask, depth=500)
        lexical_scores = lexical_all_scores[lexical_indices]
        lexical_done = time.perf_counter_ns()
        query = torch.from_numpy(vector).to(device="cuda", dtype=torch.float32)
        dense_scores = torch.mv(torch_documents, query)
        dense_all_scores = dense_scores.cpu().numpy()
        dense_indices = _exact_top_indices(
            dense_all_scores, passage_ids, excluded_mask, depth=500)
        dense_values = dense_all_scores[dense_indices]
        dense_done = time.perf_counter_ns()
        lexical = _collapse_top(lexical_indices, lexical_scores, artifact_ids, passage_ids,
                                passages, excluded_artifacts=excluded)
        dense = _collapse_top(dense_indices, dense_values, artifact_ids, passage_ids, passages,
                              excluded_artifacts=excluded)
        collapsed = time.perf_counter_ns()
        retrieval.dbsf_artifacts({"lexical": lexical, "dense": dense}, limit=10)
        finished = time.perf_counter_ns()
        return {
            "total": (finished - started) / 1e6,
            "encode": (encoded - started) / 1e6,
            "lexical": (lexical_done - encoded) / 1e6,
            "dense": (dense_done - lexical_done) / 1e6,
            "collapse": (collapsed - dense_done) / 1e6,
            "fusion": (finished - collapsed) / 1e6,
        }

    for index in range(int(warmup)):
        query_row = sequence[index % len(sequence)]
        for encoder in (v1, t0) if index % 2 else (t0, v1):
            encode_once(encoder, query_row["text"])
            serve_once(encoder, query_row)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    t0.encode([row["text"] for row in sequence[:min(100, len(sequence))]])
    current_bytes, temporary_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    measurements = {name: {"encoder": [], "serving": []} for name in ("v1", "t0")}
    encoders = {"v1": v1, "t0": t0}
    for index, query_row in enumerate(sequence):
        order = ("v1", "t0") if index % 2 else ("t0", "v1")
        for name in order:
            measurements[name]["encoder"].append(encode_once(encoders[name], query_row["text"]))
            measurements[name]["serving"].append(serve_once(encoders[name], query_row))

    summary = {}
    for name in ("v1", "t0"):
        summary[name] = {"encoder": _percentiles(measurements[name]["encoder"])}
        for component in ("total", "encode", "lexical", "dense", "collapse", "fusion"):
            summary[name][component] = _percentiles(
                [row[component] for row in measurements[name]["serving"]])
    encoder_median_ratio = summary["t0"]["encoder"]["median_ms"] / summary["v1"]["encoder"]["median_ms"]
    encoder_p95_ratio = summary["t0"]["encoder"]["p95_ms"] / summary["v1"]["encoder"]["p95_ms"]
    total_median_ratio = summary["t0"]["total"]["median_ms"] / summary["v1"]["total"]["median_ms"]
    total_p95_ratio = summary["t0"]["total"]["p95_ms"] / summary["v1"]["total"]["p95_ms"]
    gates = registry["numerical_gates"]
    official = int(query_count) == int(gates["latency_queries"])
    result = {
        "_schema": "m19-serving-benchmark-v2", "official": official,
        "query_count": int(query_count),
        "query_sequence_sha256": sha_texts([row["text"] for row in sequence]),
        "warmup": int(warmup), "repetitions": 1,
        "host": {"node": platform.node(), "platform": platform.platform(),
                 "gpu": torch.cuda.get_device_name(0)},
        "threads": {"torch": 1, "bm25": 1},
        "summary": summary,
        "loader_parity_max_abs": loader_max_abs,
        "released_loader_parity_max_abs": released_max_abs,
        "no_match_ranking_parity": ranking_equal,
        "inputs": inputs,
        "route_semantics": {
            "dense_dtype": "float32", "exclusion_before_truncation": True,
            "passage_depth": 500, "artifact_depth": 100,
            "tie_break": "descending score then ascending passage_id",
        },
        "temporary_peak_bytes": int(temporary_peak_bytes),
        "tracemalloc_current_bytes": int(current_bytes),
        "rss_high_water_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "ratios": {
            "encoder_median": encoder_median_ratio, "encoder_p95": encoder_p95_ratio,
            "encoder_median_additive_ms": summary["t0"]["encoder"]["median_ms"] - summary["v1"]["encoder"]["median_ms"],
            "encoder_p95_additive_ms": summary["t0"]["encoder"]["p95_ms"] - summary["v1"]["encoder"]["p95_ms"],
            "end_to_end_median": total_median_ratio, "end_to_end_p95": total_p95_ratio,
        },
    }
    if official:
        r = result["ratios"]
        checks = {
            "loader_parity": max(loader_max_abs, released_max_abs) <=
                gates["loader_parity_max_abs"],
            "tokenizer_boundaries": load_json(RESULTS / "m19_algebra_gates.json")[
                "tokenizer_parity"]["no_match_tokenization_equal"],
            "no_match_ranking_parity": ranking_equal,
            "pooling_identity": True,
            "resident_memory": t0.resident_table_bytes == compact_v1.resident_table_bytes +
                len(roster["terms"]) * gates["added_row_bytes"],
            "encoder_latency": max(r["encoder_median"], r["encoder_p95"]) <=
                gates["encoder_latency_ratio_maximum"] and max(
                    r["encoder_median_additive_ms"], r["encoder_p95_additive_ms"]) <=
                gates["encoder_latency_additive_ms_maximum"],
            "end_to_end_latency": max(r["end_to_end_median"], r["end_to_end_p95"]) <=
                gates["end_to_end_latency_ratio_maximum"],
        }
        result["checks"] = checks
    return result


def _serving_receipt(result):
    r = result["ratios"]
    serving = {
            "_schema": "m19-serving-gates-v2", "variant": "T0-teacher",
            "bundle_identity_sha256": result["inputs"]["bundle_identity_sha256"],
            "checks": result["checks"],
            "measurements": {
                "loader_parity_max_abs": result["loader_parity_max_abs"],
                "released_loader_parity_max_abs": result["released_loader_parity_max_abs"],
                "added_row_bytes": load_json(M19 / "registry.json")["numerical_gates"][
                    "added_row_bytes"],
                "encoder_latency_median_ratio": r["encoder_median"],
                "encoder_latency_p95_ratio": r["encoder_p95"],
                "encoder_latency_median_additive_ms": max(0.0, r["encoder_median_additive_ms"]),
                "encoder_latency_p95_additive_ms": max(0.0, r["encoder_p95_additive_ms"]),
                "end_to_end_latency_median_ratio": r["end_to_end_median"],
                "end_to_end_latency_p95_ratio": r["end_to_end_p95"],
                "temporary_peak_bytes": result["temporary_peak_bytes"],
                "rss_high_water_kib": result["rss_high_water_kib"],
            },
            "benchmark": {key: result[key] for key in (
                "query_count", "query_sequence_sha256", "host", "threads", "warmup",
                "repetitions")},
            "inputs": result["inputs"],
            "route_semantics": result["route_semantics"],
        }
    return serving


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-count", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    out = Path(args.out) if args.out else RESULTS / f"m19_serving_benchmark_{args.query_count}.json"
    if out.exists():
        result = load_json(out)
        if (result.get("_schema") != "m19-serving-benchmark-v2" or
                result.get("query_count") != args.query_count or result.get("warmup") != args.warmup):
            raise SystemExit("M19 SERVING REFUSED: existing benchmark cannot resume this request")
    else:
        result = benchmark(args.query_count, args.warmup)
        _publish(out, _json_bytes(result))
    if result["official"]:
        _publish(SERVING_RECEIPT, _json_bytes(_serving_receipt(result)))
    print(json.dumps({"official": result["official"], "query_count": result["query_count"],
                      "ratios": result["ratios"]}, indent=2, sort_keys=True))
    if result["official"] and not all(result["checks"].values()):
        raise SystemExit("M19 SERVING STOP: registered latency or serving gate failed")


if __name__ == "__main__":
    main()
