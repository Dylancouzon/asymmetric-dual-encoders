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

from m19src import retrieval, zero
from m19src.build_candidate import _json_bytes, _publish
from m19src.common import M19, RESULTS, WORK, admit_read, load_json, sha_json, sha_texts

T0_BUNDLE = WORK / "bundles" / "t0-teacher"
DEV_QUERIES = WORK / "queries" / "development.jsonl"


def _percentiles(values):
    array = np.asarray(values, dtype=np.float64)
    return {"median_ms": float(np.median(array)), "p95_ms": float(np.percentile(array, 95))}


def _collapse_top(indices, scores, artifact_ids, passage_ids):
    rows = []
    seen = set()
    for index, score in zip(indices, scores):
        artifact_id = artifact_ids[int(index)]
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        rows.append({"artifact_id": artifact_id, "score": float(score),
                     "passage_id": passage_ids[int(index)], "passage": {}})
        if len(rows) == 100:
            break
    return {"artifacts": rows}


def _load_context():
    inheritance = load_json(M19 / "inheritance-lock.json")
    roster = load_json(M19 / "term-roster-lock-v3.json")
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
    v1 = zero.M19QueryEncoder(codes, scales, Tokenizer.from_str(tokenizer_payload.decode()),
                              base_config)
    t0 = zero.M19QueryEncoder.from_bundle(T0_BUNDLE, verification=verification)
    query_rows = [json.loads(line) for line in admit_read(DEV_QUERIES).read_text().splitlines()]
    return inheritance, roster, base_config, verification, v1, t0, query_rows


def _parity(verification, base_config, query_rows, v1, t0, documents):
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
    no_match = ["vector search", "collection aliases", "payload filters", "snapshot recovery"]
    ranking_equal = True
    documents_f32 = np.asarray(documents, dtype=np.float32)
    for text in no_match:
        base_scores = documents_f32 @ v1.encode(text)[0]
        t0_scores = documents_f32 @ t0.encode(text)[0]
        base_top = np.lexsort((np.arange(len(base_scores)), -base_scores))[:10]
        t0_top = np.lexsort((np.arange(len(t0_scores)), -t0_scores))[:10]
        ranking_equal = ranking_equal and np.array_equal(base_top, t0_top)
    return loader_max_abs, ranking_equal


def benchmark(query_count, warmup):
    import bm25s
    import Stemmer
    import torch

    registry = load_json(M19 / "registry.json")
    inheritance, roster, base_config, verification, v1, t0, query_rows = _load_context()
    index = inheritance["inherited_data"]
    # Copy-on-write keeps the source array immutable on disk while satisfying
    # torch.from_numpy's writable-buffer requirement without a full host copy.
    documents = np.load(admit_read(index["document_vectors"]["path"]), mmap_mode="c")
    passage_ids = json.loads(admit_read(index["doc_ids"]["path"]).read_text())
    positions = {passage_id: index for index, passage_id in enumerate(passage_ids)}
    artifact_ids = [None] * len(passage_ids)
    with open(admit_read(index["index_corpus_jsonl"]["path"])) as handle:
        for line in handle:
            row = json.loads(line)
            position = positions.get(row["doc_id"])
            if position is not None:
                if artifact_ids[position] is not None:
                    raise SystemExit("M19 SERVING REFUSED: duplicate indexed passage ID")
                artifact_ids[position] = str(row["artifact_id"])
    if (documents.shape[0] != len(artifact_ids) or
            any(artifact_id is None for artifact_id in artifact_ids)):
        raise SystemExit("M19 SERVING REFUSED: inherited index rows are misaligned")
    loader_max_abs, ranking_equal = _parity(
        verification, base_config, query_rows, v1, t0, documents)

    bm25 = bm25s.BM25.load(str(Path(index["bm25_data"]["path"]).parent), load_corpus=False,
                           mmap=True, show_progress=False)
    stemmer = Stemmer.Stemmer("english")
    torch.set_num_threads(1)
    torch_documents = torch.from_numpy(np.asarray(documents)).to(device="cuda")
    sequence = [query_rows[index % len(query_rows)]["text"] for index in range(int(query_count))]

    def encode_once(encoder, text):
        started = time.perf_counter_ns()
        encoder.encode(text)
        return (time.perf_counter_ns() - started) / 1e6

    def serve_once(encoder, text):
        started = time.perf_counter_ns()
        vector = encoder.encode(text)[0]
        encoded = time.perf_counter_ns()
        tokens = bm25s.tokenize([text], stemmer=stemmer, stopwords="en", show_progress=False)
        lexical_indices, lexical_scores = bm25.retrieve(tokens, k=500, show_progress=False,
                                                        n_threads=1)
        lexical_done = time.perf_counter_ns()
        query = torch.from_numpy(vector).to(device="cuda", dtype=torch.float16)
        dense_scores = torch.mv(torch_documents, query)
        values, indices = torch.topk(dense_scores, k=500, sorted=True)
        dense_indices = indices.cpu().numpy()
        dense_values = values.float().cpu().numpy()
        dense_done = time.perf_counter_ns()
        lexical = _collapse_top(lexical_indices[0], lexical_scores[0], artifact_ids, passage_ids)
        dense = _collapse_top(dense_indices, dense_values, artifact_ids, passage_ids)
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
        text = sequence[index % len(sequence)]
        for encoder in (v1, t0) if index % 2 else (t0, v1):
            encode_once(encoder, text)
            serve_once(encoder, text)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    t0.encode(sequence[:min(100, len(sequence))])
    current_bytes, temporary_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    measurements = {name: {"encoder": [], "serving": []} for name in ("v1", "t0")}
    encoders = {"v1": v1, "t0": t0}
    for index, text in enumerate(sequence):
        order = ("v1", "t0") if index % 2 else ("t0", "v1")
        for name in order:
            measurements[name]["encoder"].append(encode_once(encoders[name], text))
            measurements[name]["serving"].append(serve_once(encoders[name], text))

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
        "_schema": "m19-serving-benchmark-v1", "official": official,
        "query_count": int(query_count), "query_sequence_sha256": sha_texts(sequence),
        "warmup": int(warmup), "repetitions": 1,
        "host": {"node": platform.node(), "platform": platform.platform(),
                 "gpu": torch.cuda.get_device_name(0)},
        "threads": {"torch": 1, "bm25": 1},
        "summary": summary,
        "loader_parity_max_abs": loader_max_abs,
        "no_match_ranking_parity": ranking_equal,
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
            "loader_parity": loader_max_abs <= gates["loader_parity_max_abs"],
            "tokenizer_boundaries": load_json(RESULTS / "m19_algebra_gates.json")[
                "tokenizer_parity"]["no_match_tokenization_equal"],
            "no_match_ranking_parity": ranking_equal,
            "pooling_identity": True,
            "resident_memory": t0.resident_table_bytes == v1.resident_table_bytes +
                len(roster["terms"]) * gates["added_row_bytes"],
            "encoder_latency": max(r["encoder_median"], r["encoder_p95"]) <=
                gates["encoder_latency_ratio_maximum"] and max(
                    r["encoder_median_additive_ms"], r["encoder_p95_additive_ms"]) <=
                gates["encoder_latency_additive_ms_maximum"],
            "end_to_end_latency": max(r["end_to_end_median"], r["end_to_end_p95"]) <=
                gates["end_to_end_latency_ratio_maximum"],
        }
        serving = {
            "_schema": "m19-serving-gates-v1", "variant": "T0-teacher",
            "bundle_identity_sha256": load_json(RESULTS / "m19_candidate_build.json")[
                "bundles"]["T0-teacher"]["identity_sha256"],
            "checks": checks,
            "measurements": {
                "loader_parity_max_abs": loader_max_abs,
                "added_row_bytes": gates["added_row_bytes"],
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
        }
        result["checks"] = checks
        _publish(RESULTS / "m19_serving_receipt_t0.json", _json_bytes(serving))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-count", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    result = benchmark(args.query_count, args.warmup)
    out = Path(args.out) if args.out else RESULTS / f"m19_serving_benchmark_{args.query_count}.json"
    _publish(out, _json_bytes(result))
    print(json.dumps({"official": result["official"], "query_count": result["query_count"],
                      "ratios": result["ratios"]}, indent=2, sort_keys=True))
    if result["official"] and not all(result["checks"].values()):
        raise SystemExit("M19 SERVING STOP: registered latency or serving gate failed")


if __name__ == "__main__":
    main()
