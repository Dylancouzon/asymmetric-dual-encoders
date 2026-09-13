"""Direct M18 baseline/checkpoint and one-shot confirmation evaluator."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import evaluate
import protocol
import system
from common import M18, REPO, WORK, admit_read, registry, sha_file, sha_json, write_json


def _encoder(path):
    return system._query_encoder(path)


def _exclude_query_sources(run, rows):
    """Remove every chunk of the verbatim query-bearing object before answer evaluation."""
    source_by_qid = {q["query_id"]: set(q.get("source_exclusion_docs", [q["source_doc"]]))
                     for q in rows}
    return {qid: {doc_id: score for doc_id, score in scores.items()
                  if doc_id not in source_by_qid.get(qid, set())}
            for qid, scores in run.items()}


def evaluate_rows(rows, qrels, bundles, *, include_stella=True, device="cuda"):
    reg = registry(); index = WORK / "derived/index"
    # ProjectMemory performs the complete index-manifest integrity check once.
    first_bundle = next(iter(bundles.values()))
    memory = system.ProjectMemory(index, first_bundle)
    texts, qids = [q["text"] for q in rows], [q["query_id"] for q in rows]
    prefetch = int(reg["retrieval"]["fusion"]["candidate_depth"])
    spare = max((len(q.get("source_exclusion_docs", [q["source_doc"]])) for q in rows), default=0)
    # Ask for enough spare results to remove every query-bearing source chunk below.
    bmrun = memory.bm25.run(qids, texts, prefetch + spare)
    dense, resources = {}, {}
    for name, path in bundles.items():
        enc = _encoder(path); start = time.perf_counter(); qv = enc.encode(texts)
        elapsed = time.perf_counter() - start
        dense[name] = evaluate.search(qv, memory.vectors, prefetch + spare, memory.ids, qids)
        resources[name] = {"query_encode_seconds": elapsed,
                           "milliseconds_per_query": elapsed * 1000 / max(1, len(rows)),
                           "resident_table_bytes": getattr(enc, "weight_bytes", None),
                           "bundle": str(path)}
    if include_stella:
        qv, teacher_manifest = system.encode_teacher_queries(texts, WORK / "derived/query_encodes",
                                                              name="development_raw", device=device)
        dense["stella"] = evaluate.search(qv, memory.vectors, prefetch + spare, memory.ids, qids)
        resources["stella"] = teacher_manifest
    raw_runs = {"bm25": bmrun, **dense}
    source_diagnostics = {}
    for name, run in raw_runs.items():
        excluded_by_query = {q["query_id"]: set(q.get("source_exclusion_docs", [q["source_doc"]]))
                             for q in rows}
        source_diagnostics[name] = {
            "source_opening_top10_rate_before_exclusion": float(np.mean([
                bool(excluded_by_query[q["query_id"]] & set(list(run[q["query_id"]])[:10]))
                for q in rows])) if rows else 0.0,
            "source_documents_excluded": sum(
                len(excluded_by_query[q["query_id"]] & set(run[q["query_id"]])) for q in rows),
        }
    bmrun = _exclude_query_sources(bmrun, rows)
    dense = {name: _exclude_query_sources(run, rows) for name, run in dense.items()}
    report = evaluate.compare_models(dense, bmrun, qrels, rows, prefetch=prefetch,
                                     replicates=reg["evaluation"]["bootstrap_replicates"],
                                     seed=reg["evaluation"]["bootstrap_seed"])
    report["diagnostics"] = source_diagnostics
    report["evaluation_exclusions"] = {
        "query_source_object_chunks": True,
        "reason": "indexed source chunks contain the query verbatim; qrels assess a distinct answer span",
    }
    report["resources"] = resources
    report["index_manifest_sha256"] = sha_file(index / "index_manifest.json")
    return report


def development(bundles, out, device="cuda", include_stella=True):
    rows, qrels = protocol.load_surface("development")
    report = evaluate_rows(rows, qrels, bundles, include_stella=include_stella, device=device)
    write_json(out, report); return report


def _bundle_args(values):
    out = {"zero_v1": registry()["models"]["zero_v1"]["source_path"]}
    for value in values:
        name, path = value.split("=", 1); out[name] = path
    return out


def _verify_confirmation_decision(decision, name, bundle):
    """Bind the one-shot read to the selected bytes and every evaluation identity."""
    if decision.get("_schema") != "m18-encoder-decision-v1":
        raise SystemExit("M18 CONFIRMATION REFUSED: unknown decision schema")
    selected = decision.get("selected_bundle", {})
    if name != selected.get("name"):
        raise SystemExit("M18 CONFIRMATION REFUSED: CLI bundle name differs from decision")
    root = Path(bundle).resolve()
    if root != Path(selected.get("path", "")).resolve():
        raise SystemExit("M18 CONFIRMATION REFUSED: CLI bundle path differs from decision")
    for filename, key in (("model.npz", "model_sha256"),
                          ("tokenizer.json", "tokenizer_sha256"),
                          ("config.json", "config_sha256")):
        if sha_file(root / filename) != selected.get(key):
            raise SystemExit(f"M18 CONFIRMATION REFUSED: selected {filename} hash changed")
    if sha_file(M18 / "execution-lock.json") != decision.get("execution_lock_sha256"):
        raise SystemExit("M18 CONFIRMATION REFUSED: execution lock differs from decision")
    if sha_file(REPO / "results/m18_index_manifest.json") != decision.get("index_manifest_sha256"):
        raise SystemExit("M18 CONFIRMATION REFUSED: index manifest differs from decision")
    if sha_file(REPO / "results/m18_protocol_manifest.json") != decision.get("protocol_manifest_sha256"):
        raise SystemExit("M18 CONFIRMATION REFUSED: protocol manifest differs from decision")
    reg = registry()
    recipe = {"retrieval": reg["retrieval"], "evaluation": reg["evaluation"],
              "serving": reg["serving"]}
    if sha_json(recipe) != decision.get("evaluation_recipe_sha256"):
        raise SystemExit("M18 CONFIRMATION REFUSED: evaluation recipe differs from decision")
    for record in decision.get("development_evidence", []):
        if sha_file(REPO / record["path"]) != record["sha256"]:
            raise SystemExit("M18 CONFIRMATION REFUSED: development evidence changed")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    dev = sub.add_parser("development")
    dev.add_argument("--bundle", action="append", default=[], metavar="NAME=PATH")
    dev.add_argument("--out", required=True); dev.add_argument("--device", default="cuda")
    dev.add_argument("--no-stella", action="store_true")
    conf = sub.add_parser("confirmation")
    conf.add_argument("--bundle", required=True, metavar="NAME=PATH")
    conf.add_argument("--decision", required=True); conf.add_argument("--out", required=True)
    conf.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.command == "development":
        result = development(_bundle_args(args.bundle), args.out, args.device, not args.no_stella)
    else:
        decision = json.loads(admit_read(args.decision).read_text())
        name, path = args.bundle.split("=", 1)
        _verify_confirmation_decision(decision, name, path)
        bundles = _bundle_args([args.bundle])
        result = protocol.run_confirmation(decision,
            lambda rows, qrels: evaluate_rows(rows, qrels, bundles, include_stella=False,
                                              device=args.device), args.out)
    print(json.dumps({"output": args.out, "identity": result.get("identity")}, sort_keys=True))


if __name__ == "__main__":
    main()
