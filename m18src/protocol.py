"""Build and seal leakage-safe M18 training/development/confirmation query protocols."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import WORK, admit_read, atomic_write_bytes, registry, sha_file, sha_json, write_json

MAINTAINER = {"OWNER", "MEMBER", "COLLABORATOR"}
ANSWER_CUES = re.compile(
    r"(?i)\b(?:because|you (?:can|should|need)|try |use |fixed|solution|workaround|supported|"
    r"not supported|the (?:issue|problem|reason)|this (?:is|was)|we (?:need|use|fixed|support))\b")
ERROR_CUES = re.compile(r"(?i)\b(?:error|failed?|failure|panic|crash|exception|not working|oom|out of memory|timeout|stack trace)\b")
CONFIG_CUES = re.compile(r"(?i)\b(?:config|setting|api|grpc|rest|port|cluster|replica|shard|deploy|docker|kubernetes|yaml|collection)\b")
EXACT_CUES = re.compile(r"(?:\b\d+(?:\.\d+)+\b|\b(?:HTTP|gRPC)\s*\d{3}\b|`[^`]+`|[/_.:-]|\b[A-Z][A-Z0-9_]{3,}\b)")
JARGON_CUES = re.compile(r"(?i)\b(?:qdrant|hnsw|wal|mmap|rocksdb|quantization|payload|segment|optimizer|ef_construct|ef_search|grpc|raft|rps)\b")
ALIAS_RE = re.compile(r"\b([A-Z][A-Za-z][A-Za-z0-9 /+_-]{3,60})\s+\(([A-Z][A-Z0-9_-]{1,12})\)")


def _read_jsonl(path):
    with open(admit_read(path), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _stratum(text):
    if ERROR_CUES.search(text):
        return "error_troubleshooting"
    if EXACT_CUES.search(text):
        return "exact_version_numeric"
    if CONFIG_CUES.search(text):
        return "config_api_ops"
    if JARGON_CUES.search(text):
        return "alias_jargon"
    return "concept_howto"


def _family_shape(text):
    """Union exact/backport-like titles without joining arbitrary cross-referenced threads."""
    t = str(text).lower()
    t = re.sub(r"\b(?:v?\d+(?:\.\d+){1,3}|backport|cherry[- ]pick)\b", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def _query_id(kind, source):
    return "m18q:" + kind + ":" + str(source)


def structural_candidates(corpus):
    by_thread = defaultdict(list)
    by_doc = {r["doc_id"]: r for r in corpus}
    for row in corpus:
        if row["artifact_id"].startswith("gh:thread:"):
            by_thread[row["artifact_id"]].append(row)
    candidates = []
    for family, rows in sorted(by_thread.items()):
        openings = [r for r in rows if r["kind"] == "issue_opening"]
        if not openings:
            continue
        opening = openings[0]
        answers = sorted((r for r in rows if r["kind"] == "issue_comment"
                          and r.get("author_association") in MAINTAINER
                          and len(r["text"].split()) >= 12 and ANSWER_CUES.search(r["text"])),
                         key=lambda r: (r.get("timestamp") or "", r["doc_id"]))
        if not answers:
            continue
        target = answers[0]
        text = str(opening.get("title") or "").strip()
        if len(text.split()) < 3 or text.lower() in target["text"].lower():
            continue
        candidates.append({"query_id": _query_id("issue", opening["github_number"]),
                           "text": text, "family": family,
                           "near_duplicate_family": _family_shape(text),
                           "source_doc": opening["doc_id"], "target_doc": target["doc_id"],
                           "timestamp": opening.get("timestamp") or "",
                           "stratum": _stratum(text),
                           "relevance_reason": "distinct later maintainer answer with explanatory cue",
                           "label_provenance": {"rule": "author_association+answer_cue",
                                                "author_association": target.get("author_association")}})

    # Review question -> explicit in-reply-to response. The relation is structural, not a
    # teacher judgment, and both units remain individually searchable.
    review = {r.get("github_id"): r for r in corpus if r["kind"] == "review_comment"}
    for target in sorted(review.values(), key=lambda r: r["doc_id"]):
        parent = review.get(target.get("in_reply_to_id"))
        if not parent or "?" not in parent["text"] or target.get("author_association") not in MAINTAINER:
            continue
        if len(target["text"].split()) < 8:
            continue
        text = parent["text"].strip()
        candidates.append({"query_id": _query_id("review", parent["github_id"]),
                           "text": text, "family": parent["artifact_id"],
                           "near_duplicate_family": _family_shape(text),
                           "source_doc": parent["doc_id"], "target_doc": target["doc_id"],
                           "timestamp": parent.get("timestamp") or "", "stratum": _stratum(text),
                           "relevance_reason": "maintainer review reply linked by in_reply_to_id",
                           "label_provenance": {"rule": "review_in_reply_to",
                                                "author_association": target.get("author_association")}})
    # Validate targets after construction; no artifact-level relevance fallback exists.
    return [q for q in candidates if q["target_doc"] in by_doc]


def _split(candidates, reg):
    cfg = reg["split"]
    need_dev, need_conf = int(cfg["development_per_stratum"]), int(cfg["confirmation_per_stratum"])
    # One near-duplicate shape owns one family. Keep the newest representative, then exclude the
    # entire shape from training if selected for either held-out surface.
    best = {}
    for q in candidates:
        key = q["near_duplicate_family"] or q["family"]
        cur = best.get(key)
        if cur is None or (q["timestamp"], q["query_id"]) > (cur["timestamp"], cur["query_id"]):
            best[key] = q
    grouped = defaultdict(list)
    for q in best.values():
        grouped[q["stratum"]].append(q)
    development, confirmation, heldout_shapes = [], [], set()
    realized = {}
    for stratum in reg["strata"]:
        rows = sorted(grouped[stratum], key=lambda q: (q["timestamp"], q["query_id"]), reverse=True)
        pool = rows[:need_dev + need_conf]
        # Three confirmation positions per chronological block of ten yields exactly 15/35 at
        # the target size while interleaving recency rather than putting all confirmation last.
        conf_positions = {i for i in range(len(pool)) if i % 10 in (0, 3, 6)}
        conf = [q for i, q in enumerate(pool) if i in conf_positions][:need_conf]
        conf_ids = {q["query_id"] for q in conf}
        dev = [q for q in pool if q["query_id"] not in conf_ids][:need_dev]
        development.extend(dev); confirmation.extend(conf)
        heldout_shapes.update(q["near_duplicate_family"] for q in dev + conf)
        realized[stratum] = {"available_families": len(rows), "development": len(dev),
                             "confirmation": len(conf)}
    train = [q for q in candidates if q["near_duplicate_family"] not in heldout_shapes]
    return train, development, confirmation, realized


def _documentation_training(corpus, excluded_artifacts, cap):
    rows = []
    for doc in corpus:
        if doc["kind"] != "repository_text" or doc["artifact_id"] in excluded_artifacts:
            continue
        heading = str(doc.get("title") or "").strip()
        if len(heading.split()) < 2 or heading.lower() in {"readme.md", "contents", "overview"}:
            continue
        # Deterministic query view, disclosed as a heading view rather than model-generated text.
        rows.append({"query_id": _query_id("doc", doc["doc_id"]), "text": heading,
                     "family": doc["artifact_id"], "near_duplicate_family": _family_shape(heading),
                     "source_doc": doc["doc_id"], "target_doc": doc["doc_id"],
                     "timestamp": "", "stratum": _stratum(heading),
                     "relevance_reason": "repository heading paired with its section",
                     "label_provenance": {"rule": "documentation_heading"}})
        if len(rows) >= cap:
            break
    return rows


def _alias_training(corpus, excluded_artifacts, cap):
    out = []
    for doc in corpus:
        if doc["artifact_id"] in excluded_artifacts:
            continue
        for expansion, short in ALIAS_RE.findall(doc["text"]):
            pair = hashlib.sha256((doc["doc_id"] + "\0" + expansion + "\0" + short).encode()).hexdigest()[:20]
            for view, text in (("a", short), ("b", expansion)):
                out.append({"query_id": _query_id("alias", pair + view), "text": text,
                            "family": doc["artifact_id"], "near_duplicate_family": pair,
                            "source_doc": doc["doc_id"], "target_doc": doc["doc_id"],
                            "timestamp": "", "stratum": "alias_jargon",
                            "alias_pair_id": pair, "alias_view": view,
                            "relevance_reason": "same-source parenthetical abbreviation evidence",
                            "label_provenance": {"rule": "expansion_parenthetical"}})
            if len(out) >= cap:
                return out
    return out


def _write_rows(path, rows):
    payload = b"".join((json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n").encode()
                       for r in rows)
    return atomic_write_bytes(path, payload)


def _surface_files(root, name, queries):
    qpath = _write_rows(root / f"{name}_queries.jsonl", queries)
    qrels = {q["query_id"]: {q["target_doc"]: 2} for q in queries}
    rpath = root / f"{name}_qrels.json"
    write_json(rpath, qrels)
    return {"queries": str(qpath.name), "queries_sha256": sha_file(qpath),
            "qrels": str(rpath.name), "qrels_sha256": sha_file(rpath), "count": len(queries)}


def build(corpus_path=None, out_root=None, registry_data=None):
    reg = registry_data or registry()
    out = Path(out_root or WORK / "derived" / "protocol")
    out.mkdir(parents=True, exist_ok=True)
    corpus_path = Path(corpus_path or WORK / "derived" / "corpus.jsonl")
    corpus = list(_read_jsonl(corpus_path))
    candidates = structural_candidates(corpus)
    train_struct, dev, conf, realized = _split(candidates, reg)
    min_dev = int(reg["split"]["minimum_development_per_available_stratum"])
    min_conf = int(reg["split"]["minimum_confirmation_per_available_stratum"])
    short = {s: v for s, v in realized.items()
             if v["available_families"] >= min_dev + min_conf
             and (v["development"] < min_dev or v["confirmation"] < min_conf)}
    if short:
        raise SystemExit(f"M18 PROTOCOL REFUSED: adequately populated strata miss minima: {short}")
    excluded_artifacts = {q["family"] for q in dev + conf}
    cap = int(reg["corpus"]["training_query_views_max"])
    docs = _documentation_training(corpus, excluded_artifacts, max(0, cap - len(train_struct)))
    aliases = _alias_training(corpus, excluded_artifacts, max(0, cap - len(train_struct) - len(docs)))
    train = (train_struct + docs + aliases)[:cap]
    # No held-out family or near-duplicate title shape can supply a gradient-bearing view.
    heldout_family = {q["family"] for q in dev + conf}
    heldout_shape = {q["near_duplicate_family"] for q in dev + conf}
    leaks = [q["query_id"] for q in train if q["family"] in heldout_family
             or q["near_duplicate_family"] in heldout_shape]
    if leaks:
        raise SystemExit(f"M18 PROTOCOL REFUSED: {len(leaks)} held-out families leak into training")
    train_path = _write_rows(out / "training_queries.jsonl", train)
    dev_files = _surface_files(out, "development", dev)
    conf_files = _surface_files(out, "confirmation", conf)
    overlap = []
    corpus_by_id = {r["doc_id"]: r for r in corpus}
    for q in dev + conf:
        target = corpus_by_id[q["target_doc"]]["text"].lower()
        qtoks = set(re.findall(r"\w+", q["text"].lower()))
        ttoks = set(re.findall(r"\w+", target))
        overlap.append(len(qtoks & ttoks) / max(1, len(qtoks)))
    manifest = {"_schema": "m18-protocol-manifest-v1", "split_version": reg["versions"]["split"],
                "qrels_version": reg["versions"]["qrels"], "candidates": len(candidates),
                "training_queries": len(train), "training_structural": len(train_struct),
                "training_document_headings": len(docs), "training_alias_views": len(aliases),
                "realized_strata": realized, "development": dev_files,
                "confirmation": conf_files, "confirmation_sealed": True,
                "training_queries_sha256": sha_file(train_path),
                "family_overlap_train_vs_heldout": 0, "near_duplicate_overlap_train_vs_heldout": 0,
                "relevant_source_opening_self_hits": 0,
                "query_token_overlap_with_answer": {"mean": float(sum(overlap) / max(1, len(overlap))),
                                                    "exact_query_substring_rate": float(sum(
                                                        q["text"].lower() in corpus_by_id[q["target_doc"]]["text"].lower()
                                                        for q in dev + conf) / max(1, len(dev) + len(conf)))},
                "semantics": "held-out questions over complete pinned snapshot; not historical-as-of-question",
                "structural_relevance": "only distinct maintainer answer/reply spans; thread siblings are not automatically relevant"}
    manifest["sha256"] = sha_json(manifest)
    write_json(out / "protocol_manifest.json", manifest)
    if out_root is None:
        write_json(Path(__file__).resolve().parents[1] / "results/m18_protocol_manifest.json", manifest)
    return manifest


def load_surface(name, out_root=None):
    if name not in ("development", "confirmation"):
        raise ValueError(name)
    root = Path(out_root or WORK / "derived" / "protocol")
    queries = list(_read_jsonl(root / f"{name}_queries.jsonl"))
    qrels = json.loads(admit_read(root / f"{name}_qrels.json").read_text())
    return queries, qrels


def claim_confirmation(identity, out_root=None):
    """Begin the one confirmation transaction; a stale/complete receipt is never overwritten."""
    root = Path(out_root or WORK / "derived" / "protocol")
    manifest = json.loads(admit_read(root / "protocol_manifest.json").read_text())
    receipt = root / "confirmation_read_receipt.json"
    payload = {"_schema": "m18-confirmation-read-v1", "state": "started", "reads": 1,
               "protocol_sha256": manifest["sha256"], "identity": dict(identity)}
    try:
        fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as e:
        raise SystemExit(f"M18 CONFIRMATION REFUSED: receipt already exists at {receipt}") from e
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")
        f.flush(); os.fsync(f.fileno())
    return receipt


def finish_confirmation(receipt, result_sha256):
    data = json.loads(admit_read(receipt).read_text())
    if data.get("state") != "started":
        raise SystemExit("M18 CONFIRMATION REFUSED: transaction is not started")
    data.update({"state": "complete", "result_sha256": result_sha256})
    write_json(receipt, data)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args(argv)
    print(json.dumps(build(args.corpus, args.out_root), indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
