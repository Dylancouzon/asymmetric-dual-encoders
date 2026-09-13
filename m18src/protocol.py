"""Build and seal leakage-safe M18 training/development/confirmation query protocols."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import random
from difflib import SequenceMatcher
from collections import Counter, defaultdict
from pathlib import Path

from common import REPO, WORK, admit_read, atomic_write_bytes, registry, sha_file, sha_json, write_json

MAINTAINER = {"OWNER", "MEMBER", "COLLABORATOR"}
ANSWER_CUES = re.compile(
    r"(?i)\b(?:because|you should|try |use |set |configure|fixed|resolved|implemented|"
    r"solution|workaround|is supported|not supported|the (?:issue|problem|reason)|"
    r"this (?:happens|fails|is caused|was fixed)|we (?:use|fixed|support))\b")
INFO_REQUEST = re.compile(
    r"(?is)(?:\b(?:could|can|would) you\b|\bplease\b|\bwe (?:need|want)\b).{0,80}"
    r"\b(?:provide|share|attach|send|check|confirm|try|reproduce|logs?|data|versions?|"
    r"reproducer|reproduction|more information|details|stack trace)\b|"
    r"\b(?:provide|share|attach|send)\b.{0,60}\b(?:logs?|data|versions?|reproducer|"
    r"reproduction|more information|details|stack trace)\b|"
    r"\b(?:need|want)\b.{0,60}\b(?:logs?|data|versions?|reproducer|reproduction|"
    r"more information|details|stack trace)\b")
RESOLUTION_CUES = re.compile(r"(?i)\b(?:fixed|resolves?|implemented|solution|workaround|"
                             r"closing|closed by|merged in|this (?:happens|fails|is caused))\b")
STRONG_EXPLANATION = re.compile(
    r"(?i)\b(?:because|caused by|due to|root cause|the reason|recommend|you should|should (?:use|set)|"
    r"set |configure|is supported|not supported|solution|workaround|resolved|implemented|"
    r"this (?:happens|fails|is caused))\b")
CLARIFICATION = re.compile(
    r"(?is)(?:\b(?:could|can|would) you\b|\b(?:what|which|how (?:big|many|much))\b.{0,100}\?|"
    r"\b(?:not (?:entirely )?sure|need to check|question remains|open to .*suggestions)\b)")
NON_RESOLUTION = re.compile(
    r"(?is)(?:\b(?:not|isn't|wasn't|cannot|can't|do not|don't)\b.{0,35}"
    r"\b(?:fixed|resolved|implemented|solution|workaround)\b|"
    r"\b(?:is|was|has|does)\b.{0,25}\b(?:fixed|resolved|implemented)\b[^.!?]{0,20}\?)")
STATUS_TERMS = re.compile(
    r"(?i)\b(?:ci|codespell|workflow|rebase|rebased|push|pushed|typo|build|tests?|compile)\b")
STATUS_ONLY = re.compile(
    r"(?i)\b(?:red ci|ci is green|codespell|workflow|rebase|forgot to push|typo|"
    r"not caused by this pr|maintainer approval)\b")
QUESTION_SHAPE = re.compile(
    r"(?i)(?:\?|\bhow to\b|^\s*(?:why|how|what|when|where|is|are|can|could|does|do|should)\b)")
PROJECT_LINK = re.compile(r"https?://(?:github\.com/qdrant/qdrant/(?:pull|commit|blob)/|qdrant\.tech/documentation/)")
ERROR_CUES = re.compile(r"(?i)\b(?:error|failed?|failure|panic|crash|exception|not working|oom|out of memory|timeout|stack trace)\b")
CONFIG_CUES = re.compile(r"(?i)\b(?:config|setting|api|grpc|rest|port|cluster|replica|shard|deploy|docker|kubernetes|yaml|collection)\b")
EXACT_CUES = re.compile(r"(?:\bv?\d+(?:\.\d+)+\b|\b(?:HTTP|gRPC)\s*\d{3}\b|"
                        r"`[^`]*[A-Za-z_./:-][^`]*`|\b[A-Z][A-Z0-9_]{3,}\b|"
                        r"\b[a-z][a-z0-9]*(?:[_.:/-][a-z0-9]+)+\b)")
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


def _status_matches_query(query, answer):
    qt = {x.lower() for x in STATUS_TERMS.findall(query)}
    at = {x.lower() for x in STATUS_TERMS.findall(answer)}
    return bool(qt & at)


def _credible_issue_answer(opening, row, ev):
    """High-precision structural label gate for issue/PR opening -> answer span."""
    text = row["text"]
    if NON_RESOLUTION.search(text):
        return False
    explicit = bool(ev["explicit_resolution"])
    linked_explanation = bool(ev["project_link"] and STRONG_EXPLANATION.search(text))
    event_explanation = bool(ev["closing_or_link_event"] and STRONG_EXPLANATION.search(text))
    if not (explicit or linked_explanation or event_explanation):
        return False
    if INFO_REQUEST.search(text):
        return False
    if CLARIFICATION.search(text) and not explicit:
        return False
    if STATUS_ONLY.search(text) and not _status_matches_query(opening.get("title", ""), text):
        return False
    return True


def _credible_review_answer(row):
    text = row["text"]
    return bool(ANSWER_CUES.search(text) and not INFO_REQUEST.search(text)
                and not CLARIFICATION.search(text) and not NON_RESOLUTION.search(text))


def _has_prose_question(text):
    prose = re.sub(r"(?s)```.*?```", " ", text)
    prose = re.sub(r"`[^`]*`", " ", prose)
    return "?" in prose


def _credible_audit_query(opening, text, stratum):
    # PR work-item titles are useful training context, but are not concept/how-to audit questions
    # unless the author actually phrased one as a question. Other strata deliberately admit
    # errors, identifiers and short terms.
    return not (stratum == "concept_howto"
                and opening.get("kind") == "pull_request_opening"
                and not QUESTION_SHAPE.search(text))


def structural_candidates(corpus):
    by_thread = defaultdict(list)
    by_doc = {r["doc_id"]: r for r in corpus}
    for row in corpus:
        if row["artifact_id"].startswith("gh:thread:"):
            by_thread[row["artifact_id"]].append(row)
    candidates = []
    for family, rows in sorted(by_thread.items()):
        openings = [r for r in rows if r["kind"] in ("issue_opening", "pull_request_opening")
                    and int(r.get("chunk_index", 1)) == 1]
        if not openings:
            continue
        opening = openings[0]
        def evidence(row):
            closed_after = (opening.get("state") == "closed" and opening.get("closed_at")
                            and str(opening["closed_at"]) >= str(row.get("timestamp") or ""))
            linked = any(PROJECT_LINK.search(str(x)) for x in row.get("outbound_links", []))
            events = any((e.get("timestamp") or "") >= (row.get("timestamp") or "")
                         and e.get("kind") == "issue_event"
                         and re.search(r"(?i)\b(?:closed|merged|referenced|committed)\b", e.get("text", ""))
                         for e in rows)
            explicit = bool(RESOLUTION_CUES.search(row["text"]))
            return {"explicit_resolution": explicit, "thread_closed_after_answer": bool(closed_after),
                    "project_link": linked, "closing_or_link_event": events}
        answers = sorted((r for r in rows if r["kind"] == "issue_comment"
                          and r.get("author_association") in MAINTAINER
                          and (r.get("timestamp") or "") > (opening.get("timestamp") or "")
                          and len(r["text"].split()) >= 16 and ANSWER_CUES.search(r["text"])
                          and _credible_issue_answer(opening, r, evidence(r))),
                         key=lambda r: (r.get("timestamp") or "", r["doc_id"]))
        if not answers:
            continue
        target = answers[0]
        text = str(opening.get("title") or "").strip()
        stratum = _stratum(text)
        if (len(text.split()) < 3 or text.lower() in target["text"].lower()
                or not _credible_audit_query(opening, text, stratum)):
            continue
        candidates.append({"query_id": _query_id("issue", opening["github_number"]),
                           "text": text, "family": family,
                           "near_duplicate_family": _family_shape(text),
                           "source_doc": opening["doc_id"], "target_doc": target["doc_id"],
                           "source_exclusion_docs": sorted(r["doc_id"] for r in rows
                               if r["kind"] == opening["kind"]
                               and r.get("github_id") == opening.get("github_id")),
                           "timestamp": opening.get("timestamp") or "",
                           "stratum": stratum,
                           "relevance_reason": "distinct later maintainer answer with resolution/action evidence",
                           "label_provenance": {"rule": "later_maintainer+substantive_resolution-v4",
                                                "author_association": target.get("author_association"),
                                                "resolution_evidence": evidence(target)}})

    # Review question -> explicit in-reply-to response. The relation is structural, not a
    # teacher judgment, and both units remain individually searchable.
    review = {}
    for row in corpus:
        if row["kind"] == "review_comment":
            review.setdefault(row.get("github_id"), []).append(row)
    for chunks in sorted(review.values(), key=lambda rs: rs[0]["doc_id"]):
        target = next((r for r in chunks if _credible_review_answer(r)), chunks[0])
        parents = review.get(target.get("in_reply_to_id"))
        parent = parents[0] if parents else None
        if (not parent or not _has_prose_question(parent["text"])
                or str(parent.get("author", "")).endswith("[bot]")
                or target.get("author_association") not in MAINTAINER):
            continue
        if (len(target["text"].split()) < 12 or not _credible_review_answer(target)
                or (target.get("timestamp") or "") <= (parent.get("timestamp") or "")):
            continue
        text = parent["text"].strip()
        stratum = _stratum(text)
        # Without a project-specific error/config/identifier/jargon cue, a review question relies
        # on unseen diff context ("this", "here", etc.) and is not a standalone concept query.
        if stratum == "concept_howto":
            continue
        candidates.append({"query_id": _query_id("review", parent["github_id"]),
                           "text": text, "family": parent["artifact_id"],
                           "near_duplicate_family": _family_shape(text),
                           "source_doc": parent["doc_id"], "target_doc": target["doc_id"],
                           "source_exclusion_docs": sorted(r["doc_id"] for r in parents),
                           "timestamp": parent.get("timestamp") or "", "stratum": stratum,
                           "relevance_reason": "maintainer review reply linked by in_reply_to_id",
                           "label_provenance": {"rule": "review_in_reply_to",
                                                "author_association": target.get("author_association")}})
    # Validate targets, then form a corpus-wide connected family relation over threads,
    # duplicate answer spans, title/backport shapes and explicit Qdrant thread links.
    candidates = [q for q in candidates if q["target_doc"] in by_doc]
    return _assign_union_families(candidates, corpus, by_doc)


def _assign_union_families(candidates, corpus, by_doc):
    artifacts = sorted({q["family"] for q in candidates})
    parent = {x: x for x in artifacts}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    by_shape, by_answer, by_source = {}, {}, {}
    for q in candidates:
        for table, key in ((by_shape, q["near_duplicate_family"]),
                           (by_answer, by_doc[q["target_doc"]].get("normalized_text_sha256")),
                           (by_source, by_doc.get(q["source_doc"], {}).get("normalized_text_sha256"))):
            if key and key in table:
                union(q["family"], table[key])
            elif key:
                table[key] = q["family"]
    # Conservative fuzzy union for punctuation/reworded backport titles. Comparisons are blocked
    # by the first two normalized tokens to avoid quadratic all-project similarity.
    blocks = defaultdict(list)
    for q in candidates:
        shape = q["near_duplicate_family"]
        toks = shape.split()
        blocks[tuple(toks[:2])].append((q, set(toks)))
    for rows in blocks.values():
        for i, (a, at) in enumerate(rows):
            for b, bt in rows[i + 1:]:
                jaccard = len(at & bt) / max(1, len(at | bt))
                if jaccard >= 0.82 or SequenceMatcher(None, a["near_duplicate_family"],
                                                       b["near_duplicate_family"]).ratio() >= 0.92:
                    union(a["family"], b["family"])
    link_re = re.compile(r"github\.com/qdrant/qdrant/(?:issues|pull)/(\d+)")
    for row in corpus:
        a = row.get("artifact_id")
        if a not in parent:
            continue
        for link in row.get("outbound_links", []):
            m = link_re.search(link)
            if m:
                union(a, f"gh:thread:{m.group(1)}")
    members = defaultdict(set)
    for a in artifacts:
        members[find(a)].add(a)
    for q in candidates:
        root = find(q["family"])
        q["family_group"] = "m18fam:" + hashlib.sha256("\n".join(sorted(members[root])).encode()).hexdigest()[:20]
        q["family_members"] = sorted(members[root])
    return candidates


def prospective_adjudication_pool(candidates, reg):
    """Newest bounded one-query-per-family pool, formed before any audit split exists."""
    cap = int(reg["evaluation"]["qrel_adjudication"]["pool_per_stratum"])
    best = {}
    for q in candidates:
        key = q["family_group"]
        cur = best.get(key)
        if cur is None or (q["timestamp"], q["query_id"]) > (cur["timestamp"], cur["query_id"]):
            best[key] = q
    grouped = defaultdict(list)
    for q in best.values():
        grouped[q["stratum"]].append(q)
    pool = []
    for stratum in reg["strata"]:
        rows = sorted(grouped[stratum], key=lambda q: (q["timestamp"], q["query_id"]), reverse=True)
        pool.extend(rows[:cap])
    return sorted(pool, key=lambda q: q["query_id"])


def adjudication_record(q, corpus_by_id):
    source, target = corpus_by_id[q["source_doc"]], corpus_by_id[q["target_doc"]]
    row = {"query_id": q["query_id"], "query": q["text"],
           "source_context": source["text"], "source_kind": source["kind"],
           "source_path": source.get("path"), "target": target["text"],
           "target_kind": target["kind"], "target_path": target.get("path"),
           "proposed_stratum": q["stratum"], "structural_provenance": q["label_provenance"]}
    row["candidate_sha256"] = sha_json(row)
    return row


def apply_adjudications(candidates, reg, corpus_by_id):
    cfg = reg["evaluation"].get("qrel_adjudication")
    if not cfg:
        return candidates, None
    pool = prospective_adjudication_pool(candidates, reg)
    path = Path(cfg["decisions_path"])
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        raise SystemExit(f"M18 PROTOCOL REFUSED: prospective adjudications absent at {path}")
    decisions = list(_read_jsonl(path))
    by_qid = {r["query_id"]: r for r in decisions}
    if len(by_qid) != len(decisions) or set(by_qid) != {q["query_id"] for q in pool}:
        raise SystemExit("M18 PROTOCOL REFUSED: adjudication decisions do not exactly cover pool")
    allowed_labels = {"clear", "partial", "bad"}
    accepted = []
    for q in pool:
        d = by_qid[q["query_id"]]
        candidate_sha256 = adjudication_record(q, corpus_by_id)["candidate_sha256"]
        if d.get("candidate_sha256") != candidate_sha256:
            raise SystemExit(f"M18 PROTOCOL REFUSED: stale adjudication for {q['query_id']}")
        if d.get("label") not in allowed_labels or d.get("stratum") not in reg["strata"]:
            raise SystemExit(f"M18 PROTOCOL REFUSED: invalid adjudication for {q['query_id']}")
        if not str(d.get("reason", "")).strip():
            raise SystemExit(f"M18 PROTOCOL REFUSED: reason absent for {q['query_id']}")
        if d["label"] == "clear":
            kept = dict(q)
            kept["stratum"] = d["stratum"]
            kept["label_provenance"] = {**q["label_provenance"],
                "adjudication": {"judge": cfg["judge"], "label": "clear",
                                 "reason": d["reason"], "candidate_sha256": d["candidate_sha256"]}}
            accepted.append(kept)
    displayed_path = (str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path))
    meta = {"judge": cfg["judge"], "pool": len(pool), "accepted_clear": len(accepted),
            "labels": dict(sorted(Counter(d["label"] for d in decisions).items())),
            "decisions_path": displayed_path, "decisions_sha256": sha_file(path)}
    return accepted, meta


def _split(candidates, reg):
    cfg = reg["split"]
    need_dev, need_conf = int(cfg["development_per_stratum"]), int(cfg["confirmation_per_stratum"])
    # One connected artifact/duplicate family supplies at most one audit query.
    best = {}
    for q in candidates:
        key = q["family_group"]
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
        total = min(len(rows), need_dev + need_conf)
        if total >= int(cfg["minimum_development_per_available_stratum"]) + int(cfg["minimum_confirmation_per_available_stratum"]):
            dev_n = min(need_dev, total - int(cfg["minimum_confirmation_per_available_stratum"]))
            conf_n = min(need_conf, total - dev_n)
        else:
            conf_n = min(need_conf, max(0, round(total * need_conf / (need_dev + need_conf))))
            dev_n = total - conf_n
        pool = rows[:total]
        # Three confirmation positions per chronological block of ten yields exactly 15/35 at
        # the target size while interleaving recency rather than putting all confirmation last.
        rng = random.Random(int(cfg.get("seed", 18001)) ^ int(hashlib.sha256(stratum.encode()).hexdigest()[:16], 16))
        conf_positions = set()
        for lo in range(0, len(pool), 10):
            positions = list(range(lo, min(lo + 10, len(pool))))
            rng.shuffle(positions)
            conf_positions.update(positions[:min(3, len(positions))])
        conf = [q for i, q in enumerate(pool) if i in conf_positions][:conf_n]
        if len(conf) < conf_n:
            used = {q["query_id"] for q in conf}
            conf.extend(q for q in reversed(pool) if q["query_id"] not in used
                        and len(conf) < conf_n)
        conf_ids = {q["query_id"] for q in conf}
        dev = [q for q in pool if q["query_id"] not in conf_ids][:dev_n]
        development.extend(dev); confirmation.extend(conf)
        heldout_shapes.update(q["family_group"] for q in dev + conf)
        realized[stratum] = {"available_families": len(rows), "development": len(dev),
                             "confirmation": len(conf)}
    train = [q for q in candidates if q["family_group"] not in heldout_shapes]
    return train, development, confirmation, realized


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
                            "source_doc": doc["doc_id"], "target_doc": None,
                            "timestamp": "", "stratum": "alias_jargon",
                            "alias_pair_id": pair, "alias_view": view,
                            "relevance_reason": "same-source parenthetical evidence; consistency-only, no relevance label",
                            "label_provenance": {"rule": "expansion_parenthetical"}})
            if len(out) >= cap:
                return out
    return out


def _query_only_training(structural, labeled, heldout_groups, corpus_by_id):
    """Keep non-held-out issue/PR titles as teacher-only views, never as inferred positives."""
    labeled_ids = {q["query_id"] for q in labeled}
    out = []
    for q in structural:
        if q["query_id"] in labeled_ids or q["family_group"] in heldout_groups:
            continue
        if corpus_by_id[q["source_doc"]]["kind"] not in {"issue_opening", "pull_request_opening"}:
            continue
        row = dict(q)
        row["target_doc"] = None
        row["relevance_reason"] = "natural issue/PR title; teacher-only training view without qrel"
        row["label_provenance"] = {"rule": "source-title-query-only", "positive_label": False}
        out.append(row)
    return sorted(out, key=lambda q: q["query_id"])


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
    corpus_by_id = {r["doc_id"]: r for r in corpus}
    structural = structural_candidates(corpus)
    candidates, adjudication = apply_adjudications(structural, reg, corpus_by_id)
    train_struct, dev, conf, realized = _split(candidates, reg)
    min_dev = int(reg["split"]["minimum_development_per_available_stratum"])
    min_conf = int(reg["split"]["minimum_confirmation_per_available_stratum"])
    short = {s: v for s, v in realized.items()
             if v["available_families"] >= min_dev + min_conf
             and (v["development"] < min_dev or v["confirmation"] < min_conf)}
    if short:
        raise SystemExit(f"M18 PROTOCOL REFUSED: adequately populated strata miss minima: {short}")
    excluded_artifacts = {a for q in dev + conf for a in q["family_members"]}
    heldout_family = {a for q in dev + conf for a in q["family_members"]}
    heldout_group = {q["family_group"] for q in dev + conf}
    query_only = _query_only_training(structural, train_struct, heldout_group, corpus_by_id)
    structural_train = train_struct + query_only
    cap = int(reg["corpus"]["training_query_views_max"])
    docs = []  # heading==target self retrieval is intentionally not admitted.
    aliases = _alias_training(corpus, excluded_artifacts, max(0, cap - len(structural_train)))
    train = (structural_train + aliases)[:cap]
    # No held-out family or near-duplicate title shape can supply a gradient-bearing view.
    heldout_target_hash = {corpus_by_id[q["target_doc"]]["normalized_text_sha256"] for q in dev + conf
                           if q.get("target_doc")}
    leaks = [q["query_id"] for q in train if q["family"] in heldout_family
             or q.get("family_group") in heldout_group
             or (q.get("target_doc") and
                 corpus_by_id[q["target_doc"]]["normalized_text_sha256"] in heldout_target_hash)]
    if leaks:
        raise SystemExit(f"M18 PROTOCOL REFUSED: {len(leaks)} held-out families leak into training")
    train_path = _write_rows(out / "training_queries.jsonl", train)
    dev_files = _surface_files(out, "development", dev)
    conf_files = _surface_files(out, "confirmation", conf)
    overlap = []
    for q in dev + conf:
        target = corpus_by_id[q["target_doc"]]["text"].lower()
        qtoks = set(re.findall(r"\w+", q["text"].lower()))
        ttoks = set(re.findall(r"\w+", target))
        overlap.append(len(qtoks & ttoks) / max(1, len(qtoks)))
    manifest = {"_schema": "m18-protocol-manifest-v1", "split_version": reg["versions"]["split"],
                "qrels_version": reg["versions"]["qrels"], "candidates": len(candidates),
                "structural_candidates_pre_adjudication": len(structural),
                "qrel_adjudication": adjudication,
                "training_queries": len(train), "training_structural": len(structural_train),
                "training_structural_labeled": len(train_struct),
                "training_structural_query_only": len(query_only),
                "training_document_headings": len(docs), "training_alias_views": len(aliases),
                "realized_strata": realized, "development": dev_files,
                "confirmation": conf_files, "confirmation_sealed": True,
                "corpus_sha256": sha_file(corpus_path),
                "training_queries_sha256": sha_file(train_path),
                "family_overlap_train_vs_heldout": len(
                    {q.get("family_group") for q in train} & heldout_group),
                "artifact_overlap_train_vs_heldout": len(
                    {q["family"] for q in train} & heldout_family),
                "answer_digest_overlap_train_vs_heldout": len(
                    {corpus_by_id[q["target_doc"]]["normalized_text_sha256"] for q in train
                     if q.get("target_doc")}
                    & heldout_target_hash),
                "development_confirmation_family_overlap": len(
                    {q["family_group"] for q in dev} & {q["family_group"] for q in conf}),
                "qrel_source_equals_target_count": sum(q["source_doc"] == q["target_doc"]
                                                        for q in train + dev + conf),
                "query_source_exclusion_documents": sum(
                    len(q.get("source_exclusion_docs", [q["source_doc"]])) for q in dev + conf),
                "query_token_overlap_with_answer": {"mean": float(sum(overlap) / max(1, len(overlap))),
                                                    "exact_query_substring_rate": float(sum(
                                                        q["text"].lower() in corpus_by_id[q["target_doc"]]["text"].lower()
                                                        for q in dev + conf) / max(1, len(dev) + len(conf)))},
                "semantics": "held-out questions over complete pinned snapshot; not historical-as-of-question",
                "structural_relevance": "only distinct maintainer answer/reply spans; thread siblings are not automatically relevant"}
    manifest["development_source_kinds"] = {
        stratum: dict(sorted(Counter(corpus_by_id[q["source_doc"]]["kind"]
                                    for q in dev if q["stratum"] == stratum).items()))
        for stratum in reg["strata"]}
    manifest["sha256"] = sha_json(manifest)
    write_json(out / "protocol_manifest.json", manifest)
    if out_root is None:
        write_json(Path(__file__).resolve().parents[1] / "results/m18_protocol_manifest.json", manifest)
    return manifest


def _verified_surface(name, root, manifest):
    block = manifest[name]
    qpath, rpath = root / block["queries"], root / block["qrels"]
    if sha_file(qpath) != block["queries_sha256"] or sha_file(rpath) != block["qrels_sha256"]:
        raise SystemExit(f"M18 {name.upper()} REFUSED: sealed surface hash mismatch")
    return list(_read_jsonl(qpath)), json.loads(admit_read(rpath).read_text())


def load_surface(name, out_root=None):
    if name != "development":
        if name == "confirmation":
            raise SystemExit("M18 CONFIRMATION REFUSED: use run_confirmation transaction")
        raise ValueError(name)
    root = Path(out_root or WORK / "derived" / "protocol")
    manifest = json.loads(admit_read(root / "protocol_manifest.json").read_text())
    return _verified_surface(name, root, manifest)


def lock_confirmation(identity, out_root=None):
    """Immutably lock the selected recipe/checkpoint and sealed-surface identities."""
    root = Path(out_root or WORK / "derived" / "protocol")
    manifest = json.loads(admit_read(root / "protocol_manifest.json").read_text())
    if not isinstance(identity, dict) or not identity or any(v in (None, "") for v in identity.values()):
        raise SystemExit("M18 CONFIRMATION REFUSED: decision identity must be nonempty")
    path = root / "confirmation_decision_lock.json"
    payload = {"_schema": "m18-confirmation-lock-v1", "state": "locked",
               "protocol_sha256": manifest["sha256"],
               "queries_sha256": manifest["confirmation"]["queries_sha256"],
               "qrels_sha256": manifest["confirmation"]["qrels_sha256"],
               "identity": dict(identity), "identity_sha256": sha_json(identity)}
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as e:
        old = json.loads(admit_read(path).read_text())
        if old != payload:
            raise SystemExit("M18 CONFIRMATION REFUSED: another decision is already locked") from e
        return path
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    return path


def run_confirmation(identity, evaluator, result_path, out_root=None):
    """Atomically claim the sole read, verify the lock/surface, evaluate, and seal its result."""
    root = Path(out_root or WORK / "derived" / "protocol")
    manifest = json.loads(admit_read(root / "protocol_manifest.json").read_text())
    lock_path = root / "confirmation_decision_lock.json"
    if not lock_path.exists():
        raise SystemExit("M18 CONFIRMATION REFUSED: decision lock is absent")
    lock = json.loads(admit_read(lock_path).read_text())
    expected = {"protocol_sha256": manifest["sha256"],
                "queries_sha256": manifest["confirmation"]["queries_sha256"],
                "qrels_sha256": manifest["confirmation"]["qrels_sha256"],
                "identity_sha256": sha_json(identity)}
    if lock.get("state") != "locked" or any(lock.get(k) != v for k, v in expected.items()):
        raise SystemExit("M18 CONFIRMATION REFUSED: lock, identity, or sealed surface changed")
    receipt = root / "confirmation_read_receipt.json"
    payload = {"_schema": "m18-confirmation-read-v1", "state": "started", "reads": 1,
               **expected, "identity": dict(identity)}
    try:
        fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as e:
        raise SystemExit(f"M18 CONFIRMATION REFUSED: receipt already exists at {receipt}") from e
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")
        f.flush(); os.fsync(f.fileno())
    queries, qrels = _verified_surface("confirmation", root, manifest)
    result = evaluator(queries, qrels)
    if not isinstance(result, dict):
        raise SystemExit("M18 CONFIRMATION REFUSED: evaluator must return a result object")
    result_path = Path(result_path)
    write_json(result_path, result)
    finish_confirmation(receipt, sha_file(result_path))
    return result


def claim_confirmation(*_args, **_kwargs):
    raise SystemExit("M18 CONFIRMATION REFUSED: claim-only access was removed; use run_confirmation")


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
