"""Exact passage scoring, deterministic artifact collapse and unchanged Qdrant DBSF.

The DBSF formula is forked from ``m12src/qfusion.py``. M19 applies it only after each route has
collapsed up to 500 scored passages to at most 100 unique artifacts.
"""
from __future__ import annotations

import math

import numpy as np


def _validate_depths(passage_depth, artifact_depth):
    if int(passage_depth) != 500 or int(artifact_depth) != 100:
        raise ValueError("M19 collapse depths are registered at passage=500, artifact=100")


def collapse_passage_scores(scores, passages, *, excluded_artifacts=(), excluded_text_hashes=(),
                            passage_depth=500, artifact_depth=100):
    """Filter source/copies, rank passages, then retain one best passage per artifact."""
    _validate_depths(passage_depth, artifact_depth)
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(passages),) or not np.isfinite(values).all():
        raise ValueError("passage scores must be one finite value per passage")
    excluded_artifacts = set(map(str, excluded_artifacts))
    excluded_text_hashes = set(map(str, excluded_text_hashes))
    eligible = []
    for score, passage in zip(values, passages):
        artifact_id = str(passage["artifact_id"])
        text_hash = str(passage.get("normalized_text_sha256") or "")
        if artifact_id in excluded_artifacts or (text_hash and text_hash in excluded_text_hashes):
            continue
        eligible.append((float(score), str(passage["passage_id"]), artifact_id, passage))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    scored = eligible[:passage_depth]
    collapsed = []
    seen = set()
    for score, passage_id, artifact_id, passage in scored:
        if artifact_id in seen:
            continue
        seen.add(artifact_id)
        collapsed.append({
            "artifact_id": artifact_id,
            "score": score,
            "passage_id": passage_id,
            "passage": passage,
        })
        if len(collapsed) == artifact_depth:
            break
    return {
        "artifacts": collapsed,
        "eligible_passages": len(eligible),
        "scored_passages": len(scored),
        "unique_artifacts": len(collapsed),
        "shortfall": max(0, artifact_depth - len(collapsed)),
        "exclusion_before_truncation": True,
    }


def exact_dense_artifacts(query_vector, document_vectors, passages, **collapse_kwargs):
    query = np.asarray(query_vector, dtype=np.float32)
    documents = np.asarray(document_vectors)
    if query.ndim != 1 or documents.ndim != 2 or documents.shape[1] != query.shape[0]:
        raise ValueError("query/document vector shape mismatch")
    scores = documents.astype(np.float32, copy=False) @ query
    return collapse_passage_scores(scores, passages, **collapse_kwargs)


def _dbsf_normalized(rows):
    if not rows:
        return {}
    scores = [float(row["score"]) for row in rows]
    if len(scores) == 1:
        return {rows[0]["artifact_id"]: 0.5}
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    if variance == 0:
        return {row["artifact_id"]: 0.5 for row in rows}
    std = math.sqrt(variance)
    low, span = mean - 3 * std, 6 * std
    return {row["artifact_id"]: (float(row["score"]) - low) / span for row in rows}


def dbsf_artifacts(routes, limit=10, artifact_depth=100):
    """Fuse collapsed route scores and retain route passages/contributions for explanation."""
    if int(artifact_depth) != 100:
        raise ValueError("M19 DBSF depth is registered at 100 artifacts")
    totals, support = {}, {}
    for route_name, route in routes.items():
        rows = list(route["artifacts"][:artifact_depth])
        normalized = _dbsf_normalized(rows)
        by_id = {row["artifact_id"]: row for row in rows}
        for artifact_id, contribution in normalized.items():
            totals[artifact_id] = totals.get(artifact_id, 0.0) + contribution
            row = by_id[artifact_id]
            support.setdefault(artifact_id, {})[route_name] = {
                "raw_score": row["score"],
                "dbsf_contribution": contribution,
                "passage_id": row["passage_id"],
                "passage": row["passage"],
            }
    ranked = sorted(totals, key=lambda artifact_id: (-totals[artifact_id], artifact_id))[:int(limit)]
    return [{"artifact_id": artifact_id, "score": totals[artifact_id],
             "route_support": support[artifact_id]} for artifact_id in ranked]


def artifact_ids(rows, limit=10):
    return [row["artifact_id"] for row in rows[:int(limit)]]
