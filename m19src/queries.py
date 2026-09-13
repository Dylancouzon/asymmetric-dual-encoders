"""M19 query-shape, family-split and sealed-confirmation validation."""
from __future__ import annotations

import json
import re
from collections import Counter

from m19src.common import atomic_create_bytes, sha_bytes, sha_json

LEXICAL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+#/:-]*")
PRIMARY_CLASSES = {"bare_adaptation", "short_context", "longer_control"}
SECONDARY_TAGS = {"alias", "renamed", "numeric", "version"}


def normalize_text(text):
    return " ".join(str(text).casefold().split())


def lexical_count(text):
    return len(LEXICAL.findall(str(text)))


def _validate_row(row, split, roster_terms, term_ids, tokenizer):
    required = {"query_id", "text", "term", "primary_class", "intent_family", "source_type",
                "author_id", "prospective_useful_artifact_ids", "answerable", "spent_overlap"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"query {row.get('query_id')} missing {sorted(missing)}")
    if row["term"] not in roster_terms or row["primary_class"] not in PRIMARY_CLASSES:
        raise ValueError("query term/class is outside registry")
    if "qwen" in str(row["author_id"]).casefold():
        raise ValueError("Qwen may not author M19 evaluation queries")
    if not row["answerable"] or not row["prospective_useful_artifact_ids"]:
        raise ValueError("unanswerable query must be removed before split freeze")
    if row["spent_overlap"]:
        raise ValueError("query overlaps spent development/diagnostic intent")
    tags = set(row.get("tags") or [])
    if not tags.issubset(SECONDARY_TAGS):
        raise ValueError("unknown secondary query tag")
    encoded = tokenizer.encode(row["text"], add_special_tokens=False).ids
    matches = sum(encoded.count(term_ids[term]) for term in roster_terms)
    if matches != 1 or encoded.count(term_ids[row["term"]]) != 1:
        raise ValueError("target query must match exactly one roster AddedToken")
    length = lexical_count(row["text"])
    primary = row["primary_class"]
    if primary == "bare_adaptation":
        if split != "development" or normalize_text(row["text"]) != normalize_text(row["term"]):
            raise ValueError("bare adaptation is development-only raw term")
    elif primary == "short_context" and not 2 <= length <= 5:
        raise ValueError("short context must have two to five lexical terms")
    elif primary == "longer_control" and length < 6:
        raise ValueError("longer control must have at least six lexical terms")
    if row["source_type"] == "source_authored" and not row.get("source_artifact_id"):
        raise ValueError("source-authored query lacks source artifact")


def validate_splits(development, confirmation, roster, tokenizer, registry,
                    *, exact_counts=True):
    terms = [row["term"] for row in roster["terms"]]
    term_ids = roster["selected_added_token_audit"]["term_ids"]
    all_rows = [("development", row) for row in development] + [
        ("confirmation", row) for row in confirmation
    ]
    query_ids = [row["query_id"] for _, row in all_rows]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query IDs are not unique")
    for split, row in all_rows:
        _validate_row(row, split, terms, term_ids, tokenizer)
    normalized = [normalize_text(row["text"]) for _, row in all_rows]
    intents = [row["intent_family"] for _, row in all_rows]
    if len(normalized) != len(set(normalized)) or len(intents) != len(set(intents)):
        raise ValueError("normalized query or intent family overlaps")
    for field in ("source_artifact_id", "source_family"):
        dev = {str(row[field]) for row in development if row.get(field)}
        conf = {str(row[field]) for row in confirmation if row.get(field)}
        if dev & conf:
            raise ValueError(f"{field} overlaps development/confirmation")
    if exact_counts:
        expected = registry["query_counts_per_term"]
        for term in terms:
            dev_counts = Counter(row["primary_class"] for row in development if row["term"] == term)
            conf_counts = Counter(row["primary_class"] for row in confirmation if row["term"] == term)
            if dev_counts != {"bare_adaptation": int(expected["development_bare"]),
                              "short_context": int(expected["development_short"]),
                              "longer_control": int(expected["development_longer"])}:
                raise ValueError(f"development count mismatch for {term}")
            if conf_counts["bare_adaptation"] or conf_counts["short_context"] != int(
                    expected["confirmation_short"]):
                raise ValueError(f"confirmation count mismatch for {term}")
        required_longer = len(terms) * int(expected["confirmation_longer_per_two_terms"]) // 2
        if sum(row["primary_class"] == "longer_control" for row in confirmation) != required_longer:
            raise ValueError("confirmation longer-control count mismatch")
    safety = registry["query_protocol"]
    for rows, minimum, name in (
        (development, safety["development_numeric_version_controls_minimum"], "development"),
        (confirmation, safety["confirmation_numeric_version_controls_minimum"], "confirmation"),
    ):
        controls = [row for row in rows if row["primary_class"] == "longer_control"
                    and set(row.get("tags") or []) & {"numeric", "version"}]
        term_minimum = int(safety["numeric_version_terms_minimum"])
        if len(controls) < int(minimum) or len({row["term"] for row in controls}) < term_minimum:
            raise ValueError(f"{name} numeric/version safety slice is undersized")
    return {
        "development_queries": len(development),
        "confirmation_queries": len(confirmation),
        "terms": len(terms),
        "development_text_sha256": sha_json([row["text"] for row in development]),
        "confirmation_text_sha256": sha_json([row["text"] for row in confirmation]),
        "source_artifact_overlap": 0,
        "source_family_overlap": 0,
        "normalized_text_overlap": 0,
        "intent_family_overlap": 0,
    }


def jsonl_bytes(rows):
    return b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)


def seal_splits(development_path, confirmation_path, development, confirmation):
    """Publish exact split bytes once; confirmation remains behind its read guard.

    Existing destinations are never opened to reconcile a retry. In particular, doing so for the
    confirmation destination would create a pre-claim read path. The returned hashes are computed
    from the caller-owned payload and can be bound into the later claim.
    """
    payloads = {"development": jsonl_bytes(development), "confirmation": jsonl_bytes(confirmation)}
    for path, payload in ((development_path, payloads["development"]),
                          (confirmation_path, payloads["confirmation"])):
        try:
            atomic_create_bytes(path, payload)
        except FileExistsError:
            raise SystemExit(f"M19 QUERY REFUSED: sealed split destination already exists: {path}")
    return {name + "_sha256": sha_bytes(payload) for name, payload in payloads.items()}
