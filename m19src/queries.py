"""M19 query-shape, family-split and sealed-confirmation validation."""
from __future__ import annotations

import json
import re
from collections import Counter

from m19src.common import atomic_create_bytes, sha_bytes, sha_file_unchecked, sha_json

LEXICAL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+#/:-]*")
PRIMARY_CLASSES = {"bare_adaptation", "short_context", "longer_control"}
SECONDARY_TAGS = {"alias", "renamed", "numeric", "version"}


def normalize_text(text):
    return " ".join(str(text).casefold().split())


def lexical_count(text):
    return len(LEXICAL.findall(str(text)))


def lexical_signature(text):
    """Catch case, punctuation and word-order variants without claiming semantic deduplication."""
    return tuple(sorted(token.casefold() for token in LEXICAL.findall(str(text))))


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
    exclusions = {str(value) for value in row.get("source_equivalent_artifact_ids") or []}
    if row.get("source_artifact_id"):
        exclusions.add(str(row["source_artifact_id"]))
    useful = {str(value) for value in row["prospective_useful_artifact_ids"]}
    if exclusions & useful:
        raise ValueError("source artifact or equivalent is listed as prospectively useful")


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
    lexical_signatures = [lexical_signature(row["text"]) for _, row in all_rows]
    intents = [row["intent_family"] for _, row in all_rows]
    if len(normalized) != len(set(normalized)) or len(intents) != len(set(intents)):
        raise ValueError("normalized query or intent family overlaps")
    if len(lexical_signatures) != len(set(lexical_signatures)):
        raise ValueError("lexical near-duplicate query overlaps")
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
        "lexical_near_duplicate_overlap": 0,
        "source_exclusion_sha256": sha_json({
            row["query_id"]: sorted({
                *([str(row["source_artifact_id"])] if row.get("source_artifact_id") else []),
                *(str(value) for value in row.get("source_equivalent_artifact_ids") or []),
            })
            for _, row in all_rows
        }),
    }


def jsonl_bytes(rows):
    return b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)


def _publish_or_verify(path, payload):
    """Create immutable bytes, or verify the digest of a matching interrupted publication."""
    expected = sha_bytes(payload)
    try:
        atomic_create_bytes(path, payload)
        return "created"
    except FileExistsError:
        # Hash-only verification does not return sealed confirmation content to the caller.
        if sha_file_unchecked(path) != expected:
            raise SystemExit(f"M19 QUERY REFUSED: existing sealed bytes differ: {path}")
        return "verified"


def seal_splits(development_path, confirmation_path, manifest_path, development, confirmation,
                roster, tokenizer, registry, *, exact_counts=True):
    """Validate, bind and resumably publish both split payloads and their final manifest."""
    paths = [str(development_path), str(confirmation_path), str(manifest_path)]
    if len(paths) != len(set(paths)):
        raise ValueError("query split and manifest paths must be distinct")
    validation = validate_splits(
        development, confirmation, roster, tokenizer, registry, exact_counts=exact_counts)
    payloads = {"development": jsonl_bytes(development), "confirmation": jsonl_bytes(confirmation)}
    manifest = {
        "_schema": "m19-query-split-seal-v1",
        "registry_sha256": sha_json(registry),
        "roster_sha256": sha_json(roster),
        "protocol_versions": {
            "query": registry["versions"]["query_protocol"],
            "family_split": registry["versions"]["family_split"],
        },
        "validation": validation,
        "splits": {
            name: {
                "sha256": sha_bytes(payloads[name]),
                "bytes": len(payloads[name]),
                "query_ids": [row["query_id"] for row in rows],
            }
            for name, rows in (("development", development), ("confirmation", confirmation))
        },
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    publication = {
        "development": _publish_or_verify(development_path, payloads["development"]),
        "confirmation": _publish_or_verify(confirmation_path, payloads["confirmation"]),
        # The manifest is the completion marker and is always published last.
        "manifest": _publish_or_verify(manifest_path, manifest_payload),
    }
    return manifest | {"manifest_sha256": sha_bytes(manifest_payload),
                       "publication": publication}
