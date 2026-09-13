"""Registered student-only preprocessing variants for M18.

Teacher queries, indexed documents and BM25 always receive raw text. T1 maps only event-instance
values to typed exact placeholders. T2 removes WordPiece continuation rows matching
``^##[0-9]+$`` from both the pooled sum and count. The same functions are imported by training,
evaluation and the NumPy loader.
"""
from __future__ import annotations

import inspect
import re
from collections import defaultdict

from common import sha_text

PLACEHOLDERS = {
    "uuid": "<m18_uuid>",
    "hash": "<m18_hash>",
    "timestamp": "<m18_timestamp>",
    "address": "<m18_address>",
    "random_suffix": "<m18_random_suffix>",
}

# Ordering matters: an address is also hexadecimal and an ISO timestamp begins with digits.
PATTERNS = (
    ("uuid", re.compile(r"(?<![\w-])[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}(?![\w-])", re.I)),
    ("timestamp", re.compile(r"(?<!\w)\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})(?!\w)")),
    ("address", re.compile(r"(?<!\w)0x[0-9a-f]{8,16}(?!\w)", re.I)),
    ("hash", re.compile(r"(?<![\w-])[0-9a-f]{16,64}(?![\w-])", re.I)),
    # Kubernetes-style generated name: semantic prefix + template hash + pod suffix. Requiring
    # both suffixes avoids normalizing product names such as gpt-4o-mini or qdrant-1.15.
    ("random_suffix", re.compile(
        r"\b([a-z][a-z0-9.-]{1,80})-(?=[a-z0-9]{8,10}-)(?=[a-z0-9]*\d)"
        r"[a-z0-9]{8,10}-[a-z0-9]{5}\b", re.I)),
)

DIGIT_CONTINUATION_RE = re.compile(r"^##[0-9]+$")


def normalize_t1(text: str, enabled=None) -> str:
    enabled = set(enabled or PLACEHOLDERS)
    out = str(text)
    for name, rx in PATTERNS:
        if name in enabled:
            if name == "random_suffix":
                out = rx.sub(lambda m: m.group(1) + "-" + PLACEHOLDERS[name], out)
            else:
                out = rx.sub(PLACEHOLDERS[name], out)
    return out


def t2_mask_ids(ids, tokenizer):
    """Return ids whose serialized token is not a continuation-digit row."""
    return [int(i) for i in ids if not DIGIT_CONTINUATION_RE.fullmatch(
        tokenizer.id_to_token(int(i)) or "")]


def active_ids(text, tokenizer, variant: str):
    q = normalize_t1(text) if variant == "T1" else text
    ids = list(tokenizer.encode(q).ids)
    return t2_mask_ids(ids, tokenizer) if variant == "T2" else ids


def collision_audit(records, representative_max=5):
    """Audit raw→T1 collisions, grouped by structural relevance group.

    ``records`` carry ``text`` and a stable ``relevance_group``. A normalized form shared by
    multiple groups is ambiguous. The builder refuses ambiguous forms occurring five or more
    times; rarer ones remain disclosed rather than silently treated as safe.
    """
    forms = defaultdict(lambda: {"raw": [], "raw_set": set(), "groups": set(),
                                 "changed": False, "occurrences": 0})
    pattern_counts = defaultdict(int)
    for row in records:
        raw = str(row["text"])
        norm = normalize_t1(raw)
        for name, rx in PATTERNS:
            if rx.search(raw):
                pattern_counts[name] += 1
        ent = forms[norm]
        ent["occurrences"] += 1
        ent["changed"] = ent["changed"] or norm != raw
        ent["raw_set"].add(raw)
        if len(ent["raw"]) < representative_max and raw not in ent["raw"]:
            ent["raw"].append(raw)
        ent["groups"].add(str(row.get("relevance_group", "")))
    collisions = []
    for norm, ent in sorted(forms.items()):
        if len(ent["raw_set"]) > 1 and len(ent["groups"]) > 1 and ent["changed"]:
            collisions.append({"normalized": norm, "raw_examples": ent["raw"],
                               "relevance_groups": sorted(ent["groups"]),
                               "occurrences": ent["occurrences"],
                               "ambiguous": True})
    high_frequency = [c for c in collisions if c["occurrences"] >= 5]
    return {"_schema": "m18-t1-collision-audit-v1",
            "implementation_sha256": implementation_hash(),
            "records": len(records), "pattern_counts": dict(sorted(pattern_counts.items())),
            "collisions": collisions, "ambiguous_high_frequency": high_frequency,
            "pass": not high_frequency}


def implementation_hash():
    return sha_text(inspect.getsource(normalize_t1) + inspect.getsource(t2_mask_ids)
                    + repr([(n, r.pattern, r.flags) for n, r in PATTERNS]))
