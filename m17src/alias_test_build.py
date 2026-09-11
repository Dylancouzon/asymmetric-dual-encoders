"""M17 step 2c: the 200-pair held-out alias test.

Same-intent query pairs — acronym/expansion and alias/canonical — held out of all training and
split from training pairs by query family (`registry.data.alias_heldout_test`). The test measures
whether two forms of one intent retrieve the same documents after training, which training pairs
cannot show. It is descriptive: it routes no selection
(`registry.training.decision_protocol.panel_and_alias_test_role`).

Every pair is an equivalence a SOURCE states, not one a model or this script asserts:

* `VERIFIED_BY_SOURCE` — the source document itself states the equivalence in a sentence that is
  quoted verbatim in the record's citation: a parenthetical definition ("Custom Resource
  Definition (CRD)") whose initials actually match, or an explicit "also known as" / "also called"
  / "short for" / "abbreviated as" clause. Acronym initials are checked against the expansion's
  words, so "Kubernetes (v1.31)" is not admitted as an alias.
* `PENDING_HUMAN` — the short form has more than one distinct expansion across the admitted text,
  so which sense a bare query means is a judgment. These are flagged `ambiguous_sense: true` and
  kept deliberately: the registry asks for ambiguous senses to be present, flagged.

Sources: the step-2a Kubernetes slice (CC BY 4.0) and the admitted labelled sources' stores
`squad-ctx` (SQuAD, CC BY-SA 4.0) and `mrtydi-docs` (Mr. TyDi English, Apache 2.0). HotpotQA's
5.2M-abstract store is not scanned — the yield from two stores already exceeds the target and the
scan would cost more than it adds; recorded rather than silently omitted. `fever-train` and
`esci-us` are excluded for the reasons `m17src/panel_build.EXCLUDED_SOURCES` records.

Outputs:
    work/m17/alias/alias_test.jsonl                  the pairs (text stays in work/)
    work/m17/manifest/alias_test_families.json       families/keys to EXCLUDE from training pairs
    results/m17_alias_test_manifest.json             counts, hashes, no text
    results/m17_panel_pending_judgments.jsonl        ambiguous pairs appended for review

Run after `m17src/panel_build.py --stage draft` (it appends to that review sheet), then re-run
`panel_build.py --stage screen --stage seal`. Pre-clock, no GPU, no protected access.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "m17src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from common import RESULTS, WORK, sha_file, write_json                      # noqa: E402
import panel_build as P                                                     # noqa: E402

ALIAS_DIR = WORK / "alias"
ALIAS_JSONL = ALIAS_DIR / "alias_test.jsonl"
FAMILIES_OUT = WORK / "manifest" / "alias_test_families.json"
MANIFEST_OUT = RESULTS / "m17_alias_test_manifest.json"
PENDING = RESULTS / "m17_panel_pending_judgments.jsonl"

TARGET_PAIRS = 200
K8S_SHARE = 100
SEED = 17

STOPWORDS = {"of", "the", "and", "for", "a", "an", "in", "on", "to", "with", "de", "or"}
# Both patterns are deliberately ANCHOR-ONLY and the phrases either side are recovered by word
# slicing, not by a variable-length regex group. The first version used nested quantifiers
# ("(word){1,5} \\(ABBR\\)") and spent minutes backtracking over 15 MB of documentation text
# before the first line of output; these run in one linear pass.
ABBR_RE = re.compile(r"\(([A-Z][A-Za-z]{1,6}s?)\)")
MARKER_RE = re.compile(r"\b(?:also known as|also called|short for|abbreviated as|"
                       r"sometimes called)\b", re.I)
MAX_EXPANSION_WORDS = 6
MARKER_PHRASE_WORDS = 4          # words kept either side of an "also known as" marker
BAD_EXPANSION = re.compile(r"\d|[/.]|^v\d|http|www|^see$|^the$")
# Markdown link targets sit between an expansion and its acronym ("[Node Feature Discovery]
# (https://...) (NFD)") and would be read as the expansion. Dropped before extraction.
LINK_RE = re.compile(r"\((?:https?:|mailto:|/|#)[^)]*\)")
BRACKET_RE = re.compile(r"[\[\]]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-/]*")


def initials_match(expansion, abbr):
    """True when the acronym's letters are the initials of the expansion's non-stop words.

    Tolerant of a trailing plural 's' and of internal capitals ("CustomResourceDefinition"): the
    check is that every acronym letter, in order, starts a word of the expansion.
    """
    a = abbr.rstrip("s").lower()
    words = [w for w in re.split(r"[\s\-/]+", expansion.lower()) if w and w not in STOPWORDS]
    if not (2 <= len(a) <= 6) or len(words) < 2:
        return False
    i = 0
    for w in words:
        if i < len(a) and w.startswith(a[i]):
            i += 1
    return i == len(a) and len(words) >= len(a)


def sentence_of(text, span_start):
    """The sentence containing `span_start`, trimmed to 300 characters for the citation."""
    lo = max(text.rfind(". ", 0, span_start), text.rfind("\n", 0, span_start)) + 1
    hi = text.find(". ", span_start)
    hi = len(text) if hi < 0 else hi + 1
    return " ".join(text[lo:hi].split())[:300]


def _words_before(body, pos, n=MAX_EXPANSION_WORDS):
    """Up to `n` words immediately preceding `pos`, in order."""
    return WORD_RE.findall(body[max(0, pos - 12 * n):pos])[-n:]


def _words_after(body, pos, n=MAX_EXPANSION_WORDS):
    return WORD_RE.findall(body[pos:pos + 12 * n])[:n]


def extract(doc_id, text, source, domain_hint="general"):
    """-> [{short, long, kind, citation_sentence}] from one document's text."""
    out = []
    body = " ".join(BRACKET_RE.sub(" ", LINK_RE.sub(" ", text)).split())
    for m in ABBR_RE.finditer(body):
        abbr = m.group(1)
        before = _words_before(body, m.start())
        # the shortest word-suffix whose initials are the acronym's letters wins
        for k in range(2, len(before) + 1):
            expansion = " ".join(before[-k:])
            if BAD_EXPANSION.search(expansion.lower()):
                continue
            if initials_match(expansion, abbr):
                out.append({"short": abbr, "long": expansion, "kind": "acronym-expansion",
                            "citation_sentence": sentence_of(body, m.start())})
                break
    for m in MARKER_RE.finditer(body):
        lo = max(body.rfind(". ", 0, m.start()), body.rfind("(", 0, m.start())) + 1
        avail = WORD_RE.findall(body[lo:m.start()])
        canon_words = avail[-MARKER_PHRASE_WORDS:]
        canon = " ".join(canon_words)
        seg = body[m.end():m.end() + 80]
        cut = min([i for i in (seg.find(c) for c in ",.;)") if i >= 0] or [len(seg)])
        alias_words = WORD_RE.findall(seg[:cut])
        alias = " ".join(alias_words[:MARKER_PHRASE_WORDS])
        if not canon or not alias:
            continue
        if (not alias or not canon or P.normalize(alias) == P.normalize(canon)
                or BAD_EXPANSION.search(alias.lower()) or BAD_EXPANSION.search(canon.lower())):
            continue
        # The phrase is recovered by word slicing, so a canonical name longer than the cap is
        # silently beheaded: "Balanced Random Access Distributed Storage System, also known as
        # BRADSS" yields "Access Distributed Storage System", an equivalence the source never
        # stated. Truncated (or mid-phrase) extractions are routed to a human, not verified.
        truncated = (len(avail) > MARKER_PHRASE_WORDS
                     or len(alias_words) > MARKER_PHRASE_WORDS
                     or not canon_words[0][:1].isupper())
        out.append({"short": alias, "long": canon, "kind": "alias-canonical",
                    "truncated": truncated,
                    "citation_sentence": sentence_of(body, m.start())})
    for o in out:
        o.setdefault("truncated", False)
        o.update({"doc_id": doc_id, "source": source, "domain_hint": domain_hint,
                  "evidence_doc_group": P.group_id(P.normalize(text))})
    return out


def gather(verbose=True):
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    found = []
    kdocs, _flagged, _step2a = P.k8s_docs()
    for d in kdocs:
        found += extract(d["path"], d["text"], "k8s-docs-en", "cloud-software")
    log(f"[alias] kubernetes: {len(found)} raw equivalences from {len(kdocs)} documents")
    for store, source in (("squad-ctx", "squad-train"), ("mrtydi-docs", "mrtydi-en")):
        ids, texts = P.load_store(store)
        n0 = len(found)
        for did, t in zip(ids, texts):
            found += extract(did, t, source)
        log(f"[alias] {store}: {len(found) - n0} raw equivalences from {len(ids):,} documents")
        del ids, texts
    return found


def write_exclusions(records, panel_rows, path=None):
    """Write the file `alias_pairs.py` and `support_manifest.py` read, and return it.

    Family ids and whole-view-text shas alone excluded NOTHING from the real training pool
    (`pairs_removed_by_excluded_families: 0`): a training carrier sentence shares neither with a
    held-out bare pair. So each held-out pair also contributes its normalized short form and
    expansion, and its evidence document's group id — `group_id(normalize(document text))`,
    the same key `support_manifest.document_pass` writes.
    """
    fam_ids = sorted({r["family_id"] for r in records})
    panel_fams = sorted({r["family_id"] for r in panel_rows})
    panel_shas = sorted({r["normalized_text_sha"] for r in panel_rows})
    text_shas = sorted({r["view_a_sha"] for r in records} | {r["view_b_sha"] for r in records})
    doc_ids = sorted({r["citation"]["doc_id"] for r in records})
    terms = sorted({P.normalize(r["view_a"]) for r in records}
                   | {P.normalize(r["view_b"]) for r in records})
    doc_groups = sorted({r["evidence_doc_group"] for r in records})
    blob = {
        "milestone": "M17", "step": "2c", "date": "2026-09-11",
        "purpose": ("Families and keys that must NOT appear in M17 training pairs or training "
                    "queries: the held-out alias test and the judged panel. Read by "
                    "m17src/alias_pairs.py and m17src/support_manifest.py."),
        "key_conventions": ("family_id = 'fam:' + sha256(union-find root)[:16] (this script's "
                            "own ids). The interoperable keys are the sha256(normalized text)[:16] "
                            "lists, which use m17src/support_manifest.group_id exactly: match on "
                            "those if the family ids differ. `excluded_alias_terms` are "
                            "normalized short forms and expansions; "
                            "`excluded_evidence_doc_groups` are group_id(normalize(document "
                            "text)) of the documents that evidence a held-out pair."),
        "excluded_family_ids": sorted(set(fam_ids) | set(panel_fams)),
        "alias_test": {"family_ids": fam_ids, "n_pairs": len(records),
                       "excluded_text_shas": text_shas, "citation_doc_ids": doc_ids},
        "panel": {"family_ids": panel_fams, "excluded_text_shas": panel_shas,
                  "n_queries": len(panel_rows)},
        "excluded_text_shas": sorted(set(text_shas) | set(panel_shas)),
        "excluded_alias_terms": terms,
        "excluded_evidence_doc_groups": doc_groups,
    }
    write_json(path or FAMILIES_OUT, blob)
    return blob


def build(seed=SEED, target=TARGET_PAIRS, verbose=True):
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    rng = random.Random(seed)
    found = gather(verbose=verbose)

    # ambiguity: one short form, several distinct expansions across the admitted text
    senses = defaultdict(set)
    for f in found:
        senses[P.normalize(f["short"])].add(P.normalize(f["long"]))

    # one pair per (short, long) equivalence, first citation wins, deterministic order
    seen, pairs = set(), []
    for f in sorted(found, key=lambda f: (f["source"], f["doc_id"], f["short"], f["long"])):
        key = (P.normalize(f["short"]), P.normalize(f["long"]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(f)

    # allocate: k8s share first, then the admitted labelled sources, seeded
    k8s = [p for p in pairs if p["source"] == "k8s-docs-en"]
    other = [p for p in pairs if p["source"] != "k8s-docs-en"]
    rng.shuffle(k8s)
    rng.shuffle(other)
    picked = k8s[:K8S_SHARE] + other[:max(0, target - min(len(k8s), K8S_SHARE))]
    picked = picked[:target]
    if len(picked) < target:                       # top up from whichever side has spare
        spare = k8s[K8S_SHARE:] + other[max(0, target - min(len(k8s), K8S_SHARE)):]
        rng.shuffle(spare)
        picked += spare[:target - len(picked)]

    # families: a pair joins any other pair sharing a view text or the same source document
    uf = P.Union()
    for i, p in enumerate(picked):
        uid = f"pair:{i:04d}"
        uf.add(uid)
        for k in ("short", "long"):
            t = "txt:" + P.group_id(P.normalize(p[k]))
            uf.add(t)
            uf.union(uid, t)
        d = "doc:" + P.group_id(P.normalize(p["doc_id"]))
        uf.add(d)
        uf.union(uid, d)

    records, pending = [], []
    for i, p in enumerate(picked):
        ambiguous = len(senses[P.normalize(p["short"])]) > 1
        truncated = bool(p.get("truncated"))
        domain = p["domain_hint"]
        if domain == "general":
            domain, _ = P.classify(p["long"] + " " + p["citation_sentence"])
        pid = f"alias-{i:04d}"
        rec = {
            "pair_id": pid,
            "view_a": p["long"], "view_b": p["short"],
            "view_a_form": "expansion" if p["kind"] == "acronym-expansion" else "canonical",
            "view_b_form": "acronym" if p["kind"] == "acronym-expansion" else "alias",
            "kind": p["kind"], "domain": domain, "domain_label": "heuristic",
            "slice": f"{domain}:{p['kind']}",
            "family_id": P.family_id(uf.find(f"pair:{i:04d}")),
            "source": p["source"],
            "citation": {"doc_id": p["doc_id"], "sentence": p["citation_sentence"]},
            "judgment_status": ("PENDING_HUMAN" if (ambiguous or truncated)
                                else "VERIFIED_BY_SOURCE"),
            "ambiguous_sense": ambiguous,
            "extraction_truncated": truncated,
            "evidence_doc_group": p["evidence_doc_group"],
            "distinct_expansions_seen": sorted(senses[P.normalize(p["short"])])[:6],
            "view_a_sha": P.group_id(P.normalize(p["long"])),
            "view_b_sha": P.group_id(P.normalize(p["short"])),
            "held_out_of_training": True,
        }
        records.append(rec)
        if ambiguous or truncated:
            question = (f"Does the bare query {p['short']!r} mean {p['long']!r} here? "
                        f"Other expansions seen: "
                        f"{sorted(senses[P.normalize(p['short'])])[:4]}") if ambiguous else (
                f"The extracted phrase {p['long']!r} hit the {MARKER_PHRASE_WORDS}-word cap or "
                "does not start at a phrase boundary. Is it the complete name the sentence "
                "states?")
            pending.append({
                "kind": "alias-pair", "pair_id": pid, "query": p["short"],
                "domain": domain, "partition": "alias-test",
                "reason": "ambiguous-sense" if ambiguous else "truncated-extraction",
                "candidate_doc_id": p["doc_id"],
                "candidate_title": p["long"],
                "candidate_first_300_chars": p["citation_sentence"],
                "question": question,
                "relevant_yes_no": None, "judge": None, "notes": ""})

    ALIAS_DIR.mkdir(parents=True, exist_ok=True)
    P._write_jsonl(ALIAS_JSONL, records)

    # ---- the exclusion file the training-pair builder reads
    panel_rows = P.read_jsonl(P.PANEL_JSONL) if P.PANEL_JSONL.exists() else []
    exclusions = write_exclusions(records, panel_rows)
    fam_ids = exclusions["alias_test"]["family_ids"]

    # ---- append the ambiguous pairs to the one human review sheet
    P.assert_pending_unjudged(PENDING)
    rows = [r for r in (P.read_jsonl(PENDING) if PENDING.exists() else [])
            if r.get("kind") != "alias-pair"]
    for r in rows:
        r.setdefault("kind", "panel-candidate")
    P._write_jsonl(PENDING, rows + pending)

    manifest = {
        "milestone": "M17", "step": "2c", "date": "2026-09-11",
        "script": "m17src/alias_test_build.py",
        "status": "PROVISIONAL — ambiguous-sense pairs pending human judgment",
        "gpu_used": False, "protected_payloads_opened": False, "scored_anything": False,
        "seed": seed, "target_pairs": target, "n_pairs": len(records),
        "by_source": dict(Counter(r["source"] for r in records)),
        "by_kind": dict(Counter(r["kind"] for r in records)),
        "by_domain": dict(Counter(r["domain"] for r in records)),
        "by_judgment_status": dict(Counter(r["judgment_status"] for r in records)),
        "ambiguous_sense_pairs": sum(1 for r in records if r["ambiguous_sense"]),
        "truncated_extraction_pairs": sum(1 for r in records if r["extraction_truncated"]),
        "n_families": len(fam_ids),
        "raw_equivalences_found": len(found),
        "distinct_equivalences": len(pairs),
        "stores_scanned": ["work/m17/sources/k8s_docs_en.jsonl", "squad-ctx", "mrtydi-docs"],
        "stores_not_scanned": {"hotpotqa-corpus": "5.2M abstracts; yield already over target",
                               "esci-prod": "excluded source (see panel manifest)",
                               "fever-pos": "excluded source (reserved-adjacent)"},
        "extraction": {
            "acronym_rule": ("parenthetical 'Expansion (ABBR)' where every acronym letter, in "
                             "order, starts a non-stop word of the expansion; 2-6 letters; "
                             "digits/URLs rejected"),
            "alias_rule": "'also known as' / 'also called' / 'short for' / 'abbreviated as'",
            "ambiguity": ("a short form with more than one distinct normalized expansion across "
                          "the scanned text is flagged and left PENDING_HUMAN")},
        "files": {"pairs": str(ALIAS_JSONL.relative_to(REPO)),
                  "pairs_sha256": sha_file(ALIAS_JSONL),
                  "exclusions": str(FAMILIES_OUT.relative_to(REPO)),
                  "exclusions_sha256": sha_file(FAMILIES_OUT),
                  "pending_review": str(PENDING.relative_to(REPO))},
        "role": ("read once per screen arm and once per final form beside the ordinary metrics; "
                 "routes no selection (registry.data.alias_heldout_test.reads)"),
        "limitations": [
            "A source-stated equivalence is not a relevance judgment: the pair asserts that two "
            "query forms mean the same thing, not which documents answer them.",
            "Domain labels are the panel's heuristic keyword labels.",
            "Query-side ancestry screening is m17src/panel_build.py --stage screen; the six, the "
            "reserved four and LoTTE stay deferred to the executor.",
        ],
    }
    write_json(MANIFEST_OUT, manifest)
    log(f"[alias] {len(records)} pairs "
        f"({manifest['by_judgment_status']}), {len(fam_ids)} families, "
        f"{len(pending)} ambiguous pending")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", type=int, default=TARGET_PAIRS)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    build(target=args.target, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
