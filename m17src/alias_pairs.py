"""M17 step 2b: the bulk structural alias training-pair pool.

`m17/registry.json` `data.bucket_populations_and_dose_rule.alias_training_pairs_source` admits
bulk *structural* equivalences from admitted sources only, filtered by rule and spot-checked by a
human on a random 2% sample. Individually judged pairs are reserved for the 200-pair held-out
test, which a later pre-clock step builds; nothing here is a judged pair.

Four rules were registered. Two of them have no material on this box, and that is reported as a
gap rather than replaced with something else:

  R1  k8s glossary alias forms the document itself states (`aka:` front matter in
      `content/en/docs/reference/glossary/*.md` at the pinned revision). Available.
  R2  rule-based abbreviation/expansion where the expansion appears in the same admitted
      document: the classic `Expansion (ABBR)` definition pattern, initial-matched. Available.
  R3  dataset-provided paraphrases. NOT available: none of HotpotQA, SQuAD, FEVER-train,
      Mr.TyDi or ESCI ships a paraphrase field; the built mix under `work/train/sources/` carries
      only qid/query/pos/hardneg.
  R4  Wikipedia redirect titles to their target titles inside admitted Wikipedia-derived
      datasets. NOT available: the admitted datasets carry passage ids and text, never the
      redirect graph, and no Wikipedia redirect dump is an admitted source. Adding one is a new
      source and needs Dylan's ruling.

A pair is two *query views* that differ only by the alias substitution, as PLANNING's
`k8s ingress` / `Kubernetes ingress` example requires -- not two bare terms. A carrier sentence
is taken from the admitted document that evidences the equivalence, the parenthetical definition
is removed, and the two forms are substituted into it.

Registered cautions honoured here: ambiguous abbreviations are never expanded globally (an
abbreviation with more than one distinct mined expansion is dropped, and so is an expansion with
more than one abbreviation), teacher agreement is never used as a label, and no relevance label
is attached to either view. ESCI is deliberately not mined: product listings are not prose and
the `Expansion (ABBR)` pattern is unreliable in them; recorded, not silent.

Usage (repository root, pre-clock, no GPU):

    .venv/bin/python m17src/alias_pairs.py
    .venv/bin/python m17src/alias_pairs.py --exclude-families work/m17/manifest/alias_test_families.json

Outputs: `work/m17/manifest/alias_pairs.jsonl` (gitignored pool, with text),
`work/m17/manifest/alias_pairs_summary.json`, and the committed 2% human spot-check sample
`results/m17_alias_spotcheck_sample.jsonl`.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

from support_manifest import (MANIFEST_DIR, REPO, TRAIN, WORK, group_id, iter_k8s, iter_store,
                              k8s_excluded_paths, load_exclusions, normalize, rel, sha256_file)

K8S_CLONE = WORK / "m17" / "sources" / "kubernetes-website"
GLOSSARY = K8S_CLONE / "content" / "en" / "docs" / "reference" / "glossary"
POOL = MANIFEST_DIR / "alias_pairs.jsonl"
SUMMARY = MANIFEST_DIR / "alias_pairs_summary.json"
SPOTCHECK = REPO / "results" / "m17_alias_spotcheck_sample.jsonl"
EXCLUDE_DEFAULT = MANIFEST_DIR / "alias_test_families.json"

# Mined for R2. ESCI is excluded by rule (see the module docstring); the two query-text sources
# have no documents.
MINED_STORES = {
    "hotpotqa-train": "hotpotqa-corpus",
    "fever-train": "fever-pos",
    "squad-train": "squad-ctx",
    "mrtydi-en": "mrtydi-docs",
}

# The abbreviation is located first and the expansion is read backwards from it. A single
# regex that also matched the preceding word sequence ran at ~1,950 documents/second, which is
# 45 minutes for the 5.23M-document HotpotQA corpus alone; anchoring on the parenthesis and
# slicing a fixed lead window makes the scan linear. Measured, not assumed (LEDGER step 2b).
ABBR_RE = re.compile(r"\(\s*([A-Z][A-Za-z0-9]{1,7})s?\s*\)")
WORD_RE = re.compile(r"[A-Za-z][\w./-]*\Z")
LEAD_CHARS = 120
SENT_RE = re.compile(r"(?<=[.!?])\s+")
STOP = {"of", "the", "and", "for", "in", "on", "at", "to", "a", "an", "de", "la"}
# Abbreviations that are ordinary English words or otherwise unsafe to substitute blind.
ABBR_DENY = {"it", "us", "am", "pm", "ok", "no", "on", "in", "is", "as", "at", "be", "by", "do",
             "he", "if", "me", "my", "of", "or", "so", "to", "up", "we", "all", "one", "new",
             "the", "and", "for", "was", "his", "her", "not", "you", "are", "can", "may", "man",
             "war", "art", "air", "act", "age", "ad", "id", "os", "pc", "tv", "ii", "iii", "iv"}

MIN_CARRIER_WORDS = 4
MAX_CARRIER_WORDS = 32
MAX_PAIRS_PER_LEXICON_ENTRY = 8
MAX_CARRIERS_PER_DOCUMENT = 2
# Pass 1 retains the evidencing document text so pass 2 can cut carriers without a second read of
# a 1.6 GB store. Retention is capped per lexicon entry: without the cap a corpus-wide definition
# pattern would hold millions of paragraphs in RAM for pairs the per-entry cap then discards.
CARRIER_DOCS_PER_KEY = 8


def _initials_match(words, abbr):
    """Does this word window spell `abbr`, with and without function words?"""
    a = abbr.lower()
    full = "".join(w[0].lower() for w in words if w)
    content = "".join(w[0].lower() for w in words if w and w.lower() not in STOP)
    return a in (full, content)


def find_definitions(text):
    """-> [(expansion, abbr)] for `Expansion (ABBR)` occurrences whose initials match."""
    out = []
    for m in ABBR_RE.finditer(text):
        abbr = m.group(1)
        if abbr.lower() in ABBR_DENY or not (2 <= len(abbr) <= 8):
            continue
        words = text[max(0, m.start() - LEAD_CHARS):m.start()].split()[-(len(abbr) + 2):]
        for n in range(len(abbr), min(len(abbr) + 2, len(words)) + 1):
            win = words[-n:]
            if len(win) < 2 or not _initials_match(win, abbr):
                continue
            if any(len(w) < 2 or not WORD_RE.match(w) for w in win):
                break
            out.append((" ".join(win), abbr))
            break
    return out


def _carriers(text, form, drop_parenthetical=None):
    """Sentences of `text` containing `form` as a whole phrase, trimmed to a usable query view."""
    pat = re.compile(r"(?<!\w)" + re.escape(form) + r"(?!\w)")
    out = []
    for sent in SENT_RE.split(text):
        if drop_parenthetical:
            sent = re.sub(r"\s*\(\s*" + re.escape(drop_parenthetical) + r"s?\s*\)", "", sent)
        sent = " ".join(sent.split())
        hits = pat.findall(sent)
        if len(hits) != 1:
            continue
        n = len(sent.split())
        if not (MIN_CARRIER_WORDS <= n <= MAX_CARRIER_WORDS):
            continue
        out.append(sent)
        if len(out) >= MAX_CARRIERS_PER_DOCUMENT:
            break
    return out, pat


def make_pairs(text, doc_group, source, entries, rule, counters):
    """Substitute each equivalence into carriers taken from this document."""
    pairs = []
    for form_a, form_b in entries:
        drop = form_b if rule == "R2_abbreviation_expansion" else None
        carriers, pat = _carriers(text, form_a, drop_parenthetical=drop)
        for carrier in carriers:
            view_b = pat.sub(form_b, carrier)
            if normalize(view_b) == normalize(carrier) or not view_b.strip():
                counters["substitution_noop"] = counters.get("substitution_noop", 0) + 1
                continue
            na, nb = normalize(carrier), normalize(view_b)
            pairs.append({
                "pair_id": group_id(na + "\x00" + nb),
                "family_id": group_id("docgroup:" + doc_group),
                "rule": rule,
                "source": source,
                "doc_group": doc_group,
                "form_a": form_a,
                "form_b": form_b,
                "view_a": carrier,
                "view_b": view_b,
            })
    return pairs


# --------------------------------------------------------------------------- R1

def glossary_aliases():
    """`aka:` front matter of the pinned Kubernetes glossary. Returns [(title, alias)]."""
    out = []
    if not GLOSSARY.is_dir():
        return out
    for p in sorted(GLOSSARY.glob("*.md")):
        head = p.read_text(encoding="utf-8").split("---\n")
        if len(head) < 3:
            continue
        fm = head[1]
        title = re.search(r"^title:\s*(.+)$", fm, re.M)
        aka = re.search(r"^aka:(.*?)(?=^\w+:|\Z)", fm, re.S | re.M)
        if not (title and aka):
            continue
        raw = aka.group(1).strip()
        vals = []
        for chunk in re.split(r"[\n,]", raw.strip("[]")):
            v = chunk.strip().lstrip("-").strip().strip('"').strip("'")
            if v:
                vals.append(v)
        t = title.group(1).strip().strip('"')
        for v in vals:
            if normalize(v) != normalize(t) and len(v) >= 2:
                out.append((t, v))
    return out


# --------------------------------------------------------------------------- build

def build(seed, sample_fraction, exclude_path):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    counters = {}
    gaps = []

    # R2 pass 1: mine definitions and keep only the documents that carry one.
    lexicon = {}          # (expansion_norm, abbr) -> count of evidencing documents
    abbr_to_exp = {}
    exp_to_abbr = {}
    carriers_for = []     # (source, doc_group, text, [(expansion, abbr)])
    retained = {}         # lexicon key -> documents already retained for pass 2
    excluded = k8s_excluded_paths()

    def mine(source, text):
        defs = find_definitions(text)
        if not defs:
            return False
        uniq = sorted(set(defs))
        for exp, abbr in uniq:
            key = (normalize(exp), abbr)
            lexicon[key] = lexicon.get(key, 0) + 1
            abbr_to_exp.setdefault(abbr, set()).add(normalize(exp))
            exp_to_abbr.setdefault(normalize(exp), set()).add(abbr)
        want = [(e, a) for e, a in uniq
                if retained.get((normalize(e), a), 0) < CARRIER_DOCS_PER_KEY]
        if want:
            for e, a in want:
                retained[(normalize(e), a)] = retained.get((normalize(e), a), 0) + 1
            carriers_for.append((source, group_id(normalize(text)), text, want))
        return True

    for source, store in MINED_STORES.items():
        n = 0
        for _doc_id, text in iter_store(store):
            n += mine(source, text)
        counters[f"documents_with_definitions:{source}"] = n
        print(f"  mined {source}: {n:,} documents with a definition", flush=True)
    n = 0
    for _path, _title, text in iter_k8s(excluded):
        n += mine("k8s-docs-en", text)
    counters["documents_with_definitions:k8s-docs-en"] = n
    counters["carrier_documents_retained"] = len(carriers_for)

    ambiguous_abbr = {a for a, e in abbr_to_exp.items() if len(e) > 1}
    ambiguous_exp = {e for e, a in exp_to_abbr.items() if len(a) > 1}
    counters["lexicon_raw"] = len(lexicon)
    counters["ambiguous_abbreviations_dropped"] = len(ambiguous_abbr)
    counters["ambiguous_expansions_dropped"] = len(ambiguous_exp)

    def admitted(exp, abbr):
        return abbr not in ambiguous_abbr and normalize(exp) not in ambiguous_exp

    # R2 pass 2: build views from the stored carrier documents only.
    pairs, per_entry = [], {}
    for source, doc_group, text, defs in carriers_for:
        keep = [(e, a) for e, a in defs if admitted(e, a)]
        keep = [(e, a) for e, a in keep
                if per_entry.get((normalize(e), a), 0) < MAX_PAIRS_PER_LEXICON_ENTRY]
        if not keep:
            continue
        made = make_pairs(text, doc_group, source, keep, "R2_abbreviation_expansion", counters)
        for p in made:
            per_entry[(normalize(p["form_a"]), p["form_b"])] = \
                per_entry.get((normalize(p["form_a"]), p["form_b"]), 0) + 1
        pairs += made
    del carriers_for
    counters["R2_pairs"] = len(pairs)

    # R1: the glossary's own alias statements, carried by the k8s documents that use the term.
    aka = glossary_aliases()
    counters["k8s_glossary_alias_forms"] = len(aka)
    r1 = []
    if aka:
        for path, _title, text in iter_k8s(excluded):
            dg = group_id(normalize(text))
            hit = [(t, v) for t, v in aka if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text)]
            if hit:
                r1 += make_pairs(text, dg, "k8s-docs-en", hit, "R1_k8s_glossary_aka", counters)
    counters["R1_pairs"] = len(r1)
    pairs += r1

    gaps.append({"rule": "R3_dataset_paraphrases", "state": "no material",
                 "reason": "no admitted source ships a paraphrase field"})
    gaps.append({"rule": "R4_wikipedia_redirects", "state": "no material",
                 "reason": "admitted Wikipedia-derived datasets carry passages, not the redirect "
                           "graph; a redirect dump would be a new source needing an owner ruling"})

    # Deduplicate on the unordered normalized view pair.
    seen, deduped = set(), []
    for p in pairs:
        key = tuple(sorted((normalize(p["view_a"]), normalize(p["view_b"]))))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    counters["duplicate_pairs_dropped"] = len(pairs) - len(deduped)

    ep = Path(exclude_path) if exclude_path else EXCLUDE_DEFAULT
    fams, shas = load_exclusions(ep)
    if fams or shas:
        kept = [p for p in deduped
                if p["family_id"] not in fams
                and group_id(normalize(p["view_a"])) not in shas
                and group_id(normalize(p["view_b"])) not in shas]
        counters["pairs_removed_by_excluded_families"] = len(deduped) - len(kept)
        deduped = kept
        exclusion_note = (f"applied {ep}: {len(fams)} family ids and {len(shas)} normalized-text "
                          "shas; the text shas are the interoperable key when family ids differ")
    else:
        exclusion_note = (f"{ep} absent or empty: no held-out alias test families exist yet; the "
                          "pool is unfiltered and MUST be re-filtered after that step writes it")
        print(f"  note: {exclusion_note}", flush=True)

    deduped.sort(key=lambda p: (p["rule"], p["pair_id"]))
    with open(POOL, "w", encoding="utf-8") as f:
        for p in deduped:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    rng = random.Random(seed)
    k = max(1, round(len(deduped) * sample_fraction)) if deduped else 0
    sample = rng.sample(deduped, k) if k else []
    with open(SPOTCHECK, "w", encoding="utf-8") as f:
        for p in sample:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    summary = {
        "milestone": "M17", "step": "2b", "date": "2026-09-11",
        "script": "m17src/alias_pairs.py", "script_sha256": sha256_file(Path(__file__)),
        "gpu_used": False, "protected_payloads_opened": False,
        "pairs": len(deduped),
        "pairs_by_rule": {r: sum(1 for p in deduped if p["rule"] == r)
                          for r in sorted({p["rule"] for p in deduped})},
        "families": len({p["family_id"] for p in deduped}),
        "pairs_by_source": {s: sum(1 for p in deduped if p["source"] == s)
                            for s in sorted({p["source"] for p in deduped})},
        "counters": counters,
        "rule_gaps": gaps,
        "excluded_families": {"path": rel(ep), "applied": bool(fams or shas),
                              "family_ids": len(fams), "text_shas": len(shas),
                              "note": exclusion_note},
        "spotcheck": {"path": rel(SPOTCHECK), "seed": seed,
                      "fraction": sample_fraction, "n": len(sample),
                      "sha256": sha256_file(SPOTCHECK) if SPOTCHECK.exists() else None},
        "pool": {"path": rel(POOL),
                 "sha256": sha256_file(POOL) if POOL.exists() else None},
        "family_id_method": "sha256('docgroup:' + carrier document group id)[:16]; the same id "
                            "scheme as m17src/support_manifest.py, rooted at the carrier document "
                            "group because an alias view is document-derived, not a dataset query",
        "not_judged": "bulk structural pairs only; the 200-pair held-out test is judged separately",
    }
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"  {len(deduped):,} pairs, {summary['families']:,} families, "
          f"{len(sample):,} spot-check rows", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0, help="spot-check sample seed (default 0)")
    ap.add_argument("--sample-fraction", type=float, default=0.02)
    ap.add_argument("--exclude-families", default=None,
                    help="JSON with {'families': [...]} of held-out alias test families to remove; "
                         f"defaults to {EXCLUDE_DEFAULT}, and proceeds with a note if absent")
    a = ap.parse_args(argv)
    build(a.seed, a.sample_fraction, a.exclude_families)
    return 0


if __name__ == "__main__":
    sys.exit(main())
