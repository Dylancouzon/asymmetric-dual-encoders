"""M17 step 2b: the admitted-source support manifest.

Counts deduplicated documents and queries for every currently admitted commercial-training
source plus the step-2a Kubernetes slice, assigns each source to one panel domain through the
fixed map below, emits the group/family identifiers the registry's split rules need, and applies
`data.bucket_populations_and_dose_rule.pre_lock_rule` to the measured populations.

What this script is NOT allowed to do, and does not do:

* It reads no protected surface. The six, the reserved four and LoTTE are screened inside the
  M17 executor via `m10src/protected10`, as `results/m17_k8s_source_manifest.json` records. The
  Kubernetes documents counted here are therefore *candidate* training inputs, not admitted ones.
* MS MARCO is present on this box under `work/train/stores/msmarco-pos.json` and inside
  `work/decontam/kept.json`. It is affirmatively licensed non-commercial: validation only, never
  a gradient, target, negative or generation seed (`research/m7-data-licensing.md`, rule change
  2026-09-04). `DENIED_SOURCES` refuses it by name and the run aborts if it ever appears.
* It writes no document or query text into `results/`. Text stays in gitignored `work/m17/`.

Deduplication and grouping methods, recorded verbatim in the result so a later reader does not
have to infer them from the code:

* normalization  NFKC, casefold, collapse all unicode whitespace runs to one space, strip.
* document group  sha256 of the normalized document text, first 16 hex characters. Two documents
  share a group iff their normalized text is identical. This is exact-normalized grouping, not
  MinHash near-duplicate detection; `results/m17_k8s_source_manifest.json` already carries the
  fingerprint near-duplicate screen against the development suite, and the five paths it flagged
  are excluded here by path.
* query family  union-find over admitted training queries. Two queries are joined if their
  normalized text is identical, or if they share a positive document group. A document that is a
  positive for more than `--hub-fanout` distinct queries is a hub and joins nothing: without that
  cutoff a multi-hop corpus collapses into one giant family and the registry's family-level split
  stops separating anything. Hub documents are counted and reported.

Usage (repository root, pre-clock, no GPU):

    .venv/bin/python m17src/support_manifest.py
    .venv/bin/python m17src/support_manifest.py --reuse-counts   # after alias_pairs.py

The heavy pass over the document stores is cached in `work/m17/manifest/counts_cache.json`;
`--reuse-counts` re-applies the dose rule (and picks up the alias-pair population) without
re-reading 2.9 GB of stores.
"""
import argparse
import gzip
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "work"
TRAIN = WORK / "train"
MANIFEST_DIR = WORK / "m17" / "manifest"
RESULT = REPO / "results" / "m17_support_manifest.json"
K8S_JSONL = WORK / "m17" / "sources" / "k8s_docs_en.jsonl"
K8S_STEP2A = REPO / "results" / "m17_k8s_source_manifest.json"

PANEL_DOMAINS = ["cloud-software", "general", "science-engineering", "medicine", "finance", "legal"]

# Fixed source-to-domain map. Mirrored into m17/registry.json data.source_domain_map. An
# unmapped source is 'general' by the registry's rule; every admitted source is mapped here, so
# the fallback is only a guard for a source added later without touching this table.
SOURCE_DOMAIN = {
    "hotpotqa-train": "general",     # Wikipedia multi-hop QA
    "fever-train": "general",        # Wikipedia claim verification
    "squad-train": "general",        # Wikipedia reading comprehension
    "mrtydi-en": "general",          # Wikipedia QA, English split
    "esci-us": "general",            # Amazon shopping queries; e-commerce is not a panel domain
    "nqopen": "general",             # query text only
    "triviaqa": "general",           # query text only
    "k8s-docs-en": "cloud-software",  # step-2a Kubernetes documentation slice
}
DEFAULT_DOMAIN = "general"

# Never a training input. Named explicitly so an accidental re-admission fails loudly.
DENIED_SOURCES = {"msmarco-train", "msmarco-pos", "msmarco"}

PAIR_SOURCES = ["hotpotqa-train", "fever-train", "squad-train", "esci-us", "mrtydi-en"]
QUERYTEXT_SOURCES = ["nqopen", "triviaqa"]
STORE_OF = {
    "hotpotqa-train": "hotpotqa-corpus",
    "fever-train": "fever-pos",
    "squad-train": "squad-ctx",
    "esci-us": "esci-prod",
    "mrtydi-en": "mrtydi-docs",
}

HELDOUT_MOD = 50  # m7src/trainmix.HELDOUT_MOD; re-stated so this script imports no M7 driver.


def heldout(source, qid):
    """m7src/trainmix.heldout, copied rather than imported: importing M7 data modules pulls in
    `datasets` and the M7 path/encoder environment for a two-line hash rule."""
    return int(hashlib.sha256(f"{source}:{qid}".encode()).hexdigest(), 16) % HELDOUT_MOD == 0


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def group_id(normalized):
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def rel(path):
    """Repository-relative path when it is inside the checkout, absolute otherwise (tests)."""
    path = Path(path)
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Union:
    """Minimal union-find over string keys."""

    def __init__(self):
        self.parent = {}

    def add(self, k):
        self.parent.setdefault(k, k)

    def find(self, k):
        p = self.parent
        root = k
        while p[root] != root:
            root = p[root]
        while p[k] != root:
            p[k], k = root, p[k]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# --------------------------------------------------------------------------- inputs

def load_kept():
    """Fingerprint-decontamination survivors from M7. No silent fallback: an undecontaminated
    count would be a different measurement wearing this one's name."""
    kept = json.loads((WORK / "decontam" / "kept.json").read_text())
    kept_qt = json.loads((WORK / "decontam" / "kept_querytext.json").read_text())
    for name in list(kept) + list(kept_qt):
        if name in DENIED_SOURCES and name in PAIR_SOURCES + QUERYTEXT_SOURCES:
            raise SystemExit(f"{name} is non-commercial: validation only, never training")
    return kept, kept_qt


def load_exclusions(path):
    """Held-out families and keys written by the concurrent panel/alias-test step.

    That file's own family ids are rooted in its own union-find, so they need not equal this
    script's; its `key_conventions` field names `sha256(normalized text)[:16]` -- this module's
    `group_id` -- as the interoperable key, and that is what is matched on. Returns (families,
    text shas); both empty when the file is absent, which is a reported state, not an error.
    """
    path = Path(path)
    if not path.exists():
        return set(), set()
    blob = json.loads(path.read_text())
    fams = set(blob.get("excluded_family_ids") or blob.get("families") or [])
    shas = set(blob.get("excluded_text_shas") or [])
    return fams, shas


def iter_store(store_name):
    blob = json.loads((TRAIN / "stores" / f"{store_name}.json").read_text())
    ids, texts = blob["ids"], blob["texts"]
    for i, t in zip(ids, texts):
        yield str(i), t


def k8s_excluded_paths():
    step2a = json.loads(K8S_STEP2A.read_text())
    return set(step2a["decontamination"]["result"]["flagged_paths"])


def iter_k8s(excluded):
    with open(K8S_JSONL, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["path"] in excluded:
                continue
            yield rec["path"], rec["title"], rec["text"]


# --------------------------------------------------------------------------- passes

def document_pass(out_dir, hub_fanout):
    """One pass per document store. Writes doc_id -> group_id and returns per-source counts plus
    the positive-document group lookup the family pass needs."""
    (out_dir / "doc_groups").mkdir(parents=True, exist_ok=True)
    per_source = {}
    doc_group = {}  # source -> {doc_id: group_id}, only for sources with positives
    for src in PAIR_SOURCES:
        store = STORE_OF[src]
        if store in DENIED_SOURCES:
            raise SystemExit(f"{store} is a denied source")
        n_docs = 0
        groups = set()
        mapping = {}
        path = out_dir / "doc_groups" / f"{src}.tsv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as w:
            for doc_id, text in iter_store(store):
                g = group_id(normalize(text))
                n_docs += 1
                groups.add(g)
                mapping[doc_id] = g
                w.write(f"{doc_id}\t{g}\n")
        per_source[src] = {"store": store, "documents": n_docs, "documents_deduplicated": len(groups)}
        doc_group[src] = mapping
        print(f"  {src}: {n_docs:,} documents, {len(groups):,} deduplicated", flush=True)

    excluded = k8s_excluded_paths()
    n_docs, groups = 0, set()
    with gzip.open(out_dir / "doc_groups" / "k8s-docs-en.tsv.gz", "wt", encoding="utf-8") as w:
        for path, _title, text in iter_k8s(excluded):
            g = group_id(normalize(text))
            n_docs += 1
            groups.add(g)
            w.write(f"{path}\t{g}\n")
    per_source["k8s-docs-en"] = {
        "store": "work/m17/sources/k8s_docs_en.jsonl",
        "documents": n_docs,
        "documents_deduplicated": len(groups),
        "excluded_near_duplicate_paths": sorted(excluded),
        "admission": "candidate only; the executor's protected screen has not run",
    }
    print(f"  k8s-docs-en: {n_docs:,} documents, {len(groups):,} deduplicated", flush=True)
    return per_source, doc_group


def query_pass(out_dir, per_source, doc_group, hub_fanout):
    """Deduplicated query counts and family IDs over the admitted training split."""
    kept, kept_qt = load_kept()
    # Queries held out for the judged panel or the 200-pair alias test are not training
    # population. Absent file = nothing excluded, reported in the result.
    _ex_fams, ex_shas = load_exclusions(out_dir / "alias_test_families.json")
    n_excluded = 0
    uf = Union()
    rows = []            # (source, qid, uid, norm_hash)
    by_text = {}         # normalized query text hash -> first uid
    doc_to_uids = {}     # positive group id -> list of uids

    for src in PAIR_SOURCES:
        allow = set(kept[src])
        blob = json.loads((TRAIN / "sources" / f"{src}.json").read_text())
        mapping = doc_group[src]
        n_train, texts = 0, set()
        for p in blob["pairs"]:
            qid = str(p["qid"])
            if qid not in allow or heldout(src, qid):
                continue
            nt = normalize(p["query"])
            h = group_id(nt)
            if h in ex_shas:
                n_excluded += 1
                continue
            n_train += 1
            texts.add(nt)
            uid = f"{src}:{qid}"
            uf.add(uid)
            rows.append((src, qid, uid, h))
            first = by_text.setdefault(h, uid)
            uf.union(uid, first)
            for d in p["pos"]:
                g = mapping.get(str(d))
                if g is not None:
                    doc_to_uids.setdefault(g, []).append(uid)
        per_source[src].update(queries_train=n_train, queries_deduplicated=len(texts))
        print(f"  {src}: {n_train:,} train queries, {len(texts):,} deduplicated", flush=True)

    for src in QUERYTEXT_SOURCES:
        idx = kept_qt[src]
        qs = json.loads((TRAIN / "querytext" / f"{src}.json").read_text())
        n_train, texts = 0, set()
        for i in idx:
            qid = str(i)
            if heldout(src, qid):
                continue
            nt = normalize(qs[i])
            h = group_id(nt)
            if h in ex_shas:
                n_excluded += 1
                continue
            n_train += 1
            texts.add(nt)
            uid = f"{src}:{qid}"
            uf.add(uid)
            rows.append((src, qid, uid, h))
            uf.union(uid, by_text.setdefault(h, uid))
        per_source[src] = {
            "store": None,
            "documents": 0,
            "documents_deduplicated": 0,
            "queries_train": n_train,
            "queries_deduplicated": len(texts),
            "note": "query text only; no positives, so families come from normalized text alone",
        }
        print(f"  {src}: {n_train:,} train queries, {len(texts):,} deduplicated", flush=True)

    hubs = 0
    for g, uids in doc_to_uids.items():
        distinct = set(uids)
        if len(distinct) > hub_fanout:
            hubs += 1
            continue
        it = iter(sorted(distinct))
        first = next(it)
        for u in it:
            uf.union(first, u)

    families = {}
    path = out_dir / "query_families.tsv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as w:
        w.write("source\tqid\tfamily_id\tnormalized_text_sha\n")
        for src, qid, uid, h in rows:
            fam = group_id(uf.find(uid))
            families[fam] = families.get(fam, 0) + 1
            w.write(f"{src}\t{qid}\t{fam}\t{h}\n")

    sizes = sorted(families.values(), reverse=True)
    stats = {
        "queries_total_train": len(rows),
        "queries_deduplicated_total": len(by_text),
        "families": len(families),
        "largest_family": sizes[0] if sizes else 0,
        "largest_family_sizes_top10": sizes[:10],
        "singleton_families": sum(1 for s in sizes if s == 1),
        "families_over_1000": sum(1 for s in sizes if s > 1000),
        "queries_in_families_over_1000": sum(s for s in sizes if s > 1000),
        "concentration_note": "a few giant families survive the hub cutoff through chains of "
                              "sub-cutoff documents; a family-level split must place each of them "
                              "wholly on one side, so the held-out slice, the panel split and the "
                              "200-pair alias test should be drawn outside them",
        "hub_documents_excluded_from_linking": hubs,
        "hub_fanout": hub_fanout,
        "queries_removed_as_heldout": n_excluded,
        "heldout_key_source": rel(out_dir / "alias_test_families.json"),
        "heldout_keys_available": bool(ex_shas),
    }
    print(f"  families: {stats['families']:,}, largest {stats['largest_family']:,}, "
          f"{hubs:,} hub documents", flush=True)
    return per_source, stats


# --------------------------------------------------------------------------- dose rule

def coverage_population(per_source):
    """Document-derived coverage views available before vocabulary discovery.

    A coverage view is a title/heading/first-span view of an admitted document (PLANNING: 'varied
    title/heading/span views of admitted documents'). Every deduplicated document supplies at
    least one, so the deduplicated document count is the honest lower bound on the population and
    is what the dose rule is applied to. The k8s half is reported separately because the 10%
    new-source share caps it.
    """
    new_source = per_source["k8s-docs-en"]["documents_deduplicated"]
    existing = sum(per_source[s]["documents_deduplicated"] for s in PAIR_SOURCES)
    return {"new_source_views_min": new_source, "existing_source_views_min": existing,
            "total_min": new_source + existing}


def dose_rule(general_pop, coverage_pop, alias_pop, steps=6000, batch=256):
    """registry data.bucket_populations_and_dose_rule.pre_lock_rule, applied to measurements.

    Registered shares at batch 256: 192 general views, 32 unpaired coverage views, 16 alias pairs
    (32 views). Floors: 4 pairs / 8 views, 16 coverage views. Interpretation recorded in the
    result: slots freed by shrinking alias or coverage go to general replay, which is the only
    bucket with population to absorb them; general passes are then re-checked, and only after
    that may steps fall. Steps are never raised above 6000.
    """
    registered = {"general_views": 192, "coverage_views": 32, "alias_pairs": 16}
    alias_pairs = registered["alias_pairs"]
    coverage_views = registered["coverage_views"]
    if alias_pop <= 0 or alias_pairs * steps / max(alias_pop, 1) > 4:
        alias_pairs = max(4, int(alias_pop * 4 // steps)) if alias_pop > 0 else 4
        alias_pairs = min(alias_pairs, registered["alias_pairs"])
    if coverage_views * steps / max(coverage_pop, 1) > 4:
        coverage_views = max(16, int(coverage_pop * 4 // steps))
        coverage_views = min(coverage_views, registered["coverage_views"])
    general_views = batch - coverage_views - 2 * alias_pairs
    final_steps = steps
    if general_views * steps / max(general_pop, 1) > 4:
        final_steps = min(steps, int(general_pop * 4 // general_views))
    per_run = {
        "steps": final_steps,
        "batch": batch,
        "general_views_per_batch": general_views,
        "unpaired_coverage_views_per_batch": coverage_views,
        "alias_pairs_per_batch": alias_pairs,
        "general_views": general_views * final_steps,
        "unpaired_coverage_views": coverage_views * final_steps,
        "alias_pair_draws": alias_pairs * final_steps,
        "alias_views": 2 * alias_pairs * final_steps,
    }
    per_run["total_processed_views"] = (
        per_run["general_views"] + per_run["unpaired_coverage_views"] + per_run["alias_views"])
    passes = {
        "general": round(per_run["general_views"] / general_pop, 3) if general_pop else None,
        "unpaired_coverage": round(per_run["unpaired_coverage_views"] / coverage_pop, 3)
        if coverage_pop else None,
        "alias_pairs": round(per_run["alias_pair_draws"] / alias_pop, 3) if alias_pop else None,
    }
    return {
        "registered_shares_at_256": registered,
        "resulting_shares": per_run,
        "passes_per_bucket": passes,
        "shrunk": {"alias": alias_pairs != registered["alias_pairs"],
                   "coverage": coverage_views != registered["coverage_views"],
                   "steps": final_steps != steps},
        "freed_slot_interpretation": "slots freed by an alias or coverage shrink go to general "
                                     "replay; general passes are re-checked before steps fall",
    }


def passes_at_registered_dose(general_pop, coverage_pop, alias_pop):
    """The registry's own 6000x256 numbers against the measured populations, before any shrink."""
    reg = {"general_views": 1152000, "unpaired_coverage_views": 192000, "alias_pair_draws": 96000}
    return {
        "views": reg,
        "general": round(reg["general_views"] / general_pop, 3) if general_pop else None,
        "unpaired_coverage": round(reg["unpaired_coverage_views"] / coverage_pop, 3)
        if coverage_pop else None,
        "alias_pairs": round(reg["alias_pair_draws"] / alias_pop, 3) if alias_pop else None,
        "exceeds_four_passes": [
            b for b, p in (("general", reg["general_views"] / general_pop if general_pop else None),
                           ("unpaired_coverage",
                            reg["unpaired_coverage_views"] / coverage_pop if coverage_pop else None),
                           ("alias_pairs",
                            reg["alias_pair_draws"] / alias_pop if alias_pop else float("inf")))
            if p is None or p > 4],
    }


def new_source_share(per_run_total_views, coverage_views, k8s_view_pop, share_max=0.10):
    """The 10% cap is on processed views per run, not unique queries."""
    cap = int(per_run_total_views * share_max)
    return {
        "share_max_of_processed_views": share_max,
        "processed_views_per_run": per_run_total_views,
        "new_source_view_cap": cap,
        "coverage_views_per_run": coverage_views,
        "new_source_distinct_views_min": k8s_view_pop,
        "cap_binds_before_the_coverage_bucket": coverage_views > cap,
        "passes_over_the_new_source_if_it_filled_the_cap": round(cap / k8s_view_pop, 1)
        if k8s_view_pop else None,
        "new_source_views_at_four_passes": 4 * k8s_view_pop,
        "new_source_share_at_four_passes": round(4 * k8s_view_pop / per_run_total_views, 4)
        if per_run_total_views else None,
        "rule": "k8s-derived views are drawn only inside the coverage bucket and are capped at "
                "new_source_view_cap; the remainder of the coverage bucket comes from existing "
                "admitted documents",
        "finding": "the 10% ceiling is not the binding constraint: the slice is too small to "
                   "supply it without dozens of passes, so the honest allocation is the "
                   "four-pass figure above, far below the cap",
    }


# --------------------------------------------------------------------------- main

def build(hub_fanout, reuse_counts):
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    cache = MANIFEST_DIR / "counts_cache.json"
    if reuse_counts and cache.exists():
        blob = json.loads(cache.read_text())
        per_source, family_stats = blob["per_source"], blob["family_stats"]
        print("reusing cached counts", flush=True)
    else:
        print("document pass", flush=True)
        per_source, doc_group = document_pass(MANIFEST_DIR, hub_fanout)
        print("query pass", flush=True)
        per_source, family_stats = query_pass(MANIFEST_DIR, per_source, doc_group, hub_fanout)
        del doc_group
        cache.write_text(json.dumps({"per_source": per_source, "family_stats": family_stats}))

    per_domain = {d: {"sources": [], "documents_deduplicated": 0, "queries_deduplicated": 0}
                  for d in PANEL_DOMAINS}
    for src, rec in per_source.items():
        d = SOURCE_DOMAIN.get(src, DEFAULT_DOMAIN)
        per_domain[d]["sources"].append(src)
        per_domain[d]["documents_deduplicated"] += rec.get("documents_deduplicated", 0)
        per_domain[d]["queries_deduplicated"] += rec.get("queries_deduplicated", 0)
    gaps = [d for d in PANEL_DOMAINS if not per_domain[d]["sources"]]

    general_pop = min(family_stats["queries_deduplicated_total"], 600000)
    cov = coverage_population(per_source)
    alias_pop, alias_ref = 0, None
    alias_file = MANIFEST_DIR / "alias_pairs.jsonl"
    if alias_file.exists():
        alias_pop = sum(1 for _ in open(alias_file, encoding="utf-8"))
        alias_ref = {"file": rel(alias_file), "sha256": sha256_file(alias_file)}

    applied = dose_rule(general_pop, cov["total_min"], alias_pop)
    result = {
        "milestone": "M17",
        "step": "2b",
        "date": "2026-09-11",
        "script": "m17src/support_manifest.py",
        "script_sha256": sha256_file(Path(__file__)),
        "gpu_used": False,
        "protected_payloads_opened": False,
        "methods": {
            "normalization": "NFKC, casefold, whitespace runs collapsed to one space, stripped",
            "document_group_id": "sha256(normalized document text)[:16]; exact-normalized "
                                 "grouping, not MinHash near-duplicate detection",
            "query_family_id": "union-find: identical normalized query text, or a shared positive "
                               "document group whose fan-out is at most the hub cutoff; "
                               "family_id = sha256(root uid)[:16]",
            "train_split": "M7 fingerprint-decontamination survivors (work/decontam/kept.json, "
                           "kept_querytext.json) minus the sha256 mod-50 held-out slice",
            "denied": "MS MARCO is non-commercial: validation only, excluded by name",
        },
        "source_domain_map": SOURCE_DOMAIN,
        "panel_domains": PANEL_DOMAINS,
        "per_source": per_source,
        "per_domain": per_domain,
        "unpopulated_domains_gap": gaps,
        "breadth_note": "registry vocabulary_ranking.breadth_completion needs three domains at 64 "
                        "rows each; with this map only two domains are populated, so the "
                        "vocabulary outcome can only be recorded as 'narrow'",
        "query_families": family_stats,
        "bucket_populations": {
            "general": {"population": general_pop,
                        "basis": "distinct normalized admitted training queries, capped at "
                                 "training_query_cap 600000",
                        "uncapped": family_stats["queries_deduplicated_total"]},
            "unpaired_coverage": {"population": cov["total_min"], "components": cov,
                                  "basis": "one document-derived view per deduplicated admitted "
                                           "document is the lower bound on the view population"},
            "alias_pairs": {"population": alias_pop, "source": alias_ref,
                            "basis": "distinct structural pairs from m17src/alias_pairs.py"},
        },
        "passes_at_registered_6000x256": passes_at_registered_dose(
            general_pop, cov["total_min"], alias_pop),
        "dose_rule_outcome": applied,
        "new_source_share": new_source_share(
            applied["resulting_shares"]["total_processed_views"],
            applied["resulting_shares"]["unpaired_coverage_views"],
            cov["new_source_views_min"]),
        "manifest_files": {},
        "deferred": {
            "protected_screen": "six, reserved four and LoTTE: executor only, via "
                                "m10src/protected10 (results/m17_k8s_source_manifest.json)",
        },
    }
    for p in sorted(MANIFEST_DIR.rglob("*")):
        if p.is_file() and p.name != "counts_cache.json":
            result["manifest_files"][str(p.relative_to(WORK))] = sha256_file(p)
    RESULT.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {RESULT}", flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hub-fanout", type=int, default=50,
                    help="a positive document linking more than this many distinct queries joins "
                         "no family (default 50)")
    ap.add_argument("--reuse-counts", action="store_true",
                    help="re-apply the dose rule from work/m17/manifest/counts_cache.json")
    a = ap.parse_args(argv)
    build(a.hub_fanout, a.reuse_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
