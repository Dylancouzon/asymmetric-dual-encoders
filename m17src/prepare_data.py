"""M17 step 5: the prepared-data builder.

One CLI that turns the admitted sources into the two prepared directories `train.py --data`
consumes, at a requested size, resumable per stage:

    .venv/bin/python m17src/prepare_data.py --out work/m17/prepared/s2000 --size 2000
    .venv/bin/python m17src/train.py --arm VL-A --rehearsal --data <out>/ext --steps 20 --batch 256
    .venv/bin/python m17src/train.py --arm C    --rehearsal --data <out>/base --steps 20 --batch 256

Stages, each cached under `<out>/stages/<name>.json` and skipped on a re-run:

  1 pool        admitted query pool: general / unpaired-coverage / alias buckets, the caps, the
                seeded stratified subsample and the fixed held-out slice
  2 domain      (source, document id) -> domain join, emitted as `doc_domain.tsv.gz`
  3 protected   protected-surface screen; DEFERRED TO THE CLOCK unless --protected-screen
  4 teacher     frozen stella query vectors through stella's own tokenizer and pinned instruction
  5 bank        the candidate bank: labeled positives first, then uniform admitted documents
  6 v1          zero-v1 query vectors from the released folded int8 table
  7 parity      old-vocabulary parity of the unfolded warm start against the release
  8 vocab       discovery / ranking / selection / row init / tokenizer extension
  9 cache       the candidate cache, its entropy block and the two pre-lock diagnostics
 10 manifests   `base/prepared.json` and `ext/prepared.json`

Layout. The two prepared directories share one cache and one warm start through symlinks, so
the 537 MiB bank and the 94 MiB checkpoint are stored once:

    <out>/shared/   teacher_q.npy  bank.npy  candidates.npz  cache.json  warm_start.npz(+meta)
    <out>/base/     prepared.json  student_ids.json  tokenizer.json  + symlinks into shared
    <out>/ext/      prepared.json  student_ids.json  tokenizer.json  new_rows.npy  + symlinks

What this module may not do, and does not do:

* it opens no protected surface. Every read goes through `common.admit_read`, MS MARCO is
  refused by name exactly as `support_manifest.DENIED_SOURCES` does, and the protected screen
  is a flag that is OFF for pre-clock timing work (reaching protected fingerprints materializes
  their payloads in-process, which pre-clock development may not do);
* it produces no quality number. It reads no development component and no panel;
* it writes nothing into `m17/registry.json`. The measured allocation is written by the
  orchestrator after review, from `results/m17_prepare_timing.json`.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from common import (RESULTS, WORK, admit_read, admit_write, freeze, registry, sha_array,
                    sha_file, sha_json, sha_text, sha_texts, write_json)

import cache as m17cache                                                     # noqa: E402
import support_manifest as sm                                                # noqa: E402
import vocab as m17vocab                                                     # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "work" / "m17" / "manifest"
EXCLUSIONS = MANIFEST_DIR / "alias_test_families.json"
ALIAS_PAIRS = MANIFEST_DIR / "alias_pairs.jsonl"
QUERY_FAMILIES = MANIFEST_DIR / "query_families.tsv.gz"
K8S_JSONL = REPO / "work" / "m17" / "sources" / "k8s_docs_en.jsonl"
POOL_DIR = REPO / "work" / "pool"
RELEASE_TABLE = REPO / "work" / "runs" / "p35w-2m-s2500.release.npz"
WARM_START = REPO / "work" / "runs" / "p35w-2m-s2500.npz"
TEACHER_CACHE = WORK / "prepared" / "teacher_cache"
DOC_CACHE = WORK / "prepared" / "doc_cache"
P0B_PROBES = RESULTS / "m17_tokenizer_followup.json"

STAGES = ("pool", "domain", "protected", "teacher", "bank", "v1", "parity", "vocab", "cache",
          "manifests")

# The 2000-query divergence slice (`training.overfit_divergence_check.heldout_metric`).
HELDOUT_SLICE = 2000
# Share of the bank reserved for the labeled subset's positives; the rest is the uniform draw.
LABELED_POSITIVE_BUDGET_SHARE = 0.5


# --------------------------------------------------------------------------- small helpers

def _denied(name: str):
    """`support_manifest.DENIED_SOURCES`, restated at every entry point of this module."""
    low = str(name).lower()
    if any(d in low for d in sm.DENIED_SOURCES) or "msmarco" in low or "fineweb" in low:
        raise SystemExit(
            f"M17 REFUSED: {name!r} names a source that is not a training input. MS MARCO is "
            "affirmatively licensed non-commercial (validation only: never gradients, targets, "
            "negatives or generation seeds) and FineWeb is out of this nano recipe in every "
            "role (research/m7-data-licensing.md).")
    return name


def _rng(*parts):
    """A seeded generator whose stream depends only on the named parts, never on `hash()`."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _stable_unit(*parts) -> float:
    """A deterministic value in [0, 1) for hash-threshold sampling."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def _rss_gib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 3)


def _gpu_peak_gib():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    except Exception:                                       # pragma: no cover - no torch/GPU
        pass
    return None


def _symlink(dst: Path, src: Path):
    """Point `dst` at `src` (relative), replacing an existing link."""
    dst = Path(admit_write(dst))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src.resolve(), dst.parent.resolve()))
    return dst


def words(text):
    return text.split()


def document_views(doc_key, title, text, max_views=3):
    """Title / heading / span views of one admitted document.

    `data.bucket_populations_and_dose_rule` counts one view per deduplicated document as the
    honest lower bound; this produces up to three deterministic ones. Views are deduplicated by
    normalized text so a short document does not contribute the same sentence three times.
    """
    out, seen = [], set()

    def add(kind, s):
        s = " ".join(s.split())
        n = sm.normalize(s)
        if len(n.split()) < 3 or n in seen:
            return
        seen.add(n)
        out.append({"kind": kind, "text": s})

    heading = next((ln.lstrip("# ").strip() for ln in text.splitlines()
                    if ln.startswith("#")), "")
    if title:
        add("title", title)
    if heading:
        add("heading", heading)
    w = words(text)
    add("span_head", " ".join(w[:24]))
    if len(w) >= 60:
        off = int(_stable_unit("span", doc_key) * (len(w) - 48)) + 24
        add("span_mid", " ".join(w[off:off + 24]))
    return out[:max_views]


class Timer:
    """One timed stage, with its own row/unit counts and rate."""

    def __init__(self, ctx, name, unit="rows"):
        self.ctx, self.name, self.unit, self.n = ctx, name, unit, 0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        secs = round(time.time() - self.t0, 3)
        rec = {"seconds": secs, "unit": self.unit, self.unit: int(self.n),
               f"{self.unit}_per_second": round(self.n / secs, 2) if secs > 0 and self.n else None,
               "rss_high_water_gib": _rss_gib(), "gpu_peak_gib": _gpu_peak_gib()}
        self.ctx.timings[self.name] = rec
        print(f"[stage {self.name}] {secs}s, {self.n} {self.unit}, RSS hwm "
              f"{rec['rss_high_water_gib']} GiB", flush=True)


# --------------------------------------------------------------------------- inputs

def load_families():
    """(source, qid) -> family id, from step 2b's registered `query_families.tsv.gz`."""
    fam = {}
    with gzip.open(admit_read(QUERY_FAMILIES), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            src, qid, family, _sha = line.rstrip("\n").split("\t")
            fam[f"{src}:{qid}"] = family
    return fam


def load_alias_pairs():
    pairs = []
    with open(admit_read(ALIAS_PAIRS), encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def abbreviation_inventory(pairs):
    """(abbreviations set, {abbreviation: expansion}) from the mined structural pairs.

    The inventory is the AUTHORITY `vocab.select` wants: its regex fallback misses
    vowel-carrying abbreviations such as `iam`. The abbreviation of a pair is the form with
    fewer words; a tie falls back to the shorter string.
    """
    abbrs, exp = set(), {}
    for p in pairs:
        a, b = p["form_a"], p["form_b"]
        short, long = (a, b) if (len(a.split()), len(a)) < (len(b.split()), len(b)) else (b, a)
        if len(short.split()) == 1:
            abbrs.add(short.lower())
            exp.setdefault(short.lower(), long.lower())
    return abbrs, exp


# --------------------------------------------------------------------------- stage 1: pool

def _plan(reg, avail, size):
    """Bucket targets at the full pool and, with `--size`, their proportional subsample."""
    dose = reg["data"]["measured_dose_after_pre_lock_rule"]
    steps, batch = int(dose["steps"]), int(dose["batch"])
    cap = int(reg["data"]["training_query_cap"])
    # Four passes is the registered ceiling; a bucket never needs more distinct rows than that.
    cov_target = min(avail["coverage_views"],
                     -(-int(dose["unpaired_coverage_views_per_batch"]) * steps // 4))
    alias_target = min(avail["alias_pairs"], -(-int(dose["alias_pairs_per_batch"]) * steps // 4))
    gen_target = min(avail["general"], cap - cov_target - 2 * alias_target)
    full = {"general": gen_target, "coverage": cov_target, "alias_pairs": alias_target,
            "heldout": min(HELDOUT_SLICE, gen_target // 4)}
    full_total = full["general"] + full["coverage"] + 2 * full["alias_pairs"]
    if size is None or size >= full_total:
        return full, full_total, 1.0
    f = size / full_total
    scaled = {k: max(1, int(round(v * f))) for k, v in full.items()}
    scaled["heldout"] = max(1, min(scaled["heldout"], scaled["general"] // 4))
    return scaled, full_total, f


def _stratified(items, key, target, seed_parts):
    """Seeded subsample to `target`, stratified proportionally by `key(item)`."""
    if target >= len(items):
        return list(items)
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    order = sorted(groups)
    quota = {}
    left = target
    for i, g in enumerate(order):
        q = target * len(groups[g]) // len(items) if i < len(order) - 1 else left
        q = min(q, len(groups[g]), left)
        quota[g] = q
        left -= q
    for g in order:                          # hand any remainder to the largest strata first
        if left <= 0:
            break
        extra = min(left, len(groups[g]) - quota[g])
        quota[g] += extra
        left -= extra
    out = []
    for g in order:
        rows = groups[g]
        idx = _rng(*seed_parts, g).permutation(len(rows))[:quota[g]]
        out.extend(rows[int(i)] for i in sorted(idx))
    return out


def stage_pool(ctx):
    """The admitted query pool: general, unpaired coverage and alias buckets."""
    reg, out = ctx.reg, ctx.out
    kept, kept_qt = sm.load_kept()
    ex = sm.load_exclusions(EXCLUSIONS)
    fam = load_families()
    pairs = load_alias_pairs()

    with Timer(ctx, "pool_general_scan", "queries") as t:
        general, seen_text = [], set()
        for src in sm.PAIR_SOURCES:
            _denied(src)
            allow = set(kept[src])
            blob = json.loads(admit_read(REPO / "work" / "train" / "sources" / f"{src}.json")
                              .read_text())
            for p in blob["pairs"]:
                qid = str(p["qid"])
                if qid not in allow or sm.heldout(src, qid):
                    continue
                nt = sm.normalize(p["query"])
                h = sm.group_id(nt)
                if h in ex["text_shas"] or nt in ex["terms"] or h in seen_text:
                    continue
                seen_text.add(h)
                general.append({"src": src, "qid": qid, "text": p["query"],
                                "pos": [str(d) for d in p["pos"]],
                                "family": fam.get(f"{src}:{qid}", h)})
            del blob
        for src in sm.QUERYTEXT_SOURCES:
            _denied(src)
            qs = json.loads(admit_read(REPO / "work" / "train" / "querytext" / f"{src}.json")
                            .read_text())
            for i in kept_qt[src]:
                qid = str(i)
                if sm.heldout(src, qid):
                    continue
                nt = sm.normalize(qs[i])
                h = sm.group_id(nt)
                if h in ex["text_shas"] or nt in ex["terms"] or h in seen_text:
                    continue
                seen_text.add(h)
                general.append({"src": src, "qid": qid, "text": qs[i], "pos": [],
                                "family": fam.get(f"{src}:{qid}", h)})
            del qs
        t.n = len(general)

    alias_ok = [p for p in pairs
                if p["family_id"] not in ex["families"] and p["doc_group"] not in ex["doc_groups"]]
    avail = {"general": len(general), "alias_pairs": len(alias_ok),
             "coverage_views": int(json.loads(admit_read(RESULTS / "m17_support_manifest.json")
                                              .read_text())["bucket_populations"]
                                   ["unpaired_coverage"]["population"])}
    plan, full_total, factor = _plan(reg, avail, ctx.size)
    print(f"  plan {plan} (full total {full_total}, factor {factor:.4f})", flush=True)

    general = _stratified(general, lambda r: r["src"], plan["general"], (ctx.seed, "general"))
    alias_sel = _stratified(alias_ok, lambda r: r["source"], plan["alias_pairs"],
                            (ctx.seed, "alias"))

    # Held-out: WHOLE families, so a family never straddles the split and the divergence read
    # never sees a row the optimizer sees.
    by_family = {}
    for r in general:
        by_family.setdefault(r["family"], []).append(r)
    heldout_fams, _n = draw_heldout_families(by_family, plan["heldout"], ctx.seed)

    # Coverage documents: a seeded hash-threshold draw over the registered document groups,
    # deduplicated by group, stratified by source in proportion to the deduplicated counts.
    with Timer(ctx, "pool_coverage_sample", "documents") as t:
        cov_docs = _sample_coverage_documents(ctx, plan["coverage"])
        t.n = sum(len(v) for v in cov_docs.values())

    ctx.cache_state["general"] = general
    ctx.cache_state["alias"] = alias_sel
    ctx.cache_state["heldout_families"] = sorted(heldout_fams)
    ctx.cache_state["coverage_docs"] = cov_docs
    # Persisted so `--stages domain` can resume without redoing the source scan.
    write_json(ctx.out / "pool_selection.json",
               {"general": general, "alias": alias_sel,
                "heldout_families": sorted(heldout_fams), "coverage_docs": cov_docs},
               indent=None)
    return {"plan": plan, "available": avail, "full_pool_total": full_total,
            "size_factor": factor,
            "selected": {"general": len(general), "alias_pairs": len(alias_sel),
                         "coverage_documents": sum(len(v) for v in cov_docs.values()),
                         "heldout_families": len(heldout_fams)},
            "exclusions_applied": {k: len(v) for k, v in ex.items()},
            "per_source_general": _tally(r["src"] for r in general),
            "per_source_alias": _tally(r["source"] for r in alias_sel)}


def _tally(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def _sample_coverage_documents(ctx, target_views):
    """-> {source: [doc_id, ...]} for the unpaired-coverage bucket.

    `support_manifest.coverage_population` defines the population as title/heading/span views of
    admitted documents plus the k8s slice minus its flagged paths. Documents are drawn from the
    registered `doc_groups/*.tsv.gz` (deduplicated by group), so the coverage bucket is a subset
    of exactly the population the dose rule was applied to.
    """
    man = json.loads(admit_read(RESULTS / "m17_support_manifest.json").read_text())
    per_source = man["per_source"]
    # Two views per document on average is the planning assumption; the view pass trims to the
    # exact target afterwards, so this only has to be close.
    want_docs = max(1, target_views // 2)
    k8s_pop = per_source["k8s-docs-en"]["documents_deduplicated"]
    dose = ctx.reg["data"]["measured_dose_after_pre_lock_rule"]
    total_views_per_run = int(dose["steps"]) * int(dose["batch"])
    cov_views_per_run = int(dose["unpaired_coverage_views_per_batch"]) * int(dose["steps"])
    share_max = float(ctx.reg["data"]["new_source_share_max_of_training_queries"])
    # The 10% cap is on PROCESSED VIEWS PER RUN. Coverage views are drawn uniformly from this
    # bucket, so the new source's share of the bucket is what has to stay under it.
    k8s_bucket_share_max = share_max * total_views_per_run / cov_views_per_run
    ctx.notes["new_source_cap"] = {
        "share_max_of_processed_views": share_max,
        "processed_views_per_run": total_views_per_run,
        "coverage_views_per_run": cov_views_per_run,
        "max_k8s_share_of_the_coverage_bucket": round(k8s_bucket_share_max, 4)}

    pops = {s: per_source[s]["documents_deduplicated"] for s in sm.PAIR_SOURCES}
    pops["k8s-docs-en"] = k8s_pop
    total = sum(pops.values())
    quota = {s: max(1, int(want_docs * pops[s] / total)) for s in pops}
    k8s_cap = int(k8s_bucket_share_max * want_docs)
    quota["k8s-docs-en"] = min(quota["k8s-docs-en"], k8s_cap, k8s_pop)

    out = {}
    for src in sm.PAIR_SOURCES:
        want = quota[src]
        seen_groups, picked = set(), []
        thresh = min(1.0, 4.0 * want / max(1, pops[src]))
        with gzip.open(admit_read(MANIFEST_DIR / "doc_groups" / f"{src}.tsv.gz"), "rt") as f:
            for line in f:
                if len(picked) >= want:
                    break
                did, gid = line.rstrip("\n").split("\t")
                if gid in seen_groups or _stable_unit(ctx.seed, "cov", src, did) >= thresh:
                    continue
                seen_groups.add(gid)
                picked.append(did)
        out[src] = picked
    excluded = sm.k8s_excluded_paths()
    k8s = [p for p, _t, _x in sm.iter_k8s(excluded)]
    out["k8s-docs-en"] = _stratified([{"d": d} for d in k8s], lambda r: "k8s",
                                     quota["k8s-docs-en"], (ctx.seed, "cov-k8s"))
    out["k8s-docs-en"] = [r["d"] for r in out["k8s-docs-en"]]
    return out


# --------------------------------------------------------------------------- stage 2: domain

def join_domain(doc_domain, group_domain, doc_group, key, gid, domain):
    """Record one document's domain in the join, and REFUSE a second, different label.

    Ruling A3 computes a domain from the DOCUMENT TEXT ALONE, so two labels for one document
    (or for one normalized-text group) mean the caller derived one of them from a query. That
    is the failure `vocab.discover` cannot see and Sol P1-10 asked the builder to bind.
    """
    prev_group = group_domain.setdefault(gid, domain)
    if prev_group != domain:
        raise SystemExit(
            f"M17 REFUSED: document group {gid} is labelled {prev_group!r} and {domain!r}. A "
            "document's domain comes from its own text alone (support_manifest.document_domain).")
    prev = doc_domain.setdefault(key, domain)
    if prev != domain:
        raise SystemExit(
            f"M17 REFUSED: document {key} is labelled {prev!r} and {domain!r}; one document "
            "carries one domain.")
    doc_group[key] = gid
    return domain


def draw_heldout_families(by_family, target, seed):
    """Whole query families for the fixed divergence slice, seeded and reproducible.

    Whole families only: a family that straddles the split would put a near-duplicate of a
    training query into the held-out read. Giant families (the step-2b concentration finding)
    are skipped rather than allowed to swallow the whole slice.
    """
    order = sorted(by_family)
    _rng(seed, "heldout").shuffle(order)
    # A family larger than a quarter of the slice would dominate it; at a scaled-down `--size`
    # that ceiling can exclude every family, so the fallback is the slice size itself. A family
    # bigger than the whole slice is never taken.
    for limit in (max(1, target // 4), target):
        picked, n = set(), 0
        for f in order:
            if n >= target:
                break
            if len(by_family[f]) > limit:
                continue
            picked.add(f)
            n += len(by_family[f])
        if n:
            return picked, n
    return set(), 0


def alias_specs(pair, domain):
    """The two `QuerySpec` views of one admitted alias pair.

    Both views are ordinary coverage-bucket rows of the same schema, sharing the verified pair
    id and ONE query family, so a family-level split can never separate them (registry
    `alias_admission`, `panel_split_unit`).
    """
    return [m17cache.QuerySpec(
        qid=f"alias:{pair['pair_id']}:{view}", text=pair[f"view_{view}"], source=pair["source"],
        domain=domain, bucket="coverage", family="alias:" + pair["family_id"],
        alias_pair_id=pair["pair_id"], alias_view=view) for view in ("a", "b")]


# The prepared directory each arm reads. A control pointed at the extended directory is a
# different experiment; `train._load_prepared` refuses the wrong pairing and this is its half.
ARMS_FOR_FORM = {"base": ["C", "L"], "ext": ["V", "VL", "VL-A"]}


def stage_domain(ctx):
    """The (source, document id) -> domain join, and the coverage views built from the same pass.

    One pass per store. Only the documents this pool actually needs are read out of it (the
    positives of the selected labeled queries, the sampled coverage documents and the alias
    pairs' evidence documents), which is the same labelling a whole-corpus pass would produce
    because `support_manifest.document_domain` reads the DOCUMENT TEXT ALONE.
    """
    general = ctx.cache_state["general"]
    alias = ctx.cache_state["alias"]
    cov_docs = ctx.cache_state["coverage_docs"]
    min_score, min_score_src = sm.classifier_min_score_with_source()

    need = {s: set() for s in sm.PAIR_SOURCES}
    for r in general:
        if r["pos"]:
            need[r["src"]].update(r["pos"])
    for s, ds in cov_docs.items():
        if s != "k8s-docs-en":
            need[s].update(ds)
    alias_groups = {}
    for p in alias:
        if p["source"] != "k8s-docs-en":
            alias_groups.setdefault(p["source"], set()).add(p["doc_group"])

    # group -> one representative document id, for the alias pairs' evidence documents
    group_rep = {}
    for src, groups in alias_groups.items():
        want = set(groups)
        with gzip.open(admit_read(MANIFEST_DIR / "doc_groups" / f"{src}.tsv.gz"), "rt") as f:
            for line in f:
                did, gid = line.rstrip("\n").split("\t")
                if gid in want:
                    group_rep[(src, gid)] = did
                    want.discard(gid)
                    if not want:
                        break
        need[src].update(group_rep[(src, g)] for g in groups if (src, g) in group_rep)

    doc_domain, doc_group, texts, titles = {}, {}, {}, {}
    group_domain = {}
    with Timer(ctx, "domain_store_pass", "documents") as t:
        for src in sm.PAIR_SOURCES:
            if not need[src]:
                continue
            store = _denied(sm.STORE_OF[src])
            want = need[src]
            for doc_id, text in sm.iter_store(store):
                if doc_id not in want:
                    continue
                gid = sm.group_id(sm.normalize(text))
                join_domain(doc_domain, group_domain, doc_group, f"{src}:{doc_id}", gid,
                            sm.document_domain(src, text, min_score))
                texts[f"{src}:{doc_id}"] = text
                t.n += 1
        k8s_cov = set(cov_docs.get("k8s-docs-en", []))
        for path, title, text in sm.iter_k8s(sm.k8s_excluded_paths()):
            key = f"k8s-docs-en:{path}"
            if path in k8s_cov:
                texts[key] = text
                titles[key] = title
            join_domain(doc_domain, group_domain, doc_group, key,
                        sm.group_id(sm.normalize(text)),
                        sm.document_domain("k8s-docs-en", text, min_score))
            t.n += 1

    join = ctx.out / "doc_domain.tsv.gz"
    with gzip.open(admit_write(join), "wt", encoding="utf-8") as w:
        w.write("source\tdoc_id\tgroup_id\tdomain\n")
        for key in sorted(doc_domain):
            src, did = key.split(":", 1)
            w.write(f"{src}\t{did}\t{doc_group[key]}\t{doc_domain[key]}\n")

    # The pool rows, in their final order: general, coverage, alias views, held-out.
    heldout_fams = set(ctx.cache_state["heldout_families"])
    specs, heldout_idx = [], []
    q_only_by_budget = 0
    bank_budget = int(LABELED_POSITIVE_BUDGET_SHARE * ctx.bank_docs)
    positives, labeled_order = set(), sorted(
        range(len(general)), key=lambda i: _stable_unit(ctx.seed, "labeled", general[i]["src"],
                                                        general[i]["qid"]))
    eligible = set()
    for i in labeled_order:
        r = general[i]
        pos = [f"{r['src']}:{d}" for d in r["pos"] if f"{r['src']}:{d}" in doc_domain]
        if not pos:
            continue
        if len(positives | set(pos)) > bank_budget:
            q_only_by_budget += 1
            continue
        positives.update(pos)
        eligible.add(i)

    for i, r in enumerate(general):
        pos = tuple(f"{r['src']}:{d}" for d in r["pos"]) if i in eligible else ()
        src_doc = pos[0] if pos else (f"{r['src']}:querytext-no-document"
                                      if r["src"] in sm.QUERYTEXT_SOURCES
                                      else f"{r['src']}:{r['pos'][0]}" if r["pos"] else
                                      f"{r['src']}:querytext-no-document")
        dom = doc_domain.get(src_doc, "general")
        bucket = "heldout" if r["family"] in heldout_fams else "general"
        if bucket == "heldout":
            heldout_idx.append(len(specs))
        specs.append(m17cache.QuerySpec(
            qid=f"{r['src']}:{r['qid']}", text=r["text"], source=r["src"], domain=dom,
            bucket=bucket, family="q:" + r["family"], positive_ids=pos))

    cov_views = []
    for src, ds in cov_docs.items():
        for did in ds:
            key = f"{src}:{did}"
            if key not in texts:
                continue
            for v in document_views(key, titles.get(key, ""), texts[key]):
                cov_views.append((key, src, v))
    target = ctx.stages["pool"]["plan"]["coverage"]
    cov_views = _stratified(cov_views, lambda r: r[1], target, (ctx.seed, "cov-views"))
    for key, src, v in cov_views:
        specs.append(m17cache.QuerySpec(
            qid=f"cov:{key}:{v['kind']}", text=v["text"], source=src,
            domain=doc_domain.get(key, "general"), bucket="coverage",
            family="doc:" + doc_group.get(key, key)))

    alias_source_doc = {}
    for p in alias:
        src = p["source"]
        key = (f"k8s-docs-en:{p['doc_group']}" if src == "k8s-docs-en"
               else f"{src}:{group_rep.get((src, p['doc_group']), '')}")
        dom = ("cloud-software" if src == "k8s-docs-en"
               else doc_domain.get(key, "general"))
        for s in alias_specs(p, dom):
            # The supporting unit of an alias view is its EVIDENCE DOCUMENT GROUP, not its
            # query family: two pairs from different documents can share a family, and their
            # documents may carry different domains.
            alias_source_doc[s.qid] = f"{src}:doc-group:{p['doc_group']}"
            specs.append(s)

    ctx.cache_state["specs"] = specs
    ctx.cache_state["heldout_idx"] = heldout_idx
    ctx.cache_state["source_doc"] = _source_doc_map(specs, alias_source_doc)
    ctx.cache_state["positives"] = sorted(positives)
    _write_specs(ctx, specs, heldout_idx)
    return {
        "doc_domain_tsv": {"path": sm.rel(join), "sha256": sha_file(join),
                           "documents": len(doc_domain)},
        "classifier_min_score": min_score, "classifier_min_score_source": min_score_src,
        "method": sm.domain_method(),
        "documents_by_domain": _tally(doc_domain.values()),
        "queries_by_bucket": _tally(s.bucket for s in specs),
        "queries_by_domain": _tally(s.domain for s in specs),
        "queries_by_source": _tally(s.source for s in specs),
        "labeled_queries": sum(1 for s in specs if s.positive_ids),
        "distinct_positives": len(positives),
        "labeled_positive_budget": bank_budget,
        "query_only_by_bank_budget": q_only_by_budget,
        "query_only_note": (
            "registry positive_bank_policy: the eligible LABELED subset is chosen here, before "
            "the bank is built, so every selected labeled query's positives fit the cap. The "
            "queries outside it enter the pool as query-only examples BY SELECTION; no labeled "
            "query is converted to query-only later, and the count is recorded."),
        "heldout_queries": len(heldout_idx),
        "coverage_views": sum(1 for s in specs if s.bucket == "coverage" and not s.alias_pair_id),
        "alias_views": sum(1 for s in specs if s.alias_pair_id)}


def _source_doc_map(specs, overrides=None):
    """`vocab.discover`'s `source_doc` per query: the SOURCE-QUALIFIED supporting document.

    Query-text-only sources (nqopen, triviaqa) ship no document at all. They are given one
    sentinel unit per source, deliberately: ruling A3 counts one vote per distinct supporting
    DOCUMENT, and treating each query as its own document would re-introduce exactly the
    per-occurrence counting the ruling removed. The consequence is recorded: a term supported
    only by those sources cannot reach the 20-document minimum.
    """
    out = {}
    overrides = overrides or {}
    for s in specs:
        if s.qid in overrides:
            out[s.qid] = overrides[s.qid]
        elif s.bucket == "coverage" and not s.alias_pair_id:
            out[s.qid] = s.qid.split(":", 1)[1].rsplit(":", 1)[0]
        elif s.positive_ids:
            out[s.qid] = s.positive_ids[0]
        elif s.source in sm.QUERYTEXT_SOURCES:
            out[s.qid] = f"{s.source}:querytext-no-document"
        else:
            out[s.qid] = f"{s.source}:unlabeled"
    return out


def _write_specs(ctx, specs, heldout_idx):
    path = admit_write(ctx.out / "pool.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for s in specs:
            f.write(json.dumps({"qid": s.qid, "text": s.text, "source": s.source,
                                "domain": s.domain, "bucket": s.bucket, "family": s.family,
                                "positive_ids": list(s.positive_ids),
                                "alias_pair_id": s.alias_pair_id,
                                "alias_view": s.alias_view}) + "\n")
    write_json(ctx.out / "heldout_idx.json", heldout_idx)
    write_json(ctx.out / "source_doc.json", ctx.cache_state["source_doc"])
    write_json(ctx.out / "positives.json", ctx.cache_state["positives"])


def _read_specs(ctx):
    specs = []
    with open(admit_read(ctx.out / "pool.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            specs.append(m17cache.QuerySpec(
                qid=d["qid"], text=d["text"], source=d["source"], domain=d["domain"],
                bucket=d["bucket"], family=d["family"],
                positive_ids=tuple(d["positive_ids"]), alias_pair_id=d["alias_pair_id"],
                alias_view=d["alias_view"]))
    ctx.cache_state["specs"] = specs
    ctx.cache_state["heldout_idx"] = json.loads(admit_read(ctx.out / "heldout_idx.json").read_text())
    ctx.cache_state["source_doc"] = json.loads(admit_read(ctx.out / "source_doc.json").read_text())
    ctx.cache_state["positives"] = json.loads(admit_read(ctx.out / "positives.json").read_text())
    return specs


# --------------------------------------------------------------------------- stage 3: screen

SCREEN_DEFERRAL = (
    "DEFERRED TO THE CLOCK. Building the protected fingerprint index materializes the protected "
    "payloads in-process, which pre-clock development may not do (m17/LEDGER.md). The screen "
    "runs inside the executor with --protected-screen, before any m9src import, over the k8s "
    "document texts and every new query view text; hits are dropped and counted.")


def stage_protected(ctx):
    if not ctx.protected_screen:
        return {"state": "deferred_to_clock", "reason": SCREEN_DEFERRAL, "screened": 0,
                "dropped": 0}
    sys.path.insert(0, str(REPO / "m10src"))
    import protected10                                      # noqa: E402  (deliberately late)
    idx = protected10.build()
    specs = ctx.cache_state["specs"]
    kept, dropped = [], []
    with Timer(ctx, "protected_screen", "texts") as t:
        for s in specs:
            if protected10.hits(s.text, idx):
                dropped.append(s.qid)
            else:
                kept.append(s)
            t.n += 1
    ctx.cache_state["specs"] = kept
    return {"state": "run", "screened": len(specs), "dropped": len(dropped),
            "dropped_qids_sample": dropped[:20]}


# --------------------------------------------------------------------------- stage 4: teacher

class TextVectorCache:
    """sha256(text) -> fp16 vector, on disk, reused across runs and across pool sizes."""

    def __init__(self, root, dim):
        self.root = Path(admit_write(root))
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.index_p, self.vec_p = self.root / "index.json", self.root / "vecs.f16.npy"
        self.index = (json.loads(admit_read(self.index_p).read_text())
                      if self.index_p.exists() else {})
        self.vecs = (np.load(admit_read(self.vec_p)) if self.vec_p.exists()
                     else np.zeros((0, dim), dtype=np.float16))
        if self.vecs.shape[0] != len(self.index) or (self.vecs.size and
                                                     self.vecs.shape[1] != dim):
            raise SystemExit(f"M17 REFUSED: text-vector cache {self.root} is inconsistent "
                             f"({self.vecs.shape} vs {len(self.index)} keys).")

    def get(self, texts, encode):
        keys = [sha_text(t) for t in texts]
        missing, seen = [], set()
        for k, t in zip(keys, texts):
            if k not in self.index and k not in seen:
                seen.add(k)
                missing.append((k, t))
        if missing:
            new = encode([t for _k, t in missing])
            new = np.asarray(new, dtype=np.float32)
            new /= np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-9)
            base = self.vecs.shape[0]
            self.vecs = np.concatenate([self.vecs, new.astype(np.float16)], 0)
            for j, (k, _t) in enumerate(missing):
                self.index[k] = base + j
            np.save(self.vec_p, self.vecs)
            write_json(self.index_p, self.index)
        return (self.vecs[[self.index[k] for k in keys]],
                {"requested": len(texts), "encoded": len(missing),
                 "hit_rate": round(1 - len(missing) / max(1, len(texts)), 4),
                 "cache_rows": int(self.vecs.shape[0])})


def teacher_preprocessing(ctx):
    """The frozen query preprocessing, taken from `m7/FREEZE.json`'s encoder spec."""
    spec = ctx.freeze["encoder_spec"]
    import teacher as m7teacher
    if m7teacher.QUERY_PREFIX != spec["query_prefix"]:
        raise SystemExit("M17 REFUSED: the active teacher module's query instruction differs "
                         "from m7/FREEZE.json's encoder_spec.query_prefix; the pinned "
                         "instruction is part of the cache identity.")
    return {"teacher": ctx.reg["teacher"], "revision": ctx.reg["teacher_revision"],
            "instruction": spec["query_prefix"], "max_length": int(spec["max_length"]),
            "pooling": spec["pooling"], "post_dense": spec.get("post_dense"),
            "dtype": "fp16", "normalized": "l2",
            "tokenizer": "the teacher's own tokenizer (" + spec["tokenizer_id"] + ")"}


def stage_teacher(ctx):
    import torch
    import teacher as m7teacher
    pre = teacher_preprocessing(ctx)
    specs = ctx.cache_state["specs"]
    cache = TextVectorCache(TEACHER_CACHE, int(ctx.freeze["encoder_spec"]["dim"]))

    def encode(texts):
        return m7teacher.encode(texts, prefix=pre["instruction"], max_length=pre["max_length"],
                                dtype=torch.float32, device=ctx.device, verbose=True)

    with Timer(ctx, "teacher_encode", "queries") as t:
        vecs, rep = cache.get([s.text for s in specs], encode)
        t.n = rep["encoded"]
    np.save(admit_write(ctx.shared / "teacher_q.npy"), vecs)
    ctx.cache_state["teacher_q"] = vecs
    return {"preprocessing": pre, "cache_dir": sm.rel(TEACHER_CACHE), **rep,
            "queries": len(specs), "teacher_q_sha256": sha_array(vecs)}


# --------------------------------------------------------------------------- stage 5: bank

class PoolReader:
    """The existing frozen stella document-vector memmap (`m7src/pool.py`'s `VEC_DIR`).

    Read directly rather than through `pool.build()`: that function REBUILDS a stale cache, and
    a rebuild of a 12.6 GiB artifact is not something a data preparation run may trigger by
    accident. The encoder is verified from the pool's own `meta.json` before a single row is
    used.
    """

    def __init__(self, name="stella-400M-v5"):
        d = POOL_DIR / name
        self.meta = json.loads(admit_read(d / "meta.json").read_text())
        if self.meta.get("encoder") != name:
            raise SystemExit(f"M17 REFUSED: {d}/meta.json holds vectors for "
                             f"{self.meta.get('encoder')!r}, not {name!r}.")
        if self.meta.get("encoder_revision") != registry()["teacher_revision"]:
            raise SystemExit("M17 REFUSED: the document pool was encoded at revision "
                             f"{self.meta.get('encoder_revision')}, not the registry's.")
        self.dim = int(self.meta["dim"])
        vec_p = admit_read(d / "vecs.f16")
        n = int(self.meta["n"])
        if vec_p.stat().st_size != n * self.dim * 2:
            raise SystemExit(f"M17 REFUSED: {vec_p} is not {n} x {self.dim} fp16.")
        self.vecs = np.memmap(vec_p, dtype=np.float16, mode="r", shape=(n, self.dim))
        self.spans = self.meta["spans"]

    def rows_for(self, store, doc_ids):
        """-> {doc_id: global row}. Builds the id map for one store and drops it afterwards."""
        ids = json.loads(admit_read(POOL_DIR / f"ids-{_denied(store)}.json").read_text())
        off = self.spans[store][0]
        want = set(doc_ids)
        out = {d: off + i for i, d in enumerate(ids) if d in want}
        del ids
        return out


def stage_bank(ctx):
    import torch
    import teacher as m7teacher
    specs = ctx.cache_state["specs"]
    positives = list(ctx.cache_state["positives"])
    reader = PoolReader()
    spec = ctx.freeze["encoder_spec"]

    need = {}
    for key in positives:
        src, did = key.split(":", 1)
        need.setdefault(src, set()).add(did)
    # k8s documents are admitted but absent from the frozen pool: every one of them joins the
    # bank and is encoded below with the frozen document tower. They are counted INSIDE
    # `candidate_bank_max_docs`, not added on top of it.
    k8s_texts = {f"k8s-docs-en:{p}": t for p, _title, t in sm.iter_k8s(sm.k8s_excluded_paths())}
    reserved = len(positives) + len(k8s_texts)
    if reserved > ctx.bank_docs:
        raise SystemExit(
            f"M17 REFUSED: the selected labeled subset's {len(positives)} positives plus the "
            f"{len(k8s_texts)} admitted k8s documents exceed candidate_bank_max_docs "
            f"({ctx.bank_docs}). Shrink the labeled subset before the bank, never afterwards.")
    # Uniform admitted documents, stratified by source, filling the rest of the cap.
    want_uniform = ctx.bank_docs - reserved
    man = json.loads(admit_read(RESULTS / "m17_support_manifest.json").read_text())
    pops = {s: man["per_source"][s]["documents_deduplicated"] for s in sm.PAIR_SOURCES}
    total = sum(pops.values())
    uniform_ids = {}
    with Timer(ctx, "bank_sample", "documents") as t:
        for src in sm.PAIR_SOURCES:
            want = int(want_uniform * pops[src] / total)
            have = need.get(src, set())
            picked, seen_groups = [], set()
            thresh = min(1.0, 4.0 * want / max(1, pops[src]))
            with gzip.open(admit_read(MANIFEST_DIR / "doc_groups" / f"{src}.tsv.gz"), "rt") as f:
                for line in f:
                    if len(picked) >= want:
                        break
                    did, gid = line.rstrip("\n").split("\t")
                    if did in have or gid in seen_groups:
                        continue
                    if _stable_unit(ctx.seed, "bank", src, did) >= thresh:
                        continue
                    seen_groups.add(gid)
                    picked.append(did)
            uniform_ids[src] = picked
            t.n += len(picked)

    ordered, sources = [], []
    for key in positives:                              # positives first, as registered
        ordered.append(key)
        sources.append(key.split(":", 1)[0])
    for src in sm.PAIR_SOURCES:
        for did in uniform_ids.get(src, []):
            ordered.append(f"{src}:{did}")
            sources.append(src)
    for key in sorted(k8s_texts):
        ordered.append(key)
        sources.append("k8s-docs-en")

    vecs = np.zeros((len(ordered), reader.dim), dtype=np.float16)
    missing = []
    with Timer(ctx, "bank_pool_lookup", "documents") as t:
        by_src = {}
        for i, key in enumerate(ordered):
            src, did = key.split(":", 1)
            by_src.setdefault(src, []).append((i, did))
        for src, items in by_src.items():
            if src == "k8s-docs-en":
                missing.extend(items)
                continue
            rows = reader.rows_for(sm.STORE_OF[src], [d for _i, d in items])
            for i, did in items:
                r = rows.get(did)
                if r is None:
                    missing.append((i, did))
                else:
                    vecs[i] = reader.vecs[r]
            t.n += len(items)

    enc_report = {"encoded": 0, "hit_rate": 1.0}
    if missing:
        dcache = TextVectorCache(DOC_CACHE, reader.dim)
        texts = [k8s_texts.get(ordered[i], "") for i, _d in missing]
        blank = [ordered[i] for (i, _d), t in zip(missing, texts) if not t]
        if blank:
            raise SystemExit("M17 REFUSED: no admitted text for bank documents missing from the "
                             f"frozen pool: {blank[:5]}")

        def encode(ts):
            return m7teacher.encode(ts, prefix=spec["doc_prefix"],
                                    max_length=int(spec["max_length"]), dtype=torch.float32,
                                    device=ctx.device, verbose=True)

        with Timer(ctx, "bank_document_encode", "documents") as t:
            mv, enc_report = dcache.get(texts, encode)
            t.n = enc_report["encoded"]
        for (i, _d), v in zip(missing, mv):
            vecs[i] = v

    bank = m17cache.Bank(ordered, vecs, sources, seed=ctx.seed)
    np.save(admit_write(ctx.shared / "bank.npy"), vecs)
    write_json(ctx.shared / "bank_ids.json", ordered)
    ctx.cache_state["bank"] = bank
    return {"n_docs": len(ordered), "from_pool": len(ordered) - len(missing),
            "encoded_with_document_tower": len(missing),
            "document_encode": enc_report,
            "positives": len(positives), "uniform": sum(len(v) for v in uniform_ids.values()),
            "k8s": len(k8s_texts), "per_source": _tally(sources),
            "encoder_spec_source": "m7/FREEZE.json:encoder_spec",
            "identity": bank.identity()}


# --------------------------------------------------------------------------- stage 6: v1

def stage_v1(ctx):
    import torch
    from table import Preproc, load_table
    specs = ctx.cache_state["specs"]
    art_sha = sha_file(RELEASE_TABLE)
    if art_sha != ctx.freeze["table_sha256"]:
        raise SystemExit(f"M17 REFUSED: {RELEASE_TABLE} hashes {art_sha[:12]} but "
                         f"m7/FREEZE.json records {ctx.freeze['table_sha256'][:12]}.")
    pre = Preproc(**ctx.freeze["preproc"])
    if pre.fingerprint() != ctx.freeze["preproc_fingerprint"]:
        raise SystemExit("M17 REFUSED: the FREEZE preproc does not reproduce its own fingerprint.")
    with Timer(ctx, "v1_encode", "queries") as t:
        model = load_table(admit_read(RELEASE_TABLE), variant="fp16", device=ctx.device)
        v1 = model.encode([s.text for s in specs], pre, batch=2048)
        t.n = len(specs)
        del model
        if ctx.device == "cuda":
            torch.cuda.empty_cache()
    v1 = np.asarray(v1, dtype=np.float32)
    v1 /= np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-9)
    np.save(admit_write(ctx.out / "v1_q.npy"), v1.astype(np.float16))
    ctx.cache_state["v1_q"] = v1
    return {"artifact": sm.rel(RELEASE_TABLE), "artifact_sha256": art_sha,
            "preprocessing": ctx.freeze["preproc"],
            "preproc_fingerprint": ctx.freeze["preproc_fingerprint"],
            "weights_folded": True, "queries": len(specs), "v1_q_sha256": sha_array(v1)}


# --------------------------------------------------------------------------- stage 7: parity

WARM_META_NOTE = (
    "m17src/prepare_data.py copied m7/FREEZE.json's training checkpoint verbatim (the npz bytes "
    "still hash to training_checkpoint_sha256) and added the two fields train.load_warm_start "
    "requires the artifact to state explicitly. `weights_folded: false` is DERIVED here from the "
    "checkpoint's non-empty per-token scalars (the folded release carries none) and is then "
    "PROVEN by the old-vocabulary parity check in this same stage; it is not a new claim about "
    "the bytes.")


def _materialize_warm_start(ctx):
    dst = Path(admit_write(ctx.shared / "warm_start.npz"))
    if not dst.exists():
        shutil.copyfile(admit_read(WARM_START), dst)
    got = sha_file(dst)
    want = ctx.freeze["training_checkpoint_sha256"]
    if got != want:
        raise SystemExit(f"M17 REFUSED: warm start copy hashes {got[:12]}, FREEZE records "
                         f"{want[:12]}.")
    meta = json.loads(admit_read(WARM_START.parent / (WARM_START.stem + ".meta.json")).read_text())
    z = np.load(dst)
    if z["token_weights"].size != z["rows_fp16"].shape[0]:
        raise SystemExit("M17 REFUSED: the warm start carries no per-token scalars; it is not "
                         "the unfolded p35w-2m-s2500 checkpoint.")
    meta = {**meta, "weights_folded": False, "learned_weights": True,
            "m17_meta_note": WARM_META_NOTE}
    write_json(ctx.shared / "warm_start.meta.json", meta)
    return dst, got


def stage_parity(ctx):
    import train as m17train
    dst, ck_sha = _materialize_warm_start(ctx)
    with Timer(ctx, "parity", "rows") as t:
        model, lineage = m17train.load_warm_start(dst, expect_vocab=int(ctx.reg["base_vocab"]),
                                                  device="cpu", rehearsal=False)
        rows = model.rows.detach().numpy()
        w = model.token_weights().detach().numpy()
        eff = m17vocab.effective_rows(rows, w)
        released = np.load(admit_read(RELEASE_TABLE))["rows_fp16"].astype(np.float32)
        dev = m17vocab.verify_old_vocab_parity(eff, released)
        t.n = eff.shape[0]
    ctx.cache_state["eff_rows"] = eff
    return {"warm_start": sm.rel(dst), "checkpoint_sha256": ck_sha,
            "freeze_hash_verified": lineage["freeze_hash_verified"],
            "effective_rows": list(eff.shape),
            "old_vocab_parity_max_abs": dev,
            "released_rows_source": sm.rel(RELEASE_TABLE),
            "meta_note": WARM_META_NOTE}


# --------------------------------------------------------------------------- stage 8: vocab

def stage_vocab(ctx):
    from tokenizers import Tokenizer
    from table import get_tokenizer, Preproc, tokenize
    specs = ctx.cache_state["specs"]
    eff = ctx.cache_state["eff_rows"]
    teacher_q = ctx.cache_state["teacher_q"].astype(np.float32)
    v1 = ctx.cache_state["v1_q"]
    residual = 1.0 - (teacher_q * v1).sum(1)

    hf = get_tokenizer()
    base = hf.backend_tokenizer
    pre = Preproc(**ctx.freeze["preproc"])
    base.enable_truncation(max_length=pre.max_length)
    # stella's serialized tokenizer arrives with PADDING ENABLED. The student's ragged bags are
    # built from these ids, so a padded encoding would put hundreds of [PAD] rows into every
    # query's pool. `no_padding()` is part of the frozen rule, not a convenience.
    base.no_padding()
    base_path = ctx.out / "tokenizer_base.json"
    base.save(str(admit_write(base_path)))
    # The serialized backend must reproduce the frozen tokenizer's own ids, or the extended
    # student would be tokenizing by a different rule than the released one.
    probe = [s.text for s in specs[:64]] or ["k8s ingress"]
    if [base.encode(t).ids for t in probe] != tokenize(hf, probe, pre):
        raise SystemExit("M17 REFUSED: the serialized backend tokenizer does not reproduce "
                         "m7src/table.tokenize's ids under the FREEZE preproc.")

    train_idx = [i for i, s in enumerate(specs) if s.bucket != "heldout"]
    src_doc = ctx.cache_state["source_doc"]
    records = [{"text": specs[i].text, "qid": specs[i].qid, "domain": specs[i].domain,
                "source_doc": src_doc[specs[i].qid]} for i in train_idx]
    pairs = load_alias_pairs()
    abbrs, expansions = abbreviation_inventory(pairs)

    with Timer(ctx, "vocab_discover", "queries") as t:
        stats, single = m17vocab.discover(records, residual[train_idx], base)
        t.n = len(records)
    sel = m17vocab.select(stats, ctx.reg, expansions=expansions, abbreviations=abbrs,
                          single_token=single)
    terms = [c["term"] for c in sel["terms"]]
    new_rows, pieces = m17vocab.init_new_rows(terms, base, eff)
    ext = Tokenizer.from_file(str(base_path))
    ext, tok_hash, n_added = m17vocab.extend_tokenizer(ext, terms)
    ext_path = ctx.out / "tokenizer_ext.json"
    ext.save(str(admit_write(ext_path)))
    new_ids = m17vocab.new_row_ids(ext, terms)
    probes = _p0b_probes()
    drift = m17vocab.drift_report(base, ext, eff, new_rows, new_ids, probes) if terms else []

    np.save(admit_write(ctx.out / "new_rows.npy"), new_rows)
    write_json(ctx.out / "vocab_terms.json",
               {"terms": sel["terms"], "new_ids": new_ids, "pieces_per_new_row": pieces})

    with Timer(ctx, "student_tokenize", "queries") as t:
        base_ids = [base.encode(s.text).ids for s in specs]
        ext_ids = [ext.encode(s.text).ids for s in specs]
        t.n = 2 * len(specs)
    write_json(ctx.base / "student_ids.json", base_ids, indent=None)
    write_json(ctx.ext / "student_ids.json", ext_ids, indent=None)
    shutil.copyfile(base_path, admit_write(ctx.base / "tokenizer.json"))
    shutil.copyfile(ext_path, admit_write(ctx.ext / "tokenizer.json"))
    shutil.copyfile(ctx.out / "new_rows.npy", admit_write(ctx.ext / "new_rows.npy"))

    return {"candidates": len(stats), "already_single_token": len(single),
            "selected": len(terms), "terms": terms[:64],
            "per_domain": sel["per_domain"], "breadth": sel["breadth"],
            "dropped_counts": {k: len(v) for k, v in sel["dropped"].items()},
            "notes": sel["notes"],
            "vocabulary_sha256": sel["vocabulary_sha256"],
            "abbreviation_inventory": {"source": sm.rel(ALIAS_PAIRS), "terms": len(abbrs),
                                       "expansions": len(expansions)},
            "base_tokenizer_sha256": sha_file(base_path),
            "extended_tokenizer_sha256": sha_file(ext_path),
            "tokenizer_hash": tok_hash, "added_tokens": n_added,
            "new_rows_sha256": sha_array(new_rows),
            "p0b_drift": drift,
            "source_doc_convention": _source_doc_map.__doc__.strip()}


def _p0b_probes():
    try:
        blob = json.loads(admit_read(P0B_PROBES).read_text())
        return [f["text"] for f in blob["followup"]["shared_piece_fixtures"]]
    except FileNotFoundError:                                # pragma: no cover
        return ["s3 s bucket", "k8s eks networking", "kubernetes kubectl networking"]


# --------------------------------------------------------------------------- stage 9: cache

def stage_cache(ctx):
    specs = ctx.cache_state["specs"]
    bank = ctx.cache_state["bank"]
    teacher_q = ctx.cache_state["teacher_q"].astype(np.float32)
    v1 = ctx.cache_state["v1_q"]
    manifests = {
        "v1_artifact": {k: ctx.stages["v1"][k] for k in
                        ("artifact", "artifact_sha256", "preprocessing",
                         "preproc_fingerprint", "weights_folded")},
        "teacher_query_preprocessing": ctx.stages["teacher"]["preprocessing"],
        "source_split": {"manifest_sha256": sha_file(RESULTS / "m17_support_manifest.json"),
                         "doc_domain_sha256": ctx.stages["domain"]["doc_domain_tsv"]["sha256"],
                         "pool_sha256": sha_file(ctx.out / "pool.jsonl"),
                         "heldout_sha256": sha_json(ctx.cache_state["heldout_idx"])},
        "alias": {"manifest_sha256": sha_file(ALIAS_PAIRS),
                  "exclusions_sha256": sha_file(EXCLUSIONS),
                  "family_split_sha256": sha_texts(sorted({s.family for s in specs}))},
    }
    with Timer(ctx, "cache_build", "queries") as t:
        arrays, sidecar = m17cache.build(specs, bank, teacher_q, v1, ctx.reg,
                                         cache_seed=ctx.seed, manifests=manifests,
                                         progress=max(200, len(specs) // 20))
        t.n = len(specs)
    m17cache.save(ctx.shared, arrays, sidecar)
    ctx.cache_state["cache"] = (arrays, sidecar)

    diag = {"alias_pre_lock_diagnostic": _alias_diagnostic(specs, teacher_q),
            "uninformative_list_stop": _warm_start_kl(arrays, bank, v1, ctx.reg)}
    ctx.cache_state["diagnostics"] = diag
    return {"identity_sha256": sidecar["identity"]["sha256"], "counts": sidecar["counts"],
            "entropy_diagnostic": sidecar["entropy_diagnostic"],
            "provenance_totals": sidecar["provenance_counts"]["total"],
            "teacher_v1_list_overlap": sidecar["provenance_counts"]["teacher_v1_list_overlap"],
            **diag}


def _alias_diagnostic(specs, teacher_q, floor=0.7):
    """Teacher cosine between the two views of each admitted pair; pairs below `floor` flagged."""
    by_pair = {}
    for i, s in enumerate(specs):
        if s.alias_pair_id:
            by_pair.setdefault(s.alias_pair_id, {})[s.alias_view] = i
    cos, flagged = [], []
    for pid, views in sorted(by_pair.items()):
        if set(views) != {"a", "b"}:
            continue
        c = float(teacher_q[views["a"]] @ teacher_q[views["b"]])
        cos.append(c)
        if c < floor:
            flagged.append({"pair_id": pid, "teacher_cosine": round(c, 4)})
    from common import quantiles
    return {"pairs": len(cos), "floor": floor,
            "mean": float(np.mean(cos)) if cos else None,
            "quantiles": quantiles(cos) if cos else {},
            "flagged_below_floor": len(flagged),
            "flagged_share": round(len(flagged) / len(cos), 4) if cos else None,
            "flagged_sample": flagged[:20],
            "note": "report only; no pair is dropped and no threshold is decided here"}


def _warm_start_kl(arrays, bank, v1, reg):
    """KL(teacher || warm-start student) per query over the cached list at the registered T.

    The warm-start student IS the v1 table, so this is the listwise loss the run starts from:
    an input to the registry's `uninformative_list_stop`, reported, never acted on here.
    """
    temp = float(reg["training"]["temperature"])
    cand = arrays["candidate_ids"]
    tsc = arrays["teacher_scores"].astype(np.float64)
    bank_f32 = bank.vectors.astype(np.float32, copy=False)
    kls = []
    for qi in range(cand.shape[0]):
        m = cand[qi] >= 0
        if m.sum() < 2:
            continue
        ids = cand[qi][m]
        ts = tsc[qi][m] / temp
        ss = (bank_f32[ids] @ v1[qi]).astype(np.float64) / temp
        ts -= ts.max()
        ss -= ss.max()
        pt = np.exp(ts) / np.exp(ts).sum()
        ps = np.exp(ss) / np.exp(ss).sum()
        kls.append(float((pt * (np.log(pt) - np.log(ps))).sum()))
    from common import quantiles
    a = np.asarray(kls)
    return {"queries": len(kls), "temperature": temp,
            "median_nats": float(np.median(a)) if len(a) else None,
            "mean_nats": float(a.mean()) if len(a) else None,
            "share_above_0.1_nats": float((a > 0.1).mean()) if len(a) else None,
            "quantiles": quantiles(a) if len(a) else {},
            "note": "inputs to uninformative_list_stop; reported, decided nowhere"}


# --------------------------------------------------------------------------- stage 10

def stage_manifests(ctx):
    specs = ctx.cache_state["specs"]
    out = {}
    for form, d in (("base", ctx.base), ("ext", ctx.ext)):
        for name in ("candidates.npz", "cache.json", "teacher_q.npy", "bank.npy",
                     "bank_ids.json"):
            _symlink(d / name, ctx.shared / name)
        manifest = {
            "_schema": "m17-prepared-v1",
            "form": form,
            "arms": ARMS_FOR_FORM[form],
            "warm_start": os.path.relpath(ctx.shared / "warm_start.npz", d),
            "tokenizer": "tokenizer.json",
            "teacher": ctx.reg["teacher"], "teacher_revision": ctx.reg["teacher_revision"],
            "preproc": ctx.freeze["preproc"],
            "heldout_idx": ctx.cache_state["heldout_idx"],
            "size": ctx.size, "seed": ctx.seed, "date": time.strftime("%Y-%m-%d"),
            "registry_status_at_build": ctx.reg["status"],
            "queries": len(specs),
            "stages": ctx.stages,
            "timings": ctx.timings,
            "notes": ctx.notes,
            "hashes": {
                "prepare_data_py": sha_text(Path(__file__).read_text()),
                "warm_start_npz": ctx.stages["parity"]["checkpoint_sha256"],
                "bank_ids": sha_texts(ctx.cache_state["bank"].doc_ids),
                "bank_vector_bytes": ctx.stages["bank"]["identity"]["vector_bytes_sha256"],
                "teacher_q": ctx.stages["teacher"]["teacher_q_sha256"],
                "v1_q": ctx.stages["v1"]["v1_q_sha256"],
                "tokenizer": sha_file(d / "tokenizer.json"),
                "student_ids": sha_file(d / "student_ids.json"),
                "vocabulary": ctx.stages["vocab"]["vocabulary_sha256"],
                "cache_identity": ctx.stages["cache"]["identity_sha256"],
                "doc_domain_join": ctx.stages["domain"]["doc_domain_tsv"]["sha256"],
                "query_pool": sha_file(ctx.out / "pool.jsonl"),
                "k8s_jsonl": sha_file(K8S_JSONL),
                "exclusion_file": sha_file(EXCLUSIONS),
                "alias_pairs_jsonl": sha_file(ALIAS_PAIRS),
                "support_manifest": sha_file(RESULTS / "m17_support_manifest.json"),
                "freeze": sha_file(REPO / "m7" / "FREEZE.json"),
            },
            "protected_screen": ctx.stages["protected"],
        }
        if form == "ext":
            manifest["new_rows"] = "new_rows.npy"
            manifest["hashes"]["new_rows"] = ctx.stages["vocab"]["new_rows_sha256"]
        write_json(d / "prepared.json", manifest)
        out[form] = sm.rel(d / "prepared.json")
    return out


# --------------------------------------------------------------------------- driver

class Ctx:
    def __init__(self, args):
        self.reg = registry()
        self.freeze = freeze()
        self.out = Path(admit_write(args.out))
        self.shared = self.out / "shared"
        self.base = self.out / "base"
        self.ext = self.out / "ext"
        for d in (self.out, self.shared, self.base, self.ext, self.out / "stages"):
            d.mkdir(parents=True, exist_ok=True)
        self.size = args.size
        self.seed = args.seed
        self.device = args.device
        self.bank_docs = args.bank_docs or int(self.reg["training"]["candidate_bank_max_docs"])
        self.protected_screen = args.protected_screen
        self.stages, self.timings, self.notes, self.cache_state = {}, {}, {}, {}


STAGE_FN = {"pool": stage_pool, "domain": stage_domain, "protected": stage_protected,
            "teacher": stage_teacher, "bank": stage_bank, "v1": stage_v1,
            "parity": stage_parity, "vocab": stage_vocab, "cache": stage_cache,
            "manifests": stage_manifests}

# What a resumed stage has to put back into `cache_state` before later stages run.
def _rehydrate(ctx, name):
    if name == "pool":
        sel = json.loads(admit_read(ctx.out / "pool_selection.json").read_text())
        ctx.cache_state.update(general=sel["general"], alias=sel["alias"],
                               heldout_families=sel["heldout_families"],
                               coverage_docs=sel["coverage_docs"])
        return
    if name == "domain":
        _read_specs(ctx)
    elif name == "teacher":
        ctx.cache_state["teacher_q"] = np.load(admit_read(ctx.shared / "teacher_q.npy"))
    elif name == "v1":
        ctx.cache_state["v1_q"] = np.load(admit_read(ctx.out / "v1_q.npy")).astype(np.float32)
    elif name == "bank":
        vecs = np.load(admit_read(ctx.shared / "bank.npy"))
        ids = json.loads(admit_read(ctx.shared / "bank_ids.json").read_text())
        ctx.cache_state["bank"] = m17cache.Bank(ids, vecs,
                                                [i.split(":", 1)[0] for i in ids], seed=ctx.seed)
    elif name == "parity":
        import train as m17train
        model, _ = m17train.load_warm_start(ctx.shared / "warm_start.npz",
                                            expect_vocab=int(ctx.reg["base_vocab"]),
                                            device="cpu", rehearsal=False)
        ctx.cache_state["eff_rows"] = m17vocab.effective_rows(
            model.rows.detach().numpy(), model.token_weights().detach().numpy())
    elif name == "cache":
        ctx.cache_state["cache"] = m17cache.load(ctx.shared)


def build(args, log=print):
    ctx = Ctx(args)
    want = args.stages.split(",") if args.stages else list(STAGES)
    t0 = time.time()
    for name in STAGES:
        marker = ctx.out / "stages" / f"{name}.json"
        if name not in want:
            if marker.exists():
                ctx.stages[name] = json.loads(admit_read(marker).read_text())
                _rehydrate(ctx, name)
            continue
        if marker.exists() and not args.force:
            log(f"[stage {name}] cached, skipped")
            ctx.stages[name] = json.loads(admit_read(marker).read_text())
            _rehydrate(ctx, name)
            continue
        log(f"[stage {name}] running")
        rec = STAGE_FN[name](ctx)
        ctx.stages[name] = rec
        write_json(marker, rec)
    # A partial re-run (`--stages domain,vocab`) must not erase the measurements the earlier
    # invocations paid for; the carried names are listed so no reader mistakes them for fresh.
    prev_path = ctx.out / "build_record.json"
    carried = []
    if prev_path.exists():
        prev = json.loads(admit_read(prev_path).read_text()).get("timings", {})
        for k, v in prev.items():
            if k not in ctx.timings:
                ctx.timings[k] = v
                carried.append(k)
    record = {"timings_carried_from_a_previous_build": sorted(carried),
              "_schema": "m17-prepared-build-v1", "out": sm.rel(ctx.out), "size": ctx.size,
              "seed": ctx.seed, "bank_docs": ctx.bank_docs, "device": ctx.device,
              "wall_clock_seconds": round(time.time() - t0, 3),
              "rss_high_water_gib": _rss_gib(), "gpu_peak_gib": _gpu_peak_gib(),
              "timings": ctx.timings, "stages": ctx.stages, "notes": ctx.notes,
              "source_sha256": sha_text(Path(__file__).read_text())}
    write_json(ctx.out / "build_record.json", record)
    return record


def timing_report(dirs, full_total=None):
    """Two measured sizes -> per-stage fixed cost, per-row cost and a linear extrapolation.

    A fixed load cost divided by row count is not a per-row cost, so each stage is fitted as
    `seconds = fixed + per_row * rows` from the two observations and BOTH halves are reported.
    A stage whose row count does not change between the two sizes is reported as wholly fixed
    and is never extrapolated per row.
    """
    recs = [json.loads(admit_read(Path(d) / "build_record.json").read_text()) for d in dirs]
    recs.sort(key=lambda r: r["stages"]["pool"]["plan"]["general"])
    small, large = recs
    target = full_total or large["stages"]["pool"]["full_pool_total"]
    stages, total_fixed, total_per_row = {}, 0.0, 0.0
    for name in sorted(set(small["timings"]) | set(large["timings"])):
        a, b = small["timings"].get(name), large["timings"].get(name)
        if not a or not b:
            stages[name] = {"one_size_only": a or b}
            continue
        unit = a["unit"]
        na, nb = a.get(unit, 0), b.get(unit, 0)
        rec = {"unit": unit,
               "small": {"n": na, "seconds": a["seconds"], "rate": a.get(f"{unit}_per_second")},
               "large": {"n": nb, "seconds": b["seconds"], "rate": b.get(f"{unit}_per_second")}}
        if nb != na:
            per_row = (b["seconds"] - a["seconds"]) / (nb - na)
            fixed = max(0.0, a["seconds"] - per_row * na)
            rec.update(per_row_seconds=round(per_row, 6), fixed_seconds=round(fixed, 3),
                       scales_with="queries" if unit == "queries" else unit)
            total_per_row += max(0.0, per_row) if unit == "queries" else 0.0
            total_fixed += fixed
            if unit != "queries":
                # documents/rows stages do not scale with the query pool; their large-size
                # cost is carried forward as a fixed cost of the full build instead.
                total_fixed += max(0.0, b["seconds"] - fixed)
        else:
            rec.update(fixed_seconds=b["seconds"], per_row_seconds=0.0,
                       note="row count identical at both sizes: a fixed cost, not a per-row one")
            total_fixed += b["seconds"]
        stages[name] = rec
    est = total_fixed + total_per_row * target
    return {
        "_schema": "m17-prepare-timing-v1",
        "_note": "Measured pre-clock preparation cost at two real sizes on the RTX 3080 box. "
                 "No quality number, no development or panel read, no protected surface. The "
                 "extrapolation is linear in the query count; stages measured in documents or "
                 "rows are carried as fixed costs of the full build, not divided by queries.",
        "date": time.strftime("%Y-%m-%d"),
        "host": {"gpu": "RTX 3080 10 GiB", "ram_gib": 25},
        "sizes": [{"out": r["out"], "size": r["size"], "queries": r["stages"]["domain"]
                   ["queries_by_bucket"], "wall_clock_seconds": r["wall_clock_seconds"],
                   "rss_high_water_gib": r["rss_high_water_gib"],
                   "gpu_peak_gib": r["gpu_peak_gib"],
                   "teacher_cache": {k: r["stages"]["teacher"][k]
                                     for k in ("encoded", "hit_rate", "cache_rows")},
                   "bank": {k: r["stages"]["bank"][k] for k in
                            ("n_docs", "from_pool", "encoded_with_document_tower")}}
                  for r in recs],
        "stages": stages,
        "extrapolation": {
            "full_pool_queries": target,
            "fixed_seconds": round(total_fixed, 1),
            "per_query_seconds": round(total_per_row, 6),
            "estimated_seconds": round(est, 1),
            "estimated_hours": round(est / 3600, 2),
            "fixed_costs_subtracted": sorted(
                n for n, r in stages.items()
                if isinstance(r, dict) and r.get("unit") in ("documents", "rows")),
            "caveats": [
                "linear in queries; the candidate cache is O(queries x bank) and the bank is "
                "held at the registered cap in both measurements, so its per-query cost is the "
                "real one",
                "the teacher and document encode caches were warm for repeated text; the "
                "measured `encoded` counts say how much was actually paid",
                "single process, no parallelism; the cache stage is CPU-bound numpy"]},
        "source_sha256": sha_text(Path(__file__).read_text()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timing-report", default=None,
                    help="comma-separated prepared directories; writes the two-size timing "
                         "result instead of building anything")
    ap.add_argument("--result", default=str(RESULTS / "m17_prepare_timing.json"))
    ap.add_argument("--out")
    ap.add_argument("--size", type=int, default=None,
                    help="total queries; subsamples every bucket proportionally (stratified "
                         "by source). Omit for the full admitted pool.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bank-docs", type=int, default=None,
                    help="override training.candidate_bank_max_docs (tests only)")
    ap.add_argument("--stages", default=None, help="comma-separated subset of " + ",".join(STAGES))
    ap.add_argument("--force", action="store_true", help="re-run the requested stages")
    ap.add_argument("--protected-screen", action="store_true",
                    help="run the m10 protected screen (CLOCK ONLY: it materializes the "
                         "protected payloads in-process)")
    args = ap.parse_args(argv)
    if args.timing_report:
        rec = timing_report(args.timing_report.split(","))
        write_json(args.result, rec)
        print(f"timing -> {args.result} "
              f"({rec['extrapolation']['estimated_hours']} h for the full pool)")
        return 0
    if not args.out:
        raise SystemExit("--out is required")
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    rec = build(args)
    print(f"prepared -> {rec['out']} ({rec['wall_clock_seconds']}s, RSS hwm "
          f"{rec['rss_high_water_gib']} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
