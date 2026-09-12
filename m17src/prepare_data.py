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
import re
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from common import (RESULTS, WORK, admit_read, admit_write, freeze, reassert_path_order,
                    registry, require_executable, sha_array, sha_file, sha_json, sha_text,
                    sha_texts, write_json)

import cache as m17cache                                                     # noqa: E402
import support_manifest as sm                                                # noqa: E402
import vocab as m17vocab                                                     # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO / "work" / "m17" / "manifest"
EXCLUSIONS = MANIFEST_DIR / "alias_test_families.json"
# The published record of the exclusion artifact's bytes (step 2c wrote both).
ALIAS_TEST_MANIFEST = RESULTS / "m17_alias_test_manifest.json"
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


def _group_key(gid):
    """A 64-bit key for a document-group id, for population-sized dedup sets (Sol P2-7).

    `support_manifest.group_id` is `sha256(normalized text)[:16]` — already 64 bits of hex, so
    the int is exact, not a second hash. Anything else falls back to its own 64-bit digest.
    """
    try:
        return int(gid, 16)
    except ValueError:                                  # pragma: no cover - non-hex group ids
        return int.from_bytes(hashlib.sha256(gid.encode()).digest()[:8], "big")


def _stable_unit(*parts) -> float:
    """A deterministic value in [0, 1) for hash-threshold sampling."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def _rss_gib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), 3)


# RSS high-water marks, ANONYMOUS pages and file-backed pages answer different questions
# (CLAUDE.md, "Before a long run"): a 12.4 GiB high-water that is mostly the frozen document
# memmap's file cache is not 12.4 GiB of memory this process needs. Sol step-5 P2-7 asked for
# the measurement before any redesign, so every stage records all three.
_MEM_FIELDS = ("RssAnon", "RssFile", "RssShmem", "VmHWM", "VmRSS")


def _mem_gib():
    """-> {RssAnon, RssFile, VmHWM, ...} in GiB from `/proc/self/status`, or {} elsewhere."""
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                k, _, rest = line.partition(":")
                if k in _MEM_FIELDS:
                    out[k] = round(int(rest.split()[0]) / (1024 ** 2), 3)
    except OSError:                                     # pragma: no cover - non-Linux
        return {}
    return out


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


def _lowest_hash_pick(keys, want, *seed_parts):
    """Deterministic UNIFORM selection: the `want` keys with the lowest `sha256(seed || key)`.

    Replaces the streaming "accept with probability p, stop at the quota" samplers, which
    stopped around the first quarter of a file and therefore gave later documents a much lower
    inclusion probability (Astra step-5 P1-7). Every eligible key is hashed; only the `want`
    smallest are kept, so the memory cost is the quota, not the population.
    """
    import heapq
    want = max(0, int(want))
    if not want:
        return []
    heap = []                                   # max-heap on the hash value, size <= want
    for key in keys:
        u = _stable_unit(*seed_parts, key)
        if len(heap) < want:
            heapq.heappush(heap, (-u, key))
        elif -heap[0][0] > u:
            heapq.heapreplace(heap, (-u, key))
    return [k for _u, k in sorted((-u, k) for u, k in heap)]


class Timer:
    """One timed stage, with its own row/unit counts and rate.

    `per_query_work` says how many units this stage does PER POOL QUERY, for the timing
    extrapolation: `student_tokenize` encodes every query twice, so its unit count is not the
    query count (Astra step-5 P3-20). `None` means the stage does not scale with the pool.
    """

    def __init__(self, ctx, name, unit="rows", per_query_work=None):
        self.ctx, self.name, self.unit, self.n = ctx, name, unit, 0
        self.per_query_work = 1 if (per_query_work is None and unit == "queries") \
            else per_query_work

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        secs = round(time.time() - self.t0, 3)
        rec = {"seconds": secs, "unit": self.unit, self.unit: int(self.n),
               f"{self.unit}_per_second": round(self.n / secs, 2) if secs > 0 and self.n else None,
               "per_query_work": self.per_query_work,
               "rss_high_water_gib": _rss_gib(), "gpu_peak_gib": _gpu_peak_gib(),
               "memory_gib": _mem_gib()}
        self.ctx.timings[self.name] = rec
        mem = rec["memory_gib"]
        print(f"[stage {self.name}] {secs}s, {self.n} {self.unit}, RSS hwm "
              f"{rec['rss_high_water_gib']} GiB, anon {mem.get('RssAnon')} GiB, file "
              f"{mem.get('RssFile')} GiB, VmHWM {mem.get('VmHWM')} GiB", flush=True)


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


# --------------------------------------------------------------------------- exclusions

EXCLUSION_CATEGORIES = ("general_family", "general_text_sha", "general_alias_term",
                        "general_positive_doc_group", "coverage_doc_group", "coverage_text_sha",
                        "coverage_alias_term", "alias_family", "alias_doc_group",
                        "alias_view_sha", "alias_term")


def _excluded_doc_ids(ex):
    """{source: {document id}} for every document inside an EXCLUDED document group.

    A general query is excluded by its POSITIVE DOCUMENT GROUP, and a query record carries
    document ids, not group ids; the registered `doc_groups/*.tsv.gz` is the join.
    """
    out = {}
    if not ex["doc_groups"]:
        return {s: set() for s in sm.PAIR_SOURCES}
    for src in sm.PAIR_SOURCES:
        hit = set()
        p = MANIFEST_DIR / "doc_groups" / f"{src}.tsv.gz"
        if p.exists():
            with gzip.open(admit_read(p), "rt") as f:
                for line in f:
                    did, gid = line.rstrip("\n").split("\t")
                    if gid in ex["doc_groups"]:
                        hit.add(did)
        out[src] = hit
    return out


def require_exclusions():
    """The step-2c exclusion artifact, MANDATORY and hash-verified (Sol step-5 P1-1).

    `support_manifest.load_exclusions` returns four empty sets when the gitignored
    `alias_test_families.json` is absent, which is the right permissive behaviour for the
    step-2b script (it runs before step 2c exists). For the training-pair builder it is a
    fail-open: on a box where the file was never regenerated, every held-out family, alias
    term, view text and evidence group would be admitted into training. The builder therefore
    refuses a missing file and refuses bytes that are not the ones `results/
    m17_alias_test_manifest.json` published.
    """
    if not EXCLUSIONS.exists():
        raise SystemExit(
            f"M17 REFUSED: the step-2c exclusion artifact {EXCLUSIONS} is missing. Without it "
            "the held-out alias test's families, terms, view texts and evidence document "
            "groups would be admitted into the training pool. Regenerate it with "
            "m17src/alias_test_build.py; empty exclusions are not an acceptable default here.")
    got = sha_file(EXCLUSIONS)
    man = json.loads(admit_read(ALIAS_TEST_MANIFEST).read_text())
    want = (man.get("files") or {}).get("exclusions_sha256")
    if not want:
        raise SystemExit(f"M17 REFUSED: {ALIAS_TEST_MANIFEST} records no "
                         "files.exclusions_sha256; the exclusion bytes cannot be authenticated.")
    if got != want:
        raise SystemExit(
            f"M17 REFUSED: {EXCLUSIONS} hashes {got[:12]} but the alias-test manifest records "
            f"{want[:12]}. These are not the exclusions the held-out alias test was built from.")
    return sm.load_exclusions(EXCLUSIONS)


class Exclusions:
    """ONE exclusion predicate, applied to every admitted record BY ROLE.

    Astra step-5 P1-5: a general query survived an excluded family or positive document group
    whenever its text differed, coverage sampling applied none of the four categories, and the
    reported `exclusions_applied` numbers were inventory sizes rather than evidence that any
    filter ran. `realized` counts actual drops, per category.
    """

    def __init__(self, ex, doc_ids=None):
        self.ex = ex
        self.doc_ids = doc_ids if doc_ids is not None else _excluded_doc_ids(ex)
        self.realized = {k: 0 for k in EXCLUSION_CATEGORIES}

    def _drop(self, category):
        self.realized[category] += 1
        return True

    def general(self, family, text, src, positives):
        """family, normalized-text sha, alias term, positive document group."""
        nt = sm.normalize(text)
        if family in self.ex["families"]:
            return self._drop("general_family")
        if sm.group_id(nt) in self.ex["text_shas"]:
            return self._drop("general_text_sha")
        if nt in self.ex["terms"]:
            return self._drop("general_alias_term")
        bad = self.doc_ids.get(src) or set()
        if any(str(d) in bad for d in positives):
            return self._drop("general_positive_doc_group")
        return False

    def coverage_document(self, src, doc_id, group_id):
        """A coverage view's SOURCE DOCUMENT group (checked while the documents are sampled)."""
        if group_id in self.ex["doc_groups"] or str(doc_id) in (self.doc_ids.get(src) or set()):
            return self._drop("coverage_doc_group")
        return False

    def coverage_view(self, text):
        """A coverage view's own text sha / alias term."""
        nt = sm.normalize(text)
        if sm.group_id(nt) in self.ex["text_shas"]:
            return self._drop("coverage_text_sha")
        if nt in self.ex["terms"]:
            return self._drop("coverage_alias_term")
        return False

    def alias_pair(self, pair):
        """family, BOTH view shas, BOTH alias terms, evidence document group."""
        if pair["family_id"] in self.ex["families"]:
            return self._drop("alias_family")
        if pair["doc_group"] in self.ex["doc_groups"]:
            return self._drop("alias_doc_group")
        for v in ("view_a", "view_b"):
            if sm.group_id(sm.normalize(pair[v])) in self.ex["text_shas"]:
                return self._drop("alias_view_sha")
        for f in ("form_a", "form_b"):
            if f in pair and sm.normalize(pair[f]) in self.ex["terms"]:
                return self._drop("alias_term")
        return False


# --------------------------------------------------------------------------- stage 1: pool

# The order in which the slots left under `training_query_cap` are handed out once the general
# bucket is exhausted, read from `data.cap_fill_priority` (ruling A5). `alias_all_distinct`
# raises the alias bucket from its four-pass minimum to EVERY distinct admitted pair;
# `coverage_fill` raises coverage toward its population (Sol step-5 P1-2).
CAP_FILL_STEPS = ("alias_all_distinct", "coverage_fill")


def _plan(reg, avail, size):
    """Bucket targets at the full pool and, with `--size`, their proportional subsample.

    The four-pass population is a MINIMUM to check against the dose, never a cap on distinct
    examples (Astra step-5 P1-8): the query cap is the only ceiling. Each bucket first takes
    its four-pass minimum, general then takes every distinct admitted query the cap still
    allows, and the slots that remain are handed out in the registered `cap_fill_priority`
    order until the cap is FULL (Sol step-5 P1-2) — 574,229 of a 600,000 cap was 25,771
    admitted rows left unused.
    """
    dose = reg["data"]["measured_dose_after_pre_lock_rule"]
    steps, batch = int(dose["steps"]), int(dose["batch"])
    cap = int(reg["data"]["training_query_cap"])
    priority = list(reg["data"]["cap_fill_priority"])                       # ruling A5
    unknown = [p for p in priority if p not in CAP_FILL_STEPS]
    if unknown:
        raise SystemExit(f"M17 REFUSED: data.cap_fill_priority names {unknown}, which this "
                         f"builder does not implement (known: {list(CAP_FILL_STEPS)}).")
    cov_target = min(avail["coverage_views"],
                     -(-int(dose["unpaired_coverage_views_per_batch"]) * steps // 4))
    alias_target = min(avail["alias_pairs"], -(-int(dose["alias_pairs_per_batch"]) * steps // 4))
    gen_target = max(0, min(avail["general"], cap - cov_target - 2 * alias_target))
    free = cap - gen_target - cov_target - 2 * alias_target
    for step in priority:                      # general is exhausted; hand out what is left
        if free <= 0:
            break
        if step == "alias_all_distinct":       # every distinct admitted pair, two rows each
            take = min(free // 2, avail["alias_pairs"] - alias_target)
            alias_target += max(0, take)
            free -= 2 * max(0, take)
        elif step == "coverage_fill":
            take = min(free, avail["coverage_views"] - cov_target)
            cov_target += max(0, take)
            free -= max(0, take)
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
    ex = require_exclusions()
    fam = load_families()
    pairs = load_alias_pairs()
    with Timer(ctx, "pool_exclusion_join", "documents") as t:
        excl = Exclusions(ex)
        t.n = sum(len(v) for v in excl.doc_ids.values())
    ctx.cache_state["exclusions"] = excl

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
                if h in seen_text:
                    continue
                if excl.general(fam.get(f"{src}:{qid}", h), p["query"], src, p["pos"]):
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
                if h in seen_text:
                    continue
                if excl.general(fam.get(f"{src}:{qid}", h), qs[i], src, ()):
                    continue
                seen_text.add(h)
                general.append({"src": src, "qid": qid, "text": qs[i], "pos": [],
                                "family": fam.get(f"{src}:{qid}", h)})
            del qs
        t.n = len(general)

    alias_ok = [p for p in pairs if not excl.alias_pair(p)]
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
    # Persisted so `--stages domain` can resume without redoing the source scan. Timed: writing
    # the selection scales with the pool and was outside every named timer (Sol step-5 P3-11).
    with Timer(ctx, "pool_serialize", "queries") as t:
        write_json(ctx.out / "pool_selection.json",
                   {"general": general, "alias": alias_sel,
                    "heldout_families": sorted(heldout_fams), "coverage_docs": cov_docs,
                    "exclusions_realized": excl.realized},
                   indent=None)
        t.n = len(general) + 2 * len(alias_sel)
    return {"plan": plan, "available": avail, "full_pool_total": full_total,
            "size_factor": factor,
            "cap_fill_priority": list(reg["data"]["cap_fill_priority"]),    # ruling A5
            "selected": {"general": len(general), "alias_pairs": len(alias_sel),
                         "coverage_documents": sum(len(v) for v in cov_docs.values()),
                         "heldout_families": len(heldout_fams)},
            # REALIZED drops per category, not inventory sizes (Astra step-5 P1-5).
            "exclusions_applied": dict(excl.realized),
            "exclusion_inventory": {k: len(v) for k, v in ex.items()},
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
    # One usable view per document is the only honest lower bound (`bucket_populations_and_
    # dose_rule`), so enough documents are drawn to reach the target even then; the view pass
    # fills from the REALIZED deduplicated views and trims to the exact target (P1-8). Assuming
    # two views per document silently realized ~70% of the coverage target.
    want_docs = max(1, target_views)
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

    excl = ctx.cache_state["exclusions"]

    def eligible(src):
        """Deduplicated by document group, with the excluded groups removed."""
        # Group ids are `sha256(normalized text)[:16]`: keeping them as 64-bit INTS rather
        # than as a population-sized set of Python strings is the same deduplication at about
        # half the memory (Sol step-5 P2-7).
        seen_groups = set()
        with gzip.open(admit_read(MANIFEST_DIR / "doc_groups" / f"{src}.tsv.gz"), "rt") as f:
            for line in f:
                did, gid = line.rstrip("\n").split("\t")
                key = _group_key(gid)
                if key in seen_groups:
                    continue
                seen_groups.add(key)
                if excl.coverage_document(src, did, gid):
                    continue
                yield did

    out = {}
    for src in sm.PAIR_SOURCES:
        # Uniform over the WHOLE eligible population: lowest seeded hash wins (P1-7).
        out[src] = _lowest_hash_pick(eligible(src), quota[src], ctx.seed, "cov", src)
    excluded = sm.k8s_excluded_paths()
    k8s = [p for p, _t, _x in sm.iter_k8s(excluded)]
    out["k8s-docs-en"] = _lowest_hash_pick(k8s, quota["k8s-docs-en"], ctx.seed, "cov-k8s",
                                           "k8s-docs-en")
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


def heldout_document_groups(general, is_heldout, doc_group):
    """The document groups that support a HELD-OUT general query (Astra step-5 P1-6).

    Held-out families were drawn from general queries only, and the other buckets were never
    filtered against them: a document supporting a held-out query could independently supply a
    coverage view or an alias pair, so the optimizer saw the divergence read's document.
    """
    out = set()
    for i, r in enumerate(general):
        if not is_heldout[i]:
            continue
        for d in r["pos"]:
            g = doc_group.get(f"{r['src']}:{d}")
            if g:
                out.add(g)
    return out


def select_labeled_subset(general, doc_domain, is_heldout, bank_budget, seed):
    """-> (eligible indices, positive document keys, queries made query-only by the budget).

    The eligible LABELED subset, chosen in a seeded order BEFORE the bank exists, so that every
    selected query's positives fit `labeled_positive_budget_share` of the bank cap. Ruling A4
    excludes the held-out divergence slice from the budget. A labeled query whose positives are
    ALL missing from the document join is refused by qid rather than silently converted to a
    query-only example with a fallback domain (Astra step-5 P2-18).
    """
    order = sorted((i for i in range(len(general)) if not is_heldout[i]),
                   key=lambda i: _stable_unit(seed, "labeled", general[i]["src"],
                                              general[i]["qid"]))
    positives, eligible, missing, q_only = set(), set(), [], 0
    for i in order:
        r = general[i]
        if not r["pos"]:
            continue
        pos = [f"{r['src']}:{d}" for d in r["pos"] if f"{r['src']}:{d}" in doc_domain]
        if not pos:
            missing.append(f"{r['src']}:{r['qid']}")
            continue
        # `new` first: `positives | set(pos)` copied the whole accumulated set once per labeled
        # query, including every query rejected after the budget bound (Astra step-5 P3-19).
        new = set(pos) - positives
        if len(positives) + len(new) > bank_budget:
            q_only += 1
            continue
        positives |= new
        eligible.add(i)
    if missing:
        raise SystemExit(
            f"M17 REFUSED: {len(missing)} labeled queries have known positives but NONE of them "
            f"joined the document store, e.g. {missing[:5]}. Such a query would become a "
            "query-only example with a fallback domain and the cache builder could never refuse "
            "a positive it was not given; fix the join or drop the source.")
    return eligible, positives, q_only


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
    _require_alias_groups(alias, group_rep)

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
    excl = ctx.cache_state.get("exclusions") or Exclusions(require_exclusions(), doc_ids={})
    heldout_fams = set(ctx.cache_state["heldout_families"])
    is_heldout = [r["family"] in heldout_fams for r in general]
    # ONE document/family split across every bucket (Astra step-5 P1-6).
    heldout_doc_groups = heldout_document_groups(general, is_heldout, doc_group)

    specs, heldout_idx, groups = [], [], {}
    # `training.labeled_positive_budget_share` (A4): the HELD-OUT slice is excluded from the
    # budget, so the monitoring slice never shapes the training bank.
    share = float(ctx.reg["training"]["labeled_positive_budget_share"])
    bank_budget = int(share * ctx.bank_docs)
    with Timer(ctx, "labeled_subset_select", "queries") as t:
        eligible, positives, q_only_by_budget = select_labeled_subset(
            general, doc_domain, is_heldout, bank_budget, ctx.seed)
        t.n = sum(1 for h in is_heldout if not h)

    for i, r in enumerate(general):
        pos = tuple(f"{r['src']}:{d}" for d in r["pos"]) if i in eligible else ()
        # The canonical supporting document is the query's own first joined positive, whether
        # or not the labeled budget admitted it.
        joined = next((f"{r['src']}:{d}" for d in r["pos"]
                       if f"{r['src']}:{d}" in doc_group), None)
        qid = f"{r['src']}:{r['qid']}"
        if joined:
            groups[qid] = (r["src"], doc_group[joined])
        dom = doc_domain.get(joined, "general") if joined else "general"
        bucket = "heldout" if is_heldout[i] else "general"
        if bucket == "heldout":
            heldout_idx.append(len(specs))
        specs.append(m17cache.QuerySpec(
            qid=qid, text=r["text"], source=r["src"], domain=dom,
            bucket=bucket, family="q:" + r["family"], positive_ids=pos))

    cov_views, cov_drops = [], {"heldout_document": 0, "excluded_view": 0}
    for src, ds in cov_docs.items():
        for did in ds:
            key = f"{src}:{did}"
            if key not in texts:
                continue
            if doc_group.get(key) in heldout_doc_groups:
                cov_drops["heldout_document"] += 1
                continue
            for v in document_views(key, titles.get(key, ""), texts[key]):
                if excl.coverage_view(v["text"]):
                    cov_drops["excluded_view"] += 1
                    continue
                cov_views.append((key, src, v))
    # Filled from the REALIZED deduplicated views, then trimmed to the exact target (P1-8).
    # The views the trim did NOT take are kept as the coverage RESERVE: the protected screen
    # drops rows, and a bucket that stays short of its registered target after the screen is a
    # different dose. The reserve is screened with the pool and refills it (Sol step-5 P1-2).
    target = ctx.stages["pool"]["plan"]["coverage"]
    cov_realized = len(cov_views)
    selected = _stratified(cov_views, lambda r: r[1], target, (ctx.seed, "cov-views"))
    chosen = {id(x) for x in selected}
    reserve_views = [x for x in cov_views if id(x) not in chosen]
    cov_views = selected
    for key, src, v in cov_views:
        s, g = _coverage_spec(key, src, v, doc_group, doc_domain)
        groups[s.qid] = g
        specs.append(s)
    reserve = []
    for key, src, v in reserve_views:
        s, g = _coverage_spec(key, src, v, doc_group, doc_domain)
        reserve.append((s, g))
    ctx.cache_state["coverage_reserve"] = reserve

    alias_dropped_heldout_doc = 0
    for p in alias:
        src = p["source"]
        key = (f"k8s-docs-en:{p['doc_group']}" if src == "k8s-docs-en"
               else f"{src}:{group_rep.get((src, p['doc_group']), '')}")
        gid = doc_group.get(key, p["doc_group"])
        if gid in heldout_doc_groups:
            alias_dropped_heldout_doc += 1
            continue
        dom = ("cloud-software" if src == "k8s-docs-en"
               else doc_domain.get(key, "general"))
        for s in alias_specs(p, dom):
            # The supporting unit of an alias view is its EVIDENCE DOCUMENT GROUP, not its
            # query family: two pairs from different documents can share a family, and their
            # documents may carry different domains.
            groups[s.qid] = (src, gid)
            specs.append(s)

    ctx.cache_state["specs"] = specs
    ctx.cache_state["heldout_idx"] = heldout_idx
    ctx.cache_state["source_doc"] = _source_doc_map(
        specs, groups, ctx.reg["data"].get("documentless_sources_vote", "none"))
    ctx.cache_state["positives"] = sorted(positives)
    with Timer(ctx, "spec_serialize", "queries") as t:
        _write_specs(ctx, specs, heldout_idx)
        t.n = len(specs)
    dose = _validate_buckets(ctx, specs)
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
        "labeled_positive_budget_share": share,
        "heldout_excluded_from_positive_budget": sum(
            1 for i, r in enumerate(general) if is_heldout[i] and r["pos"]),
        "query_only_by_bank_budget": q_only_by_budget,
        "query_only_note": (
            "registry positive_bank_policy: the eligible LABELED subset is chosen here, before "
            "the bank is built, so every selected labeled query's positives fit the cap. The "
            "queries outside it enter the pool as query-only examples BY SELECTION; no labeled "
            "query is converted to query-only later, and the count is recorded. Ruling A4 "
            "excludes the held-out divergence slice from the budget."),
        "heldout_queries": len(heldout_idx),
        "heldout_document_groups": len(heldout_doc_groups),
        "cross_bucket_heldout_drops": {"coverage_documents": cov_drops["heldout_document"],
                                       "alias_pairs": alias_dropped_heldout_doc},
        "exclusions_applied_in_views": {k: excl.realized[k] for k in
                                        ("coverage_text_sha", "coverage_alias_term")},
        "coverage_views_realized_before_trim": cov_realized,
        "coverage_views": sum(1 for s in specs if s.bucket == "coverage" and not s.alias_pair_id),
        "alias_views": sum(1 for s in specs if s.alias_pair_id),
        "bucket_dose_check": dose}


def _require_alias_groups(alias, group_rep, documentless=None):
    """Refuse, by pair id, an alias pair whose evidence group does not join (Sol step-5 P2-8).

    A missing join left `group_rep` without the pair's document, so the pair kept its supplied
    group id, fell back to the `general` domain and carried a support identity that was never
    derived from a document. Query-text-only sources ship no document at all (ruling A4) and
    are excepted, as is the new source, whose group id IS its document key.
    """
    documentless = set(documentless if documentless is not None else sm.QUERYTEXT_SOURCES)
    bad = [p["pair_id"] for p in alias
           if p["source"] not in documentless and p["source"] != NEW_SOURCE
           and (p["source"], p["doc_group"]) not in group_rep]
    if bad:
        raise SystemExit(
            f"M17 REFUSED: {len(bad)} alias pairs name an evidence document group that does not "
            f"resolve in their source's group table, e.g. {bad[:5]}. Such a pair would train "
            "with a fabricated `general` domain and a support identity no document backs; fix "
            "the join or drop the pairs.")
    return True


def _coverage_spec(key, src, view, doc_group, doc_domain):
    """One unpaired-coverage row and its supporting (source, document group)."""
    gid = doc_group.get(key, key)
    return m17cache.QuerySpec(
        qid=f"cov:{key}:{view['kind']}", text=view["text"], source=src,
        domain=doc_domain.get(key, "general"), bucket="coverage",
        family="doc:" + gid), (src, gid)


def _source_doc_map(specs, groups, documentless_vote="none"):
    """`vocab.discover`'s `source_doc` per query: ONE canonical joined document-group key.

    `groups` maps qid -> (source, document group id) for every example that HAS a supporting
    document. The same document yields the same key whether it appears through a general query,
    a coverage view or an alias pair, and whether or not the labeled positive budget bound it
    (Astra step-5 P1-10); the old mapping spelled the same document three ways and collapsed
    budget-bound documents onto one `source:unlabeled` key.

    An example with no supporting document at all maps to `None`: under
    `data.documentless_sources_vote == "none"` (ruling A4) nqopen and triviaqa contribute
    contexts and residuals to discovery but cast no distinct-document vote.
    """
    if documentless_vote != "none":
        raise SystemExit("M17 REFUSED: this builder implements only "
                         f"data.documentless_sources_vote == 'none'; the registry says "
                         f"{documentless_vote!r}.")
    out = {}
    for s in specs:
        g = groups.get(s.qid)
        out[s.qid] = f"{g[0]}:doc-group:{g[1]}" if g else None
    return out


def _validate_buckets(ctx, specs):
    """Realized bucket populations against the registered dose and the query cap (P1-8).

    The four-pass population is the MINIMUM each bucket needs; a shortfall is recorded (and,
    for a bucket that cannot even supply one batch, refused) rather than silently accepted.
    """
    dose = ctx.reg["data"]["measured_dose_after_pre_lock_rule"]
    steps, batch = int(dose["steps"]), int(dose["batch"])
    per_batch = {"general": int(dose["general_views_per_batch"]),
                 "coverage": int(dose["unpaired_coverage_views_per_batch"]),
                 "alias_pairs": int(dose["alias_pairs_per_batch"])}
    realized = {
        "general": sum(1 for s in specs if s.bucket == "general"),
        "coverage": sum(1 for s in specs if s.bucket == "coverage" and not s.alias_pair_id),
        "alias_pairs": len({s.alias_pair_id for s in specs if s.alias_pair_id})}
    n_heldout = sum(1 for s in specs if s.bucket == "heldout")
    plan = dict(ctx.stages["pool"]["plan"])
    # The held-out slice is drawn OUT of the general bucket, so the planned training general
    # count is the plan minus the realized slice.
    plan["general"] = max(0, plan["general"] - n_heldout)
    factor = float(ctx.stages["pool"]["size_factor"])
    out = {"realized": realized, "planned": {k: plan[k] for k in realized},
           "per_batch": per_batch, "steps": steps, "batch": batch,
           "size_factor": factor, "buckets": {}}
    for name, n in realized.items():
        need_4pass = -(-per_batch[name] * steps // 4)
        short = plan[name] - n
        out["buckets"][name] = {
            "realized": n, "planned": plan[name], "shortfall_vs_plan": max(0, short),
            "four_pass_minimum_at_full_pool": need_4pass,
            "passes_at_the_registered_dose": round(per_batch[name] * steps / n, 3) if n else None,
            # A subsampled build is a timing measurement, not the registered dose.
            "meets_four_pass_minimum": (factor >= 1.0 and n >= need_4pass)}
        if n < per_batch[name]:
            raise SystemExit(
                f"M17 REFUSED: the {name} bucket realized {n} distinct rows but one batch of the "
                f"registered dose needs {per_batch[name]}. A short batch is a different dose "
                "(registry bucket_populations_and_dose_rule.pre_lock_rule).")
    total = realized["general"] + realized["coverage"] + 2 * realized["alias_pairs"] + n_heldout
    cap = int(ctx.reg["data"]["training_query_cap"])
    out["total_rows"] = total
    out["training_query_cap"] = cap
    if factor >= 1.0 and total > cap:
        raise SystemExit(f"M17 REFUSED: the realized pool holds {total} rows, above "
                         f"training_query_cap {cap}.")
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
    reserve = ctx.cache_state.get("coverage_reserve")
    if reserve is not None:
        with open(admit_write(ctx.out / "coverage_reserve.jsonl"), "w", encoding="utf-8") as f:
            for s, g in reserve:
                f.write(json.dumps({"qid": s.qid, "text": s.text, "source": s.source,
                                    "domain": s.domain, "family": s.family,
                                    "group": list(g)}) + "\n")


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
    p = ctx.out / "coverage_reserve.jsonl"
    reserve = []
    if p.exists():
        with open(admit_read(p), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                reserve.append((m17cache.QuerySpec(
                    qid=d["qid"], text=d["text"], source=d["source"], domain=d["domain"],
                    bucket="coverage", family=d["family"]), tuple(d["group"])))
    ctx.cache_state["coverage_reserve"] = reserve
    return specs


# --------------------------------------------------------------------------- stage 3: screen

SCREEN_DEFERRAL = (
    "DEFERRED TO THE CLOCK. Building the protected fingerprint index materializes the protected "
    "payloads in-process, which pre-clock development may not do (m17/LEDGER.md). The screen "
    "runs inside the executor with --protected-screen, before any m9src import, over the k8s "
    "document texts and every new query view text; hits are dropped and counted.")


NEW_SOURCE = "k8s-docs-en"


def _new_source_rows(specs):
    """The pool rows a DEFERRED screen leaves unscreened: every row from the new source."""
    return [s for s in specs if s.source == NEW_SOURCE]


def _apply_screen(ctx, drop_qids):
    """Persist the SCREENED pool and rebuild every index derived from it (Astra step-5 P1-4).

    Dropping an early row moved every later position, so `heldout_idx` had to be rebuilt;
    dropping ONE alias view left an incomplete pair, so both views go together; and the stage
    only updated in-memory `specs`, so a restart reloaded the unscreened `pool.jsonl` while
    skipping the successful marker.
    """
    specs = ctx.cache_state["specs"]
    pairs_hit = {s.alias_pair_id for s in specs if s.alias_pair_id and s.qid in drop_qids}
    kept = [s for s in specs if s.qid not in drop_qids
            and not (s.alias_pair_id and s.alias_pair_id in pairs_hit)]
    dropped = len(specs) - len(kept)
    src_doc = dict(ctx.cache_state["source_doc"])

    # REFILL the coverage bucket from the screened reserve, up to its registered target
    # (Sol step-5 P1-2). A bucket that cannot be refilled — the alias bucket holds every
    # distinct admitted pair already — records its shortfall instead, and the dose rule is
    # re-applied below to state the passes the screened populations actually support.
    reserve = [(s, g) for s, g in (ctx.cache_state.get("coverage_reserve") or [])
               if s.qid not in drop_qids]
    # The refill restores the PLANNED TOTAL under A5's order (general, all alias pairs, then
    # coverage to the cap): a screened-out general row or alias pair is replaced by a coverage
    # view too, not only a screened-out coverage view (Sol re-check P1-2). The reserve is the
    # only bucket with spare distinct population, so coverage is the only bucket that can grow.
    plan = ctx.stages["pool"]["plan"]        # heldout rows are drawn FROM the general count
    # In a real build the pool IS the planned total; restoring the pre-screen row count is the
    # same statement and also holds for fixtures whose plan is smaller than their rows.
    target_total = max(int(plan["general"]) + int(plan["coverage"]) + 2 * int(plan["alias_pairs"]),
                       len(specs))
    target = int(plan["coverage"])
    have = sum(1 for s in kept if s.bucket == "coverage" and not s.alias_pair_id)
    rows_after_screen = len(kept)
    short = max(0, target_total - rows_after_screen)
    refill = _stratified([r for r in reserve], lambda r: r[0].source, short,
                         (ctx.seed, "cov-refill")) if short else []
    for s, g in refill:
        src_doc[s.qid] = f"{g[0]}:doc-group:{g[1]}"
        kept.append(s)
    used = {id(r) for r in refill}
    ctx.cache_state["coverage_reserve"] = [r for r in reserve if id(r) not in used]

    heldout_idx = [i for i, s in enumerate(kept) if s.bucket == "heldout"]
    ctx.cache_state["specs"] = kept
    ctx.cache_state["heldout_idx"] = heldout_idx
    ctx.cache_state["source_doc"] = {s.qid: src_doc.get(s.qid) for s in kept}
    ctx.cache_state["positives"] = sorted({p for s in kept for p in s.positive_ids})
    _write_specs(ctx, kept, heldout_idx)
    # Refuses a bucket that can no longer supply one batch of the registered dose.
    dose_check = _validate_buckets(ctx, kept)
    realized = dose_check["realized"]
    return {"rows_before": len(specs), "rows_kept": len(kept),
            "rows_dropped": dropped, "alias_pairs_dropped": len(pairs_hit),
            "heldout_rebuilt_to": len(heldout_idx),
            "positives_kept": len(ctx.cache_state["positives"]),
            "coverage_refill": {"target": target, "after_screen": have,
                                "target_total": target_total,
                                "rows_after_screen": rows_after_screen,
                                "refilled_from_reserve": len(refill),
                                "reserve_left": len(ctx.cache_state["coverage_reserve"]),
                                "remaining_shortfall": max(0, target_total - len(kept)),
                                "rule": "A5: coverage restores the planned total, replacing "
                                        "screened-out rows of ANY bucket"},
            "bucket_shortfalls_after_screen": {
                k: v["shortfall_vs_plan"] for k, v in dose_check["buckets"].items()},
            "dose_rule_reapplied": sm.dose_rule(realized["general"], realized["coverage"],
                                                realized["alias_pairs"]),
            "bucket_dose_check_after_screen": dose_check}


def stage_protected(ctx):
    specs = ctx.cache_state["specs"]
    if not ctx.protected_screen:
        # The filter is a no-op pre-clock, and the manifest says so in as many words.
        return {"state": "deferred_to_clock", "reason": SCREEN_DEFERRAL, "screened": 0,
                "dropped": 0, "receipt": None,
                "unscreened_new_source_rows": len(_new_source_rows(specs)),
                "unscreened_note": (
                    f"the bank and the pool contain {NEW_SOURCE} rows that no protected screen "
                    "has seen; a real run refuses this state (train._load_prepared) and only "
                    "`--rehearsal --data` may smoke it, loudly (ruling A4)")}
    # The status gate BEFORE `protected10` is imported at all: building its index materializes
    # the protected payloads in-process (Astra step-5 P1-1). `--stages protected` lands here too.
    require_executable(ctx.reg, rehearsal=False, what="the m10 protected screen")
    sys.path.insert(0, str(REPO / "m10src"))
    import protected10                                      # noqa: E402  (deliberately late)
    reassert_path_order()   # protected10 put m7src first; a later `import train` must stay M17's
    idx = protected10.build()

    # 1. every NEW-SOURCE DOCUMENT, screened on its FULL TEXT (Astra step-5 P1-2). A document
    #    can match a protected surface while the views sampled from it do not.
    admitted, dropped_docs, admitted_groups = [], [], set()
    with Timer(ctx, "protected_document_screen", "documents") as t:
        for path, _title, text in sm.iter_k8s(sm.k8s_excluded_paths()):
            gid = sm.group_id(sm.normalize(text))
            if protected10.hits(text, idx):
                dropped_docs.append(path)
            else:
                admitted.append({"path": path, "sha256": sha_text(text), "group_id": gid})
                admitted_groups.add(gid)
            t.n += 1
    receipt = ctx.out / "protected_receipt.json"
    write_json(receipt, {"_schema": "m17-protected-receipt-v1", "source": NEW_SOURCE,
                         "screen_version": getattr(protected10, "VERSION", ""),
                         "admitted_documents": admitted,
                         "dropped_document_paths": dropped_docs}, indent=None)

    # 2. every query view text, and every row whose supporting new-source document was dropped.
    src_doc = dict(ctx.cache_state["source_doc"])
    # The coverage RESERVE is screened with the pool: rows that refill the bucket after the
    # screen must have passed the same screen (Sol step-5 P1-2).
    reserve = ctx.cache_state.get("coverage_reserve") or []
    for s, g in reserve:
        src_doc.setdefault(s.qid, f"{g[0]}:doc-group:{g[1]}")
    drop_qids, by_text, by_document = set(), 0, 0
    with Timer(ctx, "protected_screen", "texts") as t:
        for s in list(specs) + [r[0] for r in reserve]:
            if protected10.hits(s.text, idx):
                drop_qids.add(s.qid)
                by_text += 1
            elif s.source == NEW_SOURCE:
                g = (src_doc.get(s.qid) or "").rsplit(":", 1)[-1]
                if g and g not in admitted_groups:
                    drop_qids.add(s.qid)
                    by_document += 1
            t.n += 1
    applied = _apply_screen(ctx, drop_qids)
    return {"state": "complete", "screened": len(specs), "dropped": len(drop_qids),
            "dropped_by_text": by_text, "dropped_by_document_receipt": by_document,
            "documents_screened": len(admitted) + len(dropped_docs),
            "documents_admitted": len(admitted), "documents_dropped": len(dropped_docs),
            "dropped_document_paths_sample": dropped_docs[:20],
            "receipt": {"path": sm.rel(receipt), "sha256": sha_file(receipt)},
            "pool_after_screen": applied,
            "unscreened_new_source_rows": 0,
            "dropped_qids_sample": sorted(drop_qids)[:20]}


def _screen_receipt(ctx):
    """-> {admitted new-source document paths} or None when the screen is deferred."""
    rec = ctx.stages.get("protected") or {}
    if rec.get("state") != "complete" or not rec.get("receipt"):
        return None
    blob = json.loads(admit_read(ctx.out / "protected_receipt.json").read_text())
    return {d["path"] for d in blob["admitted_documents"]}


# --------------------------------------------------------------------------- stage 4: teacher

class TextVectorCache:
    """sha256(text) -> fp16 vector, on disk, reused across runs and across pool sizes.

    The cache namespace is BOUND to the encoder/preprocessing manifest it was built under
    (Astra step-5 P2-14): changing the teacher's compute settings, tokenizer or preprocessing
    and then reusing these vectors because the text hashes still match would label old vectors
    with the new manifest. A mismatch refuses. A manifest is STAMPED only on an EMPTY cache:
    stamping a populated one relabels vectors nothing authenticated (Sol step-5 P2-6). The two
    caches in `work/m17/prepared` were stamped by hand on 2026-09-11, with the evidence
    recorded in `m17/CODEMAP.md`.
    """

    def __init__(self, root, dim, manifest=None):
        self.root = Path(admit_write(root))
        self.root.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.manifest_sha = sha_json(manifest) if manifest is not None else None
        self.manifest_stamped = False
        if manifest is not None:
            p = self.root / "manifest.json"
            if p.exists():
                prev = json.loads(admit_read(p).read_text())
                if prev.get("sha256") != self.manifest_sha:
                    raise SystemExit(
                        f"M17 REFUSED: the text-vector cache {self.root} was built under "
                        f"preprocessing manifest {str(prev.get('sha256'))[:12]} but this run "
                        f"uses {self.manifest_sha[:12]}. Matching text hashes are not matching "
                        "vectors; build a new cache directory or restore the preprocessing.")
            elif (self.root / "index.json").exists() or (self.root / "vecs.f16.npy").exists():
                raise SystemExit(
                    f"M17 REFUSED: the text-vector cache {self.root} holds vectors but no "
                    "`manifest.json`. Stamping the current preprocessing on rows nobody "
                    "authenticated would relabel vectors that may have been encoded under "
                    "another tokenizer or configuration. Re-encode into a new cache directory, "
                    "or stamp the manifest deliberately and record why.")
            else:
                self.manifest_stamped = True
                write_json(p, {"sha256": self.manifest_sha, "manifest": manifest,
                               "note": "stamped when this cache directory was created empty"})
        self.index_p, self.vec_p = self.root / "index.json", self.root / "vecs.f16.npy"
        self.index = (json.loads(admit_read(self.index_p).read_text())
                      if self.index_p.exists() else {})
        self.vecs = (np.load(admit_read(self.vec_p)) if self.vec_p.exists()
                     else np.zeros((0, dim), dtype=np.float16))
        if self.vecs.shape[0] > len(self.index) and (not self.vecs.size or
                                                     self.vecs.shape[1] == dim):
            # A crash between the vector replace and the index write leaves a vector suffix no
            # key points at (Astra lock review P2-10). The committed prefix is exactly what the
            # index describes; the suffix is dropped and its texts are re-encoded as misses.
            print(f"  [cache] {self.root}: dropping {self.vecs.shape[0] - len(self.index)} "
                  "uncommitted vector rows (index written after a crash)", flush=True)
            self.vecs = np.ascontiguousarray(self.vecs[:len(self.index)])
        if self.vecs.shape[0] != len(self.index) or (self.vecs.size and
                                                     self.vecs.shape[1] != dim):
            raise SystemExit(f"M17 REFUSED: text-vector cache {self.root} is inconsistent "
                             f"({self.vecs.shape} vs {len(self.index)} keys).")

    FLUSH_EVERY = 25_000    # texts encoded between durable flushes

    def _flush(self):
        # tmp + replace: a crash mid-write must not leave `vecs` and `index` disagreeing (the
        # constructor refuses an inconsistent cache, which would cost the whole cache).
        tmp = self.vec_p.with_suffix(".tmp.npy")
        np.save(tmp, self.vecs)
        tmp.replace(self.vec_p)
        write_json(self.index_p, self.index)

    def get(self, texts, encode, flush_every=None):
        """Vectors for `texts` in order, encoding the misses in chunks of `flush_every` and
        flushing after each: the full pool's teacher stage encodes ~550k texts and a CUDA fault
        at 120k once lost all of them (2026-09-11), because nothing was written until the end."""
        keys = [sha_text(t) for t in texts]
        missing, seen = [], set()
        for k, t in zip(keys, texts):
            if k not in self.index and k not in seen:
                seen.add(k)
                missing.append((k, t))
        step = int(flush_every or self.FLUSH_EVERY)
        for start in range(0, len(missing), step):
            chunk = missing[start:start + step]
            new = encode([t for _k, t in chunk])
            new = np.asarray(new, dtype=np.float32)
            new /= np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-9)
            base = self.vecs.shape[0]
            self.vecs = np.concatenate([self.vecs, new.astype(np.float16)], 0)
            for j, (k, _t) in enumerate(chunk):
                self.index[k] = base + j
            self._flush()
            if len(missing) > step:
                print(f"  [cache] {min(start + step, len(missing))}/{len(missing)} encoded, "
                      f"{self.vecs.shape[0]} rows durable", flush=True)
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
    # ENCODE precision and STORAGE precision are different facts (Astra step-5 P2-15): the
    # teacher runs in float32 and only the cached vectors are rounded to float16. One `dtype`
    # field could not state both, and both are cache identity inputs.
    return {"teacher": ctx.reg["teacher"], "revision": ctx.reg["teacher_revision"],
            "instruction": spec["query_prefix"], "max_length": int(spec["max_length"]),
            "pooling": spec["pooling"], "post_dense": spec.get("post_dense"),
            "encode_dtype": "float32", "storage_dtype": "float16", "normalized": "l2",
            "tokenizer": "the teacher's own tokenizer (" + spec["tokenizer_id"] + ")"}


def stage_teacher(ctx):
    import torch
    import teacher as m7teacher
    pre = teacher_preprocessing(ctx)
    specs = ctx.cache_state["specs"]
    cache = TextVectorCache(TEACHER_CACHE, int(ctx.freeze["encoder_spec"]["dim"]),
                            manifest={"role": "query", **pre})

    def encode(texts):
        return m7teacher.encode(texts, prefix=pre["instruction"], max_length=pre["max_length"],
                                dtype=torch.float32, device=ctx.device, verbose=True)

    with Timer(ctx, "teacher_encode", "queries") as t:
        vecs, rep = cache.get([s.text for s in specs], encode)
        t.n = rep["encoded"]
    with Timer(ctx, "teacher_serialize", "queries") as t:
        np.save(admit_write(ctx.shared / "teacher_q.npy"), vecs)
        t.n = len(specs)
    ctx.cache_state["teacher_q"] = vecs
    return {"preprocessing": pre, "cache_dir": sm.rel(TEACHER_CACHE), **rep,
            "cache_manifest_sha256": cache.manifest_sha,
            "cache_manifest_stamped_on_empty_cache": cache.manifest_stamped,
            "queries": len(specs), "teacher_q_sha256": sha_array(vecs)}


# --------------------------------------------------------------------------- stage 5: bank

_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')


def _stream_json_strings(path, chunk=1 << 22):
    """Yield `(index, value)` for every string in a JSON array file, without loading the array.

    The pool's `ids-<store>.json` files are flat `["a", "b", ...]` arrays written by json.dump;
    the largest holds 5.23M ids. `read_text()` + `json.loads` cost about half a gigabyte of
    Python str objects to authenticate a digest and resolve a few thousand rows (Sol step-5
    P2-7). This scans block by block and decodes one token at a time.
    """
    i, tail = 0, ""
    with open(admit_read(path), encoding="utf-8") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            buf, pos = tail + block, 0
            for m in _JSON_STRING.finditer(buf):
                tok = m.group(0)
                yield i, (tok[1:-1] if "\\" not in tok else json.loads(tok))
                i += 1
                pos = m.end()
            tail = buf[pos:]
    for m in _JSON_STRING.finditer(tail):
        tok = m.group(0)
        yield i, (tok[1:-1] if "\\" not in tok else json.loads(tok))
        i += 1


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
        """-> {doc_id: global row}. Builds the id map for one store and drops it afterwards.

        The ordered id list is AUTHENTICATED against the pool's own `id_sha256` and its span
        length before a single row is resolved (Astra step-5 P2-13): reordering an
        `ids-<store>.json` beside an unchanged memmap would otherwise attach every document id
        to another document's vector, silently.
        """
        lo, hi = self.spans[store]
        want = set(doc_ids)
        # STREAMED: `read_text()` + `json.loads` materialized the whole 5.23M-id list as Python
        # strings (about 0.5 GiB for the largest store) purely to hash it and look a few
        # thousand ids up (Sol step-5 P2-7). One pass hashes the ids in the legacy convention
        # and keeps only the rows this call asked for.
        n, out = 0, {}
        from hashing import sha_stream_list                 # the legacy digest convention
        h = hashlib.sha256()
        h.update(b"[")
        for i, doc in _stream_json_strings(POOL_DIR / f"ids-{_denied(store)}.json"):
            if i:
                h.update(b", ")
            h.update(json.dumps(doc).encode())
            if doc in want:
                out[doc] = lo + i
            n = i + 1
        h.update(b"]")
        got_sha = h.hexdigest()
        assert sha_stream_list([]) == hashlib.sha256(b"[]").hexdigest()   # same convention
        if n != hi - lo:
            raise SystemExit(f"M17 REFUSED: ids-{store}.json holds {n} ids but the pool "
                             f"span for {store} is {hi - lo} rows.")
        want_sha = (self.meta.get("id_sha256") or {}).get(store)
        if want_sha and got_sha != want_sha:
            raise SystemExit(
                f"M17 REFUSED: ids-{store}.json hashes {got_sha[:12]} but the pool meta records "
                f"{want_sha[:12]}; these ids do not describe the stored vectors.")
        if not want_sha:
            raise SystemExit(f"M17 REFUSED: the pool meta records no id_sha256 for {store}; the "
                             "id-to-vector layout cannot be authenticated.")
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
    # bank and is encoded below with the frozen document tower. They are RESERVED before the
    # uniform stratified sample and counted INSIDE `candidate_bank_max_docs`, never on top of it
    # (`training.candidate_construction.bank_sampling_amendment_a4`). Only the documents the
    # protected screen admitted enter the bank; pre-clock the receipt is absent and the filter
    # is a no-op, which `protected_screen.unscreened_new_source_rows` records.
    receipt = _screen_receipt(ctx)
    k8s_texts = {f"k8s-docs-en:{p}": t for p, _title, t in sm.iter_k8s(sm.k8s_excluded_paths())
                 if receipt is None or p in receipt}
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

            def eligible(src=src, have=have):
                seen_groups = set()                     # 64-bit int keys (Sol step-5 P2-7)
                with gzip.open(admit_read(MANIFEST_DIR / "doc_groups" / f"{src}.tsv.gz"),
                               "rt") as f:
                    for line in f:
                        did, gid = line.rstrip("\n").split("\t")
                        key = _group_key(gid)
                        if did in have or key in seen_groups:
                            continue
                        seen_groups.add(key)
                        yield did
            # Uniform over the WHOLE eligible population: lowest seeded hash wins (P1-7). The
            # old probability-then-stop walk ended around the first quarter of the file.
            picked = _lowest_hash_pick(eligible(), want, ctx.seed, "bank", src)
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
    doc_manifest = {"role": "document", "teacher": ctx.reg["teacher"],
                    "revision": ctx.reg["teacher_revision"], "instruction": spec["doc_prefix"],
                    "max_length": int(spec["max_length"]), "pooling": spec["pooling"],
                    "post_dense": spec.get("post_dense"), "encode_dtype": "float32",
                    "storage_dtype": "float16", "normalized": "l2"}
    if missing:
        dcache = TextVectorCache(DOC_CACHE, reader.dim, manifest=doc_manifest)
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
    with Timer(ctx, "bank_serialize", "documents") as t:
        np.save(admit_write(ctx.shared / "bank.npy"), vecs)
        write_json(ctx.shared / "bank_ids.json", ordered)
        t.n = len(ordered)
    ctx.cache_state["bank"] = bank
    return {"n_docs": len(ordered), "from_pool": len(ordered) - len(missing),
            "encoded_with_document_tower": len(missing),
            "document_encode": enc_report,
            "document_cache_manifest_sha256": sha_json(doc_manifest),
            "positives": len(positives), "uniform": sum(len(v) for v in uniform_ids.values()),
            "k8s": len(k8s_texts), "k8s_screen_receipt_applied": receipt is not None,
            "per_source": _tally(sources),
            "encoder_spec_source": "m7/FREEZE.json:encoder_spec",
            "identity": bank.identity()}


# --------------------------------------------------------------------------- stage 6: v1

# The released zero-v1 encoder IS the int8 folded table; `variant="fp16"` loads a different
# set of rows from the same npz and the npz hash cannot tell the two apart (Astra step-5 P1-9).
V1_VARIANT = "int8"


def _load_v1_table(path, device, variant=V1_VARIANT):
    from table import load_table
    if variant != V1_VARIANT:
        raise SystemExit(
            f"M17 REFUSED: v1 mining must use the released {V1_VARIANT} table, not {variant!r}. "
            "The v1 candidate lists and the vocabulary residuals are defined against the "
            "encoder that shipped, dequantized exactly as the released loader does.")
    return load_table(admit_read(path), variant=variant, device=device)


def stage_v1(ctx):
    import torch
    from table import Preproc
    specs = ctx.cache_state["specs"]
    art_sha = sha_file(RELEASE_TABLE)
    if art_sha != ctx.freeze["table_sha256"]:
        raise SystemExit(f"M17 REFUSED: {RELEASE_TABLE} hashes {art_sha[:12]} but "
                         f"m7/FREEZE.json records {ctx.freeze['table_sha256'][:12]}.")
    pre = Preproc(**ctx.freeze["preproc"])
    if pre.fingerprint() != ctx.freeze["preproc_fingerprint"]:
        raise SystemExit("M17 REFUSED: the FREEZE preproc does not reproduce its own fingerprint.")
    with Timer(ctx, "v1_encode", "queries") as t:
        model = _load_v1_table(RELEASE_TABLE, ctx.device)
        folded_sha = sha_array(model.rows.detach().cpu().numpy())
        v1 = model.encode([s.text for s in specs], pre, batch=2048)
        t.n = len(specs)
        del model
        if ctx.device == "cuda":
            torch.cuda.empty_cache()
    v1 = np.asarray(v1, dtype=np.float32)
    v1 /= np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-9)
    # Round ONCE, before any use, and hash exactly the representation that is stored, so a
    # resumed build reads back the same numbers a fresh build used (Astra step-5 P2-16).
    v1_16 = v1.astype(np.float16)
    np.save(admit_write(ctx.out / "v1_q.npy"), v1_16)
    ctx.cache_state["v1_q"] = v1_16.astype(np.float32)
    return {"artifact": sm.rel(RELEASE_TABLE), "artifact_sha256": art_sha,
            "variant": V1_VARIANT, "folded_rows_sha256": folded_sha,
            "preprocessing": ctx.freeze["preproc"],
            "preproc_fingerprint": ctx.freeze["preproc_fingerprint"],
            "stored_dtype": "float16",
            "weights_folded": True, "queries": len(specs),
            "v1_q_sha256": sha_array(v1_16)}


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

    # TWO encodings per query (base and extended): the unit is encodings and the full-pool work
    # count is two per query, not one (Astra step-5 P3-20).
    with Timer(ctx, "student_tokenize", "encodings", per_query_work=2) as t:
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
                        ("artifact", "artifact_sha256", "variant", "folded_rows_sha256",
                         "preprocessing", "preproc_fingerprint", "weights_folded")},
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
    # The recipe identity cannot tell two internally valid caches apart when the scoring path
    # changed (m17/CODEMAP.md, blocked-BLAS note). The ARTIFACT digest can: it covers the
    # per-array hashes and the teacher/bank/v1 realizations the lists were scored from
    # (Astra step-5 P2-17).
    # Writing the candidate arrays is query-scaled work and was outside every named timer, so
    # the extrapolation did not include it (Sol step-5 P3-11).
    with Timer(ctx, "cache_save", "queries") as t:
        sidecar = m17cache.save(ctx.shared, arrays, sidecar, artifact_inputs={
            "teacher_q_sha256": ctx.stages["teacher"]["teacher_q_sha256"],
            "bank_vector_bytes_sha256": ctx.stages["bank"]["identity"]["vector_bytes_sha256"],
            "bank_doc_ids_sha256": ctx.stages["bank"]["identity"]["doc_ids_sha256"],
            "v1_q_sha256": ctx.stages["v1"]["v1_q_sha256"]})
        t.n = len(specs)
    ctx.cache_state["cache"] = (arrays, sidecar)

    with Timer(ctx, "prelock_diagnostics", "queries") as t:
        diag = {"alias_pre_lock_diagnostic": _alias_diagnostic(ctx, specs, teacher_q),
                "uninformative_list_stop": _warm_start_kl(arrays, bank, v1, ctx.reg)}
        t.n = len(specs)
    ctx.cache_state["diagnostics"] = diag
    return {"identity_sha256": sidecar["identity"]["sha256"],
            "artifact_sha256": sidecar["artifact_sha256"], "counts": sidecar["counts"],
            "entropy_diagnostic": sidecar["entropy_diagnostic"],
            "provenance_totals": sidecar["provenance_counts"]["total"],
            "teacher_v1_list_overlap": sidecar["provenance_counts"]["teacher_v1_list_overlap"],
            **diag}


def _alias_diagnostic(ctx, specs, teacher_q, floor=0.7):
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
    # The COMPLETE flagged list is persisted beside the build; the summary keeps a sample, so
    # more than twenty flagged pairs stay individually re-verifiable (Astra step-5 P3-21).
    sidecar = ctx.out / "alias_flagged_pairs.json"
    write_json(sidecar, {"_schema": "m17-alias-flagged-v1", "floor": floor,
                         "pairs_checked": len(cos), "flagged": flagged}, indent=None)
    return {"pairs": len(cos), "floor": floor,
            "mean": float(np.mean(cos)) if cos else None,
            "quantiles": quantiles(cos) if cos else {},
            "flagged_below_floor": len(flagged),
            "flagged_share": round(len(flagged) / len(cos), 4) if cos else None,
            "flagged_sample": flagged[:20],
            "flagged_file": {"path": sm.rel(sidecar), "sha256": sha_file(sidecar)},
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

def _protected_block(ctx, specs):
    """The protected-screen receipt as the training driver reads it.

    Pre-clock the screen is OFF and the bank carries new-source rows nothing has screened; the
    count is stated here rather than left to be inferred (Astra step-5 P1-2/P1-3, ruling A4).
    """
    rec = dict(ctx.stages["protected"])
    if rec.get("state") == "complete":
        rec["unscreened_new_source_rows"] = {"pool": 0, "bank": 0}
        return rec
    bank = ctx.cache_state.get("bank")
    rec["unscreened_new_source_rows"] = {
        "pool": len(_new_source_rows(specs)),
        "bank": sum(1 for d in (bank.doc_ids if bank else []) if str(d).startswith(NEW_SOURCE))}
    return rec


def stage_manifests(ctx):
    specs = ctx.cache_state["specs"]
    out = {}
    protected = _protected_block(ctx, specs)
    timer = Timer(ctx, "manifests", "queries")
    timer.__enter__()
    timer.n = len(specs)
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
                "cache_artifact": ctx.stages["cache"]["artifact_sha256"],
                "doc_domain_join": ctx.stages["domain"]["doc_domain_tsv"]["sha256"],
                "query_pool": sha_file(ctx.out / "pool.jsonl"),
                "k8s_jsonl": sha_file(K8S_JSONL),
                "exclusion_file": sha_file(EXCLUSIONS),
                "alias_pairs_jsonl": sha_file(ALIAS_PAIRS),
                "support_manifest": sha_file(RESULTS / "m17_support_manifest.json"),
                "freeze": sha_file(REPO / "m7" / "FREEZE.json"),
            },
            "protected_screen": protected,
        }
        if form == "ext":
            manifest["new_rows"] = "new_rows.npy"
            manifest["hashes"]["new_rows"] = ctx.stages["vocab"]["new_rows_sha256"]
        write_json(d / "prepared.json", manifest)
        out[form] = sm.rel(d / "prepared.json")
    timer.__exit__()
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
    if name in ("domain", "protected"):
        # `protected` re-reads the SCREENED pool: a restart must not resume with the
        # unscreened rows just because the marker was already written (P1-4).
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


def recipe_identity(reg, fz):
    """A digest of the registry/FREEZE inputs the prepared arrays are defined against.

    Sol step-5 P1-4: none of these changed a stage marker's identity, so a prepared directory
    built under one candidate recipe could be reused, and trained on, after the registry moved.
    The training driver checks the same inputs against the CACHE it loads; this is the builder's
    half, so a re-run under a changed recipe rebuilds instead of resuming.
    """
    tr = reg["training"]
    spec = fz["encoder_spec"]
    parts = {
        "candidate_k": tr["candidate_k"],
        "candidate_mix_labeled": tr["candidate_mix_labeled"],
        "candidate_mix_query_only": tr["candidate_mix_query_only"],
        "candidate_construction": tr["candidate_construction"],
        "rng_recipe_version": m17cache.RNG_RECIPE_VERSION,
        "rng_recipe": m17cache.RNG_RECIPE,
        "temperature": tr["temperature"],
        "labeled_positive_budget_share": tr["labeled_positive_budget_share"],
        "candidate_bank_max_docs": tr["candidate_bank_max_docs"],
        "teacher": reg["teacher"], "teacher_revision": reg["teacher_revision"],
        "dose": reg["data"]["measured_dose_after_pre_lock_rule"],
        "training_query_cap": reg["data"]["training_query_cap"],
        "cap_fill_priority": reg["data"]["cap_fill_priority"],
        "documentless_sources_vote": reg["data"]["documentless_sources_vote"],
        "new_source_share_max": reg["data"]["new_source_share_max_of_training_queries"],
        "teacher_preprocessing": {k: spec.get(k) for k in
                                  ("query_prefix", "doc_prefix", "max_length", "pooling",
                                   "post_dense", "tokenizer_id", "dim", "revision")},
        "preproc_fingerprint": fz["preproc_fingerprint"],
        "table_sha256": fz["table_sha256"],
        "training_checkpoint_sha256": fz["training_checkpoint_sha256"],
    }
    return {"parts": parts, "sha256": sha_json(parts)}


def stage_identity(ctx):
    """The INPUT identity every stage marker is stamped with (Astra step-5 P2-11).

    Reusing an output directory with a changed seed, size, bank cap, exclusion file, protected
    flag or source manifest silently kept markers built from other inputs. A marker whose
    identity does not match is refused unless `--force`, and forcing a stage invalidates the
    stages that depend on it (they follow it in `STAGES`).
    """
    return {"seed": ctx.seed, "size": ctx.size, "bank_docs": ctx.bank_docs,
            "protected_screen": bool(ctx.protected_screen),
            # The registered RECIPE the stages are built under (Sol step-5 P1-4): changing a
            # candidate mix, K, the temperature, the RNG rule, the cap-fill priority or the
            # frozen teacher preprocessing must not leave the cached markers reusable.
            "recipe_sha256": recipe_identity(ctx.reg, ctx.freeze)["sha256"],
            "exclusions_sha256": sha_file(EXCLUSIONS) if EXCLUSIONS.exists() else None,
            "alias_pairs_sha256": sha_file(ALIAS_PAIRS) if ALIAS_PAIRS.exists() else None,
            "k8s_jsonl_sha256": sha_file(K8S_JSONL) if K8S_JSONL.exists() else None,
            "kept_sha256": sha_file(REPO / "work" / "decontam" / "kept.json"),
            "support_manifest_sha256": sha_file(RESULTS / "m17_support_manifest.json"),
            "query_families_sha256": sha_file(QUERY_FAMILIES) if QUERY_FAMILIES.exists() else None}


def _check_stage_identity(prev, ident, name, force):
    """-> True when the cached marker may be reused. Refuses a changed input without --force."""
    got = (prev or {}).get("_identity")
    if got == ident:
        return True
    if force:
        return False
    differ = sorted(k for k in ident if (got or {}).get(k) != ident[k]) if got else ["<absent>"]
    raise SystemExit(
        f"M17 REFUSED: the cached `{name}` stage was built under different inputs ({differ}). "
        "A prepared directory is one experiment; re-run with --force to rebuild this stage and "
        "its dependents, or build into a new --out.")


def _invalidate_dependents(out, name):
    """Persistently invalidate every stage after `name` by renaming its marker to `*.stale`.

    Sol step-5 P2-5: invalidation lived only in the in-memory `invalidated` set, so
    `--force --stages pool` rebuilt the pool and left every downstream marker on disk with a
    matching global identity — the next ordinary invocation reused stages built from the
    replaced pool. A dependent stage rewrites its own artifacts when it re-runs; what had to
    become persistent is the marker that says it need not.
    """
    moved = []
    for later in STAGES[STAGES.index(name) + 1:]:
        m = out / "stages" / f"{later}.json"
        if m.exists():
            stale = m.with_name(f"{later}.json.stale")
            if stale.exists():
                stale.unlink()
            m.rename(stale)
            moved.append(later)
    return moved


def build(args, log=print):
    ctx = Ctx(args)
    want = args.stages.split(",") if args.stages else list(STAGES)
    ident = stage_identity(ctx)
    t0 = time.time()
    ran = []
    invalidated = set()
    stale = []
    for name in STAGES:
        marker = ctx.out / "stages" / f"{name}.json"
        prev = json.loads(admit_read(marker).read_text()) if marker.exists() else None
        # `--force` applies to the REQUESTED stages only: an unrequested ancestor whose inputs
        # still match is rehydrated rather than treated as non-reusable (Sol step-5 P2-5).
        reusable = marker.exists() and name not in invalidated and \
            _check_stage_identity(prev, ident, name, args.force and name in want)
        if name not in want:
            if reusable:
                ctx.stages[name] = prev
                _rehydrate(ctx, name)
            continue
        if reusable and not args.force:
            log(f"[stage {name}] cached, skipped")
            ctx.stages[name] = prev
            _rehydrate(ctx, name)
            continue
        log(f"[stage {name}] running")
        rec = STAGE_FN[name](ctx)
        ctx.stages[name] = rec
        write_json(marker, {**rec, "_identity": ident})
        ran.append(name)
        # A rebuilt stage invalidates everything downstream of it, in the fixed stage order —
        # in memory AND on disk, so a later invocation cannot reuse a dependent marker.
        invalidated.update(STAGES[STAGES.index(name) + 1:])
        stale.extend(_invalidate_dependents(ctx.out, name))
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
    complete = ran == list(STAGES)
    record = {"timings_carried_from_a_previous_build": sorted(carried),
              "_schema": "m17-prepared-build-v1", "out": sm.rel(ctx.out), "size": ctx.size,
              "seed": ctx.seed, "bank_docs": ctx.bank_docs, "device": ctx.device,
              "stage_identity": ident,
              "stages_run_this_invocation": ran,
              "dependent_markers_invalidated_on_disk": sorted(set(stale)),
              "complete_build": complete,
              # A partial rebuild's wall clock is not the cost of preparing this size.
              "wall_clock_seconds": round(time.time() - t0, 3),
              "wall_clock_kind": "full build" if complete else "partial rebuild",
              "rss_high_water_gib": _rss_gib(), "gpu_peak_gib": _gpu_peak_gib(),
              "memory_gib": _mem_gib(),
              "timings": ctx.timings, "stages": ctx.stages, "notes": ctx.notes,
              "source_sha256": sha_text(Path(__file__).read_text())}
    write_json(ctx.out / "build_record.json", record)
    return record


OTHER_RESIDENTS_GIB = "8-9"        # measured free/used on the 25 GiB box outside this build


def _peak_memory(rec):
    """The highest RssAnon / RssFile / VmHWM any stage of one build recorded."""
    seen = [t.get("memory_gib") or {} for t in rec.get("timings", {}).values()]
    seen.append(rec.get("memory_gib") or {})
    out = {}
    for field in ("RssAnon", "RssFile", "VmHWM"):
        vals = [m[field] for m in seen if m.get(field) is not None]
        out[field] = max(vals) if vals else None
    return out


def _memory_projection(recs, target):
    """ANONYMOUS high-water measured at every built size, and projected at the full pool.

    Sol step-5 P2-7: the reported 12.4 GiB was an RSS high-water dominated by the frozen
    document memmap's file-backed pages. What decides whether the full build fits is the
    ANONYMOUS high-water plus the other residents of the box, so all three numbers are measured
    at three sizes and the anonymous slope per query is fitted, never argued.
    """
    series = []
    for r in recs:
        q = sum(r["stages"]["domain"]["queries_by_bucket"].values())
        series.append({"out": r["out"], "queries": q, "rss_high_water_gib":
                       r.get("rss_high_water_gib"), **_peak_memory(r)})
    usable = [s for s in series if s.get("RssAnon") is not None]
    projection = None
    if len(usable) >= 2:
        a, b = usable[0], usable[-1]
        per_query = ((b["RssAnon"] - a["RssAnon"]) / (b["queries"] - a["queries"])
                     if b["queries"] != a["queries"] else 0.0)
        fixed = a["RssAnon"] - per_query * a["queries"]
        projection = {
            "fitted_from": [a["queries"], b["queries"]],
            "fixed_gib": round(fixed, 3),
            "per_query_gib": round(per_query, 9),
            "projected_anonymous_high_water_gib_at_full_pool":
                round(fixed + per_query * target, 2),
            "other_residents_gib": OTHER_RESIDENTS_GIB,
            "host_ram_gib": 25}
    return {
        "measured": series,
        "projection": projection,
        "note": "RssAnon is the process's own pages; RssFile is mostly the frozen document "
                "memmap and the OS reclaims it under pressure; VmHWM is the high-water of both "
                "and is the number the earlier record reported. Compare the projected anonymous "
                f"high-water plus the {OTHER_RESIDENTS_GIB} GiB of other residents against the "
                "25 GiB host before scheduling the full-pool build."}


def timing_report(dirs, full_total=None):
    """Two measured sizes -> per-stage fixed cost, per-row cost and a linear extrapolation.

    A fixed load cost divided by row count is not a per-row cost, so each stage is fitted as
    `seconds = fixed + per_row * rows` from the two observations and BOTH halves are reported.
    A stage whose row count does not change between the two sizes is reported as wholly fixed
    and is never extrapolated per row.
    """
    recs = [json.loads(admit_read(Path(d) / "build_record.json").read_text()) for d in dirs]
    recs.sort(key=lambda r: r["stages"]["pool"]["plan"]["general"])
    # The fit uses the SMALLEST and LARGEST measured builds; any middle size is reported beside
    # them (and carries the memory series, Sol step-5 P2-7).
    small, large = recs[0], recs[-1]
    target = full_total or large["stages"]["pool"]["full_pool_total"]
    stages, total_fixed, total_per_row = {}, 0.0, 0.0
    doc_growth = []
    q_small = sum(small["stages"]["domain"]["queries_by_bucket"].values())
    q_large = sum(large["stages"]["domain"]["queries_by_bucket"].values())
    for name in sorted(set(small["timings"]) | set(large["timings"])):
        a, b = small["timings"].get(name), large["timings"].get(name)
        if not a or not b:
            stages[name] = {"one_size_only": a or b}
            continue
        unit = a["unit"]
        na, nb = a.get(unit, 0), b.get(unit, 0)
        rec = {"unit": unit,
               "small": {"n": na, "seconds": a["seconds"], "rate": a.get(f"{unit}_per_second"),
                         "rss_high_water_gib": a.get("rss_high_water_gib"),
                         "gpu_peak_gib": a.get("gpu_peak_gib")},
               "large": {"n": nb, "seconds": b["seconds"], "rate": b.get(f"{unit}_per_second"),
                         "rss_high_water_gib": b.get("rss_high_water_gib"),
                         "gpu_peak_gib": b.get("gpu_peak_gib")}}
        work = b.get("per_query_work") or a.get("per_query_work")
        # Only a stage whose row count actually GREW between the two measured sizes can be
        # fitted from them. The bank stages are held at the registered cap: `bank_pool_lookup`
        # differs by a single document and `bank_sample` shrinks as the labeled positives grow,
        # so a line through those two points has an arbitrary slope — and a negative one turns
        # into an absurd intercept (a 47.8 s cold lookup against a 10.4 s warm one projected a
        # 9.7-million-second fixed cost once a third size was added).
        grew = nb >= 1.25 * max(1, na)
        rec["scales_with_the_query_pool"] = bool(grew)
        if grew:
            per_row = (b["seconds"] - a["seconds"]) / (nb - na)
            fixed = max(0.0, a["seconds"] - per_row * na)
            rec.update(per_row_seconds=round(per_row, 6), fixed_seconds=round(fixed, 3),
                       per_query_work=work,
                       scales_with="queries" if work else unit)
            total_fixed += fixed
            if work:
                # `student_tokenize` does TWO encodings per query: the full-pool work count is
                # the query count times this stage's own work factor (P3-20).
                total_per_row += max(0.0, per_row) * float(work)
            else:
                # documents/rows stages do not scale one-for-one with the query pool; their
                # large-size cost is carried forward as a fixed cost of the full build, and the
                # growth that IS query-driven is reported as a sensitivity estimate below.
                total_fixed += max(0.0, b["seconds"] - fixed)
                doc_growth.append((name, per_row, fixed, na, nb))
        else:
            rec.update(fixed_seconds=b["seconds"], per_row_seconds=0.0,
                       note=f"row count did not grow materially between the measured sizes "
                            f"({na} -> {nb}): carried as a fixed cost of the full build, not "
                            "fitted as a per-row one")
            total_fixed += b["seconds"]
        stages[name] = rec
    est = total_fixed + total_per_row * target
    # `domain_store_pass` and the other document-unit stages are carried as fixed costs, but
    # their document counts DO grow with the query pool (more selected positives to classify).
    # The upper bound projects each such stage's document count linearly in queries and is
    # reported beside the estimate rather than folded into it (P3-20).
    growth = []
    for name, per_row, fixed, na, nb in doc_growth:
        projected = nb * (target / max(1, q_large))
        growth.append({"stage": name, "documents_at_large": nb,
                       "projected_documents_at_full_pool": int(projected),
                       "seconds_if_documents_scale_with_queries":
                           round(fixed + max(0.0, per_row) * projected, 1)})
    extra = sum(g["seconds_if_documents_scale_with_queries"] for g in growth) - \
        sum(r["large"]["seconds"] for n, r in stages.items()
            if isinstance(r, dict) and any(g["stage"] == n for g in growth))
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
                   "wall_clock_kind": r.get("wall_clock_kind", "unknown"),
                   "complete_build": r.get("complete_build"),
                   "stages_run_this_invocation": r.get("stages_run_this_invocation"),
                   "rss_high_water_gib": r["rss_high_water_gib"],
                   "peak_memory_gib": _peak_memory(r),
                   "gpu_peak_gib": r["gpu_peak_gib"],
                   "teacher_cache": {k: r["stages"]["teacher"][k]
                                     for k in ("encoded", "hit_rate", "cache_rows")},
                   "bank": {k: r["stages"]["bank"][k] for k in
                            ("n_docs", "from_pool", "encoded_with_document_tower")}}
                  for r in recs],
        "stages": stages,
        "separately_observed": {
            "cold_document_tower_encode": {
                "documents": 1648, "seconds": 45.701, "documents_per_second": 36.1,
                "what": "the admitted Kubernetes slice encoded with the frozen document tower "
                        "(m7/FREEZE.json encoder_spec, fp16), the one-off cost of filling "
                        "work/m17/prepared/doc_cache",
                "provenance": "measured on the first build of this session; both timed builds "
                              "above reused that cache and therefore report 0 encodes. Recorded "
                              "here rather than re-measured, because re-encoding would change "
                              "the bank vectors the candidate caches were built from."},
            "cold_teacher_query_encode": {
                "measured_rate_texts_per_second": 248,
                "provenance": "the first s2000 build of this session encoded its whole pool "
                              "cold: 1,949 texts in 7.868 s (commit 6fc3b6a's timing result). "
                              "The 50,000-query build encoded 39,477 texts cold in 112.7 s "
                              "(350/s), which is the `teacher_encode` slope FITTED ABOVE.",
                "full_pool_cold_seconds_if_nothing_is_cached": round(target / 248.0, 1),
                "full_pool_cold_hours_if_nothing_is_cached": round(target / 248.0 / 3600, 2),
                "what": "the headline extrapolation now INCLUDES a cold teacher encode, because "
                        "the largest measured build encoded most of its pool cold and that "
                        "per-query slope is in the fit. This slower 248/s figure is the earlier "
                        "cold measurement, kept as the conservative alternative."}},
        "extrapolation": {
            "full_pool_queries": target,
            "fixed_seconds": round(total_fixed, 1),
            "per_query_seconds": round(total_per_row, 6),
            "estimated_seconds": round(est, 1),
            "estimated_hours": round(est / 3600, 2),
            "document_stage_sensitivity_estimate": {
                "stages": growth,
                "additional_seconds_if_document_counts_scale_with_queries": round(max(0.0, extra), 1),
                "estimated_hours_with_that_sensitivity":
                    round((est + max(0.0, extra)) / 3600, 2),
                "note": "document-unit stages are carried as fixed costs of the full build. The "
                        "ones whose document count actually moved with the pool (notably "
                        "domain_store_pass, which also classifies every selected positive) are "
                        "projected here, linearly in queries. This is a SENSITIVITY ESTIMATE, "
                        "not an upper bound: the per-document slopes are noisy and were clamped "
                        "at zero where a larger build measured faster, so a genuinely slower "
                        "per-document rate at the full pool is not bounded by it (Sol step-5 "
                        "P3-11). The bank stages are excluded: the bank is held at the "
                        "registered cap, so their row count does not grow with the pool"},
            "anonymous_memory": _memory_projection(recs, target),
            "query_counts_measured": {"small": q_small, "large": q_large},
            "fixed_costs_subtracted": sorted(
                n for n, r in stages.items()
                if isinstance(r, dict) and r.get("unit") in ("documents", "rows")),
            "caveats": [
                "linear in queries; the candidate cache is O(queries x bank) and the bank is "
                "held at the registered cap in both measurements, so its per-query cost is the "
                "real one",
                "the largest measured build encoded most of its queries cold (39,477 of "
                "50,000), so the fitted `teacher_encode` slope — and therefore the headline — "
                "includes a cold teacher encode; the document cache was warm and its encode is "
                "a separately observed fixed cost",
                "the fit uses the SMALLEST and LARGEST builds; a stage whose row count did not "
                "grow by at least 25% between them is carried as a fixed cost rather than "
                "fitted, because a line through two nearly identical points has an arbitrary "
                "slope",
                "single process, no parallelism; the cache stage is CPU-bound numpy",
                "student_tokenize is extrapolated at its own work factor (two encodings per "
                "query), not one",
                "each size's wall clock states whether it was a full build or a partial "
                "rebuild; only a full build is the cost of preparing that size"]},
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
