"""The one M17 candidate cache: schema, deterministic construction and its identity hash.

Exactly the registry's `training.candidate_construction` and `candidate_mix_*`:

  * a labeled query gets one known positive (the first in the admitted dataset's own qrels
    order that is inside the bank), then 31 teacher-ranked, 16 v1-ranked, 16 uniform documents;
    a query-only example gets 32/16/16 and no positive slot;
  * quotas count NEWLY ADMITTED UNIQUE documents, walked in rank order, skipping the positive
    and duplicates; an exhausted nominal depth is continued down the same source's ranked list
    (backfill) and uniform draws never fill a teacher/v1 quota;
  * ties break by ascending bank document id;
  * uniform draws use `numpy.default_rng` seeded with SHA-256(cache_seed || raw query text hash)
    truncated to 64 bits, so a query's random slots are reproducible from the cache identity.

`has_label` and a nullable `positive_id` are stored explicitly. **A teacher hit is never a
relevance label.** Alias views are ordinary rows of this same schema carrying their verified
pair id, and each view keeps its own teacher target and its own nullable positive.

The cache identity hash covers the registry's `training.cache_identity` list: query text,
source/split manifest, alias manifest and family split, teacher revision and query
preprocessing, the bank's document ids / vector bytes / dtype, the v1 artifact and its
preprocessing, and the candidate mixes / K / seed.

Teacher-distribution entropy and p_max are recorded at build time in the
`results/m8_b2_entropy.json` block format, regardless of the listwise stop rule.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common import (admit_read, admit_write, quantiles, sha_array, sha_json, sha_texts,
                    write_json)

# candidate provenance codes, stored per (query, slot)
SRC_POSITIVE, SRC_TEACHER, SRC_V1, SRC_UNIFORM = 0, 1, 2, 3
SRC_NAMES = {SRC_POSITIVE: "positive", SRC_TEACHER: "teacher", SRC_V1: "v1",
             SRC_UNIFORM: "uniform"}


@dataclass
class Bank:
    """The fixed admitted document-vector bank. Vectors are L2-normalized, stored dtype recorded."""
    doc_ids: list
    vectors: np.ndarray                       # (N, dim), fp16 or fp32, L2-normalized
    sources: list = field(default_factory=list)
    seed: int = 0

    def __post_init__(self):
        self.doc_ids = list(self.doc_ids)
        if len(self.doc_ids) != self.vectors.shape[0]:
            raise ValueError("bank doc_ids and vectors disagree on length")
        if not self.sources:
            self.sources = ["unknown"] * len(self.doc_ids)
        self.index = {d: i for i, d in enumerate(self.doc_ids)}
        if len(self.index) != len(self.doc_ids):
            raise ValueError("bank document ids are not unique")
        # rank of each document id under ascending-id order: the registered tie-break
        self.id_rank = np.empty(len(self.doc_ids), dtype=np.int64)
        self.id_rank[np.argsort(np.asarray(self.doc_ids, dtype=object), kind="stable")] = \
            np.arange(len(self.doc_ids))

    @property
    def dim(self):
        return int(self.vectors.shape[1])

    def identity(self):
        return {"n_docs": len(self.doc_ids), "dim": self.dim,
                "dtype": str(self.vectors.dtype),
                "doc_ids_sha256": sha_texts(self.doc_ids),
                "vector_bytes_sha256": sha_array(self.vectors),
                "sample_seed": self.seed}


@dataclass
class QuerySpec:
    """One training example. Both alias views are separate QuerySpecs sharing `alias_pair_id`."""
    qid: str
    text: str
    source: str = "unknown"
    domain: str = "general"
    bucket: str = "general"                   # general | coverage | alias
    family: str = ""                          # same-intent / near-duplicate family id
    positive_ids: tuple = ()                  # in the admitted dataset's own qrels order
    alias_pair_id: str = ""
    alias_view: str = ""                      # "a" | "b" | ""

    @property
    def has_label(self):
        return bool(self.positive_ids)


RNG_RECIPE_VERSION = "m17-uniform-rng-v1"
RNG_RECIPE = ("numpy default_rng seeded with the first 8 bytes, big-endian, of "
              "SHA-256(decimal cache_seed as UTF-8 || 0x00 || "
              "SHA-256(raw query text as UTF-8) hex digest as UTF-8)")


def _query_rng(cache_seed: int, text: str):
    """`training.candidate_construction.rng`, serialized literally (see `RNG_RECIPE`).

    The registry says the seed is hashed with the raw query text's HASH, not with the raw
    text itself: the text hash is the identity the cache records, so the draw is reproducible
    from the recorded identity alone.
    """
    text_sha = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
    h = hashlib.sha256(str(int(cache_seed)).encode("utf-8") + b"\x00"
                       + text_sha.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _ranked(scores, id_rank):
    """Bank indices in the FULL registered order: descending score, ascending bank id.

    One `lexsort` per query over the whole bank. A partial `argpartition` would decide the
    cutoff on score alone, so a tie spanning the boundary could drop the smaller bank id the
    tie-break requires; at the registered 262,144-row bank a full sort is affordable.
    """
    for i in np.lexsort((id_rank, -np.asarray(scores))):
        yield int(i)


def _take(gen, quota, chosen, out_idx, out_src, code, counters):
    """Walk a ranked source until `quota` NEW unique documents are admitted."""
    taken = 0
    depth = 0
    while taken < quota:
        try:
            i = next(gen)
        except StopIteration:
            counters["shortfall_" + SRC_NAMES[code]] += quota - taken
            break
        depth += 1
        if i in chosen:
            continue
        chosen.add(i)
        out_idx.append(i)
        out_src.append(code)
        taken += 1
    if depth > quota:
        counters["backfill_" + SRC_NAMES[code]] += depth - quota
    return taken


def build(queries, bank: Bank, teacher_q, v1_q, reg, cache_seed=0, manifests=None,
          progress=None):
    """Build the candidate cache. Returns (arrays dict, sidecar dict).

    `teacher_q` / `v1_q` are (n_queries, dim) L2-normalized query vectors aligned with
    `queries`. Teacher scores are dot(q_teacher, d) against the frozen document vectors; no
    document is re-encoded and no teacher hit becomes a label.
    """
    tr = reg["training"]
    K = int(tr["candidate_k"])
    mix_l, mix_q = tr["candidate_mix_labeled"], tr["candidate_mix_query_only"]
    for m in (mix_l, mix_q):
        if sum(m.values()) != K:
            raise ValueError(f"candidate mix {m} does not sum to candidate_k={K}")
    temp = float(tr["temperature"])
    bank_f32 = bank.vectors.astype(np.float32, copy=False)

    n = len(queries)
    cand = np.full((n, K), -1, dtype=np.int32)
    tscore = np.zeros((n, K), dtype=np.float32)
    csrc = np.full((n, K), SRC_UNIFORM, dtype=np.uint8)
    has_label = np.zeros(n, dtype=bool)
    positive_id = np.full(n, -1, dtype=np.int32)
    counters = {f"{k}_{s}": 0 for k in ("count", "backfill", "shortfall")
                for s in SRC_NAMES.values()}
    per_source = {}

    for qi, q in enumerate(queries):
        ts = bank_f32 @ np.asarray(teacher_q[qi], dtype=np.float32)
        vs = bank_f32 @ np.asarray(v1_q[qi], dtype=np.float32)
        chosen, idx, src = set(), [], []

        outside = [d for d in q.positive_ids if d not in bank.index]
        if outside:
            # registry positive_bank_policy: the eligible labeled subset is chosen BEFORE the
            # bank is built, so a positive outside it is a selection error, never a query-only
            # example wearing has_label=False.
            raise ValueError(
                f"M17 cache REFUSED: labeled query {q.qid!r} has positives outside the bank "
                f"({outside}). All known positives of the selected labeled subset must fit "
                "inside the bank cap; choose the eligible subset before building the cache.")
        pos = bank.index[q.positive_ids[0]] if q.positive_ids else None
        labeled = pos is not None
        has_label[qi] = labeled
        mix = mix_l if labeled else mix_q
        if labeled:
            if mix["known_positive"] != 1:
                raise ValueError("labeled mix must carry exactly one positive slot")
            positive_id[qi] = pos
            chosen.add(pos)
            idx.append(pos)
            src.append(SRC_POSITIVE)

        t_gen = _ranked(ts, bank.id_rank)
        v_gen = _ranked(vs, bank.id_rank)
        _take(t_gen, mix["teacher_top"], chosen, idx, src, SRC_TEACHER, counters)
        _take(v_gen, mix["zero_v1_top"], chosen, idx, src, SRC_V1, counters)

        rng = _query_rng(cache_seed, q.text)
        want = mix["uniform"]
        guard = 0
        while want > 0 and len(chosen) < len(bank.doc_ids):
            for i in rng.integers(0, len(bank.doc_ids), size=max(want * 4, 8)):
                i = int(i)
                if i in chosen:
                    continue
                chosen.add(i)
                idx.append(i)
                src.append(SRC_UNIFORM)
                want -= 1
                if want == 0:
                    break
            guard += 1
            if guard > 64:
                counters["shortfall_uniform"] += want
                break

        # A bank smaller than K leaves trailing -1 slots; they are masked out of the teacher
        # distribution and counted in the shortfall counters rather than padded with a document.
        for slot, (i, c) in enumerate(zip(idx[:K], src[:K])):
            cand[qi, slot] = i
            csrc[qi, slot] = c
            tscore[qi, slot] = ts[i]
            counters["count_" + SRC_NAMES[c]] += 1
            s = bank.sources[i]
            per_source.setdefault(s, dict.fromkeys(SRC_NAMES.values(), 0))
            per_source[s][SRC_NAMES[c]] += 1
        if progress and (qi + 1) % progress == 0:
            print(f"  [m17.cache] {qi + 1}/{n}", flush=True)

    # v1/teacher list overlap is reported as "how many of the v1 top-ranked documents the
    # teacher had already admitted", which the walk skipped as duplicates.
    dup_v1 = _overlap_report(queries, bank_f32, teacher_q, v1_q, mix_l, mix_q, bank.id_rank)

    ent = entropy_diagnostic(tscore, cand, temp, K)
    sidecar = {
        "_schema": "m17-candidate-cache-v1",
        "n_queries": n, "candidate_k": K, "temperature": temp,
        "identity": identity(queries, bank, reg, cache_seed, manifests),
        "counts": {"labeled": int(has_label.sum()), "query_only": int((~has_label).sum()),
                   "alias_views": sum(1 for q in queries if q.alias_pair_id),
                   "alias_pairs": len({q.alias_pair_id for q in queries if q.alias_pair_id}),
                   "buckets": _tally(q.bucket for q in queries)},
        "provenance_counts": {"total": counters, "per_source": per_source,
                              "teacher_v1_list_overlap": dup_v1},
        "entropy_diagnostic": ent,
    }
    arrays = {"candidate_ids": cand, "teacher_scores": tscore, "candidate_source": csrc,
              "has_label": has_label, "positive_id": positive_id,
              "qids": np.asarray([q.qid for q in queries], dtype=object),
              "alias_pair_ids": np.asarray([q.alias_pair_id for q in queries], dtype=object),
              "alias_views": np.asarray([q.alias_view for q in queries], dtype=object),
              "families": np.asarray([q.family for q in queries], dtype=object),
              "buckets": np.asarray([q.bucket for q in queries], dtype=object)}
    return arrays, sidecar


def _tally(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def _overlap_report(queries, bank_f32, teacher_q, v1_q, mix_l, mix_q, id_rank, depth=None,
                    sample=256):
    """Share of the v1 nominal-depth list already present in the teacher's nominal-depth list.

    Reported because a large overlap means the v1 quota is mostly buying backfill rather than
    a second opinion; it is a diagnostic, not a rule. Measured on the first `sample` queries so
    a full cache build does not pay for a second ranking pass over everything.
    """
    d_t = depth or max(mix_l["teacher_top"], mix_q["teacher_top"])
    d_v = depth or max(mix_l["zero_v1_top"], mix_q["zero_v1_top"])
    sizes = []
    for qi in range(min(len(queries), sample)):
        ts = bank_f32 @ np.asarray(teacher_q[qi], dtype=np.float32)
        vs = bank_f32 @ np.asarray(v1_q[qi], dtype=np.float32)
        a = {i for _, i in zip(range(d_t), _ranked(ts, id_rank))}
        b = {i for _, i in zip(range(d_v), _ranked(vs, id_rank))}
        sizes.append(len(a & b) / max(1, min(d_t, d_v)))
    return {"mean_fraction_of_v1_depth_already_in_teacher_depth": float(np.mean(sizes)) if sizes
            else 0.0, "teacher_depth": d_t, "v1_depth": d_v, "queries_sampled": len(sizes)}


def entropy_diagnostic(teacher_scores, cand_ids, temp, K):
    """`results/m8_b2_entropy.json`'s block, for this cache's own candidate lists."""
    s = np.asarray(teacher_scores, dtype=np.float64) / temp
    s = np.where(np.asarray(cand_ids) < 0, -np.inf, s)
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.where(p > 0, np.log(p), 0.0)
    ent = -(p * lp).sum(axis=1)
    pmax = p.max(axis=1)
    ceiling = float(np.log(K))
    return {
        "setting": {"n_queries": int(s.shape[0]), "kl_k": int(K), "temp": float(temp),
                    "entropy_ceiling_nats": ceiling},
        "entropy_nats": {"mean": float(ent.mean()), **quantiles(ent)},
        "entropy_as_fraction_of_ceiling": float(ent.mean() / ceiling) if ceiling else None,
        "p_max_of_teacher_distribution": {"mean": float(pmax.mean()),
                                          "p50": float(np.percentile(pmax, 50)),
                                          "p95": float(np.percentile(pmax, 95))},
        "share_of_queries_below_1e-4_nats": float((ent < 1e-4).mean()),
        "share_of_queries_below_1e-2_nats": float((ent < 1e-2).mean()),
    }


def _query_records_sha(queries):
    """Ordered per-query metadata, so swapping two queries' source/bucket/positives shows up.

    Aggregate sets and counts are preserved by such a swap; this hash is not.
    """
    return sha_texts(["\x1f".join([q.qid, sha_texts([q.text]), q.source, q.domain, q.bucket,
                                   q.family, q.alias_pair_id, q.alias_view,
                                   "\x1e".join(map(str, q.positive_ids))])
                      for q in queries])


def identity(queries, bank: Bank, reg, cache_seed, manifests=None):
    """The registry's `training.cache_identity`, hashed. Every field is named, not folded in.

    The v1 artifact identity and the teacher's query preprocessing MUST be supplied: an empty
    v1 identity would let a different `v1_q` (and so different candidates) keep the same
    cache hash.
    """
    tr = reg["training"]
    man = manifests or {}
    v1 = man.get("v1_artifact")
    pre = man.get("teacher_query_preprocessing")
    if not v1:
        raise ValueError("M17 cache identity REFUSED: manifests['v1_artifact'] is required and "
                         "must be non-empty; the v1 ranking is a cache input.")
    if not pre:
        raise ValueError("M17 cache identity REFUSED: manifests['teacher_query_preprocessing'] "
                         "is required; the teacher's prompt and truncation are cache inputs.")
    parts = {
        "raw_query_text_sha256": sha_texts([q.text for q in queries]),
        "query_records_sha256": _query_records_sha(queries),
        "source_split_manifest": {
            "sources": sorted({q.source for q in queries}),
            "families_sha256": sha_texts([q.family for q in queries]),
            "buckets": _tally(q.bucket for q in queries),
            **(man.get("source_split") or {})},
        "alias_manifest": {
            "pair_ids_sha256": sha_texts([q.alias_pair_id for q in queries]),
            "views_sha256": sha_texts([q.alias_view for q in queries]),
            **(man.get("alias") or {})},
        "teacher": {"model": reg["teacher"], "revision": reg["teacher_revision"],
                    "query_preprocessing": pre},
        "bank": bank.identity(),
        "v1_artifact": v1,
        "candidates": {"k": tr["candidate_k"], "mix_labeled": tr["candidate_mix_labeled"],
                       "mix_query_only": tr["candidate_mix_query_only"], "seed": cache_seed,
                       "rng": tr["candidate_construction"]["rng"],
                       "rng_recipe": RNG_RECIPE, "rng_recipe_version": RNG_RECIPE_VERSION},
    }
    return {"parts": parts, "sha256": sha_json(parts)}


def save(out_dir, arrays, sidecar):
    out = Path(admit_write(out_dir))
    out.mkdir(parents=True, exist_ok=True)
    stored = {k: (v if v.dtype != object else np.asarray([str(x) for x in v]))
              for k, v in arrays.items()}
    sidecar = {**sidecar, "arrays_sha256": {k: sha_array(v) for k, v in stored.items()}}
    np.savez(out / "candidates.npz", **stored)
    write_json(out / "cache.json", sidecar)
    return out


def load(out_dir):
    """Load and VERIFY: a cache whose arrays no longer hash to the sidecar is not the cache."""
    out = Path(out_dir)
    z = np.load(admit_read(out / "candidates.npz"), allow_pickle=False)
    arrays = {k: z[k] for k in z.files}
    sidecar = json.loads(admit_read(out / "cache.json").read_text())
    want = sidecar.get("arrays_sha256")
    if not want:
        raise SystemExit(f"M17 CACHE REFUSED: {out}/cache.json records no per-array hashes; "
                         "it was not written by cache.save.")
    for k, w in want.items():
        if k not in arrays:
            raise SystemExit(f"M17 CACHE REFUSED: array {k!r} is missing from {out}.")
        got = sha_array(arrays[k])
        if got != w:
            raise SystemExit(f"M17 CACHE REFUSED: array {k!r} hashes {got[:12]} but the sidecar "
                             f"records {w[:12]}; the stored cache has been altered.")
    return arrays, sidecar
