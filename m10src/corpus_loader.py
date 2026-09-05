"""The M10 query corpora -> the trainer. Sources, the manifest, and the form-balanced sampler.

`data10` loads the M9 pool only, which is arm A1 and nothing else. This module adds the rest of
the corpus and the sampler the anchor is registered to use:

- **Sources are declared, not discovered** (`SOURCES`). Each names a file, a kind, and how its
  rows map onto the 12 forms. The generated half does not exist yet; its entry is here and reads a
  jsonl of the same shape (`text`/`query`/`question`, `form`, `doc`/`seed_id`), so it costs no code change.
- **Every file is hashed into the manifest** with its row count per form, because §1 records the
  corpus a number was produced from and a gitignored `work/` file is mutable.
- **The order is a pure function of (seed, step)**, exactly as in `data10`: a resumed arm draws the
  batches an uninterrupted one would (`test_trainer10`'s resume property).
- **Form-balanced sampling is the anchor default** (`instructions-m10.md`:478-485): equal
  presentation share per form, with replacement within a form. `balanced=False` is the unbalanced
  variant -- the A2 volume-control arm and the reported diagnostic, never a silent fallback.

Targets: the M9 pool's stella vectors already exist (M7's `trainq-337981` matrix and
`work/enc9/m9long-{nqopen,triviaqa}`); everything new comes from `m10src/targets10`'s
content-hash cache. Both are read as fp16 and normalized in fp32 at batch time, so no path
materializes 5.3M x 1024 fp32 (21 GB) and cold and warm reads agree bit for bit.

**Read as data, never trusted as instructions**: the harvested and generated rows are text drawn
from corpora, and nothing here executes anything they contain.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import numpy as np
import torch

import data10 as D

WORK = REPO / "work"
TOKCACHE = WORK / "m10tok"

# The 12 registered forms, in `m10src/forms.RUBRIC` order. Imported rather than retyped so a form
# cannot exist here and not there.
def form_names():
    import forms
    return tuple(forms.RUBRIC)


FORMS = form_names()
FORM_ID = {f: i for i, f in enumerate(FORMS)}

# The FORMS-12 hold-out. Never a training source; refused by path, not by convention.
HOLDOUT_FILES = {str((WORK / "m10harvest" / "harvest_forms12.jsonl").resolve())}

# --- AMBIGUITY (reported, not decided): neither the mandate nor the registry assigns a form to a
# PAQ row or to an M9-pool row, and the balanced sampler needs one per row. The reading here
# changes nothing registered: PAQ is registered as "factoid volume" (`instructions-m10.md`:360),
# and the M9 pool's own source labels map onto two of the twelve forms -- `esci-us` is a shopper's
# product search, the other five are questions. Both tables are constants so a ruling is a
# one-line change.
PAQ_FORM = "factoid"
M9_SOURCE_FORM = {"esci-us": "product", "hotpotqa-train": "factoid", "squad-train": "factoid",
                  "mrtydi-en": "factoid", "nqopen": "factoid", "triviaqa": "factoid"}

SOURCES = {
    "m9-pool":    {"kind": "m9", "n_expected": 463_314,
                   "what": "M9 real queries: queries_pair (esci/hotpotqa/squad/mrtydi, fever out) "
                           "+ nqopen + triviaqa. Targets already exist."},
    "paq-build":  {"kind": "jsonl", "path": WORK / "m10paq" / "paq_build.jsonl",
                   "form": PAQ_FORM, "n_expected": 1_000_000,
                   "what": "the build's PAQ sample, nested inside the A2 sample"},
    "paq-a2":     {"kind": "jsonl", "path": WORK / "m10paq" / "paq_a2.jsonl",
                   "form": PAQ_FORM, "n_expected": 4_037_000,
                   "what": "the A2 volume-control PAQ sample"},
    "harvest":    {"kind": "jsonl", "path": WORK / "m10harvest" / "harvest_train.jsonl",
                   "n_expected": 1_248_386,
                   "what": "A3's harvested real text, FORMS-12 hold-out already removed"},
    "generated":  {"kind": "jsonl", "path": WORK / "m10gen" / "generated_queries.jsonl",
                   "optional": True,
                   "what": "the seven generated forms; does not exist yet, same row shape"},
}

# Which sources an arm's corpus is. A2 takes the 4.037M volume-control sample and A3/A4 the 1.0M
# build sample nested inside it -- `m10/LEDGER.md` §1: "A2 is the volume control, so 'the build
# with less PAQ volume' is the coherent nesting".
ARM_SOURCES = {
    "A1":     ("m9-pool",),
    "A2":     ("m9-pool", "paq-a2"),
    "A3":     ("m9-pool", "paq-build", "harvest"),
    "A4":     ("m9-pool", "paq-build", "harvest", "generated"),
    "ANCHOR": ("m9-pool", "paq-build", "harvest", "generated"),      # A4 IS the anchor arm
}


def sha_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class Segment:
    """One source's rows: its texts, their form codes, and where their teacher vectors live.

    `array` is an (m, 1024) fp16 array (memmap or in-memory) and `rowmap[i]` is the row of
    `array` holding the target for `texts[i]`.
    """

    def __init__(self, name, texts, forms, array, rowmap):
        self.name, self.texts = name, list(texts)
        self.forms = np.asarray(forms, dtype=np.int16)
        self.array, self.rowmap = array, np.asarray(rowmap, dtype=np.int64)
        assert len(self.texts) == len(self.forms) == len(self.rowmap), name

    def __len__(self):
        return len(self.texts)


# --------------------------------------------------------------------------------- reading ----

def _rows_from_jsonl(path, default_form=None, limit=None):
    """-> (texts, form names). Accepts `text`, `query` or `question` (PAQ's field) and `form` or
    the source's default, which is what lets the generated half land here unchanged."""
    path = Path(path)
    if str(path.resolve()) in HOLDOUT_FILES:
        raise SystemExit(f"REFUSED: {path} is the FORMS-12 hold-out. Queries harvested or "
                         f"generated from held-out documents are never trained on "
                         f"(instructions-m10.md:454).")
    texts, forms = [], []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            t = r.get("text") or r.get("query") or r.get("question")
            if not t:
                raise SystemExit(f"{path}: a row carries no text field "
                                 f"(text/query/question): {list(r)}")
            f = r.get("form", default_form)
            if f not in FORM_ID:
                raise SystemExit(f"{path}: form {f!r} is not one of the 12 registered forms")
            texts.append(t)
            forms.append(f)
            if limit and len(texts) >= limit:
                break
    return texts, forms


def source_texts(name, limit=None):
    """-> (texts, form names, manifest). The read every consumer shares, targets aside."""
    spec = SOURCES[name]
    if spec["kind"] == "m9":
        texts, forms, man = _m9_texts()
    else:
        p = Path(spec["path"])
        if not p.exists():
            if spec.get("optional"):
                raise SystemExit(f"source {name!r} is not built yet: {p} does not exist")
            raise SystemExit(f"source {name!r}: {p} is missing")
        texts, forms = _rows_from_jsonl(p, default_form=spec.get("form"))
        man = {"path": str(p), "sha256": sha_file(p), "bytes": p.stat().st_size}
    if spec.get("n_expected") and len(texts) != spec["n_expected"]:
        raise SystemExit(f"source {name!r}: {len(texts):,} rows, registered {spec['n_expected']:,}")
    man.update({"source": name, "kind": spec["kind"], "what": spec["what"], "n_rows": len(texts),
                "by_form": {f: forms.count(f) for f in sorted(set(forms))}})
    if limit:
        texts, forms = texts[:limit], forms[:limit]
        man["limit"] = limit
    return texts, forms, man


def _m9_texts():
    """The 463,314 M9 real queries, labelled by source and mapped onto the form taxonomy."""
    import data as m9data
    texts, srcs, meta = _m9_labelled()
    keep = [i for i, s in enumerate(srcs) if s not in m9data.FEVER_SOURCES]
    out = [texts[i] for i in keep]
    labels = [srcs[i] for i in keep]
    for name in ("nqopen", "triviaqa"):
        t, _row = _m9_extra(name)
        out += t
        labels += [name] * len(t)
    forms = [M9_SOURCE_FORM[s] for s in labels]
    man = {"path": "m9src/data.labelled_query_pool + work/enc9/m9long-{nqopen,triviaqa}",
           "sha256": hashlib.sha256("\x00".join(out).encode("utf-8", "surrogatepass")).hexdigest(),
           "by_m9_source": {s: labels.count(s) for s in sorted(set(labels))},
           "m8_manifest_sha256": meta["m8_manifest_sha256"]}
    return out, forms, man


_M9_POOL = {}


def _m9_labelled():
    """Memoized: `labelled_query_pool` rebuilds the M8/M9 derivation and is ~30 s and a few GB."""
    if "p" not in _M9_POOL:
        import data as m9data
        _M9_POOL["p"] = m9data.labelled_query_pool()
    return _M9_POOL["p"]


_M9_EXTRA = {}


def _m9_extra(name):
    """nqopen / triviaqa: the extended-screen survivors and their cached stella vectors."""
    if name not in _M9_EXTRA:
        import longrun
        texts, row = longrun.extra_texts()[name]
        _M9_EXTRA[name] = (texts, row)
    return _M9_EXTRA[name]


def _m9_segments():
    """-> segments for the M9 pool, each pointing at the stella cache that already holds it."""
    import data as m9data
    import longrun
    texts, srcs, _meta = _m9_labelled()
    keep = np.array([i for i, s in enumerate(srcs) if s not in m9data.FEVER_SOURCES],
                    dtype=np.int64)
    qp_forms = [FORM_ID[M9_SOURCE_FORM[srcs[int(i)]]] for i in keep]
    segs = [Segment("m9-queries_pair", [texts[int(i)] for i in keep], qp_forms,
                    np.asarray(m9data.stella_query_targets()), keep)]
    for name in ("nqopen", "triviaqa"):
        t, _row = _m9_extra(name)
        v = np.load(longrun.target_dir(name) / "vecs.npy", mmap_mode="r")
        assert v.shape[0] == len(t), f"{name}: {v.shape[0]} vectors for {len(t)} texts"
        segs.append(Segment(f"m9-{name}", t, [FORM_ID[M9_SOURCE_FORM[name]]] * len(t), v,
                            np.arange(len(t), dtype=np.int64)))
    return segs


def load_segments(names, head_per_source=None, verbose=True):
    """-> (segments, manifest). The corpus a screen arm trains on.

    `head_per_source` keeps the first N rows of each source IN FILE ORDER -- a smoke device, not a
    sample (the harvest file is grouped by form), and it is applied BEFORE the target lookup so a
    smoke does not demand teacher vectors for 1.25M rows it will never draw.
    """
    segs, man = [], {"sources": [], "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    import targets10
    cache = None
    for name in names:
        t0 = time.time()
        if SOURCES[name]["kind"] == "m9":
            new = _m9_segments()
            if head_per_source:
                new = [Segment(sg.name, sg.texts[:head_per_source], sg.forms[:head_per_source],
                               sg.array, sg.rowmap[:head_per_source]) for sg in new]
            _texts, _forms, sman = source_texts(name)
        else:
            texts, forms, sman = source_texts(name, limit=head_per_source)
            cache = cache or targets10.TargetCache()
            rowmap = cache.rows_for(texts)
            miss = int((rowmap < 0).sum())
            if miss:
                raise SystemExit(
                    f"source {name!r}: {miss:,} of {len(texts):,} texts have no teacher target.\n"
                    f"Run: .venv/bin/python m10src/targets10.py --sources {name}")
            new = [Segment(name, texts, [FORM_ID[f] for f in forms], cache.vecs(), rowmap)]
        segs += new
        sman["segments"] = [s.name for s in new]
        sman["seconds"] = round(time.time() - t0, 1)
        man["sources"].append(sman)
        if verbose:
            print(f"  {name}: {sum(len(s) for s in new):,} rows ({sman['seconds']:.0f}s)",
                  flush=True)
    man["n_rows"] = sum(len(s) for s in segs)
    by_form = {}
    for s in segs:
        for fid, c in zip(*np.unique(s.forms, return_counts=True)):
            by_form[FORMS[int(fid)]] = by_form.get(FORMS[int(fid)], 0) + int(c)
    man["by_form"] = dict(sorted(by_form.items()))
    man["sha256"] = hashlib.sha256(
        json.dumps({"sources": man["sources"], "n": man["n_rows"]}, sort_keys=True,
                   default=str).encode()).hexdigest()
    return segs, man


# ------------------------------------------------------------------------------- the corpus ----

class PackedIds:
    """Pretokenized ids as one flat int32 array plus offsets, not a list of 5.3M small arrays.

    M9's pitfall 14: a Python list of ~20 token ids costs ~1 kB once the object headers are
    counted, so "tokenize everything, then batch" is tens of GB of transient heap at this scale.
    Exposes `__getitem__`/`__len__` so `data10.collate` needs no change, plus `lengths` so
    `data10.length_buckets` does not have to call `__getitem__` 5.3M times.
    """

    def __init__(self, flat, offs):
        self.flat, self.offs = flat, np.asarray(offs, dtype=np.int64)
        self.lengths = np.diff(self.offs)

    def __len__(self):
        return len(self.offs) - 1

    def __getitem__(self, i):
        return self.flat[self.offs[i]:self.offs[i + 1]]


def pack_tokenize(tok, texts, max_len=512, prefix="", batch=20_000, label="", verbose=True):
    """-> PackedIds. Chunked, so the transient heap is one batch, not the corpus."""
    parts, lens, t0 = [], [], time.time()
    for i in range(0, len(texts), batch):
        ids = tok([prefix + t for t in texts[i:i + batch]], truncation=True, max_length=max_len,
                  add_special_tokens=True)["input_ids"]
        L = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
        flat = np.empty(int(L.sum()), dtype=np.int32)
        pos = 0
        for x in ids:
            flat[pos:pos + len(x)] = x
            pos += len(x)
        parts.append(flat)
        lens.append(L)
        if verbose and label and (i // batch) % 25 == 0 and i:
            el = time.time() - t0
            print(f"    {label} {i:,}/{len(texts):,} ({i / max(el, 1e-9):,.0f}/s)", flush=True)
    L = np.concatenate(lens) if lens else np.zeros(0, dtype=np.int64)
    offs = np.zeros(L.size + 1, dtype=np.int64)
    np.cumsum(L, out=offs[1:])
    return PackedIds(np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32), offs)


def tokenize_corpus(tok, segs, man, student, max_len=512, prefix="", cache=True, verbose=True,
                    extra_ident=None):
    """-> PackedIds over every segment's texts in order, cached on the corpus identity.

    The cache key binds the manifest hash, the student (tokenizers differ), the prefix and the
    length cap -- the four things that change the ids. Re-tokenizing 5.3M texts for each of 16
    arms is minutes each; the cache is 100s of MB.
    """
    texts = [t for s in segs for t in s.texts]
    ident = {"manifest": man["sha256"], "student": student, "prefix": prefix, "max_len": max_len,
             "n": len(texts), **(extra_ident or {})}
    d = TOKCACHE / hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    if cache and (d / "offs.npy").exists():
        if verbose:
            print(f"  tokens: cached at {d}", flush=True)
        return PackedIds(np.load(d / "flat.npy", mmap_mode="r"), np.load(d / "offs.npy"))
    p = pack_tokenize(tok, texts, max_len=max_len, prefix=prefix, label="tokenize",
                      verbose=verbose)
    if cache:
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "flat.npy", p.flat)
        np.save(d / "offs.npy", p.offs)
        (d / "meta.json").write_text(json.dumps(ident, indent=1))
    return p


class TargetView:
    """Row -> fp32 unit-norm teacher vector, gathered across the segments' fp16 stores.

    Never materialized: 5.3M x 1024 fp32 is 21 GB, and the fp16 stores are memmaps.
    """

    def __init__(self, segs):
        self.segs = list(segs)
        self.bounds = np.cumsum([0] + [len(s) for s in self.segs])

    def __len__(self):
        return int(self.bounds[-1])

    def __getitem__(self, idx):
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        out = np.empty((len(idx), self.segs[0].array.shape[1]), dtype=np.float32)
        which = np.searchsorted(self.bounds, idx, side="right") - 1
        for b in np.unique(which):
            sel = np.flatnonzero(which == b)
            s = self.segs[int(b)]
            rows = s.rowmap[idx[sel] - self.bounds[int(b)]]
            out[sel] = np.asarray(s.array[rows], dtype=np.float32)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        if not np.isfinite(n).all() or n.min() < 1e-6:
            raise SystemExit("a teacher target is ~zero or non-finite; it must never reach a "
                             "trainer")
        return out / n


def corpus_forms(segs):
    return np.concatenate([s.forms for s in segs]) if segs else np.zeros(0, dtype=np.int16)


# ------------------------------------------------------------------------------- the sampler ----

def _form_batches(idx, lengths, batch_size, seed):
    """Length-bucketed batches over ONE form's rows, wrapping the ragged tail.

    Wrapping is the registered "with replacement within a form": a form smaller than a batch still
    yields a batch, and no row is dropped for landing in the tail.
    """
    order = idx[np.argsort(lengths[idx], kind="stable")]
    if len(order) == 0:
        return []
    n = int(np.ceil(len(order) / batch_size))
    take = np.resize(order, n * batch_size)              # wraps to the head of this form's order
    batches = [take[i * batch_size:(i + 1) * batch_size] for i in range(n)]
    np.random.default_rng(seed).shuffle(batches)
    return batches


class FormBalancedStream:
    """Query batches with an equal presentation share per form, addressable by step.

    `batch(k)` is a pure function of `k`: cycle `c = k // F` draws a fresh permutation of the F
    forms from `default_rng([seed, c])`, and within a cycle each form contributes exactly one
    batch -- so shares are exactly equal over every full cycle and the resume property holds.
    Each batch is one form, which is also what keeps length bucketing worth having.

    `balanced=False` samples by example over the whole corpus (`data10.length_buckets`): the A2
    volume-control arm's variant and the reported diagnostic.
    """

    def __init__(self, ids, targets, forms, pad_id, batch_size=32, seed=0, balanced=True):
        self.ids, self.T, self.pad, self.bs, self.seed = ids, targets, pad_id, batch_size, seed
        self.forms = np.asarray(forms, dtype=np.int16)
        self.balanced = balanced
        lengths = getattr(ids, "lengths", None)
        if lengths is None:
            lengths = np.array([len(ids[i]) for i in range(len(ids))], dtype=np.int64)
        if balanced:
            self.present = [int(f) for f in np.unique(self.forms)]
            self.batches = {f: _form_batches(np.flatnonzero(self.forms == f), lengths,
                                             batch_size, seed + 1 + f) for f in self.present}
            empty = [FORMS[f] for f in self.present if not self.batches[f]]
            if empty:
                raise ValueError(f"forms with no batch: {empty}")
        else:
            self.present = [-1]
            self.batches = {-1: D.length_buckets(ids, batch_size, seed=seed)}
            if not self.batches[-1]:
                raise ValueError("no full batches: corpus smaller than one batch")

    def __len__(self):
        return sum(len(v) for v in self.batches.values())

    def _pick(self, k):
        F = len(self.present)
        c, j = divmod(int(k), F)
        f = self.present[int(np.random.default_rng([self.seed, c]).permutation(F)[j])] \
            if F > 1 else self.present[0]
        bl = self.batches[f]
        return f, bl[c % len(bl)]

    def batch(self, k):
        _f, idx = self._pick(k)
        x, m = D.collate(self.ids, idx, self.pad)
        return x, m, torch.from_numpy(np.ascontiguousarray(self.T[idx]))

    def realized_shares(self, n_batches):
        """-> {form: share of presented EXAMPLES} over the first `n_batches` steps. §0b records
        these; every batch is one form and one size, so batch share is example share."""
        c = {}
        for k in range(int(n_batches)):
            f, _ = self._pick(k)
            c[f] = c.get(f, 0) + 1
        tot = max(sum(c.values()), 1)
        return {("ALL" if f < 0 else FORMS[f]): round(v / tot, 6) for f, v in
                sorted(c.items(), key=lambda kv: kv[0])}


# --------------------------------------------------------------------------------- data cut ----

def data_cut_count(registry=None):
    """The registered post-screen unique-text count A2/A3/A4 are cut to, or None while §0b is
    open. It is `min` of the three corpora and cannot be computed before generation runs."""
    reg = registry or json.loads((REPO / "m10" / "screen_registry.json").read_text())
    return reg.get("data_cut", {}).get("unique_text_count")


def apply_data_cut(segs, count, seed=0):
    """-> (segments, report). Uniform seed-0 downsample of the WHOLE corpus to `count` rows.

    Uniform over the corpus, not per source: the cut exists so A2, A3 and A4 differ in WHICH text
    they carry and never in how much (`screen_registry.data_cut`), and a per-source cut would
    additionally re-weight the sources.
    """
    n = sum(len(s) for s in segs)
    if count is None or count >= n:
        return segs, {"applied": False, "n": n, "count": count,
                      "_why": "no registered count (§0b open)" if count is None
                              else "corpus already at or below the cut"}
    pick = np.sort(np.random.default_rng(seed).choice(n, size=int(count), replace=False))
    out, lo = [], 0
    for s in segs:
        hi = lo + len(s)
        sel = pick[(pick >= lo) & (pick < hi)] - lo
        if len(sel):
            out.append(Segment(s.name, [s.texts[int(i)] for i in sel], s.forms[sel], s.array,
                               s.rowmap[sel]))
        lo = hi
    return out, {"applied": True, "n_before": n, "n_after": int(sum(len(s) for s in out)),
                 "seed": seed, "per_segment": {s.name: len(s) for s in out}}


# ---------------------------------------------------------------------------------- the arm ----

def build_query_stream(arm_or_sources, tok, student, *, batch_size=32, seed=0, balanced=True,
                       max_len=512, prefix="", head_per_source=None, cut=None, verbose=True):
    """-> (stream, manifest). Everything above, in the order an arm needs it."""
    names = ARM_SOURCES[arm_or_sources] if isinstance(arm_or_sources, str) else tuple(
        arm_or_sources)
    segs, man = load_segments(names, head_per_source=head_per_source, verbose=verbose)
    if head_per_source:
        man["head_per_source"] = head_per_source
    cut = data_cut_count() if cut == "registered" else cut
    segs, cut_rep = apply_data_cut(segs, cut)
    man["data_cut"] = cut_rep
    ids = tokenize_corpus(tok, segs, man, student, max_len=max_len, prefix=prefix,
                          verbose=verbose,
                          extra_ident={"data_cut": cut_rep, "head_per_source": head_per_source})
    stream = FormBalancedStream(ids, TargetView(segs), corpus_forms(segs),
                                pad_id=tok.pad_token_id, batch_size=batch_size, seed=seed,
                                balanced=balanced)
    man.update({"arm": arm_or_sources if isinstance(arm_or_sources, str) else None,
                "sources_used": list(names), "student": student, "max_len": max_len,
                "student_prefix": prefix, "batch_size": batch_size, "seed": seed,
                "balanced": balanced, "n_batches": len(stream),
                "n_tokens": int(ids.offs[-1]),
                "mean_tokens": round(float(ids.offs[-1]) / max(len(ids), 1), 2)})
    return stream, man


def doc_marker():
    """M9's registered document-role student marker, read from `m9/registry.json` rather than
    retyped. §Data: "document-role examples carry M9's fixed document-role marker"."""
    return json.loads((REPO / "m9" / "registry.json").read_text())["templates"]["doc_student"]


def build_doc_stream(n, tok, *, batch_size=32, seed=0, max_len=512, verbose=True):
    """-> (stream, meta) for the document-role half of the mix, from the frozen M9 pool.

    The document marker is applied HERE, once. `data10.pretokenize` used to take no prefix at all,
    so every document reached the student as raw bytes -- the query-role policy -- while its
    teacher target was the raw-bytes document encoding. The teacher side was right; the student
    side dropped the marker the recipe names.
    """
    texts, vecs, meta = D.m9_doc_pool(n, seed=seed)
    pre = doc_marker()
    ids = D.pretokenize(tok, texts, max_len=max_len, prefix=pre, verbose=verbose, label="documents")
    meta = {**meta, "student_prefix": pre, "n": len(texts), "max_len": max_len}
    return D.Stream(ids, vecs, pad_id=tok.pad_token_id, batch_size=batch_size, seed=seed), meta


class LengthsOnly:
    """Just enough of a pretokenized corpus to bucket and count with: the lengths. Lets the
    manifest report realized shares before a single teacher vector exists."""

    def __init__(self, lengths):
        self.lengths = np.asarray(lengths, dtype=np.int64)

    def __len__(self):
        return len(self.lengths)


def main():
    """Report the manifest and the realized form shares. Needs no teacher targets and no GPU."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["harvest"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shares-over", type=int, default=100_000, help="batches to realize shares on")
    ap.add_argument("--unbalanced", action="store_true")
    ap.add_argument("--out", default=str(REPO / "results" / "m10_corpus_manifest.json"))
    a = ap.parse_args()
    man, forms, words = {"sources": [], "sources_used": a.sources}, [], []
    for name in a.sources:
        texts, f, sman = source_texts(name)
        man["sources"].append(sman)
        forms += [FORM_ID[x] for x in f]
        words.append(np.fromiter((len(t.split()) for t in texts), dtype=np.int64, count=len(texts)))
        print(f"  {name}: {len(texts):,} rows {sman['by_form']}", flush=True)
    forms = np.asarray(forms, dtype=np.int16)
    words = np.concatenate(words)
    man["n_rows"] = int(len(forms))
    man["by_form"] = {FORMS[int(f)]: int(c) for f, c in zip(*np.unique(forms, return_counts=True))}
    st = FormBalancedStream(LengthsOnly(words), None, forms, pad_id=0,
                            batch_size=a.batch_size, seed=0, balanced=not a.unbalanced)
    man["balanced"] = not a.unbalanced
    man["realized_shares"] = st.realized_shares(a.shares_over)
    man["shares_over_batches"] = a.shares_over
    man["word_len"] = {"mean": round(float(words.mean()), 2), "p50": int(np.percentile(words, 50)),
                       "p95": int(np.percentile(words, 95))}
    Path(a.out).write_text(json.dumps(man, indent=1, default=str))
    print(json.dumps({k: v for k, v in man.items() if k != "sources"}, indent=1, default=str))


if __name__ == "__main__":
    main()
