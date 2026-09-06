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

**A 128-bit content hash is treated as text equality** here as well as in `targets10` -- the
hold-out guard and the corpus dedup both key on `text_hash`. Same decision, same arithmetic: at
10^7 texts a blake2b-128 collision is ~10^-24 (Codex 2026-09-05 finding 12, declined).

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

# The FORMS-12 hold-out. Refused by PATH and, since a copy defeats a path, by the CONTENT hash of
# every one of its rows: a training row whose text is a hold-out text is refused whatever file it
# arrived in (Codex 2026-09-05 finding 3 -- `cp harvest_forms12.jsonl generated_queries.jsonl`).
HOLDOUT_FILES = {str((WORK / "m10harvest" / "harvest_forms12.jsonl").resolve())}
_HOLDOUT_HASHES = {}


def text_hash(t):
    return hashlib.blake2b(t.encode("utf-8", "surrogatepass"), digest_size=16).digest()


def holdout_hashes(path=None):
    """-> {blake2b-128 of every FORMS-12 hold-out text}, or an empty set if the file is absent.

    The cache key is the file's own CONTENT digest, not its size and mtime: a rewrite that lands
    within the same wall-clock second and at the same byte length is invisible to a size+mtime key
    but is a different corpus, and a hold-out is exactly the file a silent staleness bug must
    never touch (Codex 2026-09-05 pass 3).
    """
    p = Path(path) if path else (WORK / "m10harvest" / "harvest_forms12.jsonl")
    k = (str(p), sha_file(p) if p.exists() else None)
    if k not in _HOLDOUT_HASHES:
        out = set()
        if p.exists():
            with p.open() as fh:
                for line in fh:
                    r = json.loads(line)
                    t = r.get("text") or r.get("query") or r.get("question")
                    if t:
                        out.add(text_hash(t))
        _HOLDOUT_HASHES[k] = out
    return _HOLDOUT_HASHES[k]


def refuse_holdout_texts(segs, path=None):
    """-> the count refused (always 0, or SystemExit). Applied at LOAD time, to every source.

    An ABSENT or empty hold-out raises rather than waving the corpus through: a guard that turns
    itself off when its input disappears protects nothing, and is indistinguishable from a clean
    pass in the artifact (Codex re-review 2026-09-05).
    """
    hs = holdout_hashes(path)
    if not hs:
        raise SystemExit(
            f"REFUSED: the FORMS-12 hold-out ({path or WORK / 'm10harvest' / 'harvest_forms12.jsonl'}) "
            f"is missing or empty, so no training row can be checked against it. Build it before "
            f"loading a corpus.")
    bad = [(s.name, i) for s in segs for i, t in enumerate(s.texts) if text_hash(t) in hs]
    if bad:
        where = {}
        for n, _i in bad:
            where[n] = where.get(n, 0) + 1
        raise SystemExit(f"REFUSED: {len(bad):,} training rows are FORMS-12 hold-out texts "
                         f"{where}. Queries harvested or generated from held-out documents are "
                         f"never trained on (instructions-m10.md:454), whatever file they are in.")
    return 0

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

def _rows_from_jsonl(path, default_form=None, limit=None, with_ids=False):
    """-> (texts, form names), or (texts, form names, doc ids) when `with_ids=True`. Accepts
    `text`, `query` or `question` (PAQ's field) and `form` or the source's default, which is what
    lets the generated half land here unchanged. `with_ids` defaults False so every existing
    2-tuple caller is unaffected; a row's `doc` id (or `seed_id` if `doc` is absent) is the
    provenance `load_segments` checks against the FORMS-12 hold-out's own document ids."""
    path = Path(path)
    if str(path.resolve()) in HOLDOUT_FILES:
        raise SystemExit(f"REFUSED: {path} is the FORMS-12 hold-out. Queries harvested or "
                         f"generated from held-out documents are never trained on "
                         f"(instructions-m10.md:454).")
    texts, forms, ids = [], [], []
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
            if with_ids:
                d = r.get("doc")
                ids.append(d if d is not None else r.get("seed_id"))
            if limit and len(texts) >= limit:
                break
    if with_ids:
        return texts, forms, ids
    return texts, forms


def held_out_doc_ids(path=None):
    """-> {document ids no training row may be derived from}, read from the FORMS-12 hold-out
    file's OWN rows -- `corpus10.harvest_holdout` already partitioned by document, so the file at
    `HOLDOUT_FILES` IS the held-out set (`harvest_holdout.json` records only its size, not the
    ids). Light provenance, additional to the text-hash check: a held-out document's TITLE differs
    from its held-out HEADING, so a text-only guard can miss a query the document produced that
    the (smaller) hold-out draw does not happen to repeat verbatim; this catches it by `doc` id
    instead. An absent file (no harvest built yet) is an empty set, not a refusal -- this check
    binds only sources that carry a `doc`/`seed_id` field at all.
    """
    p = Path(path) if path else (WORK / "m10harvest" / "harvest_forms12.jsonl")
    if not p.exists():
        return set()
    out = set()
    with p.open() as fh:
        for line in fh:
            r = json.loads(line)
            d = r.get("doc") if r.get("doc") is not None else r.get("seed_id")
            if d is not None:
                out.add(d)
    return out


def source_texts(name, limit=None):
    """-> (texts, form names, doc ids or None, manifest). The read every consumer shares, targets
    aside. `doc ids` is a list aligned with `texts` (each a `doc`/`seed_id` value or None), or
    plain `None` for the M9 kind, which carries no per-row document provenance."""
    spec = SOURCES[name]
    ids = None
    if spec["kind"] == "m9":
        texts, forms, man = _m9_texts()
    else:
        p = Path(spec["path"])
        if not p.exists():
            if spec.get("optional"):
                raise SystemExit(f"source {name!r} is not built yet: {p} does not exist")
            raise SystemExit(f"source {name!r}: {p} is missing")
        texts, forms, ids = _rows_from_jsonl(p, default_form=spec.get("form"), with_ids=True)
        man = {"path": str(p), "sha256": sha_file(p), "bytes": p.stat().st_size}
    # the registered count is the pool BEFORE the M10 re-screen: the re-screen is a removal whose
    # size is a measurement, not a constant, and it is reported in `man` instead.
    n_reg = man.get("n_before_rescreen", len(texts))
    if spec.get("n_expected") and n_reg != spec["n_expected"]:
        raise SystemExit(f"source {name!r}: {n_reg:,} rows, registered {spec['n_expected']:,}")
    man.update({"source": name, "kind": spec["kind"], "what": spec["what"], "n_rows": len(texts),
                "by_form": {f: forms.count(f) for f in sorted(set(forms))}})
    if limit:
        texts, forms = texts[:limit], forms[:limit]
        if ids is not None:
            ids = ids[:limit]
        man["limit"] = limit
    return texts, forms, ids, man


def _m9_texts(screen=True):
    """The 463,314 M9 real queries, labelled by source and mapped onto the form taxonomy, then
    RE-SCREENED against the M10 protected index (`rescreen10`, instructions-m10.md:462)."""
    import data as m9data
    texts, srcs, meta = _m9_labelled()
    keep = [i for i, s in enumerate(srcs) if s not in m9data.FEVER_SOURCES]
    out = [texts[i] for i in keep]
    labels = [srcs[i] for i in keep]
    for name in ("nqopen", "triviaqa"):
        t, _row = _m9_extra(name)
        out += t
        labels += [name] * len(t)
    man = {"path": "m9src/data.labelled_query_pool + work/enc9/m9long-{nqopen,triviaqa}",
           "by_m9_source": {s: labels.count(s) for s in sorted(set(labels))},
           "m8_manifest_sha256": meta["m8_manifest_sha256"],
           "n_before_rescreen": len(out)}
    if screen:
        import rescreen10
        removed, per_seg, masks = 0, {}, []
        # the three unscreened segments are the same three lists, in the same order, as `out`
        for sg in _m9_segments(screen=False):
            m, rep = rescreen10.query_keep_mask(sg.texts, sg.name, compute=False)
            masks.append(m)
            removed += rep["removed"]
            per_seg[sg.name] = {"removed": rep["removed"], "by_hit": rep["by_hit"]}
        mask = np.concatenate(masks)
        assert len(mask) == len(out), (len(mask), len(out))
        out = [t for t, k in zip(out, mask) if k]
        labels = [l for l, k in zip(labels, mask) if k]
        man["rescreen10"] = {"removed": removed, "per_segment": per_seg,
                             "protected10": rescreen10._h(rescreen10.protected_ident())}
    forms = [M9_SOURCE_FORM[s] for s in labels]
    man["sha256"] = hashlib.sha256(
        "\x00".join(out).encode("utf-8", "surrogatepass")).hexdigest()
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


def _m9_segments(screen=True):
    """-> segments for the M9 pool, each pointing at the stella cache that already holds it.

    `screen=False` is the UNSCREENED pool and exists for exactly one caller: `rescreen10`, which
    has to see the rows in order to decide which of them survive. Every training path screens.
    """
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
    if screen:
        import rescreen10
        out = []
        for sg in segs:
            m, _rep = rescreen10.query_keep_mask(sg.texts, sg.name, compute=False)
            sel = np.flatnonzero(m)
            out.append(Segment(sg.name, [sg.texts[int(i)] for i in sel], sg.forms[sel], sg.array,
                               sg.rowmap[sel]))
        segs = out
    return segs


def dedup_segments(segs):
    """-> (segments, {segment: rows removed}). EXACT text dedup, globally across sources, first
    occurrence kept.

    The registered cut is a "post-screen unique-text count" (`m10/screen_registry.json`), so both
    the count and the cut have to be taken on unique texts: `["x", "x", "y"]` is two texts, not
    three, and keeping both copies of `x` also doubles its presentation weight inside its form.
    """
    seen, out, removed = set(), [], {}
    for s in segs:
        keep = np.zeros(len(s), dtype=bool)
        for i, t in enumerate(s.texts):
            h = text_hash(t)
            if h not in seen:
                seen.add(h)
                keep[i] = True
        removed[s.name] = int((~keep).sum())
        sel = np.flatnonzero(keep)
        out.append(Segment(s.name, [s.texts[int(i)] for i in sel], s.forms[sel], s.array,
                           s.rowmap[sel]))
    return out, removed


def load_segments(names, head_per_source=None, verbose=True):
    """-> (segments, manifest). The corpus a screen arm trains on.

    `head_per_source` keeps the first N rows of each source IN FILE ORDER -- a smoke device, not a
    sample (the harvest file is grouped by form), and it is applied BEFORE the target lookup so a
    smoke does not demand teacher vectors for 1.25M rows it will never draw.
    """
    segs, man = [], {"sources": [], "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    import targets10
    cache = None
    held = held_out_doc_ids()
    for name in names:
        t0 = time.time()
        if SOURCES[name]["kind"] == "m9":
            new = _m9_segments()
            if head_per_source:
                new = [Segment(sg.name, sg.texts[:head_per_source], sg.forms[:head_per_source],
                               sg.array, sg.rowmap[:head_per_source]) for sg in new]
            _texts, _forms, _ids, sman = source_texts(name)
            if head_per_source:
                sman["head_per_source"] = head_per_source
                sman["n_rows"] = sum(len(sg) for sg in new)
        else:
            texts, forms, ids, sman = source_texts(name, limit=head_per_source)
            # provenance: a training row must not be derived from a FORMS-12 held-out document,
            # whatever its text (the text-hash guard below is defense in depth, not the primary
            # check -- a held title and a held heading are different STRINGS from the same doc).
            if ids is not None:
                hit = sum(1 for d in ids if d is not None and d in held)
                sman["held_out_doc_ids_refused"] = hit
                if hit:
                    raise SystemExit(
                        f"source {name!r}: {hit:,} rows carry a document id in the FORMS-12 "
                        f"hold-out ({len(held):,} held documents). Queries harvested or "
                        f"generated from held-out documents are never trained on "
                        f"(instructions-m10.md:454).")
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
    # the M10 re-screen identity, hoisted so the manifest hash and the token-cache key both bind
    # it: an unscreened M9 pool cannot produce a corpus that looks like a screened one.
    r = [sm.get("rescreen10") for sm in man["sources"] if sm.get("rescreen10")]
    man["rescreen10"] = r[0] if r else None
    # the FORMS-12 hold-out, by CONTENT: a copy under another name is still the hold-out
    man["holdout_rows_refused"] = refuse_holdout_texts(segs)
    man["holdout_texts"] = len(holdout_hashes())
    segs, dups = dedup_segments(segs)
    man["duplicates_removed"] = dups
    man["n_duplicates_removed"] = int(sum(dups.values()))
    for sm in man["sources"]:
        sm["duplicates_removed"] = {k: dups[k] for k in sm["segments"] if k in dups}
        sm["n_rows_deduped"] = sum(len(s) for s in segs if s.name in sm["segments"])
    if verbose and man["n_duplicates_removed"]:
        print(f"  dedup: {man['n_duplicates_removed']:,} duplicate texts removed", flush=True)
    man["n_rows"] = sum(len(s) for s in segs)
    by_form = {}
    for s in segs:
        for fid, c in zip(*np.unique(s.forms, return_counts=True)):
            by_form[FORMS[int(fid)]] = by_form.get(FORMS[int(fid)], 0) + int(c)
    man["by_form"] = dict(sorted(by_form.items()))
    # The identity is the CORPUS, not the run: `seconds` is a wall-clock measurement and would
    # otherwise put a fresh hash -- and so a fresh pretokenization -- on every single load.
    stable = [{k: v for k, v in sm.items() if k != "seconds"} for sm in man["sources"]]
    man["sha256"] = hashlib.sha256(
        json.dumps({"sources": stable, "n": man["n_rows"]}, sort_keys=True,
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


def tokenizer_ident(tok):
    """-> what identifies the TOKENIZER, not just the student's nickname.

    `student="bge-small"` is a label; two revisions of the same repo produce different ids for the
    same text, and a cache keyed on the label alone serves the older ones (Codex 2026-09-05
    finding 11). Everything here is best-effort and each field is optional -- a stub tokenizer in a
    test has none of them -- but the vocabulary is hashed in every case, which is the field that
    actually changes the ids.

    `tok.model_max_length` alone can hide a served-length difference: fastembed serves
    `min(model_max_length, max_length)` from `tokenizer_config.json`, and a `max_length` key
    beside `model_max_length` is not something `transformers` loads onto the tokenizer object at
    all (`m10/CODEMAP.md` pitfall 5 -- the exact bug that disqualified both MiniLM exports). So
    this also hashes `tokenizer_config.json` and `special_tokens_map.json` when they exist on
    disk BESIDE the tokenizer, and reads `truncation_side` off the object directly.
    """
    out = {"class": type(tok).__name__,
           "name_or_path": str(getattr(tok, "name_or_path", "") or ""),
           "revision": (getattr(tok, "init_kwargs", None) or {}).get("revision"),
           "vocab_size": getattr(tok, "vocab_size", None),
           "model_max_length": getattr(tok, "model_max_length", None),
           "truncation_side": getattr(tok, "truncation_side", None)}
    try:                                    # a fast tokenizer's `tokenizer.json`, verbatim
        out["tokenizer_json_sha256"] = hashlib.sha256(
            tok.backend_tokenizer.to_str().encode()).hexdigest()
    except Exception:
        out["tokenizer_json_sha256"] = None
    if out["tokenizer_json_sha256"] is None:
        try:
            v = tok.get_vocab()
            out["vocab_sha256"] = hashlib.sha256(
                json.dumps(v, sort_keys=True).encode()).hexdigest()
        except Exception:
            out["vocab_sha256"] = None
    d = Path(getattr(tok, "name_or_path", "") or "")
    for fname, key in (("tokenizer_config.json", "tokenizer_config_sha256"),
                       ("special_tokens_map.json", "special_tokens_map_sha256")):
        out[key] = None
        try:
            fp = d / fname
            if d.is_dir() and fp.exists():
                out[key] = hashlib.sha256(fp.read_bytes()).hexdigest()
        except Exception:
            out[key] = None
    return out


def tokenize_corpus(tok, segs, man, student, max_len=512, prefix="", cache=True, verbose=True,
                    extra_ident=None):
    """-> PackedIds over every segment's texts in order, cached on the corpus identity.

    The cache key binds the manifest hash, the student (tokenizers differ), the prefix and the
    length cap -- the four things that change the ids. Re-tokenizing 5.3M texts for each of 16
    arms is minutes each; the cache is 100s of MB.
    """
    texts = [t for s in segs for t in s.texts]
    ident = {"manifest": man["sha256"], "student": student, "prefix": prefix, "max_len": max_len,
             "n": len(texts), "tokenizer": tokenizer_ident(tok), **(extra_ident or {})}
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

def _form_draw(rows, lengths, batch_size, seed, form, occurrence):
    """One batch of ONE form, drawn WITH REPLACEMENT, then sorted by length for the padded chunk.

    §Data (`instructions-m10.md`:478-485): "each form's presentation share equal, texts drawn with
    replacement within a form". Cycling fixed length-bucketed batches is not that -- a form of
    three rows at batch 8 yielded the identical `0,1,2,0,1,2,0,1` every time it came up, and a
    large form cycled a fixed partition (Codex 2026-09-05 finding 7). The RNG is a pure function of
    (seed, form, occurrence), so the draw is still addressable by step and resume is exact.
    """
    rng = np.random.default_rng([int(seed), int(form) + 1, int(occurrence)])
    drawn = rng.choice(rows, size=int(batch_size), replace=True)
    return drawn[np.argsort(lengths[drawn], kind="stable")]


class FormBalancedStream:
    """Query batches with an equal presentation share per form, addressable by step.

    `batch(k)` is a pure function of `k`: cycle `c = k // F` draws a fresh permutation of the F
    forms from `default_rng([seed, c])`, and within a cycle each form contributes exactly one
    batch -- so shares are exactly equal over every full cycle and the resume property holds.
    Each batch is one form, which is also what keeps length bucketing worth having.

    `balanced=False` samples by example over the whole corpus (`data10.length_buckets`): the A2
    volume-control arm's variant and the reported diagnostic.
    """

    def __init__(self, ids, targets, forms, pad_id, batch_size=32, seed=0, balanced=True,
                require_forms=None):
        self.ids, self.T, self.pad, self.bs, self.seed = ids, targets, pad_id, batch_size, seed
        self.forms = np.asarray(forms, dtype=np.int16)
        self.balanced = balanced
        lengths = getattr(ids, "lengths", None)
        if lengths is None:
            lengths = np.array([len(ids[i]) for i in range(len(ids))], dtype=np.int64)
        self.lengths = np.asarray(lengths, dtype=np.int64)
        if balanced:
            self.present = [int(f) for f in np.unique(self.forms)]
            self.rows = {f: np.flatnonzero(self.forms == f) for f in self.present}
            # `require_forms` is the anchor/A4 contract: EVERY registered form must have rows, not
            # just every form already present. The old check (`for f in self.present`) could never
            # fire -- `self.present` is built FROM the forms that survived `np.unique`, so a form
            # missing entirely was silently absent from both the check and the sampler (Codex
            # 2026-09-05 whole-plan review).
            if require_forms:
                have = {FORMS[f] for f in self.present}
                missing = [f for f in require_forms if f not in have]
                if missing:
                    raise ValueError(f"forms with no rows: {missing}")
            self.n_batches = sum(int(np.ceil(len(v) / batch_size)) for v in self.rows.values())
        else:
            self.present = [-1]
            self.batches = {-1: D.length_buckets(ids, batch_size, seed=seed)}
            if not self.batches[-1]:
                raise ValueError("no full batches: corpus smaller than one batch")
            self.n_batches = len(self.batches[-1])

    def __len__(self):
        """The batches one full presentation of the corpus costs. With replacement the stream is
        endless, so this is a scale, not a bound -- `batch(k)` accepts any k."""
        return self.n_batches

    def _pick(self, k):
        F = len(self.present)
        c, j = divmod(int(k), F)
        if not self.balanced:
            bl = self.batches[-1]
            return -1, bl[c % len(bl)]
        f = self.present[int(np.random.default_rng([self.seed, c]).permutation(F)[j])] \
            if F > 1 else self.present[0]
        return f, _form_draw(self.rows[f], self.lengths, self.bs, self.seed, f, c)

    def batch(self, k):
        _f, idx = self._pick(k)
        x, m = D.collate(self.ids, idx, self.pad)
        return x, m, torch.from_numpy(np.ascontiguousarray(self.T[idx]))

    def realized_shares(self, n_batches):
        """-> {form: share of presented EXAMPLES} over the first `n_batches` steps, which is what
        §0b records. Counted from the rows actually drawn, so it is a measurement of the sampler
        and not a restatement of its design -- and so the unbalanced variant reports the per-form
        shares it produces rather than a single bucket.
        """
        idx = np.concatenate([self._pick(k)[1] for k in range(int(n_batches))])
        cnt = np.bincount(self.forms[idx], minlength=len(FORMS))
        tot = max(int(cnt.sum()), 1)
        return {FORMS[i]: round(float(c) / tot, 6) for i, c in enumerate(cnt) if c}


# --------------------------------------------------------------------------------- data cut ----

def data_cut_count(registry=None):
    """The registered post-screen unique-text count A2/A3/A4 are cut to, or None while §0b is
    open. It is `min` of the three corpora and cannot be computed before generation runs."""
    reg = registry or json.loads((REPO / "m10" / "screen_registry.json").read_text())
    return reg.get("data_cut", {}).get("unique_text_count")


def cut_arms(registry=None):
    """-> the arm names the registered cut APPLIES TO, read from the registry, not retyped.
    `A4/ANCHOR` is one registry row naming two arms."""
    reg = registry or json.loads((REPO / "m10" / "screen_registry.json").read_text())
    out = set()
    for row in reg.get("data_cut", {}).get("applies_to", []):
        out.update(x.strip() for x in str(row).split("/") if x.strip())
    return out


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


# ------------------------------------------------------------------------- cross-role guard ----

def _id_hashes(ids):
    """-> {blake2b-64 of the token-id bytes} for every row of a pretokenized corpus."""
    out = set()
    for i in range(len(ids)):
        out.add(hashlib.blake2b(np.asarray(ids[i], dtype=np.int32).tobytes(),
                                digest_size=8).digest())
    return out


def cross_role_collisions(q_ids, d_ids):
    """-> how many student inputs appear in BOTH roles.

    The same student input must never carry two different teacher targets. It can: the document
    role prepends `passage: `, so a QUERY whose text is literally "passage: X" tokenizes exactly
    like the DOCUMENT "X" -- one input, a query-prompt target and a raw-document target (Codex
    2026-09-05 finding 9). The test is a hash of the id bytes, so it costs one pass and no
    comparison of texts.
    """
    return len(_id_hashes(q_ids) & _id_hashes(d_ids))


def guard_cross_role(q_ids, d_ids, skip=False):
    """-> a report. Refuses before a step is taken; `skip=True` is for a smoke that asks."""
    if skip:
        return {"checked": False, "why": "explicitly skipped (smoke)"}
    n = cross_role_collisions(q_ids, d_ids)
    if n:
        raise SystemExit(f"REFUSED: {n:,} student inputs appear in BOTH the query and the "
                         f"document role, so the same input carries two different teacher "
                         f"targets. Remove them before training.")
    return {"checked": True, "collisions": 0, "n_query": len(q_ids), "n_document": len(d_ids)}


# ---------------------------------------------------------------------------------- the arm ----

def build_query_stream(arm_or_sources, tok, student, *, batch_size=32, seed=0, balanced=True,
                       max_len=512, prefix="", head_per_source=None, cut=None,
                       allow_uncut=False, require_forms=None, verbose=True):
    """-> (stream, manifest). Everything above, in the order an arm needs it.

    A registered CUT ARM (`screen_registry.data_cut.applies_to`) refuses to build a training
    stream while §0b has no `unique_text_count`: without the cut A2, A3 and A4 train at different
    volumes and family A's forms contrast is confounded with volume, which is the one thing it
    exists to separate. `allow_uncut=True` is the smoke escape and is RECORDED in the manifest
    (`uncut: true`), so an artifact can never look like a cut arm's.
    """
    names = ARM_SOURCES[arm_or_sources] if isinstance(arm_or_sources, str) else tuple(
        arm_or_sources)
    segs, man = load_segments(names, head_per_source=head_per_source, verbose=verbose)
    if head_per_source:
        man["head_per_source"] = head_per_source
    is_cut_arm = isinstance(arm_or_sources, str) and arm_or_sources in cut_arms()
    if is_cut_arm:
        cut = data_cut_count()
        if cut is None:
            if not allow_uncut:
                raise SystemExit(
                    f"arm {arm_or_sources!r} is a registered cut arm "
                    f"(m10/screen_registry.json data_cut.applies_to) and "
                    f"`data_cut.unique_text_count` is not registered yet, so this stream would "
                    f"train the FULL corpus. Register the count, or pass allow_uncut=True for a "
                    f"smoke -- which records `uncut: true` in the manifest.")
            man["uncut"] = True
    else:
        cut = data_cut_count() if cut == "registered" else cut
    if allow_uncut:
        # recorded even for a source-list call, which is how the smoke builds its corpus: the
        # escape must be visible in the artifact whether or not the arm was named (Codex
        # re-review 2026-09-05).
        man["uncut"] = True
    segs, cut_rep = apply_data_cut(segs, cut)
    man["data_cut"] = cut_rep
    man["is_cut_arm"] = is_cut_arm
    ids = tokenize_corpus(tok, segs, man, student, max_len=max_len, prefix=prefix,
                          verbose=verbose,
                          extra_ident={"data_cut": cut_rep, "head_per_source": head_per_source,
                                       "rescreen10": man.get("rescreen10"),
                                       "uncut": man.get("uncut", False)})
    stream = FormBalancedStream(ids, TargetView(segs), corpus_forms(segs),
                                pad_id=tok.pad_token_id, batch_size=batch_size, seed=seed,
                                balanced=balanced, require_forms=require_forms)
    man.update({"arm": arm_or_sources if isinstance(arm_or_sources, str) else None,
                "sources_used": list(names), "student": student, "max_len": max_len,
                "student_prefix": prefix, "batch_size": batch_size, "seed": seed,
                "balanced": balanced, "n_batches": len(stream),
                "n_tokens": int(ids.offs[-1]),
                "mean_tokens": round(float(ids.offs[-1]) / max(len(ids), 1), 2),
                "require_forms": list(require_forms) if require_forms else None})
    return stream, man


def doc_marker():
    """M9's registered document-role student marker, read from `m9/registry.json` rather than
    retyped. §Data: "document-role examples carry M9's fixed document-role marker"."""
    return json.loads((REPO / "m9" / "registry.json").read_text())["templates"]["doc_student"]


def _screened_doc_pool(n, seed, banned, margin=1.05, floor=2_000):
    """-> (texts, vectors, meta) for `n` documents that survive the M10 re-screen.

    `banned` must be the REAL computed ban set. An EMPTY set is treated exactly like a missing
    mask -- refused, not "nothing to remove" -- because the two are indistinguishable from here: a
    wiring bug that hands this an empty container by mistake looks identical to a genuinely clean
    6.15M-document pool, and only one of those is safe to serve silently (Codex 2026-09-05
    whole-plan review). The explicit unscreened path (`build_doc_stream(allow_unscreened=True)`)
    never calls this function at all.
    """
    if not banned:
        raise SystemExit(
            "REFUSED: the document ban set is empty, which this treats as \"mask missing\" "
            "rather than \"nothing to remove\" -- an unscreened pool must never be served "
            "silently. Build the M10 document re-screen mask (rescreen10.py --documents) and "
            "pass its real ban set, or call build_doc_stream(allow_unscreened=True) for an "
            "explicit smoke.")
    import data as m9data
    k = int(n * margin) + floor
    rows, meta = m9data.doc_pool_rows(k, seed)
    keep = np.array([int(r) not in banned for r in rows], dtype=bool)
    surv = rows[keep][:n]
    if len(surv) < n:
        raise SystemExit(f"the M10 document re-screen left {len(surv):,} of a {k:,}-row draw for "
                         f"a {n:,}-document stream -- widen `margin`")
    import pool as poolmod
    _index, vecs, _pmeta = poolmod.build()
    V = np.asarray(vecs[surv], dtype=np.float32)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    meta = {**meta, "n_drawn_before_rescreen": int(k), "n_removed_by_rescreen": int((~keep).sum()),
            "n_drawn": int(len(surv))}
    return m9data.row_texts(surv), V, meta


def build_doc_stream(n, tok, *, batch_size=32, seed=0, max_len=512, allow_unscreened=False,
                     verbose=True):
    """-> (stream, meta) for the document-role half of the mix, from the frozen M9 pool.

    The document marker is applied HERE, once. `data10.pretokenize` used to take no prefix at all,
    so every document reached the student as raw bytes -- the query-role policy -- while its
    teacher target was the raw-bytes document encoding. The teacher side was right; the student
    side dropped the marker the recipe names.

    The pool is RE-SCREENED against the M10 protected index first (`rescreen10`,
    instructions-m10.md:462: "matching pool documents are removed too"). The draw takes a margin
    and trims after the removal, so the arm still gets `n` documents and the survivors are still a
    uniform sample -- `doc_pool_rows` returns them in draw order, so a prefix of the survivors is
    a uniform sample of the survivors.
    """
    if allow_unscreened:
        # the explicit smoke escape: never routed through `_screened_doc_pool`, which now
        # refuses an empty/absent ban set outright rather than silently falling back to this.
        texts, vecs, meta = D.m9_doc_pool(n, seed=seed)
        meta["rescreen10"] = {"applied": False, "why": "allow_unscreened=True (smoke only)"}
    else:
        import rescreen10
        rows_banned, rep = rescreen10.doc_banned_rows(compute=False, verbose=verbose)
        banned = set(int(x) for x in rows_banned)
        screen_rep = {"applied": True, "n_banned_in_pool": len(banned),
                      "protected10": rescreen10._h(rescreen10.protected_ident())}
        texts, vecs, meta = _screened_doc_pool(n, seed, banned)
        meta["rescreen10"] = screen_rep
    pre = doc_marker()
    ids = D.pretokenize(tok, texts, max_len=max_len, prefix=pre, verbose=verbose, label="documents")
    meta = {**meta, "student_prefix": pre, "n": len(texts), "max_len": max_len}
    return D.Stream(ids, vecs, pad_id=tok.pad_token_id, batch_size=batch_size, seed=seed), meta


# arms that must see all 12 registered forms: A4 IS the anchor (`ARM_SOURCES`), so both names land
# on the same requirement.
REQUIRE_ALL_FORMS = {"A4", "ANCHOR"}


def resolve_arm_name(name, registry=None):
    """-> the registered arm name `name` names, resolving one `anchor_aliases` hop if needed.

    NEVER a source list: a list or tuple is refused here, before anything downstream can treat it
    as if it carried the guards a real arm name does. `F-winner` (the one alias that is prose, not
    an arm id -- resolved by `rules.F_selection_aware`, per `screen_registry.json`) is refused by
    name rather than silently resolving to nonsense.
    """
    if not isinstance(name, str):
        raise SystemExit(f"assemble_arm takes a registered arm name or an anchor_aliases key, "
                         f"never a source list: got {name!r}")
    if name in ARM_SOURCES:
        return name
    reg = registry or json.loads((REPO / "m10" / "screen_registry.json").read_text())
    aliases = reg.get("anchor_aliases", {})
    if name in aliases and not name.startswith("_"):
        val = aliases[name]
        if val in ARM_SOURCES:
            return val
        raise SystemExit(f"anchor_aliases[{name!r}] = {val!r} does not resolve to a registered "
                         f"arm id (e.g. 'F-winner' is resolved by rules.F_selection_aware, not "
                         f"this function) -- pass the resolved arm name instead")
    raise SystemExit(f"{name!r} is not a registered arm name (m10/screen_registry.json `arms`) "
                     f"or an `anchor_aliases` key")


def assemble_arm(arm_name, tok, student, *, batch_size=32, seed=0, max_len=512, n_docs=100_000,
                 prefix="", pattern="75/25", balanced=True, verbose=True, registry=None):
    """-> (batch_fn, manifest). The ONLY function a training launcher may use to build an arm's
    corpus (Codex 2026-09-05 whole-plan review: the cut, the masks, the 12-form requirement and
    the cross-role guard were each correct on their own but each OPTIONAL -- a source list or a
    flag bypassed every one of them; the fix is one mandatory path).

    Takes a registered arm name or an `anchor_aliases` key, never a source list
    (`resolve_arm_name`). Applies the registered cut for a cut arm, raising if
    `data_cut.unique_text_count` is not registered yet (no `allow_uncut` here -- that escape
    belongs to `build_query_stream` and a smoke alone). Requires the M10 re-screen masks for BOTH
    pools (no `allow_unscreened`) and cross-checks them against `results/m10_rescreen10.json` via
    `rescreen10.validate`. Requires the anchor/A4 arm to see all 12 forms, raising and naming any
    missing. Runs `guard_cross_role` on the two streams before returning. Everything applied is
    recorded in the manifest.
    """
    reg = registry or json.loads((REPO / "m10" / "screen_registry.json").read_text())
    name = resolve_arm_name(arm_name, reg)
    require_forms = FORMS if name in REQUIRE_ALL_FORMS else None
    q_stream, q_man = build_query_stream(name, tok, student, batch_size=batch_size, seed=seed,
                                        balanced=balanced, max_len=max_len, prefix=prefix,
                                        allow_uncut=False, require_forms=require_forms,
                                        verbose=verbose)
    doc_stream, doc_man = build_doc_stream(n_docs, tok, batch_size=batch_size, seed=seed,
                                          max_len=max_len, allow_unscreened=False, verbose=verbose)
    cross = guard_cross_role(q_stream.ids, doc_stream.ids)

    import rescreen10
    report = rescreen10.load_report()
    masks = {"protected10": rescreen10.protected_ident()}
    for seg in _m9_segments(screen=False):
        m, _rep = rescreen10.query_keep_mask(seg.texts, seg.name, compute=False)
        masks[seg.name] = m
    banned_rows, _rep = rescreen10.doc_banned_rows(compute=False, verbose=False)
    masks["documents"] = banned_rows
    rescreen10.validate(report, masks)

    man = {"arm": name, "requested_as": arm_name, "student": student, "batch_size": batch_size,
          "seed": seed, "max_len": max_len, "pattern": pattern, "n_docs": n_docs,
          "require_forms": list(require_forms) if require_forms else None,
          "query": q_man, "document": doc_man, "cross_role": cross,
          "rescreen10_report_validated": True}
    return D.batch_fn(q_stream, doc_stream, pattern=pattern), man


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
        texts, f, _ids, sman = source_texts(name)
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
