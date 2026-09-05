"""`arxiv-title` — the registered descriptive surface (§Surfaces, Codex B8), and its draw.

The draw is registered verbatim and is executed here, not chosen here: ID universe = every
record's `id` with the version suffix stripped, deduplicated, **sorted lexicographically**;
`numpy.random.default_rng(0).choice(N, 100_000, replace=False)` in that order; the **first 2,000
drawn are the queries** (title -> its own abstract) and the other 98,000 are distractors. Every
VERSION of a drawn paper is excluded from every training role, the harvest is fingerprint-screened
against the drawn titles and abstracts, and the set joins the protected index before any
extraction. `arxiv-title` is DESCRIPTIVE: it reads, it reports, it triggers no action
(`m10/screen_registry.json` `descriptive_contrasts`).

Source: the registered Kaggle artifact `Cornell-University/arxiv`, file
`arxiv-metadata-oai-snapshot.json`. Its sha256 and Kaggle version go in `m10/LEDGER.md` §0b before
the draw is used, which is what makes the draw reproducible rather than merely deterministic.

The credential is read from the environment or `~/.kaggle/access_token`, both outside the repo;
nothing here writes it anywhere.
"""
import hashlib, json, os, sys, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m10src"):
    sys.path.insert(0, str(REPO / p))
WORK = REPO / "work" / "m10arxiv"
ZIP = WORK / "arxiv.zip"
MEMBER = "arxiv-metadata-oai-snapshot.json"
N_DRAW, N_QUERIES, SEED = 100_000, 2_000, 0

import numpy as np


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while (b := fh.read(chunk)):
            h.update(b)
    return h.hexdigest()


def _base_id(i):
    """The version suffix stripped: `1234.5678v3` -> `1234.5678`. Old-style ids
    (`hep-th/9901001`) carry no `vN` in this field, and are returned unchanged."""
    j = i.rfind("v")
    return i[:j] if j > 0 and i[j + 1:].isdigit() else i


def stream_records(verbose=True):
    """One dict per line, straight out of the zip: 5.5 GB uncompressed, never materialised."""
    with zipfile.ZipFile(ZIP) as z:
        with z.open(MEMBER) as fh:
            for n, line in enumerate(fh):
                if verbose and n and n % 500_000 == 0:
                    print(f"  {n:,} records", flush=True)
                yield json.loads(line)


def build(verbose=True):
    """-> the draw record. Deterministic given the artifact; the artifact is pinned by sha256.

    TWO passes on purpose. Holding title and abstract for every paper while streaming is ~4 GB of
    text for 2,000 queries' worth of use; pass 1 collects only the id universe, pass 2 collects
    text for the 2,000 drawn queries alone.
    """
    WORK.mkdir(parents=True, exist_ok=True)
    ids = set()
    n_rec = 0
    if verbose:
        print("pass 1: the id universe", flush=True)
    for r in stream_records(verbose):
        n_rec += 1
        ids.add(_base_id(r["id"]))
    universe = sorted(ids)                       # lexicographic, as registered
    n = len(universe)
    sel = np.random.default_rng(SEED).choice(n, N_DRAW, replace=False)
    drawn = [universe[int(i)] for i in sel]      # IN DRAW ORDER: the first 2,000 are the queries
    queries, distractors = drawn[:N_QUERIES], drawn[N_QUERIES:]

    want = set(queries)
    text = {}
    if verbose:
        print(f"pass 2: text for the {len(want):,} drawn queries", flush=True)
    for r in stream_records(verbose):
        b = _base_id(r["id"])
        if b in want:
            text[b] = {"id": b, "title": (r.get("title") or "").strip(),
                       "abstract": (r.get("abstract") or "").strip()}
    missing = [q for q in queries if q not in text]
    if missing:
        raise SystemExit(f"{len(missing)} drawn ids have no record: {missing[:5]}")

    rec = {
        "source": {"kaggle": "Cornell-University/arxiv", "member": MEMBER,
                   "zip_sha256": sha256(ZIP), "zip_bytes": ZIP.stat().st_size},
        "n_records": n_rec, "n_unique_base_ids": n,
        "seed": SEED, "n_drawn": N_DRAW, "n_queries": N_QUERIES,
        "rule": "ids version-stripped, deduplicated, sorted lexicographically; "
                "default_rng(0).choice(N, 100000, replace=False); first 2,000 drawn are queries",
        "role": "DESCRIPTIVE surface only (arxiv-title); triggers no action",
        "first_5_query_ids": queries[:5],
        "empty_titles": sum(1 for q in queries if not text[q]["title"]),
        "empty_abstracts": sum(1 for q in queries if not text[q]["abstract"]),
    }
    (WORK / "arxiv_draw.json").write_text(json.dumps(rec, indent=1))
    (WORK / "arxiv_drawn.json").write_text(json.dumps(
        {"queries": [text[q] for q in queries], "distractor_ids": distractors}))
    # every VERSION of a drawn paper is excluded from every training role: the base ids are the
    # exclusion key, and a version suffix can never evade it
    (WORK / "arxiv_excluded_base_ids.json").write_text(json.dumps(sorted(drawn)))
    if verbose:
        print(json.dumps(rec, indent=1), flush=True)
    return rec


if __name__ == "__main__":
    build()
