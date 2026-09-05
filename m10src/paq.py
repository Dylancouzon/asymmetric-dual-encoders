"""PAQ as query text — decision 4, from Facebook's OFFICIAL release, never an HF mirror.

`instructions-m10.md`:191 registers "from Facebook's official release (never the unofficial HF
mirror); 1.0M uniform sample in the build (seed 0, file hashes pinned), 4.037M in the volume-control
screen arm A2 only; attribution recorded". A wrapper tag on someone's HF mirror is not evidence of
a licence, so the artifact is fetched from `dl.fbaipublicfiles.com` and its hash pinned here.

**Licence, primary source.** The tarball ships the grant itself: `PAQ/LICENSE` is the CC BY-SA 3.0
Unported legal code, and the release table states "The PAQ QA-pairs and metadata is licensed under
CC-BY-SA". The CC-BY-NC on the repo covers the GENERATION CODE, which is not used — we consume the
released pairs only, which `research/m7-data-licensing.md`:16 already recorded as permitted
("Using released pairs OK (share-alike); regenerating more is NC-restricted"). Model-card
attribution is required, as for the other CC BY-SA sources.

**Only the `question` field is read.** The answers are never used: this is query text for a
distillation target, and there is no supervised objective that could consume an answer.

**The draw.** Uniform without replacement over the pinned population (seed 0), then normalized-text
dedup, then the protected-index query screen with matches REMOVED — PAQ is generated over Wikipedia
and NQ/TriviaQA/HotpotQA are Wikipedia questions, so overlap with protected queries is expected and
must be measured rather than assumed. A margin is drawn so the quota still fills after removals, and
**the survivor list is SHUFFLED before truncation**: a sorted-index sequential read makes survivors
arrive in file order, so taking the first n would be a positional sample, the same defect that was
found and fixed in `harvest.draw` (see its note and `m10/LEDGER.md` §Harvest amendment 2026-09-05).

The 1.0M build sample is a **subset** of the 4.037M A2 sample. A2 is the volume control, so "the
build with less PAQ volume" is the coherent nesting; it is an implementation choice, stated here
because nothing registered specifies it.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))
SRC = REPO / "work" / "m10paq" / "PAQ" / "PAQ.filtered.jsonl"
TAR = REPO / "work" / "m10paq" / "PAQ.tar.gz"
OUT = REPO / "work" / "m10paq"

import numpy as np

# pinned 2026-09-05 from https://dl.fbaipublicfiles.com/paq/v1/PAQ.tar.gz
TAR_SHA256 = "177eefb2ddf8ab46a8d2248c058d5be52a4f2ce7614e55c1696f69fd0fe051c3"
TAR_BYTES = 1_447_064_073
POPULATION = 64_875_601                 # lines in PAQ.filtered.jsonl, counted 2026-09-05
A2_QUOTA, BUILD_QUOTA = 4_037_000, 1_000_000
SEED, MARGIN = 0, 1.30


def verify(strict=True):
    """The pinned artifact, or nothing. A silently different file is the whole risk here."""
    n = TAR.stat().st_size
    rec = {"tar": str(TAR), "bytes": n, "bytes_expected": TAR_BYTES}
    if n != TAR_BYTES and strict:
        raise SystemExit(f"{TAR}: {n} bytes, expected {TAR_BYTES}")
    h = hashlib.sha256()
    with TAR.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    rec["sha256"] = h.hexdigest()
    rec["sha256_expected"] = TAR_SHA256
    if rec["sha256"] != TAR_SHA256 and strict:
        raise SystemExit(f"{TAR}: sha256 {rec['sha256']} != pinned {TAR_SHA256}")
    rec["verified"] = rec["sha256"] == TAR_SHA256 and n == TAR_BYTES
    return rec


def _norm(q):
    return " ".join(q.split()).lower()


def read_rows(rows, verbose=True):
    """-> [question] for the given SORTED line indices, one sequential pass over 6.2 GB."""
    want, out, k, t0 = set(int(r) for r in rows), [], 0, time.time()
    with SRC.open() as fh:
        for i, line in enumerate(fh):
            if i in want:
                out.append(json.loads(line)["question"])
                k += 1
                if verbose and k % 500_000 == 0:
                    print(f"    read {k:,}/{len(want):,} ({time.time() - t0:.0f}s)", flush=True)
    return out


def draw(a2_quota=A2_QUOTA, build_quota=BUILD_QUOTA, margin=MARGIN, pilot=None, verbose=True):
    """-> (a2, build, report). Uniform, deduped, protected-screened, matches removed."""
    import protected10
    t0 = time.time()
    quota = pilot or a2_quota
    n_draw = min(int(margin * quota), POPULATION)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(POPULATION, size=n_draw, replace=False))
    if verbose:
        print(f"  drawing {n_draw:,} of {POPULATION:,} (seed {SEED}, margin {margin})", flush=True)
    qs = read_rows(idx, verbose=verbose)

    seen, deduped, n_dup = set(), [], 0
    for q in qs:
        k = _norm(q)
        if k in seen:
            n_dup += 1
            continue
        seen.add(k)
        deduped.append(q)
    if verbose:
        print(f"  dedup: {n_dup:,} of {len(qs):,} removed -> {len(deduped):,}", flush=True)

    t1 = time.time()
    ix = protected10.build(verbose=verbose)
    # `hits` returns the REASON ('exact' | 'near' | 'contains' | None), not a bool -- keep the
    # breakdown, since "how PAQ overlaps protected queries" is the interesting part of this number.
    hit = [protected10.hits(q, ix) for q in deduped]
    survivors = [q for q, h in zip(deduped, hit) if h is None]
    n_screen = sum(1 for h in hit if h is not None)
    by_reason = {}
    for h in hit:
        if h is not None:
            by_reason[h] = by_reason.get(h, 0) + 1
    if verbose:
        print(f"  protected screen: {n_screen:,} of {len(deduped):,} removed "
              f"({n_screen / max(len(deduped), 1):.2%}, {time.time() - t1:.0f}s) -> "
              f"{len(survivors):,}  by reason {by_reason}", flush=True)

    rng.shuffle(survivors)          # never take a positional prefix of a file-order list
    rep = {"source": "https://dl.fbaipublicfiles.com/paq/v1/PAQ.tar.gz",
           "licence": "CC BY-SA 3.0 (PAQ/LICENSE in the tarball); attribution required",
           "fields_used": ["question"], "population": POPULATION, "seed": SEED,
           "margin": margin, "drawn": n_draw, "dedup_removed": n_dup,
           "protected_screen_removed": n_screen,
           "protected_screen_by_reason": by_reason, "survivors": len(survivors),
           "pilot": pilot, "seconds": round(time.time() - t0, 1)}
    if pilot:
        rep["rates"] = {"dedup": n_dup / max(len(qs), 1),
                        "screen": n_screen / max(len(deduped), 1),
                        "survival": len(survivors) / max(len(qs), 1)}
        (OUT / "paq_pilot.json").write_text(json.dumps(rep, indent=1))
        if verbose:
            print(json.dumps(rep, indent=1))
        return None, None, rep

    if len(survivors) < a2_quota:
        raise SystemExit(f"{len(survivors):,} survived a {n_draw:,} draw for a {a2_quota:,} "
                         f"quota -- widen `margin` rather than take a short sample")
    a2 = survivors[:a2_quota]
    build = a2[:build_quota]                 # nested by construction; see the module docstring
    for name, sample in (("a2", a2), ("build", build)):
        p = OUT / f"paq_{name}.jsonl"
        with p.open("w") as fh:
            for q in sample:
                fh.write(json.dumps({"question": q}) + "\n")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rep[name] = {"n": len(sample), "path": str(p), "sha256": h,
                     "bytes": p.stat().st_size}
    rep["nesting"] = "build is the first build_quota of a2 after the shuffle; a2 >= build"
    rep["artifact"] = verify()
    (OUT / "paq_draw.json").write_text(json.dumps(rep, indent=1))
    if verbose:
        print(json.dumps({k: v for k, v in rep.items() if k != "artifact"}, indent=1))
    return a2, build, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=None,
                    help="measure dedup/screen rates on a small draw before committing a margin")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if a.verify_only:
        print(json.dumps(verify(), indent=1))
    else:
        draw(pilot=a.pilot)
