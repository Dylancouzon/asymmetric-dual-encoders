"""The M10 protected index = M7's protected queries + the admitted COV queries AND documents.

§Data: seeds and every generated, harvested or PAQ query are screened against "the protected
index (six + dev + reserved + LoTTE + admitted COV queries and documents)". `m7src/decontam`
supplies the first part and is not modified; this adds the COV part and caches the fingerprints,
because rebuilding them costs a pass over 404K BRIGHT documents.

The cache is keyed on the admitted component list and revisions, so admitting a component
invalidates it rather than silently serving a stale index. `seeds.SCREEN_VERSION` must name the
version in use, so a seed draw can never outlive the index it passed.

Not covered, and recorded as `m10/LEDGER.md` §3 W4: reserved-set DOCUMENT fingerprints, which do
not exist and whose construction would open the reserved corpora.
"""
import hashlib, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m10src"):
    sys.path.insert(0, str(REPO / p))
CACHE = REPO / "work" / "m10cov" / "protected10"

import numpy as np
import decontam
from cov_admit import COMPONENTS

VERSION = "2026-09-04-six+dev+reserved-queries+COV(q+d)"


def _ident():
    return {"version": VERSION,
            "components": sorted((n, rev) for cs in COMPONENTS.values() for n, _r, rev in cs)}


def build(verbose=True):
    """-> (q_ex set, q_gram sorted array, q_whole index, counts). Cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    ident = _ident()
    h = hashlib.blake2b(json.dumps(ident, sort_keys=True).encode(), digest_size=8).hexdigest()
    npz, meta_p = CACHE / f"idx-{h}.npz", CACHE / f"idx-{h}.json"

    q_ex, q_gram, q_whole, counts = decontam.protected_query_index()
    cov_short = []                      # COV QUERIES of 4-7 words, for the containment index
    if npz.exists():
        z = np.load(npz)
        cov_ex, cov_gram = z["ex"], z["gram"]
        cov_short_texts = json.loads((CACHE / f"short-{h}.json").read_text())
        counts = json.loads(meta_p.read_text())["counts"]
    else:
        from cov_screen import load_component
        ex, grams, counts = [], [], dict(counts)
        for family, comps in COMPONENTS.items():
            for name, repo, rev in comps:
                qs, ds = load_component(name, repo, rev)
                cov_short += [t for t in qs
                              if decontam.SHORT_NGRAM <= len(decontam.norm_words(t))
                              < decontam.NGRAM]
                for role, texts in (("q", qs), ("d", ds)):
                    ex += [int(decontam.exact_u64(t)) for t in texts]
                    # documents contribute all their 8-grams; queries contribute query-side grams
                    f = decontam.query_grams if role == "q" else decontam.all_grams
                    grams.append(np.unique(np.concatenate([f(t) for t in texts]))
                                 if texts else np.zeros(0, np.uint64))
                    counts[f"COV:{name}:{role}"] = len(texts)
                if verbose:
                    print(f"  {name}: {len(qs):,} q + {len(ds):,} d fingerprinted", flush=True)
        cov_ex = np.unique(np.asarray(ex, dtype=np.uint64))
        cov_gram = np.unique(np.concatenate(grams)) if grams else np.zeros(0, np.uint64)
        np.savez_compressed(npz, ex=cov_ex, gram=cov_gram)
        (CACHE / f"short-{h}.json").write_text(json.dumps(cov_short))
        cov_short_texts = cov_short
        meta_p.write_text(json.dumps({"ident": ident, "counts": counts,
                                      "n_exact": int(cov_ex.size),
                                      "n_gram": int(cov_gram.size)}, indent=1))
    merged_ex = set(q_ex) | set(int(x) for x in cov_ex)
    merged_gram = np.unique(np.concatenate([q_gram, cov_gram]))
    # A 4-7-word COV query embedded VERBATIM in a 45-word candidate matches nothing on grams --
    # a long candidate emits only 8-grams -- so it needs the containment index, which M7's
    # `protected_query_index` populates from the protected queries ONLY. Without this the COV
    # queries were half-screened (Codex 2026-09-05).
    cov_whole = decontam.short_whole_index(cov_short_texts)
    merged_whole = dict(q_whole)
    for k, v in cov_whole.items():
        merged_whole[k] = np.unique(np.concatenate([merged_whole[k], v])) \
            if k in merged_whole else v
    q_whole = merged_whole
    counts["COV:short-queries-in-containment-index"] = len(cov_short_texts)
    if verbose:
        print(f"  protected10 {VERSION}: {len(merged_ex):,} exact keys, "
              f"{merged_gram.size:,} grams", flush=True)
    return merged_ex, merged_gram, q_whole, counts


def hits(text, idx):
    """R1 test against the M10 protected index. -> 'exact' | 'near' | 'contains' | None."""
    q_ex, q_gram, q_whole, _ = idx
    return decontam.query_hits(text, q_ex, q_gram, q_whole)


if __name__ == "__main__":
    idx = build()
    print(json.dumps({k: v for k, v in idx[3].items()}, indent=1))
