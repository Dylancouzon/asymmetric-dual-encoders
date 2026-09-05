"""Seed passages for the generated forms (§Data: Wikipedia + the approved pool corpora).

Two hard rules enforced here, not downstream:
  * **MS MARCO may never seed either half.** `msmarco-pos` is present in `work/train/stores/` as a
    research-only store; `ALLOWED_STORES` is an allow-list, never a deny-list, so it cannot leak in.
  * **A seed passage that exact- or near-matches the protected index is never used** (§Data), so
    every drawn passage passes `m7src/decontam` fingerprints against the protected query index
    before it is offered to the generator.

Topical routing exists because the on-form gate is only fair if the seed can carry the form: a
consumer-health question needs a medical passage. The routing is keyword-based over the passage
text, deterministic, and recorded with the sample; the build's stratification (§Data, step 8) is a
separate, larger draw and does not inherit these keyword lists.
"""
import hashlib, json, re, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "m7src"))

ALLOWED_STORES = ("hotpotqa-corpus", "squad-ctx", "mrtydi-docs", "esci-prod")
CACHE = REPO / "work" / "m10gen" / "seeds"

# Deterministic topical routing for the seven generated forms. `general` = no filter.
ROUTE = {
    "health": r"\b(disease|symptom|patient|treatment|therapy|diagnos|syndrome|infection|cancer|"
              r"vaccin|medication|clinical|surgery|chronic|virus|blood|immune)\w*\b",
    "finance": r"\b(bank|tax|invest|stock|econom|market|inflation|currency|revenue|mortgage|"
               r"pension|insurance|budget|debt|interest rate|fiscal|trade)\w*\b",
    "howto": r"\b(software|hardware|network|server|configur|install|protocol|device|engine|"
             r"circuit|driver|comput|machine|system|maintenance|repair)\w*\b",
    "argument": "general",
    "comparison": "general",
    "yesno": "general",
    "conversational": "general",
}
MIN_WORDS, MAX_WORDS = 40, 220

# Widened routing, FIXED AND COMMITTED BEFORE THE SCAN THAT USES IT (headroom rung 2,
# `m10/HEADROOM.md`; amends T2-3, logged in LEDGER §3 T2-4). The shortfall it addresses is ROUTER
# RECALL, not corpus thinness: English Wikipedia holds ~50K medicine articles, so 8,844 health
# seeds out of 5.23M intros is the 17-keyword list above under-matching. Widening raises recall at
# held precision; relaxing `min_score` would instead admit passages that merely mention "blood"
# twice, which is why it is a later rung. These lists are never tuned to the count they produce.
ROUTE_WIDE = dict(ROUTE)
ROUTE_WIDE["health"] = (
    r"\b(disease|symptom|patient|treatment|therap|diagnos|syndrome|infection|cancer|vaccin|"
    r"medication|clinical|surgery|chronic|virus|immune|drug|disorder|medical|medicine|hospital|"
    r"physician|illness|pain|dose|dosage|pregnan|injury|heart|lung|kidney|liver|mental health|"
    r"diabet|allerg|nutrition|epidemic|pandemic|antibiotic|tumou?r|inflammat|fever|nurse|"
    r"health|bacteri|blood)\w*\b")
ROUTE_WIDE["finance"] = (
    r"\b(bank|tax|invest|stock|econom|market|inflation|currency|revenue|mortgage|pension|"
    r"insurance|budget|debt|interest rate|fiscal|trade|loan|credit|saving|salary|wage|price|"
    r"income|profit|financ|monetary|bond|equity|capital|asset|accounting|audit|payment|money|"
    r"earnings|dividend|recession|gdp|currency|tariff|subsid)\w*\b")
# `esci-prod` is excluded from the topical scan: product listings cannot serve howto/health/finance.
TOPICAL_STORES = ("hotpotqa-corpus", "squad-ctx", "mrtydi-docs")


def _iter_store(name, limit=None):
    import mix
    ids, texts = mix.load_store(name)
    return ids, texts


def _score(pat, text, head_words=25):
    """Topicality = distinct keyword hits, with a hit in the article's opening (its title and
    first clause) worth more. Counting bare occurrences let a sports biography that mentions
    "cancer researcher" once win the `health` slot on the first 282 candidates scanned."""
    hits = {m.group(0).lower() for m in pat.finditer(text)}
    head = " ".join(text.split()[:head_words])
    return len(hits) + 2 * len(({m.group(0).lower() for m in pat.finditer(head)}))


def draw(forms_wanted, per_form=40, pool_size=400_000, seed=0, store="hotpotqa-corpus",
         screen=True, verbose=True, min_score=4):
    """-> {form: [(passage_id, passage_text)]}, all screened against the protected index.

    Two passes, because the forms are not equally choosy. Pass 1 scans the whole candidate pool
    and keeps the best-scoring passages for each TOPICAL form (health, finance, howto); pass 2
    fills the `general` forms from what is left. A one-pass first-fit filled every quota from the
    first 282 candidates and handed `health` a sports biography.
    """
    assert store in ALLOWED_STORES, f"{store} is not an approved seed store"
    ids, texts = _iter_store(store)
    rng = np.random.default_rng(seed)
    cand = rng.choice(len(texts), min(pool_size, len(texts)), replace=False)
    if verbose:
        print(f"  store {store}: {len(texts):,} docs, {len(cand):,} candidates", flush=True)

    pats = {f: (None if ROUTE[f] == "general" else re.compile(ROUTE[f], re.I))
            for f in forms_wanted}
    topical = [f for f in forms_wanted if pats[f] is not None]
    general = [f for f in forms_wanted if pats[f] is None]

    # ---- pass 1: score every candidate of the right length for each topical form
    ranked = {f: [] for f in topical}
    eligible = []
    for i in cand:
        t = texts[int(i)]
        n = len(t.split())
        if not (MIN_WORDS <= n <= MAX_WORDS):
            continue
        eligible.append(int(i))
        for f in topical:
            sc = _score(pats[f], t)
            if sc >= min_score:
                ranked[f].append((sc, int(i)))
    if verbose:
        print(f"  {len(eligible):,} length-eligible; topical candidates "
              + ", ".join(f"{f}={len(ranked[f])}" for f in topical), flush=True)

    if screen:
        import decontam
        q_ex, q_gram, whole, counts = decontam.protected_query_index()
        if verbose:
            print(f"  protected query index: {counts}", flush=True)

    n_screened = n_dropped = 0
    def ok(i):
        nonlocal n_screened, n_dropped
        if not screen:
            return True
        n_screened += 1
        # `query_grams` is documented query-side-only; on a >=8-word passage it reduces to
        # `ngram_hashes`, i.e. all the passage's 8-grams, which is exactly the test wanted here:
        # does this passage contain protected query text?
        if decontam.query_hits(texts[i], q_ex, q_gram, whole):
            n_dropped += 1
            return False
        return True

    kept, used = {f: [] for f in forms_wanted}, set()
    for f in topical:                      # highest-scoring first, ties broken by doc order
        for sc, i in sorted(ranked[f], key=lambda x: (-x[0], x[1])):
            if len(kept[f]) >= per_form:
                break
            if i in used or not ok(i):
                continue
            used.add(i)
            kept[f].append((f"{store}:{ids[i]}", texts[i]))
    for i in eligible:                     # pass 2: general forms, round-robin
        if all(len(kept[f]) >= per_form for f in general):
            break
        if i in used:
            continue
        f = min(general, key=lambda g: len(kept[g]))
        if len(kept[f]) >= per_form or not ok(i):
            continue
        used.add(i)
        kept[f].append((f"{store}:{ids[i]}", texts[i]))

    short = {f: per_form - len(v) for f, v in kept.items() if len(v) < per_form}
    if verbose:
        print(f"  screened {n_screened:,}, dropped {n_dropped:,} on the protected index; "
              + ", ".join(f"{f}={len(v)}" for f, v in kept.items())
              + (f"  SHORT: {short}" if short else ""), flush=True)
    # Fable 2026-09-04: if a form's topical-candidate count in this pool is small, the FULL store
    # cannot supply the build's ~28.6K seeds at this floor, and the smoke's seeds are far stronger
    # than the build's would be. Extrapolated to the whole store, reported, action at build time.
    scale = len(texts) / max(len(cand), 1)
    projected = {f: int(len(ranked[f]) * scale) for f in topical}
    if verbose:
        print(f"  projected topical seeds in the full store: {projected}", flush=True)
    return kept, dict(store=store, pool_size=int(len(cand)), seed=seed, min_score=min_score,
                      store_size=len(texts), projected_topical_full_store=projected,
                      length_eligible=len(eligible),
                      topical_candidates={f: len(ranked[f]) for f in topical},
                      screened=n_screened, dropped_protected=n_dropped, short=short,
                      screen="protected QUERY index (six+dev+untouched-final); the protected "
                             "DOCUMENT screen runs on the build manifest at step 8")


# Bumped whenever the screen's SCOPE changes (e.g. COV joins the protected index). A cached
# draw made under an older scope can then never be served -- the old blocker was a cache key of
# `smoke-{per_form}-{forms}`, which ignored store, seed, min_score, pool size and the screen.
SCREEN_VERSION = "2026-09-04-protected-queries-only"


def _key(forms_wanted, per_form, kw):
    ident = dict(forms=sorted(forms_wanted), per_form=per_form, screen=SCREEN_VERSION,
                 store=kw.get("store", "hotpotqa-corpus"), seed=kw.get("seed", 0),
                 pool_size=kw.get("pool_size", 400_000), min_score=kw.get("min_score", 4),
                 screened=kw.get("screen", True), route={f: ROUTE[f] for f in sorted(forms_wanted)},
                 lengths=[MIN_WORDS, MAX_WORDS])
    h = hashlib.blake2b(json.dumps(ident, sort_keys=True).encode(), digest_size=8).hexdigest()
    return h, ident


def cached(forms_wanted, per_form=40, **kw):
    """Cache keyed on EVERYTHING that defines the draw, including the screen version, so a
    cached seed set can never outlive the screen it passed."""
    CACHE.mkdir(parents=True, exist_ok=True)
    h, ident = _key(forms_wanted, per_form, kw)
    p = CACHE / f"seeds-{h}.json"
    if p.exists():
        blob = json.loads(p.read_text())
        assert blob["ident"] == ident, f"{p}: cache key collision"
        return {f: [tuple(x) for x in v] for f, v in blob["seeds"].items()}, blob["meta"]
    kept, meta = draw(forms_wanted, per_form=per_form, **kw)
    meta["screen_version"] = SCREEN_VERSION
    p.write_text(json.dumps({"ident": ident, "seeds": kept, "meta": meta}))
    return kept, meta


if __name__ == "__main__":
    fs = sys.argv[1:] or list(ROUTE)
    kept, meta = cached(fs, per_form=int(__import__("os").environ.get("PER_FORM", 40)))
    print(json.dumps(meta, indent=1))
    for f, v in kept.items():
        print(f"\n=== {f} ({len(v)}) ===\n{v[0][1][:300] if v else 'NONE'}")
