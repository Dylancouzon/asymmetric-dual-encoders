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

# Widened routing, v2. **v1 is WITHDRAWN as defective, not revised to hit a number.** v1 kept the
# original form `\b(alt|alt|...)\w*\b`, which appends a wildcard to EVERY alternative, so `pain`
# matched "paints", `nurse` matched "nursery", `chronic` matched "chronicle", and on the finance
# side `capital` matched "capital city", `credit` matched "credited", `wage` matched "wages war"
# and `bond` matched a surname. A painter and a truck-painting business were routed as `health`
# seeds. v1's health count (42,380) is therefore part recall gain and part precision loss, and is
# withdrawn with it. v2 lists explicit word FORMS and anchors both ends, so a match is the word.
#
# Registered with v2, because v1's failure was that precision was asserted rather than measured:
# before the widened routing is used for a build draw, a sample of passages it newly admits is
# judged on-topic by an independent Fable subagent, and the widening is kept only if that rate
# holds. The gate and its result live in `m10/HEADROOM.md`; the lists are never tuned to the count.
def _words(*forms):
    return r"\b(?:" + "|".join(forms) + r")\b"


HEALTH_WORDS = (
    "disease", "diseases", "symptom", "symptoms", "patient", "patients", "treatment",
    "treatments", "therapy", "therapies", "therapeutic", "diagnosis", "diagnostic", "diagnose",
    "diagnosed", "syndrome", "syndromes", "infection", "infections", "infectious", "cancer",
    "cancers", "carcinoma", "vaccine", "vaccines", "vaccination", "vaccinated", "medication",
    "medications", "clinical", "clinic", "clinics", "surgery", "surgeries", "surgical",
    "surgeon", "surgeons", "chronic", "chronically", "virus", "viruses", "viral", "immune",
    "immunity", "immunology", "drug", "drugs", "disorder", "disorders", "medical", "medicine",
    "medicines", "hospital", "hospitals", "physician", "physicians", "illness", "illnesses",
    "pain", "pains", "painful", "dose", "doses", "dosage", "pregnancy", "pregnant", "injury",
    "injuries", "cardiac", "cardiovascular", "lung", "lungs", "pulmonary", "kidney", "kidneys",
    "renal", "hepatic", "psychiatric", "psychiatry", "diabetes", "diabetic", "allergy",
    "allergies", "allergic", "nutrition", "nutritional", "epidemic", "pandemic", "antibiotic",
    "antibiotics", "tumor", "tumour", "tumors", "tumours", "inflammation", "inflammatory",
    "fever", "fevers", "nurse", "nurses", "nursing", "health", "healthcare", "healthy",
    "bacteria", "bacterial", "blood",
)
FINANCE_WORDS = (
    "bank", "banks", "banking", "tax", "taxes", "taxation", "invest", "invests", "investment",
    "investments", "investor", "investors", "stock", "stocks", "economy", "economic",
    "economics", "market", "markets", "inflation", "currency", "currencies", "revenue",
    "revenues", "mortgage", "mortgages", "pension", "pensions", "insurance", "insurer",
    "budget", "budgets", "debt", "debts", "fiscal", "loan", "loans", "creditor", "creditors",
    "savings", "salary", "salaries", "wages", "income", "incomes", "profit", "profits",
    "profitable", "financial", "finance", "finances", "financing", "monetary", "equity",
    "asset", "assets", "accounting", "audit", "audits", "payment", "payments", "money",
    "earnings", "dividend", "dividends", "recession", "gdp", "tariff", "tariffs", "subsidy",
    "subsidies", "shareholder", "shareholders", "capitalism", "bankruptcy", "interest rate",
    "interest rates",
)
ROUTE_WIDE = dict(ROUTE)
ROUTE_WIDE["health"] = _words(*HEALTH_WORDS)
ROUTE_WIDE["finance"] = _words(*FINANCE_WORDS)
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
         screen=True, verbose=True, min_score=4, route=None):
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

    # T2-3's registered routing stays the default: ROUTE_WIDE **FAILED its registered
    # precision gate** (health marginal 28% on-topic, finance 38%, against >= 80%). It is
    # available by explicit argument for measurement, and is not adopted for a build draw.
    route = route if route is not None else ROUTE
    pats = {f: (None if route[f] == "general" else re.compile(route[f], re.I))
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
        # The M10 protected index, not M7's: §Data requires the admitted COV queries AND
        # documents to be in it before any seed is drawn (`m10src/protected10`, cached).
        import protected10
        q_ex, q_gram, whole, counts = protected10.build(verbose=verbose)
        if verbose:
            print(f"  protected index ({protected10.VERSION}): {counts}", flush=True)

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
                      route="ROUTE_WIDE" if route is ROUTE_WIDE else "ROUTE",
                      store_size=len(texts), projected_topical_full_store=projected,
                      length_eligible=len(eligible),
                      topical_candidates={f: len(ranked[f]) for f in topical},
                      screened=n_screened, dropped_protected=n_dropped, short=short,
                      screen="protected QUERY index (six+dev+untouched-final); the protected "
                             "DOCUMENT screen runs on the build manifest at step 8")


# Bumped whenever the screen's SCOPE changes (e.g. COV joins the protected index). A cached
# draw made under an older scope can then never be served -- the old blocker was a cache key of
# `smoke-{per_form}-{forms}`, which ignored store, seed, min_score, pool size and the screen.
SCREEN_VERSION = "2026-09-05-protected10(six+dev+reserved-q+COV-q+d)+ROUTE(T2-3)"


def _key(forms_wanted, per_form, kw):
    ident = dict(forms=sorted(forms_wanted), per_form=per_form, screen=SCREEN_VERSION,
                 store=kw.get("store", "hotpotqa-corpus"), seed=kw.get("seed", 0),
                 pool_size=kw.get("pool_size", 400_000), min_score=kw.get("min_score", 4),
                 screened=kw.get("screen", True),
                 # the route ACTUALLY passed, not `ROUTE` unconditionally: a draw made under
                 # `ROUTE_WIDE` must not be servable from a cache keyed as if it were `ROUTE`
                 route={f: (kw.get("route") or ROUTE)[f] for f in sorted(forms_wanted)},
                 lengths=[MIN_WORDS, MAX_WORDS])
    h = hashlib.blake2b(json.dumps(ident, sort_keys=True).encode(), digest_size=8).hexdigest()
    return h, ident


def supply(forms=("health", "finance", "howto"), stores=None, min_score=4, route=None,
           verbose=True):
    """Realized topical seed supply per form over the FULL approved stores — not a projection.

    Ordering matters and is fixed: a passage matching two forms is claimed by whichever form is
    processed first, so `forms` is given in priority order (health first — `nfcorpus` and
    `trec-covid`, two of the four clean-4 datasets, are biomedical). Cross-store text dedup by
    exact fingerprint, because the three Wikipedia slices overlap.
    """
    import decontam
    stores = stores or TOPICAL_STORES
    route = route or ROUTE_WIDE
    pats = {f: re.compile(route[f], re.I) for f in forms}
    counts = {f: 0 for f in forms}
    counts_nolen = {f: 0 for f in forms}
    seen, n_docs, n_len_ok = set(), 0, 0
    for store in stores:
        ids, texts = _iter_store(store)
        if verbose:
            print(f"  {store}: {len(texts):,} docs", flush=True)
        for t in texts:
            n_docs += 1
            k = int(decontam.exact_u64(t))
            if k in seen:
                continue
            seen.add(k)
            nw = len(t.split())
            length_ok = MIN_WORDS <= nw <= MAX_WORDS
            n_len_ok += length_ok
            for f in forms:                     # priority order; first match claims the passage
                if _score(pats[f], t) >= min_score:
                    counts_nolen[f] += 1
                    if length_ok:
                        counts[f] += 1
                    break
    out = dict(stores=list(stores), min_score=min_score, route="ROUTE_WIDE" if route is ROUTE_WIDE
               else "ROUTE", order=list(forms), n_docs_scanned=n_docs,
               n_unique_after_dedup=len(seen), n_length_eligible=n_len_ok,
               seeds_in_length_range=counts, seeds_ignoring_length=counts_nolen,
               length_window=[MIN_WORDS, MAX_WORDS])
    if verbose:
        print(f"  scanned {n_docs:,}, unique {len(seen):,}, length-eligible {n_len_ok:,}")
        for f in forms:
            print(f"    {f:10s} {counts[f]:8,d} in range   ({counts_nolen[f]:,} ignoring length)")
    return out


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
