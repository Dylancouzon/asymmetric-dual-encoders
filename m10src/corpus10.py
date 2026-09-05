"""Step 8: turn generated and harvested strings into the immutable build manifest.

The screens against the protected index and the six's documents already exist (`protected10.hits`,
`decontam.Inverted` streamed as in `harvest.draw`). What was missing, and is here, is the logic
that is specific to this step:

1. **The FORMS-12 hold-out** — "500 seed documents per form are set aside FIRST; queries generated
   or harvested from them are never trained on" (`instructions-m10.md`:454). *First* is the whole
   point: hold out the SEEDS, then generate, so no query can leak by having been produced before
   the split.
2. **The own-seed-passage span screen** — "any word-5-gram shared with the query's own seed
   passage (a copied span is not a query) — **for harvested text this screen is against the source
   document minus the harvested span itself, since a harvested title IS a span of its document**"
   (:447). Without the exclusion this screen rejects every harvested string by construction.
3. **A8 gate 1, diversity per form** (:461) — and note it uses a **different threshold from the
   decontamination screens**: near-duplicate is **≥ 16/32** sketch overlap with an EARLIER query of
   the same form, where the decontamination screens are **≥ 8/32**. Conflating the two would either
   gut the corpus or wave through template collapse, so `test_corpus10` asserts both constants.

A8 gate 2 (the MS MARCO distribution-overlap diagnostic) needs a stella encode and has **no
action**, so it is not gating logic and does not live here.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import numpy as np

import decontam

HOLDOUT_PER_FORM = 500          # FORMS-12
A8_NEAR_DUP_SHARE = 16          # A8 gate 1 -- NOT decontam.DUP_SHARE (8), which the screens use
A8_MAX_NEAR_DUP_RATE = 0.25     # above this the form keeps only its representatives
A8_MIN_RETAINED = 50_000        # below this the form is dropped from the build
SPAN_K = 5                      # "any word-5-gram shared with the query's own seed passage"


def holdout_seed_ids(seed_ids, n=HOLDOUT_PER_FORM, seed=0):
    """-> (held, kept) as sorted lists. Called BEFORE generation, per form.

    A uniform draw, not a prefix: seed stores arrive in corpus order (Wikipedia dump order, whose
    prefix `m10/LEDGER.md` §T2-5 measured as a 5x distortion), so a prefix hold-out would reserve a
    systematically different population from the one trained on.
    """
    ids = sorted(set(seed_ids))
    if len(ids) <= n:
        raise SystemExit(f"{len(ids)} seed ids cannot yield a {n}-document hold-out")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ids), size=n, replace=False)
    held = {ids[int(i)] for i in pick}
    return sorted(held), [i for i in ids if i not in held]


def _kgrams(words, k=SPAN_K):
    if len(words) < k:
        return set()
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def copied_span(query, source_text, exclude_span=None, k=SPAN_K):
    """True if the query shares a word-`k`-gram with its OWN seed passage -- a copied span is not
    a query. `exclude_span` is the harvested string itself, removed from the source first, because
    a harvested title IS a span of its document and would otherwise always self-match.

    The exclusion removes the span's k-grams, not the span's characters: deleting a substring would
    splice its neighbours into k-grams that never existed in the source.
    """
    qw = decontam.norm_words(query)
    sw = decontam.norm_words(source_text)
    src = _kgrams(sw, k)
    if exclude_span:
        src -= _kgrams(decontam.norm_words(exclude_span), k)
    return bool(_kgrams(qw, k) & src)


def near_dup_gate(queries, share=A8_NEAR_DUP_SHARE):
    """A8 gate 1 for ONE form. -> (representatives, exact_deduped, report).

    Sequential and order-dependent BY REGISTRATION: "a query is a near-duplicate if its
    word-8-gram bottom-32 sketch matches >= 16/32 with an EARLIER query of the same form (the
    earlier one is the representative)". So the first occurrence is always the representative.

    `representatives` is what the form becomes IF the rate is above 25%; `exact_deduped` is what
    it stays otherwise. The caller applies the action, since it also depends on `A8_MIN_RETAINED`.
    """
    seen_exact, exact_dups = set(), 0
    order = []
    for i, q in enumerate(queries):
        h = int(decontam.exact_u64(q))
        if h in seen_exact:
            exact_dups += 1
            continue
        seen_exact.add(h)
        order.append(i)

    index = {}                      # gram -> [representative positions in `reps`]
    reps, n_near = [], 0
    for i in order:
        sk = decontam.sketch(queries[i])
        counts = {}
        for g in sk.tolist():
            for r in index.get(g, ()):
                counts[r] = counts.get(r, 0) + 1
        if any(c >= share for c in counts.values()):
            n_near += 1
            continue
        r = len(reps)
        reps.append(i)
        for g in sk.tolist():
            index.setdefault(g, []).append(r)

    post_exact = len(order)
    rate = n_near / max(post_exact, 1)
    rep = {"n_input": len(queries), "exact_dups_removed": exact_dups,
           "post_exact_dedup": post_exact, "near_duplicates": n_near,
           "near_dup_rate": round(rate, 5), "representatives": len(reps),
           "share_threshold": f"{share}/32",
           "_note": "sequential by registration: the first occurrence is the representative"}
    return [queries[i] for i in reps], [queries[i] for i in order], rep


def a8_action(form, queries, share=A8_NEAR_DUP_SHARE):
    """The registered action on A8 gate 1. -> (kept, report).

    Above a 25% near-duplicate rate the form keeps ONLY its representatives -- "a real cut, never
    topped up". If fewer than 50,000 remain the form is DROPPED from the build and reported.
    At or below 25% nothing is cut: the near-duplicates stay.
    """
    reps, deduped, rep = near_dup_gate(queries, share=share)
    if rep["near_dup_rate"] > A8_MAX_NEAR_DUP_RATE:
        kept = reps
        rep["action"] = "cut to representatives (rate above 0.25, never topped up)"
    else:
        kept = deduped          # near-duplicates stay; exact duplicates are always removed
        rep["action"] = "no cut (rate at or below 0.25); exact duplicates still removed"
    rep["retained"] = len(kept)
    rep["dropped_from_build"] = len(kept) < A8_MIN_RETAINED
    if rep["dropped_from_build"]:
        rep["action"] += f"; FORM DROPPED -- {len(kept):,} < {A8_MIN_RETAINED:,} retained"
    rep["form"] = form
    return ([] if rep["dropped_from_build"] else kept), rep


def manifest_sha256(parts):
    """The manifest's identity: sorted-key JSON of the counts and hashes, nothing else."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


# ---- the step-8 driver ----------------------------------------------------------------------
#
# Composed of small stages on purpose: the expensive one (generation) is a 10-GPU-hour spend, so
# every stage around it is testable without a GPU, and `build_form` is exercised against a stub
# OpenAI-compatible server in `test_corpus10`. What is NOT tested here is the model's output
# quality -- that is what the decision-15 smoke and `m10/SMOKE.md` are for.

def partition_seeds(seed_rows, gate_ids=(), holdout_n=HOLDOUT_PER_FORM, seed=0):
    """-> (build, held, report). ORDER MATTERS and is registered.

    The 400 judged gate seeds come out first (they were shown to a judge and must never enter the
    build corpus, T2-7 ⑧), then the FORMS-12 hold-out of 500 is drawn from what remains. Drawing
    the hold-out first would let a gate seed land in it and be reported as held-out when it is
    really just excluded.
    """
    gate = set(gate_ids)
    pool = [r for r in seed_rows if r[0] not in gate]
    held_ids, keep_ids = holdout_seed_ids([r[0] for r in pool], n=holdout_n, seed=seed)
    held = set(held_ids)
    build = [r for r in pool if r[0] not in held]
    return build, [r for r in pool if r[0] in held], {
        "n_input": len(seed_rows), "gate_seeds_excluded": len(seed_rows) - len(pool),
        "forms12_holdout": len(held), "build_seeds": len(build),
        "_order": "gate seeds removed FIRST, then the hold-out drawn from the remainder"}


def draw_seeds(build, need, seed=0):
    """-> the `need` seeds to generate from, a UNIFORM draw, never a positional prefix.

    `build[:need]` would take the seed store's own order, which for a Wikipedia store is dump
    order -- the population `m10/LEDGER.md` §T2-5 measured as a 5x distortion, and the same bias
    class that had to be fixed in `harvest.draw` and `paq.draw`. Returned in store order so
    generation and the manifest stay reproducible; the draw is what is uniform, not the order.
    """
    if len(build) <= need:
        return list(build)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(build), size=need, replace=False)
    return [build[int(i)] for i in sorted(pick)]


def screen_form(rows, seed_text, protected_hit=None, doc_hit=None):
    """The per-query screens, in the registered order. -> (kept, report).

    `rows`: [{query, form, seed_id}]. `seed_text`: {seed_id: passage}. `protected_hit(q)` and
    `doc_hit(q)` are injected so this is testable without a 27M-gram index; production passes
    `lambda q: protected10.hits(q, idx)` and the six's-documents `Inverted`.

    Order is cheapest-first, and each stage's removals are reported separately because §Data
    requires "removal counts per screen, per form".
    """
    import qfilter
    rep = {"n_input": len(rows), "out_of_rubric_range": 0, "copied_span": 0,
           "protected_index": 0, "six_documents": 0}
    kept = []
    for r in rows:
        q = r["query"]
        if not qfilter.in_range(r["form"], q):
            rep["out_of_rubric_range"] += 1
            continue
        src = seed_text.get(r["seed_id"])
        if src and copied_span(q, src):
            rep["copied_span"] += 1
            continue
        if protected_hit and protected_hit(q):
            rep["protected_index"] += 1
            continue
        if doc_hit and doc_hit(q):
            rep["six_documents"] += 1
            continue
        kept.append(r)
    rep["kept"] = len(kept)
    return kept, rep


def build_form(form, seed_rows, quota, *, n_per_seed=5, gate_ids=(), base=None, seen=None,
               protected_hit=None, doc_hit=None, workers=32, margin=1.35, seed=0, verbose=True):
    """One form end to end: partition seeds, generate, screen, A8. -> (queries, report).

    `margin` over-draws seeds because the screens and A8 both remove: at 5 queries per seed a
    143,000 quota needs 28,600 clean seeds, and the smoke's contract rate was not 100%.

    **Neither the seed selection nor the final cut is a positional prefix.** `build[:need]` and
    `out[:quota]` would both take a prefix of the seed store's own order, which for a Wikipedia
    store is dump order -- the population `m10/LEDGER.md` §T2-5 measured as a 5x distortion, and
    the same bias class that had to be fixed in `harvest.draw` and `paq.draw`. Both are uniform
    draws at `seed`.
    """
    import gen
    build, held, prep = partition_seeds(seed_rows, gate_ids=gate_ids, seed=seed)
    need = int(margin * quota / max(n_per_seed, 1))
    rng = np.random.default_rng(seed)
    use = draw_seeds(build, need, seed=seed)
    if len(use) < need and verbose:
        print(f"  {form}: {len(use):,} build seeds for a {need:,} target -- short", flush=True)
    seen = seen if seen is not None else set()
    g = gen.generate(form, [(r[0], r[1]) for r in use], n=n_per_seed,
                     base=base or gen.BASE, workers=workers, label=form if verbose else "",
                     seen=seen)
    seed_text = {r[0]: r[1] for r in use}
    kept, screens = screen_form(g["queries"], seed_text,
                                protected_hit=protected_hit, doc_hit=doc_hit)
    final, a8 = a8_action(form, [r["query"] for r in kept])
    keep_set, out = set(final), []
    for r in kept:                                  # keep the rows, not just the strings
        if r["query"] in keep_set:
            keep_set.discard(r["query"])
            out.append(r)
    rng.shuffle(out)                                # never a positional prefix -- see the docstring
    report = {"form": form, "quota": quota, "seeds": prep, "seed_margin": margin, "seed": seed,
              "seeds_used": len(use), "holdout_seed_ids": sorted(r[0] for r in held),
              "generation": {k: v for k, v in g.items() if k != "queries"},
              "screens": screens, "a8": a8,
              "final": len(out), "quota_met": len(out) >= quota,
              "dropped_from_build": a8["dropped_from_build"]}
    return out[:quota], report
