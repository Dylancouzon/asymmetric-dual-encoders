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
# W10 amendment, ruled by Dylan 2026-09-05 (option (c), as the reviewer specified it). The
# registered 8-gram/16-of-32 rule is UNREACHABLE below 23 words, identical to exact dedup below 8,
# and cannot see a one-slot template at <= 15 words at ANY threshold, because every 8-gram window
# covers the changed word. Below 39 -- the point where the bottom-32 sketch stops truncating and
# the test becomes an absolute count rather than a Jaccard estimate -- a word-4-gram rule applies.
# At and above 39 the registered rule is BIT-IDENTICAL. Uniform across forms: the harvested
# measurements (keyword 0.05%, title 3.28%, claim 8.06%) show a uniform rule passes real text on
# its merits, so a provenance-conditional gate would be tuning. The 50% constant comes from the
# Jaccard intent, NOT from any value that makes a form pass or fail.
A8_LONG_FLOOR = 39              # at/above this many words the registered rule applies unchanged
A8_SHORT_K = 4                  # word-4-grams below the floor
A8_SHORT_FRAC = 0.5             # >= 50% of the SMALLER query's gram set
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


def _occurrences(hay, needle):
    """-> the start indices at which `needle` appears in `hay` (both word lists)."""
    n = len(needle)
    if not n or n > len(hay):
        return []
    return [i for i in range(len(hay) - n + 1) if hay[i:i + n] == needle]


def copied_span(query, source_text, exclude_span=None, k=SPAN_K):
    """True if the query shares a word-`k`-gram with its OWN seed passage -- a copied span is not
    a query. `exclude_span` is the harvested string itself, because a harvested title IS a span of
    its document and would otherwise always self-match.

    **The exclusion is POSITIONAL, not by gram value.** Subtracting the span's k-gram VALUES from
    the source's set removes those grams wherever else they occur, so a query copied from a
    different occurrence of the same five words would pass (Codex 2026-09-05, finding 8). Instead
    the span is located in the source and only the k-gram windows lying entirely inside one of its
    occurrences are dropped. Windows that straddle a boundary are KEPT, which is also why the
    exclusion cannot splice new grams the way deleting characters would.
    """
    qw = decontam.norm_words(query)
    sw = decontam.norm_words(source_text)
    if len(sw) < k:
        return False
    drop = set()
    if exclude_span:
        ew = decontam.norm_words(exclude_span)
        for st in _occurrences(sw, ew):
            # windows fully inside [st, st+len(ew))
            for i in range(st, st + len(ew) - k + 1):
                drop.add(i)
    src = {tuple(sw[i:i + k]) for i in range(len(sw) - k + 1) if i not in drop}
    return bool(_kgrams(qw, k) & src)


def near_dup_gate(queries, share=A8_NEAR_DUP_SHARE):
    """A8 gate 1 for ONE form. -> (representatives, exact_deduped, report).

    Sequential and order-dependent BY REGISTRATION: "a query is a near-duplicate if its
    word-8-gram bottom-32 sketch matches >= 16/32 with an EARLIER query of the same form (the
    earlier one is the representative)". So the first occurrence is always the representative.

    "An earlier query" is read LITERALLY: every earlier query, dropped ones included, so chains
    are caught (A~B, B~C, A!~C drops C) -- which IS template collapse. The narrower "earlier
    surviving representative" reading was implemented, had no caller but its own test, and is
    deleted; the reading and why it was chosen live in `m10/LEDGER.md`.

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

    index = {}                      # 8-gram -> [earlier query positions]      (registered rule)
    sindex = {}                     # 4-gram -> [earlier SHORT query positions] (W10 amendment)
    sgrams = {}                     # position -> its 4-gram set, for the "smaller set" denominator
    reps, n_near, n_short_only = [], 0, 0
    for i in order:
        w = decontam.norm_words(queries[i])
        short = len(w) < A8_LONG_FLOOR
        sk = decontam.sketch(queries[i])

        counts = {}
        for g in sk.tolist():
            for r in index.get(g, ()):
                counts[r] = counts.get(r, 0) + 1
        dup = any(c >= share for c in counts.values())

        g4 = None
        if short:
            g4 = _kgrams(w, A8_SHORT_K) if len(w) > A8_SHORT_K else {tuple(w)}
            if not dup:
                c4 = {}
                for x in g4:
                    for r in sindex.get(x, ()):
                        c4[r] = c4.get(r, 0) + 1
                if any(c >= A8_SHORT_FRAC * min(len(g4), len(sgrams[r]))
                       for r, c in c4.items()):
                    dup = True
                    n_short_only += 1

        if dup:
            n_near += 1
        else:
            reps.append(i)
        for g in sk.tolist():
            index.setdefault(g, []).append(i)
        if short:
            sgrams[i] = g4
            for x in g4:
                sindex.setdefault(x, []).append(i)

    post_exact = len(order)
    rate = n_near / max(post_exact, 1)
    rep = {"n_input": len(queries), "exact_dups_removed": exact_dups,
           "post_exact_dedup": post_exact, "near_duplicates": n_near,
           # the RAW rate drives the action; the rounded one is for display only. 50001/200001 is
           # 0.25000375 and rounds to 0.25, which would wrongly escape the > 0.25 cut.
           "near_dup_rate_raw": rate,
           "near_dup_rate": round(rate, 5), "representatives": len(reps),
           "share_threshold": f"{share}/32",
           "rule": f"W10: 4-grams >= 50% below {A8_LONG_FLOOR} words, else 8-gram {share}/32",
           "caught_only_by_short_rule": n_short_only,
           "_note": "sequential by registration: the first occurrence is the representative"}
    return [queries[i] for i in reps], [queries[i] for i in order], rep


def a8_action(form, queries, share=A8_NEAR_DUP_SHARE):
    """The registered action on A8 gate 1. -> (kept, report).

    Above a 25% near-duplicate rate the form keeps ONLY its representatives -- "a real cut, never
    topped up". If fewer than 50,000 remain the form is DROPPED from the build and reported.
    At or below 25% nothing is cut: the near-duplicates stay.
    """
    reps, deduped, rep = near_dup_gate(queries, share=share)
    if rep["near_dup_rate_raw"] > A8_MAX_NEAR_DUP_RATE:
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
    # NOTE: the held seeds are excluded from TRAINING here, which is the guarantee that matters.
    # FORMS-12 is also an EVALUATION sample -- "12 x 500 held-out generated or harvested queries,
    # overlap@10 between student and teacher" -- so 500 queries per form still have to be GENERATED
    # from these held seeds by a separate call. `build_form` returns their ids in
    # `report["holdout_seed_ids"]` for exactly that; producing the eval set is not done here.
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
    out_final = out[:quota]
    report = {"form": form, "quota": quota, "seeds": prep, "seed_margin": margin, "seed": seed,
              "seeds_used": len(use), "holdout_seed_ids": sorted(r[0] for r in held),
              "generation": {k: v for k, v in g.items() if k != "queries"},
              "screens": screens, "a8": a8,
              "final": len(out_final), "before_quota_cut": len(out),
              "quota_met": len(out_final) >= quota,
              "dropped_from_build": a8["dropped_from_build"]}
    return out_final, report


def harvest_holdout(path, per_form=HOLDOUT_PER_FORM, seed=0, out_dir=None, verbose=True):
    """FORMS-12 for the HARVESTED forms, as a post-pass over `harvest_drawn.jsonl`.

    §Data requires harvested strings to get "the same screens, quotas and hold-out as a generated
    one", and `harvest.draw` applies neither the hold-out nor the own-source screen (Codex
    2026-09-05, finding 2 → §Open questions W11). No re-draw is needed: every drawn row carries
    its source `doc` id.

    **The hold-out is by DOCUMENT and applies across ALL forms.** One Wikipedia article yields a
    title, several headings and a lead claim, so holding a doc out for `title` only would train on
    that same article's `claim` — and the rule is "queries generated or harvested from them are
    never trained on", not "from them, in that form". 500 docs are drawn per form and the UNION is
    withheld, so a form's realized hold-out can exceed 500 rows; the numbers are reported.
    """
    import json
    from pathlib import Path
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    by_form = {}
    for r in rows:
        by_form.setdefault(r["form"], set()).add(r["doc"])
    held = set()
    for f in sorted(by_form):
        ids = sorted(by_form[f] - held)          # never re-draw a doc already held for another form
        take = min(per_form, max(len(ids) - 1, 0))
        h, _ = holdout_seed_ids(ids, n=take, seed=seed) if take else ([], [])
        held.update(h)
    train = [r for r in rows if r["doc"] not in held]
    forms12 = [r for r in rows if r["doc"] in held]
    rep = {"path": str(path), "per_form_target": per_form, "seed": seed,
           "n_rows": len(rows), "held_documents": len(held),
           "train_rows": len(train), "forms12_rows": len(forms12),
           "forms12_by_form": {f: sum(1 for r in forms12 if r["form"] == f)
                               for f in sorted(by_form)},
           "train_by_form": {f: sum(1 for r in train if r["form"] == f) for f in sorted(by_form)},
           "_rule": "held by DOCUMENT across all forms; a doc held for one form is held for every "
                    "form, since one article yields a title, headings and a claim"}
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, part in (("harvest_train.jsonl", train), ("harvest_forms12.jsonl", forms12)):
            with (d / name).open("w") as fh:
                for r in part:
                    fh.write(json.dumps(r) + "\n")
        (d / "harvest_holdout.json").write_text(json.dumps(rep, indent=1))
    if verbose:
        print(json.dumps(rep, indent=1), flush=True)
    return train, forms12, rep


# ---- the diversity pilot's measurement (W10) -------------------------------------------------
#
# A8's registered gate is inert below 39 words (`m10/LEDGER.md` W10), so before the 10-box-hour
# generation run a pilot measures BOTH the registered gate and the sensitive alternatives on real
# output. None of this is a registered instrument: it exists so W10 is decided on numbers, and so
# a form whose rate is heading past 25% is caught at the prompt rather than by the blunt cut.

def _kgrams_k(words, k):
    if len(words) <= k:
        return {tuple(words)}
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def prop_near_dup_rate(texts, k=4, frac=0.5):
    """Near-dup if >= `frac` of the SMALLER query's k-gram set is shared with an earlier query.

    Degrades gracefully on short strings, which is exactly what the registered 16/32 absolute
    count cannot do: below 8 words a sketch is ONE whole-text hash, and no 8-gram rule can see a
    one-slot template at <= 15 words, since every window covers the changed word.
    """
    index, grams, n = {}, [], 0
    for t in texts:
        g = _kgrams_k(decontam.norm_words(t), k)
        counts = {}
        for x in g:
            for j in index.get(x, ()):
                counts[j] = counts.get(j, 0) + 1
        if any(c >= frac * min(len(g), len(grams[j])) for j, c in counts.items()):
            n += 1
            grams.append(g)
            continue
        grams.append(g)
        for x in g:
            index.setdefault(x, []).append(len(grams) - 1)
    return n / max(len(texts), 1), n


def opener_concentration(texts, k=4, top=10):
    """-> (share of queries in the `top` most common leading k-grams, distinct-2 ratio).

    Frame repetition -- "what should I do if my ..." over and over -- is what a one-slot template
    looks like, and it is invisible to an 8-gram rule. Distinct-2 is the classic diversity ratio:
    unique bigrams / total bigrams.
    """
    lead, bigrams, n_bi = {}, set(), 0
    for t in texts:
        w = decontam.norm_words(t)
        key = tuple(w[:k])
        lead[key] = lead.get(key, 0) + 1
        for i in range(len(w) - 1):
            bigrams.add((w[i], w[i + 1]))
            n_bi += 1
    top_share = sum(sorted(lead.values(), reverse=True)[:top]) / max(len(texts), 1)
    return top_share, (len(bigrams) / max(n_bi, 1))


def diversity_report(queries, form=None, ns=(200, 500, 1000, 2000)):
    """Everything W10 needs about one form's output, at several sample sizes.

    The rate is monotone non-decreasing in n, so a single number is a floor: the CURVE is the
    quantity. Three `frac` values are reported so no form is judged on a knife-edge constant.
    """
    _r, _d, a8 = near_dup_gate(queries)
    top_share, distinct2 = opener_concentration(queries)
    rep = {"form": form, "n": len(queries),
           "a8_registered_rate": a8["near_dup_rate_raw"],
           "a8_can_fire_below_39_words": False,
           "top10_opening_4gram_share": round(top_share, 4),
           "distinct_2": round(distinct2, 4),
           "curve": {}}
    for k in ns:
        if k > len(queries):
            continue
        rep["curve"][str(k)] = {f"frac_{int(f * 100)}": round(prop_near_dup_rate(queries[:k], frac=f)[0], 4)
                                for f in (0.4, 0.5, 0.6)}
    vals = [v["frac_50"] for v in rep["curve"].values()]
    rep["still_rising"] = bool(len(vals) > 1 and vals[-1] > vals[0] + 0.01)
    rep["_note"] = ("unregistered diagnostics; the registered gate is the a8_ row. The rate is "
                    "monotone in n, so every value here is a FLOOR for the build's 143,000.")
    return rep
