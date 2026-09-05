"""§Harvest — the real-text pipeline. Deterministic, no model in the loop.

The mandate registers four extraction rules over "the 6.15M pool documents and the Wikipedia seed
corpus, all under licences already approved for training", and this implements exactly those and
nothing else:

| rule | text | form |
|---|---|---|
| `title` | titles as-is | `title`, `keyword` (routed by length against the frozen rubric) |
| `heading` | section headings as-is | `title`, `keyword` |
| `claim` | declarative LEAD sentences of abstract-like passages, 8–40 words, a finite verb, no first person | `claim` |
| `ask` | sentences ending in `?`, with the preceding sentence kept as optional body | `factoid`, `product`, routed by the SOURCE corpus |

This is the arm-A3 corpus: **real query-like text**, the half of the coverage thesis that does not
depend on a generator. Three of the four clean-4 headline datasets have a real-text counterpart
here — scidocs↔titles, scifact↔claim sentences, trec-covid/nfcorpus↔headings — which is why A3
exists as its own arm and why the headline can rest on real text plus the teacher.

**ESCI queries are NOT harvested**: they are real user queries and are already in the M9 pool
through `esci-us`, so harvesting them would double-count arm A1's own data into A3.

**The reserved-DOCUMENT gap applies here exactly as it does to seeds** (W4): a harvested Wikipedia
lead sentence can be near-identical to a DBpedia-entity abstract and no fingerprint exists to catch
it. Registered, disclosed, unchanged — and it is why every harvested string goes through the same
protected-index screen, quotas and hold-out as a generated one.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m10src"):
    sys.path.insert(0, str(REPO / p))
OUT = REPO / "work" / "m10harvest"

CLAIM_MIN, CLAIM_MAX = 8, 40
FIRST_PERSON = re.compile(r"\b(?:I|we|We|our|Our|my|My|us|Us|me|Me)\b")
# A CLOSED finite-verb list, and only that. An earlier version added a regular-inflection branch
# (`\w{3,}(?:ed|es|s)`) to catch "X contains Y" — and a unit case caught it firing on the plural
# NOUN in "a very long noun phrase of exactly nine plain nouns", i.e. admitting a bare noun phrase
# as a declarative claim. English plural nouns and third-person verbs are the same suffix, so the
# branch cannot tell them apart without a parser, and a parser would be a model in the loop, which
# the rule is registered not to have.
#
# So the list is explicit and auditable, and the rule UNDER-fires by design: it loses "This paper
# presents a method…" and keeps "X is a Y". That is the right direction here — supply is millions
# of sentences and never binding, while a non-sentence in the claim form is a real defect.
FINITE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|does|do|did|can|could|will|would|shall|"
    r"should|may|might|must|consists|comprises|denotes|refers)\b", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# a heading in this dump's `text` field, reused from the seed store's structural rule
_TERMINAL = ('.', '!', '?', '"', "'", ')', ']', '”', '’')


def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(" ".join(text.split())) if s.strip()]


def is_claim(s):
    """Declarative, 8-40 words, a finite verb, no first person, not a question."""
    w = s.split()
    if not (CLAIM_MIN <= len(w) <= CLAIM_MAX):
        return False
    if s.endswith("?") or FIRST_PERSON.search(s):
        return False
    return bool(FINITE.search(s))


def claims_from(text, max_per_doc=1):
    """LEAD sentences only, as registered — the first sentences of the passage, not any sentence."""
    out = []
    for s in sentences(text)[:max_per_doc + 2]:
        if is_claim(s):
            out.append(s)
            if len(out) >= max_per_doc:
                break
    return out


def asks_from(text, max_per_doc=3):
    """-> [(question, preceding sentence or None)]. The body is OPTIONAL, as registered."""
    ss = sentences(text)
    out = []
    for i, s in enumerate(ss):
        if s.endswith("?") and 3 <= len(s.split()) <= 40:
            out.append((s, ss[i - 1] if i else None))
            if len(out) >= max_per_doc:
                break
    return out


def headings_from(article_text, max_per_doc=12):
    """Section headings as-is, using the seed store's structural heading rule."""
    import wikibody
    lines = article_text.split("\n")
    out = []
    for i in range(len(lines)):
        if wikibody.is_heading(lines, i):
            h = lines[i].strip()
            if h and h.lower() not in ("references", "see also", "external links", "notes",
                                       "further reading", "bibliography", "sources", "citations"):
                out.append(h)
                if len(out) >= max_per_doc:
                    break
    return out


def route_by_length(text, forms=("keyword", "title")):
    """Titles and headings are routed to the form whose registered range they fall in."""
    import qfilter
    for f in forms:
        if qfilter.in_range(f, text):
            return f
    return None


def wikipedia_pass(limit=None, out_path=None, log_every=250_000, verbose=True):
    """One streamed pass over the Wikipedia seed corpus emitting title, heading and claim rows."""
    import time
    import wikibody
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path or (OUT / "wiki_harvest.jsonl"))
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    n_art, per_rule = 0, {}
    t0 = time.time()
    with partial.open("w") as fh:
        for r in wikibody.stream(limit):
            n_art += 1
            aid, title, text = r["id"], r["title"], r["text"]
            f = route_by_length(title)
            if f:
                per_rule[("title", f)] = per_rule.get(("title", f), 0) + 1
                fh.write(json.dumps({"rule": "title", "form": f, "text": title,
                                     "src": "wikipedia", "doc": aid}) + "\n")
            for h in headings_from(text):
                f = route_by_length(h)
                if f:
                    per_rule[("heading", f)] = per_rule.get(("heading", f), 0) + 1
                    fh.write(json.dumps({"rule": "heading", "form": f, "text": h,
                                         "src": "wikipedia", "doc": aid}) + "\n")
            for c in claims_from(wikibody.lead(text)):
                per_rule[("claim", "claim")] = per_rule.get(("claim", "claim"), 0) + 1
                fh.write(json.dumps({"rule": "claim", "form": "claim", "text": c,
                                     "src": "wikipedia", "doc": aid}) + "\n")
            if verbose and n_art % log_every == 0:
                el = time.time() - t0
                print(f"  {n_art:,} articles ({n_art / el:.0f}/s, {el / 60:.1f}m), "
                      f"{sum(per_rule.values()):,} rows", flush=True)
                fh.flush()
    partial.replace(out_path)
    rep = {"source": "wikipedia", "repo": wikibody.REPO_ID, "config": wikibody.CONFIG,
           "revision": wikibody.REVISION, "n_articles": n_art, "complete": limit is None,
           "limit": limit, "rows": sum(per_rule.values()),
           "by_rule_form": {f"{a}/{b}": c for (a, b), c in sorted(per_rule.items())},
           "seconds": round(time.time() - t0, 1), "path": str(out_path),
           "bytes": out_path.stat().st_size}
    Path(str(out_path) + ".report.json").write_text(json.dumps(rep, indent=1))
    if verbose:
        print(json.dumps(rep, indent=1), flush=True)
    return rep


def arxiv_pass(out_path=None, log_every=500_000, verbose=True):
    """arXiv metadata (CC0): titles as-is, and the first declarative sentence of each abstract.

    The 100,000 base ids drawn for the `arxiv-title` diagnostic are EXCLUDED from every training
    role, so they are excluded here — that exclusion is the reason the diagnostic is clean.
    """
    import time
    import arxiv_draw
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path or (OUT / "arxiv_harvest.jsonl"))
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    excluded = set(json.loads((REPO / "work" / "m10arxiv"
                              / "arxiv_excluded_base_ids.json").read_text()))
    n, skipped, per_rule = 0, 0, {}
    t0 = time.time()
    with partial.open("w") as fh:
        for r in arxiv_draw.stream_records(verbose=False):
            n += 1
            b = arxiv_draw._base_id(r["id"])
            if b in excluded:
                skipped += 1
                continue
            title = " ".join((r.get("title") or "").split())
            f = route_by_length(title)
            if f:
                per_rule[("title", f)] = per_rule.get(("title", f), 0) + 1
                fh.write(json.dumps({"rule": "title", "form": f, "text": title,
                                     "src": "arxiv", "doc": b}) + "\n")
            for c in claims_from(r.get("abstract") or ""):
                per_rule[("claim", "claim")] = per_rule.get(("claim", "claim"), 0) + 1
                fh.write(json.dumps({"rule": "claim", "form": "claim", "text": c,
                                     "src": "arxiv", "doc": b}) + "\n")
            if verbose and n % log_every == 0:
                el = time.time() - t0
                print(f"  {n:,} records ({n / el:.0f}/s), {sum(per_rule.values()):,} rows",
                      flush=True)
    partial.replace(out_path)
    rep = {"source": "arxiv", "n_records": n, "excluded_drawn": skipped,
           "rows": sum(per_rule.values()),
           "by_rule_form": {f"{a}/{b}": c for (a, b), c in sorted(per_rule.items())},
           "seconds": round(time.time() - t0, 1), "path": str(out_path),
           "bytes": out_path.stat().st_size, "complete": True}
    Path(str(out_path) + ".report.json").write_text(json.dumps(rep, indent=1))
    if verbose:
        print(json.dumps(rep, indent=1), flush=True)
    return rep


# routing for the `ask` rule, BY THE SOURCE CORPUS as registered. Neither Wikipedia nor arXiv is
# question-bearing, so this rule only fires over the pool documents, and it is run rather than
# quietly skipped: a registered rule that yields little should report a small number, not nothing.
ASK_ROUTE = {"hotpotqa-corpus": "factoid", "squad-ctx": "factoid",
             "mrtydi-docs": "factoid", "esci-prod": "product"}


def pool_pass(stores=None, out_path=None, log_every=1_000_000, verbose=True):
    """The `ask` rule over the approved pool documents: sentences ending in `?`, with the
    preceding sentence kept as optional body, routed to a form by which store they came from."""
    import time
    import mix
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path or (OUT / "pool_harvest.jsonl"))
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    per_rule, n_docs = {}, 0
    t0 = time.time()
    with partial.open("w") as fh:
        for store in (stores or list(ASK_ROUTE)):
            form = ASK_ROUTE[store]
            ids, texts = mix.load_store(store)
            if verbose:
                print(f"  {store}: {len(texts):,} documents -> {form}", flush=True)
            for i, t in enumerate(texts):
                n_docs += 1
                if "?" not in t:                       # the cheap reject first: most documents
                    continue
                for q, body in asks_from(t):
                    k = (store, form)
                    per_rule[k] = per_rule.get(k, 0) + 1
                    fh.write(json.dumps({"rule": "ask", "form": form, "text": q, "body": body,
                                         "src": store, "doc": ids[i]}) + "\n")
                if verbose and n_docs % log_every == 0:
                    print(f"    {n_docs:,} documents ({n_docs / (time.time() - t0):.0f}/s), "
                          f"{sum(per_rule.values()):,} rows", flush=True)
    partial.replace(out_path)
    rep = {"source": "pool", "stores": list(stores or ASK_ROUTE), "n_documents": n_docs,
           "rows": sum(per_rule.values()),
           "by_store_form": {f"{a}/{b}": c for (a, b), c in sorted(per_rule.items())},
           "seconds": round(time.time() - t0, 1), "path": str(out_path),
           "bytes": out_path.stat().st_size, "complete": stores is None}
    Path(str(out_path) + ".report.json").write_text(json.dumps(rep, indent=1))
    if verbose:
        print(json.dumps(rep, indent=1), flush=True)
    return rep


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if which in ("wiki", "both"):
        wikipedia_pass(limit=lim)
    if which in ("arxiv", "both"):
        arxiv_pass()
    if which in ("pool", "both"):
        pool_pass()


# ---- the A3 draw and its screens ------------------------------------------------------------
#
# REGISTERED BEFORE THE DRAW RUNS. Quota ≈1.25M harvested strings across the three forms that
# HAVE a registered quota row: `title`, `keyword`, `claim`.
#
# **`factoid` and `product` have no quota row and are therefore NOT DRAWN.** An earlier version of
# this comment said the quota was those three "plus whatever the `ask` rule returns"; that phrase
# was never registered in `m10/LEDGER.md` §Harvest or in `instructions-m10.md` §Data, and it is
# struck. The mandate lists five harvested forms at ~250K each and also registers "a harvested form
# that falls under 100K reverts to generation at ≈143K"; the `ask` rule yields 5,605 `factoid` and
# 40,977 `product` rows, so BOTH fall under that rule. Reverting them to generation is a quota
# decision (two new prompts, two smoke windows, +286K over the registered 1.0M generation cap) and
# is reserved to Dylan at the M10.2 lock; the default is excluded. The rows are still harvested and
# their yield reported -- `pool_pass` runs rather than being quietly skipped -- and the number of
# rows dropped for having no quota row is carried in the report as `skipped_no_quota`, so the
# exclusion is visible in the artifact and not only in a ledger entry.
#
# **Draw rule:** for each form, a UNIFORM random sample (seed 0) over the union of every source's
# rows for that form, after exact-text dedup — not weighted, not balanced, not scored. Harvested
# text has no score to sort by, and uniform is the choice that preserves the corpus's own
# distribution, which is the entire reason the arm is called "real text". **The realized source
# mix is REPORTED, not fixed in advance**, so nobody has to defend a split nobody measured.
#
# **Screens, identical to a generated string's** (§Harvest: "the same screens, quotas and hold-out
# as a generated one"): the M10 protected index on the query side, and the six's documents, the
# four DEV components' documents and the admitted COV components' documents streamed against a
# candidate-side `Inverted` on the document side. Matches are REMOVED. A margin is drawn so the
# quota still fills after removals; running out raises rather than returning a short draw.

# **The frozen rubric's word range is enforced on every drawn row** (`m10src/qfilter`). `title`
# and `keyword` are already in range by construction -- `route_by_length` assigned them BY that
# range -- so this binds only `claim`, whose extraction window (`CLAIM_MIN/CLAIM_MAX`, 8-40) is a
# deliberate SUPERSET of the rubric's (8, 25). Extraction stays 8-40 so the constants keep matching
# the rows already written to disk and their `*.report.json`; the draw is the authority, and the
# per-form off-range count is reported. Enforcing each form's own frozen-rubric range is the
# largest on-form quality lever measured in M10 (`results/m10_qfilter_effect.json`), the rubric is
# the frozen gate standard, and the direction is safe: it only removes. Claim supply after the
# range is ~5.9M against a 416K quota.
QUOTA = {"title": 417_000, "keyword": 417_000, "claim": 416_000}
DRAW_SEED, MARGIN = 0, 1.5


def _iter_rows(paths):
    """Every path must exist AND its pass must have completed. A silently skipped `.partial` would
    produce a corpus missing a whole source and report it as a clean draw."""
    for p in paths:
        pp = Path(p)
        if not pp.exists():
            raise SystemExit(f"{pp}: missing -- run the harvest pass that writes it before drawing")
        rp = Path(str(pp) + ".report.json")
        if not rp.exists():
            raise SystemExit(f"{rp}: missing -- refusing to draw from an unreported pass")
        if not json.loads(rp.read_text()).get("complete"):
            raise SystemExit(f"{rp}: complete != true -- that pass did not finish; refusing to draw")
    for p in paths:
        with Path(p).open() as fh:
            for line in fh:
                yield json.loads(line)


def draw(quota=None, margin=MARGIN, paths=None, out_dir=None, verbose=True):
    """-> ({form: [row]}, report). Uniform per form, deduped, then screened; matches removed.

    **`out_dir` exists because a smoke otherwise overwrites the real artifact.** The outputs used
    to be written to the module-level `OUT` no matter what `paths` said, so a scaled-down run over
    sampled inputs produced a `harvest_draw.json` and `harvest_drawn.jsonl` at exactly the paths a
    real draw writes -- 900 rows that look like a 1.25M corpus. That is the same trap as
    `calib.run_arm`'s 90-step smoke writing to the real `P0.json` (`m10/STATUS.md`), and it bit
    here too. A smoke must pass `out_dir`; the report records which directory it wrote to.
    """
    import time
    import numpy as np
    import decontam
    import protected10
    import cov_screen
    import devsuite
    import qfilter
    from cov_admit import COMPONENTS
    quota = quota or QUOTA
    paths = paths or [OUT / "wiki_harvest.jsonl", OUT / "arxiv_harvest.jsonl",
                      OUT / "pool_harvest.jsonl"]
    t0 = time.time()

    # pass 1: reservoir per form, so 21M rows never sit in memory at once
    want = {f: int(margin * n) for f, n in quota.items()}
    rng = np.random.default_rng(DRAW_SEED)
    res, seen_n, seen_txt = {f: [] for f in want}, {f: 0 for f in want}, set()
    src_mix = {}
    skipped_no_quota, off_range, dup = {}, {f: 0 for f in want}, {f: 0 for f in want}
    for r in _iter_rows(paths):
        f = r["form"]
        if f not in want:
            # a harvested form with no registered quota row (`factoid`, `product`). Counted, not
            # drawn, and REPORTED -- see the §Harvest note above; admission is Dylan's.
            skipped_no_quota[f] = skipped_no_quota.get(f, 0) + 1
            continue
        if not qfilter.in_range(f, r["text"]):   # the frozen rubric's own range; binds `claim`
            off_range[f] += 1
            continue
        k = " ".join(r["text"].split()).lower()
        if k in seen_txt:
            dup[f] += 1
            continue
        seen_txt.add(k)
        seen_n[f] += 1
        if len(res[f]) < want[f]:
            res[f].append(r)
        else:                                   # Vitter's reservoir: uniform over the stream
            j = int(rng.integers(0, seen_n[f]))
            if j < want[f]:
                res[f][j] = r
    del seen_txt            # ~21M normalized strings; nothing below reads it and pass 2 needs the RAM
    for f, rows in res.items():
        for r in rows:
            src_mix[(f, r["src"], r["rule"])] = src_mix.get((f, r["src"], r["rule"]), 0) + 1
    if verbose:
        print(f"  reservoir: " + ", ".join(f"{f}={len(v):,}/{seen_n[f]:,}"
                                           for f, v in res.items())
              + f"  ({time.time() - t0:.0f}s)", flush=True)

    # pass 2: the screens, both directions, matches REMOVED
    flat = [r for f in sorted(res) for r in res[f]]
    texts = [r["text"] for r in flat]
    idx = protected10.build(verbose=verbose)
    q_drop = {i for i, t in enumerate(texts) if protected10.hits(t, idx)}
    if verbose:
        print(f"  query-side: {len(q_drop):,} of {len(texts):,} dropped", flush=True)
    inv = decontam.Inverted([decontam.query_grams(t) for t in texts],
                            [decontam.exact_u64(t) for t in texts])
    d_drop, per_stream = set(), {}

    def run_stream(name, it):
        t1, n, before = time.time(), 0, len(d_drop)
        for d in it:
            n += 1
            ex, near = inv.match(d, cov_screen.MIN_SHARE)
            d_drop.update(ex.tolist()); d_drop.update(near.tolist())
        per_stream[name] = dict(streamed=n, new_drops=len(d_drop) - before,
                                seconds=round(time.time() - t1, 1))
        if verbose:
            print(f"  {name}: {n:,} streamed, {per_stream[name]['new_drops']:,} new drops "
                  f"({per_stream[name]['seconds']:.0f}s)", flush=True)

    run_stream("six-docs", decontam.stream_six_docs())
    for comp in devsuite.COMPONENTS:
        run_stream(f"dev:{comp}", decontam.stream_dev_component_docs(comp))
    for _family, comps in COMPONENTS.items():
        for name, repo, rev in comps:
            _qs, ds = cov_screen.load_component(name, repo, rev)
            run_stream(f"cov:{name}", iter(ds))

    drop = q_drop | d_drop
    kept = {f: [] for f in quota}
    for i, r in enumerate(flat):
        if i not in drop:
            kept[r["form"]].append(r)
    # **Truncating a reservoir to its first `n` slots is NOT uniform.** Algorithm R fills slots
    # 0..want-1 with the stream's first `want` items and only ever overwrites slot j with a LATER
    # item, so slot j can hold initial item j and no other. Taking the first n of want slots
    # therefore over-represents stream positions [0, n) by exactly the margin (1.5x, confirmed by
    # simulation: 75.0 observed against 50.0 expected) and excludes positions [n, want) outright
    # (0.0 observed against 25.0 expected). The stream is Wikipedia dump order, whose prefix LEDGER
    # §T2-5 already measured as a 5x distortion (2001-02 core articles), so the bias would land on
    # exactly the known-skewed population. A shuffle before truncation is what the registered rule
    # ("uniform reservoir") already promises.
    for f in kept:
        rng.shuffle(kept[f])
    out, counts = {}, {}
    for f, n in quota.items():
        if len(kept[f]) < n and seen_n[f] > want[f]:
            raise SystemExit(f"{f}: {len(kept[f]):,} survived of a {want[f]:,} draw for a "
                             f"{n:,} quota -- widen `margin` rather than take a short draw")
        out[f] = kept[f][:n]
        counts[f] = dict(available=seen_n[f], drawn=want[f], survived=len(kept[f]),
                         taken=len(out[f]), short=max(0, n - len(kept[f])))
    rep = dict(quota=quota, margin=margin, seed=DRAW_SEED,
               skipped_no_quota=skipped_no_quota, off_rubric_range=off_range, exact_dups=dup,
               rubric_ranges={f: list(qfilter.RANGES[f]) for f in sorted(quota)},
               draw_rule="uniform reservoir per form over the union of sources, after exact-text "
                         "dedup; source mix reported not fixed",
               counts=counts,
               truncation="kept[] shuffled with the draw rng before truncation; see the note in "
                          "draw() -- first-n-of-reservoir is biased toward the stream prefix",
               source_mix={f"{a}/{b}/{c}": n for (a, b, c), n in sorted(src_mix.items())},
               screen=dict(n_candidates=len(texts), dropped_query_side=len(q_drop),
                           dropped_document_side=len(d_drop), dropped_total=len(drop),
                           per_stream=per_stream),
               seconds=round(time.time() - t0, 1))
    out_dir = Path(out_dir) if out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    rep["out_dir"] = str(out_dir)
    (out_dir / "harvest_draw.json").write_text(json.dumps(rep, indent=1))
    with (out_dir / "harvest_drawn.jsonl").open("w") as fh:
        for f in sorted(out):
            for r in out[f]:
                fh.write(json.dumps(r) + "\n")
    if verbose:
        print(json.dumps(counts, indent=1), flush=True)
    return out, rep
