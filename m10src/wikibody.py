"""`wikipedia-body` — the seed store that closes the health/finance supply shortfall (W3).

**Why this store and not another keyword lever.** `ROUTE_WIDE` failed its judged-precision gate
(28% / 38% on-topic vs a >= 80% bar) because the router selects on the PRESENCE of medical or
financial words, not on subject. Widening further is the same error at larger scale. The supply
shortfall is instead closed by corpus size at an unchanged router: `hotpotqa-corpus` is Wikipedia
INTRO paragraphs only, so 5.23M articles yield one candidate chunk each; the same articles' BODY
text yields ~13. A 20,000-article pilot at the registered `ROUTE` and `min_score >= 4` gave
health 1,840 / finance 3,532 seeds, i.e. ~92,000 health per 1M articles against a ~33K need.

Wikipedia is a **registered** seed source, not a new admission: `instructions-m10.md` §Data
("Seeds for the generated half: Wikipedia stratified by top-level category (CC BY-SA) and the
approved pool corpora") and `research/m7-data-licensing.md:42,61` (CC BY-SA family position,
confirmed by Dylan 2026-08-25, model-card attribution required). This is why the FineWeb closure
(`m10/EXPLORED.md`) does not reach it: FineWeb was an unregistered source whose overlap with
reserved text is unbounded, and this store's overlap is bounded by construction (below).

**The lead section is excluded.** Two reserved corpora are Wikipedia lead text — DBpedia-entity
(abstracts) and FEVER (introductory sections) — and reserved-set DOCUMENT fingerprints do not
exist and cannot be built without opening the reserved corpora (W4). Body-only text is disjoint
from lead text by construction, so this store is strictly cleaner on that axis than the intro
store already in use. It is also near-disjoint from `hotpotqa-corpus` for the same reason.

**Divergence disclosed, not fixed:** the mandate says "stratified by top-level category"; this
routes by keyword, as `seeds.ROUTE` already does for the existing stores (T2-3). Category-
membership routing over the `categorylinks` dump is the registered next lever if the gate fails —
it is the subject-level filter the judge asked for, and it is not a relaxed floor.
"""
import json, os, re, sys
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m10src"):
    sys.path.insert(0, str(REPO / p))
OUT = REPO / "work" / "m10gen"

REPO_ID = "wikimedia/wikipedia"
CONFIG = "20231101.en"
REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
# CC BY-SA 4.0 (the 2023-11 dump post-dates Wikimedia's June 2023 move to 4.0; the HF card still
# tags 3.0 — both recorded). Dual-licensed GFDL; CC BY-SA is the licence chosen for this use.
LICENCE = "CC BY-SA 4.0 (HF card tags 3.0; dual GFDL, CC BY-SA chosen)"

# Registered BEFORE the scan, and not tuned to any count it produces.
FORMS = ("health", "finance")        # `howto` clears on the existing stores and is not touched
PER_ARTICLE_CAP = 3                  # per form: one article must not become 3 x 5 near-identical
                                     # questions x 30 chunks (the A8 near-duplicate trap)
MIN_SCORE = 4                        # unchanged
_TERMINAL = ('.', '!', '?', '"', "'", ')', ']', '”', '’')


def is_heading(lines, i):
    """Is `lines[i]` a section heading in this dump's `text` field?

    The dump renders sections as a bare title line surrounded by blank lines — there is no `==`
    marker to key on — so the rule is STRUCTURAL, and the structure is the point. A first version
    tested only the line's own shape, and a Codex pass showed a lead whose opening line was short,
    capitalised and unterminated ("A Short Lead") being read as a heading, which returns the REST
    OF THE LEAD as body. That inverts the failure mode: lead exclusion is the sole mitigation for
    the reserved-document fingerprints that do not exist, so a false positive is the expensive
    error, not the cheap one. Hence: blank line before, blank line after, and at least one
    non-empty line of real lead ahead of it. A missed heading keeps lead text, which is then
    refused as a seed and costs supply — the safe direction, on purpose.
    """
    s = lines[i].strip()
    if not s or len(s.split()) > 12:
        return False
    if s.endswith(_TERMINAL) or ". " in s or "; " in s:
        return False
    if not (s[0].isupper() or s[0].isdigit()):
        return False
    if i == 0 or lines[i - 1].strip():                       # must follow a blank line
        return False
    if i + 1 >= len(lines) or lines[i + 1].strip():          # must precede a blank line
        return False
    return any(l.strip() for l in lines[:i])                 # real lead text must precede it


def body(text):
    """-> the article text AFTER the first heading line, or '' if the article has no heading."""
    lines = text.split("\n")
    for i in range(len(lines)):
        if is_heading(lines, i):
            return "\n".join(lines[i + 1:])
    return ""


def chunks(text, min_words, max_words):
    """Paragraph chunks in the registered length window; adjacent short paragraphs merged.

    Overflow is carried, not discarded: emitting `buf[:max_words]` and then clearing the buffer
    silently lost the tail of every long paragraph — a 30-word paragraph followed by a 300-word
    one produced one 220-word chunk and dropped 110 words (Codex 2026-09-05). Successive windows
    are emitted instead, and only a remainder below `min_words` is dropped.
    """
    out, buf = [], []
    for para in text.split("\n"):
        w = para.split()
        if not w:
            continue
        buf += w
        while len(buf) >= max_words:
            out.append(" ".join(buf[:max_words]))
            buf = buf[max_words:]
        if len(buf) >= min_words:
            out.append(" ".join(buf))
            buf = []
    if len(buf) >= min_words:
        out.append(" ".join(buf))
    return out


def stream(limit=None):
    from datasets import load_dataset
    ds = load_dataset(REPO_ID, CONFIG, split="train", streaming=True, revision=REVISION)
    for i, r in enumerate(ds):
        if limit is not None and i >= limit:
            return
        yield r


def verify_lead_rule(n=60, out=None):
    """Condition 2 of the Fable review: the lead rule is hand-verifiable before the scan.

    Dumps, for `n` articles, the last lead line and the detected heading, so the boundary can be
    eyeballed; and reports the share of articles with no heading at all (they contribute nothing).
    """
    rows, no_head = [], 0
    for r in stream(n):
        lines = r["text"].split("\n")
        idx = next((i for i, ln in enumerate(lines) if is_heading(ln)), None)
        if idx is None:
            no_head += 1
            rows.append(dict(title=r["title"], heading=None, last_lead_line=None))
            continue
        prev = next((lines[j].strip() for j in range(idx - 1, -1, -1) if lines[j].strip()), "")
        rows.append(dict(title=r["title"], heading=lines[idx].strip(),
                         last_lead_line=prev[-140:],
                         body_words=len(body(r["text"]).split())))
    rep = dict(n=n, no_heading=no_head, no_heading_frac=round(no_head / n, 4), rows=rows)
    if out:
        Path(out).write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    r = verify_lead_rule(int(sys.argv[1]) if len(sys.argv) > 1 else 60,
                         out=OUT / "wikibody_lead_check.json")
    print(f"no heading: {r['no_heading']}/{r['n']} ({r['no_heading_frac']:.1%})")
    for row in r["rows"][:40]:
        print(f"\n{row['title']}\n  ...lead ends: {row['last_lead_line']}\n"
              f"  heading -> {row['heading']!r}  body_words={row.get('body_words')}")


def scan(limit=None, out=None, log_every=50_000):
    """Full-dump pass in dump order -> JSONL of admitted seed chunks. No sampling, so no
    sampling question: the pilot's 5x prefix bias (13.1 chunks/article on 2001-02 core articles
    vs 4.57 on a shuffled sample) is exactly why a prefix projection is not the registered number.

    One line per admitted chunk: article id and title, form, score, chunk index, text.
    """
    import re as _re, time
    import seeds as S
    pats = {f: _re.compile(S.ROUTE[f], _re.I) for f in FORMS}
    out = Path(out or (OUT / "wikibody_seeds.jsonl"))
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    n_art = n_chunk = n_kept = 0
    per_form = {f: 0 for f in FORMS}
    t0 = time.time()
    with partial.open("w") as fh:
        for r in stream(limit):
            n_art += 1
            cs = chunks(body(r["text"]), S.MIN_WORDS, S.MAX_WORDS)
            n_chunk += len(cs)
            taken = {f: 0 for f in FORMS}
            for ci, c in enumerate(cs):
                # Scored with the article TITLE prepended. `seeds._score` gives a 2x bonus to hits
                # in the first 25 words because an intro passage opens with its title; a body
                # chunk's first 25 words are arbitrary, and a Codex pass showed the identical
                # keyword multiset scoring 6 or 2 purely on where in the chunk it fell -- a
                # positional filter, not a topical one. Prepending the title restores exactly the
                # property the bonus was written for and makes body chunks comparable to the
                # incumbent store's passages. The STORED text is the chunk alone, unchanged.
                scored = f"{r['title']}. {c}"
                for f, p in pats.items():           # first-fit, the draw's priority order
                    sc = S._score(p, scored)
                    if sc < MIN_SCORE:
                        continue
                    if taken[f] >= PER_ARTICLE_CAP:
                        # the cap must not block a LATER form: `break` here discarded every
                        # health+finance chunk after health's third (Codex 2026-09-05)
                        continue
                    taken[f] += 1
                    per_form[f] += 1
                    n_kept += 1
                    fh.write(json.dumps({"aid": r["id"], "title": r["title"], "form": f,
                                         "score": sc, "chunk_i": ci, "text": c}) + "\n")
                    break
            if n_art % log_every == 0:
                el = time.time() - t0
                print(f"  {n_art:,} articles ({n_art/el:.0f}/s, {el/60:.1f}m), "
                      f"{n_chunk:,} chunks, kept {n_kept:,} {per_form}", flush=True)
                fh.flush()
    # Atomic: the destination appears only when the pass finished, and the report is the
    # completion marker `_load_jsonl` requires. Streaming into the final path left a graceful
    # interruption looking exactly like a finished scan, and a dump PREFIX is the sampling bias
    # T2-5 forbids (Codex 2026-09-05).
    partial.replace(out)
    rep = dict(repo=REPO_ID, config=CONFIG, revision=REVISION, licence=LICENCE,
               forms=list(FORMS), min_score=MIN_SCORE, per_article_cap=PER_ARTICLE_CAP,
               length_window=[S.MIN_WORDS, S.MAX_WORDS], lead_excluded=True,
               scored_with_title_prefix=True, route="ROUTE (T2-3, unchanged)",
               complete=limit is None, limit=limit,
               n_articles=n_art, n_body_chunks=n_chunk, n_kept=n_kept, per_form=per_form,
               seconds=round(time.time() - t0, 1), path=str(out),
               bytes=out.stat().st_size)
    (OUT / "wikibody_scan.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


# ---- screening and the build draw ---------------------------------------------------------

def _load_jsonl(path=None, require_complete=True):
    """-> (all rows, deduplicated rows). Refuses a scan that did not finish."""
    import decontam
    path = Path(path or (OUT / "wikibody_seeds.jsonl"))
    if require_complete:
        rp = OUT / "wikibody_scan.json"
        if not rp.exists():
            raise SystemExit(f"{rp} is missing: there is no completed scan to draw from")
        rep = json.loads(rp.read_text())
        for k, want in (("complete", True), ("revision", REVISION), ("config", CONFIG),
                        ("per_article_cap", PER_ARTICLE_CAP), ("min_score", MIN_SCORE),
                        ("scored_with_title_prefix", True)):
            if rep.get(k) != want:
                raise SystemExit(f"scan report {k}={rep.get(k)!r}, this module expects {want!r}")
        if str(Path(rep["path"])) != str(path) or rep["bytes"] != path.stat().st_size:
            raise SystemExit(f"scan report describes {rep['path']} at {rep['bytes']} bytes; "
                             f"{path} is {path.stat().st_size}")
    rows = [json.loads(l) for l in path.open()]
    seen, out = set(), []
    rows.sort(key=lambda r: (-r["score"], r["aid"], r["chunk_i"]))
    for r in rows:
        # BOTH keys: the whitespace-lowercase key and M7's fingerprint. Two texts differing only
        # in punctuation have different lowercase keys and the SAME `exact_u64`, and the external
        # screen does not self-deduplicate (Codex 2026-09-05).
        k = (" ".join(r["text"].split()).lower(), int(decontam.exact_u64(r["text"])))
        if k[0] in seen or k[1] in seen:
            continue
        seen.update(k)
        out.append(r)
    return rows, out


def screen(rows, verbose=True):
    """-> (kept_rows, report). Every registered screen, and a match is REMOVED, not disclosed.

    Query side: the M10 protected index (six + dev + reserved QUERIES, plus admitted COV queries
    and documents) via `m10src/protected10`. Document side: an `Inverted` index over the
    candidates with the six's documents, the four DEV components' documents and the admitted COV
    components' documents streamed against it — M7's direction of travel, the same hash functions
    and the same >= 8/32 near-match threshold the COV admission screen used.

    Reserved-set DOCUMENTS are still not covered and cannot be without opening those corpora
    (`m10/LEDGER.md` §3 W4). Lead exclusion is what bounds that exposure here: DBpedia-entity is
    abstracts and FEVER is introductory sections, and this store carries neither.
    """
    import time
    import decontam
    import protected10
    import cov_screen
    from cov_admit import COMPONENTS

    texts = [r["text"] for r in rows]
    t0 = time.time()
    idx = protected10.build(verbose=verbose)
    q_drop = {i for i, t in enumerate(texts) if protected10.hits(t, idx)}
    if verbose:
        print(f"  query-side: {len(q_drop):,} of {len(texts):,} dropped "
              f"({time.time()-t0:.0f}s)", flush=True)

    inv = decontam.Inverted([decontam.all_grams(t) for t in texts],
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
            print(f"  {name}: {n:,} docs streamed, {per_stream[name]['new_drops']:,} new drops "
                  f"({per_stream[name]['seconds']:.0f}s)", flush=True)

    run_stream("six-docs", decontam.stream_six_docs())
    import devsuite
    for comp in devsuite.COMPONENTS:
        run_stream(f"dev:{comp}", decontam.stream_dev_component_docs(comp))
    for family, comps in COMPONENTS.items():
        for name, repo, rev in comps:
            _qs, ds = cov_screen.load_component(name, repo, rev)
            run_stream(f"cov:{name}", iter(ds))

    drop = q_drop | d_drop
    kept = [r for i, r in enumerate(rows) if i not in drop]
    rep = dict(n_candidates=len(rows), dropped_query_side=len(q_drop),
               dropped_document_side=len(d_drop), dropped_total=len(drop),
               kept=len(kept), per_stream=per_stream,
               seconds=round(time.time() - t0, 1))
    return kept, rep


def draw(per_form, path=None, margin=4, verbose=True):
    """The BUILD's seed selection: top-score-first per form over the screened store.

    Top-score-first is `seeds.draw`'s registered ordering and is kept, so the gate judges the
    population the build actually uses (the Fable pass's condition 6). If the gate fails on it,
    the registered next lever is category-membership routing — never a relaxed floor.

    Only the top `margin * per_form` per form are screened, not the whole store: the screen costs
    ~13 ms per candidate (a measured 39 s for 3,000) and the store holds ~500K, which is 1.8 hours
    to screen candidates that a top-score-first draw would never reach. The margin is 4x against a
    measured ~3% query-side drop rate, and running out is an error, not a silent short draw.
    """
    raw, dedup = _load_jsonl(path)
    pool, by_form = [], {}
    for f in FORMS:
        rows = [r for r in dedup if r["form"] == f]          # already score-sorted
        by_form[f] = len(rows)
        pool += rows[:margin * per_form]
    kept, srep = screen(pool, verbose=verbose)
    out, counts = {}, {}
    for f in FORMS:
        rows = [r for r in kept if r["form"] == f]
        # `(passage_id, text)` pairs, the shape `seeds.draw` returns, so the two stores are
        # interchangeable downstream (Codex 2026-09-05).
        out[f] = [(f"wikipedia-body:{r['aid']}#{r['chunk_i']}", r["text"]) for r in rows[:per_form]]
        if len(rows) < per_form and by_form[f] > margin * per_form:
            raise SystemExit(f"{f}: the {margin}x screening pool left only {len(rows)} of "
                             f"{per_form} after screening -- widen `margin` and re-draw rather "
                             f"than accepting a short draw")
        counts[f] = dict(admitted_in_store=by_form[f], screened=margin * per_form,
                         survived_screen=len(rows), taken=len(out[f]),
                         short=max(0, per_form - len(rows)),
                         min_score_taken=min((r["score"] for r in rows[:per_form]), default=None),
                         max_score_taken=max((r["score"] for r in rows[:per_form]), default=None))
    rep = dict(per_form=per_form, margin=margin, n_raw=len(raw), n_after_exact_dedup=len(dedup),
               screen=srep, counts=counts, store="wikipedia-body",
               repo=REPO_ID, config=CONFIG, revision=REVISION, licence=LICENCE,
               per_article_cap=PER_ARTICLE_CAP, min_score=MIN_SCORE, lead_excluded=True)
    (OUT / "wikibody_draw.json").write_text(json.dumps(rep, indent=1))
    if verbose:
        print(json.dumps(counts, indent=1))
    return out, rep
