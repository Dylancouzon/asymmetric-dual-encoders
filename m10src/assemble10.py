"""Step 8: assemble the per-form generation outputs into the registered training file, and
compute §0b's `data_cut` count.

`scripts/m10_gen_build.py` writes one `generated_<form>.jsonl` + `manifest_<form>.json` per form
and applies only the two screens that are LOCAL to a row: the frozen rubric's word range and the
own-seed word-5-gram copy check. Everything else it DEFERS here, by design, because the remaining
screens are corpus-wide or index-wide (`screens_deferred` in every manifest). This module runs
them, in the registered order, recording removal counts per screen per form:

  (a) the frozen rubric's word range   (`qfilter.in_range`; idempotent -- the driver applied it)
  (b) exact dedup, per form            (`decontam.exact_u64`, first occurrence kept)
  (c) the FORMS-12 hold-out BY DOCUMENT -- any row whose `doc` is in the manifest's
      `forms12_holdout_docs`. The mandate holds out 500 seed DOCUMENTS per form
      (`instructions-m10.md`:454); the driver partitioned by seed PASSAGE, so a different chunk of
      a held document could still have seeded a build query. That is the lead's ruling and it is a
      real removal, not a no-op: `corpus_loader.load_segments` refuses the whole source if one
      such row survives.
  (d) the protected-index screen and the document-side screen, EXACTLY as `harvest.draw` applied
      them: `protected10.hits` on the query side, and a candidate-side `decontam.Inverted`
      streamed against the six's documents (plus, at harvest parity, the four DEV components' and
      the admitted COV components' documents).
  (e) the own-seed copied-span check, if the manifest says the driver did not already apply it.
  (f) A8 gate 1 as amended by W10 (`corpus10.a8_action`): rate on the post-exact-dedup count,
      above 25% the form keeps only its representatives, below 50,000 retained the form is
      DROPPED and reported. Mean pairwise stella cosine is a diagnostic with no threshold and is
      NOT computed here (it needs an encode); the report carries it as a `todo`.
  (g) the registered per-form quota, a uniform seed-0 draw, applied only when the form is above
      quota. Never a positional prefix -- the file is in seed-draw order.

**The two readings this file takes, both reported so a reader can undo them.**

1. *"the six's documents"* (:445). `harvest.draw` streams the six's documents **and** the four DEV
   components' and the admitted COV components' documents against the same candidate index,
   under §Harvest's "the same screens ... as a generated one". `--streams six` restricts to the
   two literally registered screens; the default `harvest` matches what A3's harvested half was
   actually screened with, so the two halves of A4 are not screened differently. Per-stream drop
   counts are reported either way, so the registered-two number is always recoverable.
2. Exact dedup at (b) is **per form**, because A8's rate denominator is "the post-exact-dedup
   count" of that form. Cross-form duplicates are counted and reported but not removed here --
   `corpus_loader.dedup_segments` removes them globally when the corpus is loaded.

CLI::

    .venv/bin/python m10src/assemble10.py --all                      # the real run
    .venv/bin/python m10src/assemble10.py --forms finance --limit 20000 \\
        --allow-incomplete --out-dir work/m10gen/smoke_assemble      # a smoke
    .venv/bin/python m10src/assemble10.py --data-cut                 # §0b's count
    .venv/bin/python m10src/assemble10.py --gate2                    # print/run A8 gate 2

A smoke may never write the real artifact: `--limit`, `--allow-incomplete` and `--stream-limit`
all refuse to write into `work/m10gen/` (`m10/STATUS.md`, "smokes overwrite real artifacts"), and
the report is written beside its own output (`m10/CODEMAP.md` pitfall 9).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in ("m7src", "m9src", "m10src"):
    sys.path.insert(0, str(REPO / p))

import numpy as np

import protected10                 # BEFORE anything imports m9base (m10/CODEMAP.md pitfall 14)

import corpus10
import decontam
import qfilter

OUT = REPO / "work" / "m10gen"
RESULTS = REPO / "results"

# the seven approved generated forms; a form with no file on disk is SKIPPED and reported
GENERATED_FORMS = ("howto", "argument", "finance", "health", "comparison", "yesno",
                   "conversational")
OUT_NAME = "generated_queries.jsonl"          # corpus_loader.SOURCES["generated"]["path"]
REPORT_NAME = "m10_assemble10.json"
ROW_KEYS = ("text", "form", "seed_id", "doc", "prompt_sha256")
SCREEN_ORDER = ("out_of_rubric_range", "exact_dup", "forms12_holdout_doc",
                "protected_index", "six_documents", "copied_span")


# ------------------------------------------------------------------------------- reading -------

def form_paths(form, gen_dir=OUT):
    gen_dir = Path(gen_dir)
    return gen_dir / f"generated_{form}.jsonl", gen_dir / f"manifest_{form}.json"


def load_form(form, gen_dir=OUT, limit=None, allow_incomplete=False):
    """-> (rows, manifest) or (None, reason). A form with no output file is not an error."""
    jl, mf = form_paths(form, gen_dir)
    if not jl.exists() or not mf.exists():
        return None, f"no {jl.name}/{mf.name} in {gen_dir}"
    man = json.loads(mf.read_text())
    if not man.get("complete"):
        if not allow_incomplete:
            return None, (f"{mf.name}: complete != true -- generation for {form!r} has not "
                          f"finished; refusing to assemble a partial form")
        man = dict(man, _partial_read=True)
    rows = []
    with jl.open() as fh:
        for line in fh:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    for r in rows[:1]:
        missing = [k for k in ROW_KEYS if k not in r]
        if missing:
            return None, f"{jl.name}: rows are missing {missing} -- not a step-8 row"
    return rows, man


# --------------------------------------------------------------------- the per-row screens -----

def local_screens(form, rows, man, seed_text=None):
    """(a) rubric range, (b) exact dedup, (c) FORMS-12 by DOCUMENT, (e) own-seed copied span.

    (e) runs only when the manifest does not already record it. `seed_text` is injected, so this
    is testable without a seed store; production passes `seed_text_for` (or None when the driver
    already applied the screen, which it always does).
    """
    held_docs = set(man.get("forms12_holdout_docs") or ())
    if not held_docs:
        raise SystemExit(f"{form}: the manifest carries no `forms12_holdout_docs` -- the hold-out "
                         f"by document cannot be applied (instructions-m10.md:454)")
    span_done = "copied_span" in (man.get("screens_applied") or {})
    rep = {k: 0 for k in SCREEN_ORDER}
    rep["n_input"] = len(rows)
    rep["copied_span_already_applied_by_driver"] = bool(span_done)
    rep["copied_span_driver_removed"] = (man.get("screens_applied") or {}).get("copied_span")
    rep["forms12_holdout_docs"] = len(held_docs)
    seen, kept = set(), []
    for r in rows:
        q = r["text"]
        if not qfilter.in_range(form, q):
            rep["out_of_rubric_range"] += 1
            continue
        h = int(decontam.exact_u64(q))
        if h in seen:
            rep["exact_dup"] += 1
            continue
        seen.add(h)
        if r.get("doc") in held_docs:
            rep["forms12_holdout_doc"] += 1
            continue
        if not span_done:
            src = (seed_text or {}).get(r["seed_id"])
            if src is None:
                raise SystemExit(
                    f"{form}: the manifest does not record the own-seed copied-span screen and no "
                    f"seed text was supplied for {r['seed_id']!r}. Re-run with --seed-store, "
                    f"which reloads the form's seed store through scripts/m10_gen_build.seed_rows.")
            if corpus10.copied_span(q, src):
                rep["copied_span"] += 1
                continue
        kept.append(r)
    rep["kept_after_local"] = len(kept)
    return kept, rep


def seed_text_for(form):
    """The form's seed passages, {seed_id: text}, read through the DRIVER's own accessor so the
    fallback copied-span screen cannot disagree with the one generation applied."""
    sys.path.insert(0, str(REPO / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("m10_gen_build",
                                                  REPO / "scripts" / "m10_gen_build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {p: t for p, t in mod.seed_rows(form)}


# ------------------------------------------------------- the index screens (harvest parity) -----

def index_screens(texts, streams="harvest", stream_limit=None, verbose=True):
    """-> (drop indices, report). `protected10.hits` on the query side; a candidate-side
    `decontam.Inverted` streamed against the document side, exactly as `harvest.draw` does it."""
    import cov_screen
    t0 = time.time()
    idx = protected10.build(verbose=verbose)
    q_drop = {i for i, t in enumerate(texts) if protected10.hits(t, idx)}
    if verbose:
        print(f"  query-side: {len(q_drop):,} of {len(texts):,} dropped "
              f"({time.time() - t0:.0f}s)", flush=True)
    inv = decontam.Inverted([decontam.query_grams(t) for t in texts],
                            [decontam.exact_u64(t) for t in texts])
    d_drop, per_stream = set(), {}

    def run_stream(name, it):
        t1, n, before = time.time(), 0, len(d_drop)
        for d in it:
            n += 1
            ex, near = inv.match(d, cov_screen.MIN_SHARE)
            d_drop.update(ex.tolist())
            d_drop.update(near.tolist())
            if stream_limit and n >= stream_limit:
                break
        per_stream[name] = dict(streamed=n, new_drops=len(d_drop) - before,
                                seconds=round(time.time() - t1, 1),
                                truncated=bool(stream_limit and n >= stream_limit))
        if verbose:
            print(f"  {name}: {n:,} streamed, {per_stream[name]['new_drops']:,} new drops "
                  f"({per_stream[name]['seconds']:.0f}s)", flush=True)

    six_before = len(d_drop)
    run_stream("six-docs", decontam.stream_six_docs())
    six_drops = len(d_drop) - six_before
    if streams == "harvest":
        import devsuite
        from cov_admit import COMPONENTS
        for comp in devsuite.COMPONENTS:
            run_stream(f"dev:{comp}", decontam.stream_dev_component_docs(comp))
        for _family, comps in COMPONENTS.items():
            for name, repo, rev in comps:
                _qs, ds = cov_screen.load_component(name, repo, rev)
                run_stream(f"cov:{name}", iter(ds))
    elif streams != "six":
        raise SystemExit(f"--streams {streams!r}: expected 'harvest' or 'six'")
    rep = {"streams": streams, "stream_limit": stream_limit,
           "n_candidates": len(texts), "dropped_query_side": len(q_drop),
           "dropped_document_side": len(d_drop),
           "dropped_by_the_six_documents_alone": six_drops,
           "dropped_total": len(q_drop | d_drop),
           "protected10_ident": protected10._ident(),
           "min_share": cov_screen.MIN_SHARE,
           "per_stream": per_stream, "seconds": round(time.time() - t0, 1)}
    return q_drop, d_drop, rep


# ------------------------------------------------------------------------------ the quota -------

def quota_cut(rows, quota, seed=0):
    """-> (rows, report). A UNIFORM seed-0 draw, never a positional prefix: the file is written in
    seed-draw order, so a prefix would take a systematically different seed population."""
    rep = {"quota": quota, "n_before": len(rows), "seed": seed}
    if quota is None or len(rows) <= quota:
        rep["cut"] = False
        rep["n_after"] = len(rows)
        return rows, rep
    rng = np.random.default_rng(seed)
    pick = sorted(int(i) for i in rng.choice(len(rows), size=quota, replace=False))
    out = [rows[i] for i in pick]
    rep["cut"] = True
    rep["n_after"] = len(out)
    rep["removed"] = len(rows) - len(out)
    return out, rep


# ------------------------------------------------------------------------------ assemble --------

def assemble(forms=None, gen_dir=OUT, out_dir=None, limit=None, allow_incomplete=False,
             streams="harvest", stream_limit=None, seed_store=False,
             smoke_ignore_a8_min_retained=False, verbose=True):
    """-> report. Writes `<out_dir>/generated_queries.jsonl` and its report."""
    forms = list(forms or GENERATED_FORMS)
    gen_dir = Path(gen_dir)
    out_dir = Path(out_dir) if out_dir else OUT
    partial = bool(limit or allow_incomplete or stream_limit or streams != "harvest"
                   or smoke_ignore_a8_min_retained)
    if partial and out_dir.resolve() == OUT.resolve():
        raise SystemExit(
            "refusing to write the real work/m10gen/generated_queries.jsonl from a partial run "
            "(--limit / --allow-incomplete / --stream-limit / --streams six). Pass --out-dir "
            "work/m10gen/smoke_assemble (m10/STATUS.md: smokes overwrite real artifacts).")
    t0 = time.time()
    per_form, skipped, kept = {}, {}, {}
    for f in forms:
        rows, man = load_form(f, gen_dir, limit=limit, allow_incomplete=allow_incomplete)
        if rows is None:
            skipped[f] = man
            if verbose:
                print(f"  SKIP {f}: {man}", flush=True)
            continue
        st = seed_text_for(f) if seed_store else None
        k, rep = local_screens(f, rows, man, seed_text=st)
        rep["quota_registered"] = man.get("quota")
        rep["prompt_sha256"] = man.get("prompt_sha256")
        rep["manifest_complete"] = bool(man.get("complete"))
        rep["rows_on_disk"] = man.get("rows_written")
        per_form[f], kept[f] = rep, k
        if verbose:
            print(f"  {f}: {rep['n_input']:,} rows -> {len(k):,} after the local screens",
                  flush=True)
    if not kept:
        raise SystemExit("no generated form could be read -- nothing to assemble")

    # the index screens, once over every form's survivors: the index build and the document
    # streams are per-CORPUS costs, not per-form ones.
    order = [(f, i) for f in kept for i in range(len(kept[f]))]
    texts = [kept[f][i]["text"] for f, i in order]
    q_drop, d_drop, idx_rep = index_screens(texts, streams=streams, stream_limit=stream_limit,
                                            verbose=verbose)
    survivors = {f: [] for f in kept}
    for j, (f, i) in enumerate(order):
        if j in q_drop:
            per_form[f]["protected_index"] += 1
        elif j in d_drop:
            per_form[f]["six_documents"] += 1
        else:
            survivors[f].append(kept[f][i])

    # A8 gate 1, then the quota, per form
    out_rows, dropped_forms = [], []
    seen_global = {}
    for f in sorted(survivors):
        rows = survivors[f]
        per_form[f]["kept_after_index_screens"] = len(rows)
        final_texts, a8 = corpus10.a8_action(f, [r["text"] for r in rows])
        a8["mean_pairwise_stella_cosine"] = None
        a8["todo"] = ("mean pairwise stella-space cosine is a diagnostic with no threshold "
                      "(instructions-m10.md:466); it needs an encode and is not computed here")
        per_form[f]["a8_gate1"] = a8
        if a8["dropped_from_build"] and smoke_ignore_a8_min_retained:
            # SMOKE ONLY, and only on a partial run, which cannot write the real file: a 20K-row
            # smoke is under 50,000 by construction, so the registered drop would leave the quota
            # and write paths unexercised. The gate's READING is unchanged and fully reported;
            # only the 50,000 action is withheld. `a8_action` returns [] when it drops a form, so
            # the pre-drop set is re-derived from the same gate.
            reps, deduped, _r = corpus10.near_dup_gate([r["text"] for r in rows])
            final_texts = reps if a8["action"].startswith("cut to representatives") else deduped
            a8["_smoke_min_retained_not_applied"] = True
        elif a8["dropped_from_build"]:
            dropped_forms.append(f)
            per_form[f]["final_rows"] = 0
            per_form[f]["quota_cut"] = {"skipped": "form dropped by A8 gate 1"}
            continue
        rows = _keep_by_text(rows, final_texts)      # A8's action, applied to the ROWS
        rows, qrep = quota_cut(rows, per_form[f].get("quota_registered"))
        per_form[f]["quota_cut"] = qrep
        per_form[f]["final_rows"] = len(rows)
        for r in rows:
            h = int(decontam.exact_u64(r["text"]))
            seen_global[h] = seen_global.get(h, 0) + 1
        out_rows += [{k: r[k] for k in ROW_KEYS} for r in rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUT_NAME
    with out_path.open("w") as fh:
        for r in out_rows:
            fh.write(json.dumps(r) + "\n")
    return _finish(out_path, out_dir, per_form, skipped, dropped_forms, idx_rep, out_rows,
                   seen_global, forms, gen_dir, limit, allow_incomplete, streams, stream_limit,
                   t0, verbose, partial=partial)


def _keep_by_text(rows, final_texts):
    """Map A8's surviving STRINGS back onto rows, keeping row order and multiplicity."""
    want = {}
    for t in final_texts:
        want[t] = want.get(t, 0) + 1
    out = []
    for r in rows:
        if want.get(r["text"], 0):
            want[r["text"]] -= 1
            out.append(r)
    return out


def sha_file(p, chunk=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _finish(out_path, out_dir, per_form, skipped, dropped_forms, idx_rep, out_rows, seen_global,
            forms, gen_dir, limit, allow_incomplete, streams, stream_limit, t0, verbose,
            partial):
    # `partial` is PASSED, never recomputed: recomputing it here missed
    # `--smoke-ignore-a8-min-retained` and a test's smoke wrote the real
    # `results/m10_assemble10.json` -- m10/CODEMAP.md pitfall 9, caught in flight.
    rep = {
        "_what": "step 8: the generated forms assembled into the registered training file",
        "_screen_order": list(SCREEN_ORDER) + ["a8_gate1", "quota_cut"],
        "_partial": partial,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "forms_requested": forms, "gen_dir": str(gen_dir),
        "limit": limit, "allow_incomplete": allow_incomplete,
        "forms_skipped": skipped, "forms_dropped_by_a8": dropped_forms,
        "index_screens": idx_rep,
        "per_form": per_form,
        "total_rows": len(out_rows),
        "by_form": {f: per_form[f].get("final_rows", 0) for f in sorted(per_form)},
        "cross_form_duplicate_texts": int(sum(v - 1 for v in seen_global.values() if v > 1)),
        "_cross_form_note": ("counted, not removed here: A8's denominator is per form and "
                             "corpus_loader.dedup_segments removes duplicates globally at load"),
        "output": {"path": str(out_path), "sha256": sha_file(out_path),
                   "bytes": out_path.stat().st_size, "row_keys": list(ROW_KEYS)},
        "protected10_ident": protected10._ident(),
        "seconds": round(time.time() - t0, 1),
    }
    (Path(out_dir) / REPORT_NAME if partial else RESULTS / REPORT_NAME).write_text(
        json.dumps(rep, indent=1))
    rep["report_path"] = str(Path(out_dir) / REPORT_NAME if partial else RESULTS / REPORT_NAME)
    if verbose:
        print(json.dumps({k: rep[k] for k in ("total_rows", "by_form", "forms_skipped",
                                              "forms_dropped_by_a8", "seconds")}, indent=1),
              flush=True)
        print(f"wrote {out_path} and {rep['report_path']}", flush=True)
    return rep


# ------------------------------------------------------------------------------ §0b data cut ----

def data_cut(head_per_source=None, arms=("A2", "A3", "A4"), report=None, verbose=True):
    """§0b: the three post-screen unique-text counts and their `min`.

    "min of the three post-decontamination unique-text counts" (`m10/screen_registry.json`
    `data_cut.rule`). *Unique* is what `corpus_loader.load_segments` reports after
    `dedup_segments`, which is the same read a cut arm will make -- so the number cannot disagree
    with the corpus it is applied to. **This writes no registry**: the lead registers the number.
    """
    import corpus_loader as CL
    out = {"_what": "§0b: A2/A3/A4 post-screen unique-text counts and their min",
           "_rule": CL._registry_data_cut(None).get("rule"),
           "_registers_nothing": "the lead registers the number in m10/screen_registry.json",
           "head_per_source": head_per_source, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "arms": {}}
    for arm in arms:
        srcs = list(CL.ARM_SOURCES[arm])
        t0 = time.time()
        try:
            segs, man = CL.load_segments(srcs, head_per_source=head_per_source, verbose=verbose)
            out["arms"][arm] = {"sources": srcs, "unique_texts": man["n_rows"],
                                "duplicates_removed": man["n_duplicates_removed"],
                                "by_form": man["by_form"], "corpus_sha256": man["sha256"],
                                "seconds": round(time.time() - t0, 1)}
        except SystemExit as e:
            out["arms"][arm] = {"sources": srcs, "error": str(e),
                                "seconds": round(time.time() - t0, 1)}
        if verbose:
            a = out["arms"][arm]
            print(f"  {arm}: " + (f"{a['unique_texts']:,} unique texts" if "unique_texts" in a
                                  else f"UNAVAILABLE -- {a['error'].splitlines()[0]}"), flush=True)
    have = {a: v["unique_texts"] for a, v in out["arms"].items() if "unique_texts" in v}
    out["counts"] = have
    out["complete"] = len(have) == len(arms) and not head_per_source
    out["data_cut"] = min(have.values()) if out["complete"] else None
    if not out["complete"]:
        out["_why_no_number"] = ("a head-limited or incomplete read is a smoke, not the count; "
                                 "every arm must load in full and generation must be done")
    # a head-limited read is a SMOKE and must not land on the real report path
    # (`m10/CODEMAP.md` pitfall 9)
    report = Path(report) if report else RESULTS / (
        "m10_data_cut_smoke.json" if head_per_source else "m10_data_cut.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(out, indent=1))
    if verbose:
        print(f"wrote {report}: data_cut = {out['data_cut']}", flush=True)
    return out


# ------------------------------------------------------------------------------ A8 gate 2 -------

def gate2_command(gen_dir=OUT, out_path=None):
    """-> (argv, forms present). A8 gate 2 re-run for the generated forms that exist."""
    import a8_gate2
    out_path = Path(out_path) if out_path else Path(gen_dir) / OUT_NAME
    present = []
    if out_path.exists():
        with out_path.open() as fh:
            present = sorted({json.loads(l)["form"] for l in fh})
    else:
        present = [f for f in GENERATED_FORMS if form_paths(f, gen_dir)[0].exists()]
    forms = list(a8_gate2.FORMS_TODAY) + [f for f in present if f in a8_gate2.FORMS_PENDING]
    argv = [str(REPO / ".venv" / "bin" / "python"), str(REPO / "m10src" / "a8_gate2.py"),
            "--forms", *forms]
    return argv, present


def run_gate2(gen_dir=OUT, dry_run=False):
    argv, present = gate2_command(gen_dir)
    print("A8 gate 2 (report-only diagnostic, no action):")
    print("  generated forms present:", ", ".join(present) or "none")
    print("  " + " ".join(argv))
    print("  NOTE: the generated texts need teacher vectors first --")
    print("        .venv/bin/python m10src/targets10.py --sources generated")
    if dry_run:
        return {"argv": argv, "forms_present": present, "ran": False}
    r = subprocess.run(argv, cwd=str(REPO))
    return {"argv": argv, "forms_present": present, "ran": True, "returncode": r.returncode}


# ------------------------------------------------------------------------------------ CLI -------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--forms", nargs="+", default=None,
                    help="generated forms to assemble (default: every one present)")
    ap.add_argument("--all", action="store_true", help="every approved generated form")
    ap.add_argument("--gen-dir", default=str(OUT))
    ap.add_argument("--out-dir", default=None,
                    help="where generated_queries.jsonl and its report go; a partial run MUST "
                         "pass one outside work/m10gen")
    ap.add_argument("--limit", type=int, default=0, help="rows per form (smoke only)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="assemble a form whose manifest says complete != true (smoke only)")
    ap.add_argument("--streams", default="harvest", choices=("harvest", "six"),
                    help="document-side streams: harvest parity (default) or the six only")
    ap.add_argument("--stream-limit", type=int, default=0,
                    help="cap documents streamed per stream (smoke only)")
    ap.add_argument("--smoke-ignore-a8-min-retained", action="store_true",
                    help="smoke only: report A8's 50,000 drop without applying it, so the quota "
                         "and write paths are exercised at smoke scale (forces a partial run)")
    ap.add_argument("--seed-store", action="store_true",
                    help="reload seed passages for the fallback copied-span screen")
    ap.add_argument("--data-cut", action="store_true", help="compute §0b's count instead")
    ap.add_argument("--head-per-source", type=int, default=0, help="--data-cut smoke device")
    ap.add_argument("--report", default=None, help="--data-cut: where the report goes")
    ap.add_argument("--gate2", action="store_true", help="run A8 gate 2 for the forms present")
    ap.add_argument("--dry-run", action="store_true", help="--gate2: print the command only")
    a = ap.parse_args()

    if a.data_cut:
        data_cut(head_per_source=a.head_per_source or None, report=a.report)
        return
    if a.gate2:
        run_gate2(gen_dir=Path(a.gen_dir), dry_run=a.dry_run)
        return
    forms = a.forms
    if forms is None and not a.all:
        forms = [f for f in GENERATED_FORMS if form_paths(f, Path(a.gen_dir))[0].exists()]
        print("forms present:", ", ".join(forms) or "none", flush=True)
    assemble(forms=forms, gen_dir=Path(a.gen_dir), out_dir=a.out_dir, limit=a.limit or None,
             allow_incomplete=a.allow_incomplete, streams=a.streams,
             stream_limit=a.stream_limit or None, seed_store=a.seed_store,
             smoke_ignore_a8_min_retained=a.smoke_ignore_a8_min_retained)


if __name__ == "__main__":
    main()
