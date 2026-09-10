# M10.4 — THE DECISION LOCK (half B). **NOT LOCKED. NOT NEARLY DONE.**

Split from the recipe lock on Dylan's ruling, 2026-09-10 (`m10/M102_LOCK.md` header). This half
decides **how the finished model is judged**: the four C-conjuncts, their sequence, the bridge, the
reserved batch, and LoTTE read #1.

## The hard gate

**No six-dataset evaluation, no `m10-six-spent` tag, and no LoTTE read may happen until this file is
locked and an adversarial review returns clean.** That gate is what keeps the split honest: every
decision here is still fixed before the numbers it governs exist. Losing the gate turns a legitimate
sequencing choice into post-hoc judging.

## Its artifacts

| file | state |
|---|---|
| `m10/final_run_registry.json` | written, heavily reviewed, **not locked** |
| `m10/LOTTE_LOCK.md` | written, **not locked** |
| `m10src/final10.py` + tests | written, nine adversarial rounds, **not locked** |
| the SCORING path | **does not exist** — see below |

## Why this was deferred, and what must come first

`m9src/final9.py`:348 raises `SCORING PATH NOT IMPLEMENTED`. M9 left step 4's encoding unwritten and
its close-out never ran, so **there is no scoring path to reuse for either milestone.** Nine review
rounds kept finding gaps in decision code whose only caller is unwritten — that is not a care
problem, it is an ordering problem.

**Write the executor first, then lock this.** `m7src/final_run.py:verify_and_load` loads and
validates the frozen rows; `m12src/run_six_dbsf.py`:39-84 is a second six-set reference;
`m9src/final9.py`'s access machinery (`seal_protected_paths`, `acquire_lock`, `spent_tag_exists`,
`preflight`, `spend_access`) is reusable retargeted. **`score_set` is NOT reusable as-is** — it
scores M7's int8/fp16 table, BM25 and the symmetric teacher, not a nano student.

## OPEN FINDINGS — round 9, applied to NOTHING yet

Both reviewers, 2026-09-10. Tagged as they reported them. **This list is the backlog; it is not a
summary of resolved work.**

| # | finding | tag |
|---|---|---|
| 1 | Post-tag continuation's 4th condition is unevaluable in the case it was written for: a crash before the FIRST dataset write persists no registry hash to compare. Compare against `git show <BEGIN>:m10/final_run_registry.json` instead. Also persist an authenticated EMPTY run manifest (BEGIN commit, registry hash, zero datasets) *before* pushing the tag | lock text |
| 2 | The continuation state necessarily has a dirty tree (`results/` is not gitignored) and the inherited preflight refuses any `--porcelain` output. Whitelist exactly the scores file, as M7's `ALLOWED_DRIFT` does | lock text |
| 3 | Executor deliverable "M7's preflight standard" points at code that BREACHES the inherited rule: `FINAL_LOCK.md`:36 says preflight must not open six-set queries or qrels, and `m7src/final_run.py`:476-483 opens `frozen_eval/{ds}.json`. Satisfiable via `eval_manifest.json.datasets.<ds>.qids_sha256` | lock text |
| 4 | **The restated screen-registry freeze rule still does not hold.** Mechanical diff from the first contrast artifact: **8 re-stamps, not the 6 I wrote**, and `arms` and `rules` both changed after contrasts existed (`E-bs128.warmup_examples`, `E_warmup_parity`). All twelve contrast statistics stayed bit-identical. Honest form: *no field consumed by a COMPUTED contrast changes; unrun arms, descriptive records and unrun families' rules may be added or corrected pre-observation, dated* | lock text |
| 5 | **The LoTTE veto has a margin (0.004) but NO METRIC and no slice list.** Success@5 and nDCG@10 at 0.004 are different rules; choosing later would be post-hoc. M9's mandate required metrics committed before the batch | lock text |
| 6 | **M9's unsatisfiable bridge is still armed** — `m9/final_run_registry.json` 0.0003, `FINAL_LOCK.md`:45, `final9.py`:353 — and M10's lock push is M9's close-out trigger. M9 hits the same impossible rule M10 just withdrew | lock text, M9 scope |
| 7 | `min_free_gb: 120`, inherited from M9 and re-registered "verbatim", is MISSING from M10's registry. Satisfiable (~600 GB free); restore rather than silently drop | lock text |
| 8 | The extension range is stated 11–36 but its own table gives `floor((1000−98)/20)=45` and `floor((1000−267)/63)=11` → 11–45. Correct the range or show the correlated-rate calculation | lock text |
| 9 | `--recover` pins the registry but not the CODE, so a change between run and recover alters the bound silently. Pin HEAD, or persist and replay `draw_plan_sha256`/`draws_sha256` | lock text |
| 10 | `LOTTE_LOCK.md`'s manifest heading still says "after training ends" — the post-build wording the file itself withdrew — and promises four pending cells where three exist. `executed: NO` must also flip after read #1 despite the "only edit" clause | lock text |
| 11 | `M102_LOCK.md` half B references and the registry disagreed on `score_set`; half A no longer claims anything about half B, but check the registry's own text | lock text |
| 12 | Reserved batch (~9 A100-hours) has no budget line; M9's "rehearsal on open sets" was dropped unlisted | lock text |
| 13 | Mutation survivor: the decision field's `abs(v) > 1.0` bound has no test at 1.5 | code |
| 14 | Executor deliverables (NOT lock blockers): verify the real comparator file hash; bind score rows to checkpoint/system identity; compare scored qid sets against the frozen query set (pytrec_eval silently omits a run qid with no qrel); persist string qids and the run-time registry hash; iterate `partitions.all6`, never `bench/core.py:DATASETS`, which defaults to five datasets without `trec-covid` | executor |

**One number of mine to correct on sight (finding, not yet applied):** the bridge amendment says the
nDCG@10 quantum is 0.002633, ~9× the withdrawn 0.0003 tolerance. That is the BINARY figure.
pytrec_eval uses **linear** gain, so the true minimum is **0.001317** — still 4.4× the withdrawn
tolerance, so the conclusion stands and the published number does not.

## What was applied in round 9 before the split

`decide()` gained an `UNSCORABLE` sentinel — one NaN in a comparator row previously made a reached
conjunct raise and discarded verdicts already established, after the access was spent. Tests added
for the registry-hash stamp, full draw-plan index coverage, and the qid digest depending on qids.
`E-bs32` became a real A100 arm (that one belongs to half A, and shipped with it).
