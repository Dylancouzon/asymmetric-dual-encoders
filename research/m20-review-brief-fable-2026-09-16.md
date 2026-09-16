# M20 re-review brief — Fable, 2026-09-16

Round 2 of two. Round 1 (Astra, `research/m20-codex-impl-review-2026-09-16.md`) returned NO-GO with
seven P1 blockers; all are fixed on branch `m20-exec` and triaged in `m20/REVIEW_TRIAGE.md`.

You are reviewing **before** a paid cloud session that spends a one-shot, irreversible reserved-data
access and roughly $335 of compute.

## The goal your findings are measured against

M20 must produce, in one ordered sequence: the registered descriptive **reserved four** result over
an eight-system roster (one access, one transaction), a **BEIR-15** descriptive table, and a
hash-verified **archive**, without damaging any registered evidence or spending the reserved access
twice.

## Essential-only rule (Dylan, 2026-09-15) — this governs your report

Report only what is **essential** to that goal:

- correctness that would change a reported number or a shipped artifact;
- evidence or access-discipline damage (a second protected read, a lost one-shot access, a
  registered constant changed after observation, a number that cannot be reproduced);
- safety, reproducibility;
- scope deviation or over-engineering against the registered plan.

**Out of scope:** wording, style, formatting, preference restructuring, and hypothetical failures
the documents already guard against. If you find nothing essential, say so plainly. Do not
manufacture findings. This is round 2 of a two-round budget: name blockers, and record anything
smaller as debt rather than as a reason to hold the run.

## Hard access rules

- **Do not read** `results/frozen_eval/untouched-*`, `work/dev/cqadup-android.json`,
  `work/dev/cqadup-english.json`, `work/m9reserve`, `work/lotte`, or any reserved qrels cache.
  The reserved four are FEVER, DBpedia-entity, cqadup-android, cqadup-english.
- **No recursive or repo-wide content searches** across `results/` or `work/`. Open only the files
  listed below; name any extra file you need in your report.
- Read-only. Run no scripts and no tests that write. `git log`, `git diff` and reading files are
  fine.
- End your report with the exact list of files you opened.

## Your two jobs

### 1. The access boundary, which round 1 could not certify

Round 1 was not given `m8src/paths_guard.py` and therefore could not answer the most important
question. You are. Read it, and answer concretely:

- Can any code path on this branch read a reserved **query or qrel** outside the single tagged
  transaction? Consider `m20src/beir15.py` and `m20src/archive.py`, which both import
  `m8src/pre_encode.py` and so claim the **corpus-only** entry.
- Is the reserved-corpus exemption in the guard (`_reserved_corpus_cache`,
  `_reserved_cache_metadata`, `check_dataset`) narrow enough that the corpus-only claim cannot
  reach labels?
- `m13src/reserved_support.py:corpus_for` now loads the reserved corpus **inline**, under the
  `m13src.score13` claim, because importing `pre_encode` would have raised on a second claim. Is
  that inline path correct and still bounded to the corpus config?
- Does `m13src/reserved_support.py:export_reserved_payload_archive` introduce any protected read
  that would not have happened anyway? It runs inside the transaction, after scoring, and calls
  `load_payload` on each of the four datasets.

### 2. Whether the seven fixes actually fix what they claim

`m20/REVIEW_TRIAGE.md` states what each fix was. Check the code, not the table. In particular:

- **Budget.** Is `237.2 h × $1.8025 − 55.2 × $1.6636111111 = $335.72` the right conservative
  charge, and does `scripts/m13_reserved_cloud.py` actually refuse when the live quote breaks it?
- **Stage clocks.** Does one absolute deadline really reach all three tower processes, and does the
  `timeout` wrapper bound each stage separately? Can the projection in `m8src/pre_encode.py`
  under-project, for example by dividing by a rate it has not measured or by mis-ordering towers?
- **Archive.** Can `m20src/archive.py` still produce a manifest that verifies while the archive is
  incomplete or stale? Does the reserved-label export land in the exact layout the builder expects?
- **Result delivery.** Can `scripts/m20_beir15_cloud.py` still report `PASSED` without the BEIR-15
  table being durable?
- **BEIR-15 resume.** Can a fused row still be derived from a run its input did not produce? Can
  two systems on one corpus disagree about the query set without being caught?

## Files to open

Context: `CLAUDE.md`; `instructions-m20.md`; `m13/RULINGS.md` (R19, R20, R22, R23);
`m20/REVIEW_TRIAGE.md`; `research/m20-codex-impl-review-2026-09-16.md`; `m20/STATUS.md`;
`m20/SMOKE.md`.

Registration: `m20/REGISTRATION.md`; `m20/beir15_registry.json`;
`m10/final_run_registry.json` (the `reserved` key only).

Implementation: `m8src/paths_guard.py`; `m20src/roster.py`; `m20src/beir15.py`;
`m20src/archive.py`; `m13src/reserved_support.py`; `m13src/reserved_transaction.py`;
`m13src/test_reserved_support.py`; `m8src/pre_encode.py`; `scripts/m13_reserved_cloud.py`;
`scripts/m20_beir15_cloud.py`; `m13src/score13.py` (`reserved_batch`, `reserved_only_run` and
`main` only).

Reference: `m12src/qfusion.py`; `m7src/fusion.py`; `results/m20_tower_rate_benchmark.json`;
`results/m20_dbsf_reproduction.json`.

`git log --oneline -12` and `git diff --stat main...m20-exec` are permitted.

Return a verdict: **GO** or **NO-GO**, blockers first, debt separately.
