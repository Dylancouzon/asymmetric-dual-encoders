# M20 implementation review brief — Astra, 2026-09-16

You are reviewing the M20 implementation on branch `m20-exec` **before** a paid cloud session that
spends a one-shot, irreversible reserved-data access and roughly $200–340 of compute.

## The goal your findings are measured against

M20 must produce, in one ordered cloud session: the registered descriptive **reserved four**
result over an eight-system roster (one access, one transaction), a **BEIR-15** descriptive table,
and a hash-verified **archive**, without damaging any registered evidence or spending the reserved
access twice.

## Essential-only rule (Dylan, 2026-09-15) — this governs your report

Report only what is **essential** to achieving that goal:

- correctness that would change a reported number or a shipped artifact;
- evidence or access-discipline damage (a second protected read, a lost one-shot access, a
  registered constant changed after observation, a number that cannot be reproduced);
- safety, reproducibility;
- scope deviation or over-engineering against the registered plan.

**Out of scope:** wording, style, formatting, preference restructuring, and hypothetical failures
the documents already guard against. If you find nothing essential, say so plainly. Do not
manufacture findings. Rank what you do find P0/P1/P2/P3 and say which are blockers.

## Hard access rules — violating these damages the milestone

- **Do not read** `results/frozen_eval/untouched-*`, `work/dev/cqadup-android.json`,
  `work/dev/cqadup-english.json`, `work/m9reserve`, `work/lotte`, or any reserved qrels cache.
  The reserved four are FEVER, DBpedia-entity, cqadup-android, cqadup-english.
- **No recursive or repo-wide content searches** across `results/` or `work/`. Open only the files
  listed below. If you need one more file, name it in your report rather than globbing for it.
- Read-only. Run no tests, no scripts, nothing that writes.
- End your report with the exact list of files you opened, so the access log can be audited.

## Files to open

Context and authority:
`CLAUDE.md`; `instructions-m20.md`; `m13/RULINGS.md` (R19, R20, R22, R23);
`m13/RESERVED_EXECUTION.md`; `m20/PLAN.md`; `m20/STATUS.md`; `m20/SMOKE.md`.

The registration under review:
`m20/REGISTRATION.md`; `m20/beir15_registry.json`;
`m10/final_run_registry.json` (the `reserved` key only).

The implementation under review:
`m20src/roster.py`; `m20src/beir15.py`; `m20src/archive.py`; `m20src/dbsf_reproduction.py`;
`m20src/tower_rate_benchmark.py`; `m13src/reserved_support.py`; `m13src/reserved_transaction.py`;
`m13src/test_reserved_support.py`; `m8src/pre_encode.py`; `scripts/m13_reserved_cloud.py`;
`scripts/m20_beir15_cloud.py`; `m13src/score13.py` (the `reserved_batch` function only).

Reference for what the new code must not change:
`m12src/qfusion.py`; `m7src/fusion.py`; `m11/release/zero_encoder.py`;
`results/m20_dbsf_reproduction.json`; `results/m20_tower_rate_benchmark.json`.

`git log --oneline -8` and `git diff --stat main...m20-exec` are permitted.

## What changed, so you can aim

1. The reserved roster grows from three systems to eight: `nano-dense`, `zero-dense` (R20),
   `stella-query`, `bge-small-en-v1.5`, `leaf-ir-asym`, `bm25`, `zero+bm25 dbsf@100`,
   `nano+bm25 dbsf@100` (R22). Three of them share one Stella document index.
2. `m10/final_run_registry.json` gained a dated amendment that rewrites `systems_included` and adds
   two keys. `reserved_transaction._prior` no longer requires an unchanged registry hash; it
   reverses the amendment and requires the exact pre-amendment bytes.
3. The two DBSF rows are derived from persisted top-100 runs rather than re-retrieved.
4. Dense retrieval depth is 100, where M12 used 1000-then-truncate.
5. The single 55.2-hour cap became four registered stage caps, and the cloud work was split into
   two controllers with a pod STOP between them.
6. `m8src/pre_encode.py` gained a BEIR-15 batch, revision-pinned corpus loading, hash pinning for
   corpora with no frozen manifest entry, and a projection gate.

## Questions worth your attention

- Can any path in this branch read a reserved query or qrel outside the single tagged transaction?
  `m20src/beir15.py` and `m20src/archive.py` both claim the corpus-only guard entry; is that
  actually sufficient, and is the reserved-corpus exemption in `m8src/paths_guard.py` wide enough
  to matter?
- Does the amendment-reversal check in `_prior` really prove the amendment is confined to the
  roster, or can something slip through it?
- Can a `reserved.crash` resume (R23) re-score or re-open a system that already persisted, or
  produce an aggregate from incomplete outputs?
- Are the derived DBSF rows computed from the same payload and query set as their inputs, and is
  the depth-100 truncation applied before fusion everywhere?
- Is the budget arithmetic right, and does anything let the session exceed the $1,000 ceiling?
- Is the stage ordering (reserved first, then BEIR-15, then archive) actually enforced?

Return a verdict: **GO** or **NO-GO**, with blockers listed first.
