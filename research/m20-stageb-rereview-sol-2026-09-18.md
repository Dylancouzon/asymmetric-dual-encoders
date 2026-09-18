# M20 stage-B P1 re-review — Sol, 2026-09-18

Reviewer: Codex `gpt-5.6-sol`, read-only. Brief: `work/m20/logs/sol_rereview_brief.md`.
Transcript: `work/m20/logs/sol_rereview.log`. Reviewed at `cca23bb`.

# Verdict: NO-GO on entering stage B at `cca23bb`

Three essential P1 defects remain.

1. **P1 — A crash during the first export of a dataset is unrecoverable.**  
   [m13src/reserved_support.py:547](/home/dylan/asymetric-dual-encoders/m13src/reserved_support.py:547)

   The two gzip files are replaced before the dataset record is persisted at lines 556–561. If the process dies after either replacement but before the record replacement, the next process has neither a cache entry nor a recorded archive entry and refuses at line 535. Completing would then require another protected read. This breaks the registered crash/access contract after access has been spent.

   The added archive test resets the cache only after a fully completed export; it does not exercise this mid-export window.

2. **P1 — Archive reuse is not bound to the registered payload.**  
   [m13src/reserved_support.py:487](/home/dylan/asymetric-dual-encoders/m13src/reserved_support.py:487)

   `_archive_intact` verifies files only against sizes and hashes supplied by the same mutable `payload_archive.json`. The reuse path does not compare `entry["payload_hashes"]` with `cfg.manifest_path` or derive the registered semantic hashes from the archived content. A self-consistent stale or substituted archive is therefore accepted and incorporated into the reserved result.

   The reuse test only reuses the exact archive created moments earlier, so it does not demonstrate registered-payload authentication.

3. **P1 — Publish authenticates pointers, not the persisted reserved result.**  
   [m13src/reserved_transaction.py:281](/home/dylan/asymetric-dual-encoders/m13src/reserved_transaction.py:281)

   `publish` checks two fields supplied by `RESULT`: its claimed manifest hash and prior six-set hash. It does not verify the manifest against the tagged BEGIN commit, validate the result’s required structure or identity fields, or compare its summary with the authenticated atomic system outputs. Consequently, altered numbers can retain those two hashes and be inserted into the six-set result.

   The positive test actually exposes this: its deliberately incomplete synthetic result—containing only `status`, a malformed `contrasts`, and the two hashes—is accepted and published at [m13src/test_reserved_support.py:440](/home/dylan/asymetric-dual-encoders/m13src/test_reserved_support.py:440).

The memoization itself is acceptable under the stated immutable, first-open hash-checked contract. The handoff allowlist is narrowly limited to the one receipt; `_preencode` validates it before BEGIN, and BEGIN hashes and commits it.

No test was obviously vacuous, but the archive tests omit the decisive mid-export and mismatched-archive cases, while the finalization success test demonstrates insufficient authentication.

Access log:

- `CLAUDE.md` — whole
- `m20/REGISTRATION.md` — partial, relevant contract sections
- `m13/RULINGS.md` — partial, R23/R24 and surrounding material
- `research/m20-stageb-review-astra-2026-09-18.md` — whole
- `m13src/reserved_support.py` — partial, relevant scoring/archive/summarization sections
- `m13src/reserved_transaction.py` — whole
- `m13src/test_reserved_support.py` — partial, all seven added tests and nearby context
- Git inspections: `git log --oneline -6`; `git show cca23bb` limited to the permitted files

