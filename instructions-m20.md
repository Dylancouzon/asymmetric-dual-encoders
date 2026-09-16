# M20 — reserved four and BEIR-13 descriptive evaluation

**Created 2026-09-15 under owner ruling R20 (Dylan); rescoped 2026-09-16 under R22
(`m13/RULINGS.md`).** M20 is evaluation-only: one cloud session that spends the registered reserved
access and runs the broad descriptive BEIR-13 validation, then archives what it produced. The
official Nano release moved to `instructions-m22.md`; the upstream FastEmbed PRs moved to
`instructions-m23.md`. The original 2026-09-15 mandate text is preserved in git.

M20 inherits ownership from R19 (post-trigger execution moved out of M13), R20 (preview rescope)
and R22. None of them reinterprets the reserved stage, widens its estimands, or changes its alpha.

## Inherited state to verify first

- M13's six-set decision is complete and stands; it triggered the registered descriptive reserved
  stage. `results/m10_final_run.json` ends `INCOMPLETE_RESERVED`.
- **Reserved access must still be unspent.** Verify from the M13 closure, `m14/HANDOFF.md` and the
  absence of an `m8-reserved-spent` tag on origin. Never infer it from the `INCOMPLETE_RESERVED`
  label alone. Verified unspent on 2026-09-16 at planning time; re-verify at execution time.
- M14 published a research preview of `DylanCouzon/constella-nano` (weights revision `6bb167dc`)
  from the same frozen bytes. The preview changed no measured result.
- The retained A100 pod `k3aee2m68765em` and its two sibling volumes are STOP-only.

## Deliverables

1. **Pre-observation registration** (`m20/REGISTRATION.md`, `m20/beir13_registry.json`, and the
   dated `_amended_2026_09_16` block in `m10/final_run_registry.json`), pushed before any protected
   read. It adds to the reserved roster, without changing R1/R2, NDO-3 weights, B, seed or alpha:
   dense `constella-zero` (R20), `stella-query`, `bm25`, `zero+bm25 dbsf@100` and
   `nano+bm25 dbsf@100` (R22). Zero and Stella-query reuse Nano's Stella document shards; the
   inherited prohibition on *substituting* Zero+BM25 for the dense Zero row stands. If evidence
   shows the payload was opened before the amendment was pushed, document the omission; never reopen.
2. **Executor extension.** Extend and test `m13src/reserved_support.py`,
   `m13src/reserved_transaction.py`, `m8src/pre_encode.py` and `scripts/m13_reserved_cloud.py`
   from the tested three-system base to the eight-system roster and the BEIR-13 datasets, without
   touching protected data, and bind the new hashes to the transaction. Per-query nDCG@10 rows are
   persisted for every system and dataset. M13's committed base must not be executed unchanged.
3. **One reserved transaction** on the retained A100: FEVER, DBpedia-entity, cqadup-android and
   cqadup-english. Produce `results/m13_reserved_preencode.json`,
   `results/m13_reserved_manifest.json`, `results/m13_reserved_run.json`, the controller receipt
   and cost receipt, confirm the pod returned to `EXITED`, and only then update
   `results/m10_final_run.json` to end `COMPLETE`. All rows descriptive, `alpha = 0`, FEVER's
   double-contamination caveat disclosed.
4. **BEIR-13 descriptive validation** in the same cloud session, clearly labelled, with the pins,
   exclusions and per-row contact labels of `m20/beir13_registry.json`. Ten of the thirteen are
   already scored (six-set, reserved four); only the remaining corpora are encoded. It must not be
   mixed into or reinterpret M13's registered gates. Result: `results/m20_beir13_run.json` plus
   per-query rows under `results/m20_beir13_scores/`.
5. **Archive.** Raw corpora/queries/qrels and fp16 Stella document vectors for every evaluated
   dataset, hash-manifested in `results/m20_archive_manifest.json`, copied to both object storage
   and `D:` and verified by re-hash. Stopped Runpod volumes are not the archive.
6. **Ledger boundary.** Record that the reserved four, once spent, are known-test for future
   projects and unusable for re-deciding anything in this project.

## Gates and budget

- Two independent reviewers (Astra, then Sol, sequentially) before the cloud session. Each brief
  names files, forbids recursive searches, carries the reserved read-exclusion, and states the
  essential-only rule; audit the access log before accepting findings.
- `m13/RESERVED_EXECUTION.md` holds the tested implementation notes and the reserved cap of
  55.2 hours at at most $1.6636111111/hour including storage. BEIR-13 adds roughly 18M Stella
  documents; register its own hour cap in `m20/REGISTRATION.md` before renting. The $1,000 cloud
  ceiling is a ceiling, not a target.
- The reserved evaluation is a single transaction and a single access. A crash resumes only at a
  missing shard or system; a rerun is not a second access. BEIR-13 is not protected access but
  runs under the same manifest, hash and atomic-output discipline.
- Entry point for artifact paths, hashes, restore commands and the reserved checklist remains
  `m14/HANDOFF.md` (steps 1–3), read together with R20 and R22.

## Exit

`m20/STATUS.md` states: reserved four complete with receipt, BEIR-13 complete, archive verified at
both targets, pod `EXITED`, and the two items M22 needs — the result files and the archive manifest.
