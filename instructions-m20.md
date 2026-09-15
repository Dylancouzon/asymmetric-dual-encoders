# M20 — reserved four, official release and upstream FastEmbed

**Created 2026-09-15 under owner ruling R20 (Dylan).** M20 takes the protected-access, upstream
and broad-evaluation work that M14 was carrying, so that M14 can ship a research preview first.
This file creates the mandate; it does not authorize execution. A new execution session must
verify the gates below before spending access.

M20 inherits ownership from R19 (post-trigger execution moved out of M13) and R20 (preview
rescope). Neither ruling reinterprets the reserved stage, widens its scope, or changes its alpha.

## Inherited state to verify first

- M13's six-set decision is complete and stands; it triggered the registered descriptive reserved
  stage. `results/m10_final_run.json` ends `INCOMPLETE_RESERVED`.
- **Reserved access must still be unspent.** Verify this from the M13 closure, `m14/HANDOFF.md`
  and the absence of a `m8-reserved-spent` receipt. Never infer it from the
  `INCOMPLETE_RESERVED` label alone, and re-verify after M14's preview work.
- M14 published a research preview of `DylanCouzon/constella-nano` from the same frozen bytes.
  The preview changed no measured result. Cite its recorded commit/revision when describing what
  was already shown publicly.

## Deliverables

1. **Dense-Zero roster amendment.** Before opening any reserved payload, make and push a dated,
   pre-observation registry/ledger amendment adding released `constella-zero` as a query encoder
   paired with the same pinned Stella document tower used by M13 Nano. The full inherited ruling,
   including the prohibition on substituting Zero+BM25 and the instruction to document Zero's
   omission rather than reopen a spent access, is in `instructions-m14.md` under
   "Reserved-four inheritance and Zero comparison". It carries over verbatim.
2. **Executor extension.** Extend and test `reserved_support.py`, `reserved_transaction.py`,
   `pre_encode.py` and `scripts/m13_reserved_cloud.py` from M13's tested three-system base to four
   systems, without touching protected data, and bind the new hashes to the transaction. M13's
   committed base must not be executed unchanged.
3. **One reserved transaction** on the retained A100: FEVER, DBpedia-entity, cqadup-android and
   cqadup-english, against M13 Nano, dense Zero, BGE-small and LEAF asym. Zero reuses Nano's
   Stella document shards; no second corpus-scale encode. Produce
   `results/m13_reserved_preencode.json`, `results/m13_reserved_manifest.json`,
   `results/m13_reserved_run.json`, the controller receipt and cost receipt, confirm the pod
   returned to `EXITED`, and only then update `results/m10_final_run.json` to end `COMPLETE`.
   All rows are descriptive, `alpha = 0`, with FEVER's double-contamination caveat disclosed.
4. **Broad descriptive BEIR-18 validation**, clearly labelled, with exact dataset/version/config
   pins and overlap caveats. It must not be mixed into or reinterpret M13's registered gates.
5. **Official release**: final model card as a later revision of `DylanCouzon/constella-nano`,
   carrying the reserved and BEIR-18 results, private-first upload discipline, LFS-oid
   verification and downloaded-byte verification before the public revision.
6. **One clean upstream FastEmbed PR** from current upstream main with all three native entries
   and reference-derived canonical vectors. It must not include the unrelated #703 padding fix,
   and it must not be the M14 preview branch merged wholesale.
7. Only after Hub download verification may M20 recommend storage retirement. Pods and volumes
   remain STOP-only; deletion authority is not inferable from any handoff.

## Gates and budget

- Two independent reviewers before the reserved execution, per the repository review policy. The
  brief names files, forbids recursive searches, carries the reserved read-exclusion, and its
  access log is audited before findings are accepted.
- `m13/RESERVED_EXECUTION.md` holds the tested implementation notes and the conservative cap of
  55.2 hours at at most $1.6636111111/hour including storage. The recorded $1,000 cloud ceiling is
  a ceiling, not a target.
- The reserved evaluation is a single transaction and a single access. A crash resumes only at a
  missing shard; a rerun is not a second access.
- Entry point for artifact paths, hashes, restore commands and the full execution checklist
  remains `m14/HANDOFF.md`, read together with R20's rescope.
