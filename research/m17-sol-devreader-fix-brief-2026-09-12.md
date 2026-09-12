# M17 Sol review brief — the fix of Astra's nine dev-suite reader findings (2026-09-12)

You are an adversarial, READ-ONLY reviewer. Do not edit any file. Do not run anything that
writes under `results/`, `work/`, or `m17/`. Report findings only; do not confirm.

## Context

Repository `/home/dylan/asymetric-dual-encoders`, branch `m17-zero-v1.1-planning`, HEAD `4824f84`.
M17 is ON THE CLOCK, registry status `LOCKED_EXECUTABLE`. The next step is the single declared,
irreversible read of the untrained vocabulary export V0 on the pinned M7/M8 development suite
(registry `untrained_vocab_export_v0`, `reads: 1`) through `m17src/evaluate.py dev_suite_read`.

Commit `2c5321c` wired the reader. Codex Astra reviewed it (findings: final block of
`work/m17/logs/astra_devreader_review.log`) — six P1 / three P2. Commit `4824f84` claims to fix all
nine; dispositions are tabulated in `m17/REVIEW.md` (latest entry). Your job: break the FIX.
Does each finding actually close, and did the fix introduce a new way to spend the one read on a
wrong or unrecoverable number?

Executor's claims (unverified by me):

1. Gate `_require_dev_suite` requires `LOCKED_EXECUTABLE` and the three `lock.executed.v0_export`
   digests.
2. `_verify_v0_bundle` recomputes digests with `export.gate_artifact` and compares to the lock
   before constructing the encoder.
3. `_pool_identity` reproduces `m7src/heldout._verify_pool`'s checks (meta, byte size, `vecs.f16`
   sha256, row count) against `_pinned.pool` WITHOUT calling `pool.build()`; `_teacher_doc_vecs`
   hashes the memmapped cache file and compares to the cache's recorded digest
   (`teacher.PROVENANCE` / `shards.json`); refuses a cache with missing shards rather than encoding.
   Executor's own caveat: shard digests may be trust-on-first-use, hashing themselves.
4. `_query_identity` hashes ordered `(qid, text)` pairs; VERIFIED against the manifest for the two
   held-out components (`qids_ordered_sha256`, `qtexts_ordered_sha256`); RECORDED ONLY for the
   four text-backed components because the manifest pins no ordered query identity for them.
5. Full preflight (destination, digests, all six components, both vector sources) before the
   first score; a mid-scoring failure writes `state: failed` with `completed_components`.
6. `out` mandatory; refuses if result or receipt exists; receipt `started` → per-component →
   `complete` with status, digests, manifest hashes, query/vector digests, git sha, UTC,
   `reads: 1`; result JSON carries the same provenance.
7. `_enforce_production_surface`: complete pinned list, int8 resident, `serving.prefetch` depth,
   registered manifest; `_require_pinned_fields`; overrides only under `fixture=True`.
8. One `evalkit.topk_ids_scores` run feeds nDCG@10 and Recall@10; per-component and equal-weight
   macros for both.
9. End-to-end synthetic success test through the rehearsal bundle, `loader_np` and the M7 scorer.
   `pytest m17src/test_evaluate.py -q` → 26 passed. Full `pytest -q m17src` → 4 failed / 314
   passed, the 4 being pre-existing registry-status assertions in test_lock and test_train.

Executor's open concerns: (a) the pinned-pool sha256 adds a 12.6 GiB read to every preflight;
(b) text-component vector verification is only as strong as the cache's recorded digests;
(c) text-component query binding is recorded, not verified.

## Files you may open (no others; no `grep -r`, `rg`, or `find` over the repository)

- `m17src/evaluate.py`, `m17src/test_evaluate.py`, `m17src/common.py`, `m17src/lock.py`,
  `m17src/export.py`, `m17src/loader_np.py`
- `m17/registry.json` (`screen_routing_surface`, `untrained_vocab_export_v0`, `serving`,
  the lock block), `m17/REVIEW.md` (latest entry), `m17/CODEMAP.md`
- `results/m7_dev_manifest.json`
- `m7src/devsuite.py`, `m7src/heldout.py`, `m7src/evalkit.py`, `m7src/teacher.py`,
  `m7src/hashing.py`, `m7src/pool.py`
- `git diff 2c5321c 4824f84`
- You may run `.venv/bin/python -m pytest m17src/test_evaluate.py -q` (writes only pytest tmp).

## Read-exclusions (absolute)

- Never open `results/frozen_eval/untouched-*`, any reserved qrels cache, `work/m9reserve`, or
  anything concerning FEVER, DBpedia-entity, cqadup-android, cqadup-english.
- Never read anything under `work/m17/prepared/full` or `work/m17/bundles/V0` as a quality
  surface; never execute `dev_suite_read`, `--surface dev-suite`, or anything that scores an M17
  artifact on the development suite. V0 has exactly one declared read and it is not yours.

## Questions to break

1. For each of the nine: is it closed, partially closed, or closed in name only? Where a check
   compares X to Y, is Y actually an independent pinned identity, or does the code compare a
   value to itself (TOFU, recomputed both sides, defaulted field)?
2. The `_pool_identity` re-implementation: does it match `heldout._verify_pool` exactly (same
   fields, same bytes, same sha input)? Could it pass on a pool that M7's own check would refuse?
3. Does `_teacher_doc_vecs` bind the vectors that `evalkit.topk_ids_scores` actually consumes
   (same file, same slice, same dtype, same row order as the component's doc id list)?
4. Preflight: is there any path where scoring starts, or the receipt says `started`, before a
   check that can fail? Is there any path where a refusal leaves a `started` receipt that then
   blocks the legitimate read forever with no recovery instruction? Is there any path where a
   partial result JSON is written?
5. Metrics: is Recall@10 computed on the same top-k as nDCG@10, with the same qrels relevance
   threshold M7 used for this suite, and is the macro the equal-weight domain macro over the six
   components as registered (not query-weighted)?
6. `fixture=True`: can a production caller reach it? Does the CLI expose it?
7. Does the receipt's git sha reflect a clean tree, and is a dirty tree disclosed?
8. Is the 12.6 GiB pool hash acceptable for a one-time read, or does it create a realistic
   crash/timeout that spends the attempt? State it as a judgement, not a blocker, unless it is one.

## Output

Findings only, ranked P1/P2/P3 with file:line, a one-sentence failure scenario each and the
smallest fix. State explicitly whether anything still blocks the V0 read. Under 60 lines.
