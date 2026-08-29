# Codex adversarial review #4 — the pre-freeze one-shot path, after the MAJOR 1/2/4 fixes

`codex exec -s read-only -m gpt-5.6-sol -c model_reasoning_effort="high"`, 2026-08-28.
Brief: the four remaining MAJORs from `m7-codex-onepath-2026-08-28.md` had been implemented and I
asked it to break the fixes and say what they broke. Verbatim below.

**6 BLOCKER / 11 MAJOR, all actioned.** Dispositions in `m7/LEDGER.md` (Provenance, "The one-shot
path, hardened") and in the commit that follows this file. The one that mattered most: BLOCKER 6,
the freeze tag was never peeled, so the documented `git tag -a` procedure could not have passed the
guard and the final run could not have started. Verified empirically before fixing.

Two claims of mine it correctly called FALSE: that the freeze re-derives "every hash" (the BM25
cache keys were recorded and never read), and that the released system "can no longer be decided
anywhere except the dev grid" (FREEZE.json could be edited after `freeze.write`).

---

## BLOCKER

1. The new persistence ordering can wedge or destroy the confirmatory result.

Two concrete failure windows exist:

- [final_run.py:446](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:446) writes `m7_final_run.json`, then [final_run.py:451](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:451) appends `FINAL-RUN complete`. If the process dies between them, the result file is dirty but there is no completion marker. `--infra-retry` refuses because `results/m7_final_run.json` is not in its `ALLOWED_DRIFT`; `--untouched-only` refuses because there is no completion marker. The run is permanently wedged after all six were scored.
- After completion, every tail checkpoint uses in-place `write_text` at [final_run.py:484](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:484). `write_text` truncates before writing. A kill, disk-full error, or power loss can corrupt the sole confirmatory JSON. The ledger already says COMPLETE, so `--infra-retry` is forbidden; `--untouched-only` then crashes while parsing the truncated JSON at [final_run.py:136](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:136).

Observable outcome: the access is spent and the result is either unusable or trapped in a state no legitimate mode accepts.

The result needs atomic temp-write + `fsync` + `os.replace`, and the ledger needs the result SHA-256 so a crash after the replace can be finalized without rescoring.

2. `--infra-retry` is allowed twice, not once.

At [final_run.py:144](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:144):

```python
if infra_retry and n_begin > MAX_INFRA_RETRIES:
```

With `MAX_INFRA_RETRIES = 2`:

- Initial run leaves one `BEGIN`.
- First retry sees `n_begin == 1` and appends the second.
- Second retry sees `n_begin == 2`; `2 > 2` is false, so it is also allowed.
- Only a third retry is refused.

This permits three total readings. The comparison must be `>= MAX_INFRA_RETRIES`.

3. `ALLOWED_DRIFT` is content-unbounded and defeats both one-shot integrity and result integrity.

[final_run.py:84](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:84) permits any modification to the ledger/access log, not merely the scorer’s expected append. After a completed run:

- Delete the COMPLETE line while leaving one valid BEGIN.
- Invoke `--infra-retry`.
- The dirty paths are allowed, `n_complete == 0`, and the six are scored again.

Likewise, [final_run.py:86](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:86) allows arbitrary changes to `m7_final_run.json` under `--untouched-only`. The resume guard checks only that `six` is truthy and that one table hash matches at [final_run.py:137-141](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:137). A caller can alter six scores or Holm decisions, run `--untouched-only`, and the altered confirmatory result is preserved and rewritten as an apparently resumed result.

Observable outcome: a second six-set run or a plausible forged final JSON passes the intended guard.

4. The table can change after verification and still be scored as the frozen artifact.

`freeze.load_and_verify()` hashes the gitignored table once at [freeze.py:312](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:312). The final run later reopens it separately for every variant and dataset at [final_run.py:217](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:217).

Concrete ordering:

1. `load_and_verify` verifies table A.
2. Another process regenerates/replaces `<run>.release.npz` with B through `ensure_release`.
3. `score_set` loads B.
4. The result records A’s frozen hash but contains B’s scores.

Because `work/` is ignored, the clean-tree guard sees nothing. Replacement between datasets can even create a mixed A/B six-set result. The final process needs an immutable snapshot/file descriptor, not a hash followed by later pathname reopens.

5. `--untouched-only` does reread the six’s labels before its guard.

`main` calls `preflight()` before `guard()` at [final_run.py:321-322](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:321). `preflight` opens and parses all six frozen query/qrels payloads at [final_run.py:258-275](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:258).

Thus every refused second invocation and every `--untouched-only` invocation reads the six’s local qrels before the completed-run check. It does not call `score_set` on them, so it is not a second scoring pass; but if “access” means reading confirmatory labels, the answer is unequivocally yes.

Move the one-shot guard ahead of every six-payload read, while retaining static preflight before `FINAL-RUN-BEGIN`.

6. The instructed annotated freeze tag cannot pass the guard.

The error message recommends:

```bash
git tag -a m7-freeze ...
```

at [final_run.py:111](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:111), but `git ls-remote origin refs/tags/m7-freeze` at [final_run.py:107](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:107) returns the annotated tag object hash, not the peeled commit hash. Comparing it directly with `freeze_hash` always fails.

Resolve `refs/tags/m7-freeze^{}` or explicitly support both lightweight and annotated tags.

## MAJOR

1. The query/qrels fix is incomplete: query text remains completely unbound.

[final_run.py:275](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:275) hashes only sorted query IDs; [final_run.py:276](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:276) hashes qrels. There is no query-text hash in `eval_manifest.json`, and `FREEZE.json` does not hash individual frozen payloads.

Concrete trigger: change a value in `results/frozen_eval/scifact.json["queries"]` without changing its key or qrels, then commit it as part of the freeze commit. Preflight passes, strict comparator alignment passes, and the altered query is encoded and scored.

Observable outcome: a plausible confirmatory number from different query text. The comment at [final_run.py:271-273](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:271) claiming changed query text is checked is false.

2. The claim that freeze re-derives the BM25 cache keys is false.

`select_fusion` records `selected_on.bm25_run_keys` at [select_fusion.py:97](/home/dylan/asymetric-dual-encoders/m7src/select_fusion.py:97), but `load_selected_fusion` never reads or recomputes that field. Its `want` dictionary at [freeze.py:133-137](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:133) stops at the dev-manifest hash.

Concrete trigger: select fusion under bm25s/PyStemmer versions X, then upgrade to Y before freeze. The recorded keys still name X, but freeze succeeds. The final run builds fresh BM25 under Y and applies a parameter selected under X.

The new test fixture actually normalizes this hole: `bm25_run_keys` is `{}` at [test_freeze_binding.py:80](/home/dylan/asymetric-dual-encoders/m7src/test_freeze_binding.py:80), and the fixture is accepted.

3. `FREEZE.json` is not rebound to the selection at final-run time.

`load_and_verify` checks family, depth, and released-system derivation at [freeze.py:355-372](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:355), but it does not validate:

- `fusion.selected_on`
- the public selection file
- table/meta hashes inside `selected_on`
- BM25 keys
- whether `family,param` is the winning grid row
- whether the parameter even occurs in the recorded grid

Concrete trigger: after `freeze.write`, edit only `m7/FREEZE.json` before committing, changing `fusion.param` and the top-level `released_system` consistently. `load_and_verify` accepts it and the final run uses the edited parameter.

Therefore “released-system can no longer be decided anywhere except the dev grid” is false.

4. `freeze.write` has an A/B TOCTOU race between gate/selection validation and manifest construction.

At [freeze.py:220-221](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:220), gate and selection are verified against A. The table is rehashed later while constructing the blob at [freeze.py:242-245](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:242).

If another process replaces the release with B after line 221, the resulting freeze can contain:

- gate evidence for A
- fusion selected on A
- table/meta hashes for B

If A and B share preprocessing and encoder—as two checkpoints normally do—`load_and_verify` accepts B. The returned gate evidence also discards the gate’s artifact hashes at [freeze.py:192-193](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:192), hiding the inconsistency.

5. Gate validation accepts malformed or internally failing gates.

At [freeze.py:180](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:180), any truthy `PASS` is accepted. Therefore `"PASS": "false"` passes. More importantly, `"PASS": true` with a condition whose `"pass"` is false is accepted because individual conditions are never enforced.

A missing artifact block is refused through hash mismatches, but a truthy non-dict artifact crashes with `AttributeError` at line 183 rather than producing the promised refusal.

6. The early persistence still occurs after exploratory work.

The clean-4 robustness computations at [final_run.py:415-426](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:415) run before the result write at line 446. An OOM, interruption, or statistical-input error there occurs after all six and all confirmatory decisions, but before any result exists.

Also, preflight only checks comparator row presence at [final_run.py:284-293](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:284), not vector lengths, duplicate qids, or exact qid alignment. A committed `perquery.json` row with one missing qid passes preflight and fails `strict=True` only after the six have been scored.

Persist the confirmatory result before clean-4, and fully validate comparator alignment in preflight.

7. Retry does not require the same commit.

For an infrastructure retry, local `HEAD == freeze_hash` is waived at [final_run.py:99](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:99). The prior BEGIN hash is extracted at line 155 but never compared directly with `HEAD`/the current tag. Only the tree diff is inspected.

A different empty commit, or a commit changing only allowed paths, passes. That contradicts the promised identical-commit retry.

8. Teacher provenance is destroyed by a resumed tail.

The initial result captures all six encodes at [final_run.py:435](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:435). In a new `--untouched-only` process, `teacher.PROVENANCE` starts empty and accumulates only tail encodes. [final_run.py:482](/home/dylan/asymetric-dual-encoders/m7src/final_run.py:482) replaces—rather than merges—the saved provenance.

Observable outcome: after a resumed tail, the final JSON no longer records any six-set document or query cache hashes. The “auditable afterwards” claim is false.

9. Teacher remote code is verified only at freeze-write time, not when it runs.

`teacher_code.verify()` is called at [freeze.py:215-219](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:215). `load_and_verify` neither calls it nor even validates `teacher_code_pin_sha256`. If the local HF snapshot is altered after freeze, a cache miss makes the final process execute altered code/weights while all final preflight checks pass.

This should be reverified before any protected data access.

10. Encode-cache resumability and shard layout are not actually bound.

`shards.json` is written only after all shards and the combined stitch complete at [teacher.py:296-297](/home/dylan/asymetric-dual-encoders/m7src/teacher.py:296). A crash after writing hundreds of shards leaves them without persisted records; the next `verify=True` call adopts them as TOFU and refuses, forcing a complete re-encode. The advertised shard resumability is lost for new long jobs.

Also, `SHARD` is absent from the cache key and the manifest records no expected row range or shape per shard. If `SHARD` changes while reusing a hashed cache, `_combined` can load only a prefix of the old shards and never checks `off == n_rows` at [teacher.py:357-364](/home/dylan/asymetric-dual-encoders/m7src/teacher.py:357). It can write a combined file with an unwritten/zero tail, hash it, and pass later verification.

Changing `len(texts)` itself is safe because the corpus hash changes the cache directory. Changing shard layout is not.

11. Releasability is bound to a mutable side record, not the artifact.

`assert_releasable` reads only gitignored `work/runs/<id>.json` at [freeze.py:58-68](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:58). It does not bind or compare that record with the checkpoint metadata or the committed `results/m7_run_<id>.json`.

Concrete trigger: table trained with MS MARCO, but a stale/replaced `<id>.json` says clean sources. The lineage walk passes. `FREEZE.json` records only lineage IDs, not hashes/configs proving what was inspected.

## Fusion-cache edge cases

The normal production list-based key handles ordered content, `DEPTH`, and effective `k` correctly: `n_docs + DEPTH` determines `min(DEPTH, n_docs)`. JSON float formatting is not an issue because decoded dictionaries are compared structurally.

Remaining holes:

- [fusion.py:124-138](/home/dylan/asymetric-dual-encoders/m7src/fusion.py:124) trusts a caller-supplied `key` without recomputing it. A stale supplied key can consume unrelated arrays.
- `cache_key(..., depth=x)` records `x`, but retrieval always uses global `DEPTH` at [fusion.py:147](/home/dylan/asymetric-dual-encoders/m7src/fusion.py:147). A non-default supplied depth therefore mislabels the cache.
- A `doc_texts` generator is consumed while hashing at [fusion.py:104](/home/dylan/asymetric-dual-encoders/m7src/fusion.py:104), then reused exhausted for tokenization at line 144.
- When package metadata is unavailable, versions become `None` at [fusion.py:85-86](/home/dylan/asymetric-dual-encoders/m7src/fusion.py:85). Different source/editable installs lacking distribution metadata collide.
- `_read_cache` checks only query row count, not expected width, matching `ids`/`scores` shapes, index bounds, or a payload checksum. A rewritten array with an intact key is consumed.
- Cache writes are not atomic; an interrupted `.npz` raises during the next load instead of being classified as unvalidatable and rebuilt.

These are mostly not triggered by today’s `select_fusion` call with materialized lists, but the unvalidated BM25-version provenance is production-reachable.

## Grid ties

Ties systematically favor fusion. `select_on_dev` uses strict `>` at [fusion.py:183-197](/home/dylan/asymetric-dual-encoders/m7src/fusion.py:183), and iteration order is RRF first, then convex weights from 0.3 upward, then convex0. Dense-only `w=1.0` cannot replace an equal earlier point.

Concrete outcome: if `w=0.9` and `w=1.0` have identical nDCG rankings, `w=0.9` remains selected and the release is called fused despite no dev benefit. The dense endpoint is present, but the unstated tie policy privileges the more complex system.

## Test-suite assessment

The tests do not cover the dangerous production transitions.

`test_freeze_binding.py`:

- Never calls `freeze.write`; it tests helpers independently.
- Accepts empty BM25 provenance.
- Does not test TOCTOU, false gate conditions, truthy non-boolean PASS, grid/parameter consistency, lineage, or final-time selection rebinding.
- Its `load_and_verify` enum case uses a manifest with a missing table and bogus hashes at [test_freeze_binding.py:187-195](/home/dylan/asymetric-dual-encoders/m7src/test_freeze_binding.py:187). It proves that one error message contains the enum complaint, not that a valid freeze passes and only the enum mutation fails.
- The write-time enum guard at [freeze.py:230](/home/dylan/asymetric-dual-encoders/m7src/freeze.py:230) is unreachable: `derived` is always one of two literals.

`test_encode_cache.py`:

- Does not test mutated `combined.f16`.
- Its final TOFU scenario has TOFU shards, so `encode_cached` exits at [teacher.py:290](/home/dylan/asymetric-dual-encoders/m7src/teacher.py:290); it never exercises the separate combined-file refusal at lines 345–349.
- Does not test crash recovery, manifest persistence after each shard, orphan temp files, concurrency, shard-layout changes, single-shard behavior, or provenance preservation across final-run resume.
- The non-verify shard-mutation check does not consume the mutated shard: the unchanged combined file is returned. It documents nondetection but does not prove how corruption propagates when a restitch is required.

There are no tests for `final_run.guard`, its retry counter, tag peeling, result/ledger transaction, `ALLOWED_DRIFT`, resume integrity, or atomic tail persistence.

I attempted the three direct suites, but the managed read-only environment has no writable temporary directory, so they failed before import/test execution. No repository files were changed. During the audit HEAD advanced beyond `948b467`, but the reviewed target files have no diff from `948b467`; the later commits touched other material.
