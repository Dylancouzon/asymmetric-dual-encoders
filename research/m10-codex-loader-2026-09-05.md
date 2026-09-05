# Codex adversarial review — M10 corpus→trainer path, 2026-09-05 (verbatim; read-exclusion audit clean)

## Findings

1. **BLOCKER — cut arms silently train uncut.** [corpus_loader.py:470](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:470), [corpus_loader.py:504](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:504): current registry has no `unique_text_count`; `build_query_stream("ANCHOR", ...)` defaults to `cut=None`, and even `cut="registered"` becomes a no-op. A2/A3/A4 therefore train at different volumes. Fix: require the registered cut for A2/A3/A4 and raise when absent.

2. **BLOCKER — required M10 decontamination is absent.** [corpus_loader.py:172](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:172), [corpus_loader.py:536](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:536), [m9src/data.py:71](/home/dylan/asymetric-dual-encoders/m9src/data.py:71), [m9src/data.py:102](/home/dylan/asymetric-dual-encoders/m9src/data.py:102): an M9 query/document newly matching admitted COV remains eligible, leaking evaluation text into gradients. Fix: require and bind both pools to the M10 protected-index rescreen and its identity.

3. **BLOCKER — the hold-out guard protects a pathname, not the data.** [corpus_loader.py:53](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:53), [corpus_loader.py:123](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:123): copy or hard-link `harvest_forms12.jsonl` to `generated_queries.jsonl`; it is accepted as A4 training data because generated has no expected count and origin fields are ignored. Fix: verify a screened immutable manifest/protected fingerprint set, including seed/document provenance.

4. **BLOCKER — resume loses kill/plateau state.** [trainer10.py:35](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:35), [trainer10.py:68](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:68), [trainer10.py:76](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:76): resume after two cycle-end evaluations resets `evals` and `cycle_end_evals`; the third reading cannot fire the registered plateau/top-up decision. Fix: checkpoint and restore evaluation histories, kinds, counters, and extension state.

5. **HIGH — concurrent target writers can silently swap keys and vectors.** [targets10.py:131](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:131): writers A/B can append vectors as A,B but keys as B,A; both publish the same stale `n`, and reopen retains `keyB → vectorA`. Fix: enforce a single-writer lock spanning refresh, duplicate lookup, append, and metadata commit.

6. **HIGH — “unique-text” cutting operates on raw rows.** [corpus_loader.py:123](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:123), [corpus_loader.py:477](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:477): `["x","x","y"]` counts as three and may retain duplicate `x`, changing both the cut and presentation weights. Fix: exact-deduplicate globally before computing or applying the three-corpus cut.

7. **HIGH — balanced sampling is not with-replacement sampling.** [corpus_loader.py:392](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:392), [corpus_loader.py:443](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:443): a three-row form at batch eight always produces the same `0,1,2,0,1,2,0,1` batch; larger forms cycle fixed batches without replacement. Fix: derive a per-form RNG from `(seed, form-occurrence)` and call `choice(..., replace=True)`.

8. **HIGH — short cache files are padded, not refused.** [targets10.py:110](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:110): with authoritative `n=2` and half of vector two missing, `truncate(want)` zero-extends it; its nonzero surviving half is normalized and trained as a corrupted target. Fix: reject sizes below `n * row_bytes`; truncate only excess bytes.

9. **HIGH — query/document student inputs can collide.** [corpus_loader.py:308](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:308), [corpus_loader.py:530](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:530): query `"passage: X"` and document `"X"` tokenize identically, but receive query-prompt and raw-document teacher targets. Fix: detect and reject cross-role student-input collisions before training.

10. **MEDIUM — checkpoints overwrite the sole recovery point non-atomically.** [trainer10.py:35](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:35): a crash during `torch.save` corrupts the only checkpoint. Fix: write/fsync a temporary checkpoint and atomically replace the prior file.

11. **MEDIUM — token-cache identity does not identify the tokenizer.** [corpus_loader.py:331](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:331): change tokenizer revision/config while retaining `student="bge-small"`; stale token IDs are reused. Fix: include tokenizer repository, revision, vocabulary/config hashes, and selected-row hash.

12. **LOW — hash collisions are treated as text equality.** [targets10.py:48](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:48), [targets10.py:152](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:152): two distinct texts sharing the 128-bit digest collapse to one target without comparison. Fix: store enough text/secondary digest to verify equality on hits.

Tests miss or bless these failures: [test_corpus_loader.py:191](/home/dylan/asymetric-dual-encoders/m10src/test_corpus_loader.py:191) explicitly expects an absent cut to proceed; line 136 only proves a small form is not dropped; [test_targets10.py:94](/home/dylan/asymetric-dual-encoders/m10src/test_targets10.py:94) tests excess bytes, not short files or concurrency; [test_trainer10.py:41](/home/dylan/asymetric-dual-encoders/m10src/test_trainer10.py:41) resumes without evaluations. The self-comparison at `test_targets10.py:34` and pathname assertion at line 123 are partly vacuous.

The narrow step-to-stream arithmetic and wall-clock exclusion look correct. Requested `m9src/teacher.py` does not exist at either reviewed endpoint.

**NO-GO** for training a screen arm on this loader.

---

# RE-REVIEW of the fixes, same day (verbatim; read-exclusion audit clean)

Brief: the eleven fixes claimed done, with the attack surface named. Verdict below; the
gaps it found in the fixes were closed in `d6bc3d6`, and the two residuals it names are
recorded there.

NO-GO. Several fixes are real, but the “1–11 fixed” claim is too strong. The principal failures are unscreened training routes, a hold-out guard that silently disables itself, and an unenforced writer lock.

1. LOW — cut-arm refusal is real

[corpus_loader.py:710](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:710) identifies registered named arms, and [corpus_loader.py:713](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:713) refuses a missing cut unless `allow_uncut=True`; the latter records `uncut` at line 721.

Caveat: passing the equivalent source list instead of an arm name bypasses cut-arm classification. That is how the smoke calls it at [arm_smoke.py:177](/home/dylan/asymetric-dual-encoders/m10src/arm_smoke.py:177). This is acceptable only if every real launcher must pass the arm name; that invariant is not enforced here.

2. BLOCKER — unscreened M9 training routes remain

The default corpus and document builders do screen correctly:

- Query masks are required at [corpus_loader.py:300](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:300).
- `load_segments` uses screened `_m9_segments()` at [corpus_loader.py:344](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:344).
- Document masks are required by default at [corpus_loader.py:787](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:787).

But unscreened optimization paths still exist:

- `build_doc_stream(..., allow_unscreened=True)` explicitly bypasses screening at [corpus_loader.py:771](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:771).
- `_screened_doc_pool` treats a falsey ban set as permission to load the original pool at [corpus_loader.py:752](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:752).
- The arm smoke directly loads unscreened M9 queries and documents at [arm_smoke.py:106](/home/dylan/asymetric-dual-encoders/m10src/arm_smoke.py:106) and [arm_smoke.py:109](/home/dylan/asymetric-dual-encoders/m10src/arm_smoke.py:109), then performs optimizer steps at [arm_smoke.py:201](/home/dylan/asymetric-dual-encoders/m10src/arm_smoke.py:201). Even `--corpus m10` continues using those directly loaded documents.

The tests do not close this gap. The query test exercises only `query_keep_mask`, not `_m9_texts`, `_m9_segments`, or `load_segments`, at [test_corpus_loader.py:467](/home/dylan/asymetric-dual-encoders/m10src/test_corpus_loader.py:467). The document test calls `_screened_doc_pool` directly with a hand-supplied nonempty set at [test_corpus_loader.py:513](/home/dylan/asymetric-dual-encoders/m10src/test_corpus_loader.py:513), so it would pass even if `build_doc_stream` never requested a mask.

The claimed “709 of 463,314” is not validated in the permitted code. The `709` at [test_corpus_loader.py:525](/home/dylan/asymetric-dual-encoders/m10src/test_corpus_loader.py:525) is merely dummy cache-key input.

3. HIGH — hold-out protection silently turns off

Content checking is genuinely integrated into `load_segments` at [corpus_loader.py:374](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:374), and pathname refusal is real at [corpus_loader.py:168](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:168).

However, if the hold-out file is missing, `holdout_hashes` returns an empty set at [corpus_loader.py:71](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:71), and training proceeds at [corpus_loader.py:85](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:85). There is no registered-run refusal or recorded “hold-out unavailable” escape.

The cache is also keyed only by pathname at [corpus_loader.py:68](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:68). If the file is created or changed after its first read, the process retains stale hashes.

The test calls `refuse_holdout_texts` directly at [test_corpus_loader.py:376](/home/dylan/asymetric-dual-encoders/m10src/test_corpus_loader.py:376); it does not prove that a copied source is rejected through `load_segments`.

4. HIGH — most resume state is fixed, but `stopped` is never checkpointed when true

Loss, evaluation, kind, cycle-end, and example histories are saved at [trainer10.py:141](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:141) and restored at [trainer10.py:87](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:87). On ordinary resume, examples are not double-counted, and `steps_run` intentionally counts only the current process.

But every path that makes `stopped` non-null breaks before the save block:

- Non-finite loss: [trainer10.py:110](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:110)
- Non-finite gradient: [trainer10.py:117](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:117)
- Kill: [trainer10.py:132](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:132)
- Plateau: [trainer10.py:136](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:136)
- Save block only afterward: [trainer10.py:140](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:140)

Thus `stopped` is serialized syntactically but, through `train_arm`, only while it is `None`. A stopping evaluation is also absent from the latest checkpoint.

For successful resumes, `examples`, steps, and mix are not double-counted. `mix` is reconstructed from the whole restored loss count at [trainer10.py:159](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:159). One minor accounting defect remains: a non-finite loss is appended even though no step completed, so that failed attempt is included in `mix`.

5. BLOCKER — the writer lock is optional and bypassed throughout the API

`encode_missing` correctly holds the lock across refresh, lookup, encoding, append, and commit at [targets10.py:235](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:235).

But `TargetCache.append` itself neither acquires nor verifies the lock at [targets10.py:188](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:188). A second process can call it while another owns the lock. A stale `TargetCache` can also append with an obsolete `self.n`, publish the wrong count, and leave committed rows as “excess” for the next refresh to truncate.

The tests normalize this bypass: for example, they append outside any lock at [test_targets10.py:43](/home/dylan/asymetric-dual-encoders/m10src/test_targets10.py:43). The concurrency test at [test_targets10.py:178](/home/dylan/asymetric-dual-encoders/m10src/test_targets10.py:178) proves only that two callers cannot both create the lock file; it never attempts `other.append()` while the first lock is held.

Read paths are not literally non-mutating either: constructing an absent cache writes `meta.json` at [targets10.py:107](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:107) and creates zero-length store files at [targets10.py:134](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:134).

6. LOW — dedup and teacher-row preservation are implemented

Dedup occurs before the cut at [corpus_loader.py:375](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:375). Segment order and row order determine the first occurrence, and `forms` plus `rowmap` are sliced by the same selection at [corpus_loader.py:326](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:326). The surviving text therefore retains the correct teacher row.

The test at [test_corpus_loader.py:395](/home/dylan/asymetric-dual-encoders/m10src/test_corpus_loader.py:395) meaningfully checks this.

Strictly, it is hash-equivalence dedup, not exact-text dedup: only the 128-bit hash is stored in `seen` at [corpus_loader.py:320](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:320). That falls under the declined item 12 decision.

7. LOW — sampler behavior is real; one named test is weak

`_pick(k)` is a pure function of seed and step/cycle at [corpus_loader.py:586](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:586). Resume at the same global step produces the same form and row draw.

Each complete `F`-step sampler cycle contains each present form exactly once, with identical batch sizes, so example shares are exactly equal. The within-form call uses replacement at [corpus_loader.py:543](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:543).

Sorting occurs only after membership has been sampled at [corpus_loader.py:544](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:544). It changes order, not selection probabilities or form shares, so it does not reintroduce sampling bias.

The test named “sampled with replacement” at [test_corpus_loader.py:136](/home/dylan/asymetric-dual-encoders/m10src/test_corpus_loader.py:136) is effectively non-probative: it merely checks that the small form appears. A deterministic repeated batch would pass it. The later varying-batch test is better, though it still does not directly assert within-batch duplicates.

8. HIGH — short-store refusal is real, but “only a writer truncates” is unenforced

Short files raise correctly at [targets10.py:140](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:140), and ordinary construction uses `fix=False`.

However, `refresh()` is public and does not verify lock ownership at [targets10.py:150](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:150); `_check_sizes(fix=True)` likewise has no ownership check. Any caller can truncate excess data without holding the lock. Combined with the unlocked public `append`, one caller can truncate bytes another caller is appending.

On the blessed `encode_missing` path, an old reader and the writer are safe: the reader maps only the `n` published in metadata, while metadata advances after vectors and keys. The safety is convention, not enforced storage semantics.

9. HIGH — collision detection works, but it is not a mandatory corpus→trainer invariant

The hashing and refusal implementation is real at [corpus_loader.py:659](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:659) and [corpus_loader.py:680](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:680).

But neither `build_query_stream`, `build_doc_stream`, nor `train_arm` requires the guard. The only reviewed training caller that invokes it does so manually in the smoke at [arm_smoke.py:196](/home/dylan/asymetric-dual-encoders/m10src/arm_smoke.py:196). Another caller can combine the two streams and train without checking, or pass `skip=True`.

The test at [test_corpus_loader.py:434](/home/dylan/asymetric-dual-encoders/m10src/test_corpus_loader.py:434) tests the helper explicitly; it does not test enforcement on a launch path.

10. LOW — atomic replacement is implemented, but the test does not prove it

The implementation writes a same-directory temporary file, flushes, fsyncs, and calls `os.replace` at [trainer10.py:45](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:45). That protects the prior checkpoint from a process crash during serialization.

The test at [test_trainer10.py:104](/home/dylan/asymetric-dual-encoders/m10src/test_trainer10.py:104) would also pass if `save` overwrote `c.pt` directly; it observes neither interruption nor the temporary/replace operations. Also, the parent directory is not fsynced, so durability across abrupt power loss is not guaranteed, although replacement atomicity is.

11. LOW — tokenizer identity is implemented; the vocabulary test passes for the wrong reason

The key includes class, path, revision, size, and either backend JSON or vocabulary hash at [corpus_loader.py:452](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:452), and that identity enters the cache key at [corpus_loader.py:481](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:481).

But the test compares `TokA` and `TokB`, which are different class names at [test_corpus_loader.py:446](/home/dylan/asymetric-dual-encoders/m10src/test_corpus_loader.py:446) and [test_corpus_loader.py:452](/home/dylan/asymetric-dual-encoders/m10src/test_corpus_loader.py:452). Since class name is already part of the key, the two cache directories at lines 459–461 would differ even if vocabulary hashing were removed entirely. The property named by the test is therefore untested.

12. LOW — declined and documented, with broader scope than the target-cache prose suggests

The decision is documented at [targets10.py:20](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:20). No secondary equality check exists.

The same 128-bit-equals-content policy also governs hold-out detection and supposedly “exact” corpus dedup through [corpus_loader.py:60](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:60) and [corpus_loader.py:320](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:320). Thus the accepted decision is not confined to the teacher cache.

I did not execute the tests because importing this suite would read additional repository modules and data outside the hard allowlist.

NO-GO.

# THIRD PASS on the final state (b619a34), same day (verbatim; read-exclusion audit clean)

The named files at checked-out `129f681` are identical to `b619a34`; this review is therefore valid for the requested commit. Static review only—the exclusions prevent safely running tests that import unnamed data paths.

### Earlier findings

| # | Status | Evidence / regression test |
|---|---|---|
| 1 | **OPEN** | Named arms refuse a missing cut, but `allow_uncut=True` bypasses it, and passing `ARM_SOURCES["A3"]` as a list avoids cut-arm classification entirely ([corpus_loader.py:711](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:711)). `test_a_smoke_may_train_uncut_only_by_saying_so_in_the_artifact` explicitly blesses the first bypass; no test covers the source-list bypass. Either stream can be passed to `train_arm`. |
| 2 | **OPEN** | Default query/document builders require masks, but `build_doc_stream(..., allow_unscreened=True)` still returns training data, while an empty ban set silently calls the original pool ([corpus_loader.py:772](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:772), [corpus_loader.py:793](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:793)). Tests cover only the default refusal. |
| 3 | **OPEN** | Missing/content-identical hold-out checks work, but provenance is discarded: `_rows_from_jsonl` returns only text/form, ignoring `seed_id`/`doc` ([corpus_loader.py:190](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:190)). A novel query with a held-out `seed_id` passes because only its query text is hashed. The copy test cannot catch this. |
| 4 | **OPEN** | Histories and ordinary stopping checkpoints are fixed, but saving remains conditional on `run_steps` ([trainer10.py:150](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:150)). If the first attempted step after resume has a non-finite loss/gradient, the old checkpoint still says `stopped=None`; relaunch repeats it. The plateau test completes many steps and misses this. |
| 5 | **CLOSED** | `append`, `refresh`, and truncation require the owning cache handle’s writer lock ([targets10.py:152](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:152)). `test_append_outside_the_writer_lock_is_refused` and `test_only_one_writer_may_append_at_a_time` regress it. |
| 6 | **CLOSED** | Global dedup precedes cutting and preserves `rowmap` ([corpus_loader.py:324](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:324)). `test_the_cut_counts_UNIQUE_texts_across_sources` covers duplicate removal and teacher-row preservation. |
| 7 | **CLOSED** | `_form_draw` uses deterministic `choice(..., replace=True)` ([corpus_loader.py:550](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:550)). The small-form duplicate and varying-batch tests now catch the original fixed-cycle behavior. |
| 8 | **CLOSED** | Short stores raise; excess truncation requires the writer lock ([targets10.py:121](/home/dylan/asymetric-dual-encoders/m10src/targets10.py:121)). The short-store and unlocked-refresh tests cover both halves. |
| 9 | **OPEN — blocks** | Detection works, but neither `data10.batch_fn` nor `trainer10.train_arm` invokes it; only `arm_smoke` does ([data10.py:150](/home/dylan/asymetric-dual-encoders/m10src/data10.py:150), [arm_smoke.py:207](/home/dylan/asymetric-dual-encoders/m10src/arm_smoke.py:207)). The absence of a stream owner is precisely why the invariant is unenforced. The test exercises only the helper—and also blesses `skip=True`. |
| 10 | **CLOSED** | Same-directory temporary write, fsync, and `os.replace` preserve the prior checkpoint ([trainer10.py:36](/home/dylan/asymetric-dual-encoders/m10src/trainer10.py:36)). `test_a_crash_during_save_leaves_the_previous_checkpoint_intact` would fail on direct overwrite. |
| 11 | **OPEN** | Fast-tokenizer backend state is hashed, but the fallback hashes only vocabulary plus a few attributes ([corpus_loader.py:460](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:460)). Same class/repo/revision/vocabulary with different slow-tokenizer normalization or truncation side produces different IDs under one cache identity. The test changes vocabulary, not configuration. |
| 12 | **OPEN, accepted/non-blocking** | Hash equality remains text equality, as documented. No secondary comparison exists. |

### New findings

| Rank | Severity | Finding |
|---|---|---|
| 1 | **BLOCKER** | The missing-form guard is impossible to fire. `present` is constructed from `np.unique(forms)`, then `empty` checks only members of `present` ([corpus_loader.py:584](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:584)). An anchor corpus missing 1 of 12 forms silently becomes an 11-form-balanced run. `test_balanced_shares_are_equal_across_the_forms_present` blesses exactly that weaker property. |
| 2 | **BLOCKER** | The claimed 709 query removals are absent from [m10_rescreen10.json](/home/dylan/asymetric-dual-encoders/results/m10_rescreen10.json:30); it contains only the 79,630 document removals. `rescreen10.main` starts with `out={}` and rewrites the artifact, so the later `--documents` run erased the query report ([rescreen10.py:193](/home/dylan/asymetric-dual-encoders/m10src/rescreen10.py:193)). The literal `709` in a token-cache test validates nothing about the artifact. |
| 3 | **HIGH** | The hold-out cache is not content-keyed: it uses size plus integer-second mtime ([corpus_loader.py:68](/home/dylan/asymetric-dual-encoders/m10src/corpus_loader.py:68)). A same-length rewrite within one second retains stale hashes and can admit the new hold-out text. The test forces mtime forward by ten seconds, so it misses this. |

### Verdict: NO-GO

Minimum fixes to flip it:

1. Add one mandatory screen-arm assembly path that owns both streams: named arms only, no `allow_uncut`/`allow_unscreened`/source-list bypasses; require the registered cut, complete query/document masks, all 12 forms for the balanced anchor, and run the cross-role guard before returning `batch_fn`.
2. Preserve and validate hold-out provenance (`seed_id`/document identity) against an immutable screened manifest; content hashes remain defense in depth. Key any cache by the file digest.
3. Make the re-screen report retain both query and document sections and validate mask/report identity, completeness, and measured counts before a screen arm starts.
4. Save stop state even when zero new steps completed, and include tokenizer behavior/configuration in token-cache identity.
5. Register `data_cut.unique_text_count` before launching a cut arm.
