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
