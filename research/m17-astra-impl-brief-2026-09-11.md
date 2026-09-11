# Brief: adversarial implementation review of `m17src/` (M17 step 4, review 1 of 2)

You are reviewing code, read-only. Do not edit, create or run anything except the commands
listed under "You may run". Do not search recursively (`grep -r`, `find`, `rg` over the
tree); open only the files named here. List every file you read at the end of your report.

**Forbidden reads:** anything under `results/frozen_eval/untouched-*`, any reserved qrels
cache, anything under `work/m9reserve`, and any six-set or LoTTE evaluation payload. Reserved
sets are FEVER, DBpedia-entity, cqadup-android and cqadup-english. Do not open them, and do not
open anything under `work/` at all except the two file names given below.

## Files to read, in this order

1. `CLAUDE.md` (project rules), `m17/STATUS.md`, `m17/CODEMAP.md` (paths, contracts and the
   "Reuse hazards" list — those hazards are claims the code is supposed to honour).
2. `m17/registry.json` — every constant and rule the code must implement. Sections that matter
   most: `vocabulary_ranking`, `abbreviation_policy`, `data.bucket_populations_and_dose_rule`,
   `data.measured_dose_after_pre_lock_rule`, `training.*` (especially `loss_definition`,
   `candidate_construction`, `cache_identity`, `optimizer`, `overfit_divergence_check`,
   `decision_protocol`), `checkpoint_averaging`, `int8_resident_loading`.
3. `m17/LEDGER.md` sections "Step 3 — the one M17 driver" and "A3 — checkpoint 1 rulings".
4. The implementation, all of `m17src/`:
   `common.py`, `cache.py`, `vocab.py`, `train.py`, `export.py`, `loader_np.py`, `evaluate.py`,
   `support_manifest.py`, `alias_pairs.py`, `panel_build.py`, `alias_test_build.py`,
   `rehearse17.py`, `conftest.py`, and the nine `test_*.py` modules.
5. Reused legacy primitives, only these: `m7src/table.py` (class `QueryTable`, the sqrt
   occurrence weights, folding, `quantize_int8`, `save_table`), `m12src/qfusion.py`,
   `m7/FREEZE.json`.
6. Results the code wrote: `results/m17_rehearsal.json`, `results/m17_support_manifest.json`,
   `results/m17_panel_manifest.json`, `results/m17_alias_test_manifest.json`,
   `results/m8_b2_entropy.json` (the format the entropy block must match).
7. Two gitignored text files are permitted, nothing else under `work/`:
   `work/m17/manifest/alias_pairs_summary.json` and `work/m17/manifest/alias_test_families.json`.

## You may run

Only these, from the repository root, and nothing else:
`.venv/bin/python -m pytest -q -ra m17src` and `.venv/bin/python -m py_compile m17src/*.py`.
Do not run `rehearse17.py`, `support_manifest.py`, `panel_build.py`, `alias_pairs.py`,
`alias_test_build.py`, `train.py` or `export.py`.

## What this code is for

Zero is a 30,522 x 1,024 token lookup table serving as a query encoder against a frozen Stella
document index (1024d, document tower never retrained). M17 will, in a future registered 72-hour
window on one RTX 3080, add up to 3,072 new vocabulary rows (sum-initialised from constituent
pieces, jointly trained), compare a hard-candidate listwise KL loss with the old cosine-only
continuation, and test alias consistency, late-snapshot averaging and an int8 resident-row
loader. Five short screen arms (C, V, L, VL, VL-A), then two full runs of finalist and matched
control at two seeds. Selection is routed on the pinned development suite; the 553-query judged
panel and the 200-pair alias test are descriptive only.

Nothing has been trained. The registry is `DRAFT_NOT_EXECUTABLE`; `train.py` refuses a real
arm; the real data path (teacher encoding, admitted-query pool, bank mining, dev-suite and panel
readers) is deliberately unwritten and `train.py --data` expects a prepared directory that a
later step produces. This review is one of the two independent implementation reviews that the
registry's `execution_entry_missing` requires before any expensive execution.

## What I believe, and want you to break

- `losses()` in `train.py` implements `training.loss_definition` exactly: KL(teacher || student)
  with masked candidates, stable log_softmax, one shared temperature, no extra T² factor; cosine
  is `1 - dot` on normalised vectors; the anchor is a mean squared deviation of *effective folded*
  rows from the M17 initialisation; the alias term is `(2/B) * sum_pairs(1 - dot)` and both views
  keep their own teacher and candidate targets.
- The batch composition is 204 general views, 32 unpaired coverage views, 10 alias pairs at
  batch 256, drawn without replacement within a pass per bucket, reshuffled per pass with the
  run seed, resumable exactly.
- `cache.build()` walks teacher ranks skipping the positive and duplicates until 31 (or 32) new
  IDs, then v1 until 16, then uniform until 16, backfills only from ranked lists, breaks ties by
  ascending bank ID, seeds uniform draws per query from SHA-256(cache_seed || text hash), and the
  identity hash covers every item in `training.cache_identity`.
- `vocab.py` never sums over occurrences (it uses sqrt(count) per distinct piece), excludes
  [CLS]/[SEP] from new-row initialisation, applies the abbreviation policy and the per-domain and
  total caps as registered, pins `k8s`, and after ruling A3-3 counts a term's domains per distinct
  supporting document, with the document's domain coming from the source map or, for `general`
  sources, the panel builder's keyword classifier at the same threshold.
- `export.py` folds exactly once (the warm start is the *unfolded* checkpoint; the release's
  rows are already folded), averages only effective float32 folded rows over the registered
  snapshot window, quantises once, copies the frozen `encoder_spec` and asserts it, and its five
  gates fail closed. M11's gates are untouched.
- `loader_np.py`'s resident-int8 path reproduces the torch query path to 1e-6 including CLS/SEP,
  empty-query fallback, truncation at 512 and sqrt-count pooling, and reports weight bytes
  separately from RSS.
- No module can read a protected surface even by accident: the forbidden path substrings are
  checked wherever a path is opened.
- `support_manifest.py` refuses MS MARCO by name and cannot admit it through any alias.
- The tests would actually fail if each refusal were removed.

## Questions, ranked by how much they matter

1. **Correctness of the three loss terms and the batch/stream logic.** Read `losses()`,
   `forward()`, `Stream`, `build_streams()`, `RunCfg.batch_shape()` and `run()`. Find any sign,
   masking, normalisation, dtype, gradient-flow or resume bug. Does a masked-out candidate ever
   leak into the KL or the softmax denominator? Does `1 - dot` receive normalised vectors from
   the same pooling that serves? Is the anchor computed on the same "effective" rows the export
   uses, and on the M17 initialisation (post-extension) rather than the M7 checkpoint? Does the
   alias term get the right pair slots after a resume? Does `set_lr` implement warm-up 200 then
   linear decay separately for 4,000 and 6,000-step schedules? Is the divergence check read every
   500 steps on a fixed 2,000-query held-out slice and only *flagged*?
2. **Fold-once and lineage.** Trace what `load_warm_start`, `extend_model`, `_save_table`,
   `effective_rows` and `average_snapshots` do to rows and scalars. Can any path fold twice, average
   scalars separately, mix tokenizers or row scales, or accept a folded release as a warm start?
3. **Cache determinism and identity.** Can two caches with different bank bytes, v1 artifacts,
   mixes, seeds or manifests collide on `identity()`? Does `_ranked`/`_take` honour
   `candidate_construction` exactly (quota counting of *new* IDs, backfill order, tie-break)? Is
   the per-query RNG derivation what the registry says? Does the entropy block match
   `results/m8_b2_entropy.json` in structure?
4. **The A3-3 domain map.** Can per-document keyword classification of `general` sources bias
   term selection towards keyword-rich documents, or leak panel construction into training
   (the panel builder and the training map share `KEYWORDS`, `MIN_SCORE` and `classify`)? Is a
   query's `domain` guaranteed to be its document's domain and not depend on the query text? Does
   `domain_of` implement strict-majority-else-general and one cap per term?
5. **Gates that fail open.** For each of the five M17 gates and every `require_executable`,
   lineage, resume and forbidden-path refusal: construct the input that would pass it wrongly.
   Check that the tests would fail if the refusal were deleted (mutation reasoning, not running).
6. **Loader parity and measurement.** Anything in `M17QueryEncoder` that diverges from the
   torch path: unique-ID order, count weighting, normalisation epsilon, empty fallback, truncation,
   dtype of intermediate math, scale broadcasting for int8 codes.
7. **Evaluation.** `search`, `ndcg_at_k`, `recall_at_k`, the paired family bootstrap
   (percentile method, family resampling, seed), `dbsf_at` through `m12src/qfusion.py`, and
   `alias_test`. Are the dev-suite and panel surfaces really unwired, and is there any code path
   that could read them before the lock?
8. **`CLAUDE.md` violations.** Licensing (MS MARCO in any role, FineWeb), the 35M cap counted on
   the extended table plus scalars, frozen index, `results/perquery.json`, `torch.compile`,
   `m7/FREEZE.json` and M11 gates, protected paths, M13 non-interference.
9. **Rehearsal honesty.** Does `results/m17_rehearsal.json` record anything that the code did not
   in fact exercise, and are the `scaled_constants` faithful to the registry's field names?
10. What is missing that would make you refuse to authorise execution after step 5 (timings) and
    step 6 (lock)?

Be adversarial. A review that only confirms is useless. Give findings as P1 (would corrupt a
result, leak a surface, or fail open), P2 (wrong but recoverable before lock), P3 (clarity,
tests, docs), each with file and line references and one paragraph: what is wrong, the input
that triggers it, and the exact fix. Then a short list of registry or CODEMAP wording that no
longer matches the code. Finish with the list of every file you read.
