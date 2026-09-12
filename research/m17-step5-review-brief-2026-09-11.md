# Brief: adversarial review of the M17 step-5 prepared-data builder (one of two independent reviews)

You are reviewing code, read-only. Do not edit, create or run anything except the commands
listed under "You may run". Do not search recursively (`grep -r`, `find`, `rg` over the tree);
open only the files named here. List every file you read at the end of your report.

**Forbidden reads:** anything under `results/frozen_eval/untouched-*`, any reserved qrels cache,
anything under `work/m9reserve`, and any six-set or LoTTE evaluation payload. Reserved sets are
FEVER (its evaluation surfaces; the FEVER *training* store stays admitted), DBpedia-entity,
cqadup-android and cqadup-english. Do not open `work/m17/panel/`, `work/m17/alias/` or
`work/train/stores/`. Under `work/` open ONLY the files named in section 6 below.

## Files to read, in this order

1. `CLAUDE.md`, `m17/STATUS.md`, `m17/CODEMAP.md` (the `prepare_data.py` row, the
   `cache.py` row, and the whole "Reuse hazards" list — each hazard is a claim the code must
   honour).
2. `m17/registry.json`: `data.*` (sources, caps, `bucket_populations_and_dose_rule`,
   `measured_dose_after_pre_lock_rule`, `source_domain_map`, `new_term_min_*`, `discovery`),
   `training.*` (`candidate_bank_max_docs`, `positive_bank_policy`, `candidate_mix_*`,
   `candidate_construction`, `cache_identity`, `loss_definition.q_t`, `uninformative_list_stop`,
   `alias_pre_lock_diagnostic`, `new_row_initialization`, `overfit_divergence_check`),
   `execution_entry_missing` (the last two items are what step 5 had to deliver),
   `allocation_hours`, `serving`, `base_vocab`, `added_rows_max`, `teacher`, `teacher_revision`.
3. `m17/LEDGER.md` lines 120–150 (the protected-screen deferral ruling and the k8s
   decontamination record).
4. The implementation: `m17src/prepare_data.py`, `m17src/test_prepare_data.py`,
   `m17src/cache.py`, `m17src/test_cache.py`, `m17src/train.py` (`Preproc`,
   `load_warm_start`, `_load_prepared`, `_check_locked_config`, `main`), `m17src/vocab.py`,
   `m17src/support_manifest.py`, `m17src/common.py`.
5. Reused legacy code, only these: `m7src/table.py` (`tokenize`, `QueryTable`), `m7src/teacher.py`,
   `m7src/encoders.py` (the stella spec), `m7src/pool.py` (the `vecs.f16`/`meta.json` layout
   only), `m7/FREEZE.json`, `m10src/protected10.py`.
6. Results and the small text artifacts the builder wrote (no vectors):
   `results/m17_prepare_timing.json`, `results/m17_support_manifest.json`,
   `results/m17_k8s_source_manifest.json`, and under `work/` exactly:
   `work/m17/prepared/s2000/build_record.json`, `work/m17/prepared/s2000/base/prepared.json`,
   `work/m17/prepared/s2000/ext/prepared.json`, `work/m17/prepared/s2000/shared/cache.json`,
   `work/m17/prepared/s2000/stages/pool.json`, `work/m17/prepared/s2000/stages/bank.json`,
   `work/m17/prepared/s2000/stages/vocab.json`, `work/m17/prepared/s2000/vocab_terms.json`,
   `work/m17/prepared/s2000/pool_selection.json`.

## You may run

Read-only viewing of the files named here with `cat`, `sed -n`, `nl -ba`, `head`, `tail`,
`wc`, `ls` on a named path, `python3 -c` one-liners that only `json.load` a named file above
and print fields, and `grep`/`awk` **without** `-r`/`-R` over explicitly named files (a
non-recursive glob such as `m17src/*.py` is fine). Git: `git log --oneline -8`,
`git show --stat <sha>` and `git diff e1dd7b5 6fc3b6a -- m17src m17/CODEMAP.md` (the whole
step-5 diff). Plus exactly one test command from the repository root:
`.venv/bin/python -m pytest -q -ra m17src`. Do not run `prepare_data.py`, `train.py` or any
other script, and do not list directories under `work/` or `results/`.

## What this code is for

Zero is a 30,522 x 1,024 token lookup table serving as a query encoder against a frozen Stella
document index. M17 will, in a future registered 72-hour window on one RTX 3080, add up to 3,072
vocabulary rows, compare a listwise KL loss against the cosine-only continuation and test alias
consistency. Step 5 is the **prepared-data builder**: it turns the admitted training sources
into the directory `train.py --data` consumes — the query pool, the teacher targets, the
262,144-document bank, the v1 query vectors, the vocabulary extension and the candidate cache —
and it measured the preparation cost at two real sizes (2,000 and 10,000 queries) so the
allocation can be written before the lock. Nothing has been trained; no development-suite,
panel or protected read has happened; the registry is still `DRAFT_NOT_EXECUTABLE`. The first
timing forecast 20.4 h for the full 574,327-query pool, almost all in `cache.build`; commit
`6fc3b6a` blocked the scoring and partial-ordered the walk, and the rebuilt caches forecast 1.6 h.

## What I believe, and want you to break

- **Pool.** The general bucket is PAIR_SOURCES + QUERYTEXT_SOURCES restricted to
  `work/decontam/kept.json`, minus the mod-50 held-out rule, minus the step-2c exclusion file
  (families, normalized-text shas, alias TERMS, evidence document groups); positives are in the
  dataset's own qrels order; both alias views are separate `QuerySpec`s with the same family and
  never split across held-out/train; the 600,000 cap, the 10 % new-source cap on processed views
  and the four-passes ceiling are applied as registered; `--size` subsampling is seeded and
  stratified by source; MS MARCO is refused by name.
- **Domain join.** Every `QuerySpec.domain` comes from `doc_domain.tsv.gz` (document text only,
  threshold 3, k8s → cloud-software); two labels for one document or text group are refused;
  nqopen/triviaqa get one sentinel `source_doc` per source so they can never satisfy the
  20-document minimum on their own (conservative by design — say if you think it is wrong).
- **Labeled subset.** `LABELED_POSITIVE_BUDGET_SHARE = 0.5` (my choice, NOT a registered
  constant): the eligible labeled subset is chosen in seeded order before the bank exists so
  that all its positives fit in half the bank; the rest become query-only and are counted in
  `query_only_by_bank_budget`. At both timed sizes the budget did not bind. Tell me whether this
  is a protocol decision the owner must ratify before lock, and whether the choice is made from
  anything other than the training partition.
- **Teacher.** Raw text through stella's own tokenizer with the FREEZE-pinned instruction, fp16,
  L2-normalised, cached by sha256(text) with a preprocessing manifest; stella's serialized
  tokenizer ships with padding ON and the builder asserts the backend reproduces
  `m7src/table.tokenize` before use. Teacher encodes are not bit-reproducible run to run on this
  GPU, so the on-disk cache is the artifact of record.
- **Bank.** Positives of the selected subset first, then uniform stratified admitted documents to
  the cap (262,141/262,142 rows built); vectors from the existing stella `vecs.f16` memmap with
  encoder/revision/shape checks; the 1,648 admitted k8s documents encoded once with the frozen
  document tower (FREEZE `encoder_spec`) inside the cap.
- **v1** query vectors come from the released folded int8 table through `QueryTable` under the
  FREEZE preproc, artifact sha recorded in the cache identity.
- **Parity.** `load_warm_start(expect_vocab=30522)` with the real FREEZE hash check, then
  `verify_old_vocab_parity` on effective folded rows against the released table: max deviation
  9.76e-4 under tol 5e-3. The unfolded sidecar lacks `weights_folded`; the builder writes an
  augmented copy with `weights_folded: false` derived from non-empty scalars and then relies on
  parity to prove it. Break that derivation if you can.
- **Vocabulary and the two directories.** `discover` → `rank` → `select` → `init_new_rows` →
  `extend_tokenizer`; `base/` (C/L, no `new_rows`) and `ext/` (V/VL/VL-A) share one cache, bank
  and teacher file; `_load_prepared` refuses the wrong pairing. At 2,000/10,000 queries only the
  owner pin `k8s` is selected — expected because the minima are absolute.
- **Cache.** `cache.build` with the registry mixes and K=64; the identity covers every
  `training.cache_identity` item; the entropy block, the alias-view teacher-cosine distribution
  (flag < 0.7) and the warm-start KL summary are recorded, decide nothing. The new `_ranked`
  yields exactly the full lexsort order (see `test_partial_order_equals_full_lexsort_*`); blocked
  fp32 BLAS moves teacher scores by ≤ 3.6e-7, which changed two array hashes but no identity.
- **`train.py --rehearsal --data <dir>`** now smokes the real prepared directory; the bypass
  still covers only the status gate and the FREEZE hash and announces itself.
- **Timing.** `results/m17_prepare_timing.json`'s extrapolation subtracts the named fixed costs
  and is linear in queries; the cache stage is O(queries × bank) with the bank at the cap in
  both measurements. Check the arithmetic and whether any subtracted cost actually scales.

## What to report

Findings only, ranked P1 (wrong data reaches training or a registered rule is silently
violated, or a protected/denied surface can be read), P2 (a plausible single-researcher mistake
the code does not refuse, or a wrong hash/identity), P3 (everything else worth fixing). For each:
file:line, the concrete input or state that goes wrong, the observed vs. expected behaviour, the
smallest fix, and whether a test covers it. Do not report style. Do not confirm what works
except in one closing paragraph. If you believe a design choice above is a protocol decision
rather than an implementation detail, say so explicitly under a separate heading "Owner
decisions". End with the list of files you read.
