# M17 code and artifact map

| Path | Purpose |
|---|---|
| `m17/PLANNING.md`, `registry.json` | Prospective design and draft constants; no runnable training protocol |
| `m17/LEDGER.md` | Owner scope, pre-observation diagnostics, A1 adoption of the three follow-up comparisons, A2 plan-review revision |
| `m17/REVIEW.md` | Dispositions of the two Luna planning reviews |
| `m17src/planning_probe.py` | P0 resource/tokenization check; `--followup` runs P0b only. Hand-authored text, disposable random GPU weights, offline Nano tokenizer comparison. |
| `results/m17_planning_probe.json` | Original P0 observation; original source preserved at `2bca40d` |
| `results/m17_tokenizer_followup.json` | P0b: shared-piece counterexamples, broader illustrative terms, Nano vocabulary equality |
| `research/m17-vocabulary-*.md`, `m17-static-interaction-*.md`, `m17-data-training-*.md` | Three primary-source Luna research notes with exact access logs |
| `research/m17-plan-review-loss-*.md`, `m17-plan-review-data-*.md` | The two Luna planning reviews behind `REVIEW.md` |
| `research/m17-additional-avenues-*.md` | Follow-up idea shortlist that led to A1 |
| `research/m17-astra-plan-review-*.md`, `m17-astra-plan-brief-*.md` | Codex Astra whole-plan review after A2, and its brief |
| `research/m7-data-licensing.md` (Kubernetes row) | The completed step-2a licensing row: pinned SHA, clone route, quoted CC BY 4.0 text and carve-outs, acquisition-terms finding, decontamination coverage. The source is admitted for download only because this row is complete |
| `research/m17-k8s-attribution.md` | The CC BY 4.0 attribution notice, source, pinned revision and modification statement. **Ships with any derived weights**; update the modification statement if the processing changes |
| `results/m17_k8s_source_manifest.json` | Step-2a acquisition record: counts, bytes, JSONL sha256, dev-suite screen result and the deferred protected-screen blocker. No document text |
| `work/m17/sources/kubernetes-website`, `work/m17/sources/k8s_docs_en.jsonl` | Gitignored clone at the pinned SHA and the extracted English docs (path, title, stripped text, sha256). Not admitted training data until the executor's protected screen passes |
| `m17src/common.py` | Paths, registry/freeze loading, hashes and `require_executable` — the status gate every entry point calls. Sets `M7_ENCODER=stella-400M-v5` before legacy imports and puts `m17src` ahead of `m7src` on `sys.path` (else `import train` finds the legacy driver). **`admit_read(path) -> Path`**: call it in every module before opening anything (`p = common.admit_read(p)`); it resolves symlinks and refuses `frozen_eval/untouched-`, `m9reserve`, `reserved_qrels`, `lotte`, and the reserved four by name under `results/frozen_eval` or a `qrels` path — admitted FEVER *training* stores stay admitted. `admit_write` (used by `write_json`) refuses `results/perquery.json`, `results/frozen_eval/` and `m7/FREEZE.json` |
| `m17src/cache.py` | Candidate-cache schema, deterministic construction (full score/id ordering, quota walk, backfill, per-query RNG serialized exactly as `RNG_RECIPE`), provenance counts, identity hash (ordered per-query metadata; the v1 artifact and teacher preprocessing manifests are required) and the B2-format entropy block. A labeled query with a positive outside the bank is refused by qid, never converted to query-only; `save`/`load` carry and verify per-array sha256. The entropy block shares only B2's inner metric fields (quantiles, p_max, threshold shares); it does not reproduce B2's `arms` nesting, by decision (Astra P3-29). |
| `m17src/vocab.py` | Term discovery, ranking, selection (minima, abbreviation policy, caps, owner pins), count-weighted row init, `single_word` tokenizer extension, old-vocabulary parity and the P0b drift report. Ruling A3: a term's domain counts are one vote per DISTINCT supporting document (not per query occurrence), and a query record's `domain` is its source document's `support_manifest.document_domain` — `discover` requires `source_doc` and refuses two domain labels for one document. A single `[UNK]` piece is a candidate, not an existing token; a supplied `abbreviations` inventory is the authority in `select` (the regex fallback misses vowel-carrying abbreviations such as `iam`) |
| `m17src/train.py` | **The one M17 driver.** Arms C/V/L/VL/VL-A, warm start verified against `m7/FREEZE.json:training_checkpoint_sha256` with explicit unfolded metadata (`--rehearsal` bypasses the hash and says so), per-bucket without-replacement streams, the four loss terms, snapshots/recovery/resume, run record. Batch composition is `data.measured_dose_after_pre_lock_rule` (204/32/10 at batch 256), derived proportionally for other batch sizes. `run()` itself calls `require_executable`; resume refuses a checkpoint bound to a different arm/seed/phase/dose/lr/cache and restores the anchor init from it; the divergence read compares cosine + listwise on both sides and survives resume. Snapshots also store `rows_fp32` so averaging is not FP16-rounded |
| `m17src/export.py` | Fold-once effective rows (`rows_fp32` preferred), registered snapshot averaging (exactly three, complete identities, one unique registered step each), bundle build with the copied frozen `encoder_spec` (incl. `config_kwargs`), the dim/added-row/35M-parameter limits, the CC BY 4.0 `ATTRIBUTION.md` copied in whenever the vocabulary provenance lists `k8s-docs-en`, and the five M17 gates (separate from M11's, which stay bound to `m7/FREEZE.json`). `gate_artifact` verifies the four GENERATED-byte hashes (model/tokenizer/config/attribution) and not the self-reported external provenance; a fixture bundle is the explicit `fixture=True` flag only `rehearse17.py` passes, never an inferred fallback id; `gate_conformance` rejects non-finite rows/scales/outputs before comparing against `export.CONFORMANCE_TOL` (1e-5, torch vs numpy, recorded in provenance — not the registry's 1e-6 loader-vs-loader bound) |
| `m17src/loader_np.py` | Standalone numpy loader: eager fp32 vs resident int8 codes/scales with per-query dequantization, parity check, weight bytes reported separately from process RSS |
| `m17src/evaluate.py` | Exact dense retrieval, per-domain nDCG@10/Recall@10, paired family bootstrap, DBSF@100 through `m12src/qfusion.py`, alias overlap/rank correlation. `search` blocks over documents as well as queries (`DOC_BLOCK` 65,536; a 4096 x 5.2M score matrix is 85 GB). Each block keeps its WHOLE cutoff tie and selects by (-score, ascending document id) before the global merge, so a tie wider than `k` cannot lose its smaller document ids to `argpartition` and the result does not depend on the block size. Document ids must be unique or `search` refuses. `evaluate(query_ids=...)` keys the run by the panel's own `query_id` and refuses a qrels/domains/families key-set mismatch instead of scoring everything unjudged. `paired_family_bootstrap(domains=...)` recomputes the equal-weight DOMAIN MACRO inside every replicate — `delta` is that macro, `delta_query_mean` the labelled query-weighted number. Dev-suite and panel surfaces are flag-gated and unwired pre-clock |
| `m17src/panel_build.py` | **Step 2c.** Held-out-only draws (`trainmix.heldout`'s mod-50 rule) from SQuAD/HotpotQA/Mr.TyDi plus Kubernetes candidate queries; the heuristic domain classifier; families by shared gold document group and identical normalized text; the 50/50 family split; the declared corpus; the ancestor-manifest exposure screen (`STREAMS_EXPECTED` names all ten ancestor streams by exact identity and the seal requires SET EQUALITY — a missing stream makes every family exposure-unknown, it is not satisfied by a sibling's prefix); the expected-SE block; the seal. The screen records the panel and alias file hashes and `seal()` refuses a receipt written for different bytes (it re-stamps `panel_sha256_sealed` so re-sealing stays idempotent); `assert_pending_unjudged` refuses to overwrite `results/m17_panel_pending_judgments.jsonl` once any row has a `relevant_yes_no`/`judge` — no merge flag, and ingesting judgments into qrels is later work. `--reuse-cache` re-labels from `work/m17/panel/*_cache.jsonl` without re-reading 2.9 GB of stores |
| `m17src/alias_test_build.py` | **Step 2c.** The 200-pair held-out alias test: source-stated acronym/expansion and alias/canonical equivalences with quoted citations, ambiguous short forms flagged and left `PENDING_HUMAN`, families, and `write_exclusions`, the exclusion file the training-pair builder reads (now also the normalized short form/expansion of each pair and its evidence document group). A phrase that hit the four-word marker cap or does not start at a phrase boundary is `PENDING_HUMAN`, not `VERIFIED_BY_SOURCE`. Family document keys are SOURCE-QUALIFIED, and general-domain labels use the panel's pinned classifier threshold (recorded in the manifest). Refuses to rewrite an answered review sheet |
| `results/m17_panel_manifest.json` | The sealed panel: per-domain query/family counts, partitions, corpus size, relevant counts, exposure labels, pending counts, expected SE and every file hash. **PROVISIONAL** until the pending judgments are resolved |
| `results/m17_panel_selection.jsonl` | The selection partition mirrored out of `work/` because it is under 2 MB. The audit partition's text stays in `work/m17/panel/audit.jsonl`; only its hash is published |
| `results/m17_panel_pending_judgments.jsonl` | The one human review sheet: 360 Kubernetes (query, candidate) rows and the ambiguous alias pairs. `relevant_yes_no` and `judge` arrive empty and must stay empty until a person fills them |
| `results/m17_alias_test_manifest.json` | Alias-test counts by source/kind/domain/judgment status, the extraction rules and the file hashes. No pair text |
| `m17src/support_manifest.py` | **Step 2b.** Deduplicated document/query counts per admitted source, the fixed source-to-domain map (mirrored into `registry.data.source_domain_map`) plus ruling A3's `document_domain`, which sub-assigns each deduplicated document of a `general`-mapped source with `panel_build.classify` at the same threshold (`per_domain` is per-document, `per_domain_source_level` keeps the step-2b view; a pre-A3 `counts_cache.json` prints a re-run notice and falls back), document-group and query-family IDs, the measured bucket populations and the applied `pre_lock_rule`. The classifier threshold is the one `results/m17_panel_manifest.json` pinned (3), not `panel_build.MIN_SCORE` (4); which was used is recorded in `domain_assignment.classifier_min_score_source`. Drops queries whose normalized-text sha appears in step 2c's exclusion file, and queries whose normalized text IS a held-out alias term (`queries_removed_as_heldout_alias_term`). `load_exclusions` returns `{families, text_shas, terms, doc_groups}` and refuses a file lacking the term/doc-group keys. MS MARCO is refused by name inside `iter_store`, not only in the source lists. `--reuse-counts` re-applies the dose rule from `counts_cache.json` without re-reading 2.9 GB of stores |
| `m17src/alias_pairs.py` | **Step 2b.** Bulk structural alias pairs: Kubernetes glossary `aka` forms and initial-matched `Expansion (ABBR)` definitions evidenced in the same admitted document, substituted into a carrier sentence so each pair is two query *views*. Ambiguous short forms are dropped, ESCI is not mined, and `--exclude-families` removes a training pair whose family, view text, alias TERM (short form or expansion) or evidence document group is held out — family ids and full-view shas alone removed nothing |
| `results/m17_support_manifest.json` | Step-2b counts, per-domain totals, unpopulated-domain gaps, family concentration, bucket populations, passes at 6000x256, the dose-rule outcome and file hashes. No document or query text |
| `results/m17_alias_spotcheck_sample.jsonl` | The seeded random 2% sample of the alias pool for Dylan's human spot check; the only step-2b file that carries text |
| `work/m17/manifest/` | Gitignored step-2b artifacts: `doc_groups/*.tsv.gz`, `query_families.tsv.gz`, `alias_pairs.jsonl`, `alias_pairs_summary.json`, `counts_cache.json`, and step 2c's `alias_test_families.json`. Hashes are published in the result |
| `work/m17/panel/`, `work/m17/alias/` | Gitignored panel and alias text: `panel.jsonl`, `selection.jsonl`, `audit.jsonl`, `corpus.jsonl`, the two classifier caches, `ancestry_screen.json`, `alias_test.jsonl` |
| `m17src/prepare_data.py` | **Step 5. The one prepared-data builder.** Ten stages, each cached under `<out>/stages/<name>.json` and resumable: admitted query pool (general / unpaired-coverage / alias buckets, the caps, the seeded stratified `--size` subsample and the whole-family held-out slice), the `(source, document id) -> domain` join (`doc_domain.tsv.gz`, `join_domain` refuses two labels for one document or text group), the protected screen (OFF pre-clock, recorded `deferred_to_clock`), teacher query encoding, the bank, v1 query vectors, old-vocabulary parity, vocabulary selection/extension, the candidate cache with its two pre-lock diagnostics, and the two `prepared.json` manifests. Output layout: `<out>/shared/` holds the one cache, bank, teacher vectors and warm start; `<out>/base/` (arms C/L, no `new_rows`) and `<out>/ext/` (V/VL/VL-A) hold their own tokenizer, `student_ids.json` and manifest and symlink the shared files. Timings and rates are in `results/m17_prepare_timing.json` |
| `m17src/rehearse17.py`, `m17src/conftest.py`, `m17src/test_*.py` | The tiny synthetic end-to-end rehearsal and the pytest checks that run on it and on fixture-scale inputs. The rehearsal deletes an existing `--out` only when it is empty or carries the `.m17_rehearsal` marker it wrote itself |
| `results/m17_rehearsal.json`, `results/m17_rehearsal_step4.json` | The step-3 and post-review (step 4) pre-clock rehearsal records: stages, scaled fixture constants, gate results, loader parity, synthetic-only evaluation. Not a quality observation or a rate forecast |

Reproduce diagnostics from the repository root using `.venv/bin/python
m17src/planning_probe.py --gpu` or `--followup`. Both refuse to overwrite their existing result.
For a new measurement, register its reason and use a separately named result in an isolated
checkout; do not remove the recorded observation to make a command pass. The script's source,
bundle and ledger hashes are embedded in each result. Ledger changes after a measurement are
expected; git retains the version bound to that observation.

Future M17 implementation belongs in `m17src/`, small mutable artifacts in `results/m17_*`,
and heavy caches/checkpoints/bundles in gitignored `work/m17/`. Reuse existing modules where they
fit; no generic experiment framework or legacy-path rename.

The step-3 implementation now exists and rehearses end to end, but it is **not** an executable
experiment: the registry is still `DRAFT_NOT_EXECUTABLE`, `train.py` refuses a real arm, and the
real data path (teacher encoding, the admitted-source query pool, bank mining, the panel and
dev-suite readers) is deliberately unwritten — `train.py --data` expects a prepared directory
that step 5 produces. Run the checks with `.venv/bin/python -m pytest -q -ra m17src` and the
rehearsal with `.venv/bin/python m17src/rehearse17.py` (see `HARNESS.md`).

## Step-5 builder pitfalls (measured, 2026-09-11)

- stella's serialized tokenizer arrives with **padding enabled**. `Tokenizer.from_file(...)` /
  `backend_tokenizer` must be given `no_padding()` and `enable_truncation(512)` before a single
  id is taken, or every student bag is filled with `[PAD]` rows. The HF wrapper hides this, so
  the builder asserts the backend reproduces `m7src/table.tokenize`'s ids under the FREEZE
  preproc before using it.
- `work/runs/p35w-2m-s2500.meta.json` (the unfolded warm start) **does not state
  `weights_folded`**, which `train.load_warm_start` requires explicitly. `prepare_data` copies
  the npz verbatim (the bytes still hash to `training_checkpoint_sha256`) and writes an
  augmented sidecar whose `weights_folded: false` is derived from the checkpoint's non-empty
  per-token scalars and then *proved* by the parity stage. Never edit the original sidecar.
- Do not call `m7src/pool.build()` to reach the frozen document vectors: it REBUILDS a stale
  cache, and that cache is a 12.6 GiB artifact. `prepare_data.PoolReader` reads
  `work/pool/stella-400M-v5/meta.json` + `vecs.f16` directly and refuses a wrong
  encoder/revision/shape.
- `cache.build` scores the whole bank per query in numpy (two 262,144 x 1024 mat-vecs plus two
  full `lexsort`s). It is the dominant preparation cost and it is CPU-bound; see
  `results/m17_prepare_timing.json` for the measured per-query seconds and the extrapolation.
- Query-text-only sources (nqopen, triviaqa) ship no document, so `vocab.discover`'s
  `source_doc` is one sentinel unit per source. Ruling A3 counts one vote per distinct
  supporting DOCUMENT; treating each query as its own document would restore per-occurrence
  counting. The consequence is deliberate: a term supported only by those sources cannot reach
  the 20-document minimum.
- The registered bank cap cannot hold every positive of every admitted labeled query (~390k
  distinct positives against a 262,144-row bank). The builder therefore chooses the eligible
  LABELED subset before the bank exists, in a seeded order, up to half the bank, and records
  `query_only_by_bank_budget`. Queries outside that subset enter the pool as query-only
  examples by selection; `cache.build` still refuses, by qid, any labeled query whose positive
  is outside the bank.
- `train.py --rehearsal` **with** `--data` now smokes the real prepared directory instead of
  dispatching to `rehearse17`; without `--data` it is still the synthetic fixture world. The
  bypass covers the status gate and the FREEZE hash only, and the checkpoint sha is printed.

## Reuse hazards to resolve before training

- `work/train/stores/msmarco-pos.json` exists and `work/decontam/kept.json` still carries
  `msmarco-train`. MS MARCO is validation only; `support_manifest.DENIED_SOURCES` refuses it by
  name and any new data path must do the same.
- FEVER's store is pre-tokenized with spaces inside parentheses (`( EP )`). A definition pattern
  written for `(EP)` silently mines zero pairs from it; that was the diagnosis, not an absence.
- Query families are not uniform: three families hold 108,922 of 561,480 training queries after
  the 50-document hub cutoff, because chains of sub-cutoff documents still connect. Draw held-out
  slices, the panel split and the alias test outside them.
- A document's domain must be computed from the DOCUMENT text alone (`support_manifest.document_domain`),
  never from the query that retrieved it, or the same document votes differently in different
  terms. `vocab.discover` does not compute this: the CALLER supplies each query record's
  `domain` alongside its `source_doc`, and discovery only refuses two different domain labels
  for the same `source_doc`. Each supporting document votes once — do not re-introduce
  per-occurrence domain counting. Binding the labels to a source-qualified doc→domain artifact
  is step-5 data-builder work (Sol P1-10), not something discovery verifies today.
- `panel_build` must not import `support_manifest` at import time: `support_manifest` imports the
  classifier from it (lazily), and a top-level import back would make the pair circular.
- `m7src/train.py` hardcodes tokenizer/init/data assumptions and invokes development scoring;
  `sweep.one` also appends to M7 reporting files. Do not call old drivers as an M17 launcher.
- `m7src/table.py::QueryTable`, ragged bags, sqrt occurrence weights, folding and normalization
  are reusable primitives, but bind the new tokenizer/table identity and M17 output paths.
- Set `M7_ENCODER=stella-400M-v5` before importing legacy teacher modules; their default may be
  bge-base. Cache dtype, prompt, Dense projection/bias and teacher revision matter.
- The unfolded checkpoint includes learned weights. The release already folds those weights
  into the rows. Verify old-vocabulary parity before constructing new rows; never fold twice.
- New-token IDs belong only to the student. Feed the unchanged raw text through the teacher's
  own tokenizer for targets. Reuse cached targets for unchanged text/teacher preprocessing.
- Alias views share a same-intent query-family split and carry verified pair IDs in the same
  cache schema. Both views count as examples in every arm; only VL-A adds their consistency
  loss. Preserve individual teacher targets and nullable positive labels for each view.
- Keep the registry's late step-bound snapshots as well as time-bound recovery checkpoints.
  Average effective float32 folded rows only; no separate averaging of rows/scalars or mixing
  tokenizers, runs or row scales. Each full run is read as endpoint and fixed average only.
- New whole-word rows change sqrt-count sharing with constituent subwords elsewhere in a query.
  Sum initialization is not universally output-preserving; P0b demonstrates the mechanism.
- `single_word=True` uses word boundaries; underscores and punctuation deserve fixtures.
  Padding stays disabled. Preserve CLS/SEP, empty fallback and truncation parity through numpy,
  the serialized tokenizer, ONNX and the intended FastEmbed loader.
- The numpy int8 path materializes the whole table as fp32. Report process RSS separately from
  compressed weight bytes. P0 GPU rates used synthetic resident tensors and are not a forecast.
- M17 plans resident int8 codes/scales with per-query dequantization. Compare isolated fresh
  processes on identical model bytes; preserve float32 math, unique-ID/count order, fallback and
  truncation. Bench the registry's length/batch strata and retain eager loading if adoption
  checks fail. This does not redesign ONNX/FastEmbed or change the public query API.
- M11's v1 gates are deliberately tied to `m7/FREEZE.json`. Reuse their parity approach for a
  separate M17 bundle; never weaken the old gates so new bytes can masquerade as the v1 freeze.
- Copy the frozen `encoder_spec` into the candidate bundle's document-encoder metadata and
  assert its revision, dimensions, pooling, projection and prefixes match during the build.
  Record training/evaluation bank hashes in provenance. Do not require a production corpus hash
  or a new caller argument: the same encoder space supports different users' document corpora.
- Reuse `m12src/qfusion.py` for fixed DBSF@100. Quality comes from exact runs; ANN measurements
  establish serving costs only.
- No reads of `results/frozen_eval/untouched-*`, reserved qrels caches, `work/m9reserve`, or
  six-set/LoTTE payloads. Existing fingerprint products are not authority to open their inputs.

Verification for this planning batch is JSON/arithmetic consistency, Python compilation,
completed diagnostic output, links and scope diff. Legacy training/evaluation suites are not
needed for documentation and isolated feasibility measurements.
