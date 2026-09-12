# M17 planning review dispositions — 2026-09-11

Two independent Luna planning reviews, with named-file briefs and protected-read exclusions:
[loss/design](../research/m17-plan-review-loss-2026-09-11.md) and
[data/budget](../research/m17-plan-review-data-2026-09-11.md). Their access logs were inspected:
only the named design, historical guidance and permitted diagnostic artifacts are listed; no
reserved or six-set/LoTTE payload reads. These reviews do not certify an unwritten executor.

| Finding | Parent disposition and exit |
|---|---|
| Loss P1 / data F4: query-only candidates | Clarified in registry: separate labeled/query-only mixtures, `has_label`, nullable positive ID and deterministic dedup/fill; teacher hits never become labels. One shared cache schema with complete identities. |
| Loss P2: exact objective | Adopted: normalized-query definitions, shared document cache/temperature, query-teacher cosine target, batch-mean KL direction/reduction and effective-row anchor now explicit in registry. |
| Loss P3 / data F1: unpriced real workload | Already an explicit execution dependency; strengthened with phase stop/recovery rules. Reject selecting from an accidentally incomplete matrix. No synthetic throughput extrapolation or unrecorded post-lock dose/seed cuts. Real timing remains owed. |
| Loss P4: index identity | Adopted build-time equality of bundle document metadata with M7's encoder spec. Declined binding a general encoder to one production corpus hash or adding loader arguments: that would change the promised API and confuse document-space identity with corpus identity. Bank hashes remain training/evaluation provenance. |
| Data F2: broad support is conditional | Support manifest is the first data exit, with per-domain deduplicated counts and fewer rows when unsupported. No claim that current fixtures establish broad coverage. Actual manifest remains owed. |
| Data F3: frozen v1 / DBSF | Existing plan already requires pre-update folded-int8 v1 parity and fixed DBSF comparisons. Local source table, training checkpoint and lineage hashes were verified against M7's freeze. No v1/M13 assets changed. |
| Data F5: implementation breadth | Bounded to one driver/cache schema and existing primitives; optional polish can be deferred, mandatory missing evaluation/parity yields an incomplete candidate. No framework or M13 change. |
| Parent: positive-bank memory | Clarified: the selected labeled queries' known positives must fit inside the bank cap, rather than adding an unbounded extra positive bank. |

**Owner:** M17 implementing session. **Execution exit:** real inputs/panel, measured allocation,
ratified prospective protocol, working driver and two independent implementation reviews.
Planning fixes do not remove those dependencies.

## A1 scope update

Dylan's subsequent adoption of alias consistency, checkpoint averaging and int8 resident
loading is recorded in `LEDGER.md` A1. The earlier Luna reviews covered the preceding draft;
they are not represented as reviews of this amendment or its future implementation. A1's
planning check covers the matched alias views, fixed forms/read counts, budget arithmetic,
audit isolation, links and preservation of the original results. Implementation reviews must
cover the pair sampler/loss, snapshot/resume/export path and selected-row dequantization too.

## Codex Astra whole-plan review after A2 — 2026-09-11

Brief: [m17-astra-plan-brief](../research/m17-astra-plan-brief-2026-09-11.md). Log with the full
report: `research/m17-astra-plan-review-2026-09-11.log`. Read-only, xhigh effort. Its access
log lists only the eighteen named files; no reserved, six-set or LoTTE payload. Astra confirmed
the arithmetic and found no violation of the cap, frozen index or M13 isolation, and would keep
A2 non-executable. Dispositions:

| Finding | Disposition and exit |
|---|---|
| P1 decision rules incomplete | Adopted: `decision_protocol` block pins the six dev components, the technical metric (`cqadup-programmers` dense nDCG@10), eligibility, control-map walk, tie, seed-disagreement, no-survivor and averaging-eligibility rules, and int8 artifact binding. |
| P1 panel could still decide via vetoes | Adopted: ambiguous-sense veto and coverage-slice alternative removed from every predicate; panel and alias test are report-only. |
| P1 sealed panel not necessarily unseen by warm start | Adopted: ancestor decontamination against all M7 training manifests through approved fingerprint interfaces; exposure-unknown labeling where ancestry is unavailable. |
| P1 rehearsal/V0 observation boundary | Adopted: rehearsal scores synthetic fixtures only; protocol, vocabulary hash, tokenizer hash, seeds and cache identity committed before V0 or any real read. |
| P1 Kubernetes evidence absent | Adopted: proposal row added to `research/m7-data-licensing.md` with the required checklist; still not admitted for download. |
| P2 bucket repetition hidden by "2.6 passes" | Adopted: per-bucket populations, four-pass ceiling, share-then-steps shrink rule, processed-view basis for the 10% cap, structural bulk alias pairs with spot checks. |
| P2 teacher entropy wrong stop criterion | Adopted: replaced with warm-start student-teacher KL rule; surviving matrix defined (C, V). Entropy kept descriptive. |
| P2 candidate mixture underdefined | Adopted: `candidate_construction` block: positive choice, unique-hit quotas, backfill, tie order, RNG derivation, bank sampling, provenance counts. |
| P2 alias weight 1.6x per view | Adopted: alias term batch-normalized (2/B times pair sum); pre-lock teacher-pair cosine and gradient-share diagnostic. |
| P2 count-weighted initialization | Adopted: new row = sum sqrt(c_j) R_j over the isolated term's pieces, special tokens excluded. |
| P2 discretionary vocabulary rules | Adopted: `vocabulary_ranking` block: residual, support weight, ordering, domain assignment, abbreviation threshold, breadth condition. `k8s` exception kept. |
| P2 anchor inertness not established | Adopted: described as coefficient continuity; optimizer pinned to M7's Adam defaults; anchor update share measured in smoke. |
| P2 preparation underpriced, pair verification cost | Adopted in part: bulk structural pairs remove the per-pair judging cost; measured preparation costs at two sizes added to execution-entry requirements. Real timing still owed. |
| P3 loader decided by latency noise | Adopted: 200 batches x 5 processes, nearest-rank quantile, natural-reuse fixtures, Linux labeling; Mac gate separate. |

**Owner:** M17 implementing session. **Exit:** unchanged; the executor and its two independent
implementation reviews remain owed. This planning review does not certify an unwritten executor.

## Codex Astra implementation review of `m17src/` — 2026-09-11 (step 4, review 1 of 2)

Brief: [m17-astra-impl-brief](../research/m17-astra-impl-brief-2026-09-11.md). Log:
`research/m17-astra-impl-review-2026-09-11.log` (gitignored, local). Read-only, xhigh effort. A
first attempt read nothing because the brief permitted only two Python commands; the second read
the 37 named files and the two permitted `work/m17/manifest` summaries, nothing else. Its pytest
and `py_compile` runs failed inside Codex's read-only sandbox (no writable temp directory), so
the 141-test pass was ours to confirm; it was. Astra found the loss terms, KL direction,
temperature, masking, anchor rows and learning-rate schedule correct for valid inputs, no
document-tower training, no `torch.compile`, no change to M11's gates. It would not authorize
execution: 18 P1, 8 P2, 3 P3, mostly in how inputs are constructed, validated and resumed.

Dylan (2026-09-11): "make sure we don't over-engineer, this is supposed to be a fairly easy
re-training." Dispositions therefore keep the fixes a careless researcher would need and record
what was dropped as adversarial-only. Fixes landed in two Opus passes: driver/cache/export/loader
(`e2db77e`), then the builders and evaluator (next commit).

| Finding | Disposition and exit |
|---|---|
| P1-1 batch dose 192/32/16 instead of the measured 204/32/10 | Fixed: `RunCfg` consumes `data.measured_dose_after_pre_lock_rule`; the test that enshrined the old fractions now asserts the registered dose. |
| P1-2 prepared-data path not arm-aware, hashes empty, arbitrary post-lock steps/seeds | Fixed: `_load_prepared` refuses `new_rows` for C/L and requires them for V/VL/VL-A, fills tokenizer/vocabulary/cache hashes from the files, and real runs refuse an unregistered steps/batch/seed. |
| P1-3 held-out alias pairs enter the alias stream; short batches on empty buckets | Fixed: pairs only from training-bucket rows with exactly two views (a/b) of one family; an undersupplied bucket raises. |
| P1-4 resume accepts a different experiment | Fixed, minimal: checkpoint carries arm, seed, phase, steps, composition, temperature, lrs, anchor weight, hashes, table shape and the anchor init; any difference refuses; stream population length checked. No digest framework. |
| P1-5 warm start not verified against the freeze | Fixed: file sha checked against `m7/FREEZE.json:training_checkpoint_sha256` (the unfolded checkpoint), explicit fold metadata and one finite positive scalar per row required; `--rehearsal` announces its bypass. |
| P1-6 averaging accepts a fourth snapshot / missing identities | Fixed: exactly three, complete identities, one unique registered step each. |
| P1-7 cache identity ignores v1 identity and per-query metadata; arrays trusted on load | Fixed: empty v1/preprocessing refused, ordered per-query record hash added, per-array sha verified on load. |
| P1-8 labeled queries silently become query-only | Fixed: refused by qid; the eligible subset must be chosen before the cache. |
| P1-9 partial ranking breaks the tie rule | Fixed: one full lexsort by (-score, bank id); tie test across block boundaries. |
| P1-11 protected-path checks not at every open site; `run()` bypasses the status gate | Fixed, minimal: a 15-line `common.admit_read` (realpath, forbidden substrings; admitted FEVER training stores stay admitted) at the real open sites; `train.run()` calls `require_executable`. Dropped: canonical artifact-type admission layer (adversarial-only). |
| P1-12 writers can hit `results/perquery.json`; rehearsal `rmtree` on any dir | Fixed: `common.admit_write` refuses the three frozen destinations; rehearsal deletes only an empty dir or one carrying its own marker. |
| P1-14 gates check internal consistency, not the locked artifact | Fixed: all provenance hashes, `config_kwargs`, dim 1024, row count, 35M cap with scalars, directory entries; Kubernetes attribution copied into the bundle when provenance lists `k8s-docs-en`. |
| P1-15 conformance fails open on NaN | Fixed: non-finite rows/scales/outputs/errors and bad scales refused before the tolerance; 1e-5 torch-vs-numpy bound recorded in provenance, distinct from the 1e-6 loader bound. |
| P2-19 fp16 snapshot rows before folding/averaging | Fixed: fp32 rows saved beside the legacy table; export prefers them. |
| P2-20 RNG recipe differs from the registered one | Fixed: implemented literally with a golden-value test; recipe and version in identity. |
| P2-23 divergence check mixes objectives, resets on resume | Fixed: separate cosine/listwise accumulators in state, same components both sides, 2,000 unique disjoint held-out queries for real runs. |
| P3-28 vacuous assertions; no numerical KL/anchor test | Fixed: assertion replaced; hand-computed KL and anchor fixtures added. |
| P3-29 entropy block flattens B2's `arms` nesting | Accepted as registered: only the inner metric block is shared with B2; noted in CODEMAP. No code change. |
| Rehearsal could not run as a script (`import train` resolved to M7's driver) | Found during the fix pass: `common.py` path order corrected. Pytest had hidden it. |
| P1-10 held-out alias families never excluded from training (0 pairs removed) | Fixed: the exclusion file carries normalized held-out terms and evidence document groups; the pair builder and the manifest's query pass drop matches; an old-format file is refused. Artifacts regenerated (step 4d). |
| P1-13 MS MARCO refusal only by three exact names in configured lists | Fixed: the store reader refuses any name or path containing `msmarco`. |
| P1-16 evaluator keyed by position while the panel contract keys by query id | Fixed: `evaluate(query_ids=...)` with key-set validation; the panel test goes through the public evaluator. |
| P1-17 stale ancestry screen could certify a rebuilt panel | Fixed: the screen records panel and alias hashes; `seal()` refuses a mismatch. |
| P1-18 alias extraction can verify a truncated phrase | Fixed: phrases that hit the word cap or start mid-phrase are `PENDING_HUMAN`. Regenerated alias test: 120 verified, 80 pending (was 141/59). |
| P2-21 classifier threshold 4 vs the panel's pinned 3; query domain not guaranteed document-derived | Fixed: the pinned threshold is read from the panel manifest and recorded; discovery requires a source document and refuses conflicting labels for one document. Support manifest regenerated. |
| P2-22 abbreviation detection / `[UNK]` | Fixed in part: `[UNK]` piece counts as a candidate; the supplied inventory is documented as authoritative (the "ignored inventory" half did not reproduce). Punctuation-led units such as `.NET` remain unsupported, recorded. |
| P2-24 bootstrap estimates the query mean, report shows the domain macro | Fixed: domain macro recomputed inside each replicate; query mean kept and labelled. |
| P2-25 exact search allocates 4096 × corpus | Fixed: document blocking with exact merged top-k, ties by document id. |
| P2-26 rebuilding erases human judgments; seal ignores them | Fixed in part: builders refuse to overwrite answered review rows. Judgment ingestion into qrels is built when the judgments exist, not before. |
| P3-27 rehearsal parity/resume claims overstated | Accepted as a limitation, no code change: the rehearsal certifies that the modules run and refuse, not lineage parity; the real smoke in step 5 interrupts an active run. |

**Dropped as adversarial-only (Dylan's over-engineering ruling):** a canonical artifact-type read
admission layer, config digests beyond the fields listed, packaging contracts beyond the
attribution copy, judgment ingestion before judgments exist, `.NET`-style punctuation units.

**Owner:** M17 implementing session. **Exit:** Codex Sol implementation review (2 of 2) on the
fixed code, then one P1-only re-check if Sol finds P1s; then step 5 timings.

## Codex Sol implementation review of `m17src/` — 2026-09-11 (step 4, review 2 of 2)

Brief: [m17-sol-impl-brief](../research/m17-sol-impl-brief-2026-09-11.md) (the Astra brief with
the review number changed; Sol did not see Astra's report). Log:
`research/m17-sol-impl-review-2026-09-11.log` (gitignored). Read-only, high effort, on the code
after the Astra fixes and the step-4d regeneration. Access log: the 37 named files and the two
permitted `work/m17/manifest` summaries; one non-recursive grep of `LEDGER.md` returned three
lines outside the requested sections; no protected path. Codex's sandbox again could not run
pytest or `py_compile` (no writable temp dir); the 206-test pass is ours. Sol confirmed the
KL/cosine/anchor/alias arithmetic, warm-up and decay, the held-out cadence, the cache walk and
tie-break, the RNG recipe, fold-once averaging over fp32 rows, numpy/torch loader parity,
`domain_of`, DBSF via M12 and the domain-macro bootstrap. 10 P1, 8 P2, 4 P3.

| Finding | Disposition and exit |
|---|---|
| P1-8 exact search not exact for ties wider than k (per-block argpartition) | Fixed: full (-score, doc id) ordering inside each block before the cut; test with >k ties across blocks. |
| P1-9 evaluation intersects keys (BM25 run, bootstrap baseline, alias views) | Fixed: exact key equality required, mismatch raises. |
| P1-4 export accepts a same-size tokenizer that is not the snapshot's; endpoint export needs no identity | Fixed: tokenizer sha compared with the snapshot's record; endpoint export requires the same identity fields as averaging; `gate_artifact` claim narrowed to what it checks. |
| P1-5 fixture marker inferred from a non-CLS fallback id bypasses dim/cap checks | Fixed: explicit `fixture` provenance flag set only by the rehearsal; fallback id range checked in every mode. |
| P1-7 seal accepts a prefix-matched ancestor stream set | Fixed: exact stream names required, set equality. |
| P1-3 resume binding omits cadence, held-out size, snapshot steps, held-out slice | Fixed: those four added to the binding. Dropped: prepared-manifest and code-hash binding (adversarial-only). |
| P1-6 protected reads/writes bypassable at some openers | Fixed in part: `sha_file`, the synthetic-fixture check and `loader_np --json` go through the helpers. Dropped: universal admission at every builder `open` and rglob removal; the builders read admitted training stores only, by name. |
| P1-1 status string is not a lock receipt; `run()` accepts any config | Dropped as framework: the step-6 lock commits protocol, hashes and seeds in git; the registry status flip is that commit. `run()` refuses unregistered steps/batch/seed via `_load_prepared` on the real path. Screen-before-final ordering is the operator's registered procedure, not a code gate. |
| P1-2 prepared inputs not bound by one manifest; no old-vocab parity on the real path | Deferred to step 5: the prepared-data builder is deliberately unwritten; its output will carry the bank/teacher/tokenizer hashes and run `verify_old_vocab_parity` before extension. Recorded in `execution_entry_missing`. |
| P1-10 document→domain map is a caller assertion | Deferred to step 5 with P1-2: the data builder emits the (source, doc id) → domain join from the document pass. |
| P2-1 non-alias arms skip the alias supply check | Fixed: alias requirement for every arm. |
| P2-5 alias-test families merge on bare doc id; domains at threshold 4 | Fixed: source-qualified keys; pinned threshold passed and recorded. Alias test regenerated. |
| P2-6 abbreviation "replacement" not atomic | Fixed: the expansion is admitted at the abbreviation's rank if eligible. |
| P2-7 panel/alias manifests carry no script or registry hash | Fixed: both added. |
| P2-8 uniform draws stop after a 65-batch guard | Fixed: sampling without replacement from remaining ids. |
| P2-2 cache identity does not require source/alias manifests | Dropped: manifests are always hashed when computed (Astra P1-7 fix); requiring separate manifest files is step-5 builder work. |
| P2-3 entropy block vs B2 nesting | Already dispositioned (Astra P3-29): shared inner fields only, by decision. |
| P2-4 loader validation and the fresh-process benchmark harness | Deferred: the benchmark is the registered loader phase after the final runs, not step 4. |
| P3-1 rehearsal resume is post-schedule; parity check tautological | Accepted as limitation (Astra P3-27); the step-5 smoke interrupts an active run. |
| P3-2 cosmetic assertions | Fixed: `or True` removed. |
| P3-3 tokenizer gate truncation claim | Fixed: serialized truncation length and CLS/SEP ids checked. |
| P3-4 classifier method string says query+document | Fixed: document-only method string for the training map; bias described as a heuristic topical preference. |
| Drift: registry fraction fields, bucket views, panel/alias statuses, STATUS pair count, CODEMAP claims | Fixed in the registry, STATUS and CODEMAP. |

**Owner:** M17 implementing session. **Exit:** one P1-only re-check by Codex Sol of the fixed
files, then step 5 timings. Deferred items are step-5 entry conditions, listed in the registry.

## Codex Sol P1-only re-check — 2026-09-11 (the single re-check)

Brief: [m17-sol-recheck-brief](../research/m17-sol-recheck-brief-2026-09-11.md). Log:
`research/m17-sol-recheck-2026-09-11.log` (gitignored). Read-only on commit `cc88fc1`; access
log: `REVIEW.md`, `registry.json` and the 22 `m17src` modules, nothing under `work/` or
`results/`. Of the 28 P1s across both reviews, 25 were confirmed fixed with the refusing line and
its test named; the dropped and deferred P1s were judged to have no plausible careless-researcher
trigger before step 5. Three remained, fixed directly in the same session (`m17src` diff of a
dozen lines plus three tests; 223 tests pass; rehearsal re-run to `results/m17_rehearsal_step4.json`):

| Remaining P1 | Fix |
|---|---|
| Control arms recorded an empty vocabulary identity, so the new completeness rule refused the matched control's export | Controls record `base-vocab:<tokenizer sha>`; test on `snapshot_identity`. |
| `train.run`, `export.build_bundle` and the rehearsal root created directories before the write guard | `admit_write` on the output root first, in all three; test with a `results/frozen_eval` root. |
| The rehearsal skipped the teacher/revision check | Checked in every mode; the synthetic warm start carries the registered teacher; mismatch test. |

**Exit:** step 4 closed. Review loop capped here (two reviews plus one re-check). Step 5 next.

## Codex Astra step-5 review (2026-09-11)

Brief: [m17-step5-review-brief](../research/m17-step5-review-brief-2026-09-11.md). Log:
`research/m17-astra-step5-review-2026-09-11.log` (gitignored; the report is its last section).
Read-only, on the step-5 builder, the cache, the driver and the two timed prepared directories.
Astra's sandbox again could not run pytest, so its coverage notes describe the reviewed tests,
not executions. 10 P1, 8 P2, 3 P3, plus four owner decisions. Owner ruling A4 (LEDGER.md)
settled the four before this batch; the fixes below implement it. All 275 `m17src` tests pass and
both timed sizes were rebuilt with the fixed builder.

| Finding | Disposition and exit |
|---|---|
| P1-1 `--protected-screen` reaches `protected10.build()` without the status gate | Fixed: `require_executable(reg, rehearsal=False, ...)` runs before the import, including under `--stages protected`; test asserts `protected10` is never imported while the registry is a draft. |
| P1-2 the screen never reads Kubernetes document text | Fixed: the full text of every new-source document is screened, `protected_receipt.json` records the admitted path + sha + group, and the bank, coverage views and alias views are filtered through that receipt; pre-clock the filter is a no-op and the manifest states `unscreened_new_source_rows`. Test with a stub screen: an innocent view whose document matched is dropped. |
| P1-3 training accepts a deferred screen | Fixed: `_load_prepared` refuses `state != "complete"` on a real run; `--rehearsal --data` proceeds with one loud line naming the unscreened pool and bank row counts (ruling A4). Test both branches. |
| P1-4 the screened pool is neither persisted nor reindexed | Fixed: `_apply_screen` rewrites `pool.jsonl`, rebuilds `heldout_idx`, `source_doc` and `positives`, drops BOTH views of an affected pair, and `_rehydrate("protected")` re-reads the screened pool. Drop-and-resume test on a fixture. |
| P1-5 the exclusion contract is only partly applied | Fixed: one `Exclusions` predicate applied by role (general: family, text sha, alias term, positive document group; coverage: source document group, view text sha/term; alias: family, both view shas, both terms, evidence group). `exclusions_applied` now reports realized drops per category; the inventory sizes moved to `exclusion_inventory`. Fixture with one excluded item per category. On the real pool it realizes 98 general, 115 coverage and 2 text-sha drops where the old code reported inventory only. |
| P1-6 the divergence split does not isolate families across buckets | Fixed: `heldout_document_groups` is computed once, after the held-out families are drawn, and any coverage view or alias pair whose supporting document group backs a held-out query is dropped and counted. Test on the helper. |
| P1-7 both document samplers are biased toward file prefixes | Fixed: `_lowest_hash_pick` — lowest `sha256(seed‖key)` over the whole eligible population, memory bounded by the quota. Test: the last decile of a 10,000-key population is drawn as often as the first. |
| P1-8 four passes are not enforced against the realized coverage population | Fixed: enough documents are drawn for one view each, the bucket is filled from realized deduplicated views and trimmed to target, and `_validate_buckets` records every realized population against the dose, the four-pass minimum and the query cap, refusing a bucket that cannot supply one batch. The plan test now asserts the minimum direction. |
| P1-9 v1 mining used the fp16 rows | Fixed: `_load_v1_table` loads `variant="int8"` and refuses anything else; the cache identity carries `variant` and the folded-rows sha. Test that the two variants differ. |
| P1-10 supporting-document identities inflate or collapse support | Fixed: one canonical `"<source>:doc-group:<gid>"` key per document in every role and under the budget; documentless sources map to `None` (ruling A4, `data.documentless_sources_vote`). `vocab.discover` counts no vote for `None` and still refuses two domains for one real document. Test: 18 real documents plus nqopen and triviaqa do not reach 20. |
| P2-11 stage reuse is not bound to its inputs | Fixed: each marker carries `_identity` (seed, size, bank cap, protected flag, exclusion/alias/k8s/kept/support-manifest/family shas); a mismatch refuses unless `--force`, and a forced stage invalidates every later stage in the fixed `STAGES` order. Test: a changed seed refuses. Observed on the real directories before the rebuild. |
| P2-12 prepared hashes recorded but never verified | Fixed: `_verify_prepared_hashes` checks `student_ids`, tokenizer, `teacher_q`, `bank` and `new_rows` against `prepared.json` and requires candidate rows = teacher rows = student id lists, before the model is built. Test: a corrupted `teacher_q.npy` refuses. |
| P2-13 pool document ids are not authenticated | Fixed: `PoolReader.rows_for` verifies the pool meta's `id_sha256` (legacy `sha_stream_list`) and the span length. Test with a reordered and a truncated id file. |
| P2-14 vector caches have no preprocessing identity | Fixed: both caches carry a `manifest.json`; a mismatch refuses. The existing caches adopt the current manifest once and the stage record says `cache_manifest_adopted`. Test. |
| P2-15 the teacher manifest misstates precision | Fixed: `encode_dtype: float32` and `storage_dtype: float16` recorded separately and bound into the cache identity. The precision itself is unchanged. Test. |
| P2-16 fresh and resumed v1 stages differ numerically | Fixed: the stage rounds to fp16 once, before any use, and hashes exactly the stored representation. Test. |
| P2-17 the recipe identity cannot enforce the artifact-of-record rule | Fixed: `cache.save` stamps `artifact_sha256` over the array hashes plus the teacher/bank/v1 input digests; `train` records both and binds resume to the artifact digest. Test: two caches with one recipe, different digests. |
| P2-18 missing document joins silently erase labels | Fixed: `select_labeled_subset` refuses, by qid, a labeled query whose positives are all missing from the join. Test. Both timed sizes build without tripping it. |
| P3-19 repeated set copies in the labeled selection | Fixed: `new = set(pos) - positives` against the budget, and the stage is timed (`labeled_subset_select`). |
| P3-20 the timing forecast omits work and misclassifies scaling | Fixed in part: `student_tokenize` extrapolates at its own work factor (two encodings per query); label selection, the pre-lock diagnostics and the manifests are timed; each size records whether its wall clock is a full build or a partial rebuild. `domain_store_pass` stays a carried fixed cost, with an explicit `document_stage_growth_upper_bound` beside the estimate rather than a per-query re-classification — the assumption is stated, not hidden. |
| P3-21 more than twenty flagged alias pairs cannot be identified | Fixed: the complete flagged list is written to `alias_flagged_pairs.json` and hashed in the diagnostic; the summary keeps the sample. |

**Owner decisions raised (not decided here).** Astra's report lists four; Dylan ruled on all
four in LEDGER.md A4 on 2026-09-11 and the registry records them as
`accepted_plan_revision_a4`. They are reproduced verbatim as raised:

- The 0.5 positive-budget share needs ratification before lock. It determines which examples
  receive positive injection and the bank's composition, and eligibility was computed over all
  `general` records before their final buckets were assigned, so the divergence holdout took
  part. Register how holdout positives interact with the budget.
- Reserving every Kubernetes document needs an explicit bank-allocation rule: those 1,648 rows
  are reserved before proportional sampling, which the registry's general description of uniform
  stratified sampling does not specify.
- The documentless-source convention needs correction, not merely documentation: preventing
  nqopen/triviaqa alone from reaching twenty is conservative, but adding two fictional document
  votes to eighteen real documents is not.
- Pre-clock training on deferred Kubernetes inputs requires resolution against the ledger; the
  real-data rehearsal switch does not itself authorize the admission exception.

**Smoke and resume after the fixes** (pre-clock, `--rehearsal --data`, both on the rebuilt
`s10000` directory; run records under `work/m17/runs/`). VL-A on `ext`, 400 steps at batch 256:
the driver printed the unscreened-row warning (11 pool rows, 1,648 bank documents), the
warm-start sha `0f544275cbce` matching `m7/FREEZE.json`, and ran at 21-22 ms/step. Interrupted at
step 18 with `--checkpoint-minutes 0.0`, the relaunch reported `resumed at step 18 from
recovery.pt` and finished at step 400 with the step-1 history entry carried across the restart —
and it resumed under `--checkpoint-minutes 0.5`, confirming the interval is not a resume-bound
field. C on `base`, 100 steps, ran clean. `grad_shares` is present at step 1 and at the last read
in both runs; the anchor observation is in `m17/FINDINGS.md`.

**Owner:** M17 implementing session. **Exit:** step 5 timings regenerated from the fixed
builder, the smoke/resume run recorded, then the step-5 close-out. This is one review of the
step-5 builder; it is not the second independent review that expensive execution requires.

## Codex Sol step-5 review (2026-09-11)

The SECOND independent review of the step-5 builder, after the Astra fixes. Brief:
[m17-sol-step5-brief](../research/m17-sol-step5-brief-2026-09-11.md) (addendum to Astra's).
Log: `research/m17-sol-step5-review-2026-09-11.log` (gitignored; the report is its last
section). Read-only, on `m17src` plus the two timed prepared directories; its sandbox again
could not run pytest, so its coverage notes describe the reviewed tests, not executions. 4 P1,
4 P2, 3 P3, plus one owner decision. Of Astra's 21 findings it confirms P1-1, P1-4, P1-6, P1-7,
P1-9 and P1-10 closed and leaves P1-2/3/5/8 partly open through the fail-open and after-screen
paths below. All 290 `m17src` tests pass; all three sizes were rebuilt with the fixed builder.

| Finding | Disposition and exit |
|---|---|
| P1-1 required step-2c exclusions fail open when the artifact is absent | Fixed: `prepare_data.require_exclusions` refuses a missing `alias_test_families.json` and verifies its sha256 against `results/m17_alias_test_manifest.json:files.exclusions_sha256`. `support_manifest.load_exclusions` keeps its permissive empty-set behaviour, which step 2b needs (it runs before step 2c exists) and the builder no longer uses. Test: missing file, wrong bytes, and the matching case. |
| P1-2 the builder neither fills the registered query cap nor refuses a four-pass shortfall after the screen | Fixed under ruling **A5** (`data.cap_fill_priority`): each bucket takes its four-pass minimum, general takes every distinct admitted query the cap allows, then the remaining slots go to `alias_all_distinct` (all 15,393 pairs) and `coverage_fill`. The realized full pool is now 600,000 rows, not 574,229. The coverage views the trim did not take are kept as a RESERVE, screened with the pool, and refill the bucket to target after the protected screen; a bucket that cannot be refilled records its shortfall, `support_manifest.dose_rule` is re-applied to state the resulting passes, and a bucket below one batch still refuses. Tests: exhausted-general cap fill, unknown priority step, after-screen refill, and the below-one-batch refusal. |
| P1-3 a fresh run can train on arrays from two builds | Fixed: `cache.load` recomputes `artifact_sha256` from the arrays on disk (one `artifact_digest` shared with `save`) and refuses a sidecar that does not describe them; every hash recorded in `prepared.json` is now REQUIRED (`if exp` removed); `_check_cache_belongs_here` compares the sidecar's identity and artifact digests with the manifest and its `artifact_inputs` with this directory's `teacher_q.npy`/`bank.npy`/`bank_ids.json` and the manifest's v1 digest. Test: `candidates.npz` + `cache.json` copied from build B into directory A load cleanly and are refused. |
| P1-4 a locked candidate recipe can silently reuse and train an older recipe | Fixed: `train._check_locked_recipe` (real runs) compares the loaded cache's mixes, K, RNG rule and recipe version, teacher revision and query preprocessing with the locked registry/FREEZE; `prepare_data.recipe_identity` puts a digest of those inputs (plus temperature, dose, cap, cap-fill priority, bank cap and the FREEZE fingerprints) into every stage marker's `_identity`, so a changed recipe rebuilds instead of resuming. Tests both halves; verified against the rebuilt `s2000/ext`. |
| P2-5 `--force` dependency invalidation is only in memory | Fixed: a rebuilt stage renames every dependent marker to `*.stale` on disk, and `--force` applies to the REQUESTED stages only, so an unrequested valid ancestor is rehydrated instead of being treated as non-reusable. `dependent_markers_invalidated_on_disk` is in the build record. Test on a marker fixture. |
| P2-6 the preprocessing binding trusts pre-existing vectors without evidence | Fixed: a manifest is stamped only on an EMPTY cache directory; a populated cache without one refuses. The two existing caches were stamped BY HAND on 2026-09-11 (`stamped_by_hand` in each `manifest.json`, pitfall in `m17/CODEMAP.md`): both were created and filled by this same builder from commit `5fe24b9` under exactly the preprocessing they record, so the stamp asserts nothing the git history does not already show. The test that expected adoption now expects the refusal. |
| P2-7 the 12.4 GiB peak and full-pool memory safety | Measured first, as asked. The 12.4 GiB was a **VmHWM**: on the rebuilt s10000 the peak occurs in `bank_pool_lookup` with `RssFile` 8.42 GiB (the frozen document memmap, reclaimable) while `RssAnon` never exceeds 3.07 GiB. Every stage now records `RssAnon`/`RssFile`/`VmHWM`. Cheap fixes applied: `PoolReader.rows_for` streams and hashes the id list instead of materializing 5.23M Python strings, and both population-sized `seen_groups` sets use 64-bit int keys — `bank_sample` fell from 4.54-4.64 GiB to 2.21 GiB anonymous. Three sizes were then built (2k/10k/50k) and the anonymous slope per query is fitted and projected at the full pool in `results/m17_prepare_timing.json:extrapolation.anonymous_memory`, beside the 8-9 GiB of other residents on the 25 GiB box. |
| P2-8 a missing alias evidence-group join silently fabricates its domain | Fixed: `_require_alias_groups` refuses by pair id any alias pair whose evidence group does not resolve in its source's group table; query-text-only sources (ruling A4) and the new source, whose group id IS its document key, are excepted. Test. |
| P3-9 `grad_shares` divides by the sum of component norms | Fixed: the denominator is `‖∇rows Σ terms‖`; the per-term proportions are kept separately as `component_norm_fraction` and shares need not sum to one. Test: aligned (0.5/0.5), orthogonal (shares sum to √2) and exactly cancelling (total norm 0, shares `None`, not 50/50). |
| P3-10 `--checkpoint-minutes nan` bypasses the maximum | Fixed: `math.isfinite` before the bounds check. Test for `nan` and `inf`. |
| P3-11 the forecast omits query-scaled serialization; the document "upper bound" is not one | Fixed: `cache_save` and the pool/spec/teacher/bank writes are inside named timers and enter the extrapolation; the projection is renamed `document_stage_sensitivity_estimate` and its note says the clamped, noisy per-document slopes do not bound a slower full-pool rate. |

**Owner decision raised (not decided here).** Sol raised one, reproduced verbatim as raised:

- "The original four Astra decisions are ratified by A4. One new protocol choice remains: after
  guaranteeing each bucket's four-pass minimum, the registry does not specify whether the
  remaining 25,771 slots should go first to coverage, the remaining 393 alias pairs, or another
  deterministic mixture. Because that changes pool composition, the owner should register the
  allocation priority before the cap-filling fix."

Dylan registered it as **A5** on 2026-09-11 while this batch was in flight:
`data.cap_fill_priority = ["alias_all_distinct", "coverage_fill"]` with
`accepted_plan_revision_a5`. The builder reads `data.cap_fill_priority` directly and refuses a
priority step it does not implement; the recommended default it was written against is the
registered order.

**Rebuilds, timing and memory after the fixes.** All three sizes were built with the fixed
builder (`--force` for 2,000 and 10,000, whose stage identities changed; 50,000 fresh):
129.9 s, 142.3 s and 518.5 s, all full builds. `results/m17_prepare_timing.json` now fits the
smallest against the largest and reports 82.7 s fixed plus 0.009078 s per query, **5,529.7 s
(1.54 h)** for the 600,000-query pool, 1.57 h with the document-stage sensitivity estimate. It
is higher than the previous 0.94 h for four measured reasons: the pool is now 600,000 rows
rather than 574,229 (A5), the 50,000 build encoded 39,477 queries COLD so a cold teacher encode
is inside the fitted slope, `cache_save` and the pool/spec/teacher/bank writes are now timed,
and a stage whose row count did not grow by 25% between the two fitted sizes is carried as a
fixed cost instead of being fitted (a 47.8 s cold `bank_pool_lookup` against a 10.4 s warm one
otherwise projected a 9.7-million-second intercept — the latent hazard the third size exposed).

Memory, measured rather than argued: peak `RssAnon` / `RssFile` / `VmHWM` are 3.03/8.43/12.18,
3.07/8.42/12.21 and 3.44/7.44/11.41 GiB at 2k/10k/50k. The 12.4 GiB of the earlier record is a
VmHWM reached in `bank_pool_lookup`, where `RssFile` alone is 8.4 GiB of frozen-document memmap
that the OS reclaims; anonymous memory never exceeded 3.44 GiB. The fitted anonymous slope is
8.5e-6 GiB per query on a 3.01 GiB base, projecting **8.11 GiB anonymous at the full pool**;
with the 8-9 GiB of other residents on the 25 GiB box that is about 17 GiB, below the ~20 GiB
line, so nothing about the bank or the pool is being changed on memory grounds.

**Smoke and resume after the fixes** (pre-clock, `--rehearsal --data`, on the rebuilt `s10000`
`ext` directory; `work/m17/runs/smoke-resume-VL-A-2`). VL-A, 100 steps at batch 256: the driver
printed the unscreened-row warning (11 pool rows, 1,648 bank documents) and passed the new
cache-belongs-here check. Launched with `--checkpoint-minutes 0.0`, killed once `recovery.pt`
existed, and relaunched with the registered smoke interval `--checkpoint-minutes 0.2`: the
relaunch printed `resumed at step 1 from recovery.pt`, carried the step-1 history entry across
the restart and finished at step 100 at 24 ms/step — under a CHANGED interval, which is the
evidence that the interval is not a resume-bound field. Corrected `grad_shares` (denominator
`‖∇rows Σ terms‖`): listwise 0.887 at step 1 and 0.883 at step 100, teacher cosine 0.251 to
0.268, alias consistency 0.0032 to 0.0024, anchor exactly 0, against a total row-gradient norm
of 2.99e-2 falling to 2.66e-2. The shares sum slightly above one because the cosine and listwise
gradients partly oppose, which the old denominator hid; `m17/FINDINGS.md` carries the corrected
numbers.

**Owner:** M17 implementing session. **Exit:** done — the three sizes are rebuilt, the timing
result regenerated and the resume smoke re-run. Step-5 close-out next.

## Codex Sol step-5 P1-only re-check (2026-09-11, the single re-check)

Brief: [m17-sol-step5-recheck-brief](../research/m17-sol-step5-recheck-brief-2026-09-11.md).
Log: `research/m17-sol-step5-recheck-2026-09-11.log`. Read-only; its sandbox could not run
pytest, so its coverage notes describe the tests, not executions. Of the 14 step-5 P1s (Astra
P1-1…P1-10, Sol P1-1…P1-4), 12 confirmed fixed with the refusing line and the covering test
named; two remained and were fixed directly in the same session:

| Remaining P1 | Disposition and exit |
|---|---|
| Sol P1-2 the after-screen refill restored only the coverage target, so a screened-out general row or alias pair left the pool below the A5 planned total | Fixed: `_apply_screen` refills coverage from the screened reserve until the pre-screen row count (= the planned total in a real build) is restored; the record carries `target_total`, `rows_after_screen` and the rule. Test: an alias-view hit plus a coverage row → three coverage rows return. |
| Sol P1-4 the locked-recipe check compared only mixes, K, RNG, model/revision and three preprocessing fields, so an older cache differing in `score_ties`, backfill, tokenizer or `post_dense` passed | Fixed: the cache identity now carries the whole registered `candidate_construction` block and `train._check_locked_recipe` compares it and the COMPLETE preprocessing object (`prepare_data.teacher_preprocessing`'s fields, incl. `post_dense`, tokenizer and both precisions) with the registry/FREEZE; unknown extra fields refuse. Tests: `score_ties`, `post_dense`, an extra field. |

The re-check also confirmed the A5 cap-fill order (general, all distinct alias pairs, coverage
to exactly the cap, deterministic). The review loop is closed here: two reviews plus one
re-check. **Exit:** step 5 closed; step 6 (full-pool build and lock) next.

## Codex Astra step-6 lock review (2026-09-11)

Brief: [m17-astra-lock-brief](../research/m17-astra-lock-brief-2026-09-11.md). Log:
`research/m17-astra-lock-review-2026-09-11.log` (gitignored). Read-only on commit `7f96762`;
access log: `m17src` modules and tests, `m7src/teacher.py`, `m7src/table.py`, the five `m17/`
documents, `git show --stat`; nothing under `work/` or `results/`. Eleven findings on
`m17src/lock.py` (new), the training gate and the chunked teacher-cache flush. Triage rule
(owner, 2026-09-11): smallest fix per real bug; findings whose only trigger is an adversary
editing files behind the manifests are dropped.

| Finding | Disposition and exit |
|---|---|
| P1-1 executed hashes copied from the manifest, never checked against the bytes V0 folds | Fixed for V0: `export_v0` hashes the tokenizer, `new_rows` (`sha_array`, the builder's convention) and warm start and refuses a mismatch; training already verifies the same bytes in `_load_prepared`. Ordered term→row-id verification beyond the size check: dropped (adversarial substitution of a same-size tokenizer). |
| P1-2 training never checks the executed lock; `EXECUTABLE` admitted real training | Fixed: `require_executable(training=True)` needs `LOCKED_EXECUTABLE` for a real arm (`EXECUTABLE` admits preparation only); `train._check_executed_lock` compares the loaded `prepared.json` hashes with `lock.executed.{ext,base}_hashes` for the form being trained. Tests in `test_lock.py`. |
| P1-3 a receipt's existence does not prove THIS build was screened | Partly fixed: the receipt bytes must hash to the manifest's recorded `sha256` (production shape `{path, sha256}`). Binding the receipt to the pool identity: dropped (needs a forged manifest). |
| P1-4 base form bound only by its tokenizer; held-out slice unbound | Fixed for the base form: `lock.executed.base_hashes` records every `EXECUTED_HASHES` field for the control arms and training compares per form. Held-out subset substitution: dropped (adversarial; the slice sits in the cache identity's inputs). |
| P1-5 a complete `--size` build or another seed would lock | Fixed: `_check_full_build` refuses `size != None` and a seed other than the locked prepared seed in both halves. Tests. `kept`/family hashes: covered transitively by `query_pool`/`doc_domain_join` and the recipe/exclusion hashes; not added. |
| P1-6 tie band, learning rates, vocabulary caps outside the bound block; fixed files not rechecked at execution | Fixed: `protocol_identity` binds the whole registry minus `status`/`lock`/`allocation_hours`; the executed half rehashes the fixed manifests and refuses a change. Tests (tie band → 0; edited support manifest). |
| P1-7 export identity lacks the cache ARTIFACT digest, so snapshots from two runs sharing a recipe could be averaged | Fixed: `candidate_cache_artifact_sha256` joins `export.IDENTITY_FIELDS` and every snapshot's metadata; the rehearsal's in-memory cache records a stated `rehearsal-no-artifact:` value instead of a blank (a real run refuses a blank digest already). |
| P1-8 the clock definition (first full-pool `--protected-screen`) exempts charged work and permits resets | Removed from the code: `lock.clock` now says the start rule is an OWNER ruling recorded in LEDGER.md before the first `--protected-screen` invocation; raised to Dylan (STATUS). |
| P2-9 the builder's receipt is a dict, `Path(dict)` crashed the executed half | Fixed (with P1-3); `_fake_build` now uses the production shape. |
| P2-10 the flush is not atomic across `vecs`/`index` | Fixed: the constructor drops an uncommitted vector suffix (rows beyond the index) and re-encodes those texts as misses; test. Chunking changing the teacher's length-sorted batches: accepted — the encode is float32 and normalized per row, and identity binds to the realized bytes, never to a byte-exact re-encode. |
| P2-11 executed `--dry-run` exported V0 | Fixed: `--dry-run` refuses for the executed half. |

Astra's verdict on the two-half sequencing: "defensible" — the registry's lock text and
PLANNING's observation boundary allow executed identities on the clock before any read; neither
the pre half's reads nor the V0 gates are a development or panel read; replacing the placeholder
hours is expressly permitted; ×2 is a chosen ceiling, not a demonstrated bound, and exceeding it
triggers the registered stop/recovery policy. STATUS step 6's older wording ("lock, then clock")
is superseded by this section. **Owner:** M17 implementing session. **Exit:** fixes landed;
Codex Sol reviews the fixed code next (alternating rule), then at most one P1 re-check.

## Codex Sol step-6 lock review (2026-09-11, second review)

Brief: [m17-sol-lock-brief](../research/m17-sol-lock-brief-2026-09-11.md). Log:
`research/m17-sol-lock-review-2026-09-11.log` (gitignored). Read-only on commit `2c157be`;
access log: twelve `m17src` modules/tests read directly, the rest scanned by one permitted
non-recursive grep, `REVIEW.md`, `registry.json`; nothing under `work/` or `results/`. Every
Astra fix confirmed with the refusing line and the former trigger named; **no remaining P1, no
new P1/P2**. Sol also checked the real builder's manifest shape field by field (`size`
integer-or-null, `seed`, receipt `{path: sm.rel, sha256}`, `build_record.json` types, base
manifest without `new_rows` compared as `None` on both sides), that the whole-registry protocol
hash cannot invalidate its own pre half, that V0's hash conventions match the builder's and its
table equals step 0 of `train.extend_model`, and that the suffix recovery cannot drop committed
rows. Coverage gaps noted and accepted (owner's no-over-engineering rule): `export_v0` is
exercised by the manual smoke on `work/m17/prepared/s2000` (five gates pass), not by a test;
no test averages two snapshots with differing cache-artifact digests; the executed
`--dry-run` refusal has no test. **Exit:** review loop closed (Astra → fixes → Sol, nothing
remaining, no re-check needed). Step 6 proceeds to the pre half once the full build's
`build_record.json` says `full build`.

## Codex Astra — step 6b import fix (2026-09-11)

Brief: [m17-astra-6b-fix-brief](../research/m17-astra-6b-fix-brief-2026-09-11.md). Report:
`research/m17-astra-6b-fix-review-2026-09-11.md`. Read-only, on the `d8f2b43` fix for the
`parity` crash. No P1; two P2s, both fixed in `bfa7644`.

| Finding | Disposition and exit |
|---|---|
| P2-1 a legacy `train` already cached in `sys.modules` survives a path-only fix | Fixed: `common.reassert_path_order()` evicts a cached m7src `train`. |
| P2-2 the regression test covered the helper, not the call site | Fixed: a call-site test with a stub `protected10` that fails when the production call is removed (mutation-checked). |

Astra could not answer its Q1–4 on the first pass (the read list was too narrow). A re-check with a
wider list followed: [m17-astra-6b-recheck-brief](../research/m17-astra-6b-recheck-brief-2026-09-11.md),
report `research/m17-astra-6b-recheck-review-2026-09-11.md`.

**Re-check outcome (2026-09-12).** Q1: the only shared basenames are `train` (m7src/m17src) and
`run_arm` (m7src/m10src); none spans all three trees. Q2: no shared-name import happens before
`parity`, so the six reused stage markers are sound. Q3: no live comparison holds the old
`prepare_data.py` hash — the lock compares against the amended value and the whole `lock` block is
excluded from the registry identity — so the dated amendment is sufficient. Q4: nothing downstream
reads or refuses the `partial rebuild` build record. **Verdict: proceed to the executed half**, with
no fix required first.

| Finding | Disposition and exit |
|---|---|
| P2-3 (new) on a protected-index cache miss `protected10.build()` lazily imports `m10src/cov_screen.py`, which puts m7src first again after our repair | **DEFERRED, not fixed now.** The fix changes `prepare_data.py`, whose sha the lock binds, and the finished build is what trains. Exit: required before any FUTURE screened rebuild — reassert the path order after `build()` as well (and cover path mutation during `build()` in the test), as a dated amendment. |

## Codex Astra — dev-suite reader review (2026-09-12)

Brief: [m17-astra-devreader-brief](../research/m17-astra-devreader-brief-2026-09-12.md). Log:
`work/m17/logs/astra_devreader_review.log` (gitignored). Read-only on the newly wired dev-suite
reader; six P1s (all blocking the one irreversible V0 read) and three P2s. All fixed in
`m17src/evaluate.py` and `m17src/test_evaluate.py`; no other module changed.

| Finding | Disposition and exit |
|---|---|
| P1-1 the gate accepts an incomplete or altered lock (`{"status": "EXECUTABLE"}` passed) | Fixed: `_require_dev_suite` requires `LOCKED_EXECUTABLE` **and** `lock.executed.v0_export`'s three digests (`_executed_v0_export`). Protocol-hash matching beyond `require_executable`: not added (the executed half already refuses a moved protocol). `m17src/evaluate.py` |
| P1-2 the locked V0 artifact is never verified before the encoder is built | Fixed: `_verify_v0_bundle` recomputes model/tokenizer/config digests with the exporter's own conventions (`export.gate_artifact`, not a reimplementation) and refuses unless all three equal `lock.executed.v0_export`. `m17src/evaluate.py` |
| P1-3 document-vector identity unbound (held-out checked by row count; text vectors accepted from `encode_cached`) | Fixed: `_pool_identity` reproduces `m7src/heldout._verify_pool`'s pinned-pool verification against `_pinned.pool` — meta fields, byte size and the 12.6 GiB `vecs.f16` SHA-256, computed once per read — **without** `pool.build()` (which rebuilds the artifact); `_teacher_doc_vecs` hashes the encode-cache file actually memmapped and compares it to the cache's own recorded digest (`teacher.PROVENANCE`/`shards.json`), refuses a cache missing shards rather than encoding, and records both digests in the receipt. `m17src/evaluate.py` |
| P1-4 text-component queries are not bound (permuted `q_texts` passed every check) | Fixed: `_query_identity` hashes the ordered `(qid, text)` pairs per component and verifies the ordered qid/qtext digests wherever the manifest pins them. The two held-out components are therefore VERIFIED; the four text-backed components pin no ordered query identity, so their pair digest is RECORDED in the receipt and the result and labelled as such — the manifest was not edited to add one (owner ruling required to pin it). `m17src/evaluate.py` |
| P1-5 validation happened after scoring had begun | Fixed: `dev_suite_read` preflights the output destination, the bundle digests, all components (identities, queries and both vector sources) before the first score; a failure after scoring starts still writes a `state: failed` receipt naming the components already completed. `m17src/evaluate.py` |
| P1-6 neither provenance nor the single-read boundary survives | Fixed: `out` is mandatory, the read refuses if the result or `*.receipt.json` exists (no `--force`), and the receipt is written `started` before scoring and updated to `complete` — registry status, bundle digests, per-component manifest hashes, query and vector digests, git sha, UTC timestamps, `reads: 1`. The result JSON carries the same provenance; `--surface dev-suite` now requires `--out`. `m17src/evaluate.py` |
| P2-7 production arguments could redefine the registered surface | Fixed: `_enforce_production_surface` requires the complete pinned list, int8/resident loading, the registered `serving.prefetch` depth and the registered manifest; `_require_pinned_fields` refuses a component whose pinned hash fields are missing instead of skipping their checks. Subset/fp16/eager/other-k are reachable only under `fixture=True`. `m17src/evaluate.py` |
| P2-8 Recall@10 was lost | Fixed: one `evalkit.topk_ids_scores` run per component feeds both `per_query_ndcg` and `recall_at_k`; per-component values and the equal-weight component macro of each are kept in the result and the receipt. `m17src/evaluate.py` |
| P2-9 tests never exercised successful reader scoring | Fixed: `test_dev_suite_read_scores_through_the_loader_and_the_m7_scorer` runs the reader end to end on the rehearsal's real exported bundle through `loader_np` and the M7 scorer (tie pair, a dropped self-hit), asserting both metrics, the receipt fields and the no-overwrite refusal; plus refusal tests for the gate, the bundle digests, the surface arguments and the missing pinned field. `m17src/test_evaluate.py` |

**Owner:** M17 executing session. **Exit:** fixes landed; `pytest m17src` = 314 passed with only
the four pre-existing registry-status failures. The V0 read remains unspent (`read: false`).

## Codex Sol — dev-suite reader fix review (2026-09-12)

Brief: [m17-sol-devreader-fix-brief](../research/m17-sol-devreader-fix-brief-2026-09-12.md). Log:
`work/m17/logs/sol_devreader_fix_review.log` (gitignored). Read-only, on the Astra fixes above.
Six P1s, three P2s, one P3 (retain). All dispositioned below; every code change is in
`m17src/evaluate.py` and `m17src/test_evaluate.py`.

| Finding | Disposition and exit |
|---|---|
| P1-1 a Python caller could pass `fixture=True` and an invented `reg` | Fixed: `dev_suite_read(bundle, out, allow_dev_suite=)` is production-only — it loads the on-disk registry itself and has no manifest/subset/loader/depth/fixture parameter (asserted by signature in the tests). The overridable form is the test-only `_dev_suite_read_fixture`; the CLI cannot reach it. `m17src/evaluate.py` |
| P1-2 the four text components' ordered (qid,text) pairs are recorded, not pinned | **RECORDED-ONLY, unchanged; owner decision required.** Fact-finding: `m7src/devsuite` builds `work/dev/<name>.json` straight from `load_dataset` and records only the five manifest hashes — no per-component build provenance and no ordered query-text digest anywhere on disk. The only other copy of those texts is the mutable, unpinned HuggingFace dataset cache (which also holds two reserved subsets), so re-deriving proves nothing a manifest pin would. Pinning needs an owner-approved, dated manifest amendment; the manifest and registry were not edited. |
| P1-3 `verify=False` let the teacher cache be adopted trust-on-first-use | Fixed as a REFUSAL: `_teacher_doc_vecs` now calls `teacher.encode_cached(..., verify=True)`, and the manifest's `_pinned.active_encoder` repo/revision/dim is enforced against the teacher and the loaded width. Fact-finding on the real caches (metadata only): all four stella doc caches record EVERY shard `trusted_on_first_use: true` (nq-250k 5/5, hotpotqa 105/105 plus a TOFU `combined.f16`, each cqadup 1/1), so **`verify=True` refuses all four** — the V0 read is blocked on this until the owner rules (re-encode the four caches, ~5.5M documents, or accept TOFU with a dated disclosure). Model/revision/pooling/prefix/max_length/corpus are bound by the cache key and `meta.json` = `stella_en_400M_v5` @ `ffeb2b7e…`, mean-l2 + `2_Dense_1024` (1024d). `m17src/evaluate.py` |
| P1-4 an arbitrary `out`, a non-atomic existence test and an ignored `read` flag | Fixed: production writes ONLY to `V0_READ_PATH` (or a `read_path` the registry names) and refuses any other destination; `lock.executed.v0_export.read` must still be `false`; the receipt is claimed with `os.open(O_CREAT\|O_EXCL)`, so two processes cannot both start the read. The flag is not flipped in code — the orchestrator records the read. `m17src/evaluate.py` |
| P1-5 a crash after `started` left an unrecoverable receipt | Fixed: each component's per-query nDCG/Recall is written into the receipt (tmp + `os.replace`) as it completes; a receipt in `started`/`failed` is CONTINUED for the remaining components only when every field in `IDENTITY_FIELDS` matches the fresh preflight, and otherwise refused by field name. A complete receipt is never continued; an empty claim left by a preflight refusal is released. No other recovery path. `m17src/evaluate.py` |
| P1-6 a dirty tree makes the recorded git sha a lie | Fixed: a production read refuses unless `git status --porcelain --untracked-files=no` is empty, and records the sha and the (empty) porcelain in the receipt and the result. Comparing an operative registry/protocol digest beyond this: not added — the executed lock half already binds it. `m17src/evaluate.py` |
| P1-7 “the checkout is not `4824f84`” | **DROPPED.** HEAD moved only because the review brief itself was committed; the allowed files are unchanged, and the receipt records the actual execution commit of a clean tree. |
| P2-8 pytrec_eval and a Python sort could resolve a rank-10 tie differently | Fixed: `_ndcg_and_recall` takes both metrics from ONE `pytrec_eval.RelevanceEvaluator` (`ndcg_cut.10`, `recall.10`) over the same run, with M7's qrels handling (`evalkit`). `m17src/evaluate.py` |
| P2-9 `_pool_identity` omitted M7's active-encoder check and hashed a constructed path | Fixed: `encoders.active()` is compared to `_pinned.pool.encoder`, and the resolved `pool.vecs.filename` must equal the file that was hashed. `m17src/evaluate.py` |
| P2-10 the end-to-end test monkeypatched `_dev_doc_vecs` | Fixed: `v0_world` now builds a TINY REAL teacher encode cache (redirected `teacher.ENC`) and the end-to-end test runs the production `_teacher_doc_vecs`; new fixture tests cover TOFU, a corrupted shard, a row-count mismatch, a pinned-dimension mismatch, another teacher, a missing shard, and `_pool_identity` (pass, wrong row count, another active encoder, same-size-different-content, truncated file). `m17src/test_evaluate.py` |
| P3-11 keep the 12.6 GiB sequential hash | Kept, as recommended. |

**Owner:** M17 executing session. **Exit:** fixes landed; `pytest m17src/test_evaluate.py` = 32
passed, `pytest -q m17src` = 320 passed, 4 failed (the pre-existing registry-status four). The V0
read stays unspent (`read: false`) and is now blocked on TWO owner decisions: the TOFU teacher
caches (P1-3) and whether the four text components' query texts are pinned (P1-2).

## Codex Astra — dev-suite reader re-check (2026-09-12)

Log: `work/m17/logs/astra_devreader_recheck.log` (gitignored). Read-only, over the Astra and Sol
fixes above. Three P1s, two P2s, plus a disposition table for the original nine. Both owner
rulings landed in `m17/LEDGER.md` the same day; this round is the last code before the V0 read.

| Finding | Disposition and exit |
|---|---|
| P1-1 a concurrent process could "resume" a live read and overwrite its checkpoints | **NOT FIXED — operational.** The V0 read is one orchestrated detached process (`work/m17/logs/run_v0_read.sh`); no second process is started, and the launcher is the only sanctioned entry. An advisory whole-run lock would be new machinery for a risk the operating procedure already excludes. |
| P1-2 an orphaned EMPTY claim blocks the read forever; a git-status refusal escapes cleanup | **Fixed.** `_claim_receipt` treats a zero-byte receipt as an abandoned claim: it removes it and re-claims once under the same `O_CREAT\|O_EXCL` semantics (`O_EXCL` cannot truncate-and-continue). The preflight `try` now extends through receipt construction — `_git_sha`, `_git_porcelain`, `_check_resumable` — to the FIRST `write_json` of the receipt, so every refusal in between releases the empty claim. Test: `test_an_orphaned_empty_claim_is_reclaimed_and_a_late_refusal_releases_it`. `m17src/evaluate.py`, `m17src/test_evaluate.py` |
| P1-3 the pool digest memo survives across attempts | **Fixed.** `_reset_pool_memo()` is called at the start of every `_dev_suite_read` attempt, so the 12.6 GiB digest is shared by the two held-out components of ONE attempt only. The end-to-end test poisons `_POOL_VERIFIED` and asserts the reader clears it (it fails without the production call); `test_pool_identity_refuses_a_changed_pool_…` now uses that production reset instead of a test-only `clear()`. `m17src/evaluate.py`, `m17src/test_evaluate.py` |
| P2-4 no operative protocol-hash matching before the claim | **NOT FIXED.** `m17/registry.json` is committed and the read refuses a dirty tree, so a post-lock protocol change is a visible commit recorded in the receipt's `git_sha`; protocol changes are owner-only and dated (CLAUDE.md). A second digest comparison would restate the lock, not add a check. |
| P2-5 receipt writes are atomic but not fsynced | **NOT FIXED.** Single-box run with resume: a machine crash loses at most the components since the last persisted checkpoint, which the continuation re-scores under the identity comparison. Durability against power loss buys nothing the resume path does not already provide. |
| Original nine (Astra dev-reader review) | Re-check dispositions: P1-1 **partial** (residual = P2-4 above, not fixed); P1-2 **closed**; P1-3 **partial** (TOFU refusal stands; now gated by the dated disclosure under Ruling 1, and the memo issue is P1-3 above); P1-4 **owner-ruled** — Ruling 2 keeps the four text components' query texts recorded-only, labelled `query_text_binding: "recorded_only (owner ruling 2026-09-12, LEDGER)"` in the receipt and the result, with the two held-out components `"verified"`; P1-5 **closed**; P1-6 **closed** by P1-2 above (P2-5 not fixed); P2-7, P2-8, P2-9 **closed**. |

**Ruling 1 implementation.** `m17/tofu_disclosure.json` names the four stella dev caches, their
per-shard digests and (where stitched) the `combined.f16` digest, copied from each cache's own
`shards.json`, plus the SHA-256 of the spot-check summary (tracked as
`results/m17_teacher_spotcheck.json`). `_teacher_doc_vecs` still calls
`encode_cached(verify=True)` FIRST; only the two trust-on-first-use refusals from
`m7src/teacher.py` are caught, and only for a cache whose disclosure entry still equals its
current `shards.json` — then the bytes actually scored are re-hashed and compared to the
disclosed digest. An undisclosed TOFU cache, a changed cache, or any other refusal still refuses.
`component_identities[*].document_vectors` records `tofu_disclosed`, the disclosure file's
SHA-256 and the ruling string. Test: `test_only_a_disclosed_tofu_cache_is_accepted`.

**Owner:** M17 executing session. **Exit:** `pytest m17src/test_evaluate.py` = 34 passed;
`pytest -q m17src` = 322 passed, 4 failed (the pre-existing registry-status four). The V0 read is
unspent (`read: false`) and unblocked; the launcher is `work/m17/logs/run_v0_read.sh`.
