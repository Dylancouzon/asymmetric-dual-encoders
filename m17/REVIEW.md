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
