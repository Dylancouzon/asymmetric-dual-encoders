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
