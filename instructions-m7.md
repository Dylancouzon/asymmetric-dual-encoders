# M7 — Train and release a Qdrant query encoder (self-directed mandate)

Read CLAUDE.md first: M0–M6 are complete; this file is the binding M7 mandate. You are a future Claude session running self-directed inside WSL2 on the Windows/RTX 3080 box. Dylan delegated all technical decisions; time and compute are unconstrained. The goal is the best benchmark number that survives adversarial review — the evaluation protocol below is what makes the number publishable, so it binds as hard as the goal. This plan went through three adversarial gates (2026-08-25); do not quietly relax anything below.

## Mission

Train a token→vector lookup-table query encoder against a frozen off-the-shelf document tower; release under the Qdrant org if it clears the release bar. Query time = tokenize → fetch rows → weighted average → normalize; no transformer. Drops into the M5 two-collection Edge prototype (0.9 ms/query).

Novelty verified 2026-08-25 (`research/m7-novelty.md`); re-run the freshness check before the report ships. Research and web searches run in Sonnet subagents, never inline — raw sources stay out of your context; you keep the conclusions.

The central open question is structural: LightRetriever's table works inside a doc space co-trained to be additively predictable from query tokens; a frozen bge-base space was never optimized for that. Stage 0 answers this before any large training spend.

Known headwind, pre-registered: the clean data stack has almost no in-domain relevance pairs for the six test sets (ESCI is e-commerce; MIRACL/Mr.TyDi and TriviaQA are Wikipedia; the six are scientific, biomedical, financial, argumentative). This is the primary risk to the release bar. Shaping synthetic queries to the six's query forms (claim verification, long counter-arguments, finance-style questions) is a query-form mitigation only — it does not supply in-domain documents, vocabulary, or relevance structure; document-side transfer remains unmitigated and the report says so. Add a dev diagnostic by query form. If Stage 0 passes and the domain gap caps the number, that is the report's stated finding.

## Decision authority

- You decide: architecture, objectives, data mix within the licensing rules, schedule, experiments (subject to the mandatory ablations), teacher swaps within constraints, when to stop.
- Dylan decides (ping with data, don't block on anything else): money (cloud GPUs, paid APIs — bring a cost estimate); the HF release (explicit go); any new synthetic seed corpus (FineWeb/C4 are not approved — `research/m7-data-licensing.md`).
- Hard constraints: no teacher or component from a vendor shipping a competing vector search product (vendor evidence in `research/m7-teacher-shortlist.md`); the training-data rules in `research/m7-data-licensing.md` (decided stack, exclusions, fingerprint decontamination); the evaluation protocol below.
- CC BY-SA, one rule for the whole family, **confirmed by Dylan 2026-08-25**: training on CC BY-SA text with model-card attribution, weights not treated as derivative works (details in m7-data-licensing.md). MIRACL/Mr.TyDi, Wikipedia synthetic seeds, and the NQ/SQuAD/HotpotQA/FEVER pair sets are all approved for training.

## Comparators and tiers

Frozen per-query nDCG@10 vectors for all preregistered tier and dev-reference comparators are committed in `results/perquery.json`; `results/eval_manifest.json` + `results/frozen_eval/` pin the dataset content (corpus hashes, vendored queries and qrels) the pairing assumes. All tier decisions pair against those vectors — never re-run a comparator system. `scripts/validate_perquery.py` must pass before any pairing is trusted; the provenance note in `results/FINAL_MATRIX.md` documents its two allowlisted cells. The M4 caches (`artifacts/`, 12 GB) do not travel; doc encodes are re-derived on the 3080.

Reference avg-6 rows: potion-retrieval-32M 0.3601 (best static) · BM25 0.4174 · LR dense single websearch table 0.4320 (like-for-like single-table row) · **LR dense per-task 0.4583 — LR's strongest dense config, oracle-flavored: the release comparator** · **OpenSearch doc-v3-gte 0.4868 — best zero-query-compute measured: the aim** · arctic-m-v1.5 0.5264 (matrix winner). Doc-index cost rides along: LR indexes at 1536d ≈ 3.07 GB/1M docs, a 768d table at 1.54 GB — a release-bar win at half the doc-index cost is a materially stronger claim; say so.

**Tier candidates are fixed:** Tiers 2 and 3 are judged on the released artifact — the single int8 dense table with its one query-preprocessing rule. Tier 1 is judged on the released zero-neural-query-compute system, which may be the table + BM25 fusion, labeled as such. Tiers are decided only by paired bootstrap CIs on unrounded per-query scores (procedure below); an unresolved comparison falls to the lower tier. (The ±0.007 band was an M2-era estimate; it appears in no M7 rule.)

- **Tier 1 — released system CI-resolved above opensearch-doc-v3-gte: the aim.** The "best zero-query-compute system" headline additionally requires the int8 dense table alone to CI-resolve above BM25, plus the fusion ablation (BM25 / table / fusion / incremental gain). A weak table rescued by BM25 does not earn the headline. Report the fusion next to the LR hybrid rows (0.4720 per-task, 0.4594 websearch) for context.
- **Tier 2 — int8 dense table CI-resolved above lr-dense-pertask (0.4583): the release bar**, deliberately LR's strongest dense config, labeled as such.
- Tier 3 — int8 dense table CI-resolved above BM25 only: publishable frontier point; release is Dylan's call.
- Tier 4 — everything else: negative result published next to the M6 projection failure; stop.

## Teacher

Default BAAI/bge-base-en-v1.5 — rationale, size math, alternatives, and vendor evidence in `research/m7-teacher-shortlist.md`. Pin the HF revision hash at kickoff; the release references it. Teacher selection and any swap use dev runs and official published numbers only; the teacher's six-set symmetric row is measured once, inside the final run, as the retention-ceiling row, with per-dataset retention in the report.

## Architecture

- Table trained fp32; released fp16 and int8. int8 = symmetric per-row absmax quantization, no calibration set (the M3 recipe, which was quality-free for LR).
- Init default: teacher-derived token representations (each vocab token forwarded through the frozen teacher, pooled its way); random and raw-input-embedding inits as controls.
- Vocab coverage: measure per-token update counts; low-update rows stay regularized toward init; deterministic unseen-token behavior; report coverage on train and dev.
- Learned per-token scalar weights: positive and bounded (softplus), IDF-initialized, length-normalized. A long-query hypothesis to ablate, not a fix.
- Prefix: two mandatory variants — no prefix, fixed runtime prefix tokens — with byte-for-byte conformance checks; double application forbidden. Prefix-conditioned rows are exploratory only.
- Conformance test file before the first training run: special tokens, padding, repeated tokens, truncation, max length, empty queries, near-zero-norm sums, prefix handling.

## Objectives

- **A — contrastive InfoNCE** against precomputed frozen doc vectors: dedup across sources; dataset-aware batching; false-negative filtering by teacher-score margin; compare BM25-mined, teacher-mined, and mixed negatives. Frozen doc vectors make very large negative pools nearly free — exploit that first.
- **B — distillation** to teacher query embeddings: normalized cosine plus a ranking-preservation term (KL over top-k similarities against frozen doc vectors); report embedding agreement and retrieval agreement separately on dev.
- **C — B-init, then A-finetune.**

Objective-by-dataset field table required before training: fields used, rights, positive construction, usable pair counts after filtering. TriviaQA contributes query text only — it feeds B, never A.

## Stage 0 — representation compatibility (mandatory, before large training)

1. **Distilled table** (objective B on held-out text): cosine/MSE to teacher query vectors, overlap@10 against the teacher's retrieved lists, retrieval nDCG vs the dev reference rows.
2. **Capacity probe — diagnostic only, categorically ineligible for any gate:** deliberately overfit a table on the dev queries with the contrastive objective, trained to loss plateau on a logged budget. Falsifiable bar: the overfit table must CI-resolve above the BM25 dev row. If even unlimited overfitting on dev cannot beat BM25 on dev, the frozen-tower tax is structural → negative-result path, earned. If it passes but the distilled table fails, the problem is objective or data — continue.

## Evaluation protocol (violating it voids the report)

- **Partition ledger before the first training run:** TRAIN / DEV / KNOWN-TEST (the six — labeled development-informed) / UNTOUCHED-FINAL (default: BEIR FEVER, Climate-FEVER, DBpedia-entity — Wikipedia-based, admissible because fingerprint filtering guards generic-Wikipedia overlap; label the caveat). Fingerprint decontamination (exact + near-duplicate) runs TRAIN↔DEV, TRAIN↔KNOWN-TEST, and TRAIN↔UNTOUCHED-FINAL, removal counts logged. Source-level license evidence (not wrapper tags) recorded per dev and untouched-final set at kickoff; drop and log any set that fails the affirmative-license standard; if the untouched partition empties, the report says so. Headline aggregates never mix partitions.
- **Dev suite, pinned here, hashes recorded in `eval_manifest.json` before any candidate result exists:** BEIR NQ subsampled (all qrels-positive docs + random distractors to 250K docs, seed 0, all test queries), HotpotQA, CQADupStack **programmers** and **physics** (the non-Wikipedia sets with real qrels), held-out training slices (pairs with sha256(pair_id) mod 50 == 0), including a long-query slice (held-out queries ≥64 WordPiece tokens). Macro = equal weight per component. Touché is banned (its args.me corpus is ArguAna's source family); Quora is banned (no license, even for eval). All selection, tuning, objective choice, and fusion fitting happen on dev and only dev. Dev reference rows computed once: BM25, potion-retrieval-32M, bge-base symmetric.
- **Go/no-go gate (dev only, ~2 days in), judged on a named checkpoint trained on TRAIN data only:** Stage-0 distilled table CI-resolved above the potion-retrieval-32M dev row; capacity probe passes its bar; the candidate CI-resolved above BM25 on the dev macro; int8 equivalent — paired bootstrap on identical queries, one-sided 97.5% upper bound of (fp16 − int8) below 0.005 on the dev macro. Pass → full program. Fail → negative-result report, stop. Report to Dylan either way.
- **Six-set access has exactly two authorized classes.** (a) Harness validation, pre-freeze, logged: reproduce bge-small ArguAna 0.6034, bge-small SciFact 0.7127, bm25 FiQA 0.2532, each to ≤0.003 — no new-model numbers before this passes. (b) The single final run. The final scorer is the sole reader of six-set and untouched-final qrels, reading only `results/frozen_eval/` after verifying `eval_manifest.json` corpus hashes against the fresh download; it appends every access to `m7/LEDGER.md` itself. Scoring a new model against six-set qrels outside (b) is forbidden.
- **The final run:** freeze the complete recipe (config, code, fusion params, preprocessing, dev-suite hashes) in a commit **pushed to GitHub** — the remote timestamp is the external witness; the final-run script refuses to start unless HEAD equals that pushed hash and the ledger holds no prior final-run entry. It covers the six + the untouched-final sets (untouched-final scored after the six, disclosed only in the report; no recipe or model change afterward). Crash handling: an infrastructure-only retry (identical commit, config, and inputs) is allowed once per crash, logged; a fix that changes code requires a new pushed freeze commit with the diff and its classification, the aborted attempt's partial scores deleted, and the report disclosing that test access preceded the revision. No relabeling a later run as final.
- **Bootstrap, pre-registered:** paired on the frozen per-query vectors; resample queries within each dataset; recompute the macro per replicate; B=10,000; fixed logged seed; per-dataset CIs alongside the macro; TREC-COVID's n=50 makes its CI wide — say so. **Confirmatory decisions are exactly three final-run comparisons** — int8 table vs lr-dense-pertask, int8 table vs BM25, released system vs OpenSearch — one-sided, Holm step-down at family α = 0.025. The dev int8-equivalence gate is a separate dev-stage decision. Everything else is exploratory and labeled.
- **Fusion:** one family (RRF or convex — pick on dev), every parameter including BM25's (bm25s-lucene defaults) frozen on dev before any test access. No per-dataset weights, normalization, or routing.
- **Mandatory ablations** (run and reported whatever they say): flat vs learned token weights; dense vs BM25-alone vs fusion with incremental gain; the two prefix variants; the three inits; int8 (dev gate, re-reported at final). Per-instruction tables only as a separately labeled oracle analysis, optional. The experiment ledger records stopped, failed, and OOM runs.
- The released table gets its own ANN sweep (M5: lookup vectors were harder for HNSW — −2.1 nDCG at default ef on FiQA vs −0.7 for bge-small, recovered at ef=512).
- Costs: three numbers, decimal MB — query asset, doc index, hydration/load.

## Working files and reporting (the headless-machine contract)

The box is headless; Dylan follows progress on GitHub. Two standing rules:

- **Commit and push frequently** — after every completed experiment, status change, or ledger append. Small commits, subject-only messages per CLAUDE.md. Dylan granted standing commit/push for this session on the M7 work branch (2026-08-25; recorded here to satisfy CLAUDE.md's explicit-approval rule for that branch only — never main, never force-push, no caches or checkpoints).
- **State is split across small files under `m7/` so answering one question never means reading everything:**
  - `m7/STATUS.md` — one screen, rewritten every push: current stage, what's running, last result, next step, open blockers. Dylan reads this first.
  - `m7/RESULTS.md` — append-only experiment table: run id, config pointer, dev metrics, verdict.
  - `m7/EXPLORED.md` — avenues tried and closed: what, why it was killed, pointer to evidence. Check it before starting anything new.
  - `m7/LEDGER.md` — the protocol ledgers: partition record, freeze record, every six-set and untouched-final access, crash re-runs.

  Details live in configs and results JSON and get pointed at, never restated. Load `STATUS.md` plus the one file the task needs, not the whole set.

## Ops

You run inside WSL2 Ubuntu; Claude Code is launched from WSL, never PowerShell. The repo is cloned into the WSL home directory (ext4), never `/mnt/c`. Host-side setup is Dylan's and lives in `setup-windows.md`.

Bring-up order, before any new numbers: (1) `nvidia-smi` (confirm 10 vs 12 GB), CUDA version, free disk; (2) Python env per `bench/`, re-download the six and dev sets via HF, run `scripts/validate_perquery.py` and verify `eval_manifest.json` corpus hashes; (3) the named harness-validation cells; (4) encode-throughput benchmark on 10K docs → extrapolate wall-clock per approved corpus and write per-stage peak RAM/disk budgets before any full encode.

Rules: doc vectors in fp16 memmaps, chunked GPU brute force — never assume a corpus fits VRAM; encode caches keyed by (model revision, dtype, tokenizer, prefix, max length, corpus hash), resumable by shard; initial peak-VRAM benchmark per training config, with gradient accumulation, mixed precision, and chunked similarity available; long runs in tmux; all work on the M7 branch under the standing commit/push grant above.

## Reuse, do not rebuild

`bench/` (MTEB-validated harness, ≤0.003 on every baseline) · `results/` including `perquery.json`, `eval_manifest.json`, `frozen_eval/`, `FINAL_MATRIX.md` · `scripts/` (bootstrap significance, costs, validate_perquery, freeze_eval_assets) · the M5 Edge prototype · `research/` (M1/M2 notes plus the three m7-* files).

## Deliverables

HF weights under the Qdrant org (Dylan's explicit go) · decision-report Artifact updating M6, through CLAUDE.md's review gates, carrying the partition ledger, the run ledger, per-dataset teacher retention, the mandatory ablations, and the labeled comparator table · the Edge demo running our table · every decision logged in CLAUDE.md.
