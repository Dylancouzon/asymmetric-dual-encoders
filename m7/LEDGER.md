# M7 protocol ledger

Append-only. Records the partition ledger, the freeze record, every six-set and
untouched-final access, and crash re-runs.

## Bring-up

- 2026-08-25 — Machine confirmed: RTX 3080 10 GB VRAM, 25 GB RAM, 16 cores, 946 GB free ext4, nvcc 12.6.

## Bring-up (continued)

- 2026-08-25 — Env: Python 3.12.14 venv, torch 2.8.0+cu126 (CUDA available, RTX 3080),
  transformers 4.57.6, datasets 5.0.1, pytrec-eval-terrier 0.5.10. Lock: `m7/requirements.lock.txt`.
- 2026-08-25 — `scripts/validate_perquery.py`: OK, 54 cells (4 allowlisted per FINAL_MATRIX.md).
- 2026-08-25 — `scripts/verify_manifest.py` (new): all six datasets re-downloaded from HF and
  matched to `results/eval_manifest.json` on n_docs/n_queries/corpus_ids/corpus_text/qids/qrels,
  and `results/frozen_eval/` matched to the fresh download. Frozen comparator pairing is valid.
- 2026-08-25 — **SIX-SET ACCESS, class (a) harness validation** (`m7src/validate_harness.py`,
  `results/m7_harness_validation.json`): bge-small ArguAna 0.6038 (want 0.6034, +0.0004);
  bge-small SciFact 0.7127 (0.0000); bm25 FiQA 0.2532 (-0.0000). All within the 0.003 standard.
  No new-model number was scored against six-set qrels in this access.

## Partition ledger (2026-08-25)

Doc-encode dtype decision, logged with evidence: teacher fp16 vs fp32 agrees to cosine
1.000000 on 10K FiQA docs (`results/m7_throughput.json`) and to |Δ nDCG@10| ≤ 3e-4 on both
CQADupStack dev components (`m7src/dtype_check.py` output). **fp16 for dev and training
corpora (2.4x throughput), fp32 for the six-set and untouched-final final run.**

### TRAIN
Approved sources only (`research/m7-data-licensing.md`): HotpotQA train qrels (BEIR),
FEVER train qrels (BEIR), SQuAD train, Amazon ESCI (US locale), MIRACL-en, Mr. TyDi-en,
NQ-open + TriviaQA question text (objective B only — TriviaQA evidence docs keep their own
copyright), plus self-generated synthetic queries over approved seeds if needed.
Excluded by decision: MS MARCO, Quora, S2ORC, PubMed, StackExchange (new dumps), GooAQ,
ELI5, WikiAnswers, sentence-transformers/embedding-training-data as a blanket source.

### DEV (pinned, hashes in `results/m7_dev_manifest.json` — frozen before any candidate result)
| component | docs | queries |
|---|---|---|
| nq-250k (all qrels-positive + rng(0) distractors to 250K) | 250,000 | 3,452 |
| hotpotqa (full BEIR corpus) | 5,233,329 | 7,405 |
| cqadup-programmers | 32,176 | 876 |
| cqadup-physics | 38,316 | 1,039 |
| heldout-train / heldout-longq (built by trainmix.py) | see m7_dev_manifest.json | |

Banned from dev: Touché (args.me is ArguAna's source family), Quora (no license).

### KNOWN-TEST (development-informed)
The six: scifact, nfcorpus, fiqa, arguana, scidocs, trec-covid. Content pinned by
`results/eval_manifest.json` + `results/frozen_eval/`, re-verified on this machine.

### UNTOUCHED-FINAL
- BEIR **FEVER** — admissible (CC BY-SA, fever.ai/download/fever/license.html, verbatim).
- BEIR **DBpedia-entity** — admissible with caveat (test collection MIT,
  github.com/iai-group/DBpedia-Entity/blob/master/LICENSE; underlying DBpedia abstracts
  CC BY-SA 3.0 + GFDL, dbpedia.org/about).
- BEIR **Climate-FEVER** — **DROPPED. Fails the affirmative-license standard.** A Sonnet
  primary-source sweep (2026-08-25) found no license statement at climatefever.ai, in
  arXiv:2012.00614 including appendices, or in github.com/tdiggelm/climate-fever-dataset
  (no LICENSE file, README silent). Only HF mirrors assert CC-BY-SA-4.0 — a wrapper tag,
  which this project does not accept as evidence (the same rule that excluded Quora).

### Dev/eval source-level license evidence (recorded at kickoff, per the eval-use standard)
- NQ — CC BY-SA 3.0, first-party but **no longer on the live README**: declared by Google's
  maintainers in merged PR #11 (2019-06-10, commit c307fa7030) and silently dropped by an
  Aug-2019 commit. Cite the commit, not the live page. The repo's LICENSE file is Apache-2.0
  and covers code only.
- HotpotQA — CC BY-SA 4.0, dataset and underlying Wikipedia corpus, hotpotqa.github.io.
- CQADupStack — CC BY-SA 3.0, stated verbatim in the ADCS 2015 paper ("released in line with
  the original licence of the StackExchange dump"), i.e. the 2014 dump, predating Stack
  Exchange's 2024 no-LLM-training clickwrap. Eval-only use here. The official download host
  (nlp.cis.unimelb.edu.au) was unreachable during the sweep; the paper text is the anchor.
  Note the HF wrapper tags contradict each other (BeIR cc-by-sa-4.0 vs mteb apache-2.0) —
  exactly why wrapper tags are not evidence.
- ESCI — Apache 2.0 at repo root (the repo is the dataset). Caveat: unanswered issue #21
  (opened 2024-11-12) asks whether Apache-2.0 covers the data; Amazon has not replied.
- MIRACL, Mr. TyDi — Apache 2.0, LICENSE files confirmed, no caveats.
- FEVER — CC BY-SA per fever.ai's own license page (per-article Wikipedia terms, 3.0 fallback).
- **BEIR itself is not a license authority**: its Apache-2.0 covers packaging/code only and
  its README disclaims per-dataset licensing.

## Incident: WSL OOM, 2026-08-25 ~23:10

Three memory-heavy jobs ran concurrently (decontamination indexing DBpedia's 4.6M abstracts at
14.7 GB RSS, the 5.23M-doc HotpotQA teacher encode at 4.7 GB, the asset freeze at 4.1 GB). Peak
hit 24 GB of 25 GB and the kernel killed a process; WSL went down. **This repeated the M4
lesson already recorded in CLAUDE.md** ("strictly sequential jobs") and it is now enforced by
`run_stage0.sh` rather than by intention.

Damage and recovery:
- Encode caches survived intact — shard-resumable by design, 73/105 HotpotQA shards kept.
- Decontamination lost entirely and was rewritten memory-bounded (below).
- Asset freeze lost its manifest write; rewritten to stream corpus hashes (`m7src/hashing.py`,
  byte-identical to the M4 `sha(json.dumps(...))` convention, verified against the frozen
  scifact entry).
- No results were lost: nothing had been scored yet.

Decontamination redesign: the index is now built over the TRAIN side (queries, and the ~1M
documents that are actually positives) and the protected corpora are STREAMED against it. Peak
RAM is one train index (~0.4 GB) whether the protected corpus is 70K CQADupStack documents or
4.6M DBpedia abstracts. Sketch reduced from bottom-64 to bottom-32 with the share threshold
scaled to match (8/32, still an estimated Jaccard of 0.25).

## Decontamination rule scope (decision, 2026-08-25)

R1 and R2 remove; R3 measures and discloses. The reasoning, recorded because it is a narrowing
of the naive reading of "removal counts logged":

- **R1 — remove on QUERY overlap, all partitions.** Query overlap is the leakage that matters.
- **R2 — remove on positive-document overlap with the six.** This is the contamination map
  (S2ORC, PubMed, CORD-19, NutritionFacts, StackExchange-finance, args.me) enforced at
  fingerprint level instead of by source name.
- **R3 — measure and disclose document overlap with DEV and UNTOUCHED-FINAL, do not remove.**
  `hotpotqa-corpus` IS the dev HotpotQA corpus and `fever-pos` is drawn from the untouched
  FEVER corpus, so a removal rule there deletes the sources rather than decontaminating them:
  it would forbid training on any Wikipedia data while evaluating on any Wikipedia benchmark.
  What removal protects — the test queries and qrels — is enforced by R1 and by the
  final-scorer ledger. Every comparator in the M4 matrix has the same property (LightRetriever
  and OpenSearch doc-v3-gte both trained on MS MARCO; bge-small on a large web mix), so the
  comparison stays like-for-like. The report states the measured overlap rates and labels BEIR
  FEVER as in-domain; DBpedia-entity is the clean generalization probe.

## Decontamination results (2026-08-26)

`results/m7_decontam.json`. 353,519 TRAIN pairs in → **352,145 kept**.

- **R1 (query overlap, removed)**: 1,329 pairs. hotpotqa-train 1,103 near / 0 exact ·
  fever-train 162 near / 2 exact · squad-train 42 near · esci-us 17 exact · mrtydi-en 3.
- **R1 on the query-text-only sources** (`results/m7_decontam_querytext.json`): nq-open 213
  removed of 86,112 (2 exact, 211 near) · TriviaQA 155 of 135,651 (10 exact, 145 near).
- **R2 (positive-document overlap with the six, removed)**: 45 pairs, from 23 of 855,324 unique
  TRAIN positive documents — a near-dup rate of **3e-05** against the six's 272,117 documents,
  0 exact duplicates. The source-level contamination map was already doing the work; the
  fingerprint pass confirms it rather than rescuing it.
- **R3 (measured, disclosed, not removed)**: CQADupStack dev 1 document. **DBpedia-entity
  (untouched-final): 15,523 exact + 79,595 near-duplicate TRAIN positives, a rate of 9.32%.**

### The DBpedia finding changes the report's framing

DBpedia-entity was the intended *clean* generalization probe — the one untouched-final set with
no training data drawn from it. It is not clean: 9.3% of our training positives near-duplicate
one of its documents. The cause is structural, not a mistake in the mix — DBpedia abstracts are
Wikipedia lead paragraphs, and so are HotpotQA's documents and SQuAD's contexts.

Consequence, to be stated plainly in the report: **after Climate-FEVER was dropped for licensing,
the untouched-final partition has no clean member.** BEIR FEVER shares its corpus with
fever-train by construction; DBpedia-entity has 9.3% document overlap. Both rows are reported
with their overlap rate attached, and neither is presented as an uncontaminated generalization
number. The mandate anticipated this ("if the untouched partition empties, the report says so");
what actually happened is weaker than empty and needs saying in those terms.

## Adversarial review of the protocol code (Fable, 2026-08-26, pre-results)

Run deliberately **before any candidate number existed**, so no finding could be weighed against
a result worth keeping. 3 BLOCKER / 6 MAJOR / 10 MINOR. All blockers and majors actioned:

- **B1 — tier decisions pairing against a re-run comparator.** `final_run.py` put a freshly
  computed BM25 row into the same dict the confirmatory loop read, so C2 (int8 table vs BM25)
  would have paired against BM25 recomputed on this box instead of the frozen per-query vector,
  violating "never re-run a comparator system". Fixed: the comparator side is now always
  `boot.from_perquery_json`, and it aborts if a frozen vector is missing. The fresh BM25 run
  remains, used only as a fusion input and an exploratory row.
- **B2 — the freeze pinned code but none of the decisive inputs.** The table lives under the
  gitignored `work/`, and preprocessing, fusion and the released-system choice were command-line
  flags applied after the freeze. Fixed: `m7/FREEZE.json` (new, `m7src/freeze.py`) pins the
  table's sha256 and byte size, its metadata hash, the preprocessing fingerprint, the dev-selected
  fusion spec, the released-system choice, and the dev/eval/perquery manifest hashes.
  `final_run.py` now reads all of it from that committed file, recomputes the table hash, and
  takes no recipe argument at all.
- **B3 — a silent undecontaminated training path.** `decontam_querytext.py` still imported
  helpers the memory rewrite had deleted, so it could not run; and `mix.query_texts` *silently
  fell back to unfiltered text* when its kept-file was absent. Fixed: script ported to the shared
  `decontam.query_hits`, and both `mix.query_texts` and `pseudoq.build_decontaminated` now raise
  rather than fall back. This one had teeth — it recovered the 213 + 155 removals above.
- **M1 — `--infra-retry` laundered anything, and its own precondition was unsatisfiable** (the
  run appends to the ledger before scoring, so a retry always faces a dirty tree). Fixed: retry
  now parses the prior `FINAL-RUN-BEGIN freeze=… table=…` marker, requires the same table hash,
  and requires `git diff prior..HEAD` to touch nothing but `m7/LEDGER.md`.
- **M2 — the gate silently used 3 of the pinned dev components**, dropping HotpotQA (where BM25
  is strongest) and both held-out slices, which could flip G3. Fixed: the gate defaults to
  `dev_eval.dev_components()`, asserts the reference rows cover every component, and prints the
  text-backed subset that BM25/potion comparisons necessarily run on.
- **M3 — R3 did not measure two of the corpora it claimed to.** nq-250k (dev) and FEVER
  (untouched-final) were never swept. Fixed in `decontam.py`; `decontam_r3_extra.py` produces the
  two missing rows without repeating the completed 30-minute run.
- **M4 — `BENCH_DATASETS` could silently redefine "the six"** (its default is the M2-era five).
  Fixed: `final_run` asserts the list equals the six before anything else.
- **M5 — conformance theatre.** The degenerate-query fallback was documented and advertised as
  "the [CLS] row" but used `rows[0]`, which is **[PAD]**; the test compared the fallback against
  itself, so the false claim would have shipped in the model card. And the released
  save→load→encode path had zero assertions. Fixed: explicit `CLS_ID = 101`, a non-circular test
  against `tok.cls_token_id`, and a real round-trip test (fp16 max|Δ| 1.1e-04, int8 2.8e-03,
  weights and update counts preserved). Suite is now 30 checks, all passing.
- **M6 — fusion selected at depth 100 on dev, applied at depth 1000 at final.** Both RRF and
  min-max convex fusion are depth-sensitive, so the frozen parameter would have been applied to a
  different function. Fixed: one `fusion.DEPTH = 1000` used by both.

Minors actioned: structured ledger marker instead of a substring match; pseudo-queries now pass
R1; content-hashed cache keys for hard-negative mining and the doc pool (a name-and-count key
would reuse stale vectors); dead `or True` in the CQADupStack loader removed (verified inert —
all qrels are score 1); argv validated before any test access; `decontam_*` scripts wrapped in
`main()` so importing them cannot execute a memory-heavy job.

Held open and disclosed rather than fixed: R1's near-duplicate test degenerates to exact match
for queries under 8 words (most NQ/FEVER-style questions), and `heldout-train` is a
seen-document/unseen-query slice (SQuAD gives ~5 questions per context), so it rewards
document-anchored memorization during dev selection. Both go in the report.

The reviewer independently verified as sound: the paired bootstrap (genuinely paired, within-
dataset resampling, correct one-sided inversion, p=0 reported as a bound), Holm's step-down,
`upper_bound_one_sided`'s tail and argument order for the int8 gate, int8 quantization including
the zero-row case, self-hit removal parity between dense and BM25 paths, `encode_cached`'s
content-hashed keys and atomic shard writes, that `train.py` reads no dev or test qrels anywhere,
and that both logged narrowings (R3 measure-not-remove; mod-50 at query granularity) are
correctly reasoned — the second strictly stronger than the mandate's literal wording.

## Held-out dev slices, and a finding about query length (2026-08-26)

Built from the frozen pool: every held-out positive plus rng(0) distractors to ~200K docs.

| component | docs | queries |
|---|---|---|
| heldout-train | 199,227 | 7,325 (esci 1,598 · fever 2,151 · hotpotqa 1,717 · squad 1,790 · mrtydi 69) |
| heldout-longq | 199,999 | **55** |

**The training mix contains essentially no long queries.** Held-out query length is p50 = 13
WordPiece tokens, p90 = 24, max = 111. Only 55 of 7,325 reach the mandated ≥64-token threshold,
and 54 of those 55 come from HotpotQA.

This matters well beyond the slice being small. ArguAna's queries average 193 words — roughly
250+ tokens, an order of magnitude past anything in TRAIN. So:

- **The long-query slice cannot validate long-query behaviour.** n=55 gives a CI far too wide to
  resolve anything, exactly as TREC-COVID's n=50 does in the M4 matrix. It is kept (the mandate
  pins it) and weighted equally as specified, with its n and CI width reported next to it.
- **The mandate's "long-query hypothesis" for learned per-token weights and length normalisation
  is untestable on this dev suite.** That ablation will be run and reported, but its result
  speaks to 13-token queries, not to 250-token ones.
- **The ArguAna row in the final matrix is an extrapolation, not a validated prediction.** M1
  already flagged ArguAna as the stress case for bag-of-tokens query encoders; we now know we
  have no dev signal on it whatsoever. The report says so rather than letting the six-set average
  imply the coverage was there.

No approved source fixes this: long argumentative queries live in args.me / idebate, which is
ArguAna's own source family and excluded by the contamination map. Same structural wall as the
document-side domain gap.

### TRAIN ↔ held-out decontamination outcome

`results/m7_decontam_heldout.json`: **2,211 further pairs removed**, leaving **349,934** TRAIN
pairs. By source: fever-train 1,847 · hotpotqa-train 295 · squad-train 64 · esci-us 4 · mrtydi-en 1.

This pass was not in the naive reading of the mandate and it earned its place. The mod-50 rule
guarantees held-out queries are *exactly* disjoint from TRAIN, not near-duplicate-disjoint —
and FEVER turns out to contain many near-identical claims about the same entity, so 1,847 of them
straddled the split. Without this pass the heldout-train dev component would have been scoring a
model on paraphrases of its own training queries, and every dev-based selection decision built on
it would have been inflated.

### R3 sweeps completed (the two the first run missed)

- **nq-250k (dev)**: 406 exact + 3,942 near of 856,515 TRAIN positives — 0.46%.
- **FEVER (untouched-final)**: 47,289 exact + 96,573 near — **11.3%**. Expected in direction
  (fever-pos is drawn from this corpus by construction) but the rate covers all TRAIN positives,
  so hotpotqa/squad/mrtydi Wikipedia documents contribute to it too.

All five protected corpora are now measured: six 3e-05 · cqadupstack-dev ~0 · nq-250k-dev 0.46% ·
DBpedia-entity 9.32% · FEVER 11.3%. The two untouched-final sets are the two most overlapped —
which is the finding, not a coincidence: both are Wikipedia, and so is most of TRAIN.

### The held-out slices were rebuilt against the full pool (2026-08-26)

The first construction gave each slice ~200K documents: all held-out positives plus random
distractors. The teacher then scored **0.8383** on heldout-train and **0.9915** on heldout-longq,
and its dev macro rose from 0.6106 (four components) to 0.7120 (six). Both slices were
near-saturated — a random distractor drawn from 6M documents is almost never confusable with the
true positive — so neither could discriminate between candidates, and under equal per-component
weighting they would have made the go/no-go gate easier to pass for no methodological reason.

Rebuilt with the **entire 6,169,142-document pool** as the corpus. That makes each slice as hard
as a real 6M-document retrieval task, adds no teacher-derived bias (mining hard distractors with
the teacher would bias the component toward the teacher's own ranking, which is the thing being
measured), and is cheaper: the document vectors are the pool memmap itself, so nothing is copied.

heldout-longq remains n=55 and is reported with its CI width attached, the same treatment
TREC-COVID's n=50 gets in the M4 matrix.

## Stage 0.1 — closed-form representation compatibility (2026-08-26)

`m7src/stage0_ridge.py`, `results/m7_stage0_ridge.json`. The MSE-optimal flat-weight
bag-of-tokens approximation of the frozen teacher's query encoder, from one ridge solve per
lambda (30,522² fp64 Cholesky, ~9.5 TFLOP each). This is the **global optimum of flat-weight
distillation under squared loss**, so no training run can beat it at that objective — which is
what makes it a clean answer to the mandate's central structural question rather than one more
data point.

Fitted on **571,329 decontaminated TRAIN queries**; bag matrix 8,205,703 nnz; Gram 1.8% dense;
**vocabulary coverage on TRAIN queries 0.895** (so ~3,200 of 30,522 rows never receive an update
and fall to the unseen-row policy).

| lambda | train cos | overlap@10 | proxy macro-3 |
|---|---|---|---|
| 1e-4 | 0.9117 | 0.485 | 0.4534 |
| 1e-3 | 0.9117 | 0.487 | 0.4535 |
| **1e-2** | **0.9110** | **0.490** | **0.4542** |
| 1e-1 | 0.9044 | 0.464 | 0.4367 |
| 1 | 0.8724 | 0.351 | 0.3604 |

Proxy references on the same three components: teacher 0.5722 · BM25 0.4083 · potion 0.3525.

**Verdict: the structural bet holds.** A frozen, off-the-shelf bge-base document space *is*
additively predictable from query tokens — cosine 0.9110 with the teacher's own query vectors,
half its top-10 recovered, and BM25 beaten by 4.6 points with flat weights and zero gradient
steps. The mandate's stated worry (LightRetriever's table works because its document tower was
co-trained to be additively predictable; a frozen tower was never optimised for that) does not
bite hard enough to kill the approach.

Three qualifications the report must carry:

1. **Retention is 79%, not 95%.** The honest headline is not "lookup tables match transformers"
   but "a frozen off-the-shelf document space is additively predictable enough from query tokens
   to beat BM25 at zero query compute". The flat table sits between BM25 and the teacher, nearer
   BM25.
2. **The optimum is interior and lambda barely matters below it** (0.4534 → 0.4542 across three
   orders of magnitude, then falling). The binding constraint is therefore *representational*,
   not statistical: more data or better regularisation will not move this. Only a more expressive
   query function (learned per-token weights) or a better-aligned objective (contrastive) can.
3. **Per-component retention is very uneven, and unevenly in the wrong direction.** nq-250k
   0.7285/0.8198 = 89%; cqadup-programmers 0.2724/0.4240 = **64%**. StackExchange-style question
   retrieval is the closest analogue in dev to FiQA's domain, and it is where the bag-of-tokens
   approximation is weakest. That makes FiQA the six-set row most at risk — and FiQA is also
   where BM25 is unusually weak (0.2532), so the dense and fusion stories pull opposite ways
   there. Flagged now, before any six-set number exists.
