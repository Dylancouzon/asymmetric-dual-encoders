# M7 protocol ledger

The load-bearing record: partitions, licence evidence, every six-set access, decontamination
counts, gate results, freeze record, incidents. Detail lives in `results/m7_*.json` and is
pointed at, never restated.

> **Compacted 2026-08-26** after the go/no-go gate. Every protocol-required fact is kept
> verbatim; settled justification prose was cut to one line each. The full original narrative is
> in git history (`git log -p m7/LEDGER.md`) and in the results JSONs.

## Environment

- Box: RTX 3080, **10 GB VRAM**, 25 GB RAM (peak budget 18 GB), 16 cores, ext4, nvcc 12.6.
- Stack: Python 3.12.14, torch 2.8.0+cu126, transformers 4.57.6, datasets 5.0.1,
  pytrec-eval-terrier 0.5.10, qdrant-edge-py 0.8.0, Qdrant server v1.19.0. Lock:
  `m7/requirements.lock.txt`.
- Teacher: **BAAI/bge-base-en-v1.5 @ a5beb1e3e68b9ab74eb54cfd186867f64f240e1a**.
- Doc-encode dtype: **fp16 for dev + training, fp32 compute for the final run; fp16 at rest
  everywhere**, matching the M4 convention the frozen comparators were produced under. Evidence:
  cosine 1.000000 vs fp32 on 10K docs, |Δ nDCG| ≤ 3e-4 on both CQADupStack components.

## Verification

- `scripts/validate_perquery.py` OK, 54 cells (4 allowlisted per FINAL_MATRIX.md).
- `scripts/verify_manifest.py`: all six datasets re-downloaded and hash-matched to
  `results/eval_manifest.json`; `results/frozen_eval/` matched the fresh download. Frozen
  comparator pairing is valid.
- **SIX-SET ACCESS, class (a) harness validation** (2026-08-25, `results/m7_harness_validation.json`):
  bge-small ArguAna 0.6038 (want 0.6034), SciFact 0.7127 (0.0000), bm25 FiQA 0.2532 (−0.0000).
  All within 0.003. No new-model number was scored against six-set qrels in this access.
- Conformance suite 30/30 (`m7src/test_conformance.py`), including the real save→load→encode path.

## Partitions

**TRAIN** — approved sources only (`research/m7-data-licensing.md`). Final count after all
decontamination: **349,934 pairs** + 221,395 query-text-only rows for objective B. Per-source
fields, rights, positive construction and counts: `results/m7_field_table.md`.

**DEV** (pinned; hashes in `results/m7_dev_manifest.json`, frozen before any candidate result):
nq-250k 250,000/3,452 · hotpotqa 5,233,329/7,405 · cqadup-programmers 32,176/876 ·
cqadup-physics 38,316/1,039 · heldout-train (corpus = the full 6,169,142-doc pool)/7,325 ·
heldout-longq (same corpus)/**55**. Banned: Touché (args.me is ArguAna's source family), Quora
(no licence). BM25 and potion have no row on the two held-out slices (their corpora are pool row
indices carrying no document text), so those comparisons run on the four text-backed components.

**KNOWN-TEST** — the six, development-informed. Content pinned by `results/eval_manifest.json` +
`results/frozen_eval/`.

**UNTOUCHED-FINAL** — BEIR FEVER and DBpedia-entity. **Climate-FEVER dropped: fails the
affirmative-licence standard** (no statement at climatefever.ai, in arXiv:2012.00614 incl.
appendices, or the GitHub repo; only HF mirrors assert CC-BY-SA-4.0, and a wrapper tag is not
evidence here — the same rule that excluded Quora).

### Source-level licence evidence (eval-use standard)
- **NQ** CC BY-SA 3.0 — first-party but **not on the live README**: declared in merged PR #11
  (2019-06-10, commit `c307fa7030`) and silently dropped Aug 2019. Cite the commit. Repo LICENSE
  is Apache-2.0, code only.
- **HotpotQA** CC BY-SA 4.0, dataset and Wikipedia corpus, hotpotqa.github.io.
- **CQADupStack** CC BY-SA 3.0, verbatim in the ADCS 2015 paper (the 2014 Stack Exchange dump,
  predating the 2024 no-LLM-training clickwrap). Eval-only here. HF wrapper tags contradict each
  other (BeIR cc-by-sa-4.0 vs mteb apache-2.0) — why tags aren't evidence.
- **FEVER** CC BY-SA, fever.ai's own licence page.
- **ESCI** Apache 2.0 at repo root. Caveat: unanswered issue #21 asks whether it covers the data.
- **MIRACL, Mr. TyDi** Apache 2.0, LICENSE files confirmed.
- **DBpedia-entity** test collection MIT (iai-group/DBpedia-Entity); abstracts CC BY-SA 3.0 + GFDL.
- **BEIR itself is not a licence authority** — its Apache-2.0 covers packaging only.

## Decontamination

Rules (narrowed deliberately; reasoning kept because it is part of the protocol):
- **R1 remove** on query overlap, all partitions. Query overlap is the leakage that decides scores.
- **R2 remove** on positive-document overlap with the six — the contamination map enforced at
  fingerprint level rather than by source name.
- **R3 measure and disclose, do not remove**, for DEV and UNTOUCHED-FINAL documents.
  `hotpotqa-corpus` **is** the dev HotpotQA corpus and `fever-pos` comes from the untouched FEVER
  corpus, so removal there would delete the sources rather than decontaminate them — it would
  forbid training on Wikipedia while evaluating on any Wikipedia benchmark. What removal protects
  (test queries and qrels) is enforced by R1 and the final-scorer ledger, and every comparator in
  the M4 matrix has the same property, so the comparison stays like-for-like.

Method: blake2b-64 word hashes, polynomial rolling word-8-grams, bottom-32 sketch, ≥8/32 shared
(est. Jaccard ≥ 0.25). Index built over the TRAIN side, protected corpora streamed against it, so
peak RAM is ~0.4 GB regardless of corpus size.

Results (`results/m7_decontam.json`, `..._querytext.json`, `..._heldout.json`):
- R1: 1,329 pairs. Plus nq-open −213, TriviaQA −155.
- R2: 45 pairs, from 23 of 855,324 unique positives — **3e-05** against the six. The source-level
  map was already doing the work.
- TRAIN↔held-out: **2,211 further pairs** (fever-train 1,847 — FEVER contains many near-identical
  claims that straddled the mod-50 split). Without this pass, `heldout-train` would have scored
  models on paraphrases of their own training queries.
- R3 overlap: six 3e-05 · cqadupstack-dev ~0 · nq-250k-dev 0.46% · **DBpedia-entity 9.32%** ·
  **FEVER 11.3%**. The two untouched-final sets are the two most overlapped — both are Wikipedia,
  and so is most of TRAIN.

**Consequence: the untouched-final partition has no clean member.** Both rows are reported with
their overlap rate attached; neither is presented as an uncontaminated generalisation number.

Held-out slice rule: mod-50 applied at **query** granularity, not pair — strictly stronger than
the mandate's literal wording (per-pair holdout would leave a held-out query's text in TRAIN via
its other positives). Disclose: `heldout-train` is a *seen-document/unseen-query* slice (SQuAD
gives ~5 questions per context), so it rewards document-anchored memorisation during dev selection.

## Stage 0

**0.1 closed-form ridge** (`train_cos` below is an IN-SAMPLE fit residual over the 571,329
fitting queries, not a held-out agreement figure; in bge's anisotropic space it also needs a
baseline to be interpretable — overlap@10 is the honest agreement number and says half the
teacher's top-10 is not recovered) (`results/m7_stage0_ridge.json` pending a full-suite re-eval): the
MSE-optimal flat-weight bag-of-tokens approximation of the teacher's query encoder — the global
optimum of flat distillation under squared loss. Fitted on 571,329 TRAIN queries; vocab coverage
**0.895**. Best λ=1e-2: **0.4542** dev proxy macro-3, train cos 0.9110, overlap@10 0.490 (vs
teacher 0.5722, BM25 0.4083, potion 0.3525). λ curve 1e-4→10: .4534 .4535 **.4542** .4367 .3604
.2004 — interior optimum, so the binding constraint is *representational*, not statistical.
At λ=10 the rows barely move, giving ~0.20: **the teacher-derived init alone is a poor table**,
useful only as a regularisation anchor.

**0.2 capacity probe** — PASS at ~1.0000 across components, d=+0.5917. **Near-vacuous**: 23.4M
parameters against ~3,500 dev queries makes memorisation trivial. It falsifies only the
hypothesis that good retrieval is *inexpressible* here. The load-bearing Stage-0 evidence is the
ridge probe, which generalises.

**Objective grid** (dev proxy macro-3): **p1-objB distillation 0.4548** · p1-objC B→A 0.3721 ·
p1-objA contrastive 0.3248. Details `m7/RESULTS.md`.

**Contrastive InfoNCE with random negatives is destructive**, from two initialisations: monotone
0.3532→0.3248 over 12k steps, and 0.4449→0.3721 (−7.3) over 8k steps from a healthy checkpoint.
`reg_init` tested and exonerated (weakest at the high update counts where C degraded fastest).
The first stated mechanism ("random negatives trivially separable") **does not survive arithmetic**
— at τ=0.02 with 32,768 negatives the loss is ~3.4, not ~0 — so the cause is being identified by
measurement (`m7src/diag_scores.py`: score geometry, softmax mass per temperature, and what
fraction of the *hardest* negatives the `fn_margin` filter removes) rather than by ablation.

**This contradicts a mandate premise.** `instructions-m7.md` says "Frozen doc vectors make very
large negative pools nearly free — exploit that first." Scale without hardness wasted the
objective; few-and-hard beats many-and-easy. Phase 2 was expanded to the mandate's full
comparison (BM25-mined / teacher-mined / mixed) and the BM25 arm built
(`train.mine_bm25_negatives`, mined within each query's own doc store).

## GO/NO-GO GATE: **GO** (2026-08-26 03:03)

`results/m7_gate_p1-objB.json`. Full six-component dev suite; BM25/potion on the four text-backed.

| condition | result |
|---|---|
| G1 Stage-0 table > potion | **PASS** d=+0.0994 CI=[0.0910,0.1078] p<1e-4 |
| G2 capacity probe > BM25 | **PASS** d=+0.5917 (near-vacuous, above) |
| G3 candidate > BM25 | **PASS** d=+0.0270 CI=[0.0188,0.0353] p<1e-4 |
| G4 int8 equivalence (bar 0.005) | **PASS** d=+0.0001, upper=0.00053 |

Text-backed macros: candidate fp16/int8 **0.4795** · BM25 0.4525 · potion 0.3801 · teacher 0.6106.
**Retention 0.7853** text-backed / 0.8073 all six — agreeing with the ridge probe's 79% from a
different method. `p1-objC` fails G3 (d=−0.0383) as expected.

### The GO is one component wide (correction, 2026-08-26, post-review)

G3's per-dataset breakdown, which was in the committed JSON from the start and in none of the
prose until an adversarial review flagged it:

| vs BM25 | delta | CI95 | n |
|---|---|---|---|
| nq-250k | **+0.1445** | [+0.1312,+0.1580] | 3,452 |
| cqadup-physics | +0.0152 | [−0.0040,+0.0346] | 1,039 |
| cqadup-programmers | −0.0203 | [−0.0429,+0.0023] | 876 |
| **hotpotqa** | **−0.0316** | **[−0.0395,−0.0236]** | 7,405 |

The +0.0270 macro is carried entirely by nq-250k — the component whose query distribution is most
represented in TRAIN (86K nq-open rows feed objective B, guarded only by an R1 near-dup test that
degenerates to exact match under 8 words). The candidate **loses to BM25 CI-resolved on
HotpotQA** and directionally on cqadup-programmers, i.e. it wins on the in-distribution component
and loses on the ones most like the six.

**Projection to the six: ~0.41** = 0.785 text-backed retention x a plausible bge-base six-set row
of ~0.52 (bge-small measured 0.5042). That is below BM25's 0.4174 on the six and far below the
0.4583 release bar; substituting the 64% cqadup retention for FiQA-like sets lowers it further.
**On today's evidence the best candidate projects to Tier 4.** The gate answers "is the program
alive" (yes); it does not support "on track", and nothing in the record claimed otherwise before
this entry, which is the omission.

**Checkpoint substitution, logged in advance:** the driver named `p1-objC` before any result
existed; gating a knowingly-inferior checkpoint would be a false negative, so `p1-objB` was also
gated and both are reported. Selecting on dev is within the protocol — the gate is a dev-stage
decision.

## Strategy pivot: stop, research, re-plan for Tier 1 (Dylan's call, 2026-08-26)

Trigger: the corrected projection puts the best candidate at ~0.41 on the six (Tier 4). Direction:
do not grind phase 2 forward; research properly, revise the plan, aim for Tier 1, restart the
model work if needed. **The plan itself lives in `m7/STATUS.md`** (rewritten, not appended).

Decision record: **Tier 1 is unreachable by the dense table alone** — it needs ~94% retention on
bge-base or ~91% on bge-large, neither plausible for a bag encoder. Tier 1 therefore requires
better teacher x retention 85-88% x fusion. Full arithmetic in STATUS.

**Not invalidated by a model-side restart, and to be preserved:** the eval protocol, partition
ledger, decontamination, pinned dev suite, frozen comparator vectors, freeze/final-run machinery,
and both adversarial reviews. Only architecture and training recipe change.

Two results that narrow the search (proved, not researched):
- A doc-side linear map is nearly a no-op: `q.(Ad) = (A^T q).d`, so it reparametrises the table we
  already optimise rather than adding capacity.
- Centering documents cannot change ranking at all: `q.(d-mu) = q.d - q.mu` is a per-query
  constant. (Centering the QUERY side does change ranking.)

## Other findings that constrain the report

- **Dev cannot validate long queries.** Held-out length p50=13 WordPiece tokens, p90=24, only 55
  of 7,325 at ≥64; ArguAna's are ~250. The ArguAna row is an extrapolation and the learned-weight
  "long-query hypothesis" is untestable here. No approved source fixes it (args.me is ArguAna's
  own family). heldout-longq keeps its n and CI width attached, as TREC-COVID's n=50 does in M4.
- **Learned per-token weights buy +0.0006** over the flat closed form (no CI yet) — currently
  unjustified complexity in the artifact.
- **FiQA is the six-set row most at risk**: ridge retains 64% of the teacher on
  cqadup-programmers vs 89% on nq-250k, and StackExchange-style retrieval is the nearest dev
  analogue. FiQA is also where BM25 is weakest, so dense and fusion pull opposite ways.
- **int8 is quality-free** on two checkpoints (upper bound 0.00053 vs a 0.005 bar), replicating
  M3's LightRetriever finding for our own table. Released query asset **23.4 MB int8**.
- **Held-out slices were rebuilt** against the full 6.17M pool: with ~200K random distractors the
  teacher scored 0.8383/0.9915 and the slices could not discriminate, inflating the teacher's dev
  macro from 0.6106 to 0.7120. Random distractors from 6M docs are almost never confusable.
  Mining hard distractors with the teacher was rejected as biasing the component toward the
  teacher's own ranking — the thing being measured.

## Reviews

**Fable, pre-results, protocol code only** (2026-08-26, deliberately before any candidate number
existed): 3 BLOCKER / 6 MAJOR / 10 MINOR, all blockers and majors actioned.

B1 tier decisions paired against a **re-run** BM25 instead of the frozen vectors · B2 freeze
pinned code but not the table bytes / preprocessing / fusion · B3 `decontam_querytext` couldn't
run and `mix.query_texts` **silently fell back to unfiltered text** (fixing it recovered 368
overlapping queries) · M1 `--infra-retry` laundered anything and its precondition was
unsatisfiable · M2 gate silently used 3 of 6 dev components · M3 R3 never swept nq-250k or FEVER ·
M4 `BENCH_DATASETS` could redefine "the six" · M5 the degenerate-query fallback used `rows[0]` =
**[PAD]** while documenting "[CLS]", and its test compared the fallback to itself · M6 fusion
selected at depth 100, applied at depth 1000. All fixed; see `git log` and the modules.

**Fable, post-results, results + plan** (2026-08-26): 1 BLOCKER / 6 MAJOR / 4 MINOR.
BLOCKER — the GO headline omitted that the win is one component wide (corrected above).
MAJOR — only the *sampled* positive was masked from negatives (ESCI averages ~13.5/query, and the
`fn_margin=0` arm would have confounded its own reading); the fn-mask rate was never logged, so the
leading collapse suspect ran unobserved all grid; objective C's regulariser anchored to the teacher
init rather than the B checkpoint; phase 2's arms all inherited the suspect tau/lr/fn_margin;
"two independent ways" was one estimator family with two optimizers; two remedies dismissed too
fast (see EXPLORED). All actioned. Also flagged: `train_cos` is an in-sample residual and lost its
qualifier in prose; the +0.0006 learned-weights claim has no CI and is confounded.

Verified sound by that review: the paired bootstrap (genuinely paired, within-dataset resampling,
correct one-sided inversion), Holm step-down, `upper_bound_one_sided`'s tail and argument order,
int8 quantisation incl. the zero-row case, self-hit removal parity across dense and BM25 paths,
`encode_cached`'s content-hashed keys and atomic shard writes, that `train.py` reads no dev/test
qrels anywhere, and both logged narrowings (R3 measure-not-remove; mod-50 at query granularity).

Held open and disclosed rather than fixed: R1's near-dup test degenerates to exact match for
queries under 8 words (most NQ/FEVER-style questions); `heldout-train` is seen-document.

## Incidents

- **2026-08-25 ~23:10 WSL OOM (self-inflicted).** Three memory-heavy jobs concurrently hit 24 of
  25 GB; kernel killed a process and WSL went down. Repeated the M4 lesson already in CLAUDE.md.
  Nothing scored yet, so no results lost; encode caches survived (shard-resumable). Fixes:
  decontamination rewritten TRAIN-side-indexed, pool index per-store and lazy, `encode_cached`
  returns a memmap, hashes streamed, jobs strictly sequential via the `run_stage0*.sh` drivers.
  Peak-RAM budget now explicit at 18 GB. Recurring-mistake list: `m7/CODEMAP.md`.
- **2026-08-26 05:52 reboot — Windows Update, not a crash.** Event 1074, `TrustedInstaller.exe`,
  `NT AUTHORITY\SYSTEM`, reason "Operating System: Upgrade"; Kernel-Boot Event 20 reports last
  shutdown success **true**. No Event 41/6008/bugcheck, no thermal or power event. Gate finished
  03:03; box idle ~3 h before. Nothing lost. **Host action for Dylan: stop Windows Update
  rebooting mid-run** (active hours / pause / no-auto-restart-with-logged-on-users).
- A cosmetic bug crashed the gate's *printer* after its JSON was written (G4 is an equivalence
  bound with no `ci95` key). Verdicts never at risk; each field is now guarded independently.
