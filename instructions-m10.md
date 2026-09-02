# M10 — nano v2: the retry, built around coverage (mandate)

*Written 2026-09-01 by the planning session on Dylan's direction ("M9 failed to achieve our goals
… make M10 a retry … build something unimpeachable by competitors"). Evidence `m10/PLANNING.md`;
M9's record `m9/FINDINGS.md`. Adversarial review: gpt-5.6-terra, read-only, three passes
(`research/m10-codex-plan-2026-09-01.md`, `-plan2-`, `-plan3-`, `-plan4-`; full logs are gitignored `.log` files beside them); every finding and its
disposition is in PLANNING §8. The M9 model is **nano**; M7's table is **zero**; the product is
still the pair on one stella index.*

## What binds from M7 and M9 — exhaustive

From `instructions-m7.md`: decision authority; vendor and source-licence rules (as relaxed in
CLAUDE.md); frozen-comparator validation and pairing; pushed-freeze, ledger, crash/abort
disclosure; Sonnet-only research subagents; the headless reporting files under `m10/`
(STATUS/RESULTS/EXPLORED/LEDGER); the eval-use licence standard (source-level evidence before a
selection set is used); CLAUDE.md's tightness rule. From `instructions-m9.md`: the bars, the
permitted claim and its qualification, the six-set transaction design (`m9/FINAL_LOCK.md`,
re-registered for M10 with new hashes and the one change in §Final run), the LoTTE atomic-read
protocol, R1/R2/R3, the stage discipline (no evaluation before its lock; "diagnosed defect" =
implementation divergence only), the 70 MB fp16 query-asset target, screen-in-closed-form first,
the watch-long-runs checklist. **M10 supersedes** M9's data pool, mix, optimizer, schedule, plateau
rule, phase-2 handling, screen arm list, and selection surfaces. Where M10 changes a carried
mechanism, M10 controls. Read `m8/CODEMAP.md` and `m9/CODEMAP.md` before writing code.

## Why M9 missed, and what M10 changes

M9's candidate retains **93.8%** of the teacher on NQ and **50–71%** on the two CQADupStack dev
components (`m9/FINDINGS.md` §1). The thesis: the 463K-query pool (all Wikipedia QA and product
search, longest query 108 words) did not cover the forms the model is tested on, and the student
had the capacity. **The thesis is tested before anything is built** (§Screen family A), and the
test's reach is stated: it covers the query families the COV surface contains (§Surfaces), and
the six-set transaction remains the only test of the forms COV lacks. M10 changes, in order of
evidence:

1. **Coverage** — a query corpus of ~4.5M texts spanning 12 query forms (§Data). Evidence: the
   per-component spread; LEAF's ~1.8M query-like texts across five styles and its one-epoch
   ablation (queries-only 46.7 vs queries+docs 60.7).
2. **Optimizer regime** — LEAF's small batch and cyclic anneals (1-epoch loss bs16 0.4194 ≈ bs32
   0.4214 ≪ bs256 0.4593; 3 cycles 1e-4→1e-5). M9 ran 113 examples/step at constant LR; its plateau
   rule read un-annealed points and the anneal it never modelled added +0.004.
3. **A ranking-aware phase-2 loss registered at lock and priced by a screen arm** (M9's
   symptom gate was never specified). Evidence: listwise KL distillation at 90–95% retention for
   17–32M students (mxbai-edge-colbert); EmbedDistill's score-distillation ablation; M8's `R-LIST`.
4. **Warm start from the M9 candidate** as a registered init arm (2306.11550: init moves retention
   up to 6 points). The M9 checkpoint is already at 94% on NQ.

The student cap stays at 35M (LEAF's class; the pair's story); decision 6 makes a larger tier a
conditional owner decision, not a silent refusal.

## Owner rulings already made (Dylan, 2026-09-01)

- **Git:** M9 is merged to `main` after the close-out cleanup; M10 execution work happens on branch
  **`m10-work`** under the headless commit-and-push contract; merges to main at stage boundaries
  need Dylan's go. M9's registered six-set close-out still runs from `m9-work`, because `guard9`
  pins that branch (`m9src/guard9.py:35`); the branch is kept until then.

## Owner decisions (defaults apply until Dylan rules; each is recorded in `m10/LEDGER.md`)

| # | decision | default while open |
|---|---|---|
| 1 | Ratify M9's final-lock amendment **together with the close-out amendment that strikes M9's reserved conditional** (§Stage plan, M10.2): M9's close-out is six-only and cannot spend the reserved access | blocks the close-out only |
| 2 | Money: one A100/H100 for ≈ 80–110 GPU-hours (≈ **$120–280** at $1.5–2.5/h) and/or hosted open-weights generation (≈ 1.1B tokens ≈ **$110–330**); prices unverified Sept 2026 | box-only path when the box is reachable |
| 3 | FineWeb as a **seed** for synthetic queries (the 2026-08-30 approval covered document-side text; `research/m7-data-licensing.md` records FineWeb as not approved for seeding) — needs a source-level rights review | seeds = Wikipedia + the approved pool corpora |
| 4 | PAQ (machine-generated questions over Wikipedia; data CC BY-SA, generation code CC BY-NC) as query text | include; 1.0M uniform sample in the build (seed 0, pinned revision), 4.037M in the volume-control screen arm A2 only; attribution recorded |
| 5 | Confirm: FineWeb **documents are excluded** from M10 (no reserved-set document fingerprints exist and creating them would open reserved corpora — `m9/LEDGER.md` §1.3) | excluded |
| 6 | If the capacity probe clears 85% on the CQA-2 components, scope a **>35M tier** as a separate frontier point (its 109M student is 768-hidden, so part of any clear is output width, which family G already buys) | nano ships at ≤35M regardless |
| 7 | Confirm: LoTTE read #1 withdrawn unexecuted in M9; renumbering M10/M11/M12 | as recorded in `m9/STATUS.md` and CLAUDE.md |

## Goal, bars, and the permitted claim — unchanged from M9

nano = a ≤35M transformer query encoder serving stella-400M's frozen 1024d index (the SAME index
as zero). Bars, paired on `results/perquery.json` (sha `6b18e3dd…`, irreplaceable):

| | avg-6 | NDO-4 |
|---|---|---|
| teacher ceiling (stella symmetric) | 0.5744 | 0.5640 |
| **AIM: leaf-ir-asym** | **0.5155** (89.7% retention) | 0.5233 |
| **RELEASE BAR: bge-small** | **0.5042** (87.8%) | 0.5046 |

C1 nano-dense > bge-small (release); C2 nano-dense > leaf-ir-asym (aim). Both dense-vs-dense;
fused rows descriptive only. **C2 is a whole-system comparison** (stella documents + nano queries
vs arctic documents + LEAF queries): different document towers, index sizes, encode costs and
disclosed teacher overlap. It supports exactly M9's verbatim headline sentence and **no statement
about nano versus LEAF's query tower**; the report carries both systems' retention against their
own teachers, index bytes, document-encode cost and query latency beside the number. NDO-4 and
reserved NDO-3 stay descriptive. Additional mandatory disclosures: per-dataset retention, the
synthetic-data provenance table (generator revision and terms, prompts, seed sources and ids,
counts per form, removal counts per screen), the selection-surface table (licence evidence,
revisions, sizes, removals from training data), dose in examples / tokens / GPU-hours beside
LEAF's ~100 A100-hours, and the dev-reuse count.

## Stage plan

- **M10.0 DIAGNOSIS + SCREEN LOCK** (no six-set, reserved or LoTTE output exists during this stage).
  (a) Rank-bottleneck probe — **done on the Mac 2026-09-01** (`results/m10_rank_probe_mac.json`,
  PLANNING §9): the reconstruction-optimal (PCA) 384-d subspace of stella's query space retains
  99.5% of a single query distribution's retrieval and 90–93% once it is fit to three distributions
  at once; at 640 it retains 98–100%. L2 regression is a reconstruction objective, so this is the
  subspace an L2-trained 384-hidden linear-head student is pushed toward. **It is evidence that
  output width binds under L2 distillation, not a bound on every 384-d subspace** (a ranking-optimal
  one may do better). **Action taken:** the head stays linear (a nonlinear head has no fastembed
  path) but the pooled **feature widens** by concatenating mean-pooled states of three layers
  (§Recipe); screen family G decides it against the 384-d and 768-d alternatives.
  (a2) Serving-parity check for the multi-layer head: export the per-token head over the three
  layers' token states, let fastembed mean-pool, compare to the in-graph pooled output on M9's
  parity sample (min-cos ≥ 1−1e-4, max-abs ≤ 1e-3). Must pass before family G is locked.
  (b) Capacity probe (`m9src/capacity_probe.py`, 109M student, M9 anchor dose, unchanged): ≤75%
  on the CQA-2 components → capacity is not binding at that dose; ≥85% → decision 6 fires. Between
  → reported, no action.
  (c) Per-component DEV-6 read of the M9 candidate incl. `heldout-longq` (the baseline row).
  (d) **COV admission** (§Surfaces): for every candidate component, commit to `m10/LEDGER.md` its
  primary-source licence URL and terms, HF repo and revision, corpus size, query count, qrels
  format and the retrieval metric, its corpus-level contamination check and its fingerprint
  screen against the six and the reserved four. A component is named COV only after that record
  is pushed. Then **every admitted COV corpus, query set and document set joins the protected
  index** (`m8src/protected_filter`) before any PAQ or synthetic text is constructed.
  (e) **Screen lock**: `m10/LEDGER.md` §0 fixes every arm of §Screen (eleven arms), order, doses,
  seeds, the τ rule, surfaces, the thirteen contrasts, multiplicity control, confirmation design
  and outcome→action maps.
- **M10.1 DATA.** Generation (per-form smoke of 200 queries, read by a person, rate measured,
  before scaling), PAQ samples, decontamination against the protected index (now including COV)
  and the six's documents, the FORMS-12 hold-out, teacher targets, hard-candidate mining (§Data),
  `results/m10_data_manifest.json` with hashes and the provenance table. **After the manifest is
  immutable and before any arm**, the τ rule is executed and its table recorded.
- **M10.2 SCREEN + RECIPE LOCK.** The arms of §Screen, the confirmation runs, then one pushed lock
  commit with every field of M9's M9.2 list filled — including the phase-2 loss, the best-to-best
  plateau/extension rule on annealed checkpoints, the final-run registry and LoTTE read #1's
  manifest. Codex and Fable review the pushed lock. **Then, and only then,** M9's close-out runs:
  its registered six-set transaction **amended before execution to six-only** (the `if C1 then
  execute` reserved conditional is struck; disclosed and ratified under decision 1), so M9 cannot
  spend the reserved access. Its rows are a forecasting calibration and a whitepaper frontier
  point; nothing in M10 may change on them. LoTTE read #1 then runs as the registered veto on
  the selected recipe.
- **M10.3 BUILD.** One candidate under the kill and extension rules; export + parity; freeze;
  pre-freeze review; LoTTE read #2 (audit only).
- **M10.4 FINAL.** The six-set transaction → decision → the registered reserved conditional.

## Data

**Query corpus (~4.5M unique texts, query role):**

| source | build count | licence | role |
|---|---|---|---|
| M9 real queries (hotpotqa, squad, esci, mrtydi, nqopen, triviaqa; fever out) | 463K | CC BY-SA / Apache | real forms |
| PAQ sample (decision 4) | 1.0M | CC BY-SA (data) | factoid volume — capped so it cannot dominate |
| **Synthetic, Qwen3-8B (Apache-2.0; revision pinned), 4-bit, vLLM on the box or hosted** | 3.0M | generated under the generator's terms; provenance pinned; **not redistributed** without review | form breadth |

**Form taxonomy — 12 forms, 250K each** (quotas locked at M10.1; ±10% realized): factoid question ·
how-to / troubleshooting question with title and body · scientific claim (a statement) · long
counter-argument paragraph (120–220 words) · finance / personal-economics question · paper-title
query · 2–4-word keyword query · consumer-health question · product-search query · comparison
question · yes/no verification question · conversational multi-sentence request. **Seeds:**
Wikipedia stratified by top-level category (CC BY-SA) and the approved pool corpora; FineWeb only
if decision 3 approves it, and then only after URL-domain exclusion of every source family below
and the screens below. **Contaminating source families — never seeds, never regression text, and
excluded from COV** (`research/m7-data-licensing.md` map): S2ORC / Semantic Scholar; PubMed;
NutritionFacts.org and its mirrors; CORD-19; StackExchange personal finance (money.SE) and Reddit
finance; args.me / idebate; every six-set and reserved corpus.

**Screens on every generated or PAQ query, thresholds fixed here (M7's fingerprints,
`m7src/decontam.py`):** exact `blake2b-64` match or word-8-gram bottom-32 sketch ≥ 8/32 against
(i) the protected index (six + dev + reserved + LoTTE **+ admitted COV queries and documents**),
(ii) the six's documents, (iii) the query's own seed passage (a copied span is not a query);
word-4-gram containment for 4–7-word queries. The M9 real-query pool and the document pool are
re-screened against the COV additions (R1 removes matching queries; matching pool documents are
removed too). Removal counts per screen, per form and per COV component are recorded **before any
COV component is scored**. **FORMS-12 hold-out:** 500 seed documents per form are set aside first;
queries generated from them are never trained on.

**Document corpus (document role):** the M9 pool, 6.15M documents, re-screened as above. FineWeb
documents are excluded (decision 5).

**Mix:** by *example*, decided by screen family B; default **75% query-role / 25% document-role**
examples per step. Query-role examples get raw bytes (prompt policy (b)); document-role examples
carry M9's fixed document-role marker; teacher targets use the s2p template for queries and raw
bytes for documents. The same student input never maps to two teacher targets.

**Hard candidates for phase 2:** a fixed bank of 1M pool documents (seed 0). For every query text,
the teacher's top-64 bank documents **excluding the query's own seed document**, plus 64 bank
documents drawn uniformly per step. Mining is exact brute force on frozen stella fp16 vectors
(4.5M × 1M × 1024 ≈ 9.2e15 FLOP; on the RTX 3080 that is minutes at tensor-core rates and is
**measured on a 10K-query smoke before the full pass**); if the measured full pass exceeds 4 h,
the registered fallback is Qdrant HNSW over the bank with recall@64 audited against exact
top-64 on the smoke sample (≥ 0.98 required) and the method recorded in the manifest.

## Recipe (defaults; screen families decide the marked items)

- **Student:** bge-small-en-v1.5 [family F]. **Feature [family G]:** masked mean-pooled hidden
  states of layers 12, 8 and 4 concatenated (1152-d; MiniLM-L6: layers 6, 4, 2) → Linear(1152→1024)
  → L2 normalize; +0.8M parameters (34.2M total for bge-small). The head is warm-started in closed
  form (ridge on the concatenated features) for the bge-small init; the M9-candidate init keeps its
  384-d head and adds zero-initialized columns for the two extra layers. Exported per token so
  fastembed's mean pooling reproduces the pooled output exactly (M10.0-a2).
- **Phase 1:** plain L2 on unit-norm teacher vectors, fp32 loss, bf16 autocast (M9 form).
- **Phase 2 [family D]:** cycle 3 continues from the phase-1 checkpoint with
  L2 + λ·KL( softmax(t·Dᵀ/τ) ‖ softmax(s·Dᵀ/τ) ) over the 128 candidates per query-role example;
  document-role examples keep L2 only. **λ = 1.0.** **τ rule (locked at M10.0-e, executed once
  after the M10.1 manifest is immutable):** draw 10,000 training queries with seed 0, equal thirds
  from real / PAQ / synthetic; for τ ∈ {0.01, 0.02, 0.05} compute the teacher's softmax over each
  query's 128 candidates and its effective support exp(H); choose the τ whose median exp(H) lies
  in [8, 16]; ties → the smaller τ; none in range → τ = 0.02, disclosed. Recorded with the entropy
  table. Neither λ nor τ is tuned on any selection surface.
- **Optimizer:** AdamW β=(0.9, 0.999), eps 1e-8, wd 0.01 on dim>1, clip 1.0. **Batch 32 examples**
  [family E].
- **Schedule:** 3 cycles of equal example count, each linear 1e-4→1e-5; 2,000 warmup steps in
  cycle 1. Evaluation at every cycle end (annealed) and at cycle midpoints (curve watch only).
- **Init [family C]:** bge-small (default) or the M9 candidate.
- **Dose:** **50M examples** registered. Tokens follow the mix: at 75/25, 37.5M × ~35 + 12.5M × ~230
  ≈ **4.2B tokens**; at 50/50 ≈ 6.6B. At M9's measured mixed rate of 18,984 tok/s that is
  **2.6–4 days on the RTX 3080** before any batch-32 throughput penalty, which family E measures
  and the lock records as the build's wall-clock budget. **Extension:** one more cycle (+16.7M
  examples) if the last cycle-end improved the best COV macro by ≥ 0.003; at most two; integer cap
  83.4M examples.
- **Kill:** non-finite loss/grad; two consecutive cycle-end evaluations more than 0.0056 below the
  best. **Plateau** is read best-to-best on annealed checkpoints only.
- **Seeds:** one shipping seed for the build; confirmation seeds at screen dose per §Screen.
  Full-dose replicas stay waived unless Dylan reinstates them.

## Screen — seven families, thirteen contrasts, locked at M10.0-e

**Screen dose = 2.5M examples** (5% of the build; ≈ 209M tokens ≈ 3 h per arm at 75/25 and
18,984 tok/s), full 3-cycle schedule compressed to that dose, one seed, identical evaluation.
Throughput is recorded for every arm and decides nothing. Order: A, B, C, D, E, F, G.
**Anchor** = the full M10 corpus (A3's data), mix 75/25, bge-small init, L2 only, bs 32, 1152-d feature.

| family | arms | contrasts | rule and default |
|---|---|---|---|
| **A — data (the thesis)** | A1: M9 pool (463K unique texts) · A2: M9 pool + **4.037M PAQ = 4.5M unique texts** (factoid forms only — the volume control) · A3: M9 pool + 1.0M PAQ + 3.0M synthetic = 4.5M unique texts (form breadth at matched volume) | A3−A2 (forms at equal volume) · A3−A1 · A2−A1 (volume) | **Three registered outcomes on A3−A2:** corrected lower bound > MDE 0.0056 → coverage **resolved on the COV families**, build proceeds; point ≥ MDE and lower bound > 0 but ≤ MDE → **positive, not resolved**, build proceeds and the report says so; otherwise → **M10 stops before any build and returns to Dylan with all three rows**. A2−A1 is reported as the volume effect; if it resolves, the build keeps both volume and forms |
| **B — mix** | 100/0 · 75/25 · 50/50 query/document, **matched query presentations** (1.875M query examples in every arm; document examples 0 / 0.625M / 1.875M on top; totals 1.875M / 2.5M / 3.75M; the document cost in tokens and hours is reported) | 75/25−100/0 · 50/50−100/0 · 50/50−75/25 | resolved winner; default 75/25 |
| **C — init** | bge-small · M9 candidate | 1 | resolved winner; default bge-small |
| **D — objective** | anchor · anchor with phase-2 KL in cycle 3 | 1 | resolved winner; default phase 1 only |
| **E — batch** | 32 · 128 at equal examples and identical schedule | 1 | resolved winner; default 32 (LEAF) |
| **F — student** | bge-small · MiniLM-L6-v2 at equal examples | 1 | resolved winner; default bge-small |
| **G — output width** | feature = last layer only (384, M9's head) · last two of the three layers (768) · three layers (1152) | 1152−384 · 1152−768 · 768−384 | resolved winner; **default 1152** (the probe's evidence that width binds under L2, PLANNING §9; the screen, not the probe, decides) |

A2 exists only as a control; the build never uses more than 1.0M PAQ. **Equal examples** holds for
every family except B, which is matched on query presentations by design.

**Rule, per contrast (families B–G):** the difference in COV macro (family-weighted, §Surfaces)
between the two arms' final checkpoints; paired stratified bootstrap over queries within component,
B = 20,000, seed 0; a contrast **resolves** when the point estimate ≥ MDE 0.0056 **and** the
one-sided lower bound at the **0.025/13 quantile** (Bonferroni over the thirteen contrasts) is > 0,
and the sign is stable across the last two cycle-end checkpoints. **Family A's contrast A3−A2 is
exempt from this rule and uses only its three-outcome rule in the table** (resolved requires the
corrected lower bound > MDE); A3−A1 and A2−A1 are descriptive. **Confirmation:** for every decision whose
non-default option won, both the winner and the default are re-trained with two more seeds at
screen dose (at most four such decisions, largest margins first; the rest revert to default);
the decision stands only if the winner's margin exceeds the largest seed range observed in
either arm. Worst-case confirmation cost (B's 50/50 arm winning plus three other decisions) is
≈ 3.9B tokens ≈ 2.4 days, plus the synthesized selected-recipe run (≤ 0.5B); PLANNING §5 has the
arithmetic. Every screen verdict is artifact-specific at screen dose; never "resolved" in the
report's sense.

## Surfaces

- **COV** — the primary selection surface: licensed, decontaminated, **qrel-bearing** retrieval
  components, admitted at M10.0-d, **weighted equally per family** (slices within a family are
  averaged first). Candidate families and components, each requiring the admission record above:
  **forum-technical** — cqadup-programmers, cqadup-physics (M7 dev, CC BY-SA 2014 dump, eval-only);
  **consumer-health** — MTEB MedicalQARetrieval (questions over NIH consumer-health pages);
  **long technical questions** — BRIGHT slices biology, earth-science, psychology, robotics,
  sustainable-living; **economics** — BRIGHT economics; **legal** — MTEB LegalBench
  consumer-contracts-QA and corporate-lobbying. **At least four families must survive admission**
  or M10 returns to Dylan. COV contains no scientific-claim, paper-title or argument retrieval —
  no licensed, non-contaminating, qrel-bearing set exists for those forms — so **family A's verdict
  is a verdict about coverage on the COV families**, and those three forms are tested only by the
  six-set transaction (FORMS-12 reports them descriptively before that).
- **DEV-6** secondary, reported beside every COV read. SCREEN-3 is retired.
- **FORMS-12**: 12 × 500 held-out synthetic queries, overlap@10 between student and teacher
  rankings over the 1M bank, per form. **Descriptive only** — teacher agreement on generated
  queries is a coverage diagnostic, not retrieval quality.
- **LoTTE-clean** (7 slices, macro over slices): **read #1** after the recipe lock. Before LoTTE
  opens, the **selected recipe is trained once as a single synthesized arm at screen dose** (its
  checkpoint hash committed); read #1 scores that checkpoint and the anchor's in one atomic batch.
  Veto rule (M9 §7, unchanged): the selection is vetoed if the selected recipe's 7-slice macro is
  worse than the anchor's by more than 0.004 AND the one-sided 97.5% paired-bootstrap upper bound
  (B = 10,000, seed 903, paired within slice) on (selected − anchor) is below −0.004; a veto means
  **the anchor recipe builds**. No other action may follow from read #1. **Read #2** pre-freeze,
  audit only. No third read.
- **Dev→six calibration**: M9's close-out rows (available only after M10.2) and M7's; forecasting
  only, never gating.
- Dev reuse is counted (`m8src/dev_reuse_m8.py`) and published.

## Final run

`m9/FINAL_LOCK.md` re-registered verbatim for M10 in the M10.2 lock with new BOUND-AT-FREEZE hashes
and an `m10-six-spent` tag — bridge as phase 1, C1/C2 with the empirical 0.0125-quantile bootstrap
bound and the Holm sign-flip conjunct, NDO-4 descriptive, FEVER labelled — **with one change: the
reserved conditional is `if C1 or C2 passes then execute`**, so an aim claim never stands without
its descriptive reserved rows. Zero alpha on the reserved batch is unchanged. `m9src/final9.py`'s
scoring path is written and reviewed before M9's close-out and reused.

## Costs

M9's cost protocol (`instructions-m9.md` §Costs) unchanged; the frontier is reported per index
configuration, naming the one measured (`m9/RESULTS.md` rounds 1–4). TurboQuant 4-bit (Qdrant 1.19)
joins the M11 all-in quantization comparison.

## Unimpeachable by competitors — what the report must carry

1. Comparators byte-verified against the official artifacts, scored in the same exact-search
   harness, frozen as per-query vectors before any M10 number existed; the bridge check proves
   the harness unchanged.
2. Pre-registration with pushed commits as the external witness; one six-set transaction per
   milestone, each disclosed (M7, M9 close-out, M10); the reserved four's single access; dev-reuse
   count and the selection-surface table published.
3. Contamination handled three ways: stella's disclosed overlap (ArguAna, FiQA) at every headline;
   NDO-4 rows; reserved NDO-3 and LoTTE as surfaces no decision touched.
4. Training data affirmatively licensed, no MS MARCO (LEAF trained on MS MARCO queries; the report
   states our stack, not theirs), attribution recorded; synthetic queries with a permissive
   generator, per-query provenance, and removal counts per screen.
5. Compute disclosed in examples, tokens and GPU-hours beside LEAF's ~100 A100-hours.
6. System-level framing only (§Goal): no claim isolates nano from its document tower.
7. Reproducibility: code, corpus manifests with hashes, model revisions, seeds, statistics code
   with tests, and the full screen table including losing arms and confirmation seeds.

## Deliverables

Frozen candidate + `m10/FREEZE.json` (`assert_releasable` with a proper run record), the frontier
update, the M10 section of the report artifact, decisions logged in CLAUDE.md, handoff to M11.

## Out of scope (reopening conditions in PLANNING §7)

Document-side co-adaptation (breaks the pair; M11+ as its own system) · a >35M nano (decision 6
scopes a separate tier only) · teacher change (stella-1.5B measured worse; Qwen3-0.6B never
screened and not the pair) · a nonlinear head (no fastembed path; the width comes from linear multi-layer pooling instead) · MS MARCO in any form ·
FineWeb documents (decision 5) · any change to zero.
