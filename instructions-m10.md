# M10 — nano v2: the retry, built around coverage (mandate)

*Written 2026-09-01 by the planning session on Dylan's direction ("M9 failed to achieve our goals
… make M10 a retry … build something unimpeachable by competitors"). Evidence `m10/PLANNING.md`;
M9's record `m9/FINDINGS.md`. Adversarial review: gpt-5.6-terra, read-only, six passes
(`research/m10-codex-plan-2026-09-01.md`, `-plan2-` … `-plan6-`; full logs are gitignored `.log`
files beside them); every finding and its disposition is in PLANNING §8, including two reviewer
dissents the owner can overrule. **Amended 2026-09-01 after pass 6 on Dylan's compute ruling:** M10
runs on a rented GPU budget or not at all (§Owner rulings, §Compute); the review amendments taken
with it are decision 9 and PLANNING §8's amendment block. The M9 model is **nano**; M7's table is
**zero**; the product is still the pair on one stella index.*

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

M9 retained **93.8%** of the teacher on NQ and **50–71%** on the two CQADupStack dev components
(`m9/FINDINGS.md` §1): its 463K Wikipedia-QA and product queries did not cover the forms the model
is tested on, and its 384-wide linear head is the subspace L2 regression cannot push past once
queries are diverse (PLANNING §9). Parameter count is not the constraint. Both causes are tested
before anything is built (§Screen, families A and G). M10 changes, in order of evidence (PLANNING §3):

1. **Coverage** — ~4.5M query texts across 12 forms (§Data).
2. **Optimizer regime** — LEAF's batch 32 and three linear 1e-4→1e-5 cycles; plateau read on
   annealed checkpoints, best-to-best.
3. **A ranking-aware phase-2 loss**, registered here and priced by a screen arm, never symptom-gated.
4. **A wider linear output** — three mean-pooled layers concatenated (1152-d) before the head.
5. **Warm start from the M9 candidate** as a screen arm.

**The student cap is 35M and it is hard** (Dylan, 2026-09-01: "109M is not an option. This isn't
low compute anymore. 33M was already in the upper bound of what I think is acceptable"). bge-small
with the three-layer head is 34.5M; MiniLM-L6 with it is 23.9M. The cheaper student gets the
benefit of a tie (§Screen, family F).

## Owner rulings already made (Dylan, 2026-09-01)

- **Git:** M9 was merged to `main` on 2026-09-01 after the repo cleanup (its six-set close-out is
  still pending and runs from `m9-work`); M10 execution work happens on branch **`m10-work`**
  under the headless commit-and-push contract; merges to main at stage boundaries
  need Dylan's go. M9's registered six-set close-out still runs from `m9-work`, because `guard9`
  pins that branch (`m9src/guard9.py:35`); the branch is kept until then.
- **Compute (Dylan, 2026-09-01): "M10 won't be done on a 3080. M10 will be done on a GPU budget, if
  allowed, or not at all."** Every M10 stage from M10.0-c on runs on a rented GPU under the budget in
  §Compute, or M10 does not run. The RTX 3080 is not an execution target for any M10 step; it only
  supplies M9's frozen checkpoint once (§Compute). Every dose and duration here is set from the
  budget, not from the box.

## Owner decisions (defaults apply until Dylan rules; each is recorded in `m10/LEDGER.md`)

| # | decision | default while open |
|---|---|---|
| 1 | Ratify M9's final-lock amendment **together with the close-out amendment that strikes M9's reserved conditional** (§Stage plan, M10.2): M9's close-out is six-only and cannot spend the reserved access | blocks the close-out only |
| 2 | **GPU budget approval** (§Compute): expected **$400–715** with generation on the GPU (239–269 GPU-hours) or **$465–895** if generation is hosted (209 GPU-hours plus $110–330), hard ceiling **$1,000**, itemized in PLANNING §6; prices unverified Sept 2026. Refusal closes M10 unstarted | no GPU stage runs until approved; the Mac list in `m10/STATUS.md` continues |
| 3 | FineWeb as a seed — **ruled out 2026-09-01** (delegated): Wikipedia and the approved corpora carry the topics; FineWeb adds a rights review and a blocklist for no measured gain. Reopening condition in `m10/EXPLORED.md` | closed |
| 4 | PAQ (machine-generated questions over Wikipedia; data CC BY-SA, generation code CC BY-NC) as query text | include, from Facebook's official release (never the unofficial HF mirror); 1.0M uniform sample in the build (seed 0, file hashes pinned), 4.037M in the volume-control screen arm A2 only; attribution recorded |
| 5 | FineWeb documents — **excluded 2026-09-01**: no reserved-set document fingerprints exist and creating them would open reserved corpora (`m9/LEDGER.md` §1.3) | closed |
| 6 | A >35M tier — **ruled out by Dylan 2026-09-01**; 35M is a hard cap | closed |
| 7 | Confirm: LoTTE read #1 withdrawn unexecuted in M9; renumbering M10/M11/M12 | as recorded in `m9/STATUS.md` and CLAUDE.md |
| 8 | Second build seed: the identical recipe and data at seed 1, as the headline's replication band (`m7/FINDINGS.md` 9); ≈ 100 GPU-hours ≈ $150–250 | runs after the seed-0 build only if ≥ 100 GPU-hours at the billed price remain under the ceiling after every mandatory line of PLANNING §6. The lock fixes, before any seed-0 six-set output exists, a seed-1 boolean, its data and model seeds, and its six-set row labels. **Seed 0 alone controls C1/C2, release, recipe and headline; seed 1's rows are descriptive inside the same transaction and can trigger no action** |
| 9 | The 2026-09-01 review amendments taken with the compute ruling: dose 200M examples, screen dose 5M, arms D-KL1 / D-NCE / G-1536, the COV resolution number, the seed-rank field, extension capped by budget (PLANNING §8, amendment block and pass 7) | adopted; any item Dylan strikes reverts to the pass-6 text |

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
  (a) Mac diagnostics — **done 2026-09-01**, `m10/RESULTS.md`, PLANNING §9–9c: the rank probe
  (a 384-d subspace keeps 99.5% of one query distribution and 90–93% of three; 98–100% at 640 —
  evidence about the class under L2 regression, not a bound), the head-width probe (a frozen
  bge-small retains more with each pooled layer), and the serving-parity check (fastembed 0.8.0
  reproduces the three-layer per-token head to 2e-7; 34.5M parameters). **Action taken:** the head
  stays linear; the pooled feature widens to three layers (§Recipe); family G decides. Repeat the
  parity check on MiniLM's head before family F may select it, on the four-layer head before G's
  1536 arm runs, and on the trained artifact with M9's locked parity sample before the freeze.
  (b) Capacity probe (`m9src/capacity_probe.py`, unchanged) — **optional, report-only**: no outcome
  changes an M10 action under the hard cap; runs only on an idle GPU; its 109M student is
  768-hidden, so any gain is partly width.
  (c) **[GPU; gated on decision 2 — nothing from here on runs before the budget is approved, and
  nothing scales before the day-one rate benchmark in §Compute has re-derived PLANNING §6]**
  Per-component DEV-6 read of the M9 candidate incl. `heldout-longq` (the baseline row).
  (d) **COV admission** (§Surfaces): for every candidate component, record in `m10/LEDGER.md` §2
  its primary-source licence URL and terms, HF repo and revision, corpus size, query count, qrels
  format and metric, its corpus-level contamination check and its fingerprint screen against the
  six and the reserved four. A component is named COV only after that record is pushed. **COV
  admits only surfaces no M10 decision has read**: the two CQADupStack dev components were scored
  by the Mac diagnostics (PLANNING §9, 86 raw reads) and stay in DEV-6. Then every admitted COV
  corpus, query set and document set joins the protected index (`m8src/protected_filter`) before
  any seed is drawn or any PAQ or synthetic text is constructed. The **COV resolution number**
  (§Surfaces) is measured on the admitted surface and pushed before (e); it decides nothing.
  (e) **Screen lock**: `m10/LEDGER.md` §0 (skeleton committed 2026-09-01) fixes every arm of
  §Screen (fourteen arms), order, doses, seeds, the τ rule, surfaces, the sixteen contrasts,
  multiplicity control, confirmation design and outcome→action maps.
- **M10.1 DATA.** Generation under the §Data contract (200-query smoke per form first), PAQ
  samples, decontamination against the protected index (now including COV) and the six's
  documents, the FORMS-12 hold-out, teacher targets, hard-candidate mining (§Data),
  `results/m10_data_manifest.json` with hashes and the provenance table. **After the manifest is
  immutable and before any arm**, the τ rule is executed and its table recorded.
- **M10.2 SCREEN + RECIPE LOCK.** The arms of §Screen, the confirmation runs, then one pushed lock
  commit with every field of M9's M9.2 list filled — including the phase-2 loss, the best-to-best
  plateau/extension rule on annealed checkpoints, the GPU-hour allocation under the ceiling (build,
  extensions, decision 8), the final-run registry and LoTTE read #1's manifest. Codex and Fable review the pushed lock. **Then, and only then,** M9's close-out runs:
  its registered six-set transaction **amended before execution to six-only** (the `if C1 then
  execute` reserved conditional is struck; disclosed and ratified under decision 1), so M9 cannot
  spend the reserved access. Its rows are a forecasting calibration and a whitepaper frontier
  point; nothing in M10 may change on them. **Wherever those rows are published — frontier table,
  whitepaper, model card — they carry M9's build-provenance disclosure with them:** the M9
  candidate's provenance spans TWO build lock states (mid-build infrastructure repair, session
  force-reopened; `m9/LEDGER.md` "M9.3 BUILD PROVENANCE DISCLOSURE" and `m9/BUILD_LOG.md`), and it
  may never be described as an uninterrupted single-lock run. LoTTE read #1 then runs as the
  registered veto on the selected recipe.
- **M10.3 BUILD.** One candidate under the kill and extension rules; export + parity; freeze;
  pre-freeze review; LoTTE read #2 (audit only); then the seed-1 replica if decision 8's condition
  holds (descriptive, never a choice).
- **M10.4 FINAL.** The six-set transaction → decision → the registered reserved conditional.

## Data

**Query corpus (~4.5M unique texts, query role):**

| source | build count | licence | role |
|---|---|---|---|
| M9 real queries (hotpotqa, squad, esci, mrtydi, nqopen, triviaqa; fever out) | 463K | CC BY-SA / Apache | real forms |
| PAQ sample (decision 4) | 1.0M | CC BY-SA (data) | factoid volume — capped so it cannot dominate |
| **Synthetic, Qwen/Qwen3-8B (Apache-2.0; revision `b968826d…`), bf16 via vLLM on the rented GPU; registered fallback: hosted open-weights inference of the same revision if the smoke's end-to-end projection of the full generation job exceeds 60 GPU-hours (PLANNING §5)** | 3.0M | generated under the generator's terms; provenance pinned; **not redistributed** without review | form breadth |

**Form taxonomy — 12 forms, 250K each** (quotas locked at M10.1; ±10% realized): factoid question ·
how-to / troubleshooting question with title and body · scientific claim (a statement) · long
counter-argument paragraph (120–220 words) · finance / personal-economics question · paper-title
query · 2–4-word keyword query · consumer-health question · product-search query · comparison
question · yes/no verification question · conversational multi-sentence request. **Seeds:**
Wikipedia stratified by top-level category (CC BY-SA) and the approved pool corpora. FineWeb is
out of M10 (decisions 3 and 5). **Contaminating source families — never seeds, never regression text, and
excluded from COV** (`research/m7-data-licensing.md` map): S2ORC / Semantic Scholar; PubMed;
NutritionFacts.org and its mirrors; CORD-19; StackExchange personal finance (money.SE) and Reddit
finance; args.me / idebate; every six-set and reserved corpus.

**Generation contract (M10.1):** generator pinned by HF repo + revision in the manifest (bf16
weights; the hosted provider and its served revision recorded if the fallback fires); vLLM sampling temperature
0.8, top-p 0.95, `max_new_tokens` per form (60, or 400 for the argument and conversational
forms), `seed = blake2b-64(seed_passage_id)`; the reply must parse as one JSON list of exactly n
strings (`m10src/forms.parse`, strict — no preamble); one retry on a contract failure, then the
seed is dropped; exact-duplicate queries removed. **Smoke:** 200 queries per form, read by Dylan,
who is the approver; a form passes when ≥ 90% of replies meet the contract and ≥ 80% of a
50-query sample are judged on-form; a failing form's prompt may be revised at most twice, each
revision recorded in `m10/LEDGER.md` §1 before the next smoke. **Seeds are pre-filtered:** a seed
passage that exact- or near-matches the protected index is never used.

**Screens on every generated or PAQ query, thresholds fixed here (M7's fingerprints,
`m7src/decontam.py`):** exact `blake2b-64` match or word-8-gram bottom-32 sketch ≥ 8/32 against
(i) the protected index (six + dev + reserved + LoTTE **+ admitted COV queries and documents**),
(ii) the six's documents; (iii) any word-5-gram shared with the query's own seed passage (a
copied span is not a query); word-4-gram containment for 4–7-word queries. The M9 real-query pool
and the document pool are re-screened against the COV additions (R1 removes matching queries;
matching pool documents are
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
the teacher's top-64 bank documents **excluding the query's own seed document** (real and PAQ
queries have none; nothing is excluded), plus 64 bank documents drawn uniformly per step with
`numpy.default_rng(904 + step)`. Mining is exact brute force on frozen stella fp16 vectors
(4.5M × 1M × 1024 ≈ 9.2e15 FLOP; minutes on an A100 at tensor-core rates, and
**measured on a 10K-query smoke before the full pass**); if the measured full pass exceeds 4 h,
the registered fallback is Qdrant HNSW over the bank with recall@64 audited against exact
top-64 on the smoke sample (≥ 0.98 required) and the method recorded in the manifest. For every
synthetic query the manifest also records the teacher's rank of its seed passage against the 64
mined candidates (1 = above all of them, 65 = below all). Provenance only, read by no M10 rule; it
exists so a round-trip quality filter can be registered at M10.1 without a second generation pass.

## Recipe (defaults; screen families decide the marked items)

- **Student:** bge-small-en-v1.5 is the **screen anchor** (every non-F family is screened on it);
  the **build student is decided by family F**, MiniLM-L6-v2 by default. Before F can select
  MiniLM, its three-layer head passes the same export and fastembed parity check as bge-small's
  (M10.0-a; the Mac can run it). **Feature [family G]:** masked mean-pooled hidden states of
  layers 12, 8 and 4 concatenated (1152-d; MiniLM-L6: layers 6, 4, 2) → Linear(1152→1024) → L2
  normalize; head 1.18M parameters, 34.5M total for bge-small. G's fourth arm adds layer 2
  (MiniLM-L6: layer 1) → 1536-d, head 1.57M, 34.9M total, under the cap; its head passes the same
  parity check first. Warm-started in closed form (ridge)
  for the bge-small init; the M9-candidate init keeps its 384-d head and zero-initializes the two
  extra layers' columns. Exported per token so fastembed's mean pooling reproduces it exactly.
- **Phase 1:** plain L2 on unit-norm teacher vectors, fp32 loss, bf16 autocast (M9 form).
- **Phase 2 [family D]:** cycle 3 continues from the phase-1 checkpoint with
  L2 + λ·KL( softmax(t·Dᵀ/τ) ‖ softmax(s·Dᵀ/τ) ) over the 128 candidates per query-role example;
  document-role examples keep L2 only. **λ = 1.0.** Family D also prices two from-the-start
  variants at equal examples. **D-KL1:** the same L2 + λ·KL from the first step of cycle 1. **D-NCE**
  (synthetic query-role examples only): with s the student's unit-norm output, p the frozen unit-norm
  stella vector of the query's seed passage encoded in the document role (raw bytes, as the index
  encodes documents), and d₁…d₁₂₈ the frozen unit-norm vectors of the query's 128 candidates (which
  already exclude the seed), the term is the cross-entropy of the 129-way softmax over
  [s·p/τ, s·d₁/τ, …, s·d₁₂₈/τ] with the positive as the target, averaged over the synthetic
  query-role examples in the batch; loss = L2 + λ·NCE, λ = 1.0; τ is the τ-rule value reused
  unchanged (the rule is calibrated on the 128-candidate teacher softmax and is not re-run for the
  129-logit set — disclosed). A synthetic example whose seed passage a screen removed contributes L2
  only and is counted in the manifest; real and PAQ queries and document-role examples keep L2 in
  that arm. **τ rule (locked at M10.0-e, executed once
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
- **Dose:** **200M examples** registered — LEAF's dose (6.7M texts × 30 epochs ≈ 201M; PLANNING §5),
  three cycles of 66.7M. Tokens follow the mix: at 75/25, 150M × ~35 + 50M × ~230 ≈ **16.8B**; at
  50/50 ≈ 26.5B. Query epochs ≈ 33 over 4.5M texts, document epochs ≈ 8 over the 6.15M pool. At the
  planning rate of 560 examples/s (§Compute) that is ≈ **100 GPU-hours**; the first screen arm's
  measured rate replaces it, and family E's batch-32 penalty is recorded in the lock as the build's
  GPU-hour line. **Extension:** let m_k be the COV macro (full precision, the locked formula and
  evaluation hashes) at the end of cycle k. After every cycle k ≥ 3, one more cycle of 66.7M examples
  (linear 1e-4→1e-5, as cycle 3) starts iff m_k − max(m₁, …, m_{k−1}) ≥ 0.003 **and** the extension
  cycles already run are fewer than `max_extension_cycles`, an integer the lock fixes from the
  approved dollars minus every mandatory line of PLANNING §6 at the measured rates and the billed
  price (§Compute). Whole cycles only; a cycle whose projected cost plus billed spend to date would
  exceed the ceiling does not start. The lock records m_k's formula, the evaluation hashes,
  `max_extension_cycles`, and the spend source.
- **Kill:** non-finite loss/grad; two consecutive cycle-end evaluations more than 0.0056 below the
  best. **Plateau** is read best-to-best on annealed checkpoints only.
- **Seeds:** one shipping seed for the build (seed 0); confirmation seeds at screen dose per §Screen.
  A full-dose seed-1 replica runs only under decision 8 and is descriptive.

## Screen — seven families, sixteen contrasts, locked at M10.0-e

**Screen dose = 5M examples** (2.5% of the build; ≈ 420M tokens at 75/25; ≈ 2.5 GPU-hours per arm
at 560 examples/s), full 3-cycle schedule compressed to that dose, one seed, identical evaluation:
**COV at every cycle end; DEV-6 once, at the final checkpoint** (its 5.2M-document hotpotqa and
6.17M-row heldout components cost ~13 GB of reads per pass — M9's practice). Throughput is
recorded for every arm and decides nothing. Order: A, B, C, D, E, F, G.
**Anchor** = the full M10 corpus (A3's data), mix 75/25, bge-small init, L2 only, bs 32, 1152-d feature.

| family | arms | contrasts | rule and default |
|---|---|---|---|
| **A — data (the thesis)** | A1: M9 pool (463,314 queries) · A2: M9 pool + PAQ (factoid forms only — the volume control) · A3: M9 pool + 1.0M PAQ + 3.0M synthetic (form breadth). **A2 and A3 are cut to the identical post-screen unique-text count** (the smaller of the two after decontamination, the larger downsampled with seed 0) and both hashes are locked before any arm | A3−A2 (forms at equal volume) · A3−A1 · A2−A1 (volume) | **Three registered outcomes on A3−A2:** corrected lower bound > MDE 0.0056 → coverage **resolved on the COV families**, build proceeds; point ≥ MDE and lower bound > 0 but ≤ MDE → **positive, not resolved**, build proceeds and the report says so; otherwise → **M10 stops before any build and returns to Dylan with all three rows**. A2−A1 is reported as the volume effect; if it resolves, the build keeps both volume and forms |
| **B — mix** | 100/0 · 75/25 · 50/50 query/document, **matched query presentations** (3.75M query examples in every arm; document examples 0 / 1.25M / 3.75M on top; totals 3.75M / 5M / 7.5M; the document cost in tokens and GPU-hours is reported) | 75/25−100/0 · 50/50−100/0 · 50/50−75/25 | resolved winner; default 75/25 |
| **C — init** | bge-small · M9 candidate | 1 | resolved winner; default bge-small |
| **D — objective** | anchor · D-KL3: phase-2 KL in cycle 3 · D-KL1: L2 + KL from cycle 1 · D-NCE: L2 + InfoNCE on the seed passage from cycle 1 (§Recipe) | D-KL3−anchor · D-KL1−anchor · D-NCE−anchor | resolved winner with the largest margin; default phase 1 only |
| **E — batch** | 32 · 128 at equal examples and identical schedule | 1 | resolved winner; default 32 (LEAF) |
| **F — student** | bge-small (34.5M with the head) · MiniLM-L6-v2 (23.9M) at equal examples | 1 | bge-small only if it wins **resolved**; **default MiniLM-L6-v2** — owner preference for the low-compute point (2026-09-01); MiniLM is 2× cheaper to train and serve (`m9/RESULTS.md`), and M9's screen had it −0.0026 unresolved |
| **G — output width** | feature = last layer only (384, M9's head) · last two of the three layers (768) · three layers (1152) · four layers (1536, §Recipe) | 1152−384 · 1152−768 · 768−384 · 1536−1152 | resolved winner; **default 1152** (the probe's evidence that width binds under L2 and is still rising at three layers, PLANNING §9–9b; the screen, not the probe, decides) |

A2 exists only as a control; the build never uses more than 1.0M PAQ. **Equal examples** holds for
every family except B, which is matched on query presentations by design. **Definitions:** a
decision's *margin* is the COV macro difference between the winner's and the default's final
checkpoints in the original screen; an arm's *seed range* is max minus min of its COV macro over
its three seeds. If family F selects
MiniLM-L6, the other families' verdicts (taken on bge-small) transfer to it by assumption, and the
report says so; the synthesized selected-recipe arm and LoTTE read #1 are the check on that transfer.

**Rule, per contrast (families B–G):** the difference in COV macro (family-weighted, §Surfaces)
between the two arms' final checkpoints; paired stratified bootstrap over queries within component,
B = 20,000, seed 0; a contrast **resolves** when the point estimate ≥ MDE 0.0056 **and** the
one-sided lower bound at the **0.025/16 quantile** (Bonferroni over the sixteen contrasts) is > 0,
and the sign is stable across the last two cycle-end checkpoints. **Family A's contrast A3−A2 is
exempt from this rule and uses only its three-outcome rule in the table** (resolved requires the
corrected lower bound > MDE); A3−A1 and A2−A1 are descriptive. **Confirmation:** for every decision whose
non-default option won, both the winner and the default are re-trained with two more seeds at
screen dose (at most four such decisions, largest margins first; the rest revert to default);
the decision stands only if the winner's margin exceeds the largest seed range observed in
either arm. Worst-case confirmation cost (B's 50/50 arm winning plus three other decisions) is
85M examples ≈ 47 GPU-hours with its 16 evaluations, plus the synthesized selected-recipe run (5M);
PLANNING §5 has the arithmetic. Every screen verdict is artifact-specific at screen dose; never "resolved" in the
report's sense.

## Surfaces

- **COV** — the primary selection surface: **qrel-bearing** retrieval components with an
  affirmative licence at the dataset's primary source (M7's eval-use standard), decontaminated,
  and **read by no M10 decision**, admitted at M10.0-d, **weighted equally per family**
  (slices within a family averaged first). Candidates, draft records in `m10/COV_CANDIDATES.md`
  (primary-source licences checked 2026-09-01): **consumer-health** MTEB MedicalQARetrieval (CC BY
  4.0 at MedQuAD; NIH sources); **BRIGHT**, one family, six slices (biology, earth-science,
  economics, psychology, robotics, sustainable-living; CC BY 4.0 on the benchmark at its primary
  source — the same dataset-level standard that admitted CQADupStack and the six, whose documents
  are also third-party text; the document-rights caveat is disclosed and the data is evaluation-only,
  never redistributed); **legal** MTEB LegalBenchCorporateLobbying (CC BY 4.0; ConsumerContractsQA
  refused, CC BY-NC); **finance** LEDGER (CC BY 4.0 annual-report QA) once its structure and a
  100K-chunk cap are verified. Climate-FEVER refused (no licence at its primary source, as in M7).
  **At least three families must survive admission** or M10 returns to Dylan; the report names the
  family count (Codex pass 6 preferred four — with the CQADupStack pair demoted, four needs LEDGER;
  Dylan may raise the floor). The two CQADupStack components are **DEV**, reported beside every COV
  read, never in the macro. COV contains no
  scientific-claim, paper-title or argument retrieval (no licensed, non-contaminating set exists),
  so **family A's verdict is a verdict about coverage on the COV families**; those three forms are
  tested only by the six-set transaction (FORMS-12 reports them descriptively before that).
  **Resolution number (M10.0-d, before the lock; descriptive, decides nothing):** with the contrast
  rule's own bootstrap (paired, stratified within component, B = 20,000, seed 0), measure the
  distance between the point estimate and the one-sided 0.025/16 lower bound for the COV-macro
  difference between two frozen comparators that are candidates in no M10 family — **e5-small-v2 and
  gte-small** (both `results/FINAL_MATRIX.md` rows) — each scored symmetrically on the admitted
  surface. Only the distance is recorded, never which comparator led. It is the first disclosed COV
  read (`m10/LEDGER.md` §4) and is published beside every contrast, so a reader can tell an unresolved
  verdict from an invisible one. No rule reads it; the screen runs all fourteen arms regardless.
- **DEV-6** (incl. the two CQADupStack components) secondary, reported beside every COV read.
  SCREEN-3 is retired.
- **FORMS-12**: 12 × 500 held-out synthetic queries, overlap@10 between student and teacher
  rankings over the 1M bank, per form. **Descriptive only** — teacher agreement on generated
  queries is a coverage diagnostic, not retrieval quality.
- **LoTTE-clean** (7 slices, macro over slices; its corpora — ~2.8M passages — are encoded with
  stella once, ≈ 1.3 GPU-hours at the assumed 600 docs/s, budgeted in PLANNING §6): **read #1**
  after the recipe lock.
  Before LoTTE opens, the **selected recipe is trained once as a single synthesized arm at screen
  dose** (its checkpoint hash committed); read #1 scores that checkpoint and the anchor's in one atomic batch.
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

## Compute and costs

**M10 runs on one rented A100 80 GB** (an H100 80 GB if its cost per example measures lower on the
smoke), ≥ 500 GB persistent disk that survives stopping the instance, SSH, and a GitHub deploy key so
the headless commit-and-push contract holds. Provider is Dylan's choice; $1.5–2.5/h assumed,
unverified Sept 2026. The instance is stopped between stages. **Planning rate: 560 examples/s at
batch 32** — LEAF's realized rate (201M examples in ~100 A100-hours, PLANNING §5) — and 600 stella
documents/s for encodes (3× the box's 210); both unmeasured on our stack. **Day-one rate benchmark
(≈ 2 GPU-hours, before any scale-up):** stella docs/s on 20K pool documents; training examples/s at
batch 32 for 2,000 steps on each of the 75/25 anchor mix, the 50/50 mix and MiniLM-L6; generation
requests/s per form on the 200-query smoke, end to end (prefill, decode, retries, JSON failures);
the provider's billed price. **PLANNING §6 is re-derived from these and pushed before generation
starts, and again before the remaining screen arms.** Budget, from PLANNING §6: mandatory lines
**239–269 GPU-hours ≈ $400–715** with generation on the GPU, or 209 GPU-hours plus hosted generation
≈ **$465–895**; ≈ $40 of disk is inside both. Hard ceiling **$1,000**. Allocation order at the lock:
every mandatory line first at the measured rates and billed price, then the second build seed
(decision 8) if ≥ 100 GPU-hours remain, then whole extension cycles (≈ 33 GPU-hours each) from the
remainder; at the high end of either scenario nothing optional fits and the plan still completes.
Wall-clock ≈ 2.5–3 weeks on one GPU; screen arms are independent and may run on 2–4 GPUs at the same
total cost. On the instance, `results/perquery.json` (tracked) is sha-verified against `6b18e3dd…`
before any scoring, and M9's parity sample is regenerated from `m9/registry.json` (tracked; seed and
sha256 pinned) — `m9src/port.py` refuses a sample that does not hash to the lock, in which case the
512 texts are transferred from the box. **From the box, once, sha-verified:** `work/m9long/ckpt/last.pt` (M9's frozen candidate,
`9d631b2c…`) — family C, M10.0-c and M9's close-out need it — plus any file `m9src/guard9.py`
hashes that lives under `work/` (listed on the box before transfer). Everything else (pool, dev
suite, fingerprints, teacher targets, six/reserved/LoTTE/COV encodes) is re-derived from the repo
and HF (≈ 12 GPU-hours plus a day of CPU and network). Reserved-set encodes run only if the reserved
conditional fires.

M9's cost protocol (`instructions-m9.md` §Costs) unchanged; the frontier is reported per index
configuration, naming the one measured (`m9/RESULTS.md` rounds 1–4). TurboQuant 4-bit (Qdrant 1.19)
joins the M12 all-in quantization comparison.

## Unimpeachable by competitors — what the report must carry

1. Comparators byte-verified against the official artifacts, scored in the same exact-search
   harness, frozen as per-query vectors before any M10 number existed; the bridge check proves
   the harness unchanged.
2. Pre-registration with pushed commits as the external witness; one six-set transaction per
   milestone, each disclosed (M7, M9 close-out, M10); the reserved four's single access; dev-reuse
   count and the selection-surface table published.
3. Contamination handled three ways: stella's disclosed overlap (ArguAna, FiQA) at every headline;
   NDO-4 rows; reserved NDO-3 and LoTTE as surfaces no decision touched.
4. Training data affirmatively licensed, no MS MARCO, attribution recorded; synthetic queries from
   an open-weights generator with pinned revision, per-query provenance and removal counts.
5. Compute disclosed in examples, tokens and GPU-hours beside LEAF's ~100 A100-hours.
6. System-level framing only (§Goal): no claim isolates nano from its document tower.
7. Reproducibility: code, corpus manifests with hashes, model revisions, seeds, statistics code
   with tests, and the full screen table including losing arms and confirmation seeds.

## Deliverables

Frozen candidate + `m10/FREEZE.json` (`assert_releasable` with a proper run record — it walks
`work/runs/<id>.json` for `cfg.sources` and `cfg.init` across the whole lineage and fails CLOSED on
a missing ancestor, which is why M9's freeze was refused; the record must be **derived from the
run's own hash-bound `manifest.json`, never asserted by hand**, because a post-hoc record that
happens to satisfy a licence guard is precisely the artifact that must not be fabricated; see
`m9/BUILD_LOG.md`), the frontier
update, the M10 section of the report artifact, decisions logged in CLAUDE.md, handoff to M12.

## Out of scope (reopening conditions in PLANNING §7)

Document-side co-adaptation (inside M10 it breaks the pair; a tower co-trained against both query paths at once keeps it and is recommended 2026-09-01 as the M12 candidate, Dylan's call, PLANNING §7) · any student above 35M
(hard cap, Dylan 2026-09-01) · teacher change (stella-1.5B measured worse; Qwen3-0.6B never
screened and not the pair) · a nonlinear head (no fastembed path; width comes from linear
multi-layer pooling) · MS MARCO in any form · FineWeb in any role (decisions 3 and 5) · any change to zero.
