# M8 plan — DRAFT v2 (2026-08-29), post first adversarial gate

**Status: DRAFT v2.** Not yet a pre-registration. v1 (git history) was gated by a fresh Codex
gpt-5.6-sol pass (`research/m8-planning/codex-plan-gate-2026-08-28.md`) which returned **STOP with
17 findings**; every finding is actioned or explicitly disposed below (§6 maps them). No protected
set has been touched; no training has run. Sources: five independent reviews in
`research/m8-planning/` plus the gate. New scope from Dylan 2026-08-28: **ONNX/fastembed
compatibility is a requirement** (§5) — eventually, not day-1; benchmark numbers still dominate.

## 0. What is frozen, what is amendable, what needs Dylan

Per the gate (finding 2), the v1 line "nothing here is frozen" was false. The boundary:

| class | items |
|---|---|
| **FROZEN** (registered in `instructions-m8.md` before M7's number existed) | The four confirmatory sets (FEVER, DBpedia-entity, cqadup-android, cqadup-english; hash-pinned, un-scored); the paired frozen-M7-vs-frozen-M8 comparison in ONE M8 access; the statistics family *shape* (Holm + raw CI + simultaneous bound, dependence-preserving, same as M7's tier rule); six-set scoring is descriptive-only, labelled "development-informed"; comparator sources on the reserved four = frozen M7 system + the frozen `fusion.bm25_run` builder + published numbers as labelled context ONLY; minimum release bar = beats frozen M7 CI-resolved on the reserved sets; licensing/decontamination rules; dev-only selection; one-access freeze/ledger protocol. |
| **AMENDABLE, but only before the first M8 number exists, in writing, with reasoning** | Macro weighting over the four; the exact hypotheses/α/family membership and extra legs; dev-suite composition; probe designs; anything in this draft not listed above. |
| **OWNER-RULING REQUIRED before it may enter the confirmatory candidate** | E1 nonlinear query head (scope change); E3 two-artifact release / doc-side head (architecture change); E5 index-time corpus adaptation of the confirmatory candidate (protocol change); E2 generator licensing; E6 training-time-only second teacher; E4 M9 reserve; E7 vocab-rule rewrite. Until ruled, D1/D3/D5 are **research rows, not candidates**. |

Consequence (gate finding 1): **v1's "freeze OpenSearch per-query vectors on the reserved four" is
DELETED.** Scoring any system on the reserved sets makes them development-visible per the mandate;
"computed and sealed" does not reverse access. On the reserved four the comparators are frozen M7,
frozen BM25, and published numbers as labelled context — the missing external anchor is stated as a
limitation, not papered over. (`m7_bars_clean4.json` is not precedent: it was arithmetic over
already-frozen six-set vectors.)

---

## 1. Corrected diagnosis — findings vs hypotheses

Verified findings (each checked against a repo artifact, several re-verified this session):

1. **Phase B lands +0.0008 above a 2-minute closed-form ridge solve; the entire trained gain over
   closed form comes from Phase A** (83 s, ~3.7 epochs over 340,850 pairs, plateaus by 2,500
   steps). `m7_stage0_ridge_stella.json`, run logs.
2. **924,704 licence-clean (query, positive) pairs are discarded**: `pseudoq.build()` holds
   `(store, doc_id)` per span and returns text only. Verified `pseudoq.py`.
3. **The pseudo pool is 86.5% ESCI+HotpotQA, first-sentence-only.** Verified `pseudoq.py:84-86`.
4. **The mined-negatives close is contradicted by the ledger**: matched-step arms +0.0112/+0.0111
   clear the bar; the loss appears only under a proxy step-selector the ledger records as ranking
   arms backwards; outcome "NOT IDENTIFIED". Verified LEDGER:335.
5. **The bigram −0.0301 close tested residual fitting toward the query-vector target on an earlier
   table; the joint closed-form solve measures +0.0101 (5K rows) / +0.0143 (10K rows), CIs
   excluding 0.** Joint retrain explicitly open. `m7_bigram_*.json`.
6. **The doc-side-map dismissal rests on a wrong sentence** (LEDGER:577 — the map applies to cached
   vectors, one GEMM, no re-encode) while the repo's own algebra shows it is non-absorbable under
   renormalization (rank agreement 0.000).
7. **Retention is inverted from the project's prior**: ArguAna (longest queries) 0.929;
   trec-covid/fiqa (short queries) 0.667/0.673. Recomputed from `m7_final_run.json`.
8. **The release container is 93,886,950 bytes, not 31.3 MB** — the exporter writes fp16 AND int8
   payloads. Verified by unzipping. M8 ships an int8-only format and reports payload vs container.
9. **M7 vs LR-dense-websearch is a statistical tie, not a win**: +0.00194 [−0.0153, +0.0195],
   re-verified this session by paired bootstrap on the frozen vectors. The honest sentence is
   "matches LR's single-table dense system at 1/15 the artifact bytes". LR-dense-**pertask**
   (0.4583) is an **instruction-oracle** comparator (per-dataset instruction tables), NOT a
   corpus-adaptation analogue — v1 misstated this; D3 is an unanchored hypothesis whose evidence
   starts at its probe.

Hypotheses (plausible, probe-gated, NOT established — gate finding 9):

- **H1 (pair starvation):** Phase A's plateau is supply-limited. G2's 0.99999 shows in-sample
  expressibility only; a 3.7-epoch plateau does not by itself separate pair supply from objective,
  sampling, or optimization. Probe B3.
- **H2 (degenerate KL):** with 31 uniform distractors at temp 0.02 the teacher target is
  effectively one-hot, so no teacher *ranking* information reaches the student. The supporting
  arithmetic uses marginal score statistics; the actual 32-way entropy distribution over sampled
  candidate sets is what B2 measures.
- **H3 (short-query loss is recoverable):** the retention inversion is consistent with a
  linearity tax on short queries. The v1 compressed-sensing "provably information-preserving"
  claim is withdrawn (RIP/incoherence never measured for this table; normalization discards
  scale); B1' and B4 measure instead of assuming.

Instrument facts that govern sizing: the dev recipe-perturbation band is 0.0027–0.0078 (a DEV
selection fact; per the ledger it does NOT deflate a frozen confirmatory comparison — gate finding
14); the reserved-4 equal-weight macro confirms Δ ≈ 0.0096 half-width for dissimilar systems,
≈ 0.005 for near-siblings. **+0.02 is the planning target for the structural direction** (a sizing
heuristic, not a statistical conclusion); power is computed for the exact frozen estimand at
registration time.

---

## 2. The M8 design

### 2a. Structure: one frozen recipe, then one structural direction (gate finding 11)

The v1 "recipe floor as free riders" is replaced by an explicit two-stage design. The confirmatory
claim is **system-level** (v2 replaces v1); per-lever attribution is dev-descriptive only and is
labelled as such everywhere.

- **Stage R (recipe rebuild) produces ONE frozen recipe R1 before any structural family trains.**
  Contents, each entering only if its probe clears its registered bar, else the M7 setting stays:
  ICT pairs at the B3-selected fraction; listwise distillation over teacher top-N candidates with
  split temperatures (contingent on B2); one continuous mixed objective with replay across phases
  (contingent on a registered matched three-arm test: M7-sequential vs mixed-replay vs
  listwise-only, equal optimizer updates, B-target retention tracked); negatives per the
  matched-steps arm of B13; hparams from B13's registered selection rule; pool rebuilt with
  per-source quotas ≤25%, multi-span/doc, Wikipedia ICT added. R1 is frozen and hash-pinned before
  Stage S begins. R1-vs-M7-recipe is reported descriptively on exploratory dev; it is not a
  confirmatory claim.
- **Stage S (structural family) trains candidate families UNDER frozen R1** from the currently
  in-scope menu (§2c), each with a registered kill/keep gate. A pre-declared mechanical rule —
  highest exploratory-dev worst-group gain among survivors, ties broken by smaller artifact bytes —
  selects **exactly one** candidate. If no family survives its gate, the candidate is R1 alone
  (recipe-only v2), which is a legitimate v2 per the mandate.
- **Shadow gate:** the one candidate crosses the shadow-dev suite once, go/no-go, no fallback and
  no second crossing (gate finding 13). Shadow-dev is **new, never-scored data frozen before
  Phase 0** (see 2d); M7's existing dev components cannot become shadow by relabeling.
- **Seeds:** three seeds for the finalist; aggregation rule pre-declared per architecture class
  before seeds run — table-averaging only for identically-parameterized aligned tables, otherwise
  mechanical median on the pre-declared statistic. Never best-seed.
- Then fusion re-selection (mandatory), freeze, and the single reserved-4 access with M7's
  hardened one-shot mechanics inherited verbatim (spent-receipt, exclusive lock, strict hashes,
  atomic write, snapshot, single infra-retry — copied from `m7/LEDGER.md`, gate finding 17).

### 2b. Phase 0 probes — costed first, bounded subset first (gate findings 12, 16)

Before any probe runs: (i) benchmark each new code path on a 10K-doc/1K-query slice and publish a
serial GPU/RAM/disk schedule in `m8/LEDGER.md`; (ii) resolve the chain-cost discrepancy (recipe
report says ~30 min B+A from logs; architecture report estimated 3–4 h) by one timed run. "One
week" in v1 was an aspiration, not a schedule.

Every probe registers at LEDGER-time: input hashes, split, endpoint, comparator, exact
threshold/CI rule, multiplicity treatment, tie rule, the no-survivor outcome, and the unique
direction it gates. Draft thresholds below are proposals to be frozen verbatim (or amended in
writing) at registration; **a probe with no registered bar does not run.**

Wave 1 (cheapest, most decision-relevant):

| # | probe | cost | gates | draft bar |
|---|---|---|---|---|
| B2 | Entropy/teacher-mass quantiles of the ACTUAL sampled 32-way candidate sets, uniform vs top-200, temp ∈ {0.02, 0.05, 0.1} | 10 min | the listwise-objective leg of R1 | median teacher-target entropy < 0.05 nats under the current sampler confirms H2; the listwise arm then runs |
| B3 | ICT pairs mixed into Phase A at {0, .25, .5, .75}, frozen B checkpoint, **equal optimizer updates and sampling exposure**, dense AND fused, OOD read | ~1 h | the ICT leg of R1 | best arm − baseline ≥ +0.005 OOD, sign non-negative on both OOD components |
| B9 | SVD of Δ = W_final − W_init; eval rank-truncated at r ∈ {16, 64, 256, 1024} | 1 multieval | low-rank regularization rider of R1 | flat-to-r=64 (≤0.002 macro loss) ⇒ rider allowed into R1's registered menu |
| B1' | **Teacher order-sensitivity diagnostic** (renamed per gate finding 8): teacher on token-shuffled + sorted-unique dev queries | minutes | context for D5/E1 only — NOT a ceiling, gates nothing alone | descriptive; reported with B4 |
| B10 | Scoring-rule family sum/max/top-k/LSE, existing table, exact search only (no ANN confound) | ~1–2 h | pooling-rule rider | any member > sum by ≥ +0.005 OOD, Holm over the family |
| B12 | 4-bit/PQ quantization + `ann_sweep.py` interaction | 1–2 GPU-h | byte envelope for D2 | ≤ 0.002 macro loss at 4-bit ⇒ D2 sized at 4-bit |

Wave 2 (needs wave-1 outcomes or more implementation):

| # | probe | cost | gates | draft bar |
|---|---|---|---|---|
| B4 | **Bag-generalization ceiling**: train an expressive permutation-invariant model (DeepSets-style MLP over token embeddings) on an exploratory split of dev queries, eval on a grouped holdout; plus token-bag recoverability measured on the actual table | hours–1 day | whether the order-free class has headroom; informs E1 and the whole-milestone framing | holdout ≥ trained-table + 0.02 ⇒ class has headroom; below ⇒ doc side / lexical arm is where M8 spends |
| B5 | Index-time adaptation on ONE OOD **dev** corpus (spans → ridge-toward-W₀, 3 λ) | ~2 h | D3 (research row until E5) | ≥ +0.005 on that component |
| B6 | Doc-side map, frozen table, cached pairs, OOD read | ~2 h | D1 (research row until E3) | ≥ +0.005 OOD |
| B7 | Block-CG joint solve vocab curve V ∈ {30.5K control, 64K, 128K}, self-trained tokenizer | half day–1 day after benchmark | D2 | monotone slope with 64K−30.5K ≥ +0.005 on held-out dev queries |
| B8 | Bare-target + doc-centroid target blend, closed form | ~2 h | target-design leg of R1 | best α − current ≥ +0.005 OOD |
| B13 | A-phase screening grid (temp × n_neg × steps) + matched-steps negatives arms + riders (EMA, token dropout, per-row lr) | <2 h | R1 hparams + the negatives disposition | REGISTERED AS A SCREEN: selects a region; one confirm arm per adopted setting at matched steps must clear +0.005 OOD before entering R1 |
| B14 | Doc-side instruction refit, closed form, on the two OOD dev corpora only (NOT the full 5.5M-doc text-backed suite — cost per gate finding 16) | ~2–4 h | doc-side instruction rider | ≥ +0.005 on the OOD pair |
| B15 | Context-averaged row init (Wada-style, 100 contexts/token) vs single-forward init, closed form on the OOD pair | ~half day | init leg of R1 | ≥ +0.005 OOD (restored from literature report; dropped in v1 — gate finding 17) |
| B16 | MEV/self-similarity (Ethayarajh) over the ten cached teacher candidates vs measured table nDCG | GPU-minutes | teacher-selection rule for the background sweep; closes an EXPLORED open item | \|ρ\| ≥ 0.5 ⇒ usable screening rule; below ⇒ negative result, written down |

If more than one direction survives its gates, the §2a mechanical rule chooses; multiple surviving
*probes* feed R1 only through their individually-registered bars.

### 2c. The structural menu (Stage S)

**In scope today (no owner ruling needed):**

- **D2 — compositional capacity**: self-trained tokenizer (64–128K, multi-word merges legal), rows
  initialized from teacher forwards (or B15 winner), trained through the forward under R1. Ships at
  ≤66 MB int8 if B12 clears 4-bit, ≤132 MB otherwise. Gate: B7 + B12.
- **D4' — lexical arm, bounded**: BM25F over title/text (weights fitted on dev, frozen before any
  reserved contact; re-registers the fusion function). Gate: registered dev bar ≥ +0.005 fused OOD.
  **Full-dose dual-index question expansion is REMOVED from the confirmatory menu** (gate finding
  15: ~303M generations ≈ 263–702 days on this box). A bounded expansion probe (≤50K docs, one dev
  corpus) may run as research if E2 lands; it does not extrapolate to the reserved system.
  Constraint carried from v1: D4' may not be the sole winning direction — the mandate requires a
  stronger v2 *table* (gate finding 5), enforced by the dense co-condition in §2e.
- **R1-only** — the recipe rebuild alone, if every structural family dies at its gate.

**Research rows until Dylan rules (§0):** D1 doc-side head (E3), D3 index-time adaptation (E5 — and
only ever confirmatory under OS-level isolation: pinned image digest, network off, only
corpus/doc-vector inputs mounted, no `results/frozen_eval` mount, open-syscall audit, output schema
= table + provenance, λ-rule and seed frozen beforehand; if that harness is not built, D3 stays
research-only), D5 nonlinear post-pool head (E1).

**Explicitly out** (unchanged from v1, plus gate additions): higher table dims (identity-linear MRL
heads off a 1024-d hidden state); absorbable transforms as *capacity* (they may appear only as
registered training priors, killed on cross-domain validation, not algebra — Codex round-1 #6);
full late interaction (compute); full-dose doc2query; another 31 MB unigram table with better
hyperparameters as the sole change. (The teacher question moved OUT of this list on Dylan's
2026-08-29 push — see §2f.)

### 2f. Teacher and data workstreams — upgraded per Dylan (2026-08-29)

Dylan: *"are we not revisiting the teacher choice? Are there models we haven't considered? Should
we make changes to the training data? Should we do more tests with licensed data? I want to make
sure that we have explored all the options."* The v1/v2 draft under-weighted both. Upgraded from
background items to first-class Phase-0 workstreams, with the Codex-#21 guard kept: **screening
happens now with the current closed-form frame; any swap decision is re-probed under the final M8
architecture (frozen R1 + winning structural family) before it is put to Dylan** — teacher
ordering can change when the tokenizer/objective changes, so the screen prunes, it never picks.

**T — teacher sweep, run now, three prongs:**

1. **Unblock the two never-probed shortlist survivors.** granite-embedding-english-r2 and
   gte-modernbert-base were excluded ONLY because `stage0_ridge` builds a float64 Gram (50,368² =
   20.3 GB > budget) — a solver-memory limit, not merit. The block-CG solver built for B7 removes
   the limit. Probe both, with the 30,522-vocab CG control so the solver change is not confounded
   (the comparability caveat binds the OLD sweep's ordering, so these two enter as a NEW
   CG-frame sweep that re-probes stella + the top incumbents in the same frame).
2. **Candidates excluded by stale arithmetic, admissible under E7's byte-cap rewrite:**
   Qwen3-Embedding-0.6B (Apache-2.0, 151K vocab, MRL → 38 MB int8 table at 256-d; vendor:
   Alibaba, OK-with-justification), stella_en_1.5B_v5 (MIT, Qwen vocab → ~155 MB at 1024-d; encode
   cost ~3x measured before committing). Both get closed-form table probes IF Dylan sets the byte
   cap (E7). ONNX-portability recorded per candidate (Qwen3-Embedding ONNX status to be checked in
   the sweep).
3. **Fresh sweep for anything released or missed since `m7-teacher-shortlist-2026-08-26.md`**,
   filters: licence permits derived-weight release; vendor rule (relaxed form); table byte cap per
   E7; ONNX-exportable; probed on the closed-form TABLE criterion only (never the tower — Spearman
   0.000 stands); B16's MEV/self-similarity screen applied first if it validates.

Decision rule unchanged from the mandate: a swap needs the closed-form table criterion, an
off-family read, re-probe under the final M8 frame, and Dylan's sign-off.

**DATA — the untested hypothesis, stated honestly:** the clean-stack tax measured *MS-MARCO-shaped*
data at +0.006. It says nothing about **genre-diverse clean data** — scientific/technical/legal
registers, the exact genres of the clean-4 failure — because none was ever collected. Workstream:

1. **Rights review sweep (Sonnet, now):** USPTO patents (public domain), EUR-Lex (reuse grant),
   US federal/CFR, PMC-OA **commercial-use** subset, arXiv abstracts (CC0 metadata; SciDocs
   citation-graph adjacency needs a contamination read), plus anything else the sweep surfaces.
   Output: per-source licence verdict + contamination map against ALL protected sets (six,
   reserved four, any new shadow/M9 reserve).
2. **A registered genre-diversity probe in Stage R:** add the cleared technical corpora as ICT/
   pseudo-span sources under the ≤25%-per-source quota; matched arm vs the Wikipedia-only rebuild;
   read on the OOD dev groups. Bar: ≥ +0.005 OOD to enter R1.
3. **PMC-OA trade-off is Dylan's call (new E8):** licence-clean but its use makes NFCorpus and
   TREC-COVID *training-adjacent*, weakening the six as a descriptive continuity read (the
   reserved four are unaffected). Options: exclude (default), include and disclose, or include
   only in a labelled research arm.
4. Synthetic training queries stay gated on E2; MS MARCO stays permanently out (measured, and
   terms unchanged).

### 2d. Dev instrument (gate finding 13 + inherited protocol)

- **Exploratory dev**: M7's pinned suite (already burned as an instrument, fine for exploration) +
  rebalanced weighting: selection statistic = median/worst-group gain over {CQA group, Wikipedia/QA
  group, heldout groups}, never the arithmetic-mean macro.
- **Shadow dev**: NEW never-scored components, frozen (hash-pinned, licence-verified) before
  Phase 0. Candidates to review for licence + contamination: BEIR sets not otherwise used and not
  reserved (e.g. Quora is licence-out; **Touché-2020 is banned by inherited M7 dev protocol** —
  proposing it in v1 was an error (gate finding 17); using it would require an explicit written
  pre-result amendment with Dylan's sign-off, which we do NOT request), non-BEIR options (LoTTE
  subsets — licence review needed; SciQ; TREC collections with clean terms). A Sonnet sweep
  produces the candidate list with licence status; the pick is registered before Phase 0. If no
  clean shadow set exists, the shadow gate is dropped WITH a written note — not silently weakened.
- **Dev-reuse counter** from day one; published like `m7_dev_reuse_count.json`.
- M7's clean-4 are burned diagnostics; never dev evidence.

### 2e. Confirmatory statistics (gate findings 3, 4, 5, 14)

Registered in `m8/LEDGER.md` before the first M8 number, as executable code
(`m8/final_decide.py`-style), inheriting M7's exact family shape:

- **Estimand**: equal-weight four-set macro (inherited default). The grouped macro
  (FEVER + DBpedia + CQA-pair/2)/3 is a **registered sensitivity analysis**, not the primary
  (v1 had this backwards; grouping also widens the half-width by upweighting 400-query DBpedia).
- **Hypotheses (one-sided, paired, dependence-preserving), family α = 0.025, Holm + raw CI + the
  simultaneous one-sided bound at α/m from the same bootstrap draws (M7's three-leg rule,
  inherited verbatim), m = 3:**
  - **C1 (release, primary): fused-M8 > fused-M7.**
  - **C2 (release co-condition): dense-M8-table > dense-M7-table** — this is what makes the v2 a
    stronger *table* per the mandate, and blocks a BM25F-rescued weak table from shipping.
  - **C3 (absolute floor): fused-M8 > BM25** (frozen builder) on the same macro.
  - Ship requires **all of C1, C2, C3**. No OpenSearch leg (finding 1); published numbers appear as
    labelled context only.
- **Point guard**: C1 point gain must additionally be ≥ +0.005 — registered as a product-margin
  guard, explicitly NOT a minimum-effect hypothesis test.
- **Worst-group guard**: no reserved domain group (Wikipedia pair / CQA pair) may regress vs M7 by
  more than 0.01 point estimate — registered as a ship-blocker with descriptive status.
- **Disclosures registered now**: per-dataset rows; both group subsets; the overlap caveats
  (FEVER-train is a TRAIN source, 11.3% doc overlap; DBpedia is Wikipedia, 9.32%; the CQA pair is
  within-family transfer from dev) stated at the rows, as clean-4 was in M7.
- **Weak-null caveat inherited**: M7's calibration note (mildly anti-conservative in one check,
  uniform weak-null FWER not established) is carried into the ledger text.
- Power computed for this exact estimand/family at registration; the +0.02 planning target is
  labelled as planning-only.

---

## 3. ONNX / fastembed requirement (new scope, Dylan 2026-08-28; approved 2026-08-29)

Ruling: M8 must be servable via ONNXRuntime in fastembed **eventually**; day-1 stella port not
required; benchmark numbers dominate; community conversion by us or a model change are both
acceptable; deferral of goals to M10 is acceptable. Dylan approved this section's approach
("ONNX works for me", 2026-08-29).

Feasibility is now verified (`research/m8-planning/onnx-feasibility-2026-08-29.md`): stella's
export blocker is two config-flag defaults (`use_memory_efficient_attention`, `unpad_inputs`), the
structurally identical gte-large-en-v1.5 ships first-party ONNX with those flags off, Xenova
produced a working (unmerged, unvalidated) stella export once, and fastembed already ships a
non-ONNX bespoke model class (Qdrant/bm25) — direct precedent for the table encoder. Verdict:
**days, not weeks; not blocked.** No existing artifact is reusable as-is; we redo the export with
our own parity check.

Plan:

1. **The released query encoder ships AS an ONNX graph in M8** (Gather → count/sqrt pool →
   MatVec-free normalize, table as int8 initializer + dequant). Trivial by construction; a
   registered parity check (bit-identical rankings vs the int8 reference path on one dev
   component) makes it a release artifact, not a demo. This is the fastembed-native form of the
   deliverable and costs ~a day.
2. **Doc tower**: stella stays the teacher (its selection is measured on the shipped-table
   criterion; every probed alternative is CI-resolved below). A **parity-verified ONNX export of
   stella_en_400M_v5** is an M8 engineering task with M10 as the fallback landing zone: cosine
   parity ~1.0 and nDCG delta ≤ 0.001 on one dev component vs the PyTorch path, fp32 and fp16
   variants measured. If conversion proves blocked (custom-attention export failure), that fact —
   not serving convenience — feeds the teacher-swap consideration with Dylan.
3. **Teacher-probe filter**: any future teacher candidate must be ONNX-exportable (or already
   exported) to enter a probe shortlist. Added to the background sweep spec.
4. Index-side tooling (BM25 builder, any adaptation/fitting scripts) are offline build steps, not
   served models; ONNX does not apply. Whether fastembed should eventually host index-build
   utilities is a product question for Dylan, out of M8 scope.
5. D1/D5, if ever ruled in scope, are ONNX-trivial (MatMul/activation nodes) and inherit the same
   parity check.

---

## 4. Decision items for Dylan (updated)

- **E1 — scope of "zero"**: nonlinear post-pool head (2 MB MLP / 3K-param DyT): keeps ~0.7 ms /
  ~34 MB / instant cold start; forfeits "no learned computation". Now informed by B1'/B4 rather
  than gated on a mislabeled ceiling probe. Required before D5 may be a candidate.
- **E2 — generator licensing (re-scoped and shrunk)**: prompted Apache-2.0 instruct LLM (Qwen3
  line) for (a) synthetic *training* queries, (b) a bounded research-only expansion probe.
  Full-dose doc-side expansion is off the table regardless (compute, finding 15).
- **E3 — two-artifact release** (doc-side head / index-time recipe): required before D1/D3 may be
  candidates.
- **E4 — M9 reserve (time-critical)**: M8 burns the last untouched partition; freeze a new reserve
  now or amend instructions-m9.md. Candidate review runs as a Sonnet sweep either way.
- **E5 — index-time adaptation protocol**: only under the OS-level isolation harness in §2c;
  approve building that harness, or D3 stays research-only.
- **E6 — training-time-only second teacher** for ensemble ranking targets: does the vendor rule
  bind components that never ship?
- **E7 — vocab-rule rewrite (now material)**: replace "vocab ≤ ~50K" with a released-artifact byte
  cap (proposal: ≤120 MB int8; hard ceiling 233 MB = LR int8 parity). This directly decides whether
  Qwen3-Embedding-0.6B and stella-1.5B may enter the §2f teacher probes, and sets D2's size.
  Dylan's 2026-08-29 guidance leans generous: "storage can be fairly cheap" for this use case
  (inference-free/low-power deployments won't have massive datasets), so the cap should not be the
  binding constraint — cold-start/latency and the vs-bge-small optics matter more than raw bytes.

**Dimension note (Dylan asked, 2026-08-29, "stella is 4096?"):** stella_en_400M_v5 is natively
**1024-d** (hidden_size 1024) and M7 shipped 1024-d; the advertised 2048–8192 dims are
identity-activation linear MRL heads, so every dim above 1024 is provably rank-deficient — 4–8x
storage for zero ranking capacity (verified from config.json in the architecture review; stella's
card independently reports 1024-d within ~0.001 MTEB of 8192-d). Same cap applies to stella-1.5B:
hidden 1536, so its 8960-d head is ~5.8x redundant; if it ever wins a probe we serve ≤1536. The
"dimension down to 512 to halve the doc index" lever (old P9) demotes to nice-to-have under the
storage guidance.
- **E8 — PMC-OA commercial subset as training data**: licence-clean, but using it makes NFCorpus
  and TREC-COVID training-adjacent, weakening the six's descriptive continuity read (reserved four
  unaffected). Exclude (default) / include+disclose / research-arm-only?

---

## 5. Inherited-obligation matrix (gate finding 17)

| inherited item | disposition in M8 |
|---|---|
| sqrt-pooling trained-through, full-chain arm (carried lever) | Redesigned arm allowed by mandate; runs only if B10/B13 make pooling look live; its own pre-registration; NOT revived at arm (a). |
| negatives/step confound (carried) | B13 matched-steps design; disposition registered from its confirm arm. |
| doc2query full dose (carried, blocked on E2) | Confirmatory: dead on compute (finding 15). Research probe only, bounded, if E2 lands. |
| teacher revisit (carried, swap bar) | UPGRADED to workstream T (§2f): CG-frame re-sweep incl. the two solver-blocked survivors + byte-cap-admissible candidates + fresh releases; screen now, re-probe under final M8 frame, Dylan sign-off; runs parallel to Stage R, never blocks it. |
| mandatory ablations / ANN sweep / cost reporting (M7 report standards) | Inherited: final report carries ablation table for R1 legs (dev-descriptive), `ann_sweep.py` on the final candidate, cost rows split payload/container/doc-index/hydration. |
| one-shot mechanics (spent-receipt, tag-peel check, lock, hashes, atomic write, snapshot, one infra-retry) | Copied verbatim from m7/LEDGER.md into m8/LEDGER.md before the access; the guard/freeze-binding test suites run against the M8 paths. |
| M7 report addendum | LR-websearch row added as labelled exploratory TIE (+0.0019 [−0.0153, +0.0195]) — corrected from v1's "win". |
| exporter fix | int8-only artifact + parity check (§3.1) — also closes finding 8. |
| Wada context-averaged init / MEV probe (literature) | Restored as B15/B16. |
| Touché-2020 as dev | Withdrawn (banned in inherited dev protocol); shadow candidates come from the §2d Sonnet sweep instead. |

## 6. Gate-finding map (v1 → v2)

1 OpenSearch freeze → deleted (§0). 2 frozen table → §0. 3 statistics → §2e. 4 grouped macro →
sensitivity only (§2e). 5 no-table ship → C2 co-condition; D4' bounded (§2c/2e). 6 D1/D3/D5 →
owner-gated research rows (§0/§2c). 7 D3 isolation → OS-level spec or research-only (§2c). 8 B1
relabel → B1' + B4 (§2b). 9 diagnoses → hypotheses H1–H3 (§1). 10 LR misstatement → corrected +
re-verified (§1.9). 11 recipe floor → Stage R frozen before Stage S (§2a). 12 probe gates → per-
probe registered bars, no-bar-no-run (§2b). 13 shadow/seeds → new frozen shadow data, one crossing,
per-class aggregation (§2a/2d). 14 band misuse → point-guard labelling + power at registration
(§2e). 15 doc2query dose → removed from confirmatory (§2c). 16 costing → benchmark-first schedule
(§2b). 17 dropped items → §5 matrix.

## 7. Next steps

1. Dylan reviews E1–E8 (E4 first — time-critical; E7 next — it gates two teacher probes).
2. Sonnet sweeps (launched 2026-08-29): teacher candidates (§2f prong 2/3, with ONNX + licence +
   byte-cap columns); data rights review (§2f DATA-1); shadow-dev candidates (§2d). ONNX
   feasibility fact-check DONE (`onnx-feasibility-2026-08-29.md`).
3. Second Codex gate on THIS draft; findings actioned; then and only then transcribe into
   `m8/LEDGER.md` as executable pre-registrations with hashes and code-backed decision rules.
