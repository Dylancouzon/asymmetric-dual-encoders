# M8 plan — v5 FINAL DRAFT (2026-08-29), ready for LEDGER transcription

**Status: planning complete.** This is a clean rewrite (v1–v4 in git history) after four
adversarial reviews — three Codex gates (17 → 14 → 8 findings, converging;
`research/m8-planning/codex-plan-gate*-2026-08-2*.md`) and one Opus scientific-judgment review
(`research/m8-planning/opus-plan-review-2026-08-29.md`) — and after Dylan ruled **all twelve
decision items** (§4). No protected set touched; no training run. The next session transcribes
this into `m8/LEDGER.md` as executable pre-registrations per `m8/NEXT-SESSION.md`.

Working release names, LOCKED by Dylan: **`qdrant/constella-zero-m8`** (this milestone's table)
and **`qdrant/constella-nano-m9`** (M9's distilled tower) — the no-compute/low-compute pair.
Constella = constellation + stella: navigate by fixed stars, no engine. If the teacher leaves the
stella lineage, naming reopens with Dylan.

## 0. Inheritance: frozen / amendable / ruled

| class | items |
|---|---|
| **FROZEN** (`instructions-m8.md`, registered before M7's number) | The four confirmatory sets (FEVER, DBpedia-entity, cqadup-android, cqadup-english; hash-pinned, un-scored); paired frozen-M7-vs-frozen-M8 in ONE access; statistics family shape (Holm + raw CI + simultaneous bound, dependence-preserving); six-set scoring descriptive-only, labelled development-informed; comparator BARS from frozen M7 + frozen `fusion.bm25_run` + published numbers as context; minimum release bar = beats frozen M7 CI-resolved on the reserved sets; licensing/decontamination rules; dev-only selection; one-access freeze/ledger protocol. |
| **AMENDABLE only before the first M8 number, in writing** | Macro weighting; exact hypotheses/α/legs; dev-suite composition; probe designs; the E12 descriptive-comparator addition (registered as such). |
| **RULED by Dylan 2026-08-28/29** | E1–E12, see §4. |

## 1. Diagnosis — verified findings vs hypotheses

Verified against repo artifacts (spot-checked this session where noted):

1. Phase B lands +0.0008 above a 2-minute ridge solve; all trained gain comes from Phase A (83 s,
   ~3.7 epochs of 340,850 pairs, plateaus at 2,500 steps). `m7_stage0_ridge_stella.json`.
2. 924,704 licence-clean (query, positive) ICT pairs are discarded by `pseudoq.build()`
   (provenance held, returned as text only). Verified.
3. The pseudo pool is 86.5% ESCI+HotpotQA and first-sentence-only. Verified `pseudoq.py:84-86`.
4. The mined-negatives close is contradicted by the ledger (matched-step +0.0112/+0.0111 clear the
   bar; loss only under a proxy step-selector recorded as ranking arms backwards; "NOT
   IDENTIFIED"). Verified LEDGER:335.
5. The bigram −0.0301 close tested the wrong frame; the JOINT closed-form solve measures +0.0101
   (5K rows) / +0.0143 (10K), CIs excluding 0. Joint retrain open. `m7_bigram_*.json`.
6. The doc-side-map dismissal rests on a wrong sentence (map applies to cached vectors, one GEMM);
   with renormalization it is genuine non-absorbable capacity (rank agreement 0.000 absorbed vs
   direct). LEDGER:577 + `m7_absorb_check.json`.
7. Retention inverts the prior: best on longest queries (ArguAna 0.929), worst on shortest
   (trec-covid 0.667, fiqa 0.673). Recomputed from `m7_final_run.json`.
8. The frozen release container is 93,886,950 bytes (fp16 AND int8 payloads). Verified. M8 ships
   int8-only and reports payload vs container bytes.
9. M7 vs LR-dense-websearch is a TIE (+0.00194 [−0.0153, +0.0195], re-verified by paired
   bootstrap): the honest sentence is "matches LR's single-table system at 1/15 the bytes".
   LR-dense-pertask (0.4583, the missed bar) is an instruction-oracle config, not a
   corpus-adaptation analogue.
10. Instrument facts: dev recipe-perturbation band 0.0027–0.0078 (a DEV-selection fact; does not
    deflate a frozen confirmatory comparison); reserved-4 equal-weight macro half-width ≈0.0096
    (dissimilar) / ≈0.005 (near-sibling). +0.02 is the planning target for the structural
    direction — planning-only, power computed at registration.

Hypotheses (probe-gated, NOT established): **H1** Phase A is pair-starved (B3); **H2** the KL term
is degenerate — one-hot to ~1e-4 nats under 31 uniform distractors at temp 0.02 (B2 measures the
actual candidate-set entropy); **H3** short-query loss is partly recoverable in-class (B17 routes;
the v1 compressed-sensing "proof" is withdrawn — never measured for this table).

M7's overlap disclosures (FEVER 11.3%, DBpedia 9.32% TRAIN-document overlap) are **M7-mix
placeholders**; M8 recomputes them for its final mix (§2f-DATA).

## 2. Design

### 2a. Pipeline (order is binding)

**Protected-partition freezes (LoTTE + M9-reserve inventories) → protected-query filter build →
teacher freeze (workstream T) → noise-floor measurement → Stage R (one assembly + one validation
gate) → Stage S (one finalist by executable rule) → three-seed aggregation → int8 export → ONNX
parity → fusion instantiation → immutable candidate manifest → ONE mandatory LoTTE shadow crossing
→ freeze → reserved-4 doc pre-encode (all scored systems) → the single access.**

- Any post-manifest mutation invalidates the shadow crossing. A teacher change after Stage R
  begins = full R/S restart.
- **Stage R** enumerates every degree of freedom with its M7 fallback: (1) ICT pair fraction (B3);
  (2) listwise distillation arm — candidate sampler + split temps (B2-triggered performance arm);
  (3) phase structure — one registered three-arm test (sequential / mixed-replay / listwise-only,
  equal updates, B-target retention tracked); (4) negatives (B13, matched steps); (5) hparams
  (B13); (6) target design (B8); (7) row init (B15); (8) pool spec — ONE frozen composition:
  per-source quotas ≤25%, multi-span/doc, Wikipedia ICT, the genre bundle (§2f-DATA), and a dosed
  **synthetic-query component (E2)** with its own bar; (9) riders (B9 low-rank, B10 pooling, B14
  doc instruction). Probe outputs are tri-state: adopt named setting / keep named fallback / stop
  direction. B13 confirms ONE complete named configuration jointly (single confirm arm vs the
  complete fallback), never per-axis adoptions.
  **The fused-objective lever (recipe P3) is consciously EXCLUDED per ruling E11** (strict C2: the
  dense table must stand alone because hybrid is not everyone's deployment); recorded here so no
  future session re-derives it as an oversight.
  **Assembly:** adopted settings form one bundle; ONE common-frame validation — assembled-R1 vs
  R0, matched updates/data/seed policy, dense AND fused endpoints, bar sized from the confirmatory
  arithmetic (registration deliverable) — decides R1 vs wholesale fallback to R0. **R0 :=** the
  registered M7 recipe settings instantiated under the selected teacher, current filters, M8's
  data volume/precision/seed policy.
- **Fusion operator** (family grid, depth, dev components, frozen `bm25_run`; amended only if D4'
  BM25F re-registers the lexical function BEFORE Stage R) is frozen before Stage R and applied
  identically at every fused read; the final invocation instantiates parameters only.
- **Stage S** (menu §2c): fixed within-family selection rules (D2's vocab by its own nested dev
  split); family finalists vs R1-alone on one named group vector (registered groups/precision/
  aggregation/budget; worst-group as an explicit formula) with a practical-equivalence band;
  tie-break = total downloadable bytes + doc-index delta. Registered outcomes: no survivor →
  candidate is R1-alone; multiple → the rule picks. **Release-format rule: int8, always** — the C2
  identity, the proven-quality-free format, what the ONNX graph embeds; all sizing/eligibility/
  tie-breaks at int8 (D2 at 128K×1024 = 131.6 MB fits the 233 MB cap without any quantization
  experiment). A 4-bit variant is research-only in M8 and never ships.
- **Seeds:** three; aggregation pre-declared per architecture class (table-average only for
  identically-parameterized aligned tables; else mechanical median). Never best-seed.
- **Shadow gate (mandatory, STOP-on-failure):** the manifest (table/tokenizer/fusion/ONNX/doc-side
  hashes) crosses LoTTE once. **GO threshold = the minimum detectable effect from the joint power
  simulation, not zero**; registered branches: GO → access; NO-GO → "defer to M9, panel
  preserved" (default) or report-only — never a second crossing, never a fallback candidate.
- **Reserved-4 pre-encode (before the access, after freeze):** document vectors for every scored
  system (M8, frozen M7, and E12's comparators bge-small + LR-dense-websearch) are pre-encoded and
  hash-pinned by a named, reviewed script that is physically unable to open the untouched
  query/qrel payloads. Doc encoding reads no queries/qrels and produces no ranking; it is the same
  contact class as the mandated decontamination. The guarded access is then a minutes-long
  gather/rank/metric step. Budget: ~10.12M docs ≈ 20.6 GB fp16 per teacher-dim system; disk ample
  (781 GB free); schedule published before Phase 0.
- One-shot mechanics inherited verbatim from `m7/LEDGER.md` (spent-receipt, tag-peel check,
  exclusive lock, strict hashes, atomic write, snapshot, single infra-retry), guard + freeze-
  binding test suites ported to the M8 paths.

### 2b. Phase-0 probes

**Before any probe:** (i) benchmark every new code path on a 10K-doc/1K-query slice; publish the
serial GPU/RAM/disk schedule (including the reserved pre-encode and the timed B+A chain that
settles the ~30-min-vs-3-h discrepancy); (ii) **measure M8's noise floor**: two matched null
replicates (seed change; ±10% A-steps) on every endpoint a bar will read. **Every bar is then set
at ≥2x the measured floor, or uses the B3 template (sign-consistency on both OOD components + a
seed replicate).** The draft bars below are proposals; LEDGER freezes the numbers. **No bar, no
run — enforced by `m8src/probe_guard.py`, not prose.** Probe outputs are tri-state (§2a);
diagnostics can only trigger separately-registered performance arms; cross-frame conclusions
(closed-form → trained) are reconfirmed on the assembled candidate.

Wave 1: **B2** (candidate-set entropy quantiles, uniform vs top-200, 10 min — diagnostic for the
listwise arm) · **B3** (ICT fractions {0,.25,.5,.75}, equal updates and exposure, dense+fused OOD
— the template probe) · **B17** (**in-domain oracle-generalization**: 50/50 query split on the dev
CQA components, oracle table on one half, score the other against the 0.481 teacher ceiling;
REGISTERED ROUTING RULE: held-out ≥ ~0.45 ⇒ supervision/objective is the story, R1 is the
milestone's center of gravity; stalls ≤ ~0.40 ⇒ the class caps in-domain and D2/D1/D4' carry the
milestone; between ⇒ both, budget split as registered) · **B9** (SVD rank truncation; adopt
low-rank delta at registered rank or keep fallback) · **B10** (sum/max/top-k/LSE scoring family,
exact search only).

Wave 2: **B7** (block-CG vocab curve 30.5K-control/64K/128K — gates D2) · **B6** (doc-side map on
frozen table; **precondition: demonstrated fused one-file doc ONNX graph**, then the quality bar —
gates D1) · **B8** (bare + doc-centroid targets, closed form) · **B13** (A-grid + matched-steps
negatives + riders; one complete configuration confirmed jointly) · **B14** (doc-side instruction
refit, the two OOD dev corpora only) · **B15** (context-averaged row init).

Inside workstream T: **B16** (MEV/self-similarity — descriptive-only; may not prune unless
separately validated on fresh clean-screen artifacts).

Removed from M8's calendar: **B1'/B4** (E1 made them decision-irrelevant here; recorded as M9
planning diagnostics), **B5** (E5: index-time adaptation is research-only AFTER the final access),
**B12** (superseded by the int8-always rule; a 4-bit sweep may run post-finalist as research),
**B11** (fusion complementarity — moot under E11's strict C2; the fusion operator is frozen
mechanics, not a lever).

Dev-reuse counter runs from evaluation #1 (`m8_dev_reuse_count.json`).

### 2c. Stage-S menu

- **D2 — compositional capacity** (in scope): self-trained tokenizer (64–128K, multi-word merges),
  rows initialized per B15's winner, trained through the forward under R1. Sized at int8 under the
  233 MB cap; gate B7. **Registered coverage spec** (Opus #7): minimum-updates-per-reachable-row
  criterion, targeted rare-row span sampling with the pool expansion to meet it, a
  coverage-vs-capacity diagnosis rule for any failure, and the "bag mass on cold rows vs per-query
  retention" diagnostic run on existing artifacts first.
- **D1 — doc-side head** (in scope per E3, conditional): linear 1024→1024 / 2-layer MLP / →512
  variants over cached teacher vectors, jointly trained; **preconditions: fuses into ONE doc-side
  ONNX file as plain MatMul/activation nodes (E3's hard condition, tested at B6 entry), and — per
  strict C2 (E11) plus gate-4's G4-4 — a D1 win alone does NOT make a qualifying v2 table: a
  qualifying table change (R1 or D2) must also survive** for the release path to open.
- **D4' — lexical arm, bounded** (in scope, auxiliary): BM25F title/text, weights dev-fitted and
  frozen before Stage R (re-registers the fusion function). May never be the sole qualifying
  change. Full-dose dual-index question expansion stays dead on compute (263–702 days); a bounded
  ≤50K-doc research probe may run under E2, never extrapolated to the reserved system.
- **R1-only** — legitimate iff ≥1 qualifying change survived (§2e).
- **Research-only, never candidates:** D3 index-time adaptation (E5 — one labelled measurement
  allowed AFTER the final access), D5 nonlinear head (E1 — out entirely; the tiny-compute niche is
  constella-nano's).
- **Kill-list stands** (algebra/arithmetic): higher table dims (identity-linear MRL heads off a
  1024-d hidden state); absorbable transforms as capacity (allowed only as registered training
  priors, killed on cross-domain validation); full late interaction; another 31 MB unigram table
  with better hyperparameters as the sole change.

### 2d. Dev and shadow instruments

- **Exploratory dev**: M7's pinned suite, selection on median/worst-group gain over registered
  groups (never the arithmetic-mean macro). M7's clean-4 are burned diagnostics.
- **Shadow**: LoTTE (E10, adopted under the written "not literally CQADupStack" reading;
  CC BY-SA, pre-clickwrap 2021 dump), hash-frozen at the very start of Phase 0 — **after the
  overnight overlap measurement** (community-list intersection + document-hash overlap vs reserved
  android/english and dev physics/programmers; any hit drops the offending slice and goes to the
  wake-up note; material overlap reopens E10). Registered coverage statement: the shadow reads the
  CQA half of the estimand only; no clean Wikipedia/entity-shaped shadow exists (candidates were
  swept and failed on licence) — the FEVER/DBpedia half is guarded instead by the six-set
  no-regression guard and the worst-group guard (§2e). Touché-2020 stays banned.
- **M9 reserve** (E4: both): EUR-Lex + USPTO retrieval sets. Overnight freezes ONLY the corpus and
  query-text inventories (what the contamination filter needs); qrel construction and the final
  hash-pin follow a reviewed procedure with a PROVISIONAL sanity report (Opus #20).

### 2e. Confirmatory statistics and ship rule

Registered as executable code before the first M8 number. Estimand: equal-weight four-set macro
(grouped variant = sensitivity only). One-sided paired hypotheses, family α = 0.025, Holm + raw CI
(two-sided 95% lower endpoint > 0, unrounded) + simultaneous one-sided bound at α/m from the same
draws, m = 3:

- **C1: fused-M8 > fused-M7** (primary).
- **C2: dense released-M8 system > frozen dense M7 system** (strict, per E11) — endpoints fully
  frozen; presented as a system comparison, not table causality.
- **C3: fused-M8 > BM25** (absolute floor, frozen builder).

**Ship requires C1 ∧ C2 ∧ C3, PLUS:** a **qualifying v2 table** (enumerated: objective-family,
data-construction, feature/tokenizer, row-init-construction, structural riders, doc-side head —
though D1 alone does not satisfy the requirement; NOT qualifying: seed/steps/temps/negative
counts/lr/pool sizes/ordinary tuning; distinct int8 payload necessary but insufficient); the
**+0.005 point guard on C1** (product margin, not a hypothesis); the **worst-group guard** (no
reserved group regresses vs fused-M7 by >0.01 point estimate); and the **six-set no-regression
guard** (descriptive, frozen vectors, zero new access: M8 must not fall below M7 on the six by
more than a registered margin — the anti-memorization ship-blocker, since all four reserved sets
are training-adjacent and FINDINGS #13's signature would otherwise read as a win there).

**Descriptive context inside the same access (E12, registered before any M8 number):**
bge-small-en-v1.5 and LR-dense-websearch scored on the reserved four, outside the Holm family, no
ship consequence — the external anchor for the report. FEVER rows carry the E9 proxy-provenance
caveat; all legs additionally reported FEVER-excluded; the cancellation argument is stated
conditional on a stella-lineage teacher.

Registration deliverables: decision code (draws, seed, stratified paired resampling, strict qid
alignment, Holm ordering/ties, α/3 bound), the **joint power simulation of the full ship rule**
publishing minimum detectable effects AND **P(ship)** under the surviving levers' EV — put in
front of Dylan before Phase 0 spends its week (a knowing report-only choice beats a discovered
one), and the weak-null calibration caveat carried verbatim.

### 2f. Workstreams

**T — teacher (opens Phase 0).** Order: protected freezes → filter → screens. Fixed student frame
per screen; fit-list regenerated through the filter (M7's list had protected hits; unusable);
provenance rows per candidate (registry proxy convention). Probes: stella-1.5B (breaks WordPiece
compatibility — fingerprints rebuilt if it wins), harrier-oss-v1-0.6b (training data undisclosed —
contamination black box, needs a ruling before adoption), granite-r2 + gte-modernbert as CG-frame
controls, incumbent re-probed in the same frame. **Swap bar (Opus #8): the challenger must beat
the incumbent on the table criterion by MORE than the swap's CI-widening penalty** (near-sibling
≈0.005 → dissimilar ≈0.0096 half-width, stated numerically from the power sim), and the swap
charges its real costs (double reserved pre-encode; FEVER-cancellation loss; compatibility
rebuild). **Same-teacher is the registered default.** Dylan signs any swap. ONNX feasibility
evidence (successful local export or family precedent; absence of an artifact ≠ failure) is
assessed for every finalist BEFORE the freeze.

**DATA.** Cleared (primary-source verified, zero eval overlap): USPTO (37 CFR), EUR-Lex
(2011/833/EU, TDM-silence noted), US federal. OUT: bulk arXiv, SEC EDGAR, HackerNews, post-2024
StackOverflow. PMC-OA: EXCLUDED (E8; revisit only if the genre probe shows a biomedical-specific
gap — the PMID measurement is deferred to that condition). **Enforcement precedes probes**: R1/R2
filters extended over six + reserved + shadow + M9-reserve and run over every corpus (Wikipedia
ICT included); post-filter hashes frozen; M8-mix overlap rates recomputed and replace the M7-era
figures. **Genre probe**: one frozen bundle (USPTO+EUR-Lex+federal at registered shares, capped
total technical share), matched examples/updates vs the Wikipedia-only arm, endpoint = registered
OOD groups + a technical non-protected exploratory group from held-out cleared-corpus
pseudo-queries; whole-bundle in/out, no post-hoc cherry-pick. **Synthetic queries (E2)**: a dosed
registered component of the pool spec with its own bar (EmbedDistill's +query-generation prior:
gains amplify as the student shrinks); Qwen3-line prompted, dedup vs all benchmarks, per-query
provenance.

## 3. ONNX / fastembed (scope approved)

Verified feasibility (`research/m8-planning/onnx-feasibility-2026-08-29.md`): stella's export
blocker is two config flags; gte-large-en-v1.5 (same architecture) ships first-party ONNX; verdict
days-not-weeks. Plan: (1) **constella-zero ships AS an ONNX graph** (Gather → sqrt-count pool →
normalize, int8 initializer) — fastembed-native, with the BM25-bespoke-class fallback; (2) a
parity-verified export of the SELECTED teacher is an M8 task, M10 the fallback landing zone;
ONNX-infeasibility (demonstrated, not absence-of-artifact) is the only ONNX-based teacher
exclusion; (3) D1, if it survives, ships fused into the doc graph — one file (E3's condition);
(4) **parity runs on the final aggregated int8 artifacts BEFORE shadow**, full conformance
fixture suite + vector/cosine tolerances + top-k tie policy + nDCG delta bound, all pinned in the
manifest; (5) index-side tooling is offline, not served.

## 4. Rulings (all twelve, Dylan, 2026-08-28/29 — the authoritative list; prior wording superseded)

| # | ruling |
|---|---|
| E1 | **Pure lookup is the product.** No query-side neural head in M8, not even research ("if people have some compute capability, there's no reason to not use M9"). |
| E2 | **Synthetic Qwen3 training queries approved** ("green light if no downsides" — downsides recorded: style bias, self-limited by the OOD bar; half-day GPU). |
| E3 | **Doc-side head approved CONDITIONALLY**: must fuse into the doc ONNX graph as plain nodes — one served file, no custom pipeline — and clear its probe. |
| E4 | **Reserve BOTH M9 sets** (EUR-Lex + USPTO, frozen construction procedure, never scored in M8). |
| E5 | **Index-time adaptation: research-only, end of project** ("seems over engineered… afraid of the accusations"). Never in the confirmatory candidate. |
| E6 | **Training-only second teacher allowed if licence-clean**; vendor rule binds shipped components; model-card documented. |
| E7 | **Byte cap 233 MB int8** ("storage can be fairly cheap"); cold-start/latency and optics matter more than bytes. |
| E8 | **PMC-OA excluded** (delegated "include if it moves the needle, otherwise exclude" → excluded: its unique value is duplicated by cleaner sources; cost is the NFCorpus/TREC-COVID honesty read). |
| E9 | **FEVER: label + sensitivity read** (proxy-provenance caveat at the rows; all legs also FEVER-excluded; cancellation stated conditional on a stella-lineage teacher). |
| E10 | **LoTTE adopted as the mandatory shadow** under the written "not literally CQADupStack" reading — pending the overlap measurement (§2d); shadow STOP-on-failure. |
| E11 | **STRICT C2** ("we want something that looks good on benchmarks too; hybrid should be the default but isn't to everyone") — the dense table must beat M7's dense table; the fused-objective lever stays consciously excluded. |
| E12 | **Comparators inside the access: YES, bge-small + LR-dense-websearch**, descriptive only, outside the Holm family, registered before any M8 number. |
| E13 | **FineWeb (Qdrant/FineWeb-10B): measure first, ship-decide later** (2026-08-29). The affirmative-licence standard stays in force for the RELEASED stack; a FineWeb arm joins the registered data probe under the clean-stack-tax design — fully filtered/decontaminated (R1/R2 + near-dup vs ALL protected partitions), matched exposure, never released, refused by the release guard. If it clears the probe bar by a margin worth shipping, the licensing ruling comes back to Dylan WITH the number; note that Qdrant redistributing FineWeb is itself evidence of the company's posture, but a wrapper tag — including our own — is not a licence. |

## 5. Inherited-obligation matrix

| item | disposition |
|---|---|
| sqrt full-chain arm | Own registration slot at R1-assembly time (run or formally deferred with owner-visible reasoning); B10/B13 inform, do not falsify; never revived at arm (a). |
| n-gram rows (carried lever) | **Superseded by D2** (the no-whitespace tokenizer IS the n-gram direction, non-overlapping form). If D2 dies, additive rows need their own registration — no auto-revival. |
| negatives/step confound | B13 matched-steps; disposition from its single joint confirm arm. |
| doc2query full dose | Dead on compute for confirmatory; bounded research probe only (E2). |
| teacher revisit | Workstream T opens Phase 0; swap bar incl. CI-widening penalty; same-teacher default; Dylan signs. |
| M7 mandatory ablations (flat-vs-learned weights, prefix variants, init controls, dense/BM25/fusion decomposition, int8) | Mapped per eligible architecture in LEDGER; each adopted or not-applicable WITH reason. |
| ANN sweep + cost reporting | `ann_sweep.py` on the final candidate; cost rows split payload/container/doc-index/hydration. |
| one-shot mechanics | Copied verbatim; guard + freeze-binding suites run against M8 paths. |
| M7 report addendum | LR-websearch row as labelled exploratory TIE (+0.0019 [−0.0153, +0.0195]). |
| exporter fix | int8-only artifact + §3 parity (closes diagnosis #8). |
| Wada context-averaged init / MEV | B15 / B16 (descriptive). |
| Touché-2020 | Banned, stays banned. |

## 6. Next steps

1. Overnight session executes `m8/NEXT-SESSION.md` (LEDGER transcription with the registration
   deliverables, guards, freezes, noise floor, schedule, then workstream T screens + wave-1
   probes as far as the guards allow).
2. Wake-up note at the top of `m8/STATUS.md` collects everything needing Dylan (P(ship) readout,
   LoTTE overlap result, harrier provenance question, any teacher-swap case).
3. This file is superseded by `m8/LEDGER.md` at transcription and then moves to
   `research/m8-planning/PLAN-final-2026-08-29.md`.
