# M8 plan — DRAFT for adversarial gate + Dylan review (2026-08-28)

**Status: DRAFT.** Not a pre-registration. Nothing here is frozen; no protected set has been
touched; no training has run. Produced by the M8 planning session from five independent reviews
(four Opus 5 brainstormers + one Codex gpt-5.6-sol adversarial pass, all in
`research/m8-planning/`), with the load-bearing factual claims spot-verified against the repo.
Dylan's framing for this round: *all previous settled decisions can be revisited, all assumptions
can be broken — as long as the project stays replicable and defendable on benchmarks.*

The draft has three parts: (1) the corrected diagnosis of M7's miss, (2) the M8 design — a probe
battery that picks one structural direction, plus the protocol that makes the result believable,
(3) the decision items only Dylan can rule on.

---

## 1. The corrected diagnosis — what M7's miss is actually made of

Five independent reviews converged on the same rewrite of M7's closing claim. **"The gap is
architectural" was inferred, not measured**: the clean-stack tax is evidence about MS MARCO, not
about architecture, and the alternative — objective + supervision starvation — was never
eliminated. The specific defects, each verified against repo artifacts:

1. **Phase B is a 16,000-step reimplementation of a 2-minute ridge solve.** B lands +0.0008 above
   closed form (`m7_stage0_ridge_stella.json` 0.4973 vs `p35b-2m` 0.4981 proxy); the entire recipe
   gain over closed form comes from Phase A — 2,500 steps × 33 ms = **83 seconds**, ~3.7 epochs
   over 340,850 pairs, plateaus there. The recipe is **pair-starved, not capacity-starved** (the G2
   overfit probe hits 0.99999 in-sample).
2. **The "KL" distillation term is degenerate.** Distractors are 31 uniform draws from the 2M bank;
   at temp 0.02 the teacher target is one-hot to ~1e-4 nats (arithmetic from
   `m7_diag_scores.json`). **No teacher ranking information reaches the student anywhere in the
   recipe** — Phase B is pure regression to the teacher query vector, the criterion FINDINGS #3
   proved mis-ranks. A specification defect, not a closed avenue.
3. **924,704 licence-clean (query, positive) pairs are being discarded by a `return` statement.**
   `pseudoq.build()` holds `(store, doc_id)` for every span and returns text only; retaining it
   gives ICT/Contriever-style pairs with teacher doc vectors already cached — 3.7× Phase A's
   supply at zero new encoding.
4. **The pseudo pool is 86.5% ESCI+HotpotQA and first-sentence-only** (verified,
   `pseudoq.py:84-86`), i.e. the "query-shaped" pretraining text is two genres, neither matching
   the confirmatory genres.
5. **The mined-negatives close is contradicted by the ledger itself** (verified, LEDGER:335): at
   matched 2,500 steps `teacher16` +0.0112 and `mixed32` +0.0111 clear the bar; they lose only
   under a proxy step-selector the ledger separately records as ranking arms backwards. Recorded
   outcome: "NOT IDENTIFIED", not refuted.
6. **The bigram close tested the wrong frame.** −0.0301 was closed-form residual fitting toward the
   query-vector target onto an earlier table. The *joint* closed-form solve measures **+0.0101 at
   5K rows, +0.0143 at 10K** [CIs excl. 0] — ~+0.0042/doubling. Joint retrain is explicitly open.
7. **A doc-side linear map was dismissed on a factually wrong sentence** (LEDGER:577 "changing the
   document map means re-encoding the corpus" — it applies to cached vectors, one GEMM, no teacher
   re-run). With renormalization it is genuine non-absorbable capacity (rank agreement 0.000 with
   the absorbed form, the repo's own measurement).
8. **Retention is inverted from the project's prior**: best on the longest queries (ArguAna 0.929),
   worst on the shortest (trec-covid 0.667, fiqa 0.673). The loss is concentrated exactly where a
   pooled bag is provably information-preserving (compressed-sensing bound: invertible up to ~150
   distinct tokens at d=1024) — i.e. where a *nonlinear post-pool head* could in principle recover
   the bag-of-words ceiling.
9. **We missed a bar defined by corpus adaptation, using the only unadapted system in the
   comparison.** LR-dense-**pertask** 0.4583 vs LR-dense-**websearch** 0.4320: the adaptation delta
   (+0.0263) exceeds our whole miss (−0.0243). BM25 and OpenSearch are corpus-fitted by
   construction. Meanwhile our like-for-like row is a WIN: our single table 0.4339 vs LR's single
   websearch table 0.4320, at 1/15 the bytes — the M7 report should carry this row (labelled
   exploratory; per-query vectors already frozen).
10. **Operational bug (verified):** the frozen release container is **93.9 MB**, not 31.3 MB — the
    exporter writes `rows_fp16` *and* `rows_int8` into the same npz. M8 needs an int8-only
    deployment format and must report payload and downloadable bytes separately.

Two instrument facts that govern everything below: the recipe-perturbation band is
0.0027–0.0078, and every effect M7 ever adopted sits inside it; the reserved-4 panel confirms
Δ ≈ 0.010–0.012 (dissimilar systems) or ≈ 0.007 (near-siblings). **Therefore M8 is one structural
direction sized ≥ +0.02, not a lever ladder.** Lever-stacking inside the band is unconfirmable and
scientifically hollow even if it passes.

---

## 2. The M8 design

### 2a. Phase 0 — the probe battery (≈ one week, no protected set touched, ~zero risk)

Ordered by information-per-hour. Each has a pre-registered read-out; results go in `m8/LEDGER.md`
as they land. Everything reads the OOD dev subset next to the macro.

| # | probe | cost | what it decides |
|---|---|---|---|
| B1 | **Bag-ceiling**: teacher on token-shuffled + sorted-unique dev queries | minutes | The ceiling of every order-free query encoder. If shuffled ≈ teacher, the 0.367→0.481 OOD gap is linearity/supervision tax and a nonlinear head or better objective can chase it; if it collapses, the class is capped and M8 moves to the doc side / lexical arm. **The single most decision-relevant measurement available.** |
| B2 | **KL degeneracy check**: entropy of teacher target under uniform vs top-200 distractors, temp ∈ {0.02, 0.05, 0.1} | 10 min numpy | Confirms defect #2 before touching the objective. |
| B3 | **ICT pairs go/no-go**: retain span provenance, mix pseudo pairs into Phase A at {0, .25, .5, .75} from the frozen B checkpoint | ~45 min | The pair-starvation thesis (defect #1/#3). |
| B4 | **In-domain generalization ceiling**: 50/50 query split on dev CQADupStack, oracle table on one half, score the other | hours | Architecture-caps vs supervision-caps — the honest version of "the gap is architectural". |
| B5 | **Index-time corpus adaptation**: 100K spans from one OOD dev corpus → ridge-toward-W₀, 3 λ | ~2 h | The adaptation lever (diagnosis #9). Null closes it in an afternoon. |
| B6 | **Doc-side map, frozen table**: fit M alone on cached pairs, eval OOD | ~2 h | Whether the joint (table, doc-map) arm is worth a chain (defect #7). |
| B7 | **Vocab-size curve**: block-CG joint solve at V ∈ {30.5K control, 64K, 128K} with a self-trained tokenizer | half day | The capacity direction's slope beyond 10K rows (defect #6), before buying any chain. |
| B8 | **Bare target + doc-centroid target blend**, closed form, α grid | ~2 h | Two open target-design items in one pass. |
| B9 | **SVD rank truncation of Δ = W_final − W_init**, eval at r ∈ {16, 64, 256, 1024} | 1 multieval | Whether the trained delta is low-rank (regularization headroom + the "approximability unexplained" item). |
| B10 | **Scoring-rule family**: sum / max / top-k / LSE reduction over token rows, existing table | <1 h | Whether sum-pooling is even the right reduction (never tested). |
| B11 | **Fusion oracle ceiling + complementarity** of table vs BM25 on dev | ~1 h | How much residual fused training (2c) can chase. |
| B12 | **4-bit / PQ quantization sweep** on the shipped table | 1–2 GPU-h | The byte envelope: whether capacity ships at 66 MB instead of 132 MB. |
| B13 | **A-phase screening grid** (temp × n_neg × steps, + matched-steps negatives arms, + EMA/dropout/per-row-lr riders) | <1 h total | Resolves the never-run `phase3_hparams` axis and the negatives/step confound; pre-registered as a **screen, never an adjudication**. |
| B14 | **Doc-side instruction refit** (closed form on dev corpora re-encoded with a fixed doc prompt) | 2–4 h | The "cheapest untried structural lever" from M7's own research notes. |

Dev-reuse note: B3/B13 are trained arms and count toward the published dev-reuse tally; the rest
are closed-form or eval-only. All get logged in the M8 analogue of `m7_dev_reuse_count.json`.

### 2b. The structural directions the probes choose among

Pre-registered menu; the probe outcomes plus the decision rules below pick **one primary** (plus
free riders). Expected values are the reviewers' estimates, not promises.

- **D1 — Learned doc-side head over frozen stella vectors, jointly trained with the table**
  (gate: B6, supported by B1/B4). Linear 1024→1024, 2-layer MLP, and 1024→**512** variants — the
  512 variant halves the customer's doc index AND the table (16 MB int8). Negative bank survives
  (head applies to cached vectors). Release becomes two artifacts (table + doc head) — Dylan item
  E3. On-ramp to a stella LoRA co-train only if the head shows ≥0.02 OOD headroom (that escalation
  is days of GPU + ~28 h re-encode, one shot).
- **D2 — Compositional capacity: self-trained tokenizer (64–128K, multi-word merges) with rows
  trained through the forward** (gate: B7 slope + B12 bytes). Ships at 66–132 MB int8 (33 MB at
  4-bit×512-d). Complements D1; competes with it for the milestone's one confirmatory shot.
- **D3 — Index-time corpus adaptation recipe** (gate: B5). Ships zero extra bytes; turns the
  deliverable into a Qdrant index-build step. Needs Dylan's protocol ruling (E5) because the table
  is fitted (documents-only) on the deployment corpus — the same operation BM25's idf does, but it
  must be pre-registered with the query/qrel paths provably unreadable to the fitting code.
- **D4 — Lexical-arm upgrade + fusion mechanics** (gate: B11; doc2query leg gated on E2).
  BM25F (title/text) nearly free; dual-index question expansion (Doc2Query++-style, 30 q/doc,
  prompted Apache-2.0 generator, separate question index fused convexly) is the only lever with
  published gains on five of our six datasets; a DeepImpact-style own term-weighting model is the
  high-ceiling expensive form. Query side stays statistics-only throughout.
- **D5 — Nonlinear post-pool head (2 MB MLP / separable DyT)** (gate: B1 must pass + Dylan's scope
  ruling E1). Recovers the bag-of-words ceiling in principle; forfeits the word "zero", keeps
  0.55–0.7 ms and instant cold start.

**Recipe floor (runs under whichever direction wins, since the objective defects are direction-
independent):** ICT pairs at the B3-selected fraction; real listwise distillation over teacher
top-N candidates with a split temp (fixes defect #2; also the principled fix for the vacuous
false-negative check — an unlabelled positive gets teacher mass instead of a hard negative label);
one continuous mixed objective with B-style replay through A (Codex #15) instead of the
cosine→InfoNCE handoff; matched-steps negatives verdict from B13; hparams from the B13 region;
pool rebuilt with per-source quotas (≤25%/source), multi-span/doc, Wikipedia added as an ICT
source; training scored against the fused system where D4 is in play (gate: B11). Synthetic
training queries (Qwen3-prompted) only if B3 says pairs are the binding constraint and Wikipedia
ICT saturates first.

**Explicitly NOT in the menu** (with reasons, so they stay dead): higher table dims (provable
no-op: stella's MRL heads are identity-linear off a 1024-d hidden state); post-pool *linear* maps,
per-token scalars, centering/whitening (absorbable — the algebra stands); full late interaction
(compute arithmetic: ~66 h/dataset on this box); teacher swap as an opening move (probe only
post-2026-08 candidates with ratio > 0.72 on the closed-form criterion, in the background, Sonnet
sweep first; the architecture must be fixed before a teacher comparison is meaningful — Codex #21);
another 31 MB unigram table with better hyperparameters (the one thing all five reviews agree
cannot recover 0.024).

### 2c. Protocol and evaluation (the part that makes any number believable)

Actions **before any M8 training run**, in `m8/LEDGER.md`:

1. **Comparator freeze on the reserved four, NOW**: encode
   `opensearch-neural-sparse-encoding-doc-v3-gte` over the reserved-4 corpora (~10 h), write
   per-query vectors into `results/perquery.json` under the same regime as M7's comparators —
   computed and sealed before any M8 design reads them. Decide (cost call): LR-dense-websearch row
   if affordable. Precedent: `m7_bars_clean4.json`.
2. **Bars.** Primary: **fused-M8 > fused-M7** on the reserved-4 **grouped** macro
   (FEVER + DBpedia + (android+english)/2, /3 — the CQA pair is one domain, not two), Holm at the
   registered family α, raw CI leg, point gain ≥ +0.005 (outside the perturbation band). Secondary
   registered legs: dense-vs-dense, **absolute floor: fused-M8 > BM25 CI-resolved on the same
   grouped macro** (an M8 that beats M7 but sits under BM25 does not ship), and fused-M8 vs the
   frozen OpenSearch row (context leg). Pre-registered subgroup disclosures: Wikipedia pair, CQA
   pair, per-dataset rows — the reserved four are *more* train-adjacent than M7's six (FEVER-train
   is a TRAIN source, DBpedia is Wikipedia) and 2 of 4 flatter memorization-shaped gains; the
   disclosure is registered before any number exists.
3. **Dev suite redesign (OOD-first).** Selection weight ≥ half on families absent from TRAIN and
   from disclosed teacher data; CQA subforums = one group; Wikipedia/QA = one group; selection on
   median/worst-group gain, not the arithmetic-mean macro (the 0.915-dev/0.755-final gap is the
   measured cost of the old macro). Shadow-dev partition: architecture families see the exploratory
   half; only frozen family winners see the shadow half. Candidate new OOD dev component:
   Touché-2020/args.me (argumentative; needs primary-source licence verification; ArguAna-adjacency
   disclosed). M7's clean-4 are burned diagnostics, not new dev evidence.
4. **Seed policy.** Three seeds/perturbations for every finalist; pre-declared equal-weight table
   average ("table soup", zero runtime bytes) or mechanical median — never best-seed.
5. **Six-set continuity**: M8 may score M7's six descriptively, always labelled
   "development-informed at milestone level" (already registered in instructions-m8.md).
6. **Exporter fix**: int8-only release format; bit-identical ranking check against the int8
   variant; report payload vs container bytes (diagnosis #10).
7. **Dev-reuse counter** from day one (M8 analogue of `m7_dev_reuse_count.json`).

### 2d. Execution order (after Phase 0 and Dylan's rulings)

1. Freeze 2c (dev metric, shadow suite, seed policy, bars, comparator vectors).
2. Phase 0 battery → write verdicts to `m8/LEDGER.md`, kill/keep each direction by its
   pre-registered gate.
3. Recipe floor rebuild (objective + pool + ICT) with the B13-selected region → one candidate
   family per surviving direction, exploratory dev only.
4. Family winners → shadow dev, three seeds, mechanical selection.
5. Fusion re-selection on the final candidate (mandatory — a changed checkpoint invalidates w).
6. One frozen system → the single reserved-4 access, paired against frozen M7, per the bars.

---

## 3. Decision items for Dylan (bundled; none assumed)

- **E1 — Scope: what does "zero" mean?** A 2 MB nonlinear post-pool head (or DyT, 3K params) keeps
  ~0.7 ms query latency, ~34 MB artifact, instant cold start, no tokenizer/transformer change — but
  forfeits the literal "no learned computation at query time" label. Is the product "no transformer
  at query time" (head allowed) or "literally table lookup only"? Asked now with B1 pending; only
  acted on if B1 shows the ceiling is worth it.
- **E2 — Generator licensing ruling (re-scoped).** Not doc2query-the-model: a *prompted* Apache-2.0
  instruct LLM (Qwen3 line — the precedent `research/m7-data-licensing.md` already names clean),
  used for (a) dual-index question expansion on the BM25/lexical arm, (b) synthetic training
  queries. MS-MARCO-fine-tuned generators stay out by our own inheritance standard.
- **E3 — Two-artifact release.** A doc-side head (D1) or index-time fitting recipe (D3) makes the
  release "query table + doc-side component". Advance notice; the report would carry both cost
  lines.
- **E4 — The M9 reserve problem (time-critical, only legal now).** M8 burns the last untouched
  partition. Either freeze a new never-scored reserve now (candidates need licence + contamination
  review — e.g. Touché-2020 has licence questions; suggest reviewing 2–3 candidates and freezing
  two), or accept M9 ships without a confirmatory panel and instructions-m9.md is amended
  accordingly.
- **E5 — Index-time adaptation optics (D3).** Fitting the table on deployment-corpus *documents* at
  index build is operationally identical to BM25 computing idf, but a reviewer can call it
  "fitting on the eval corpus". If D3 survives its probe, the pre-registration would be:
  documents-only, frozen script, query/qrel paths unreadable in code, system-vs-system labelling.
  Green-light putting that protocol in front of the confirmatory run, or keep D3 out of the
  confirmatory candidate (research row only)?
- **E6 — Ensemble distillation targets (P11).** A training-time-only second teacher's *rankings*
  (never its vectors, never shipped) — does the vendor rule bind training-time-only components?
- **E7 — Vocab-rule rewrite.** The "vocab ≤ ~50K" teacher filter was derived from fp16 arithmetic;
  int8 (measured quality-free) and MRL truncation change it. Proposed replacement: "released table
  ≤ some byte cap Dylan picks (e.g. ≤120 MB int8, or ≤LR's 233 MB), whatever the vocab". Affects
  the background teacher sweep only.

---

## 4. Sources and verification trail

- `research/m8-planning/opus-recipe-2026-08-28.md` — recipe/objective (Phase A starvation, KL
  degeneracy, ICT pairs, fused-objective, screening grid).
- `research/m8-planning/opus-architecture-2026-08-28.md` — capacity/bytes (index-time adaptation,
  tokenizer swap, doc-map, 4-bit, absorbable kill-list, fragmentation analysis).
- `research/m8-planning/opus-premises-2026-08-28.md` — premises (retention inversion, bag ceiling,
  bar redesign, comparator freeze, M9 reserve, lexical ladder, LR-websearch under-claim).
- `research/m8-planning/opus-literature-2026-08-28.md` — literature (EmbedDistill, Wada/SWE, DyT,
  Doc2Query++/HyPE, SAE latent terms, Ethayarajh MEV, fusion optimum, clean-generator licensing).
- `research/m8-planning/codex-adversarial-2026-08-28.md` (+ brief) — adversarial (negatives close
  contradiction, bigram wrong-frame, B→A forgetting, temp/effective-negatives, pool composition,
  reserved-suite weakness, grouped metric, exporter bug, execution order).
- Spot-verified by the orchestrator this session: release container 93,886,950 bytes with dual
  payloads; LEDGER:335 matched-step negatives +0.0112/+0.0111 "NOT IDENTIFIED"; pseudoq.py:84-86
  pool composition 400,001+400,001+12,484+18,844+94,655 = 924,704.
- Known unreconciled detail: B+A chain wall-clock (recipe agent ~30 min from logs vs architecture
  agent 3–4 GPU-h estimate) — measure once before scheduling around it.

Open follow-ups after gating: turn this into `m8/LEDGER.md` pre-registrations (per-item, with
falsifiers and decision rules), file the M7-report under-claim fix (LR-websearch row) as a small M7
addendum task, and start the two background Sonnet sweeps (post-2026-08 teacher candidates;
M9-reserve dataset candidates with licence status).
