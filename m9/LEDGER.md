# M9 ledger — protocol, rulings, and the numbers a rule reads

Mandate: `instructions-m9.md`. Evidence: `m9/PLANNING.md`. Runs: `m9/RESULTS.md`. Closed avenues:
`m9/EXPLORED.md`. Modules: `m9/CODEMAP.md`. Constants a machine reads: `m9/registry.json` and
`results/m9_lock_constants.json`. Nothing here restates those.

---

## §0 M9.0 SCREEN LOCK — 2026-08-30

Everything below is fixed **before** any target encoding, training, retrieval evaluation, or LoTTE
access. `m9src/guard9.py` binds it mechanically: an arm opens a **run token** at launch (lock
commit, branch, lock-file hashes, code hashes, data hashes, arm id) and the result write refuses
unless every one of those is byte-identical at write time. Amending the lock mid-run therefore
kills the run rather than blessing it.

Two adversarial reviews, both gpt-5.6-sol, both read-exclusion honoured and log-audited with zero
reserved-set reads:

- **Pass 1** on the first draft — `research/m9-codex-lock-2026-08-30.md`, **DO NOT COMMIT**,
  7 BLOCKER / 8 MAJOR / 4 MINOR + a post-number-freedom table + an arithmetic audit. Disposition §10.
- **Pass 3** on v3 — `research/m9-codex-lock3-2026-08-30.md`, **v3 is broken; do not let `m9s1`
  open stage B.** It caught a false statement in this ledger: the warm-start ridge λ was described
  as selected on the training residual and had in fact been selected on **SCREEN-3, a dev
  surface**. The anchor run in flight at the time was killed and quarantined; §3.2a now selects λ
  on a training-only holdout. Disposition §12.
- **Pass 2** on the amended lock **and the code that executes it** —
  `research/m9-codex-lock2-2026-08-30.md`, **DO NOT COMMIT. DO NOT SPEND THE 6 GPU-hours.**
  Its finding was that the amendment had moved several failures out of the prose and into the
  code. Disposition §11. Its closing recommendation is what M9.1 was restructured around:
  *"the only defensible next GPU action is a corrected, fully guarded anchor curve — not all nine
  arms."*

**Staging (§3).** Stage A is the pilots plus the anchor arm, its warm-start contrast and the seed
replica. Stage B — the five contrast arms — runs only if the anchor clears the registered
**adequacy gate** (§4.4), and runs in the order **student → prompt → mix → teacher A → teacher B**
(§9.18). `m9src/screen.py:require_predecessors` enforces both the order and the
gate; it is not advice.

### Owner approvals carried in (Dylan, 2026-08-30 planning session)

| ruling | status |
|---|---|
| FineWeb approved as nano's regression-text seed corpus | **NOT EXERCISED — excluded by the mandate's own precondition** (§1.3) |
| M9 execution on branch `m9-work`, frequent commit+push; merges to main need Dylan's go | in force; the guard requires HEAD on `origin/m9-work` |
| Query asset: quality first, 70 MB target, exceedable with logged measured justification | in force. **Unit defined**: decimal MB of *total shipped artifact bytes* (ONNX graph + tokenizer + config), measured by the port pilot — not the theoretical weight product. Weights alone: bge-small nano 67.508 MB, MiniLM nano 46.215 MB |

### M9.0 amendment to the mandate — the teacher-screen surface (**RATIFIED by Dylan 2026-08-30**,
after a plain-language walkthrough of the three options; the challenger had already lost on
measurement, so the surface decided nothing)

`instructions-m9.md` fixes the tuning-dev macro at M7's **six** pinned components for every screen
arm. That is satisfiable for any stella arm and **physically impossible for a challenger teacher**:
the two large components carry 5,233,329 and 6,169,142 documents, so re-encoding full dev in a
challenger's space costs 11.72M document encodes per challenger — **46.5 GPU-hours** for
stella-1.5B and **22.3** for Qwen3-0.6B at this box's measured rates (`results/m9_throughput_probe.json`),
≈69 GPU-hours and ~48 GB of extra vectors for the pair, against a whole-screen budget of ~6.

**Amendment, made before any number was observed and logged here per CLAUDE.md's protocol rule:**

- **Teacher contrasts (arms 2, 3 vs 1) are decided on SCREEN-3** = `nq-250k` +
  `cqadup-programmers` + `cqadup-physics`, at **family weights** 0.50 nq / 0.25 / 0.25 (§4.1).
- **Student, prompt and mix contrasts are decided on all six pinned components (DEV-6),
  equal weight** — exactly as the mandate requires. This costs nothing: stella document vectors
  exist for all six.
- **If a challenger teacher wins**, student/prompt/mix cannot run on DEV-6 in that teacher's
  space. That branch does not proceed on a proxy: it stops and returns to Dylan with the priced
  options (fund the full-dev challenger encode, or accept SCREEN-3 for the remaining arms).

Every stella arm is scored on **both** surfaces, so no contrast ever crosses surfaces.

---

## §1 Data

### 1.1 Query pool (locked)

`work/m9_screen_queries.json` — **242,786** texts, sha256
`77c7ce118feab016bd9991c295b15859ba6af582e557d3dbd98508433fc49238`; row map
`work/m9_screen_rows.npy` into the cached stella target matrix. Provenance:
`results/m9_screen_pool.json`. Built by `m9src/data.py`.

| step | detail |
|---|---|
| source | M8's extended-filter survivor list `work/m8_trainq_texts.json` (337,981 texts, manifest sha `da0f208e…`), re-labelled by source via an order-preserving two-pointer alignment against the identical `train.build_arrays(Cfg(), pool.build())` derivation it was cut from (asserted exact, 337,981/337,981) |
| exclusion | `fever-train` dropped — FEVER is reserved AND stella-disclosed. 95,195 texts removed |
| kept | esci-us 73,030 · hotpotqa-train 81,743 · squad-train 84,713 · mrtydi-en 3,300 |
| screened against | six + dev + untouched-final + **the reserved four** + LoTTE shadow + M9-reserve |
| shape | words: mean 10.64, p50 9, p95 25, max 108. WordPiece: mean 15.32, p95 32, max 143 |

**`nqopen` and `triviaqa` are EXCLUDED from all of M9** (not deferred). They are decontaminated
against M7's protected set only; the M8 *extended* screen needs the `short_whole_index`
containment structure, which the persisted fingerprint npz does not carry and which can only be
rebuilt inside `m8src/protected_filter.py`, the one module holding the G2 capability. Admitting
220,632 texts — **47.6%** of the non-FEVER query corpus — after the screen had selected a recipe
would mean selecting on one training distribution and building on another. The cost is bounded:
the abundant resource for a LEAF-style recipe is *document* text, of which §1.2 has 6.15M
pre-screened rows, and the mix arm is what prices it. **Reopens in M10 only.**

**Long-query training coverage is ABSENT and is not repaired by anything in M9.1.** The pool's
longest query is 108 words against ArguAna's 174-word average, and training on long
`"passage: "`-prefixed documents does **not** exercise long-*query* input behaviour. The head+tail
probe **will not run in M9**; first-512 truncation is stated as a limitation. `heldout-longq`
(55 queries) is descriptive only and **may not change any M9.2 decision** — recipe, checkpoint,
teacher, student, prompt, mix, grid or weight.

### 1.2 Document pool (locked)

The frozen stella document pool `work/pool/stella-400M-v5/` (6,169,142 rows × 1024 fp16, per-store
`id_sha256` pinned), **minus** the `fever-pos` span [808389, 820938) and **minus** the 7,190
`banned_pool_rows` (B2 mask, pool-identity bound). Union measured, not subtracted:
**6,149,679 eligible rows** (276 banned rows fall inside the FEVER span).

Arm-6 **candidate list**: **400,000** eligible rows drawn by
`numpy.random.default_rng(9).choice(eligible, replace=False)` **in draw order, not sorted** — rows
sha256 `216e8783c52df2ea220c526c9063dc8d70dff3e352c3509d045850aff3495e98`. (Sorting global pool row
ids would make any prefix a low-row prefix — `esci-prod` first — so the sample would be
store-biased rather than uniform.) The materialized schedule consumes **189,002 documents, each
seen exactly once**, leaving 210,998 candidates of headroom. Doc targets under stella are the
existing pool vectors, so they cost zero teacher compute. WordPiece length of a
`"passage: "`-prefixed candidate: mean 94.57, p50 64, p95 306, max 512.

*What arm 6 can and cannot answer:* it estimates single-pass regression over the **pre-screened M7
document pool** (Wikipedia/Amazon-product/SQuAD/mrTyDi). It is **not** evidence about LEAF-style
broad web-text regression, because FineWeb is excluded (§1.3).

### 1.3 FineWeb — EXCLUDED (verification)

The mandate permits FineWeb only if **pre-existing, non-reversible reserved-set fingerprint
artifacts** support exact + near-**document** checks against DBpedia and both reserved CQA slices,
and forbids creating such fingerprints now. Verified:

| artifact | what it holds | covers the requirement? |
|---|---|---|
| `work/decontam/m8_protected_query_index.npz` | `exact` (78,620 u64) + `grams` (4,220,227 u64) over protected **queries**; `short_whole_index` not persisted | no — query-side only |
| `results/m7_decontam.json` R3 block | **counts** of train-doc overlap vs dbpedia / fever / cqa-untouched | no — the streamed document index was discarded |
| `work/decontam/banned_pool_rows.npy` | a row mask over *our* pool + per-store `pool_id_sha256` | no — not a reserved-corpus fingerprint |

No persisted reserved-set **document** fingerprint exists (`m7src/decontam.py` writes
`kept.json`/`summary.json` only; `m8src/protected_filter._cqa_index()` builds its CQA document
index in memory per call). Building one would open reserved corpora. **FineWeb is excluded from
M9. It may reopen in M10** — not within M9, because the M9.4 reserved access happens after all
training is finished and so could not change it anyway.

### 1.4 R1 / R2 / R3, verbatim

Fingerprints (`m7src/decontam.py`): blake2b-64 word hashes over `norm_words` (lowercased
alphanumeric runs); `exact_u64` = blake2b-64 of the normalized text; near-duplicate = polynomial
rolling word-**8**-grams, bottom-**32** sketch, **≥ 8/32** shared (est. Jaccard ≥ 0.25); short-query
paths add word-**4**-grams for 4–7-word queries plus a verbatim containment index.

- **R1 (removes).** A TRAIN pair whose *query* exact- or near-matches any protected query.
  Protected = the six + dev + untouched-final, extended by M8 with the reserved four, the LoTTE
  shadow and the M9-reserve. M9 inherits the extended screen and re-runs nothing.
- **R2 (removes).** A TRAIN pair whose *positive document* exact/near-duplicates a document of
  **the six**. Output: `work/decontam/kept.json`.
- **R3 (measures, does not remove).** Overlap of TRAIN positive documents against dev and
  untouched-final corpora, disclosed in `results/m7_decontam.json`. M9 adds no R3 pass and opens no
  reserved bytes; the M7 disclosure carries forward verbatim, including dbpedia-entity near-dup
  rate 9.298% and fever-untouched 11.178%.

---

## §2 Students, teachers, environment — all pinned

| student | repo @ revision | params (incl. head) | fp16 weights |
|---|---|---|---|
| **anchor** | `BAAI/bge-small-en-v1.5` @ `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | 33,754,240 | 67.508 MB |
| challenger | `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | 23,107,456 | 46.215 MB |

Both share the `bert-wordpiece-30522` tokenizer, so every token count in §3 is student-invariant.
Architecture: backbone → mean pooling over the attention mask → `Linear(hidden, 1024, bias=True)`
(PyTorch default init under seed 0) → L2 normalize (eps 1e-12). **Size does not discriminate**;
the student decision is made on quality alone.

| teacher | repo @ revision | dim | pooling | encode path |
|---|---|---|---|---|
| **incumbent** | `NovaSearch/stella_en_400M_v5` @ `ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20` | 1024 (`2_Dense_1024`) | mean | M7's frozen `m7src/teacher.py` — untouched |
| challenger A | `NovaSearch/stella_en_1.5B_v5` @ `7817065102fd9e1b031fe874e910c01f40b2f001` | 1024 (`2_Dense_1024`) | mean | `SentenceTransformer` |
| challenger B | `Qwen/Qwen3-Embedding-0.6B` @ `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 1024 | last-token | `SentenceTransformer` |

The challengers use `SentenceTransformer` because M7's teacher module implements neither
last-token pooling nor per-repo dense-head selection. That is only admissible because the ST path
**reproduces** the frozen one: on 64 real screen-pool texts through stella-400M both ways,
min-cos **0.99959**, max-abs **1.45e-4** (fp16-vs-fp32 rounding) — `m9src/teacher9.parity_vs_frozen`.

Environment pins, framework versions, per-component qid manifests and code hashes:
`results/m9_lock_constants.json`. `M7_ENCODER` is **assigned, not defaulted**, and a conflicting
value in the environment is rejected rather than inherited.

---

## §3 The screen — a batch pilot, then seven sequential arms

| stage | id | varies | teacher | student | prompt | mix | seed | decided on |
|---|---|---|---|---|---|---|---|---|
| **A** | `m9s1` | anchor | stella-400M | bge-small | (b) | query-only | 0 | both |
| **A** | `m9s1c` | **no warm start** (diagnostic, decides nothing) | stella-400M | bge-small | (b) | query-only | 0 | DEV-6 |
| B | `m9s1b` | training seed (**reported**, read by no rule) | stella-400M | bge-small | (b) | query-only | **1** | DEV-6 |
| ~~B~~ | ~~`m9s2`~~ | **WITHDRAWN** (§9.20) — stella-1.5B measured −0.00229 against a +0.010 bar | | | | | | |
| ~~B~~ | ~~`m9s3`~~ | **WITHDRAWN** (§9.20) — a smaller nominal edge than the one that just lost | | | | | | |
| B | `m9s4` | student | selected | MiniLM-L6 | (b) | query-only | 0 | DEV-6 |
| B | `m9s5` | prompt | selected | selected | (a) | query-only | 0 | DEV-6 |
| B | `m9s6` | mix | selected | selected | selected | 70/30 **by token** | 0 | DEV-6 |

Order is fixed and enforced in code. **The batch-32-versus-128 pilot was registered and then
removed before any arm ran**: two matched epochs give batch 32 four times the optimizer updates
and compress a separate warmup+cosine schedule into a miniature, so it would have measured early
optimization speed rather than the batch size that wins at final dose. LEAF's `bs=32` finding was
made at roughly a hundred times this dose. Batch size is locked at **128**. If a teacher challenger
fires, the run stops (§0 amendment).

**Role byte-templates (literal, locked).**
- QUERY policy **(b) — baseline**: student input = raw query bytes. Teacher target =
  `"Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "`
  + query (verbatim from stella's `config_sentence_transformers.json`; Qwen3's own string has no
  trailing space and is used verbatim for that teacher).
- QUERY policy **(a)** — alternate: student input = the same s2p string + query; teacher target
  unchanged, so arms 1 and 5 share one target cache and the contrast is purely student-input.
- DOC (arm 6 only): student input = `"passage: "` + document text; teacher target = raw document
  bytes. Disjointness of the two student input sets is **asserted on the pinned tokenized inputs**
  at plan-build time, not assumed from the prefix string.
- **Challenger branch:** if a challenger were selected, arm 6's document targets would be these
  same 111,906 raw documents re-encoded in that teacher's space (priced at 26.6 min for Qwen3,
  16.0 min for stella-1.5B) — the frozen pool vector is a stella-only shortcut.

### 3.1 Dose — identical optimizer updates and identical non-pad tokens

| quantity | value |
|---|---|
| query texts | 242,786, 16 epochs |
| examples (arms 1–5) | **3,884,576** |
| batch size | **128** examples (subject to the P pilot) |
| optimizer steps | **30,349** — the last batch is 32 examples and **is** trained, at its natural size |
| **T_base non-pad tokens** | **59,507,872** (16 × 3,719,242); per-step budget **1,960.8** |
| checkpoints | **7,588** · 15,175 · 22,762 · **30,349** — the step at which each quarter of the examples *completes* (the first was one step early in the draft). Decision reads the last; the last two feed the rank-stability rule, **by step id, never by position** |
| checkpoint **surfaces** | SCREEN-3 at all four (0.65 GB of document vectors — nearly free, and it carries the dose-response curve); **DEV-6 at the FINAL checkpoint only.** A DEV-6 pass streams 23.3 GB, and *interleaved with training* it fragments the CUDA allocator badly enough to cost more wall-clock than the training it measures (§9.15). Reading it once, at the end, removes the interleaving. Consequence, fixed before any arm ran: the adequacy gate and the rank-stability rule both read **SCREEN-3**, the only surface with four points; the decision contrasts read **DEV-6 at the final checkpoint**, which is where they were always defined |
| seed | 0 (data order, init, dropout); arm 1b seed 1 |
| optimizer | AdamW, β (0.9, 0.999), eps 1e-8, weight decay 0.01 on tensors with dim > 1 and 0.0 otherwise, grad-clip 1.0 global norm, no gradient accumulation |
| LR | `warmup: 1e-4·(step+1)/910`; then `1e-5 + 0.5·(1e-4−1e-5)·(1+cos(π·(step−910)/(30349−910−1)))`. The `−1` matters: with `steps−warmup` the last executed step is short of the endpoint and `lr_final` is never reached |
| precision | bf16 autocast backbone; **loss in fp32**; retrieval matmul fp32 from fp16 document memmaps |
| max sequence | 512, dynamic padding to the longest member of the batch |
| epoch order | one `default_rng(seed)`, a fresh `permutation(n)` per epoch, concatenated |

**Two arms are token-matched rather than example-matched**, because their inputs are not the same
length as the baseline's and an example-matched version would confound the factor under test with
the token dose (Codex pass 1 BLOCKER-2, pass 2 BLOCKER-2):

- **arm 5 (prompt policy (a))** prepends ~20 tokens to every query. Sixteen full epochs would be a
  *larger* non-pad dose than the baseline, so instead it holds **30,349 optimizer updates and
  59,507,872 non-pad tokens** and sees correspondingly fewer presentations of the same pool.
- **arm 6 (mix)** holds the same 30,349 updates and 59,507,872 tokens, split **70/30 by token**:
  41,655,512 query and 17,852,365 document tokens realized — a **0.30000** document share and
  five tokens over `T_base` in total. Batch size floats: mean **95.83**, range 68–124.
  Loss is the plain mean over the batch — **no role weighting**.

The token budget is tracked **cumulatively**, not per step: each role fills until its running total
reaches its share of the tokens due by the end of that step, so per-step rounding cannot accumulate
over 30,349 steps. The batcher asserts, rather than hopes, that there are exactly 30,349 non-empty
batches and that no stream runs dry — a short arm must never be able to look like a full one. Both
schedules are **materialized and their realized counts recorded at lock time**
(`results/m9_lock_constants.json` → `token_budget.students.*.mix_schedule`), not estimated.

Baseline arms keep fixed 128-example batches; their texts are one length family, so per-step tokens
are near-constant. The asymmetry is deliberate, and the realized per-step token distribution is
reported for every arm.

### 3.2 Objective (phase 1)

`loss = mean_over_batch( sum_over_dim( (normalize(student_out, eps=1e-12) − target)² ) )`, target =
the L2-normalized teacher vector, computed in fp32. No auxiliary term (LEAF's ablation rejected
them; MSE+cosine is affine-redundant under normalized outputs). Phase 2 is out of scope for M9.1.

### 3.2a Head warm start — closed-form, every arm (M9.0 amendment, before any arm ran)

The head is **initialized to the closed-form ridge solution** from the frozen backbone's
mean-pooled outputs to the teacher targets, then trained normally with everything else:
n_fit **60,000** screen-pool texts, seed **21**, ridge on a trace-normalized Gram, bias column
included.

**λ is selected on a training-only holdout** (`m9src/warmfit.py`): fit on the first 50,000 of the
60,000, score the remaining 10,000 under the **actual normalized objective**
`‖normalize(XA) − Y‖²`, locked grid {1e-6 … 1}, ties to the larger λ. No dev surface is read, and
`nano.warm_start_head` refuses to run until that selection exists and matches the registry.
The v3 text claiming λ came from the *training residual* is withdrawn (§9.12): the probe had in
fact taken the SCREEN-3 argmax, and a residual criterion would not have rescued it either — a ridge
residual is monotone in λ, so it merely picks the bottom of the grid.

Why, measured before any arm ran (`results/m9_head_probe.json`, a `-diag` artifact no decision may
cite): a frozen bge-small backbone plus this head alone scores **0.3463** on SCREEN-3 — **50.8%**
of the 0.6822 teacher ceiling — while a *random* head after 2,000 trained steps reached **12.4%**.
At ~1% of LEAF's dose, a random head spends a large share of the entire budget re-deriving a linear
map that has a closed form, and a screen run that way would partly rank arms by how fast each one
recovers from its own initialization rather than by the factor under test. **The estimand this creates, stated rather than assumed away.** Every arm gets the identical
*algorithm*, but not an identical *effect*: each arm fits its own head, in its own feature space,
under its own teacher's targets, so a contrast measures **the factor plus its refit**, end to end.
The teacher screen becomes "index quality *and* linear decodability after a calibrated head fit";
the prompt arm is "prompt plus refit in that prompt's feature space"; the mix arm is warm-started
on query targets only, so it tests *adding documents to a query-calibrated start*, not an
independently optimized mixed recipe. Every M9 claim carries that wording. The v3 sentence "so no
contrast moves" is withdrawn (§9.13).

**Stage-0 dose.** The warm start is an extra supervised phase — 60,000 frozen-backbone forwards,
60,000 teacher-target accesses and a 385×385 solve — and it is **not** part of the registered SGD
dose. Every arm reports it separately, so the retention curve can be read against both the SGD
budget and total compute, and so it is never silently compared to LEAF's random-head recipe as
though the doses matched.

Two consequences worth stating. **The model now has no random initialization at all** — pretrained
backbone plus a deterministic head — so the seed controls only data order and dropout, and the
seed-replica floor F measures exactly that and nothing else. And **arm `m9s1c`** repeats arm 1
*without* the warm start at the identical dose: a registered **DIAGNOSTIC contrast** that prices
the warm start at the screen dose and feeds M9.3's budget estimate. It decides nothing and is
marked decision-ineligible in the registry.

### 3.3 fp16 target cache acceptance

Numerical only, and it is the **only** fp16 gate: against live fp32 on a locked sample of 10,000
screen-pool texts stratified into 10 equal-count WordPiece-length deciles (seed 11, ids
materialized and hashed at M9.0), require min-cos ≥ **0.9999** and max absolute coordinate error
≤ **1e-3**. Failure → fp32 targets for every arm, logged. The earlier "retrieval macro shift"
clause is **withdrawn**: training texts carry no qrels and cannot produce a SCREEN-DEV macro, so
the clause was unexecutable. The document side's fp16 storage is M7's frozen convention, unchanged
and out of M9's scope.

---

## §4 Decision surfaces and rules

### 4.1 Surfaces

- **DEV-6** = the six pinned M7 dev components, equal weight, 20,152 queries. Decides student,
  prompt, mix, batch size, and the seed floor.
- **SCREEN-3** = `nq-250k` + `cqadup-programmers` + `cqadup-physics`, **family weights 0.50 /
  0.25 / 0.25**. Decides the teacher only. Family weighting exists because equal component
  weighting would give CQADupStack — 35.7% of the queries — 66.7% of the macro, while the
  confirmatory six contain no CQA at all and the reserved NDO-3 macro is 50% CQA. Decision
  sensitivity under equal-component and query-pooled weights is reported, and a teacher swap
  additionally requires **direction stability across all three weightings**.

Both surfaces stay labelled DEV. Per-component **ordered qid sha256 and counts are pinned** in
`results/m9_lock_constants.json`; the statistic asserts every arm matches the manifest exactly, so
two systems cannot quietly agree to drop the same hard queries. The dev-reuse counter continues
(M7+M8 cumulative 494); `m9src` re-derives it, never hand-edits it.

### 4.2 Statistic and thresholds

`m9src/screen_stats.py`, one implementation. Align identical qids per component against the pinned
manifest; per-query nDCG@10 differences; **B = 20,000** bootstrap replicates resampling n_d queries
with replacement within each component; component means combined at the surface's weights; fixed
seed 0; resample indices drawn once and shared across every contrast on that surface. The reported
bound is the empirical quantile named by the rule.

**Power basis, pre-registered before any arm ran** (`results/m9_power_prereg.json`): over **2,031**
system pairs in M7/M8's committed dev per-query dumps, the per-query nDCG@10 *difference* SD has
p50 0.085 and **p90 0.126**; the corresponding macro SE is **0.00205** on SCREEN-3 and **0.00259**
on DEV-6. At the p90 planning value a 97.5% one-sided bound clears zero at a point estimate of
**0.0040 / 0.0051**. Caveat recorded: those pairs are mostly close variants of one table family
and may understate the spread between two different student backbones.

| decision | surface | passes iff | else |
|---|---|---|---|
| **teacher swap** | SCREEN-3, family weights | point ≥ **0.010** AND 1.25%-quantile lower bound > 0 (Bonferroni over the two challenger contrasts) AND direction stable across all three weightings | **stella-400M stays.** If both challengers pass, the larger point wins; on an exact tie, stella stays |
| **batch size** | DEV-6 | macro(bs32) − macro(bs128) ≥ **MDE** AND 2.5%-quantile lower bound > 0 | **batch 128** |
| **student** | DEV-6 | point ≥ **MDE** AND 2.5%-quantile lower bound > 0 AND sign agrees at the last two checkpoints | **bge-small** |
| **prompt** | DEV-6 | same form | **policy (b)** |
| **mix** | DEV-6 | same form | **query-only** |

**MDE = 0.0056**, one number, fixed at M9.0. It is 1.96 × the DEV-6 macro SE implied by the p90
per-query difference SD (0.1195) over the 2,031 historical dev contrasts:
SE = (1/6)·√(Σ_c s²/n_c) = 0.002876 → 0.005636, rounded to 0.0056.

**Withdrawn before any arm ran:** the earlier `max(0.0051, 2F)` form with F a seed-replica range.
A single absolute difference between two seeds is one half-normal draw, not an estimated σ — it can
sit near zero under large real seed variance or inflate the threshold arbitrarily, and calling it a
noise floor would repeat `m8/CODEMAP.md` pitfall 18 at K = 2 instead of K = 3. Arm `m9s1b` still
runs in stage B, and its seed sensitivity is **reported beside every contrast**; no rule reads it.

**Rank stability.** A student/prompt/mix decision that adopts a challenger must also have the same
sign at checkpoints 3 and 4. The screen runs at ~1% of LEAF's example dose and ~0.5% of its
optimizer updates, so an early-training ranking is not automatically the final-dose ranking; the
checkpoint curve is the registered instrument for saying whether it is. A sign flip between the
last two checkpoints ⇒ **not stable at screen dose** ⇒ the default is taken and the arm is
recorded as diagnostic.

**Scope of every screen verdict.** These are *artifact-specific* selections at the screen dose
under seed 0 — not claims that a recipe is superior. The bootstrap quantifies query resampling
conditional on one fitted artifact and contains **no** training-uncertainty term at all. No screen
result may be reported as resolved, confirmed, or significant.

### 4.4 The adequacy gate — stage A → stage B

Read once, on the anchor arm, on **SCREEN-3** (the only surface with four checkpoints once DEV-6
moved to the final one — §3.1), before any challenger teacher is encoded:

| condition | threshold | why |
|---|---|---|
| retention at the final checkpoint | ≥ **0.60** of the 0.68223 SCREEN-3 ceiling | below this the student sits so far from the teacher surface that a contrast between two arms is dominated by early imitability rather than by the factor under test |
| late slope, macro(ckpt4) − macro(ckpt3) on SCREEN-3 | ≤ **0.02** | a curve still climbing steeply at the last checkpoint is an early-training snapshot whose ranking need not survive to final dose |

**PASS** → stage B runs as registered. **FAIL** → stage B does **not** run; M9.1 reports the anchor
curve, the warm-start contrast and a priced dose-response extrapolation, and the
teacher/student/prompt/mix questions go back to Dylan with a budget attached rather than being
answered at an inadequate dose. Both outcomes are actions, which is the test `m8/FINDINGS.md` §4
requires of any screen worth building.

**What this gate is not.** It is a **budget trigger**, not a calibrated ranking-adequacy test.
Neither threshold is tied to a measured rank-concordance rate; 0.60 was chosen as "clearly on the
teacher surface" and 0.02 as "no longer climbing fast", and both are judgement. The slope is read
where the cosine schedule has already decayed the LR to 1e-5, so a small late gain can mean the
schedule shut learning down rather than that the representation converged — and any *negative*
slope also passes it. It decides only whether spending stage B is worth it; it certifies nothing
about whether a stage-B contrast would rank the same way at final dose. The verdict is
**recomputed** by `require_predecessors`, never read from a stored boolean, and the retention
denominator is the registry's pinned ceiling with its artifact hash checked.

### 4.3 What a screen result may NOT do

Ship, set a bar, touch the six, or touch LoTTE. Its only outputs are the M9.2 recipe selection and
the retention-vs-dose curve that prices M9.3.

---

## §5 Descriptive reporting fixed at lock

**Length bins** (raw query word count): `[1,5] · [6,10] · [11,20] · [21,50] · [51,∞)`.
**Fragmentation bins** (subwords per word, punctuation stripped **before** tokenizing —
`m8/CODEMAP.md` pitfall 14): `[1.0,1.2) · [1.2,1.5) · [1.5,2.0) · [2.0,∞)`.
Reported per arm with tokenizer fertility, `[UNK]` rate, truncation rate, retained-token fraction.
Any pooled within-dataset slope carries its per-component variance share and a leave-one-out slope
before it is believed (pitfall 13).

---

## §6 M9.1 pilots — acceptance fixed at lock

| pilot | pass condition |
|---|---|
| **bridge-tolerance dry run** | `m9src/bridge_dryrun.py` freezes a dev-side per-query reference for the bge-small anchor on DEV-6, then re-derives it end to end in a **fresh process**: zero missing / extra / reordered qids and max per-query \|Δ nDCG@10\| ≤ **3e-4**. It runs on DEV, never on the six — the six-set bridge is phase 1 of the sole six-set transaction and spending it early would consume the access |
| **ONNX export** | both students, real weights, **opset 17**, zero custom-domain ops; parity vs torch on **every** example of the locked sample: min-cos ≥ **1 − 1e-4**, max-abs ≤ **1e-3** |
| parity sample | **512** texts: 256 query-pool + 256 documents, **51/51/51/51/52** per length bin per side (longest bin first, shortfall flowing to the next bin down), seed 12, ids materialized and hashed at M9.0 |
| **fastembed** | `TextEmbedding.add_custom_model()` accepts the description for any student that can still win; full serving parity needs a published repo path and is M10's step. Recorded either way |
| **artifact size** | total shipped bytes (ONNX + tokenizer + config) measured and compared against the 70 MB decimal target |
| **fp16 target gate** | `m9src/fp16_gate.py`, and `screen.query_targets()` refuses to hand out fp16 rows until it has passed. §3.3's thresholds on the locked 10,000-text sample |
| **throughput** | the anchor arm **is** the measurement — it reports realized ex/s, tokens/step and per-checkpoint evaluation seconds. There is no separate pilot and **no dose fallback**: a single dose is registered, and if it proves unaffordable that is a finding reported to Dylan, not a threshold a session can steer itself across |

---

## §7 LoTTE — untouched at M9.1, and it is a veto, not a confirmation

LoTTE-clean (7 slices, 20,122 q, macro over slices, never pooled) is **not read during M9.1**; its
batch manifest is an M9.2 artifact. The rule is fixed now so it cannot be chosen after an outcome:
**read #1** scores, in one atomic batch, two already-trained checkpoints pinned by hash — the
screen-selected recipe and the registered fallback (stella-400M × bge-small × prompt (b) ×
query-only) at equal dose. It is a **non-inferiority veto**: the selection stands unless the
selected recipe's 7-slice macro is worse by more than 0.004 **and** the 97.5% one-sided paired
bootstrap upper bound on (selected − fallback) is below −0.004; then the fallback is adopted and
the screen result is recorded as vetoed. Adoption may not trigger any retraining choice. Read #1
may also select the fusion weight from the fixed grid. **Read #2** is audit-only, pre-freeze. No
third read. Forum-heaviness (same family as reserved CQA, and as SCREEN-3's 50% family weight) is
disclosed.

---

## §8 Protocol carried forward, unchanged

Reserved four unspent · `results/perquery.json` irreplaceable, never rewritten · frozen comparator
pairing · one six-set transaction with the bridge as its phase 1 · Holm at family α = 0.025 over
the two confirmatory contrasts · Sonnet-only research subagents · every review brief carries the
reserved read-exclusion and the log is grepped afterwards · `m8/CODEMAP.md` read before writing
code · watch-long-runs checklist for anything > 10 min · **`git status m7/ results/` after running
anything out of `m7src/`**.

`m9src/` imports `m7src` and `m8src` and edits neither. `m9src/m9base.py` installs `paths_guard` at
import. No G2 allowlist entry is created for M9.1; none is needed.

---

## §9 Withdrawn claims and corrections

*(never compressed away)*

1. **Withdrawn 2026-08-30, before any arm ran:** "the mix arm's documents cover the long-query
   limitation." False — long documents do not exercise long-*query* input behaviour. M9 has **no**
   long-query training coverage; see §1.1.
2. **Withdrawn 2026-08-30:** the fp16 target-cache "query retrieval macro shift ≤ 1e-4" clause.
   Unexecutable — training texts carry no qrels. Replaced by §3.3's numerical gate.
3. **Corrected 2026-08-30:** the first draft's arm-6 document split (582,686) was arithmetically
   wrong (72,836 × 8 = 582,688) and the whole example-matched design was replaced by §3.1's
   token-matched design.
4. **Corrected 2026-08-30:** the first draft's parity sample asked for 64 per bin per side across
   five bins (320) while also asking for 256 per side. Now 51/51/51/51/52.
5. **Amended 2026-08-30, before any arm ran:** the head is warm-started in closed form (§3.2a).
   The first draft trained it from a random init, which the head probe showed costs most of an
   affordable budget. LEAF's own recipe trains from random init — at roughly a hundred times this
   dose, where it does not matter.
6. **Withdrawn 2026-08-30:** the searchsorted estimate of arm 6's document count (111,906) and its
   53-token overshoot. The schedule is now materialized by the real batcher at lock time and its
   realized counts recorded, so there is no estimate left to be off by.
7. **Withdrawn 2026-08-30:** `MDE = max(0.0051, 2F)`. See §4.2.
8. **Withdrawn 2026-08-30:** the batch-32-versus-128 pilot and both fallback doses. See §3 and §6.
9. **Corrected 2026-08-30:** checkpoint 1 was 7,587; four epochs complete at step **7,588**.
10. **Corrected 2026-08-30:** the document candidate list was **sorted** by global pool row id, so
    any prefix of it was a low-row prefix — `esci-prod` first — and arm 6's documents would have
    been a store-biased sample rather than a uniform one. It now keeps RNG draw order.
11. **Corrected 2026-08-30:** the cosine LR denominator was `steps − warmup`, so the schedule
    stopped one step short and never reached `lr_final`.
12. **WITHDRAWN 2026-08-30, and the run that used it was killed:** "λ is selected on the training
    residual, never on a dev surface" (§3.2a, v3). False. `m9src/head_probe.py` evaluated every λ
    on **SCREEN-3** and took the argmax — a dev surface — and the artifact's own `_status` field
    said the number was optimistic for exactly that reason. Caught by Codex pass 3. The anchor arm
    then in flight (≈11,000 of 30,349 steps) was **killed and quarantined**, λ selection moved to a
    training-only holdout under the real normalized objective, and the anchor re-run from scratch.
    `results/m9_head_probe.json` remains as a `-diag` artifact and no rule may cite it.
13. **WITHDRAWN 2026-08-30:** "every arm gets the identical treatment, so no contrast moves"
    (§3.2a, v3). The algorithm is identical; the effect is not, because each arm refits its head in
    its own feature space. §3.2a now states the estimand instead.
14. **Corrected 2026-08-30:** the adequacy gate was described as a ranking-adequacy test. It is a
    **budget trigger** with two judgement thresholds; §4.4 says so.
15. **Recorded 2026-08-30 (operational, not a claim), and corrected twice:** interleaving a DEV-6
    evaluation with training collapses the training rate from ~2,000 to ~340 ex/s — the card pinned
    at 9,985/10,240 MiB, power down from 288 W to 150 W at 96% "utilisation", which is CLAUDE.md's
    allocator-thrash signature, not work. `torch.cuda.empty_cache()` after each evaluation was
    tried first and **did not fix it**: the arena is fragmented, not merely cached. The fix is
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, pinned in `m9src/m9base.py` so it is part
    of the lock rather than an operator's export, together with reading DEV-6 once at the end
    instead of interleaving it. Two anchor attempts were killed and re-run over this.
18. **Amended 2026-08-30, before any teacher number was observed:** stage B's order is
    student → prompt → mix → the two teacher arms, not teacher-first. The mandate ordered the
    teacher arms first because a swap changes arm 4's baseline; that reason is preserved by a
    **stronger rule** rather than by ordering — a challenger win **voids** every student/prompt/mix
    decision taken under the incumbent and returns the milestone to Dylan, which is already what
    §0's conditional branch requires. What the reorder buys: the three arms that actually set the
    M9.2 recipe cost ~35 minutes each and need no teacher encode, while the two teacher arms cost
    2.1 and 1.4 hours and can only either change nothing or stop everything. No teacher comparison
    existed when this was written — the stella-1.5B arm had died at `loss nan` inside its own
    encoder (§9.19).
19. **Recorded 2026-08-30:** encoding with **stella-1.5B in fp16 produced NaN for 24 of 242,786
    real query texts** (0.01%) — an overflow in its attention path, invisible to a 256-row sampled
    norm check and surfacing only as `loss nan` a thousand steps into a two-hour arm. Two fixes:
    challenger teachers now encode in **bf16** (fp32's exponent range at fp16's memory cost), and
    `build_plan` checks **every** target for finiteness and unit norm instead of a corner of them.
17. **Corrected 2026-08-30:** the session manifest was keyed on the lock commit as well as the
    fingerprint, so committing an arm's own result would have voided every arm already run. It is
    keyed on the **fingerprint** alone; `check_state()` still requires the guarded files clean and
    HEAD pushed on `m9-work`. The anchor was re-run under the corrected guard — deterministically,
    reproducing checkpoint 1 to six decimals for the third time.
16. **Corrected 2026-08-30:** the adequacy gate and the rank-stability rule now read **SCREEN-3**
    rather than DEV-6, because DEV-6 no longer has four checkpoints. The decision contrasts are
    unchanged — they always read the final checkpoint, and DEV-6 is still evaluated there.

20. **WITHDRAWN 2026-08-30, on measurement and the owner's ruling: the teacher screen.** `m9s2`
    (stella-1.5B) ran to completion before an unrelated guard failure voided its artifact, and its
    checkpoint curve sits **behind** the stella-400M anchor at every point —
    0.44649 / 0.48034 / 0.49225 / **0.49775** against 0.44814 / 0.48121 / 0.49459 / **0.50004**, a
    final delta of **−0.00229** where the swap rule needs **+0.010**. A teacher 3.75× larger with a
    materially better MTEB score (0.5837 vs 0.5609) distils *worse*, which is M8's `T1` finding
    generalising from tables to towers. Qwen3-0.6B's nominal edge (+0.004) is smaller than the one
    that just lost (+0.023). **stella-400M stands as the teacher by default and by Dylan's product
    preference — one document model and one collection shared by `zero` and `nano` — not by a
    registered measurement.** The figures above are diagnostic: they informed a decision not to
    spend two GPU-hours, and no M9 claim rests on them.
21. **Recorded 2026-08-30 (operational):** editing `m9src/guard9.py` mid-screen voided every arm in
    flight, because the guard's own module sits in the `protocol` scope it enforces. Care had been
    taken over `LEDGER.md` and `registry.json` and none over the guard itself. Cost: the
    eligibility of five completed arms, though their document caches and numbers survived. **All
    guarded-file edits are batched into windows between arms from here.**

---

## §14 M9.2 — the recipe lock for the seven-day build

Lives in **`m9/M92_LOCK.md`**, deliberately outside this file: a guarded protocol file cannot be
edited while an arm is in flight, and the build's recipe needed drafting while the screen ran.
Constants a machine reads are generated into `work/m9long/config.json` by `m9src/make_config.py`
from the screen's decision artifact — **not** by parsing `M92_LOCK.md`, which is the human record.
It shows the arithmetic for every constant. The build's own scope (`build`) pins the trainer, the
watchdog, the config and the manifest.

Registered there and not repeated here: dose 5/5/90 by token · 113 examples a step · warmup →
stable → decay-on-demand with a 59,507,872-token cooldown · the four-part kill envelope · the
first-eval gate against this run's own step-0 baseline · stop-on-evidence rather than on a horizon.

---

## §15 Codex review #5 — the seven-day trainer

`research/m9-codex-longrun-2026-08-30.md`. Verdict **DO NOT LAUNCH**, seven blockers, all actioned.
The first was that a fresh run **crashed before its first optimizer step** (`warmfit.ARTIFACT` did
not exist), which proved the fresh-start path had never been executed end to end. The most
consequential was that the loss was **not the plain example mean its own docstring claimed** —
token shares set the batch composition and were then applied again as objective weights, giving a
95-token document ~6× the weight of a 16-token query. Also: integrity compared two copies of a
*declaration* rather than bytes; resume silently accepted a different recipe; decay was not
resumable; the mandated kill envelope did not exist; and the trainer ran outside the guard.
Its dose recommendation (5/5/90) and cooldown scale (the anchor's own 59.5M tokens) were both
adopted over my drafts, and its advice to skip the capacity probe was taken.

---

## §16 Codex review #6 — the watchdog

`research/m9-codex-watchdog-2026-08-30.md`. Verdict **DO NOT LAUNCH UNATTENDED**, six blockers, all
actioned. The first defeated every stopping rule in §15's work: the trainer exits *normally* after
a first-eval failure, a regression, a plateau or a completed cooldown, and the watchdog — seeing
only "no PID" — restarted it, after which the first-eval gate could never fire again. The second is
the sharper lesson: **the throughput guard written specifically for the measured `m9s2` slowdown
would not have caught `m9s2`**, because it used the cumulative session mean, which takes ~five days
to fall below half after a 5× drop on day three and re-baselines onto the degraded rate at every
restart. Also: `exists()`-then-write is not a lock; a fresh-start wedge was invisible because both
staleness checks were guarded on a heartbeat that did not exist yet; the plateau rule compared
evaluations ~164M tokens apart while demanding a 1B-token span, so it could never fire; and a crash
between checkpoint and history append lost that evaluation permanently.

---

## §10 Codex lock review — disposition

`research/m9-codex-lock-2026-08-30.md`. All 19 findings + 7 post-number freedoms actioned.

| finding | disposition |
|---|---|
| B1 decision surface contradicts the mandate | **adopted, as the reviewer's preferred structure** — §0 amendment: teacher on SCREEN-3, student/prompt/mix on DEV-6, challenger branch stops for Dylan |
| B2 arm 6 violates the equal-token budget | **adopted** — §3.1 token-matched fixed-compute contrast |
| B3 guard does not freeze the protocol | **adopted** — run tokens in `m9src/guard9.py`; branch pinned to `m9-work`; diagnostic runs forced to a `-smoke` id and marked ineligible |
| B4 arms not reproducibly specified | **adopted** — §2 revisions, §3.1 LR formula / partial-batch / epoch order, `M7_ENCODER` assigned not defaulted, challenger arm-6 targets defined and priced |
| B5 statistic accepts a shrunken surface | **adopted** — pinned ordered qid hashes; single arm-report function with fixed orientation, shared resamples and the two-challenger outcome table |
| B6 parity sample arithmetic impossible | **adopted** — 51/51/51/51/52; the fp16 retrieval clause withdrawn (§9.2) |
| B7 47.6% of query text deferred | **adopted** — `nqopen`/`triviaqa` excluded from all of M9 (§1.1) |
| M1 CQA over-weighted | **adopted** — family weights + three-weighting direction stability |
| M2 0.004 threshold unstated | **adopted** — empirical power basis measured pre-training; MDE = max(0.0051, 2 × measured seed floor) |
| M3 early-dose ranking | **adopted** — dose raised to 16 epochs, rank-stability rule, batch-32-vs-128 pilot added as arm P |
| M4 CI ignores training uncertainty | **adopted** — arm 1b seed replica supplies the floor; verdicts scoped artifact-specific (§4.2) |
| M5 arm-6 pool narrow and repeated | **adopted** — single-pass over 111,906 unique documents; scope stated; FineWeb reopens in M10, not M9 |
| M6 LoTTE "confirmation" | **adopted** — renamed a non-inferiority veto with an uncertainty term and hash-pinned checkpoints |
| M7 long-query coverage misstated | **adopted** — claim withdrawn (§9.1); head+tail probe will not run; `heldout-longq` may not change a decision |
| M8 throughput fallback manipulable | **adopted** — §6 pinned manifest, warmup exclusion and formula |
| MIN 70 MB unit undefined | **adopted** — decimal MB of total shipped bytes, measured |
| MIN B=10,000 too few below the 1.25% quantile | **adopted** — B = 20,000 |
| MIN `"passage: "` disjointness | **adopted** — asserted on the pinned tokenized inputs |
| MIN eligible-row arithmetic assumed | **adopted** — union measured: 6,149,679 (276 banned rows sit inside the FEVER span) |

---

## §11 Codex lock review #2 — disposition

`research/m9-codex-lock2-2026-08-30.md`. Verdict **DO NOT COMMIT. DO NOT SPEND THE 6 GPU-hours** on
the v2 lock; its core finding was that the v1 amendment had moved several failures out of the prose
and into the code. Its closing recommendation — *run one corrected, fully guarded anchor curve* —
became §3's staging. All 6 BLOCKER / 11 MAJOR / 3 MINOR actioned.

| finding | disposition |
|---|---|
| B1 amendment unratified; STOP advisory only | **adopted in code** — `screen.require_predecessors` refuses every stage-B arm until the adequacy gate passes, and `decide()` records the STOP note. Ratification is a **standing item for Dylan** (§0, `m9/STATUS.md`); nothing that depends on it can run meanwhile |
| B2 equal-token dosing false (arms 5 and 6) | **adopted** — arm 5 is now token-matched too; the batcher tracks a cumulative budget, asserts 30,349 non-empty steps and asserts no stream runs dry; both schedules materialized at lock time |
| B3 run token does not freeze the experiment | **adopted** — one **session manifest** frozen before the first arm; per-run tokens are one-use and consumed atomically; `eligible()` recomputes instead of trusting a boolean; diagnostics confined to `work/m9smoke/` where the decision loader cannot address them |
| B4 fingerprint omits inputs; stale code hashes | **adopted** — the fingerprint now covers `m7src/train.py`, `m7src/mix.py`, the banned-row mask and `results/m9_lock_constants.json`; starting files are compared against the hashes the registry pins; constants regenerated after the final code change |
| B5 fp16 gate absent | **adopted** — `m9src/fp16_gate.py` implements it and `screen.query_targets()` refuses fp16 rows until it has passed |
| B6 port pilot crashes | **adopted** — registry-key drift fixed, run token opened, `--no-strict` removed, size rule and fastembed registration made pass conjuncts |
| M7 batch pilot uninformative | **adopted** — the pilot and both fallback doses were **removed**; batch 128 locked with the reason recorded |
| M8 `2F` not a noise bound | **adopted** — MDE is one registered number, 0.0056; the seed replica is reported, never read by a rule |
| M9 teacher screen may rank early imitability | **adopted** — the adequacy gate (§4.4) sits between the anchor and any challenger encode |
| M10 statistic not fully locked | **adopted** — quantile method registered, bootstrap draws chunked (a DEV-6 draw is 3.2 GB in one block), rank stability reads registered **step ids**, `seed_sensitivity` goes through `align()` |
| M11 longq subsetting conditionally identical | **adopted** — corpus identity, vector shape, qrels and qid containment all asserted |
| M12 arm order and fallbacks unenforced | **adopted** — `require_predecessors`; the fallbacks no longer exist to be unenforced |
| M13 document sample source-biased | **adopted** — draw order kept; the realized schedule now consumes **189,002** documents from a uniform draw |
| M14 Qwen template deviates from the mandate | **adopted as a recorded deviation** — each teacher's own repository template is used, because a teacher's targets are only meaningful under the prompt it was trained with. Listed in §0's amendment for Dylan; it binds only stage-B arms, which cannot run yet |
| M15 caches and auxiliary outputs unguarded | **partially adopted** — the symmetric ceiling and parity artifacts are in the fingerprint's blast radius via the code hashes; per-chunk content hashing of challenger caches is a **stage-B prerequisite**, recorded here and not yet built |
| M16 port acceptance incomplete | **adopted** — total shipped bytes measured against the 70 MB decimal target, fastembed registration required, fp16 graph emitted |
| M17 six-hour estimate incomplete | **adopted** — there is no estimate left to defend: the anchor arm *is* the measurement, and there is no fallback for a number to steer |
| MIN 1 LR never reaches `lr_final` | **adopted** — denominator `steps − warmup − 1` |
| MIN 2 checkpoint off-by-one | **adopted** — 7,588 |
| MIN 3 validation sample hashes absent | **adopted** — both materialized into `results/m9_lock_constants.json` and mirrored into the registry |

---

## §12 Codex lock review #3 — disposition

`research/m9-codex-lock3-2026-08-30.md`. Verdict **v3 is broken; do not let `m9s1` open stage B**,
with the in-flight anchor to be finished and quarantined. It was instead **killed at ~11,000 of
30,349 steps** — because it was quarantined anyway, and because it had by then degraded to 786 ex/s
on allocator thrash (§9.15), so finishing it would have cost 1.7 hours for a diagnostic.

| finding | disposition |
|---|---|
| **B1 λ was DEV-tuned while the ledger said otherwise** | **adopted in full** — claim withdrawn (§9.12), `m9src/warmfit.py` selects λ on a training-only 50K/10K split under the real normalized objective, `warm_start_head` refuses without it, anchor re-run |
| MAJOR "identical treatment, no contrast moves" is false | **adopted** — §3.2a states the estimand: factor **plus refit**, per arm, end to end (§9.13) |
| MAJOR the warm start adds uncounted dose | **adopted** — Stage-0 dose (examples, non-pad tokens, teacher-target accesses, seconds) reported separately from the SGD dose by every arm |
| MAJOR `m9s1c` does not fully price the warm start | **adopted as wording** — its estimand is the **fixed-SGD-dose warm-start delta**, one seed, not compute-matched. Extra random-head seeds are a stage-B item, not claimed now |
| MAJOR both adequacy numbers arbitrary; slope ≠ convergence | **adopted** — §4.4 renames it a **budget trigger** and states that neither threshold is calibrated, that the LR has already decayed to 1e-5 where the slope is read, and that a negative slope also passes |
| **B adequacy authorization and ceiling unguarded** | **adopted** — the ceiling is a registry constant with its artifact hash checked at read time, and `require_predecessors` **recomputes** the verdict instead of trusting `"pass": true` |
| B1(A) STOP still advisory | **adopted** — `require_predecessors` refuses every arm once an eligible decision artifact names a non-incumbent teacher |
| B3 tokens overwritable; write/consume non-atomic; `eligible()` trusts copied metadata | **adopted** — `begin_run` refuses a consumed id, the result is written to a temp file and published only after the token is consumed, and `eligible()` additionally requires a consumed token this session issued |
| B4 stale code hashes in the constants | **adopted** — constants regenerated as the last step before the commit |
| M10 "chunking" still retained every block | **adopted** — blocks are regenerated from the seed on demand and discarded, so two contrasts still share draws but nothing holds 3.2 GB |
| M11 longq subsetting under-asserted | **adopted** — full ordered doc-id hash and a vector-content check, not first/last |
| M15 caches trusted by shape | **adopted** — challenger cache chunks carry content hashes in `chunks.json` and a stitch refuses unless every part is accounted for; a failed ceiling now **raises** instead of leaving the arm looking complete |
| M16 parity tested the graph that is not shipped | **adopted** — the fp16 graph carries its own parity row and is a pass conjunct; the parity sample's hash is verified against the lock |
| MIN 1/3 stale registry LR string, unverified sample hash, duplicate top-up | **adopted** — all three |
| B2 arm 6 realizes five tokens over baseline | **accepted and recorded, not fixed** — 0.08 parts per million on 59.5M tokens, from stopping at the first example that crosses each cumulative target. Exact equality would require splitting an example |
| M14 Qwen template deviates from the mandate | **unchanged, and still unratified** — it binds only stage-B arms, which cannot run until Dylan rules on §0 |
| M17 cost model incomplete | **partially adopted** — Stage-0 dose is now reported, so the anchor's own timing no longer under-reports itself; a full execution-graph cost model is a stage-B item |

---

## §13 Codex lock review #4 — disposition, and the over-engineering audit

`research/m9-codex-lock4-2026-08-30.md`. It reviewed the six changes made after pass 3, none of
which anyone had checked, and answered the question it was asked: **what is not earning its cost.**

> *"Bluntly: the discipline is now eating the experiment."*

**Not earning its cost, and now removed or reduced:** the all-or-nothing session fingerprint;
re-running deterministic incumbent arms after challenger-only edits; SCREEN-3 "rank stability"
sold as a test of the DEV-6 decision surface; provisional `decide()` artifacts masquerading as
final ones; duplicated code hashes in two places; a ledger long enough to contradict itself
(§3 said SCREEN-3 adequacy while §4.4 still said DEV-6 — reconciled).

**Earning its cost, and kept:** the training-only warmfit; the warm-start ablation (+0.0265 for
0.5% of wall-clock); anchor-first staging; full-row finiteness and unit-norm checks; bf16
challenger encoding; materialized token schedules with exhaustion and non-empty-step assertions;
pinned qid manifests and fixed contrast orientation; refusing to re-threshold the fp16 parity
result; atomic result publication.

| finding | disposition |
|---|---|
| **B1 the stage-B reorder is a protocol breach, and "void later" is not equivalent to teacher-first** | **withdrawn, order restored to the mandate's.** The reviewer is right that teacher-first *prevents* invalid downstream experiments while void-later runs them and promises to discard them. The reorder was justified by a time budget that no longer exists (Dylan: 15 hours available, then a 3-day run) |
| **B2 bf16 does not invalidate the fp16 cache; combined not rebuilt after a repair** | **adopted** — compute dtype, config overrides, transformers version and the encoder's own code hash are in the cache key; `combined.f16` is deleted whenever any chunk is written |
| **B3 provisional decisions are eligible and can become the de facto decision** | **adopted** — interim output is `results/m9_screen_state.json`, stamped ineligible; the final artifact is written only when every mandatory arm exists, carries `complete: true` and the sha256 of each arm it read, and deletes the provisional it supersedes. The STOP check reads both |
| M dependency-scoped fingerprints | **adopted** — eight scopes (protocol, data, train, eval, challenger, port, fp16, bridge); a run declares what it depends on. A challenger repair no longer voids the incumbent anchor, which is what three 1.5-hour re-runs bought nothing for |
| M `eligible()` proves a token was consumed, not that the payload is the one it wrote | **adopted** — the token stores the body's sha256 and `eligible()` recomputes it |
| M the allocator setting is not pinned | **adopted** — assigned, and a conflicting value refused, exactly as `M7_ENCODER` already was |
| M rank stability is a proxy, not a stability test of the decision surface | **adopted as naming** — the registry calls it a **proxy veto** |
| M the port pilot is not a gate | **adopted** — it exits non-zero on failure, records the fp16 graph's max-abs as well as its cosine, and registers the **fp16** graph as fastembed's `model_file` (naming the 135.6 MB fp32 graph demonstrated a serving route for something we do not intend to ship) |
| warmfit's split is index-contiguous, not random | **adopted** — a seeded permutation; fit p95 was 26 words against validation's 16. Selection now reads full-precision objectives, and the artifact says λ is **anchor-calibrated and globally reused**, not per-arm calibrated |
| the K=2 seed replica is read by no rule | **kept as reporting.** It is already run and it costs nothing to report; §4.2 already says no rule reads it |

**Added in the same pass, on Dylan's authorisation:** a **capacity probe** (`m9cap-diag`) — the
identical recipe and dose on a **109M** student. The anchor sits at 73.2% of the ceiling and the
aim needs ~90%; the curve says more epochs on 242,786 queries asymptote near 74%. The probe
separates "the data volume is the wall" from "33M parameters cannot represent this teacher's query
space", before three days of compute are committed to one of those answers. It is diagnostic and
out of M9's scope for selection — the mandate caps nano at 35M.

---

## §17 Codex review #7 — final pre-launch audit

`research/m9-codex-prelaunch-2026-08-30.md`. Verdict: **M9 is not closed and the build is not safe
to launch** — nine launch blockers plus a list of file inconsistencies.

The first blocker is the one to remember: `longrun.py` would have raised `NameError: tput` at
**step 500**, a dangling reference left when rolling throughput replaced the cumulative mean — and
`test_resume.py` hid it by setting `log_every = 10**9`. **A test that disables the code path it is
meant to cover is not a test**, which is `m8/CODEMAP.md` pitfall 17 in a new disguise.

Fixed in the same session: the `tput` crash; a first-eval gate that could never fire because step 0
is itself in history; `make_config.py` silently inventing a default config and ignoring the screen's
mix and prompt verdicts; and `test_resume.py` writing the **real** terminal marker, which would have
made the watchdog refuse to launch.

**Still blocking, and recorded in `m9/STATUS.md`'s handoff:** prompt policy (a) is not implemented
in `prepare`; a registered stop never runs the cooldown the lock promises; `open_session()` compares
the whole fingerprint so build artifacts generated after it can reject the build's own run token;
`write_status()` can crash the watchdog on a heartbeat that lacks training fields; and the
checkpoint-stale and eval-overdue checks only log.

Also actioned: RESULTS still carried the withdrawn "training residual" λ wording and no `m9s2`
diagnostic row; EXPLORED and the §3 arm table still presented the teacher screen as live; STATUS
said stage B was running, cited the DEV-6 ceiling for a gate that reads SCREEN-3, and described
`nqopen`/`triviaqa` as excluded when the build admits them.

**On guard scoping, the reviewer's ruling is adopted:** `guard9.py` stays in the `protocol` scope —
changing the code that decides eligibility during an arm *must* invalidate that arm. The repair is
to scope `open_session()` the way `eligible()` already is, or to give M9.3 its own session; the
"batch edits between arms" discipline is operational hygiene, not a fix.

## §18 Pre-launch hardening — five further adversarial passes, 2026-08-30 evening

All five ran before the M9.1 re-screen opened its session; every finding actioned same-day.
Briefs carried the read-exclusion; every review log audited clean of reserved-set reads.

| pass | scope | verdict → disposition |
|---|---|---|
| Codex #8 (`scratchpad`, actioned in `0c912e8`) | the five #7 blocker fixes | DO NOT LAUNCH: 5 BLOCKER / 4 MAJOR — mix verdict ignored by make_config, first-eval gate impossible (step-0 row), stale-corpus blessing, decisions outliving voided arms, cooldown not durable. All fixed |
| Codex #9 (verify, `eb6d1da`) | the #8 fixes | 3 BLOCKER residue — identity must bind BYTES not declarations; m9-decisions needs the challenger scope; manual decay not durable. Fixed |
| Fable fresh-context (`0eadaf9`) | whole launch path | DO NOT LAUNCH: 1 BLOCKER / 6 MAJOR — eval pause inside the rolling throughput window turns one slow eval into a registered permanent stop; policy-(a) prefix never reached eval (would confound m9s5); stale-heartbeat 80s restart race; no two-watchdog guard; untimed git push; cleanup gaps. All fixed |
| Codex #10 (`2506343`) | the Fable fixes | window reset sat BEFORE checkpoint I/O; initial launch unguarded; stale-lock takeover racy. Fixed |
| Codex #11 (`cf7632e`) | the #10 fixes | takeover still racy (path vs inode) → replaced with flock (kernel-released, no staleness state); smoked live |

**Registered before m9s6 ran** (M92_LOCK §4): mix verdict `70/30` → the 5/5/90 build; `query-only`
→ make_config refuses, shares become an owner decision. **Registered schedule change** (§5/§6):
plateau and the stable token cap enter the cooldown automatically; under the watchdog the trainer
also anneals before the wall-clock horizon (`--anneal-before-deadline`, 1.25× margin at measured
rate); every decay entry checkpoints immediately.

Also: `m9-status` is a status-only orphan branch (the watchdog worktree no longer checks out the
repo tree — closes the 2026-08-29 read-exclusion surface); corpora meta verified by 200-sample
re-tokenization per corpus and identity-stamped (student, source/kept/pool-rows hashes); Dylan
RATIFIED the §0 amendment (recorded at §0).

**§18.1 The screen, re-run and final (2026-08-30 21:37).** Teacher stella-400M · student
**bge-small** (MiniLM −0.0026) · prompt **(b)** (policy (a) −0.0204) · mix verdict **query-only**
(−0.0060) **overridden by owner ruling to 5/5/90** — grounds, decomposition and caveat in
`m9/M92_LOCK.md` §4 and `registry.json owner_rulings`. `m9s1b`/`m9s1c` withdrawn before the re-run
(read by no rule; measured pre-chunking). **The chunked-backward fix is the day's real catch**: a
padded-batch OOM that would have killed the seven-day build at step 0, found by `m9s6` in the
screen's twin of the same loop.

### Owner ruling 2026-08-30 — seed replicas waived; 168 h build authorized

| item | ruling |
|---|---|
| two extra preregistered seeds (mandate §Recipe/Seeds) | **waived** — ~14 GPU-days for a reporting-only figure; `registry.json owner_rulings.seed_replicas_waived`. Report states seed variability unmeasured, as a limitation |
| M9.3 build | **GO at the locked 168 h horizon**, recipe unchanged (`M92_LOCK.md`) |

### Codex unattended-operation review, 2026-08-31 (mid-build, log `research/m9-codex-unattended-2026-08-31.log`)

Verdict "not safe to call failure-proof": 2 BLOCKER, 5 MAJOR, 3 MINOR. Reserved-read audit of the
log: clean (only the brief's own exclusion text and two naming lines of `M92_LOCK.md`).

| # | finding | disposition |
|---|---|---|
| B1 | `TimeoutExpired` escaping `gpu_ok()` reached the loop's outer `except`, so a wedged driver disabled restart/terminal/deadline supervision for the rest of the run | **FIXED** — probe made advisory and non-raising |
| B2 | nothing supervised the watchdog; its death was silent | **FIXED** — `m9src/guardian.sh`, verified live against SIGKILL. Safe by flock: a relaunch while one lives exits immediately |
| M6 | corrupt `deadline.json` read as "missing" → a fresh 168 h horizon on restart | **FIXED** — fails closed; file+dir fsync |
| m9 | 180 s timeout killed the bash wrapper, not a hung `git push` | **FIXED** — `timeout -k 10 120` binds git itself; `start_new_session` |
| M3 | deadline stop stays armed during decay → a late slowdown can truncate the cooldown | **OPEN, trainer-side.** Only bites below ~18,940 tok/s sustained (live: 25,970). Scheduled with M4/M5 |
| M4 | corrupt `last.pt` has no fallback to the newest readable `stepN.pt` | **OPEN, trainer-side** |
| M5 | eval exceptions uncontained; a transient one kills the trainer | **OPEN, trainer-side** |
| M7/m8/m10 | `SESSION.json` init outside the trainer lock; eval crash window; argv-heuristic process match | **accepted, documented.** m10 bit the operator during this very repair — a `pgrep` pattern matched the operator's own shell, which was killed. Trainer unaffected; kill by exact PID from `ps -eo pid,args` |

Trainer PID 225232 was never restarted by this repair; `deadline.json` reused, not reset.

### Codex review of the M3/M4/M5 trainer patch, 2026-08-31 — **REJECTED, not applied**

Log `research/m9-codex-patch-review-2026-08-31.log`; reserved-read audit clean. Verdict: *"Do not
restart the live trainer onto this code."* The patch was reverted from disk the same minute; the
live trainer (pid 225232) never executed it. Rejected version kept for redesign.

| blocker | why v1 was wrong | what the redesign must do |
|---|---|---|
| M3 | trainer grace of 6 h is fiction: `watchdog.py:484` gives up 1,800 s after the deadline and refuses to restart a dead trainer past it, so the effective grace is ~30 min and a cooldown crash still ends the run | the fix must span BOTH files; watchdog must honour decay |
| M4 | `int(q.stem[4:])` runs before the `try` → any stray `step*.pt` crashes startup; "readable" ≠ schema-valid; and `reconcile_history()` runs BEFORE fallback, so a rewind leaves future eval rows, stale plateau/regression history, and a `best` disagreeing with history | no silent rewind — fall back only when no history/checkpoint frontier is later than the candidate, else write terminal state and stop |
| M5 | counter is process-global, so the watchdog's 5 h eval-stale restart resets it before three failures (3 intervals = 5.4 h nominal, 10.8 h at floor) — the bound never binds. Worse, a caught exception can resume training after eval perturbed CPU/CUDA RNG, **changing later gradients** — a locked-mathematics violation | bounded immediate retries at the same boundary, RNG/state restored after each failure, counter checkpointed, watchdog coordinated |

Also fixed from the same review: sentinel's `[ -f ]`-then-`stat` race against the watchdog's
heartbeat unlink (empty arithmetic → bash exit, i.e. the alarm dying silently); unanchored `pgrep`
in sentinel and guardian (the sentinel reported "2 trainer processes" within a minute of arming —
it was counting the bash launch wrapper); no multi-guardian detection; no proof-of-life.

**Standing decision for the redesign:** these three are low-probability risk-reduction on a healthy
run (M3 binds only below ~18,940 tok/s; live rate 25,971). Whether to touch a verified running
build at all is itself a question for the next review, not an assumption.

### M9.4 final-run lock written mid-build as a DISCLOSED AMENDMENT, 2026-08-31

**Gap found:** M9.2 locked the build recipe but none of the final-run fields the mandate requires
(no system manifest, claim table, statistics block, or reserved manifest; `registry.json` had no
final-run key). Written now because protocol may only be written ahead of the numbers it affects,
and **no six-set, reserved, LoTTE or confirmatory output exists in M9**. Author's observations at
time of writing, disclosed for influence assessment: SCREEN-3 build evals at steps 0/15,000/30,000
and throughput telemetry — nothing else.

**This is NOT the M9.2 preregistration and must never be described as one. Dylan's ratification is
a standing open item** (`registry.final_run.ratified_by_owner: false`).

Reviewed: `research/m9-codex-finallock-2026-08-31.log` — 7 BLOCKER / 6 MAJOR / 1 MINOR, all
actioned in `m9/FINAL_LOCK.md` + `registry.final_run`. Reserved-read audit clean.

| finding | fix |
|---|---|
| **B1 (the M7-equivalent bug)** `boot.py` exposes NO 0.0125 quantile and rounds `one_sided_lower_2.5` to 4dp; a caller using `ci95_raw[0]` (2.5%) would pass contrasts the mandate fails | decision field is `lower_q0125_raw = np.quantile(draws, 0.0125, "linear")`, full precision, **only that field decides**; rounded fields explicitly not the gate. Requires `m9src/final_stats.py` — the "no new statistics code" claim was **withdrawn as false** |
| B2 not the M9.2 preregistration | classified as a disclosed amendment with the observation set stated; owner ratification pending |
| B3/B4 crash rules self-contradictory; a state could be both retry-eligible and spent | durable state machine: the `m9-six-spent` tag is **pushed before the first protected read**; `--infra-retry` admissible iff that tag is absent from origin; `--recover` recomputes decisions without re-reading the six; preflight may not open six-set queries/qrels |
| B5 constants not machine-registered (executor could read the M9.0 screen defaults B=20,000/seed 0) | mirrored into `registry.final_run`; prose/registry disagreement is a hard abort |
| B6 C1 computable against either the fresh bridge row or the frozen row (differing by up to 3e-4) | bridge row is **validation-only and discarded**; C1/C2 use only frozen rows from `perquery.json` pinned at sha256 `6b18e3dd…` |
| B7 reserved section was placeholders ("pinned hashes", "restart semantics") | complete manifest: systems INCLUDED/OMITTED, estimands R1/R2 + directions, B/seed/interval, renormalized leave-one-out, 120 GB gate, per-system atomic crash semantics |
| M1 `C1 fail / C2 pass` cell wrong — it suppressed a permitted aim claim | corrected: no release, **aim claim permitted**; C2 does not gate the ship and C1 does not gate the claim |
| M2/M3 disclosures unbound; headline referenced not quoted; "unrestricted" not banned | headline reproduced verbatim, paraphrase forbidden; each disclosure binds a value; all three forbidden words listed |
| M4/M5 `paired_dep` defaults (`strict=False`, `k=len(aligned)`) allow a silent 5-dataset macro; shared draws not guaranteed by the API | assert exactly six datasets + identical frozen qids, then `strict=True`; one frozen draw plan, digest serialized, reused by both contrasts |
| M6/m1 manifest not revision-complete; `FINAL-BEGIN` misnamed | hashes/revisions pinned per row; ledger marker is `FINAL-RUN-BEGIN` |

### Unattended memory-safety finding, 2026-08-31 — the trainer is the kernel's top OOM victim

Found while diagnosing two Codex reviews dying mid-run. The box has 25 GB; the trainer's RSS is
19.8 GB, but **`RssAnon` is only 2.1 GB** — the other 18.5 GB is memory-mapped corpus/target
pages, i.e. reclaimable page cache (`VmSwap` 0). Real memory pressure is therefore low.

**The risk is the ordering, not the volume.** `oom_badness` counts file-backed pages, so the
trainer's `oom_score` is **1073** — the highest on the box. Any *other* process triggering an OOM
would get the seven-day trainer killed first, though its own pages are reclaimable and it is the
innocent party. Lowering its score needs `CAP_SYS_RESOURCE` (no root here).

**Mitigation, permission-free:** `m9src/sacrificial.sh` sets `oom_score_adj=1000` on itself and
execs the helper, because *raising* a process's own score requires no privilege. **Every helper
tool run alongside the build — Codex reviews especially — must go through it**, so the helper
loses any OOM contest with the trainer. Verified: wrapper reports `oom_score_adj=1000`.

### Codex code review of `final_stats.py`, 2026-08-31 — "not fit to decide release as written", now fixed

Log `research/m9-codex-finalstats-2026-08-31.log`. Verdict reversed by the fixes below; 16/16 tests.

| finding | disposition |
|---|---|
| **CRITICAL: `method="linear"` is not the empirical quantile.** At B=10,000 the empirical 0.0125 quantile is the **125th order statistic** (`inverted_cdf`); NumPy's `linear` interpolates toward the 126th and returns a weakly **higher, more permissive** bound — able to flip the irreversible gate in the candidate's favour | **FIXED** — gate is `method="inverted_cdf"`; registry records the correction and its reason. Test asserts `inverted_cdf <= linear`, i.e. never more permissive |
| caller-supplied `conf` could change B, seeds, quantile, alpha | **FIXED** — `_assert_matches_registry` refuses any deviation; tests must pass `allow_unregistered=True` explicitly |
| contrast identities/order unenforced; a reversed pair silently tests the wrong direction | **FIXED** — contrasts asserted against `registry.contrasts` |
| plan replicate counts not checked across datasets | **FIXED** — inconsistent B refused |
| **the lock OVERCLAIMED**: "one frozen sign plan shared by C1 and C2" is only a same-seed guarantee — no sign plan or digest is materialized, unlike the bootstrap draw plan | **CORRECTED in lock, registry and code comment.** Sharing is a comparability device, not a condition of validity |
| gate field, 1/6 weighting, resampling unit, `alternative="greater"` tail | confirmed correct |

**Answer recorded for the report (asked because I could not settle it myself):** sharing bootstrap
indices across C1 and C2 **does not affect either interval's marginal validity**. It correlates
their Monte-Carlo error only — common random numbers for comparability. Do not describe it as
anything stronger.

**Operational note:** three background Codex runs were killed mid-exploration without emitting
findings; the same brief, narrowed and run in the FOREGROUND with the module inlined, completed in
under ten minutes. Prefer foreground + inlined code + explicit "do not explore" for review briefs.

### M9.3 BUILD PROVENANCE DISCLOSURE — build session force-reopened 2026-08-31 06:20 UTC

**This build is NOT an uninterrupted single-lock run and must never be described as one.** Its
provenance spans two build lock states, though every scientific input is unchanged.

| item | value |
|---|---|
| old `SESSION-build.json` fingerprint | `24dcbe852bde` |
| new tree fingerprint | `2b1e2ae214bc` |
| command | `open_session(force=True, run_id="m9-build")` |
| resume boundary | `work/m9long/ckpt/last.pt` at step ~78,754 (~0.645B tokens), written pre-amendment |
| invalidated | prior guard9 runs/tokens under `SESSION-build.json` **only** |
| NOT touched | `work/m9tokens/SESSION.json` was neither forced nor reopened; screen arms `m9s1`, `m9s4`, `m9s5`, `m9s6`, `m9-decisions`, `m9-adequacy` remain under their original lock and stay valid |

**Changed, classified:** *protocol bookkeeping* — `m9/LEDGER.md` (review logs), `m9/registry.json`
(ADDED `final_run`, bumped `stage`). *Approved infrastructure repair* — `m9src/longrun.py`,
`m9src/watchdog.py` (M3 anneal protection + M4 fail-fast diagnostic).

**Unchanged:** recipe, dose, sampling specification, `work/m9long/config.json`,
`work/m9long/manifest.json`, `m9/M92_LOCK.md`, `m9src/make_config.py`, and all data/train/eval
scope files. The checkpoint's `config_hash` and `manifest_hash` are unchanged and its resume guard
verifies both. **No dose, sampling or recipe change was introduced by this intervention.**

**Reason:** the trainer restart was blocked because mandatory protocol logging and the approved
build repair moved scoped fingerprints. **Review:** five adversarial passes on the M3/M4 diff
(`research/codex_m3{,b,c,d,e}.log`), approved on the fifth; amendment pushed before the force.

### Consequence, and a standing rule for the rest of the build

`m9/LEDGER.md` and `m9/registry.json` are in guard9's **protocol** scope, and `DEPS["m9-build"]`
includes it — so **any** edit to them makes the trainer refuse to start. The watchdog's automatic
crash-recovery calls `begin_run` too, so protocol logging during the build silently disables
unattended restart: a crash days later would not recover.

**Rule from now until the build ends: no writes to `m9/LEDGER.md`, `m9/registry.json` or any other
protocol-scope file.** Build-period notes go to `m9/BUILD_LOG.md`, which is in no scope, and are
merged into the ledger after the build completes.
