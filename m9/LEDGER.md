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

Adversarial review of the first draft: `research/m9-codex-lock-2026-08-30.md` (gpt-5.6-sol,
verdict DO NOT COMMIT — 7 BLOCKER / 8 MAJOR / 4 MINOR + a post-number-freedom table + an
arithmetic audit). **All findings actioned**; the disposition table is §10. Read-exclusion
honoured, log audited: zero reserved-set reads.

### Owner approvals carried in (Dylan, 2026-08-30 planning session)

| ruling | status |
|---|---|
| FineWeb approved as nano's regression-text seed corpus | **NOT EXERCISED — excluded by the mandate's own precondition** (§1.3) |
| M9 execution on branch `m9-work`, frequent commit+push; merges to main need Dylan's go | in force; the guard requires HEAD on `origin/m9-work` |
| Query asset: quality first, 70 MB target, exceedable with logged measured justification | in force. **Unit defined**: decimal MB of *total shipped artifact bytes* (ONNX graph + tokenizer + config), measured by the port pilot — not the theoretical weight product. Weights alone: bge-small nano 67.508 MB, MiniLM nano 46.215 MB |

### M9.0 amendment to the mandate — the teacher-screen surface (needs Dylan's ratification)

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

Arm-6 **candidate list**: the first **400,000** eligible rows drawn by
`numpy.random.default_rng(9).choice(eligible, replace=False)`, sorted — rows sha256
`89fb550ec82fcfb9577aa55038f10889a88296723b8b0449c46af705bdc17ae6`. Arm 6 consumes this list
**in order** until its token target is met: **111,906 documents, each seen exactly once**
(§3.1). Doc targets under stella are the existing pool vectors, so they cost zero teacher compute.
WordPiece length of a `"passage: "`-prefixed candidate: mean 94.57, p50 64, p95 306, max 512.

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

| # | id | varies | teacher | student | prompt | mix | seed | decided on |
|---|---|---|---|---|---|---|---|---|
| P | `m9p-bs32` / `m9p-bs128` | batch size | stella-400M | bge-small | (b) | query-only | 0 | DEV-6 |
| 1 | `m9s1` | anchor | stella-400M | bge-small | (b) | query-only | 0 | both |
| 1b | `m9s1b` | **training seed** | stella-400M | bge-small | (b) | query-only | **1** | DEV-6 |
| 2 | `m9s2` | teacher | stella-1.5B | bge-small | (b) | query-only | 0 | SCREEN-3 |
| 3 | `m9s3` | teacher | Qwen3-0.6B | bge-small | (b) | query-only | 0 | SCREEN-3 |
| 4 | `m9s4` | student | selected | MiniLM-L6 | (b) | query-only | 0 | DEV-6 |
| 5 | `m9s5` | prompt | selected | selected | (a) | query-only | 0 | DEV-6 |
| 6 | `m9s6` | mix | selected | selected | selected | 70/30 **by token** | 0 | DEV-6 |

Order is fixed and may not change after `m9p-bs128` starts. Arm 1b exists to **measure** the
training-noise floor the decision threshold reads, instead of importing M8's table-specific 0.004
(§4.2). If a teacher challenger fires, the run stops (§0 amendment).

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
| checkpoints | 7,587 · 15,175 · 22,762 · **30,349** (decision reads the last; the last two also feed the rank-stability rule) |
| checkpoint **surfaces** | SCREEN-3 at all four (0.65 GB of document vectors, free, and it carries the dose-response curve); **DEV-6 at checkpoints 3 and 4 only** — a DEV-6 pass streams 23.3 GB at this box's measured 174 MB/s, so four passes per arm would cost more wall-clock than the training they measure. Fixed before any arm ran; every rule stays executable because the decision reads checkpoint 4 and rank stability reads 3 and 4 |
| seed | 0 (data order, init, dropout); arm 1b seed 1 |
| optimizer | AdamW, β (0.9, 0.999), eps 1e-8, weight decay 0.01 on tensors with dim > 1 and 0.0 otherwise, grad-clip 1.0 global norm, no gradient accumulation |
| LR | `warmup: 1e-4·(step+1)/910`; then `1e-5 + 0.5·(1e-4−1e-5)·(1+cos(π·(step−910)/(30349−910)))` |
| precision | bf16 autocast backbone; **loss in fp32**; retrieval matmul fp32 from fp16 document memmaps |
| max sequence | 512, dynamic padding to the longest member of the batch |
| epoch order | one `default_rng(seed)`, a fresh `permutation(n)` per epoch, concatenated |

**Arm 6 is a fixed-compute contrast** (Codex BLOCKER-2): identical **30,349** optimizer updates and
identical **59,507,872** non-pad tokens, split **70/30 by token**, not by example. Per step it
fills 70% × 1,960.8 query tokens from the locked query schedule and 30% from the locked document
candidate list, both consumed in order; batch size therefore floats (≈90 queries + ≈4 documents).
Targets: **41,655,510** query tokens (11.20 epochs of the pool) and **17,852,415** document tokens
(**111,906 documents, single pass, no repeats**). Loss is the plain mean over all examples in the
batch — **no role weighting**. The realized per-step token distribution is reported for every arm.

The baseline arms keep fixed 128-example batches (their texts are one length family, so per-step
tokens are near-constant); this asymmetry with arm 6's token-budgeted batches is deliberate,
recorded, and reported.

**If the P pilot selects batch 32**, every arm falls back to the registered 8-epoch dose:
1,942,288 examples, **60,697** steps, checkpoints 15,174 / 30,348 / 45,522 / 60,697, warmup 1,821,
T_base 29,753,936. No other dose is permitted.

### 3.2 Objective (phase 1)

`loss = mean_over_batch( sum_over_dim( (normalize(student_out, eps=1e-12) − target)² ) )`, target =
the L2-normalized teacher vector, computed in fp32. No auxiliary term (LEAF's ablation rejected
them; MSE+cosine is affine-redundant under normalized outputs). Phase 2 is out of scope for M9.1.

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

**MDE = max(0.0051, 2 × F)** where **F = |macro(m9s1) − macro(m9s1b)|** on DEV-6, the measured
training-noise floor at the screen dose. 0.0051 is 1.96 × the p90 DEV-6 macro SE. `F` is
registered as *measured at M9.1 by arm 1b before any student/prompt/mix contrast is read*, and
`m9src/screen_stats` refuses to decide those three until the arm-1b artifact exists. **Stated
limitation:** F is a range over K = 2 seeds and is therefore a very noisy estimate of σ
(`m8/CODEMAP.md` pitfall 18); it is used only to *raise* the threshold, never to lower it below
0.0051.

**Rank stability.** A student/prompt/mix decision that adopts a challenger must also have the same
sign at checkpoints 3 and 4. The screen runs at ~1% of LEAF's example dose and ~0.5% of its
optimizer updates, so an early-training ranking is not automatically the final-dose ranking; the
checkpoint curve is the registered instrument for saying whether it is. A sign flip between the
last two checkpoints ⇒ **not stable at screen dose** ⇒ the default is taken and the arm is
recorded as diagnostic.

**Scope of every screen verdict.** These are *artifact-specific* selections at the screen dose
under seeds 0 and 1 — not claims that a recipe is superior. The bootstrap quantifies query
resampling conditional on one fitted artifact; F is the only training-uncertainty term in the
rule, and it is coarse. No screen result may be reported as resolved, confirmed, or significant.

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
| **throughput pilot** | pinned manifest: **20,000** query texts (seed 13 from the locked pool) + **5,000** documents (first 5,000 of the locked candidate list), at the registered batch settings, first 2 batches discarded as warmup. GPU-hours = Σ over the 9 runs of (examples ÷ measured ex/s) + measured challenger-teacher encode seconds |

**Fallback trigger:** if the pilot's GPU-hour total exceeds **12**, every arm falls back to the
registered 8-epoch dose (§3.1). The manifest, the warmup exclusion and the formula are fixed above
precisely so the trigger cannot be steered.

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
