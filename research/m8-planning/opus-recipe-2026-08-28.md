# M8 planning — Opus brainstormer: training recipe and objective (2026-08-28)

*Verbatim final report of an Opus 5 subagent briefed on m7/RECIPE.md, FINDINGS.md, EXPLORED.md,
LEDGER.md, instructions-m8.md, m7src/train.py, pseudoq.py, program.py and the cited result
artifacts. Numbers are the agent's own reads of repo artifacts; the orchestrator spot-verified the
negatives matched-step claim (LEDGER.md:335) and the pseudo-pool composition (pseudoq.py:84-86).*

---

# Part 1 — Three measured facts that reorder the whole lever list

**(A) Capacity is not the binding constraint; generalization is.** G2's overfit probe reaches macro
**0.99999** with train=eval on dev queries (`m7_gate_p35w-2m-s2500.json`). A 30,522×1024 table has
31.3M free parameters against 340,850 labelled pairs. So every "add capacity" lever (bigram rows,
richer pooling) is attacking the wrong constraint, and every "restrict / regularize / supply more
supervision" lever is attacking the right one. This inverts the priority order implied by
`LEDGER.md § Capacity levers`.

**(B) Phase B reproduces the closed-form ridge solution, and Phase A does all the work — in 83
seconds.** Under stella:

| artifact | proxy-3 |
|---|---|
| closed-form ridge, flat table, λ=0.01 (`m7_stage0_ridge_stella.json`) | 0.4973 |
| `s1-objB` — B, 8k steps, no pseudo mix | 0.4903 (*below* the ridge) |
| `p35b-2m` — B, 16k steps, 924,704-span pseudo mix | **0.4981** (ridge **+0.0008**) |
| `p35a-2m-1e3` / `p35w-2m-s2500` — after Phase A | 0.5109 / **0.5106** |

16,000 SGD steps of Phase B land **+0.0008** from a two-minute linear solve. The entire measured
improvement over closed form, **+0.0125 proxy**, comes from Phase A — which runs 2,500 steps at
33 ms/step = **83 seconds**, and plateaus there (`p35a-2m-1e3-x4000`: peak 0.5119@2500, then flat).
Phase A is not step-limited. It is **pair-limited**: 2,500 × 512 = 1.28M draws over 340,850 pairs
≈ 3.7 epochs, after which it memorizes. *The one part of the recipe that does anything runs for 83
seconds on 3.7 epochs of data.* That is where M8's money goes.

**(C) The KL term is not a KL — it carries no soft target.** `step_b` builds the candidate set as
the query's positive plus **31 uniformly random rows from the 2M bank**, then takes the teacher's
softmax at `temp=0.02`. From `m7_diag_scores.json`: positive mean cosine 0.6797, random-negative
mean 0.2854, p99 0.4344, and only 32.65 of 32,768 random negatives outscore the positive. So for
31 uniform draws:

- expected distractors beating the positive: 31 × 32.65/32768 = **0.031**
- teacher logit gap at the *mean*: (0.680 − 0.285)/0.02 = 19.7 → p(positive) = 1 − 8.6e-8
- at the *99th-percentile* distractor: (0.680 − 0.434)/0.02 = 12.3 → p(positive) = 1 − 1.4e-4

**The teacher's target distribution is one-hot to ~1e-4 nats.** `KL(teacher‖student)` with a
one-hot target is exactly cross-entropy — i.e. `kl_weight=1.0, kl_k=32` is a second, *weaker*
InfoNCE (31 negatives) bolted onto Phase B, not rank distillation. **No teacher ranking information
reaches the student anywhere in the recipe.** Phase B is therefore pure regression to the teacher's
query vector, which explains (B) exactly: regression's optimum is the ridge, and SGD finds it.

This also re-frames `FINDINGS.md` #3 ("cosine agreement with the teacher's query vector is not the
metric, and it mis-ranks"). The project *measured* that the query-vector target mis-ranks — and
then trained 16,000 steps, plus 50% of every Phase-B batch and all 220,632 query-text-only rows, on
precisely that target and nothing else.

**Corollary ("which objectives move the optimum vs merely reweight"):** the student is linear in
`W` up to the final normalize, so the *model class* is fixed and no objective changes what the
table can express (G2 already says the class can express everything). What an objective changes is
*which* `W` in a 31M-dim space is selected — and that split is clean:

- **Cannot move the optimum meaningfully:** anything that is a function only of `cos(q(W), t_q)` —
  its optimum is the ridge, measured at 0.4973/0.5181. Plus everything `m7_absorb_check.json`
  already proved absorbable (centering, whitening, top-PC removal, per-token scalars).
- **Genuinely different optimum:** anything defined on `q·d` over a document set — InfoNCE, a
  *real* KL over confusable candidates, margin-MSE on teacher score gaps, listwise/LambdaLoss.
  These weight table directions by ranking impact rather than by target variance, and discard
  directions of the teacher's query vector that are orthogonal to the discriminative doc subspace.
  Regression spends 31M parameters matching directions that cannot change a ranking.

So: **the objective is the whole game, the ranking half of the objective is currently a no-op, and
Phase A is starved of pairs.**

---

# Part 2 — The cost model that makes this cheap

Measured from the logs, not estimated:

| operation | cost |
|---|---|
| Phase A arm, 2,500 steps, from a frozen B checkpoint | **83 s** (33 ms/step) |
| Phase B chain, 16,000 steps | ~9 min compute, ~20–25 min with in-training proxy evals |
| **Full pinned dev suite, 18 variants at once** (`multieval`, incl. hotpotqa 5.23M docs + pool 6.17M) | **14 min**, 15.2 GB peak RSS |
| teacher encode, fp16 | ~1,000 texts/s → 1M texts ≈ 17 min |
| negative bank in VRAM | 2M × 1024 fp16 = 4.1 GB of the 3080's 10 GB |

The experimental loop is therefore: train N A-arms (N × 83 s), score all N in **one 14-minute
pass**. A 16-arm A-phase grid is a **~40-minute experiment**. This is the single most
under-exploited fact in the repo — `phase3_hparams` never ran, and three of its nine arms are
A-phase-only and cost 83 seconds each.

---

# Part 3 — Proposals

### P1. Give pseudo-queries their source document as a positive (self-supervised contrastive at scale) — **the bet**

**What.** `pseudoq.build()` draws spans from five TRAIN doc stores and returns *text only*,
discarding which document each came from. But `poolmod.PoolIndex` maps `(store, doc_id) → pool
row`, and `build()` already has both in hand — retaining them is a ~10-line change. That converts
924,704 unlabelled spans into 924,704 **(query, positive-document) pairs** with the teacher's
frozen doc vector already in the pool, at zero licensing cost and zero new encoding. Phase A's pair
supply goes 340,850 → ~1.27M (3.7×), and the pseudo half of Phase B stops being trained on the
criterion `FINDINGS.md` #3 says mis-ranks.

**Expected effect.** Large by this project's standards — forecast +0.01 to +0.03 dev macro, with a
*much* better out-of-domain profile than any lever tried so far, because these positives are drawn
from the whole 6.2M-doc pool rather than from six QA datasets. Two independent reasons: (i) Phase A
is at 3.7 epochs and plateaus; 3.7× the pairs moves the plateau, and it is the only phase that
beats closed form; (ii) this is ICT / Contriever-style independent cropping, whose entire published
purpose is exactly this regime — abundant unlabelled documents, scarce labelled pairs.

**Compute.** Nothing new to encode (span targets and pool doc vectors both cached). One Phase-A arm
83 s; a full B+A chain ~30 min. The pseudo→pool-row index is a dict build, minutes. Fits VRAM and
the 18 GB RAM budget unchanged.

**Risk.** ICT positives reward lexical overlap — the table could learn term matching, which BM25
already does for free, and BM25 fusion would then stop adding. Three mitigations, all cheap: (a)
`_span` currently takes only the **first sentence**, the highest-overlap span in the document —
switch to random sentence-aligned windows (`_long_span` already does this); (b) remove the span
from its own document before the doc is encoded, ICT's standard 90%-removal trick — this *does*
cost a re-encode, so make it the second arm, not the first; (c) keep the teacher-cosine term as an
anchor so the target stays semantic. Second risk: the pseudo positives are 43% Amazon product text
and 43% HotpotQA Wikipedia, so genre skew persists — P7 addresses it.

**Probe (<2h).** Retain provenance; rebuild the pseudo pool index; run **one** Phase-A arm from the
*existing frozen* `p35b-2m` B checkpoint with pseudo pairs mixed into `step_a` at fractions
{0, 0.25, 0.5, 0.75} — four arms × 83 s — and score all four plus the baseline in one 14-min
multieval. **Total ≈ 45 min.** Read the out-of-domain subset next to the macro.

**Touches.** Nothing closed. Lever #2 (pseudo-query coverage, ADOPTED) is the *dose* of these spans
in the cosine term; this is a different use of the same asset. A matched no-pseudo control already
exists (`p4x-nopseudo-*`).

### P2. Make the KL an actual KL: draw distractors from the teacher's top-N, not uniformly

**What.** Change the `dist` sampling in `step_b` from `rng.integers(0, nb, ...)` to a sample from
the teacher's top-100..1000 for that query. With confusable candidates the teacher's softmax at
`temp=0.02` becomes genuinely soft (positive ≈0.85, hard negatives ≈0.80 → 2.5-logit gap → p(top1)
≈ 0.28), so the ordering *among* candidates transfers — which is what distillation is for and what
fact (C) shows is currently absent.

**Expected effect.** This is the difference between "we distil the teacher's ranking" and "we do
not". Uncertain in size but structurally load-bearing; +0.005 to +0.02 with a plausible chance of
nothing if the teacher's fine ordering among hard candidates is unlearnable by a bag-of-tokens
student. Pair with a **temp split**: `temp_kl` (soft target, ~0.05–0.1) separate from
`temp_infonce` (0.02, which `m7_diag_scores.json` shows is the elbow — effective negatives 3.0 at
0.01, 28.9 at 0.02, 5,305 at 0.05). Sharing one temp between a distillation target and a
contrastive denominator is a conflation nobody has tested.

**Compute.** Mining the teacher top-N for 561K train queries against the 6.2M pool:
`mine_hard_negatives` already exists and is chunked; budget 1–3 h once, cached. Then Phase-B chains
at the usual ~30 min.

**Risk.** This overlaps the **CLOSED mined-hard-negatives avenue** — and the close does not bind
here: (1) the LEDGER's own text says the result is "NOT IDENTIFIED", not refuted; (2) the closure's
diagnosed mechanism (memorization) is a mechanism for **hard negatives in the InfoNCE denominator
on labelled pairs**, not for **soft targets in a distillation term**; (3) fact (C) makes this a
**specification defect, not a hyperparameter**: a term named `kl` with `kl_weight=1.0` that
provably carries ≤1e-4 nats of information is a bug. Also: the vacuous false-negative check
(`EXPLORED.md`) — a soft target is the natural fix, since an unlabelled positive gets high teacher
mass instead of being force-labelled negative. That is an argument *for* this change.

**Probe (<2h).** Free version, no training at all: take 2,000 cached train queries, compute the
teacher's candidate distribution entropy under (a) 31 uniform distractors and (b) 31
teacher-top-200 distractors, at temp ∈ {0.02, 0.05, 0.1}. **If (a)'s entropy is ~0, fact (C) is
confirmed as a defect in 10 minutes of numpy.** Then one Phase-B+A chain (~30 min) with mined
distractors at the corrected temp.

### P3. Train against the *fused* score, because the released system is the fusion

**What.** The released system is `convex0 w=0.8` fusion with BM25, worth **+0.057** on the six —
an order of magnitude more than any recipe lever this project has moved. Yet the table is trained
as a standalone retriever. Make the training objective the deployed objective: in `step_a`/`step_b`,
score candidates as `w·s_dense + (1−w)·s_bm25` with BM25 frozen, and apply InfoNCE/KL to the fused
score. The gradient then pushes the table toward the residual BM25 cannot cover instead of
re-deriving lexical matching.

**Expected effect.** Structurally correct and unexplored anywhere in the repo. Realistic +0.005 to
+0.02 on the *fused* system, which is the number that ships. It also attacks the clean-4 failure
directly: on the four sets with no teacher overlap the table sits **−0.0311 below BM25**, so the
dense side is contributing least exactly where the lexical side is carrying — the two are
redundant, not complementary.

**Compute.** Build a `bm25s` index over the training doc stores, score each (query, candidate) pair
offline once, cache alongside the mined candidates. Hours, not days. Training cost unchanged.

**Risk.** (i) Circularity with fusion selection: train at the incumbent `w=0.8`, re-select after,
pre-register the comparison as fused-vs-fused. (ii) It couples the released artifact to a specific
BM25 implementation — pin harder (the fifth review already caught a `bm25s`-upgrade hazard).
(iii) If Dylan ever wants the dense-only artifact, this optimises the wrong thing for it; report
both.

**Probe (<2h).** No training needed for the go/no-go: compute, on the four text-backed dev
components, the **per-query complementarity** of the current table and BM25 (rank correlation of
their runs, and the oracle-fusion ceiling at `DEPTH=1000`). If the oracle ceiling is far above the
achieved `convex0 w=0.8` macro, there is real residual to train toward. ~1 h with cached runs.

### P4. The never-run `phase3_hparams` sweep, redesigned as an A-phase grid — and it resolves the negatives/step confound in the same pass

**What.** `program.phase3_hparams` exists and has never run; `temp=0.02` and `n_neg=32768` have
been fixed since phase 1. The grid as written mixes B-phase and A-phase knobs. **Split it:** from
the frozen `p35b-2m` checkpoint each A-arm is **83 seconds**. Design a single 16–20 arm A-phase
grid crossing `n_neg ∈ {8k, 32k, 128k, 512k}` × `temp ∈ {0.01, 0.02, 0.05}` × `steps_a ∈ {1500,
2500}`, **plus the four negatives arms (`bank`/`teacher16`/`bm2516`/`mixed32`) at matched
`steps_a`** — exactly the matched-steps design `instructions-m8.md` names for the confound. Cost:
~25 min of training, one 14-min multieval. **Under an hour resolves a never-tested hyperparameter
axis and a formally-unidentified closed avenue.**

**Expected effect.** Individually small — probably each inside the 0.0027–0.0078
recipe-perturbation band, so *no single arm will be adjudicable*. Its value is (a) map-making;
(b) it identifies the negatives/steps confound at near-zero cost; (c) `n_neg` interacts strongly
with P1/P2 — with 4× the pairs and mined candidates the optimal `temp`/`n_neg` will not be
0.02/32,768, and inheriting them would confound the main bet.

**Risk.** **Multiplicity is the whole risk.** Pre-register it as a **screen, not an adjudication**:
it may only select a *region* to confirm with one arm under a proper bar, and its numbers may never
be quoted as effects.

### P5. Constrain the table instead of expanding it: low-rank / structured delta

**What.** Fact (A): 31.3M free parameters against 340K pairs, rows updated at Zipf-distributed
rates. Reparameterize the learned change as `W = W₀ + U Vᵀ`, r ≈ 64–256 — 2M–8M parameters instead
of 31M, sharing statistical strength across rows. Not on the absorbable list (it *restricts* the
class, not reparameterizes it). Export unchanged — still a dense 31.3 MB int8 table.

**Expected effect.** Genuinely uncertain, ±. But it is the only proposal that attacks the
over-parameterization directly, and `reg_init` — the only regularizer in the recipe — is measured
at **exactly zero** effect (`p4-reg0-a`: −0.0000, ci [−0.0001, 0.0001]). The recipe currently has
no working regularizer at all.

**Probe (<2h, free).** Take the shipped table, compute `Δ = W_final − W_init`, SVD it, evaluate
rank-truncated tables at r ∈ {16, 64, 256, 1024} on the dev suite. One multieval pass. If quality
is flat to r=64, the other 960 directions are noise; if it degrades immediately, kill P5 in an
afternoon. **This same probe partly answers "is stella's approximability explained?" — the
under-diagnosed item still open in `EXPLORED.md`.**

### P6. Target design: bare-prefix targets *and* the doc-centroid blend (do them together)

**What.** (a) **Bare (unprefixed) teacher vectors** — `EXPLORED.md` lists this as under-diagnosed
and never fitted; the runtime-prefix ablation is a different question. (b) **Blend the target
toward the teacher's doc-side geometry**: `t = α·normalize(q_teacher) + (1−α)·normalize(mean of the
query's positive doc vectors)`. The table's job is to rank documents, not to imitate a query tower;
the centroid of the relevant documents is the ranking-optimal single vector under a cluster model,
and it lives in exactly the space retrieval scores against.

**Expected effect.** (a) small, ±0.005 — value is closing a listed open item cheaply. (b) plausibly
larger and a real change of optimum, per the Part-1 corollary.

**Compute.** (a) re-encode 561K train queries unprefixed ≈ 9 min, ridge ≈ 2 min/λ. (b) centroid
targets free from cached pool vectors; sweep α ∈ {0, 0.25, 0.5, 0.75, 1} closed-form, ~15 min.

**Risk.** Centroid targets only exist for labelled pairs — unless P1 lands, in which case every
pseudo-query gets one too. `stage0_ridge` builds the Gram in float64 — do not widen the probe
beyond 30,522 vocab in this frame.

**Probe (<2h).** Entirely closed-form. One script, both axes, one multieval. **Highest
information-per-minute item on the list; run on day one alongside P4.**

### P7. Data mix — the genre gap, and what is actually available

The mix is Wikipedia-QA plus e-commerce (ESCI). Zero scientific, biomedical, financial, forum or
argumentative text — which is *exactly* the six. The contamination map blocks the obvious fixes
(S2ORC→SciFact/SciDocs, PubMed→NFCorpus/TREC-COVID, StackExchange-finance→FiQA, args.me→ArguAna).
The clean-4 result (table below BM25 by −0.0311) is what a genre-starved bag-of-tokens model looks
like. Genuinely available and unexplored:

- **Full Wikipedia as an ICT source** (CC BY-SA, already approved). Millions of spans across every
  topic. Contaminates none of the six. ~17 min encode per million docs. **Cheapest genre
  diversification available; feeds P1 directly.**
- **Public-domain technical corpora nobody has considered**: USPTO patents (public domain, densely
  technical), EUR-Lex (explicit reuse grant), US federal documents/CFR. Technical/formal register,
  clean rights, **zero contamination of the six or the reserved four**. Rights review needed but
  likely clean.
- **PubMed Central OA-Commercial subset** — licence-clean, but contaminates NFCorpus and
  TREC-COVID as continuity reads. State the trade to Dylan; do not decide here.
- **arXiv abstracts** — metadata CC0, but SciDocs is built on the citation graph of the same
  papers. Needs rights *and* contamination review.
- **Not available**: StackExchange (2024 no-LLM clickwrap; and any new subforum is eval-adjacent to
  M8's reserved CQADupStack pair), Quora, GooAQ, ELI5, FineWeb/C4.

More spans from *existing* stores is nearly free: `_span` returns the **first sentence only**, one
span per document — that is why 3 of 5 stores exhaust and the 2M request realises 924,704. Drawing
k random sentence-aligned windows per document multiplies the pool 5–20× with no new corpora and no
rights review — encode 2M spans ≈ 33 min.

### P8. Synthetic training *queries*, not doc2query — and the licensing answer

doc2query is a doc-side expansion: re-encodes the document index, changes the server side of the
release. **Synthetic query generation for training pairs** uses the same generator, touches nothing
that ships, and feeds Phase A directly. For a pair-starved Phase A, the second is strictly better
value and should be the M8 form of this lever.

**Licensing, reasoned:** `doc2query--`/`msmarco-t5` are MS-MARCO-trained, therefore tainted by our
own standard (the same inheritance `freeze.assert_releasable` refuses, one level up). flan-t5 is
Apache-2.0 but the FLAN mixture provenance is a second review. **The clean answer is Qwen3
(Apache-2.0)** — and `research/m7-data-licensing.md` already records the precedent: Qwen3-Embedding
generated its synthetic stage with Qwen3-32B, named there "the clean synthetic precedent."
Qwen3-0.6B/1.7B fits the 3080. The file's caveats bind: document the generator's terms separately
from seed rights, run memorization/near-duplicate filters against seeds and all benchmarks, retain
per-query seed provenance.

**Cost.** Qwen3-1.7B on a 3080 ≈ 300–800 queries/min → 500K synthetic queries is ~half a day to a
day of GPU. Do it *after* P1 proves the pair-starvation thesis, since ICT gives 924K pairs today
for free.

**Risk.** Synthetic queries inherit the generator's distribution. Seed diversity (P7) matters more
than volume.

### P9. Polyak/EMA averaging of the table over Phase A

Free variance reduction in an over-parameterized model doing 3.7 noisy epochs; keep an EMA buffer
and export the average. Same economic class as `sqrt` pooling: no bytes, no query-time cost,
identical int8 codes. Expected +0.002–0.006. Probe: one A arm with EMA at three decay rates, folded
into P4's multieval pass. Risk: nil.

### P10. Token-dropout augmentation on the query bag

Drop 10–25% of query tokens during training. Robustness to *which* tokens appear — the
bag-of-tokens analogue of paraphrase variation. Cost: zero. Probe: three arms in the P4 grid. Risk:
at high rates it destroys short queries; cap by query length.

### P11. Ensemble *rankings* as the distillation target — the only viable multi-teacher form

Multi-teacher on **vectors** is architecturally excluded (incompatible spaces). But once P2 makes
the KL a real distillation term, the *target distribution over candidates* is space-free — it can
come from an ensemble (stella + a second strong retriever + BM25), which is standard and reliably
beats a single teacher. Cost: one extra scoring pass over cached candidates. **Vendor-rule note for
Dylan:** a training-time-only teacher is not a shipped component, so the rule arguably does not
bind — but reopening vendor/licensing constraints is Dylan's call. Flag it, do not assume it.

### P12. Row-frequency-aware learning rates

Effective update counts per row span ~6 orders of magnitude (Zipf). Lever #5 failed, but that was a
**post-hoc interpolation** between finished tables, not a **training-time schedule** — a materially
different intervention. Per-row lr ∝ 1/√count. Cost: zero. Probe: two arms in the P4 grid.

---

# Part 4 — Ranking by expected value per week

| # | lever | why it ranks here |
|---|---|---|
| **1** | **P1 — pseudo-queries get positives (ICT pairs)** | Attacks the diagnosed binding constraint; 3.7× the supply; fixes "half the batch trains on a criterion we proved mis-ranks"; costs nothing new; go/no-go probe 45 minutes. |
| **2** | **P2 — make the KL a real KL** | The recipe's rank-distillation term carries ≤1e-4 nats. A defect, not a knob. The 10-minute entropy probe confirms or kills before GPU time. |
| **3** | **P4 + P6 + P9/P10/P12 as one screening pass** | Sub-hour; resolves a never-run sweep and the negatives/step confound; answers the bare-target open item closed-form; prevents P1/P2 inheriting stale hparams. Pre-register as a screen, never an adjudication. |
| **4** | **P3 — train against the fused score** | Fusion is worth +0.057 — ten times any recipe lever — and the released system *is* the fusion; the table is optimized for a metric nobody ships. Complementarity probe ~1 h. |
| **5** | **P7 (Wikipedia + more spans/doc) — the cheap half** | Zero-rights-review, ~50 min of encoding, and it is what makes P1 scale past 924K pairs. |

Below the line but queued: P5 (free SVD probe first), P8 (after P1 proves the thesis), P11 (needs
Dylan's ruling first).

---

# Part 5 — Two warnings for `m8/LEDGER.md` before any lever runs

**1. M8's reserved confirmatory four are far more in-distribution than M7's six were, and every
lever ranked above will look better on them than it deserves.** FEVER's train split is a training
source (11.3% disclosed TRAIN-document overlap); DBpedia is Wikipedia (9.32%);
CQADupStack-android/english are within-family transfer from the pair that *is* M7's out-of-domain
dev subset. M7's six were scientific, biomedical, financial and argumentative — and retention fell
to 0.755 there. `FINDINGS.md` #11 and #13 establish that this dev culture over-rewards
in-distribution gains and that the memorization signature is a gain concentrated near training.
**The reserved four are shaped to reproduce exactly that failure at the confirmatory stage.**
Mined negatives were closed because their gain was `heldout-train` +0.0297 and `hotpotqa` +0.0187 —
on FEVER and DBpedia that same mechanism would read as a win. Consequences: (i) pre-register a
mechanism disclosure alongside the four (per-dataset decomposition + an overlap-restricted subset,
the analogue of clean-4) before any number exists; (ii) keep M7's six as the honest generalization
read, labelled development-informed.

**2. No single lever above can be adjudicated on the current dev instrument.** The
recipe-perturbation band is 0.0027–0.0078 and the out-of-domain subset resolves nothing below
~0.005 at n=1,915. Either (a) fix the instrument first — widen the out-of-domain dev read
(Touché-2020/args.me is the one clean argumentative candidate, with the ArguAna-adjacency caveat
and the usual primary-source licence verification) — or (b) pre-register a single stacked arm
(P1+P2+P4's selected region) against the frozen M7 recipe, sized to clear the band, rather than a
ladder of individually-unresolvable knobs.

---

# Part 6 — The single bet

**If only one recipe lever can run: P1 — give the pseudo-queries their source document and train
Phase A on them.**

The three measured facts converge on it. Capacity is not the constraint (G2 = 0.99999). Phase B is
a 16,000-step reimplementation of a two-minute ridge solve (+0.0008), so the entire recipe's value
lives in a Phase A that runs for 83 seconds on 3.7 epochs of 340,850 pairs and plateaus there. And
924,704 spans — each one a document's own sentence, each with the teacher's frozen doc vector
already sitting in the pool — are currently being spent on a cosine-to-query-vector target that
this project's own `FINDINGS.md` #3 proved mis-ranks. The fix retains a `(store, doc_id)` tuple
that `pseudoq.build()` already holds and throws away, costs no new encoding, needs no licensing
ruling, touches no closed avenue, and the go/no-go is four 83-second arms plus one 14-minute
multieval.

It is the largest supply of unused, licence-clean, correctly-shaped supervision in the repository,
and it is being discarded by a `return` statement.
