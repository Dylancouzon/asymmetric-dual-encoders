# M8 planning — Opus brainstormer: premise-questioning (2026-08-28)

*Verbatim final report of an Opus 5 subagent briefed on CLAUDE.md, instructions-m7/m8.md,
m7/STATUS/RECIPE/FINDINGS/EXPLORED.md, LEDGER sections, FINAL_MATRIX.md and the raw result
artifacts (it recomputed per-dataset macros from m7_final_run.json itself).*

---

## Three numbers nobody has put next to each other, which reframe M8

**(a) Retention is not uniform — and it is inverted from everyone's prior.** Per-dataset
table/teacher on the six:

| | arguana | scifact | nfcorpus | scidocs | fiqa | trec-covid |
|---|---|---|---|---|---|---|
| int8 table | 0.5916 | 0.6101 | 0.3124 | 0.1677 | 0.3728 | 0.5490 |
| teacher | 0.6369 | 0.7796 | 0.4134 | 0.2395 | 0.5536 | 0.8234 |
| **retention** | **0.929** | 0.783 | 0.756 | 0.700 | 0.673 | **0.667** |

ArguAna — 193-word queries, the dataset everyone predicted would kill a bag-of-tokens encoder — has
the **highest** retention by a mile. The worst two are the **shortest**-query sets. The
teacher-overlap confound does not explain it: ArguAna *and* FiQA are both on stella's disclosed
list, and they sit at the two extremes. **The bag gets better as queries get longer, and the loss
is concentrated where queries are short.** Every intuition M8 inherits about where to spend is
pointed the wrong way.

**(b) The class is nowhere near its own in-class ceiling, and nobody measured the ceiling.** On the
out-of-domain dev pair: closed-form ridge **0.350** → trained table **0.3672** → teacher
**0.4806**. Training bought 13% of the gap. `m7_ridge_vs_trained.json` shows the trained-minus-
ridge effect (+0.0205) lands on `hotpotqa`/`heldout-train`, not out-of-domain. Meanwhile the
capacity probe hits 0.99999 by memorising 3,452 queries. So: unlimited in-sample capacity, 13% of
the generalisation gap captured, and **no measurement anywhere of what the best possible order-free
query encoder could do.** "The gap is architectural" was inferred from the clean-stack tax — a
*data* experiment. It is a statement about MS MARCO, not about architecture.

**(c) On the like-for-like row, M7 already won and the report may be under-claiming.**
`lightretriever-dense-websearch` (LR's *single* table, the same product shape as ours) = 0.4320.
Ours = 0.4339 — with a 400M frozen tower instead of a co-trained 1.5B one, a 31 MB artifact instead
of 466 MB (15×), and a 1024-d index instead of 1536-d. The bar we missed (0.4583) is LR's
*per-task oracle* config — a different table per dataset, a thing a single released artifact
structurally cannot be.

---

## Premise 1 — the frozen document tower

**Verdict: REOPEN-with-probe (cheap form) + ESCALATE-to-Dylan (expensive form). The cheap form was
dismissed on a sentence.**

*Does the mandate forbid it?* No. `instructions-m7.md` §Decision authority: "You decide:
architecture, objectives…". CLAUDE.md names this exact premise as reopenable. Licence: stella is
MIT, derived from gte-large (Apache-2.0). Vendor: NovaSearch clean. Nothing blocks it.

*Steelman AGAINST:* the only existence proof of a co-trained tower with a bag query side is
LightRetriever — 1.5B doc tower, co-trained end-to-end, lands at 0.4320 single-table. Our frozen
400M tower lands at 0.4339. Co-training bought them nothing measurable at 4× the parameters.
Co-training also destroys the free negative bank (`n_neg=32768` from a 2M frozen bank exists
precisely because doc vectors are static; a training tower forces in-batch B≈128 with GradCache).
And a co-trained tower makes the release two artifacts and forces every adopter to re-encode — the
"drop it onto your existing stella index" story dies.

*Steelman FOR — the version to actually run:* **you do not need to touch the tower to break the
frozen-space premise.** `m7/LEDGER.md:577` corrected the algebra — a doc-side map is NOT absorbable
once documents are renormalised (rank agreement 1.000 unnormalised, 0.000 normalised) — and then
concluded *"It changes nothing we can do."* That conclusion does not follow; it says the opposite.
A doc-side map `d → normalize(g(d))`, trained **jointly with the table**: (i) needs zero corpus
re-encoding at training time — it applies to cached frozen vectors, so the 2M negative bank
survives; (ii) costs one 1024×1024 matmul per doc at index time; (iii) trains in the same ~20 min;
(iv) can output **512 or 256 dims**, halving/quartering *both* the doc index (2.05 → 1.02 →
0.51 GB/1M) *and* the table (31 → 16 → 8 MB int8). That last point is the strongest Qdrant-product
argument in this list: index bytes are the customer's real bill.

This is the on-ramp: it tests "is the frozen space the binding constraint?" at ~1% of the cost of a
LoRA co-train. If the doc-map gains ≥0.02 on the out-of-domain dev pair, escalate to Dylan for the
tower fine-tune with real numbers.

*Arithmetic for the expensive form:* LoRA on stella-400M fits 10 GB. Training is days (negative-
bank problem). Re-encoding after training ≈ 100 texts/s → **28 h for the 10.1M-doc reserved-4
corpus alone**. One-shot, no-iteration budget.

**Next step:** pre-register a joint `(table, doc-map)` arm — linear 1024→1024, 2-layer MLP,
linear 1024→512 — adopted on the out-of-domain dev subset with the standard bar, disclosing that
the doc side gains a released artifact. Do this **first**; it is the cheapest structural test
available.

## Premise 2 — the teacher

**Verdict: KEEP; REOPEN only a bounded probe of post-2026-08 models judged on the ratio, not the
ceiling. Reject bigger/wider teachers on arithmetic.**

The decisive number: stella's **approximability ratio 0.7156 is the highest of all ten candidates**
(bge-base 0.6855, e5-base 0.6734, bge-large 0.6132, arctic-l 0.5261, gte-large 0.4315). And
Spearman(ceiling, table) = 0.000. So: +0.020 via a swap at constant ratio needs a teacher-symmetric
六-set row of 0.6009 (~MTEB-Ret 61.5) inside vocab ≤50K / dim ≤1024 / commercial licence — **no
such model is known to exist**. +0.020 via the ratio at constant ceiling needs 0.7156 → 0.7504 —
possible in principle (spread 0.43–0.72) but we are at the top of the observed distribution with no
mechanism to search on (the "approximability unexplained" open item).

**Bigger teachers are worse on arithmetic**: 4096-d → 8.2 GB/1M doc index, 125 MB table, ~10x
encode. Qwen3-Embedding-0.6B at 151,669 vocab → 621 MB int8 table (worse than the 466 MB LR
artifact we exist to undercut). The vocab×dim budget binds before quality does; MRL truncates dim,
never vocab.

**Next step:** one Sonnet sweep for permissive retrieval encoders since 2026-08 with vocab ≤50K,
dim ≤1024; probe ≤3 closed-form on the existing criterion; require **ratio > 0.72**. If any
candidate has a ~50K vocab, budget a float32 Gram *for all candidates* so comparability survives
(the granite/gte-modernbert exclusion is a solver-memory limit, not merit). Free correlational work
first (anisotropy, effective rank, in-sample ridge R² vs ratio over the ten cached candidates) to
convert "stella is best" into a searchable attribute.

## Premise 3 — where the "zero" actually is

**Verdict: ESCALATE-to-Dylan, but only after two probes. The algebra is decisive and sharper than
expected.**

*The algebra.* A post-pooling matmul is exactly absorbable: `M(Σ_t √c_t w_t) = Σ_t √c_t (M w_t)` —
zero new capacity. Insert **any** nonlinearity — `v → σ(Av)`, gating, ReLU — and absorption fails.
The boundary is precise: **linear-after-pooling is free and worthless; nonlinear-after-pooling is
the first genuinely new capacity that costs nothing at index time.**

*The consequence.* For ranking, the final L2 is a no-op, so the query encoder IS a linear map from
the √count vector `x ∈ R^30522` to `R^1024`. A typical BEIR query has 10–30 distinct tokens; a
30,522-atom dictionary measured in 1024 dimensions supports sparse recovery for roughly s <
1024/(2 ln(n/m)) ≈ **150**. **The pooled vector is information-theoretically invertible for short
queries** — the bag is *not* lost by summing. Therefore the ceiling of "table + nonlinear head" is
not the linear-in-bag ceiling; it is the **bag-of-words ceiling** — an order-blind transformer. And
it predicts the retention pattern in (a): recovery holds for FiQA/TREC-COVID/NFCorpus (10–30
tokens, worst retention, most to gain) and fails for ArguAna (~250 tokens ≫ 150, already 0.929 —
averaging genuinely is a good summary of a 200-word argument). The measured retention curve and the
compressed-sensing bound agree.

*Cost.* Current int8 query path 0.3795 ms, 31.84 MB, 0.224 s hydration. A 2-layer int8 head
(1024→1024→1024, 2.1M params): +2.1 MB (+6.6%), +~0.1–0.3 ms, cold start unchanged, no tokenizer
change, no attention, constant in query length. Total ≈ 0.55–0.7 ms vs a 33M transformer's 5 ms +
1.3 s load. Every operational property survives; the only casualty is the *word* "zero".

*How to put it to Dylan.* Not "may we add a neural net" but: "Group B is labelled
zero-neural-query-compute. A 2 MB MLP head keeps 0.7 ms / 34 MB / instant cold start but forfeits
that label. Here is what it buys: X. Is the product 'no transformer at query time' or 'literally no
learned computation'?"

**Next step, in order:** (i) the **bag-ceiling probe** — re-encode dev queries with the teacher on
*token-shuffled* input (and the sorted-unique bag) and score. ~2,000 queries, minutes. Measures the
ceiling of every order-free query encoder in one shot. If shuffled-teacher ≈ teacher, the whole
0.3672 → 0.4806 gap is linearity tax and the head is the highest-value lever in M8. If it collapses
to ~0.38, we are at the class ceiling, the head is dead, and M8 must move to the doc side or the
lexical arm. **The single most decision-relevant measurement available; costs an afternoon.**
(ii) Only if (i) passes: a dev-only MLP-head arm; require ≥0.02 on the out-of-domain subset before
spending Dylan's attention.

## Premise 4 — the bar and the product framing

**Verdict: REOPEN — the registered M8 bar is necessary but not sufficient, and the fix is only
legal now.**

1. **The bar is purely relative with no floor.** M7's clean-4 C2 = −0.0311: the table is below BM25
   there. An M8 that beats M7 by +0.012 on the reserved four could still be below BM25. **Add an
   absolute leg: the released system must CI-resolve above BM25 on the reserved-4 macro.** BM25 is
   the one comparator `instructions-m8.md` §3 already permits there.
2. **The primary comparison should be system-level.** M7 proved it: the table missed its bar
   (−0.0243), the *system* tied the aim (+0.0043) and beat every Group-B row descriptively. Make
   **fused-M8 vs fused-M7** the primary confirmatory leg and the dense table a labelled secondary —
   fusion re-selected on dev for each.
3. **On the reserved four we have no OpenSearch or LightRetriever row at all** — M8's headline
   would have no external anchor. Fixable only now: encode
   `opensearch-neural-sparse-encoding-doc-v3-gte` over the reserved-4 corpora (~10 h for 10.1M
   docs), freeze its per-query vectors into `perquery.json` **before any M8 design decision**,
   exactly as `m7_bars_clean4.json` was precomputed. LR is probably not worth it (1.5B + 1536-d
   over 10.1M docs ≈ 31 GB, ~40 h), but the websearch single-table row is the honest like-for-like
   comparator if affordable.

Separately, free: **the M7 report should carry the LR-websearch comparison** (0.4339 vs 0.4320,
single table vs single table, 15× smaller) as a labelled exploratory row from `perquery.json`.

## Premise 5 — BM25 as the lexical arm

**Verdict: REOPEN, and one EXPLORED entry is wrong for our architecture.**

Fusion contributed +0.0572 over the table and +0.0737 over BM25, from the weakest lexical partner
in existence. The ladder: BM25 0.4174 → fused 0.4911 (measured); BM25F ~0.42–0.44 → ~0.50;
BM25+doc2query full dose ~0.44–0.46 → ~0.51–0.52; own learned sparse doc encoder 0.48-class →
**~0.53–0.55 fused** — above arctic-embed-m-v1.5 (0.5264), the matrix winner. Highest ceiling
anywhere in this project.

**The EXPLORED correction.** `research/m7-research-2026-08-26b.md` §1 killed doc2query on Weller et
al. ("expansion helps weak retrievers and harms strong dense ones") and noted Doc2Query++ recovers
gains only "by keeping a second index and fusing at scoring time … a different architecture, and it
reintroduces an index we do not have." **We now have exactly that index.** The negative finding
applies to the arm we would never expand (dense); the positive finding applies to the arm we would
(BM25 — precisely the "weak retriever" expansion helps). The M7 probe (+0.0054 [−0.0007, +0.0114]
at N=5/doc) was fusion-agnostic and at 1/8 the published dose. Re-scope the lever as **"expand the
BM25 arm only"**; it is no longer a trap. Still blocked on Dylan's generator ruling — now worth
chasing.

**Cheapest first move:** BM25F over title/text, weights fitted on dev and frozen. Near-zero cost;
five of the ten datasets in play have real titles. Caveat: BM25 is the *frozen fusion function*, so
any change re-registers the whole fusion selection.

**Middle path worth a probe:** own doc-side term-weighting model (DeepImpact-shaped: re-weight
existing terms, no vocabulary expansion). Keeps the inverted-index shape, no vendor/circularity
issue. Risk: a term-weighting model may lose much more than +0.006 to the MS MARCO exclusion —
unmeasured.

## Premise 6 — multi-vector / late interaction

**Verdict: KEEP single-vector for the confirmatory run — the arithmetic rules it out on this box —
but REOPEN one nearly-free piece.**

Full late interaction, priced: docs average 166.9 tokens; per-token doc vectors at 128-d int8 →
216 GB for the reserved-4 corpora (storage survives); compute kills it — exact MaxSim on FEVER
alone ≈ 3.6e15 FLOP ≈ 66 h on a 3080, four orders above the 5 s single-vector brute force. Pooling
docs to 32 vectors cuts storage, not the ratio. Feasible only as retrieve-then-rerank — a different
product and claim. Plus: ME-BERT's own table shows single-vector winning at 50-token passages, and
there is no evidence multi-vector heads bolt onto a frozen tower — premise 6 collapses into
premise 1.

*The free piece.* Qdrant's native MaxSim with one doc vector degenerates to exactly our
sum-pooling. The interesting reduction is the other one: `max_t (w_t · d)` or top-k — trivially
\|q\| batched searches merged client-side (~12 × 0.48 ms ≈ 6 ms, doc index unchanged). **Nobody has
tested whether sum is the right reduction. Sum / max / top-3-sum / log-sum-exp is testable on dev
today, with the existing table, in under an hour, at zero training cost.** If any member beats sum,
a free architectural win that also changes what the table should be trained for; if none, a
one-line EXPLORED entry closing a family.

## Premise 7 — eval economics on the reserved four

**Verdict: good news, and it changes which levers are worth running.**

Calibrating off M7's clean-4 interval (per-query paired sd ≈ 0.284 for dissimilar systems), the
reserved four (n = 6666, 400, 699, 1570) give an equal-weight 4-macro half-width ≈ **0.0094** for
dissimilar systems, ≈ **0.0051** for near-siblings (sd ≈ 0.15); inflate ~15–20% for Holm at family
α=0.025. **M8 needs Δ ≈ 0.010–0.012 to confirm, ≈ 0.007 if near-sibling.** The reserved four are a
statistically friendlier panel than the six (9,335 queries, no n=50 set).

**What it rules out.** The recipe-perturbation band is 0.005, and every adopted M7 lever effect
sits inside it. Lever-stacking is not merely likely to fail — it is scientifically hollow even if
it passes. M8 has one confirmatory access and needs **one change worth ≥0.02** to be safely
confirmable with margin. Spend the milestone on one structural direction plus its ablations, not a
lever programme.

**Two panel problems, fixable only now.** FEVER-train is a TRAIN source and FEVER-test shares its
corpus (11.3%); DBpedia is Wikipedia (9.32%); android/english are within-family transfer from the
dev components we selected on. Every one of the four carries a caveat and 2 of 4 flatter us.
Pre-register now: report the 4-macro **plus** a Wikipedia-pair subset and a CQADupStack-pair
subset, exactly as M7 pre-registered clean-4.

**Now-or-never:** after M8 burns these four, **M9 has no untouched partition left.**
`instructions-m9.md` sets a bar on the six — development-informed twice over by then. Either
reserve a fifth and sixth set now (freeze assets, never score) or accept M9 ships without a
confirmatory panel. Raise with Dylan.

## Premise 8 — the one I went hunting for

**Verdict: REOPEN. The load-bearing claim of M7's closing paragraph — "the gap is architectural,
which is where M8 should spend" — was never measured.**

It is inferred from the clean-stack tax, which is evidence about MS MARCO, not about architecture.
The alternative never eliminated: **the objective and the in-domain-supervision gap**, which the
numbers actively support (ridge 0.350 → trained 0.367 → ceiling 0.481; training captured 13%; every
out-of-domain lever moved within 0.0040 — below the instrument's resolution, not zero).

Three cheap measurements settle it, none touching a protected set:
1. **The bag-ceiling probe** (premise 3). Minutes.
2. **The in-domain generalisation ceiling**: split the CQADupStack dev components' queries 50/50,
   train an oracle table on one half with in-domain positives, score the held-out half. The
   capacity probe made honest. If the held-out half reaches ~0.45 against a 0.481 ceiling, the
   architecture is fine and data/objective is the whole story; if it stalls at ~0.38, the
   architecture caps and premises 1/3/5 are the only places to spend. Hours.
3. **The doc-side instruction** — flagged in `m7-research-2026-08-26b.md` §2 as "the cheapest
   untried structural lever we have", genuine non-absorbable capacity, never run. Re-encode dev
   corpora with a fixed doc instruction, refit the closed-form table, compare. ~2–4 h; costs the
   released system nothing at query time.

Adjacent, open, cheap: bare-target closed-form (~1–1.5 h); `phase3_hparams` never ran.

---

## Top 3 premise-changes, ranked by expected value

**1. The frozen doc *space* — a jointly-trained doc-side map (premise 1, cheap form).** Genuine new
capacity by the project's own corrected algebra; buildable with existing machinery in a 20-minute
run; no corpus re-encode; can shrink the doc index and table together at 512-d; dismissed on a
single unexamined sentence. The honest on-ramp to the tower fine-tune.

**2. Measure the ceiling before choosing the lever (premises 3 + 8).** Bag-ceiling probe, in-domain
oracle-generalisation probe, sum/max/top-k scoring sweep. Under a day total; determines whether M8
stays in this artifact class at all. If the ceiling probe says the class is exhausted, this saves
the milestone; if not, it justifies asking Dylan for the 2 MB nonlinear head with a number
attached.

**3. Upgrade the lexical arm (premise 5).** Fusion is where the released system lives; it runs on
the weakest partner. BM25F nearly free; doc2query re-scoped to the BM25 arm only; own learned
doc-term weighting has the highest ceiling measured here (~0.53–0.55 fused).

**Escalations to bundle into one message to Dylan:** (i) the scope question — "no transformer" vs
"no learned computation", with 0.7 ms / 34 MB / instant-cold-start numbers; (ii) the doc2query
generator ruling, re-scoped to the lexical arm; (iii) advance notice that a doc-side map/tower
makes the release two artifacts; (iv) **the M9 reserve problem — M8 burns the last untouched
partition.** (i) and (iv) are time-critical; (iv) is only legal before M8 begins.

## If I were starting M8 from scratch

Stop treating the query side as the interesting half. The measurements say the query table is close
to what a linear-in-bag map can do against a space never built for it; the biggest per-dataset
losses are on *short* queries where the pooled vector is provably information-preserving; and the
one place we already win is fusion — the arm where the *document* side does the work. Build a
**jointly-trained asymmetric pair with an aggressively cheap query side and a deliberately
expensive, deliberately ours document side**: keep the 30,522-row WordPiece table as the query
encoder, but train it against a **learned doc-side head over the frozen stella vectors** — jointly,
contrastively, with the 2M-vector negative bank intact — outputting **512 dims**, so the
deliverable is a 16 MB query artifact and a *halved* document index, which is the number a Qdrant
customer actually pays. Alongside it, a doc-side lexical arm we own — term re-weighting first,
expansion if Dylan clears a generator — and fuse, because the released system is the fusion and
pretending otherwise cost M7 its headline. Spend the first week on nothing but the three bounding
probes, pre-register a single structural direction with a ≥0.02 target (the panel arithmetic makes
anything smaller unconfirmable *and* inside the perturbation band), and freeze OpenSearch's
per-query vectors on the reserved four before making a single design choice. The headline to aim at
is not "we beat LightRetriever's oracle config" — it is **"zero-transformer query side, matches a
33M transformer's retrieval quality at half its document index and 1/10th its query latency"**,
which the fusion ladder says is reachable and which is the claim Qdrant can actually sell.
