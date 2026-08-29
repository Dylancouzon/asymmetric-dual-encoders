# Untried levers for the M7 lookup-table query encoder — ranked sweep (2026-08-27)

Scope read before writing: CLAUDE.md, m7/STATUS.md, m7/LEDGER.md, m7/EXPLORED.md, m7/RESULTS.md,
m7/CODEMAP.md, research/m7-novelty.md, research/m7-codex-review-2026-08-27.md (ideas section),
m7src/train.py Cfg, m7src/pseudoq.py.

Deliberately NOT listed (already tried, running, or queued — do not double-spend):
count-saturation pooling eval-only (lever #4, running), A-only bigram residual rows (Codex idea #2,
queued exploratory), B-phase extension past 16k (Codex idea #4, queued), TRAIN-count row
interpolation toward B (Codex idea #3), closed-form bigram (CLOSED, structural), doc2query
(parked, Dylan's licensing call), pseudo-query scale-up per se (lever #2 adopted at 2m; further
scale is the queued B extension), any per-token scalar weighting / centering / whitening
(absorbable, `results/m7_absorb_check.json`).

A framing fact that shaped the ranking: Stage-0 established the binding constraint is
**generalisation, not expressivity** (ridge optimum interior; capacity probe ≈1.0 on dev).
So the highest-expected-value levers are ones that change the *training signal* (data and loss),
not ones that bolt runtime machinery on. For training-signal levers the "absorbable?" test does
not apply — they move the rows themselves; the ceiling they push is the generalisation gap.
For runtime levers I state non-absorbability explicitly, because that has been the graveyard.

Cost basis: one full B16k→A2500 chain ≈ 3–4 h on the 3080 (inferred from RESULTS cadence:
several chains per overnight). Eval-only full-suite pass ≈ 0.3–1 GPU-h (hotpotqa's 5.2M-doc
brute force dominates). All ideas below keep the frozen teacher, the frozen doc vectors, the
clean data stack, and — except where loudly flagged — the exact gather→weighted-combine→normalize
query path with 0 extra MB.

---

## 1. Vector-PRF at zero neural cost (Rocchio over the frozen doc vectors), eval-only probe

**Mechanism.** Two-pass search: v0 = table query vector → retrieve top-k → v1 =
normalize(α·v0 + β·mean(top-k frozen doc vectors)) → final retrieval with v1. No transformer,
no training, no new artifact bytes; the "expansion model" is the document index we already ship.
The mechanism argument for *this* system specifically: PRF replaces part of a weak query vector
with an average of strong teacher doc vectors. The lit says gains anti-correlate with retriever
strength, and our end-to-end query representation is the weakest part of the system by
construction — the same helped-regime argument EXPLORED.md already accepts for doc2query.

**Why not absorbable.** A table maps a token multiset to a *fixed* vector; the PRF vector depends
on the corpus being searched at query time. No table can reproduce it. (It is a query-path
change, though — see flag.)

**Evidence.** Li, Mourad, Zhuang, Koopman, Zuccon, TOIS 2023 (arXiv 2108.11044): training-free
vector PRF (Average/Rocchio) on ANCE/TCT-ColBERT — consistent gains on MAP/recall-oriented
metrics, "does not improve RR and nDCG@1 in a consistent manner." Honest read: for *strong*
dense retrievers, shallow-metric gains are mixed; nobody has measured it on a bag-of-tokens
query encoder, which is exactly the weak-encoder regime where the mechanism predicts the most.
This is a hypothesis test, not a literature-backed sure thing.

**Cost.** 0 MB. ~0.5–1 GPU-h (one extra brute-force pass per dev component, small α/β/k grid —
pre-register the grid: e.g. k∈{5,10}, β∈{0.1,0.3,0.5}, one winner by dev macro).
**Falsifier.** Dev macro delta ≤ 0 at every grid point → closed in one afternoon.
**P(>+0.005 dev macro): 0.25.**

**FLAGS, loud.** (a) This changes the released system's story from "one lookup, one search" to
"one lookup, two searches" — ~2× query latency (still ~2 ms class, still zero neural compute, and
the cold-start story is untouched). Precedent exists in-project: the aim-bar candidate already
allows labeled BM25 fusion; PRF needs the same labeled treatment and, per the house rule, a
protocol note written **before** the numbers. (b) ANN interaction unknown — PRF vectors are
doc-space centroids; check at the ANN sweep, not after. (c) If adopted, comparators stay as
released (they could all run PRF too; the report must say the comparison is system-vs-system,
not representation-vs-representation).

---

## 2. Long-span B distillation (length curriculum on the adopted lever-#2 machinery)

**Mechanism.** `pseudoq._span` caps pseudo-queries at the first sentence, ≤32 words, and the real
TRAIN queries sit at p50=13 wordpieces. The table has therefore *never been trained on what a mean
of 150–300 rows should look like*, yet ArguAna — 1 of the 6 confirmatory sets — has ~250-word
queries, flagged since M1 as the stress case, and dev's only long-query slice is 55 HotpotQA
queries. Add a second span pool: multi-sentence spans of 64–256 words from the same decontaminated
doc stores, mixed into objective B at a pre-registered fraction (e.g. 15–25% of the pseudo half).
Zero new licensing surface (same approved corpora), zero new code beyond `_span`, and the mandate
explicitly pre-authorises synthetic queries "including long counter-arguments."

**Why not absorbable.** Training-data lever; changes the rows. Not a runtime transform.

**Evidence.** LightRetriever trains its token table on full-length natural data and holds ~95%
average retention across BEIR *including* ArguAna (arXiv 2505.12260) — so a trained mean-of-rows
can represent long text when it has seen long text. The complexity-diversity result
(arXiv 2602.09448) finds query *diversity of form* is what buys generalisation, complexity alone
insufficient. No paper isolates "train the bag on long spans, win on long queries" — this is an
inference from mechanism plus the known ArguAna failure mode of bag-of-token encoders.

**Cost.** 0 MB. One chain ≈ 3–4 GPU-h (+ ~1 h to encode the new span pool with the frozen teacher).
**Falsifier, cheap and pre-registerable.** Before the chain: teacher-agreement (cosine +
overlap@10, no qrels needed — EXPLORED.md already notes these are implemented) of the *current*
candidate on held-out long spans vs short spans. If the long-span agreement is not materially
worse than short (no gap to close), kill before training. After the chain: same metric + full dev
suite non-regression.

**P(>+0.005 dev macro): 0.15** — say this plainly: dev barely measures long queries, so the dev
macro is the wrong place for the payoff to appear. The real target is the six (a +0.03 on ArguAna
alone is +0.005 on the six-set macro). Rank it high anyway because it is the only lever on this
list aimed at a *known, named* weakness of a *confirmatory* dataset, it is cheap, and its
falsifier costs nearly nothing. Risk to hedge: one table serving two length regimes may trade
short-query quality away — the dev suite catches that side.

---

## 3. ICT-style A-phase pairs: span → source-doc positives (scale the phase that actually wins)

**Mechanism.** Every adopted gain since the teacher swap has come from objective A or from
feeding B more query text. A currently trains on only 340,850 qrels pairs; the pool is 6.17M
frozen doc vectors. Inverse-Cloze-Task construction: take spans (the pseudo-query machinery
already samples them, decontaminated), positive = the source document's *existing pool vector*,
negatives = the random bank that already wins. This multiplies A-phase supervision ~10–20× at
zero licensing cost and zero teacher re-encode (positives are rows we already have). Optionally
drop the sampled sentence from nothing — no doc re-encode is possible or needed; the span≠doc
asymmetry is the point.

**Why not absorbable.** Training-data lever; changes the rows.

**Evidence.** ICT (Lee et al., arXiv 1906.00300) and Contriever (arXiv 2112.09118, cropping +
contrastive) established span→source contrastive as the strongest unsupervised dense recipe —
Contriever is the best unsupervised dense bi-encoder on BEIR nDCG@10 and beats BM25 on
Recall@100; AugTriever (arXiv 2212.08841) extends it with scalable pseudo-queries. None of them
do it against a *frozen* doc tower with a table student — that combination is ours — but the
supervision signal is proven.

**Cost.** 0 MB. One chain ≈ 3–4 GPU-h (pair construction is bookkeeping over existing artifacts).
**Falsifier.** One arm at a fixed mix fraction (e.g. 50% ICT / 50% qrels pairs in A), full-suite
compare vs the candidate under the standing signflip+CI bar. A resolved loss closes it.
**P(>+0.005 dev macro): 0.30** — the most likely single-chain win on this list. Risk: span→own-doc
is lexically easy for a linear model, so the gradient may be mostly what B already taught; the
50/50 mix arm answers that directly.

---

## 4. Train-through nonlinear pooling (conditional on lever #4's eval-only read)

**Mechanism.** Lever #4 (binary/cap2/sqrt) is being evaluated on a table *trained for mean
pooling* — an eval-only mismatch that understates what multiplicity-dependent pooling can do.
If any arm is ≥ neutral, retrain one chain with the winning mode **in the forward pass** so the
rows adapt to it. Same for an elementwise max/mean blend (v = normalize(a·mean + b·max), a,b
learned scalars): max is an order statistic across gathered rows, so it is genuinely outside the
linear family, still gather+combine+normalize, zero latency cost.

**Why not absorbable.** Multiplicity-dependent pooling is the one query-side transform besides
n-gram rows proven non-absorbable to machine precision (`results/m7_absorb_check.json`). Max
pooling is likewise non-linear in the gathered multiset.

**Evidence.** The strongest inference-free system we must beat made exactly this choice: the
OpenSearch doc-only query side is **binary presence × IDF**, not counts (arXiv 2411.04403;
BEIR-13 50.35, +3.3 over prior inference-free SOTA) — the binary half is capacity, the IDF half
is absorbable for us. TILDE-family query sides are binary too. Max-vs-mean evidence is
off-domain but directionally consistent (max pooling repeatedly beats mean for salient-token
tasks; e.g. the MIL/hallucination line reports +1.5–2.5 AUROC — weak evidence, stated as such).

**Cost.** 0 MB. One chain ≈ 3–4 GPU-h per mode; run at most one, chosen by the eval-only probe.
**Falsifier.** The running lever-#4 probe *is* the gate: if all three eval-only arms are resolved
losses, spend nothing further on counts (the max-blend arm survives that kill, barely — it tests
a different nonlinearity, and I'd only run it if pooling shows any life at all).
**P(>+0.005 dev macro): 0.20** (conditional; unconditional ~0.12). Short queries rarely repeat
tokens, which caps the count-family upside; the max blend is the more interesting half.

---

## 5. Joint B+A: keep a distillation anchor inside the contrastive phase

**Mechanism.** The recipe is sequential B→A; during A the only teacher signal is `reg_init`'s
decaying pull. The bigram post-mortem showed A's *deviations from the teacher are the gains* —
but that cuts both ways: on rare rows with few A updates, drift is noise, and Codex idea #3
(interpolation back toward B) is a post-hoc patch for exactly that. The loss-level version:
A steps carry a small auxiliary cosine/KL term to the teacher query vector on the B mix
(e.g. 0.1–0.3 weight), so common rows keep earning ranking gains while rare rows stay anchored.
One arm, one weight, pre-registered.

**Why not absorbable.** Loss lever; changes the rows.

**Evidence.** LightRetriever's auxiliary KL alignment is worth +0.5 BEIR (removal costs −0.5,
arXiv 2505.12260 ablation) — small but real, in the closest published system. Multi-task
distill+contrastive is the standard recipe in bge/E5-class training. Margin-MSE
(arXiv 2010.02666) is the classic evidence that matching a teacher's *score structure* transfers
better than matching nothing when the student is weak — though its usual teachers (MS MARCO
cross-encoders) are license-dead here, the frozen stella scores are a legal teacher signal we
already possess.

**Cost.** 0 MB. One chain ≈ 3–4 GPU-h.
**Falsifier.** Full-suite compare vs candidate; also read the per-component split — the
hypothesis predicts the gain concentrates on heldout-train/hotpotqa (rare-vocab components), so
a flat gain profile would refute the mechanism even if the macro ticks up.
**P(>+0.005 dev macro): 0.25.**

---

## 6. Checkpoint soup over the A-phase peak (and across seeds), eval-only

**Mechanism.** A-phase proxy curves peak-and-turn (`p2x-rn-3e4` peaked at 500; s2500 chosen by
every-500 proxy), and best-step re-runs wobble ~0.0007 from CUDA-atomics nondeterminism. Average
the saved tables in a window around the selected step (they are just matrices; the average of
tables is a table), optionally across 2–3 seeds of the same recipe. Smooths the step-selection
noise the protocol currently absorbs by re-running.

**Why not absorbable — n/a, and that's fine.** This is not runtime capacity and doesn't claim to
be; it is a training-side variance reduction that produces different (better-averaged) rows.

**Evidence.** Model-soup-for-embeddings (Jina, 2024/25 writeup; Wortsman et al. soups): merging
checkpoints matches the best individual checkpoint and removes overtraining defects. Gains in
the literature are a few tenths of a point — consistent with our step-to-step wobble.

**Cost.** 0 MB, ~0.3 GPU-h (pure eval; checkpoints already on disk if `save_table` kept them —
if only best-step artifacts were kept, cost includes one re-run with dense checkpointing, +3 h).
**Falsifier.** One full-suite eval of the souped table vs the candidate. Ten-minute read.
**P(>+0.005 dev macro): 0.10** — expected gain is +0.001–0.003; listed because the cost is near
zero and it stacks with everything else. Protocol note: souping is a *selection rule change*;
pre-register the window before reading its number.

---

## 7. Self-mined (student-error) negatives, one curriculum arm

**Mechanism.** Teacher-mined hard negatives resolvedly HURT (−0.0034 vs random); the standing
conclusion is "random only". But teacher-mined ≠ student-mined: ANCE-style negatives are docs the
*current table* ranks highly and the qrels say are wrong — they target the student's own
confusions, which for a bag-of-tokens model are characteristically lexical (shared tokens, wrong
meaning), a failure class random negatives almost never sample. Mine top-100 by the current
table, filter with the existing `fn_margin` teacher check, mix ~25% into the bank.

**Why not absorbable.** Training-data lever.

**Evidence.** ANCE (arXiv 2007.00808) built the case that negatives from the *retriever being
trained* beat static negatives; DRAGON (arXiv 2302.07452) got BERT-base to BEIR 47.4 with
progressive supervision rather than one-shot hard negatives. Counter-evidence in-house: the
mined-negatives loss above, and the capacity-mismatch literature warning that hard negatives can
hurt weak students. Genuinely two-sided.

**Cost.** 0 MB. Mining pass is minutes (the pool loop-order fix), one chain ≈ 3–4 GPU-h.
**Falsifier.** Full-suite compare; a resolved loss closes the whole negatives family for good
(random would then have beaten teacher-mined, provided-hard, and self-mined).
**P(>+0.005 dev macro): 0.15.**

---

## 8. Length-bucket-conditioned token weights (tiny genuine capacity, unproven)

**Mechanism.** A per-token scalar weight is absorbable; a weight w(t, b) that varies by query
**length bucket** b (say 4 buckets: ≤8, 9–16, 17–48, >48 wordpieces) is not separable into
per-row scales — it is equivalent to interpolating between per-bucket tables. Learn the 30,522×4
weight matrix in the A phase (rows shared, weights bucketed). Directly encodes "function words
matter differently in a 6-token query than in a 200-token argument," which is the cheapest
version of the short/long conflict in idea 2.

**Why not absorbable.** Non-separable dependence on |T|: folding w(t,b)·row_t into the row
requires one scale per row, but each row needs *b different* scales; `m7_absorb_check`'s
per-token-scalar proof covers only b=1. (Any *purely* length-dependent global factor still
cancels under L2 — the capacity lives strictly in the t×b interaction.)

**Evidence.** None direct — no published system conditions static token weights on query length.
Adjacent support only: verbosity/length normalisation is a classic BM25 lever (b parameter), and
the opensearch binary-vs-count choice is itself a crude length interaction. Treat as a
mechanism-only bet.

**Cost.** ~0.24 MB fp16 (30,522×4). One chain ≈ 3–4 GPU-h.
**Falsifier.** If idea 2's teacher-agreement probe shows no short/long gap, this dies with it.
Otherwise one arm, full-suite bar.
**P(>+0.005 dev macro): 0.10.**

---

## Traps — things that will be proposed and should be refused or flagged

- **Softmax "attention-lite" pooling over learned per-token logits** — a trap. The softmax
  denominator is a per-query scalar and the final L2 normalize kills it, so softmax pooling ≡
  exp-scaled per-token weights ≡ absorbable. Zero capacity dressed up as attention.
- **Ensembling tables / averaging two trained tables of different recipes as a *runtime* claim** —
  the average of tables is a table; fine as training (idea 6), meaningless as capacity.
- **IDF or corpus-statistics weighting at query time** — per-token scalar, absorbable, and the
  trained weights are already IDF-like (spearman −0.44, EXPLORED.md).
- **Margin-MSE / rankers as extra teachers** — the loss is legal (idea 5 uses stella's own
  scores), but every strong public cross-encoder/reranker teacher is MS MARCO-trained: using one
  to *generate training signal* is the same licensing question that parked doc2query. Dylan's
  call, not a session's.
- **Per-token ReLU scoring (SPARTA-style, score = Σ_t max(0, r_t·d))** — genuine capacity
  (SPARTA, arXiv 2009.13013, showed token-level interaction beats sequence-level with an
  inference-free query side), but it **breaks the single-vector query path**: no longer one ANN
  search, needs per-token retrieval + fusion or index-time precomputation (which is the sparse
  family we already compare against). Only worth revisiting if the table's ceiling is hit and a
  architecture renegotiation with Dylan is on the table anyway. Saying it loudly, as instructed.
- **PRF (idea 1) is half-trap**: it is the best gain/cost ratio on the list *and* a release-story
  change. Do not let a good dev number launder the two-search architecture in quietly — the
  protocol note and the labeling decision come first.

## What I would do first and why

**This afternoon (eval-only, ~1.5 GPU-h total, both pre-registered in one LEDGER entry):**
(1) the **idea-2 falsifier probe** — teacher-agreement on long vs short held-out spans; it costs
minutes and decides two ideas (2 and 8) before any training money; (2) **Vector-PRF grid**
(idea 1) — highest probability-per-GPU-hour on the list and zero interaction with the training
pipeline; (3) **soup** (idea 6) if checkpoints exist on disk. None of these touch the candidate
chain, so they can run while the review-#3 items 1–4 (which STATUS correctly puts before any
GPU-heavy spend) are being done.

**First overnight chain: idea 3 (ICT A-pairs).** It scales the exact phase that has produced
every real gain, uses only artifacts already on disk, has the strongest literature behind it, and
its failure mode (lexically-trivial pairs) is informative — it would tell us the A-phase gains
are saturated by B-adjacent signal, which reprices ideas 5 and 7 downward immediately.

**Second overnight: idea 2 (long spans) if its probe showed a gap**, because it is the only lever
pointed at a named confirmatory-set weakness (ArguAna) rather than at the dev macro, and the
final claim is the six. Ideas 4/5/7 queue behind, each gated on the cheap read that reprices it
(lever-#4 result for 4; ICT outcome for 5 and 7).

One meta-note, per the standing directive: nothing above is a projected tier-changer on its own —
the honest expected values are +0.003–0.010 each with maybe 40% total probability that at least
two land. The compounding path to the aim bar is (adopted lever #2) + one data lever + fusion,
and the cheap probes above are how to find the data lever without burning the ablation budget
STATUS has already committed.

## Sources

- LightRetriever: https://arxiv.org/abs/2505.12260 (ablations: raw-embedding table −11.2 BEIR;
  instructions +1.6; aux KL ±0.5; ~95% retention)
- OpenSearch inference-free v3: https://www.alphaxiv.org/abs/2411.04403v2 (BEIR-13 50.35, +3.3
  over prior inference-free SOTA; query side = binary presence × IDF)
- Vector PRF: https://arxiv.org/abs/2108.11044 (training-free Rocchio/Average; MAP/recall gains,
  shallow metrics inconsistent)
- Query Encoder Distillation via Embedding Alignment: https://arxiv.org/abs/2306.11550 (2-layer
  student keeps 92.5% of full DE on BEIR — frozen-doc query distillation headroom)
- ICT: https://arxiv.org/abs/1906.00300 · Contriever: https://arxiv.org/abs/2112.09118 ·
  AugTriever: https://arxiv.org/pdf/2212.08841
- Margin-MSE: https://arxiv.org/abs/2010.02666
- DRAGON: https://arxiv.org/abs/2302.07452 (BERT-base BEIR 47.4, progressive supervision)
- Complexity-diversity for training queries: https://arxiv.org/pdf/2602.09448
- SPARTA: https://arxiv.org/abs/2009.13013
- Model soup for embeddings: https://jina.ai/news/model-soups-recipe-for-embeddings/
