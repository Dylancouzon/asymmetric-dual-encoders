# M7 research round 2, 2026-08-26 — the angles the plan had not considered

Four parallel Sonnet briefs, commissioned to fight assumptions before any training starts, on the
principle that the expensive-to-redo decisions (teacher, dimension, doc-side transforms) must be
settled *first*. **Every number here is second-hand until we measure it.** Round 1 is
`m7-research-2026-08-26.md`, whose central algebra claim this project has since disproved.

## 1. The finding that saves a corpus re-encode: doc2query expansion is a trap

The plan was going to consider appending generated queries to documents before encoding — a
zero-query-cost lever that attacks vocabulary mismatch directly. **The controlled literature says
it fails, and fails worse the stronger your encoder is.**

- **Weller et al., "When do Generative Query and Document Expansions Fail?"** (EACL 2024 Findings,
  arXiv 2309.08541): 11 expansion techniques × 12 datasets × 24 retrievers. Headline is a **strong
  negative correlation between retriever strength and expansion gain** — expansion helps weak and
  unsupervised retrievers and **harms strong supervised dense ones**, because it adds recall-useful
  terms along with noise that blurs the top of the ranking.
- **PqE** (dl.acm.org/doi/10.1007/978-981-97-9431-7_8) corroborates from another group: +7 nDCG@10
  for *unsupervised* Contriever, dropping to **+1–2 on Recall@1k only** for Contriever-FT and BGE —
  our teacher's quality class.
- **Doc2Query++** (arXiv 2510.09557) does recover gains on frozen Contriever (+0.01 to +0.07 across
  NFCorpus/SciDocs/FiQA/ArguAna/SciFact) — but by keeping a **second index and fusing at scoring
  time**, not by folding expansions into one document vector. That is a different architecture, and
  it reintroduces an index we do not have.

**Consequence: dead as posed.** Do not re-encode a corpus for it. Recorded in EXPLORED.md.

## 2. The genuinely open doc-side lever: a document-side instruction

Nobody has published a doc-side-instruction ablation for the bge / e5 / gte / stella families. The
field convention is "no document prompt", and E5-Mistral (arXiv 2401.00368) states outright *why*
— **"We do not modify the document side with any instruction prefix. In this way, the document
index can be prebuilt"** — i.e. it is an engineering convenience, **not a measured result**.

This matters to us specifically because an instruction changes the doc vectors **non-linearly**, so
unlike a doc-side linear map it is *not* absorbable into our query table: it is genuine new
capacity. And it is nearly free to test — encode the docs with a fixed instruction string and
re-measure. Cheapest untried structural lever we have.

## 3. Multi-vector documents point the wrong way for OUR corpus

- **ME-BERT** (Luan et al., TACL 2021, arXiv 2005.00181) is exactly multi-vector-doc /
  single-vector-query with a max operator. Its own table: at 400-token passages ME-BERT 34.4 vs
  single-vector 32.2, **but at 50-token passages single-vector WINS (44.2 vs 42.0).** Index cost
  ~3× (MS MARCO: 34.2 GB → ~100 GB at m≈3). And it is trained **end-to-end** — no evidence that
  multi-vector heads can be bolted onto a frozen tower post-hoc.
- Our six are titles+abstracts, i.e. the short regime where ME-BERT's own numbers favour a single
  vector. **Chunking is likewise inapplicable** — there is nothing to chunk.

## 4. The n-gram lever is weaker than the plan assumed

STATUS ranked n-gram rows as "the only structurally new lever". The algebra still says that, but
the *evidence* is worse than assumed:

- **Sent2Vec's own ablation** (arXiv 1703.02507) — the founding n-gram-row reference — reports that
  bigrams help **supervised classification** (82.0 vs 81.4) but **"doesn't help much when it comes
  to unsupervised evaluations"** (STS/similarity). Unsupervised cosine similarity is the regime
  retrieval lives in. It reports **no retrieval numbers at all**.
- No modern (BEIR/MTEB-era) retrieval result using word/phrase lookup rows was found. Confirmed
  again that Model2Vec/potion and sentence-transformers statics use pure subword rows, no n-grams.
- Arbitrary bigrams over a 30K vocab is ~10^9 candidate rows before pruning, so the artifact cost
  is real and the phrase-selection problem is unsolved for this setting.

**Consequence: keep n-grams, but demote them below the cheaper non-absorbable levers.** The
original-claim argument survives; the expected-value argument does not.

## 5. Count saturation gains cross-family support — and a 2026 precedent

- **NUMEN** (arXiv 2601.15205, Jan 2026): character 3/4/5-grams hashed into up to 32,768 dims,
  aggregated, then **`v ← log(1+v)`** — explicitly BM25/SPLADE-inspired tf damping — then L2
  normalized. **Zero learned parameters.** On DeepMind's LIMIT (1K queries / 50K docs) it reaches
  **93.90% Recall@100, beating BM25's 93.6%** and crushing 7B dense encoders at matched dimension.
- Caveats that must travel with that number: LIMIT is a *synthetic adversarial* benchmark, not
  BEIR; the setup is **symmetric** (no frozen doc tower); and 32,768-dim fp32 rows are 128 KB per
  vector, the opposite direction from our 23–31 MB budget.
- Still: BM25 (`tf/(tf+k1)`), SPLADE (`log(1+ReLU)`) and now NUMEN independently converge on
  sublinear count damping. No published ablation isolates linear-count vs saturated-count in a
  *dense token-vocabulary* bag on BEIR — so this stays our experiment, now with real indirect
  support rather than just our own algebra.
- One contrast worth keeping honest: LightRetriever's own ablation found that fixing its **sparse**
  arm's per-token value to 1 instead of true term frequency cost only +0.0 BEIR. Different claim
  (count vs no count, sparse arm) from ours (saturated vs linear, dense arm), but it is a data
  point against counts mattering much.

## 6. Order-binding (HRR/VSA) has no retrieval evidence whatsoever

Circular convolution / holographic reduced representations / random-permutation binding would give
order sensitivity at zero learned cost. The best head-to-head (Recchia et al., PMC4405220) is on
**psycholinguistic synonym and similarity tasks** (TOEFL, ESL, Rubenstein-Goodenough) at 6M–418M
tokens, and finds **random permutation scales where circular convolution does not**. There is **no
BEIR-era, or even TREC-era, IR number for binding anywhere.** Plausible mechanism, genuinely
unmeasured; if ever tried, use permutation, not FFT.

## 7. The elephant: two transformer layers is worth 92.5%

**Query Encoder Distillation via Embedding Alignment** (arXiv 2306.11550, ACL SustaiNLP 2023) is
the closest published analogue to our architecture — asymmetric, **frozen document tower**, small
distilled query tower, measured on BEIR nDCG@10:

| query tower | BEIR nDCG@10 retention |
|---|---|
| 2-layer BERT | **92.5%** |
| 4-layer BERT | **96.2%** |
| our lookup table (measured) | **78.5%** |

We need 82–88%. Priced against our calibration, a 92.5%-retention query tower gives 0.470 on the
six with the *current* teacher (clears the 0.4583 release bar) and 0.506 with gte-large (clears the
0.4868 Tier-1 aim **dense-only, before any fusion**). Cost: ~0.1–0.5 ms vs our 0.023 ms.

**This is M9's mandate** (renumbered from M8 on 2026-08-28; now `instructions-m9.md`: a LEAF-style distilled small query tower). So the
literature says M9 will likely clear M7's bars comfortably. That reframes M7 honestly: **M7's
contribution is the zero-transformer point on the cost frontier, not the quality winner.** Worth
saying out loud before we over-invest in closing a gap that the next milestone closes by design.

Also correcting how we cite LightRetriever: its **−11.2 BEIR** embedding-bag ablation compares raw
input-embedding rows against rows forwarded through its *own trained model*. Both pool by plain
mean. So it is evidence that **table content** matters enormously, **not** that bag pooling is
hopeless — we have been citing it slightly wrong.

## 8. The negative result, if we end up reporting one, is now a theorem

**"On the Theoretical Limitations of Embedding-Based Retrieval"** (DeepMind, arXiv 2508.21038)
proves the number of realizable top-k document subsets is bounded by embedding dimension, and
builds LIMIT to exhibit it — where strong 7B encoders score single digits. The SIGIR 2026
reproduction (arXiv 2605.03824) finds even reasoning-targeted retrievers collapse on LIMIT+.

This reframes our multi-hop ceiling from "our bag is weak" to **"single-vector retrieval is
provably limited here, and a bag is the cheapest way to reach that limit."** Strong citation
whichever way the result lands, and it is an argument for the fusion arm on principle, not just
empirically.

## 9. Regularisation: our reg_init has a name, and our row-scaling appears novel

- **L2-SP** (arXiv 1802.01483) is exactly penalising `‖θ − θ₀‖²` toward the pretrained start
  instead of toward zero: **+1.1 to +4.6 top-1** across four CNN transfer benchmarks, better than
  decay-to-zero in every reported cell, **and the gain grows as training data shrinks** — our
  regime. Caveat: CNN vision classification only; no embedding-table or retrieval precedent exists.
- **No paper regularises embedding rows by their observed per-row update count.** Our
  `reg_init × 1/(1+updates)` scaling appears to be genuinely new, if minor.
- SPLADE's FLOPS regulariser is **not** our mechanism: it shapes the deployed index's posting-list
  distribution, not per-row training exposure. A 2025 follow-up (arXiv 2505.15070) exists precisely
  because its frequency-linked side effects are considered a production *flaw*.

## 10. Synthetic query data: useful, but not a substitute

- **Gecko/FRet** (arXiv 2403.20327): **6.6M** synthetic (task, query, positive, negative) tuples,
  two-step generate-then-relabel with RRF ensembling. Synthetic-only → synthetic+real is
  **+2.54 MTEB Retrieval** (53.16 → 55.70). Generating LLM is **not disclosed**.
- **E5-Mistral** (arXiv 2401.00368): **500K** examples / 150K unique instructions / ~180M tokens,
  25% GPT-3.5 + 75% GPT-4. Retrieval is the category most starved by synthetic data alone:
  **46.9 synthetic-only → 52.2 +MS MARCO → 56.9 full.** Cost never reported.
- Both are *contrastive training on generated labels*. **Our case is different and easier**: the
  label is the frozen teacher's own embedding of the query text, so a synthetic query needs to be
  distributionally useful, not correct. That makes the −10-point synthetic-only retrieval penalty
  above a poor guide to our setting — but nobody has published the ablation that isolates query
  *realism* from query *coverage* for distillation, so it stays untested.
