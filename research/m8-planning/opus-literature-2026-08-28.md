# M8 planning — Opus brainstormer: literature sweep (2026-08-28)

*Verbatim final report of an Opus 5 subagent that read the prior research/ notes first and
web-searched for what is new since ~May 2026 or missed. Its own caveats: several numbers are
LLM-assisted extractions from abstracts/HTML (the Doc2Query++ table, EmbedDistill's ablation and
the SSE numbers should be re-pulled from PDFs before entering any report), and three of the most
useful items are unreplicated single sources.*

---

## 1. Inference-free / zero-query-compute retrieval — what's new

**Nothing overturns the M7 novelty verdict.** No dense token→vector lookup table trained against a
*frozen off-the-shelf* teacher has been published. Confirmed via arXiv sweeps on `inference-free`,
`query encoding`, `query-side`, `cacheable/amortized/precomputed` in cs.IR. No OpenSearch doc-v4
(v3 still current; lineage arXiv 2411.04403, 2504.14839).

| Paper | Date | What | Relevance |
|---|---|---|---|
| **EmbedDistill** [2301.12005] (Google) | 2023 | **Missed by all prior notes.** Student inherits the teacher's **frozen document encoder**, trains only a tiny query encoder. Theorem 3.1 + Prop 3.2 formally argue why freezing the doc tower shrinks the deviation bound. | Closest published theoretical justification for our exact architecture; carries the best ablation in this sweep (§6). |
| V-SPLADE [2605.30917] | 2026-05 | Inference-free multimodal LSR | Off-modality; framing is spreading |
| Multivector reranking [2601.05200] | 2026-01 | Inference-free LSR first stage, 24x speedup | Architecture validation |
| Score-Only Distillation [2607.11465] | 2026-08 | Row-centered PairMSE on score vectors, 0.6B student, ≤50% gap recovery | Objective candidate; authors report mixed external transfer — weak for OOD, our weak spot |
| HARNESS-LM [2605.23572] | 2026-05 | Sponsored-search production, L2 alignment + contrastive refinement, <600M student, 27x lower latency | Industry point on the ScalingNote rung |
| KAHM [2605.02950] | 2026-05/06 | Already in m7-novelty.md as nearest miss | Unchanged |

**Verdict:** novelty claim survives; nothing forces an M8 re-scope.

## 2. Static embeddings — the strongest new signal in the sweep

**(a) Wada et al., "Static Word Embeddings for Sentence Semantic Representation" — arXiv
2506.04624, EMNLP 2025.** Missed by all prior notes; the closest published *methodological* sibling
to M7, and every design choice differs from ours:
- **Row construction:** each word's row = mean of its **contextual** vectors over **100 sampled
  sentences** from a corpus, not a single isolated forward pass. Multi-subword words get their
  pieces averaged into one row.
- **Vocab: 150k whole words** (English), not 30k wordpieces. Dim reduced to 256/512.
- **ABTT PCA at the sentence level**: PCA over 100k sentence embeddings, discard first ⌊d/100⌋
  components.
- **Objective: softmax-KD over in-batch cosine similarities** (τ=0.05) against a frozen GTE-base
  teacher — *listwise* distribution match, not MSE/ridge.
- Numbers: MTEB avg 51.97 at d=256 vs Model2Vec 51.33, SimCSE 48.86; STS avg 79.4. Code:
  github.com/twadada/swe4semantics.
- Caveat: STS/MTEB-general, **not BEIR retrieval**. Treat the *recipe* as the import, not the
  number.

**(b) SSE — Stable Static Embedding** (HF technical report, MIT, 2026). EmbeddingBag → mean pool →
**separable DyT**: `y_k = γ_k · tanh(α_k · x_k + β_k)` per dimension → L2. NanoBEIR-en mean nDCG@10
0.5124 vs own static baseline 0.5068, static-retrieval-mrl-en-v1 0.4957.
- **Skepticism:** single-author blog, unreplicated, NanoBEIR tiny, trained on MS MARCO 45% with
  NanoMSMARCO in the eval. Isolated DyT effect +0.0056; non-separable DyT worse. Do not cite.
- **Why it matters anyway:** a per-dimension nonlinearity after pooling is **not absorbable into
  the table** — the cheapest published construction that escapes the absorbability proof. 3×dim ≈
  3,072 parameters, O(1) query cost.

**Model2Vec/potion:** no 2026 release (latest potion-multilingual-128M, 2025-05). Tokenlearn's
objective confirmed: **MSE between static output and base-model output on real text, trained
through the pooling** — i.e. "pooling trained through at scale", already on our M8 carried-in list,
is what the strongest static family actually does.

**WordLlama**: Llama-derived token codebook, 16MB @ 256-dim; licence-dirty. Skip.
**SwiftEmbed** [2510.24793]: static-lookup serving, 1.12ms p50. Engineering only.

## 3. N-gram / phrase / hash rows

No retrieval paper applies hashed n-grams to dense retrieval. Live work is n-gram memory for LMs;
tricks transfer to the bigram-row lever:
- **Tokenizer-Agnostic Engram** [2607.29065] — polynomial hashing with hash-equivalence for
  byte-equivalent token sequences.
- **Tensorizing Engram** [2606.08347] — n-gram memory as **CP-factorized shared factors**, matching
  performance at a fraction of the parameters. Direct answer to "bigram rows blow up the artifact":
  rank-r CP factorization is O((V+V)·r) instead of O(V²), still pure lookup.
- Multi-hash ID embeddings [2608.27413] — 98% table reduction at production scale (recsys).
- Precedent nobody cites: **Universal Sentence Encoder DAN** (1803.11175) averages **word *and*
  bigram** embeddings then feeds a feedforward DNN. Our two open capacity levers (n-gram rows,
  nonlinear post-pool head) are exactly USE-DAN, published 2018.
- Wada's 150k *word* rows are the low-entropy subset of the n-gram idea — only sequences the
  tokenizer splits get a row. Better-conditioned than arbitrary bigrams, with published success.

## 4. Learned sparse, doc-side, statistics-only query side

2026 was busy on efficiency, not quality:

| Paper | Number | Note |
|---|---|---|
| **Vocabulary Transfer** [2607.00004] | ModernBERT + VT = 52.4 nDCG BEIR, +4.7 | Migrates encoders to a normalized sparse-friendly vocabulary (semantic init + activation-potential calibration). "Generalizes to inference-free architectures." Anonymous repo, unreplicated. Our uncased 30,522 vocab spends the case-normalization half; the *calibration* half is not spent. |
| **Latent Terms** [2605.29384] | matches/beats base single-vector; large gain on LIMIT | **Sparse autoencoders on a frozen retriever** extract a Zipfian BM25-ready vocabulary with no retrieval supervision. Directly applicable: a doc-side, frozen-teacher, zero-query-compute lexical channel derived from **our own teacher** — makes the fusion partner non-arbitrary. No code confirmed. |
| SPLARE [2603.13277] (Naver) | outperforms vocabulary LSR multilingual/OOD | SAE-instead-of-vocabulary LSR |
| SAE for SPLADE [2604.21511] | comparable, better efficiency | ditto |
| DF-FLOPS [2505.15070] | −2.2 MRR in-domain, better on 12/13 cross-domain, ~10x faster | Applied to SPLADE-Doc. Cross-domain *improvement* from penalizing high-DF terms — notable given our OOD weakness. No weights confirmed. |
| ESPLADE [2509.16621] | 32K vs 100K vocab competitive within budget; pretrained init beats random | Argues *initialization*, not size, is the lever |
| LACONIC [2601.01684] | 60.2 nDCG@10 MTEB | Llama LSR — not inference-free, licence-dirty. Comparator only. |

**Vendor/licence read:** the only clean-vendor inference-free sparse weights remain OpenSearch
doc-v3 (excluded: comparator/circularity) and naver/splade-v3-doc (CC BY-NC — fails). **Nothing new
shipped Apache-2.0 from a clean vendor in 2026.** The Latent-Terms/SAE route needs no vendor at all
— build the sparse channel out of the teacher we already ship.

## 5. Theory of token-linear approximability — the cheap thing we never did

- **The canonical instrument exists: Ethayarajh, arXiv 1909.00512 (EMNLP 2019).** Defines
  **self-similarity**, **intra-sentence similarity**, and **maximum explainable variance (MEV)** —
  the fraction of variance in a word's contextual vectors captured by their first PC, i.e.
  *literally the quality ceiling of a single static row for that word*. Headline: <5% of variance
  in contextualized representations is explainable by a static embedding; upper layers markedly
  more context-specific; anisotropy everywhere.
- **Compositionality of embeddings** [2509.19332]: additive signal is strongest in
  deep-but-not-top layers and **declines at the top layer**. If it holds for our teachers, the
  final-layer pooled vector we distill onto is the *least* token-linear representation the teacher
  has — a mechanistic story for the retention cap, and an argument for probing intermediate-layer
  targets on the query side (the doc side is fixed by the index).
- No 2026 paper predicts distillability-into-a-static-table from teacher statistics. **Open
  ground** — and ten measured teacher/table pairs sit in `m7_learnability_report.json`; the
  correlation study costs GPU-minutes.

## 6. Co-training / asymmetric capacity — best ablation in the sweep

**EmbedDistill's NQ-dev cumulative ablation, 11.3M-param query student** (teacher 110.1M BERT-base,
R@5 72.3): direct training 24.8 → +score distillation 44.3 → **+inherit teacher's frozen doc
embeddings 56.1** → **+query embedding matching 61.1** → **+query generation (synthetic queries)
64.3**. MS MARCO: 11.3M student = 95% teacher retention; 67.5M = 96% on BEIR. Stated finding:
*"embedding matching consistently outperforms score-only distillation across all sizes, gains
amplified for smaller students."* No student is smaller than a table — strongest prior available.

Others: CARE [2604.10937] (align-frozen then unfreeze-both); DevRev-Search [2601.04646] —
"index-preserving query-only adaptation", same structural constraint as ours; DistilVDR
[2608.10636] — capacity on the doc side, 70M query side; ADE-SPL [2204.07120] — sharing the
**final projection** is what rescues asymmetric dual encoders. **Nothing found trains the doc tower
to compensate for a fixed weak query encoder** — that structural inversion is unpublished.

## 7. Tiny-compute middle ground

- ScalingNote BERT-1L/8M: 49.56 R@50 vs 7B tower 55.15 (known).
- HeceTokenizer [2604.10665]: 1.5M-param BERT-tiny with syllable tokenization beats a 200x larger
  model — Turkish-specific, but an existence proof that tokenization-as-inductive-bias buys more
  than parameters at that scale.
- MAM-AI [2606.29580]: EmbeddingGemma-300M on-device; claims "on-device retrieval is essentially
  solved". The framing our report competes with; address head-on.
- **Algebraic note:** a single post-pooling matmul is absorbable — not a middle ground. The genuine
  sub-1M-param middle grounds are exactly three: nonlinear post-pool head, n-gram rows,
  multiplicity-dependent pooling — the same three `m7_absorb_check.json` identified. The literature
  adds no fourth.

## 8. Hybrid fusion

**We are already at the published optimum.** Bruch, Gai, Ingber [2210.11934 / TOIS 2023]: convex
combination beats RRF in- and out-of-domain; normalization choice provably minor; CC is
sample-efficient with one parameter. Exactly our `convex0 w=0.8`.

2026 additions, none better for a zero-query-compute system: DESA [2608.15851] (query-side ⇒ costs
us compute; doc-side half may be extractable); EAHR [2608.07152] (efficiency); SSCC [2606.28367]
(per-source calibrated thresholds; also: beyond a strong reranker, fusion tricks give "no reliable
gain").

**Cautionary negative, directly on our headline:** [2608.02112] — Dutch hybrid retrieval,
systematic simplex weight search, 10-fold CV: all 50 selections put **zero weight on the static
embedding**; BM25+Qwen dominates. Their static is symmetric (both sides), not our system — but
reviewers will reach for it; state our fusion claim against it explicitly.

## 9. doc2query with a commercially clean generator

**The licensing answer is clean and it is not the model weights, it's the fine-tuning data.**
docTTTTTquery / doc2query-- / macavaney/doc2query-t5-base-msmarco are T5 (Apache-2.0 weights)
**fine-tuned on MS MARCO** — derived weights inherit the non-commercial restriction under exactly
the reasoning that excluded MS MARCO from our stack. No Apache/MIT expansion model trained without
MS MARCO was found. **The route that works is prompting, not fine-tuning:** a general instruct LLM
under Apache-2.0 (Qwen2.5/Qwen3-Instruct, Mistral-7B-v0.3, Apertus) generates queries zero/few-shot
— no MS MARCO anywhere; the generated text is ours.

The 2026 recipe to import is **Doc2Query++** [2510.09557] — topic modeling → LLM topic labeling →
keyword selection → prompted LLaMA-3.1-8B-Instruct (not fine-tuned), **30 queries/document**. On
our own datasets (BM25 → doc2query → Doc2Query++): NFCorpus 0.3223 → 0.3338 → **0.3415**; SCIDOCS
0.1430 → 0.1495 → **0.1568**; FiQA 0.2466 → 0.2786 → **0.2972**; ArguAna 0.3437 → 0.3475 →
**0.3561**; SciFact 0.6776 → 0.6927 → **0.6945** (Contriever columns rise similarly).

Two things the M7 probe did not test: **N=30, not 5**, and **Dual-Index Fusion** — generated
queries embedded into a *separate* index, scored `S = (1−α)S_t + αS_q` (α=0.5); on FiQA the
dual-index form beat naive appending by 11.1% relative. Same idea independently from **HyPE**
[2607.29402]: hypothetical questions at index time, question-question matching, no added query
latency (RAG metrics, not BEIR-comparable — cite Doc2Query++ for quality, HyPE for the
architecture).

**Why this is the highest-leverage doc-side lever for us specifically:** our doc index holds
teacher vectors of *documents*, while our table is trained to land where teacher *query* vectors
live. A question-vector index puts the doc side into the query distribution — closing the asymmetry
from the side we are allowed to move, with zero query compute, inside the frozen-teacher premise.

---

# Ranked shortlist: 5 most actionable imports

1. **Dual-index question expansion (Doc2Query++/HyPE), clean prompted generator.** High EV (+0.02
   to +0.05 avg-6 plausible; the only lever with published gains on five of our six datasets).
   Distinct from the parked M7 probe on dose (30 vs 5) and construction (separate question index +
   convex fusion, not text append). Zero query compute; generator = Apache-2.0 instruct LLM. Probe:
   two dev CQADupStack components, 10 questions/doc with Qwen3-Instruct, α ∈ {0.25, 0.5, 0.75},
   pre-registered.
2. **Synthetic-query augmentation of the distillation set (EmbedDistill's "+query generation").**
   Medium-high, compounds with #1. Measured +3.1 R@5 on an 11.3M student, gains amplify as the
   student shrinks. Sidesteps the data-licensing bind. Probe: closed-form ridge on real vs
   real+200k generated queries; read macro and OOD subset separately.
3. **Non-absorbable post-pool head (separable DyT, then 2-layer MLP if DyT resolves).** Low-medium
   (+0.005 order) but nearly free and one of only three constructions the algebra leaves open.
   USE-DAN is the decade-old precedent. Probe: freeze the shipped table, train only (α,β,γ) per
   dimension on the phase-A objective; falsifier = clear the recipe-perturbation band.
4. **Word-level rows for multi-wordpiece words (Wada's 150k-word vocab; CP-factorized if it
   grows).** Medium; the correctly-conditioned retry of the bigram close (which was
   closed-form-only). Import Wada's other two choices, never tested here: **context-averaged row
   initialization** (100 sentences/token vs our single isolated forward) and **softmax-KD over
   in-batch similarities** instead of pointwise ridge. Probe: rows for the top-20k split words
   (+20 MB int8), retrain phase A jointly, dev only.
5. **Teacher approximability metric (Ethayarajh self-similarity / MEV).** Low direct quality, high
   option value; closes a named under-diagnosed item. Probe: over 100 contexts/token, compute
   final-layer self-similarity and MEV for the ten already-scored teachers; Spearman against
   measured table nDCG; |ρ| < 0.5 is a negative result to write down.

**Runner-up worth one line to Dylan:** Qwen3-Embedding-0.6B (Apache-2.0, Alibaba OK-with-
justification) is normally disqualified by its 151k vocab, but it is **MRL-truncatable**: at dim
256 the table is 151k × 256 int8 ≈ 38 MB. The vocab-size rule as written rejects it for the wrong
reason (fp16 arithmetic).
