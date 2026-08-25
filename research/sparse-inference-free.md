# Inference-free learned sparse retrieval: survey for edge retrieval scoping

Query-time budget assumed throughout: tokenization only, or tokenization + a fixed lookup
table. No transformer forward pass on the query. All numbers below are BEIR nDCG@10 unless
stated otherwise. "UNVERIFIED" marks anything not confirmed against a primary source (paper,
official model card, or official repo) in this pass.

---

## 1. SPLADE doc-only variants (naver)

All three share the same query-side design: **the query is tokenized and every query term
gets equal weight (or its raw count) — no query model runs at all.** Scoring is
`sum over query terms j of doc_weight(j)`, so the entire ranking signal lives in the
document-side sparse vector, computed once at indexing time. This is confirmed directly in
the SPLADE v2 paper section 3.3 ("the ranking score is simply given by
`s(q,d) = Σ_{j∈q} w_j^d`... everything can be pre-computed offline").

| Model | HF repo | Query-side compute | BEIR avg nDCG@10 | Notes |
|---|---|---|---|---|
| SPLADE-v2-doc | not released as a standalone repo (described in Formal et al. 2021, arXiv:2109.10086) | tokenize, uniform weight | **not reported** in the SPLADE v2 paper's BEIR table (Table 2 only covers SPLADE-sum/max/distil, symmetric variants). MS MARCO MRR@10 = 0.322 is the only public number. | UNVERIFIED for BEIR — treat as no public BEIR number, not as a bad one |
| Efficient SPLADE (naver/efficient-splade-V-large-doc + naver/efficient-splade-V-large-query) | `naver/efficient-splade-V-large-doc`, `naver/efficient-splade-V-large-query` (also VI-BT-large-doc/query) | **Not literally query-free** — the "query" side of this family (called BT-SPLADE in the source paper, Lassance & Clinchant, "An Efficiency Study for SPLADE Models," arXiv:2207.03834) runs a real but tiny transformer, BERT-tiny, at query time. Quote: "use a very efficient PLM on the query encoder, namely BERT-tiny." | BT-SPLADE-S 39.2, BT-SPLADE-M 42.1, BT-SPLADE-L 44.5 (paper's BEIR averages, size variants of the same doc/query-tiny design) | Model cards report MS MARCO only (38.8 MRR@10, 98.0 R@1000); no BEIR number on the HF page itself. **This family does not satisfy "no transformer at query time"** — it minimizes query compute, it doesn't eliminate it |
| SPLADE-v3-Doc | `naver/splade-v3-doc` | tokenize, uniform weight, explicitly "no inference on query side" (HF card) | **47.0** (BEIR-13 avg, per HF model card) | MRR@10 (MS MARCO dev) = 37.8. Base model Luyu/co-condenser-marco, 30,522-dim output. This is the strongest true (zero-transformer) SPLADE doc-only number found |

Bottom line for this family: **SPLADE-v3-doc (47.0 avg) is the one genuinely query-inference-free SPLADE checkpoint with a solid public BEIR number.** SPLADE-v2-doc has no public BEIR score. "Efficient SPLADE" / BT-SPLADE is a different, weaker guarantee (near-zero, not zero, query compute) and shouldn't be filed under the same claim.

---

## 2. uniCOIL, TILDE/TILDEv2, DeepImpact, EPIC, SparTerm

Source for the head-to-head numbers: SPRINT (Kamalloo, Lin, et al., "A Unified Toolkit for
Evaluating and Demystifying Zero-shot Neural Sparse Retrieval," arXiv:2307.10488), which
benchmarks all of uniCOIL, DeepImpact, SPARTA, TILDEv2, SPLADEv2 on the small BEIR-5 slice
plus more. Cross-checked against the Pyserini docs and TILDE GitHub repo for the
inference-free claim (a description elsewhere states plainly: "DeepImpact and TILDEv2 can be
viewed as uniCOIL models without a query encoder").

| Model | Query-side compute | HF/checkpoint | SciFact | NFCorpus | FiQA-2018 | ArguAna | SciDocs | BEIR avg (5-set) |
|---|---|---|---|---|---|---|---|---|
| uniCOIL | **Not inference-free by default** — original design runs a BERT query encoder (`castorini/unicoil-msmarco-passage`). Pyserini also ships a "pre-tokenized queries with pre-computed weights" fast path, but the reported/benchmarked numbers use the real query encoder | `castorini/unicoil-msmarco-passage`, `castorini/unicoil-noexp-msmarco-passage` | 0.686 | 0.333 | 0.289 | 0.387 | 0.144 | 0.428 |
| DeepImpact | **Inference-free.** Query = tokenize, uniform/binary weight; all learned weighting is document-side | No official HF checkpoint found — original release is a Lucene impact index built from a private checkpoint (`public.ukp.informatik.tu-darmstadt.de`); not cleanly redistributable | 0.633 | 0.312 | 0.266 | 0.374 | 0.146 | 0.415 |
| TILDEv2 | **Inference-free.** Query is tokenized into a binary vector; scoring sums precomputed document-side log-probabilities for the intersecting terms — 0.1ms per query, no per-query model call | `ielab/TILDEv2-noExp`, `ielab/TILDEv2-TILDE128/150/200-exp`, `ielab/TILDEv2-docTquery-exp` (all on HF) | 0.647 | 0.318 | 0.266 | 0.476 | 0.146 | 0.425 |
| SPARTA | query encoder required (BEIR checkpoint `BeIR/sparta-msmarco-distilbert-base-v1`), not inference-free | `BeIR/sparta-msmarco-distilbert-base-v1` | 0.598 | 0.301 | 0.198 | 0.279 | 0.126 | 0.340 |
| SPLADEv2 (full, symmetric, for reference) | query encoder required, not inference-free | `naver/distilsplade_max` | 0.693 | 0.334 | 0.336 | 0.478 | 0.158 | 0.470 |
| EPIC (MacAvaney et al., SIGIR 2020) | doc-side importance prediction; original paper design still runs a query encoder | **No public checkpoint found** — research-code only | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| SparTerm (arXiv:2010.00768, the direct ancestor of SPLADE) | importance-predictor + gating controller on both sides; not designed as doc-only | **No public checkpoint found** | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |

Bottom line: of this group, **TILDEv2 and DeepImpact are the two with a genuinely zero-compute query side and released numbers** — TILDEv2 slightly ahead (0.425 vs 0.415 avg) and it has a clean HF checkpoint (DeepImpact's does not). uniCOIL and SPARTA need a real query encoder despite reputation as "sparse and cheap." EPIC and SparTerm are dead ends for this project: no released weights to point at.

---

## 3. doc2query / docT5query family (document expansion + plain BM25)

Query-time compute here is **zero in the strongest sense** — no sparse-vector query
processing at all, just standard BM25 over an inverted index whose documents were expanded
offline with a T5 model.

Source: BEIR paper itself (Thakur et al., 2021, arXiv:2104.08663, Table 2 — verified directly
against the paper text).

| Dataset | BM25 | docT5query (BM25 + T5 doc expansion) |
|---|---|---|
| SciFact | 0.665 | 0.675 |
| NFCorpus | 0.325 | 0.328 |
| FiQA-2018 | 0.236 | 0.291 |
| ArguAna | 0.315 | 0.349 |
| SCIDOCS | 0.158 | 0.162 |
| Overall (18 BEIR datasets) | — | **+1.6% over BM25 on average, wins on 11/18 datasets** (paper's summary claim) |

This is arguably the simplest possible "inference-free at query time" option: BM25 needs no
model at all on either side at query time, and the only neural cost is a one-time doc-side T5
pass at indexing. The gains over plain BM25 are real but modest (biggest lift on FiQA-2018 and
ArguAna, near-zero on SciFact/NFCorpus/SCIDOCS).

---

## 4. BM25 — the canonical zero-compute baseline

Same BEIR paper, Table 2, verified:

| Dataset | BM25 nDCG@10 |
|---|---|
| SciFact | 0.665 |
| NFCorpus | 0.325 |
| FiQA-2018 | 0.236 |
| ArguAna | 0.315 |
| SCIDOCS | 0.158 |

BM25 needs no model on either side, ever. It is the floor every "inference-free" sparse method
in this doc is trying to beat, and on the small-corpus/scientific datasets (SciFact, NFCorpus,
ArguAna) it's already a strong baseline that TILDEv2/DeepImpact only beat by a few points.

---

## 5. Qdrant-native options

**BM25 (Qdrant/Edge):** standard BM25 over an inverted/sparse index. Zero compute on both
sides, the same numbers as row 4. No model artifact to manage at all — this is the
zero-engineering-risk option.

**BM42:** combines BM25 IDF-style term stats with attention weights pulled from a transformer
run over the document (and, per the official article, also over the **query** — "queries
undergo the same attention-based processing as documents"). **BM42 is not query-inference-free**
and should not be listed as a candidate for this project. Worse: Qdrant's own article was
corrected post-publication — "BM42 does not outperform BM25" — and the piece now explicitly
labels BM42 "experimental," not production-ready. Exclude it.

**miniCOIL (`Qdrant/minicoil-v1`, FastEmbed handle `Qdrant/minicoil-v1`):** each vocabulary
word gets a small (4-dim, 8/16-dim variants were tested) learned "meaning vector," distilled
offline from `jina-embeddings-v2-small-en` over a 30,000-word vocabulary. At query time this
is tokenize + stem + **dictionary lookup** into the fixed word→vector table — no transformer
runs. (The Qdrant docs describe inference-free sparse retrievers generally as relying on
"precomputed document representations offline, and simple operations for obtaining query
representations online"; the miniCOIL source article does not spell out the query path
explicitly, so the "no model at query time" claim is inferred from the lookup-table design
plus that general framing — flag this one item as **UNVERIFIED in the strict sense**, though
it's consistent with everything else published about the model.)

Published numbers are thin and don't overlap the requested 5-dataset slice:

| Dataset | BM25 | miniCOIL |
|---|---|---|
| MS MARCO | 0.237 | 0.244 |
| NQ | 0.304 | 0.319 |
| Quora | 0.784 | 0.802 |
| FiQA-2018 | 0.252 | 0.257 |
| HotpotQA | 0.634 | 0.633 (tie) |

No SciFact / NFCorpus / ArguAna / SciDocs numbers are published for miniCOIL — **UNVERIFIED /
not available** for those four. The one overlapping figure (FiQA-2018) shows a small lift over
BM25 (0.257 vs 0.252), smaller than TILDEv2's or SPLADE-v3-doc's margin on the same dataset.

---

## 6. OpenSearch neural-sparse-encoding-doc-v1/v2/v3 (the "inference-free" family)

All of these are explicitly built and marketed as inference-free: the HF model cards state
directly that "queries use a tokenizer and a weight look-up table to generate sparse
vectors," with no neural computation at query time. Two papers underlie the v2/v3 line:
arXiv:2411.04403 ("Towards Competitive Search Relevance for Inference-Free Learned Sparse
Retrievers" — IDF-aware penalty + dense/sparse ensemble distillation, claims +3.3 nDCG@10 over
prior inference-free models) and arXiv:2504.14839 ("Exploring ℓ₀ Sparsification for
Inference-free Sparse Retrievers," SIGIR 2025 — claims new SOTA among inference-free sparse
models, "comparable to leading Siamese sparse retrieval models").

Numbers below are read directly off each model's official HF card (BEIR-13, same 13-dataset
split used by SPLADE-v3-doc):

| Model | HF repo | Params | BEIR avg | SciFact | NFCorpus | FiQA | ArguAna | SCIDOCS |
|---|---|---|---|---|---|---|---|---|
| doc-v1 | `opensearch-project/opensearch-neural-sparse-encoding-doc-v1` | 133M | 0.490 | 0.716 | 0.352 | 0.344 | 0.461 | 0.154 |
| doc-v2-mini | `opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini` | 23M | 0.497 | 0.699 | 0.336 | 0.338 | 0.480 | 0.164 |
| doc-v3-distill | `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill` | 67M | 0.517 | 0.708 | 0.345 | 0.356 | 0.520 | 0.163 |
| doc-v3-gte | `opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte` | 133M | **0.546** | 0.725 | 0.360 | 0.407 | 0.520 | 0.455 |

**doc-v3-gte is the best inference-free sparse number found in this whole survey (0.546 avg),
beating naver/splade-v3-doc's 0.517-comparable-scale number** (note: splade-v3-doc's 47.0 is
also BEIR-13, so the comparison is apples-to-apples-ish) and clearly ahead of TILDEv2/DeepImpact
(0.42 range) and BM25 (BEIR-13 average would land well under 0.45). doc-v3-gte's SCIDOCS score
(0.455) is a genuine outlier — 2-3x every other model in this doc — worth double-checking
against the paper if SCIDOCS specifically matters to your eval, since a jump that large on one
dataset while everything else moves normally is the kind of thing worth a second look before
citing.

**Yes — OpenSearch's doc-v3-gte is the state of the art among genuinely query-inference-free
sparse retrievers, per every source found in this pass.** Nothing else in this document,
released and BEIR-scored, beats 0.546 while keeping the query side to tokenize + lookup.

---

## 7. 2024-2026 hybrid: zero-compute dense + zero-compute sparse at query time

**LightRetriever** (arXiv:2505.12260, "A LLM-based Text/Hybrid Retrieval Architecture with
Extremely Faster Query Inference") is the clearest published match for this exact idea. Its
hybrid mode:

- **Dense side:** an LLM-based encoder is trained normally, but at serving time the query
  vector is built by **averaging cached per-token embeddings looked up from a table** — no
  transformer forward pass on the query. This is the same "token-vector lookup" idea named in
  the brief.
- **Sparse side:** the query's sparse vector is literally token counts (BM25/SPLADE-shaped,
  zero neural cost).
- The two scores are **linearly interpolated** for final ranking.

Reported BEIR (15-dataset) numbers, hybrid vs. full uncompressed baseline:

| Backbone | Full baseline | LightRetriever hybrid | Retention |
|---|---|---|---|
| Mistral-7B | 57.5 | 54.5 | ~94.8% |
| Qwen-7B | 56.6 | 53.8 | ~95.1% |

Paper reports >1000x query-encoding speedup on GPU (20x on CPU-only) for ~5% average nDCG@10
cost. Code promised at `github.com/caskcsg/lightretriever`; **no HuggingFace checkpoint id
found** in the paper text — UNVERIFIED whether trained weights are actually published yet or
code-only.

No other 2024-2026 paper matching this exact "zero-transformer dense + zero-transformer
sparse, combined" description turned up in this pass — flagging that this may be an
incomplete sweep of a fast-moving area rather than a confirmed absence.

---

## What's practical to run locally (Mac, document-side encoding)

All of these are small BERT-scale encoders (23M-133M params) — CPU or Metal/MPS inference on
a Mac is trivial, no GPU cluster needed, main cost is corpus size x pass count at index time:

- **OpenSearch doc-v2-mini** (23M) — cheapest to run, 0.497 avg. Good first pass.
- **OpenSearch doc-v3-distill** (67M) — 0.517 avg, still light.
- **OpenSearch doc-v3-gte** (133M) — 0.546 avg, best quality, still a normal BERT-base-sized
  model, runs fine on a Mac (M-series or even Intel) for a BEIR-scale corpus (thousands to low
  millions of docs).
- **naver/splade-v3-doc** — co-condenser-marco based (BERT-base scale, ~110M), same ballpark,
  0.470 avg.
- **TILDEv2 (`ielab/TILDEv2-noExp`)** — BERT-base scale, cheap, 0.425 avg, and has the
  advantage of a genuinely tiny per-query cost (0.1ms) since document-side inference is a
  one-time vocab-scale pass.
- **miniCOIL** — the lightest of all: a 30k x 4-dim lookup table plus one-time distillation
  from a small sentence encoder; trivially fast to run and index on a Mac, but the quality
  claim is unverified outside 5 non-classic-BEIR datasets.

## Qdrant-sparse-vector compatibility

Every model in section 6 and 2 (SPLADE-family, uniCOIL/TILDEv2/DeepImpact, OpenSearch
doc-v1/v2/v3) emits a standard `{token_id: weight}` sparse vector over a BERT vocabulary —
this is exactly Qdrant's native sparse vector format and indexes directly with no
transformation. miniCOIL's 4-dim-per-token output needs Qdrant's multi-vector-per-token
(named vector / COIL-style) support rather than a plain scalar sparse vector — check the
`qdrant-search-quality` / `qdrant-edge` skill docs before assuming a drop-in fit if you pick
miniCOIL specifically. BM25/BM42/docT5query all reduce to plain BM25 sparse vectors, also
native.
