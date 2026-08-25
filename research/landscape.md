# Landscape: alternatives to a trained asymmetric dual encoder for edge retrieval

Scope: what an edge-retrieval system could use *instead* of training a custom asymmetric pair (big cloud model on documents, near-zero-compute lookup+average on queries). Four buckets: small conventional bi-encoders, static/zero-inference encoders, off-the-shelf asymmetric-compatible pairs, and large-model reference ceilings.

**Scale warning.** Retrieval scores below come from at least four different, non-interchangeable scales:
- **MTEB v1 / BEIR-en (15 datasets, nDCG@10)** — the classic scale, used by bge-small, e5-small, gte-small, arctic-embed, LEAF, e5-mistral. This is the scale to compare against.
- **MTEB v2 (English)** — a reworked, larger task suite; scores are numerically different (mid-60s is "average," not mid-50s). Used by mdbr-leaf-mt's headline claim.
- **NanoBEIR** — small-sample ("Nano") versions of BEIR datasets (~50 queries each), used by sentence-transformers' static model cards. Not comparable in absolute terms to full BEIR.
- **MMTEB (multilingual)** — used for potion-multilingual-128M; a different task/language mix again.

Every number below is tagged with its scale. Do not average across scales.

All numbers verified against primary sources (HF model cards / raw READMEs / model-index YAML eval blocks, the Model2Vec results table, and the LEAF/LightRetriever/BGE-M3 papers) on 2026-08-24, via direct fetch, not the MTEB Space UI (which wasn't scraped). Where a number could only be found via a secondary source or search-engine summary, it is marked UNVERIFIED.

---

## 1. Small conventional bi-encoders (10–120M params)

MTEB v1 / BEIR-en nDCG@10, average over the 15 English retrieval datasets, plus the 5 requested per-dataset scores. All per-dataset numbers were recomputed directly from each model's own `model-index` eval metadata (or, for all-MiniLM-L6-v2, from the MTEB `results` GitHub repo, since its HF card carries no eval YAML) — not copied from a secondary table — so the row averages are self-consistent.

| Model | HF repo | Params | Dim | License | Disk (fp32 safetensors) | BEIR-en avg (15) | ArguAna | FiQA-2018 | NFCorpus | SCIDOCS | SciFact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` | 22.7M | 384 | Apache-2.0 | 90.9 MB | **41.95** | 50.17 | 36.87 | 31.59 | 21.64 | 64.51 |
| E5-small-v2 | `intfloat/e5-small-v2` | 33.4M | 384 | MIT | 133.5 MB | **49.04** | 41.67 | 37.43 | 32.45 | 17.77 | 68.85 |
| Arctic-embed-xs | `Snowflake/snowflake-arctic-embed-xs` | 22M | 384 | Apache-2.0 | 90.3 MB | **50.15** | 52.08 | 34.52 | 30.89 | 18.36 | 64.51 |
| GTE-small | `thenlper/gte-small` | 33.4M | 384 | MIT | 66.8 MB (fp16 on hub) | **49.46** | 55.44 | 39.35 | 34.77 | 21.38 | 72.70 |
| BGE-small-en-v1.5 | `BAAI/bge-small-en-v1.5` | 33.4M | 384 | MIT | 133.5 MB | **51.68** | 59.55 | 40.34 | 34.31 | 20.52 | 71.28 |
| Arctic-embed-s | `Snowflake/snowflake-arctic-embed-s` | 33M | 384 | Apache-2.0 | 132.9 MB | **51.98** | 56.87 | 39.68 | 32.54 | 19.42 | 69.92 |
| Granite-embedding-small-english-r2 | `ibm-granite/granite-embedding-small-english-r2` | 47M | 384 | Apache-2.0 | 95.3 MB | **50.9** (card's own BEIR-15 figure) | UNVERIFIED (not broken out) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| **mdbr-leaf-ir** (see §3, listed here too as it's the current best in class) | `MongoDB/mdbr-leaf-ir` | 23M | 384 (MRL to 256) | Apache-2.0 | 90.3 MB | **53.55** | 58.4 | 38.4 | 35.8 | 19.7 | 70.0 |

Notes:
- Two BEIR-en numbers exist for MiniLM-L6-v2 in the wild: 41.95 (MTEB results repo, revision `8b3219a`, what LEAF's card also cites) vs. 42.92 (cited on Model2Vec cards, likely a newer MTEB point-release run). Both are real, ~1 point apart; treat as noise, not a real model difference.
- **Best-in-class at ≤100M as of 2026-08-24 is mdbr-leaf-ir** (23M, 53.55 BEIR-en symmetric / 54.03 asymmetric — see §3), per its own model card's claim of #1 on the public BEIR leaderboard for ≤100M models. Among models with no teacher-alignment trick, **BGE-small-en-v1.5 (51.68) and Arctic-embed-s (51.98)** remain the strongest plain small bi-encoders; **Granite-embedding-small-english-r2 (47M, 50.9)** is the newest entrant of that kind, released Aug 2025, replacing IBM's older 30M model (BEIR 49.1) and adding 8192-token context.
- IBM Granite's own r2 paper reports MTEB-v2(41) = 61.1 for the small model, vs. 63.22 (gte-small) and 61.32 (e5-small-v2) on the same v2 scale per MongoDB's table — consistent cross-check between two independent primary sources.
- One caveat on the AI-search-summarized claim "granite-embedding-97m-multilingual-r2 scores 60.3, best <100M" that came up during research: this could not be verified from a primary source and contradicted itself in the same answer (60.3 vs 60.5); treat as UNVERIFIED/likely hallucinated. It's also a multilingual model, not the "small English retrieval" class asked about.
- e5-mistral-, jina-v3/v4/v5-scale, and other >120M small-ish models are out of scope for this table (see §4 / §3 notes) even where cheap; jina-embeddings-v5-text-nano (239M, BEIR 56.06 per its own model page) is representative of the size class just above this bucket if the ceiling needs relaxing.

---

## 2. Static / zero-inference embedding models

No transformer forward pass: a per-token lookup table (optionally averaged/weighted), the same compute class as a lookup+average query encoder. Scores below are the **Model2Vec MTEB "Ret" column** (its own eval harness, English MTEB v1-based retrieval subset — the model cards call this "MTEB Retrieval Score") except where noted as NanoBEIR.

| Model | HF repo | Params | Dim | License | Disk | MTEB Retrieval (or NanoBEIR) | Notes |
|---|---|---|---|---|---|---|---|
| potion-base-2M | `minishlab/potion-base-2M` | 1.89M | 256 | MIT | 7.6 MB | 22.99 | distilled from bge-base-en-v1.5, general-purpose (not retrieval-tuned) |
| potion-base-4M | `minishlab/potion-base-4M` | 3.78M | 256 | MIT | 15.1 MB | 28.43 | ditto |
| potion-base-8M | `minishlab/potion-base-8M` | 7.56M | 256 | MIT | 30.2 MB | 31.11 | ditto |
| potion-base-32M | `minishlab/potion-base-32M` | ~32M | 512 | MIT | — | 32.67 | ditto |
| **potion-retrieval-32M** | `minishlab/potion-retrieval-32M` | 32.3M | 512 | MIT | 129.2 MB | **35.06** | fine-tuned from potion-base-32M specifically for retrieval; best static retrieval model per Model2Vec's own comparison |
| static-retrieval-mrl-en-v1 | `sentence-transformers/static-retrieval-mrl-en-v1` | ~31.3M (30,522-vocab × 1024-dim table) | 1024 (MRL-truncatable) | Apache-2.0 | 125.0 MB | 34.95 (MTEB Ret) / **0.5031 NanoBEIR nDCG@10** | NanoBEIR figure is on a different, smaller-sample scale — don't compare it to the MTEB-Ret column |
| potion-multilingual-128M | `minishlab/potion-multilingual-128M` | ~101M-effective (vocab-table-bound) | 256 | MIT | 512.4 MB | 37.86 (Retrieval, **MMTEB** scale, 101 languages) | not comparable to the English-only rows above |
| GloVe 300d avg (baseline) | `sentence-transformers/average_word_embeddings_glove.6B.300d` | — | 300 | — | — | 21.80 | reference floor, pre-neural |

Notes:
- All Model2Vec (`potion-*`) numbers come from the same primary table, verified by pulling the raw README twice (from `potion-base-4M` and `potion-retrieval-32M`) and cross-checking the shared `all-MiniLM-L6-v2` baseline row (42.92 both times) — internally consistent.
- **potion-retrieval-32M is the strongest static model for retrieval specifically**: reaches 81.7% of all-MiniLM-L6-v2's retrieval score (35.06 vs. 42.92) at zero inference compute, and edges out static-retrieval-mrl-en-v1 (34.95) head-to-head on the same MTEB-Ret metric.
- Model2Vec's own README states models are distilled by mean-pooling a real sentence-transformer's outputs over a corpus, then PCA + SIF-reweighting the resulting per-token vectors — i.e., the same "lookup + average" family as this project's proposed query path, but used symmetrically (same static model for queries and docs).
- No newer static-model release beyond these was found as of 2026-08-24; Model2Vec's own GitHub results page (fetched) lists no additional English retrieval-tuned model past potion-retrieval-32M.

---

## 3. Aligned asymmetric pairs requiring no training by us

### MongoDB LEAF — confirmed real, confirmed numbers

`MongoDB/mdbr-leaf-ir` (retrieval) and `MongoDB/mdbr-leaf-mt` (multi-task) are 23M-param students **distilled to be representation-aligned with their teachers**, explicitly enabling mixed-checkpoint inference: encode documents with the big teacher at index time, encode queries with the 23M student at request time. This is packaged two ways — call the two checkpoints yourself, or load the ready-made `-asym` variant (`MongoDB/mdbr-leaf-ir-asym`, `MongoDB/mdbr-leaf-mt-asym`) that already pairs them.

| Model | Teacher | Task | Params | Dim | License | BEIR-en avg (symmetric) | BEIR-en avg (asymmetric: teacher docs + leaf queries) |
|---|---|---|---|---|---|---|---|
| mdbr-leaf-ir | `Snowflake/snowflake-arctic-embed-m-v1.5` (109M) | IR/retrieval | 23M | 384 (MRL to 256) | Apache-2.0 | **53.55** | **54.03** |
| mdbr-leaf-mt | `mixedbread-ai/mxbai-embed-large-v1` | classification/clustering/STS/summarization | 23M | 1024 (MRL to 256) | Apache-2.0 | 63.97 (**MTEB v2 English**, not BEIR) | not published in BEIR-en terms |

Confirmed directly from the `mdbr-leaf-ir` and `mdbr-leaf-ir-asym` model cards (raw READMEs) and the LEAF paper (ACL 2026, arXiv:2509.12539, Vujanic & Rückstiess, MongoDB Research): "documents are encoded with the larger teacher model, while queries can be encoded faster and more efficiently with the compact leaf model... Retrieval results in asymmetric mode are often superior to the standard mode."

Full per-dataset BEIR-en breakdown, pulled from Table 1 of the LEAF paper (nDCG@10; note the paper's own numbers are 53.9/54.8, ~0.15–0.25 pts off the HF card's 53.55/54.03 — same eval, likely a later revision/rerun on the card):

| Model | ArguAna | ClimateFEVER | CQADupstack | DBPedia | FEVER | FiQA2018 | HotpotQA | NFCorpus | NQ | Quora | SCIDOCS | SciFact | TREC-COVID | Touché2020 | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| leaf-ir (asym.) | 59.0 | 37.5 | 42.4 | 45.0 | 86.5 | 41.3 | 68.5 | 36.2 | 61.2 | 86.0 | 20.3 | 70.2 | 82.6 | 30.1 | 54.8 |
| leaf-ir (symmetric) | 58.4 | 34.6 | 42.3 | 44.6 | 86.6 | 38.4 | 68.1 | 35.8 | 58.9 | 86.3 | 19.7 | 70.0 | 80.3 | 30.2 | 53.9 |
| arctic-embed-xs (23M, no alignment trick) | 52.1 | 29.9 | 40.1 | 40.2 | 83.4 | 34.5 | 65.3 | 30.9 | 54.8 | 86.6 | 18.4 | 64.5 | 79.4 | 32.8 | 50.9 |
| MiniLM-L6-v2 (23M) | 50.2 | 20.3 | 41.3 | 32.3 | 51.9 | 36.9 | 46.5 | 31.6 | 43.9 | 87.6 | 21.6 | 64.5 | 47.2 | 16.9 | 42.3 |
| Teacher: arctic-embed-m-v1.5 (109M) | 59.5 | 36.9 | 45.0 | 45.6 | 88.4 | 42.4 | 72.2 | 36.2 | 62.5 | 87.4 | 21.5 | 71.6 | 84.6 | 31.4 | 56.1 |

**This is exactly the asymmetric-inference pattern the project is scoping** — a 23M student aligned to a 109M teacher, with the asymmetric configuration beating both the symmetric small model and (on average) coming within 2.3 points of the teacher running full-size on both sides. The remaining gap to the teacher is concentrated in FiQA-2018 (−4 pts) and NQ/HotpotQA, less so in ArguAna/FEVER/SciFact.

### Prior art on query-only distillation (predates and likely informs LEAF)

- **EmbedDistill** (Google, arXiv:2301.12005, 2023): "novel asymmetric architectures for student models which realize better embedding alignment without increasing online inference cost," reporting 1/10-size asymmetric students retaining 95–97% of teacher performance on MSMARCO. The name is close enough to EmbeddingGemma's stated "Geometric Embedding Distillation" training objective that it's plausibly the same lineage, though EmbeddingGemma's card doesn't cite it directly (see below).
- **Query Encoder Distillation via Embedding Alignment** (arXiv:2306.11550, 2023): shows a 2-layer BERT query encoder retains 92.5% of full dual-encoder BEIR performance via unsupervised distillation, framed explicitly as "a trivially simple recipe" baseline for asymmetric efficiency — same goal as this project, much smaller student than LEAF.
- **LightRetriever** (arXiv:2505.12260, 2025): the closest analog in spirit to this project's target architecture. Query encoding is reduced to "no more than an embedding lookup" — cache the LLM's per-token embeddings offline, then average the query's token embeddings at request time, no forward pass at all. Document side stays a full LLM encoder (trained jointly, symmetric-then-decoupled). Results (BEIR, 15 datasets, nDCG@10): Full-Llama3.1-8b symmetric = 56.8; LightRetriever-Llama3.1-8b (lookup-only queries) = 54.4 (−2.4, keeps ~96%); a middle-ground baseline using only the LLM's first transformer layer for queries scores 50.1, actually worse than pure lookup despite costing real inference. LightRetriever-Llama3.1-8b (54.4) already beats BGE-M3 dense+sparse (49.6) and approaches E5-Mistral-7b (56.9) and LLM2Vec-Llama8b (56.6). This is direct evidence that lookup+average queries can be competitive against a real transformer query encoder when the document side is strong and training is joint — worth reading in full before finalizing this project's approach.
- **EmbeddingGemma** (`google/embeddinggemma-300m`, Google, gated repo): 300M, trained with "a hybrid objective of Noise-Contrastive Estimation and Geometric Embedding Distillation from larger teacher models," explicitly built with "the same research and technology used to create Gemini models." However: the model card does **not** state that EmbeddingGemma is representation-aligned for mixed-checkpoint use with the Gemini Embedding API (no asymmetric-mode instructions like LEAF's), and at 300M it's outside the 10–120M small-model range and outside the near-zero-compute range this project targets. Treat the "relationship to Gemini embeddings" as shared training lineage/technique, not a confirmed asymmetric-inference pair — UNVERIFIED as an asymmetric option.
- **Jina embeddings v5** (`jina-embeddings-v5-text-nano`, 239M; `-small`, 0.6B): distilled from Qwen3-Embedding-4B, and the family does use asymmetric query/document prefixing ("Query:" / "Document:") — but that's standard asymmetric *prompting* within one checkpoint, not confirmed mix-and-match between the nano/small student and a larger Jina checkpoint's document embeddings. UNVERIFIED as a teacher-swap pair. jina-embeddings-v5-text-nano's own BEIR score is 56.06, but at 239M it's well above this project's target compute budget and off the "no training by us" small-student premise.

---

## 4. Reference ceiling: strong large models

Both computed independently from each model's own `model-index` YAML eval block on its HF card (averaging the same 15 BEIR-en datasets, same method as §1), except BGE-M3 which doesn't publish an English-only BEIR number (see note).

| Model | HF repo | Params | Dim | License | BEIR-en avg (15) | ArguAna | FiQA-2018 | NFCorpus | SCIDOCS | SciFact |
|---|---|---|---|---|---|---|---|---|---|---|
| E5-mistral-7b-instruct | `intfloat/e5-mistral-7b-instruct` | ~7.11B (Mistral-7B backbone) | 4096 | MIT | **56.89** (recomputed from card; matches the widely-cited 56.9) | 61.88 | 56.59 | 38.62 | 16.30 | 76.41 |
| BGE-M3 (dense mode) | `BAAI/bge-m3` | ~568M (XLM-RoBERTa-large backbone) | 1024 | MIT | **49.6** (dense+sparse hybrid, English BEIR-15; UNVERIFIED for dense-only, see note) | — | — | — | — | — |

Note: BGE-M3's own paper (arXiv:2402.03216) reports MIRACL, MKQA, and MLDR — multilingual/cross-lingual/long-document benchmarks — as its headline numbers, and never reports a standalone English BEIR-15 score; the model is positioned as a multilingual/long-context generalist, not an English BEIR specialist. The 49.6 figure above (dense+sparse combined, not pure dense) comes from LightRetriever's independent evaluation (arXiv:2505.12260, Table 2), the only primary source found reporting BGE-M3 on this exact 15-dataset scale. A pure-dense BGE-M3 English BEIR-15 number could not be located from a primary source in this pass — mark as a gap, not a confident number.

---

## Sources fetched directly (primary)

- HF model cards / raw READMEs: BAAI/bge-small-en-v1.5, BAAI/bge-m3, intfloat/e5-small-v2, intfloat/e5-mistral-7b-instruct, sentence-transformers/all-MiniLM-L6-v2, thenlper/gte-small, Snowflake/snowflake-arctic-embed-xs, Snowflake/snowflake-arctic-embed-s, ibm-granite/granite-embedding-small-english-r2, minishlab/potion-base-{2M,4M,8M,32M}, minishlab/potion-retrieval-32M, minishlab/potion-multilingual-128M, sentence-transformers/static-retrieval-mrl-en-v1, MongoDB/mdbr-leaf-ir, MongoDB/mdbr-leaf-mt, MongoDB/mdbr-leaf-ir-asym, google/embeddinggemma-300m, jina-embeddings-v5-text-nano
- HF `api/models/*/tree/main` for exact disk sizes; `embeddings-benchmark/results` GitHub repo for all-MiniLM-L6-v2 per-dataset scores (its own card lacks eval YAML)
- Papers (arXiv HTML, fetched in full): LEAF (2509.12539), LightRetriever (2505.12260), EmbedDistill (2301.12005, abstract), Query Encoder Distillation via Embedding Alignment (2306.11550, abstract), BGE-M3 (2402.03216), Granite Embedding R2 (2508.21085, via model card)
- Model2Vec results page: github.com/MinishLab/model2vec/blob/main/results/README.md
