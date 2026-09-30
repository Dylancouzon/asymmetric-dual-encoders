# Sonnet research agent (literature and published reference numbers)

Date 2026-09-30. Access: web only; forbidden from reserved-dataset qrels searches. Returned text below, verbatim. Dispositions are in `m15/LOG.md` (2026-09-30) and `m15/PLAN.md`.

---

## Report: published references for the BEIR-15 comparison (checked 2026-09-30)

Numbers below came from the web only: raw JSON in the MTEB results repo, model cards, arXiv, Qdrant docs. I did not touch the four protected datasets' qrels or labels.

### 1. Published per-dataset nDCG@10

The Stella and bge-small "card" columns come from each model card's `model-index` metadata, [Stella](https://huggingface.co/NovaSearch/stella_en_400M_v5) and [bge-small](https://huggingface.co/BAAI/bge-small-en-v1.5). The leaf-ir column comes from [embeddings-benchmark/results, MongoDB__mdbr-leaf-ir](https://github.com/embeddings-benchmark/results/tree/main/results/MongoDB__mdbr-leaf-ir/2e46f5aac796e621d51f678c306a66ede4712ecb). CQADupstack is the 12-forum mean.

| Dataset | Ours (Stella) | Stella card | Delta | bge-small card | leaf-ir |
|---|---|---|---|---|---|
| scifact | .7796 | .7823 | -.003 | .7128 | .7056 |
| nfcorpus | .4134 | .4149 | -.002 | .3431 | .3653 |
| scidocs | .2395 | .2504 | **-.011** | .2052 | .1977 |
| trec-covid | .8234 | .8521 | **-.029** | .7590 | .8197 |
| fiqa | .5536 | .5606 | -.007 | .4034 | .3917 |
| arguana | .6369 | .6424 | -.006 | .5955 | .5938 |
| fever | .8207 | .9099 | **-.089** | .8664 | .8660 |
| dbpedia | .4603 | .4988 | **-.039** | .4003 | .4536 |
| cqadupstack | .4448 | .4436 | +.001 | .3905 | .4254 |
| msmarco (dev) | .4373 | .4369 | +.000 | .4083 | .4057 |
| nq | .6371 | .6907 | **-.054** | .5018 | .5956 |
| hotpotqa | .6950 | .7174 | **-.022** | .6994 | .6935 |
| touche2020 | .2980 | .3146 | **-.017** | .2604 | .3081 |
| quora | .8912 | .8958 | -.005 | .8878 | .8617 |
| climate-fever | .2907 | .4353 | **-.145** | .3184 | .3485 |
| 15-set average | .5614 | .5897 | -.028 | .5168 | .5355 |

Delta is ours minus the Stella card. Bold marks a gap above 0.01.

Averages each source gives:
- **Stella:** 58.97 from the card metadata (I computed the mean of its 15 values).
- **bge-small:** the card's own table says [Retrieval (15) = 51.68](https://huggingface.co/BAAI/bge-small-en-v1.5). This matches our .5171 to within .0003.
- **leaf-ir:** the [mdbr-leaf-ir card](https://huggingface.co/MongoDB/mdbr-leaf-ir) says 53.55. The mean of the 15 MTEB-repo values is 53.55.
- **leaf-ir-asym:** the [card](https://huggingface.co/MongoDB/mdbr-leaf-ir-asym) says 54.03, which matches our .5402. The card gives no per-dataset numbers and does not state the dataset count. The 53.55 match suggests it averages 15 datasets.
- **LEAF paper:** [Table 1](https://arxiv.org/html/2509.12539) covers 14 datasets and leaves out MSMARCO. It gives 53.9 for leaf-ir and 54.8 for asym, with arctic-embed-m-v1.5 documents. Do not mix these with the card's 15-dataset figures.
- **mdbr-leaf-ir-asym** has no folder in the MTEB results repo. Only leaf-ir and leaf-mt do.

**Documented reasons for the Stella gaps.**
- **Prompts.** The model creator states the card numbers used "e5-mistral prompts", meaning per-task instructions, plus max_len=400 and bf16 inference. These settings were missing from the README. Source: [discussion 21](https://huggingface.co/NovaSearch/stella_en_400M_v5/discussions/21).
- **Instruction text.** The e5 instructions are in [unilm/e5/utils.py](https://github.com/microsoft/unilm/blob/master/e5/utils.py). They differ from the single `s2p_query` prompt for most of the flagged sets:
  - FEVER: "Given a claim, retrieve documents that support or refute the claim".
  - ClimateFEVER: "Given a claim about climate change, retrieve documents that support or refute the claim".
  - NQ: "Given a question, retrieve Wikipedia passages that answer the question".
  - DBPedia: "Given a query, retrieve relevant entity descriptions from DBPedia".
- **MSMARCO fits this explanation.** Its e5 instruction is word-for-word the `s2p` text, and our MSMARCO delta is +.000.
- **Quora and CQADupstack** use different e5 instructions, yet our gaps there are small (-.005 and +.001). The s2s-prompt idea for these two is not documented anywhere I found.
- **Length and dimension.** The card recommends 512 tokens and says 1024d is 0.001 below 8192d on the MTEB score. Neither explains gaps of 0.05 to 0.14.
- **The published numbers vary by protocol.** The MTEB results repo has a separate run of the same revision, through mteb's `InstructSentenceTransformerModel` ([stella_models.py](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/model_implementations/stella_models.py)). It gives TREC-COVID .766, ArguAna .609, SCIDOCS .232, and CQADupstack .415. Its FEVER, ClimateFEVER, and HotpotQA files are HardNegatives subsets, so they are not comparable. Sources: [results repo, Stella folder](https://github.com/embeddings-benchmark/results/tree/main/results/NovaSearch__stella_en_400M_v5).
- **Contamination.** I did not verify a contamination explanation for the ClimateFEVER and FEVER gaps.

**BM25 references.** Our .4006 is not a match to any single published figure.
- [MTEB baseline-bm25s](https://github.com/embeddings-benchmark/results/tree/main/results/mteb__baseline-bm25s/0_1_10): 39.84 over 15 datasets, with MSMARCO dev at .2189.
- LEAF cards ([asym card](https://huggingface.co/MongoDB/mdbr-leaf-ir-asym)): BM25 41.14, dataset count not stated.
- LEAF paper: BM25 with k1=0.9 and b=0.4, 42.5 over 14 datasets.
- [Pyserini BEIR page](https://castorini.github.io/pyserini/2cr/beir.html): multifield averages 43.7 and flat averages 42.7 over the same 14 datasets, with CQADupstack as a mean (.299 multifield, .302 flat). That page has no MS MARCO row.
- BEIR paper BM25 for MS MARCO: not found. I could not extract Table 2.

### 2. Is this the MTEB English retrieval set?

Confirmed for MTEB(eng, v1). The [benchmark definition](https://github.com/embeddings-benchmark/mteb/blob/main/mteb/benchmarks/benchmarks/benchmarks.py) lists 14 retrieval tasks plus `MSMARCO` on the `dev` split, which makes 15. The [BGE docs](https://bge-model.com/tutorial/4_Evaluation/4.2.1.html) say "MTEB directly use the open source benchmark BEIR in its retrieval part, which contains 15 datasets". The bge-small card labels its column "Retrieval (15)".

MTEB(eng, v2) is a different set. It has 10 retrieval tasks, including ClimateFEVERHardNegatives, FEVERHardNegatives, HotpotQAHardNegatives, Touche2020Retrieval.v3, and only the Gaming and Unix CQADupstack forums. The v1 description says v2 "removes common fine-tuning datasets such as MSMARCO".

BEIR proper has 18 datasets on [Pyserini](https://castorini.github.io/pyserini/2cr/beir.html). It adds BioASQ, Signal1M, TREC-NEWS, and Robust04 to the 14 non-MSMARCO sets above.

Names in the literature:
- "BEIR" (LEAF: 14 datasets)
- "Retrieval (15)" on the bge-small card
- "MTEB(eng, v1) retrieval"
- "15 English tasks from BeIR" ([LightRetriever](https://arxiv.org/html/2505.12260))

I could not confirm the literal string "BEIR-15" in a primary source.

### 3. Literature, one line each

| Work | What it does | Pre-empts the claim? |
|---|---|---|
| [LEAF 2509.12539](https://arxiv.org/abs/2509.12539) (ACL 2026) | 23M query student aligned to a frozen arctic-embed-m-v1.5 document tower | Partly. One student, one tower the authors chose. No cost frontier. |
| [pyNIFE](https://github.com/stephantul/pynife) (Nov 2025, no paper) | Static NIFE query models aligned to mxbai-large and gte-modernbert. Reports 59.2 on NanoBEIR, with the metric label unclear in the README. | Partly. Same lookup-table idea. No BEIR-15 and no menu. |
| [2306.11550](https://arxiv.org/abs/2306.11550) (2023) | Two-layer BERT query encoder by embedding alignment, keeping 92.5% on BEIR | Partly. It is the distillation baseline. |
| [LightRetriever 2505.12260](https://arxiv.org/abs/2505.12260) (ICLR 2026) | Dense query = lookup then average of token vectors, over 15 BEIR tasks. Keeps about 95%. | Closest on the lookup and benchmark. The document LLM is trained with it, so it is not a frozen third-party tower. |
| [BCT 2003.11942](https://arxiv.org/abs/2003.11942), [FCT 2112.02805](https://arxiv.org/abs/2112.02805) | Compatible embeddings across model versions for image retrieval | No. Background. |
| [Drift-Adapter 2509.23471](https://arxiv.org/abs/2509.23471) (EMNLP 2025) | Adapter maps new-model queries into the old index space and recovers 95 to 99% of recall | No. Model upgrades, not a query-cost menu. |
| [Align Then Adapt 2604.03403](https://arxiv.org/abs/2604.03403) | Aligns a strong query embedder with a light document embedder | No. The light side is the documents. |
| [CARE 2604.10937](https://arxiv.org/abs/2604.10937) (ACL 2026) | Light BERT query encoder with an LLM document encoder, Chinese medical | No. Domain-specific. |
| [2601.04646](https://arxiv.org/abs/2601.04646) | Query-only fine-tuning that preserves the index, for multi-tenant search | No. Domain adaptation. |
| [Static retrieval blog](https://huggingface.co/blog/static-embeddings) | static-retrieval-mrl-en-v1 reaches 87.4% of all-mpnet-base-v2 on NanoBEIR, 397x faster on CPU | No. Symmetric and not aligned to another tower. |
| [potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) | Model2Vec static model, MTEB retrieval 35.06 against 42.92 for MiniLM | No. Same reason. |
| [OpenSearch neural sparse 2411.04403](https://arxiv.org/abs/2411.04403), [SPLADE v2 (SPLADE-doc) 2109.10086](https://arxiv.org/abs/2109.10086), [Li-LSR 2505.01452](https://arxiv.org/abs/2505.01452) | Sparse retrieval with document-only encoding and a token weight table for queries | No. The index is the authors' own sparse index. |
| [thinletter qwen3-embedding-0.6b-query-clients](https://huggingface.co/thinletter/qwen3-embedding-0.6b-query-clients) (2026-09-16) | Quantized copies of the same tower as query clients, scored as percent of fp32 on four BEIR sets | Slight overlap with "the tower itself" as one option. No frontier. |

I found no work that combines a frozen third-party tower, a menu from lookup table to the tower itself, and a BEIR-15 cost frontier. This is absence within about 10 searches, not proof of absence.

### 4. Qdrant, from the [hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/), [quantization](https://qdrant.tech/documentation/manage-data/quantization/), and [Edge](https://qdrant.tech/documentation/edge/) docs

- **RRF.** "RRF considers the positions of results within each query and boosts those that appear closer to the top in multiple sets of results." The constant k defaults to 2. Setting k is available as of v1.16.0. Weighted RRF is available as of v1.17.0.
- **DBSF** (available as of v1.11.0). "DBSF keeps the raw scores from each query but normalizes their distributions before combining." It normalizes with the mean and sample standard deviation, using the 3-sigma extremes as endpoints. Scores are not clipped to [0, 1]. The docs add that neither DBSF nor weighted RRF dominates, so choose using an eval set.
- **TurboQuant** (available as of v1.18.0). "TurboQuant is a quantization method developed by Google. It operates by applying a fast random rotation to vectors before compression". The default is `bits4` at 8x. The other options are bits2 (16x), bits1_5 (24x), and bits1 (32x). The doc says "Test TurboQuant on your data before committing". It carries no beta or GA label. Against binary quantization it "offers better recall at lower speed and equivalent storage budgets".
- **Older methods.** Scalar quantization (v1.1.0) is float32 to uint8, 4x. Binary quantization (v1.5.0) is "a single bit", 32x. Product quantization (v1.2.0) goes up to 64x. Binary also has 1.5-bit and 2-bit modes, and asymmetric quantization (v1.15.0).
- **Qdrant Edge.** "Qdrant Edge is in beta." It is "a lightweight, embedded vector search engine for in-process retrieval with a minimal memory footprint and no background services." It has Python and Rust bindings.

### 5. Why some preprints became reference points

- **[MTEB 2210.07316](https://arxiv.org/abs/2210.07316):** 58 datasets, open code, and a public leaderboard. This is why the Stella and bge-small cards quote an MTEB retrieval average.
- **[Matryoshka 2205.13147](https://arxiv.org/abs/2205.13147):** one training trick that model authors could adopt at no cost. Semantic Scholar shows 351 citations, and the Stella, LEAF, and static models all ship MRL dimensions.
- **[LIMIT 2508.21038](https://arxiv.org/abs/2508.21038):** a theory result paired with a small dataset where strong models fail. Semantic Scholar shows [136 citations](https://api.semanticscholar.org/graph/v1/paper/arXiv:2508.21038?fields=citationCount), accepted to ICLR 2026.
- **[LEAF](https://api.semanticscholar.org/graph/v1/paper/arXiv:2509.12539?fields=citationCount):** released hub models that ranked first on the leaderboard for their size. It has only 6 citations so far.
- BEIR and MTEB citation counts: not retrieved, because of rate limits.
