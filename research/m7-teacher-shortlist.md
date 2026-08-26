> **STALE as of 2026-08-26.** This shortlist was built under the strict vendor rule (no vendor
> with any competing vector-search product). Dylan relaxed that rule: vendors whose vector offering
> is far from their main business are now admissible with heavy justification — see the "Vendor
> rule, relaxed" section in CLAUDE.md for the operationalised tiers. Candidates dismissed below on
> *vendor* grounds alone (Alibaba gte/Qwen3, Microsoft e5, IBM granite, Snowflake arctic) may now
> be viable. The licence rule and the vocab x dim size arithmetic still bind unchanged.
> Re-run this sweep before committing to a teacher.
>
> **DONE — see `m7-teacher-shortlist-2026-08-26.md`, which supersedes this file.** Kept only for the 2026-08-25 audit trail; do not use its rankings.

# M7 teacher document-encoder shortlist (2026-08-25, Sonnet web sweep, specs verified on HF cards)

Decision made with Dylan 2026-08-25: **default teacher BAAI/bge-base-en-v1.5**; the executing session may swap to a measurably better teacher that passes both hard constraints (permissive license, vendor ships no competing vector search product). Vocab size is a first-class criterion: the released lookup table is vocab × dim.

## Model specs

| Model | Params | Emb dim | Tokenizer / vocab | Max seq | License | Retrieval quality (source) | MRL | Query prefix / instruction |
|---|---|---|---|---|---|---|---|---|
| Snowflake/snowflake-arctic-embed-m-v1.5 | 109M | 768 | BERT WordPiece, 30,522 | 512 | Apache 2.0 | 55.14 MTEB v1 Retrieval (card); **0.5264 avg-6 measured by us** | Yes, to 256 | "Represent this sentence for searching relevant passages: " on queries; docs bare |
| Snowflake/snowflake-arctic-embed-m-v2.0 | 305M | 768 | XLM-RoBERTa, ~250,002 | 8,192 | Apache 2.0 | 55.4 MTEB v1 Retrieval (card) | Yes, to 256 | Same prefix; multilingual. Vocab → 384 MB fp16 table |
| BAAI/bge-small-en-v1.5 | 33.4M | 384 | BERT WordPiece, 30,522 | 512 | MIT | **0.5042 avg-6 measured by us** | No | Same prefix as arctic, optional in v1.5 |
| BAAI/bge-base-en-v1.5 | 109M | 768 | BERT WordPiece, 30,522 | 512 | MIT | 53.25 MTEB Retrieval (card); **not yet measured on our six — first M7 job** | No | Same prefix, optional |
| intfloat/e5-base-v2 | 109M | 768 | BERT WordPiece, 30,522 | 512 | MIT | ~48.7 BEIR (leaderboard, unofficial) | No | Mandatory "query: " / "passage: " |
| Alibaba-NLP/gte-base-en-v1.5 | 137M | 768 | BERT WordPiece, 30,522 | 8,192 | Apache 2.0 | 54.09 MTEB Retrieval (card) | No | None |
| Alibaba-NLP/gte-modernbert-base | 149M | 768 | ModernBERT BPE (vocab unconfirmed) | 8,192 | Apache 2.0 | 55.33 BEIR (card) | Unconfirmed | None |
| ibm-granite/granite-embedding-english-r2 | 149M | 768 | ModernBERT, 50,368 | 8,192 | Apache 2.0 | 59.5 avg across benchmarks (card, not retrieval-only) | Yes (512/256/128) | Appears prefix-free |
| ibm-granite/granite-embedding-small-english-r2 | 47M | 384 | ModernBERT, 50,368 | 8,192 | Apache 2.0 | **0.4947 avg-6 measured by us** | Yes | Same family |
| nomic-ai/nomic-embed-text-v1.5 | 137M | 768 | WordPiece ~30,522 | 8,192 | Apache 2.0 | 62.28 (card, all-MTEB not retrieval-only) | Yes (to 64) | Mandatory task prefixes (search_query:/search_document:) |
| nomic-ai/modernbert-embed-base | 149M | 768 | ModernBERT | 8,192 | Apache 2.0 | "outperforms nomic-v1.5" (card) | Yes (768/256) | Nomic-style prefixes |
| mixedbread-ai/mxbai-embed-large-v1 | 335M | 1024 | BERT WordPiece, 30,522 | 512 | Apache 2.0 | 54.39 MTEB Retrieval (card) | Yes (512/256) | Same "Represent this sentence…" prefix |
| Qwen/Qwen3-Embedding-0.6B | 0.6B | 32–1024 | Qwen BBPE, 151,936 | 32,768 | Apache 2.0 | 61.83 MTEB v2 En Retrieval (card) | Yes | Instruct template. Vocab → 311 MB fp16 table |
| google/embeddinggemma-300m | 300M | 768 | SentencePiece, 262,144 | 2,048 | **Gemma terms — fails permissive rule** | 65.11 MTEB v2 En mean (card) | Yes | task:/title: templates |
| NovaSearch/stella_en_400M_v5 | 400M | 1024 | WordPiece 30,522 (gte-large backbone) | 512 | MIT | Strong MTEB standing (exact figure unverified) | Yes (512–8192 heads) | s2p instruct prompt |
| MongoDB/mdbr-leaf-ir | 23M | 768 | WordPiece 30,522 | 512 | Apache 2.0 | **0.5123 avg-6 measured by us** | Yes (256) | arctic-style prefix |
| BAAI/bge-large-en-v1.5 | 335M | 1024 | WordPiece 30,522 | 512 | MIT | 54.29 MTEB Retrieval (card) | No | Same optional bge prefix |
| BAAI/bge-m3 | 568M | 1024 dense | XLM-R 250,002 | 8,192 | MIT | Multilingual/multi-function; no English-retrieval edge over bge-base on MTEB | No | None |
| jinaai/jina-embeddings-v3 | 570M | 32–1024 | XLM-R SentencePiece, 250,002 | 8,192 | **CC BY-NC 4.0 — disqualified** | 65.52 MTEB avg (all tasks) | Yes | Task LoRA adapters |

Unverified in this pass (flagged, not guessed): gte-modernbert MRL support and exact vocab; e5-base-v2 official retrieval number; stella exact MTEB figure. No sub-400M 2025–2026 permissive model clearly beat this list; BGE-M3 (MIT) is strong but 568M.

## Vendor → competing vector-search product evidence

| Vendor | Model(s) | Competing product | Evidence |
|---|---|---|---|
| Snowflake | arctic-embed | Cortex Search (uses Arctic Embed M as its backbone) | https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview |
| Microsoft | e5 | Azure AI Search | https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-create-index |
| Alibaba | gte, Qwen3-Embedding | OpenSearch Vector Search Edition / AnalyticDB | https://www.alibabacloud.com/help/en/open-search/vector-search-edition/introduction-to-vector-search-edition |
| IBM | granite-embedding | watsonx.data managed Milvus | https://www.ibm.com/docs/en/watsonxdata/standard/2.2.x?topic=components-milvus |
| Nomic AI | nomic-embed | Nomic Atlas vector search | https://docs.nomic.ai/atlas/capabilities/vectors |
| Mixedbread | mxbai-embed | Mixedbread vector stores | https://www.mixedbread.com/docs/vector-stores/data-models |
| Google | embeddinggemma | Vertex AI Vector Search | https://medium.com/google-cloud/introducing-vertex-ai-vector-search-2-0-from-zero-to-billion-scale-90ed666dac43 |
| MongoDB | mdbr-leaf-ir | Atlas Vector Search | https://www.mongodb.com/products/platform/atlas-vector-search |
| Jina AI (acquired by Elastic, Oct 2025 per search results — verify before citing) | jina-embeddings-v3 | Jina Search Foundation; Elasticsearch vector search | https://jina.ai/deepsearch/ |
| BAAI | bge | **None found** — non-profit research lab | https://huggingface.co/BAAI |
| NovaSearch | stella | **None found** — open research group; backbone derives from Alibaba's gte-large (provenance counts toward defensibility) | (no product URL found) |

## Why bge-base-en-v1.5 is the default

Only strong candidate whose vendor has no vector search product at all. MIT license. BERT vocab keeps the table at 30,522 × 768 = 23.4M params → 46.9 MB fp16 / 23.4 MB int8, decimal MB (a 256-dim MRL table would be 15.6 MB, but bge has no MRL). Family already validated in our harness (bge-small in the M4 matrix). Cost vs arctic-m: roughly two points of teacher ceiling. Per the M7 evaluation protocol, teacher selection uses dev-suite runs and official published numbers only; bge-base's six-set symmetric row is measured once, inside the final matrix run, as the retention-ceiling row.

Same-vendor alternatives, for the delegated swap: **bge-large-en-v1.5** is the first candidate — +1 point of teacher ceiling (54.29 vs 53.25 MTEB-Ret) for a 62.5 MB table (same 30,522 vocab at 1024d), a 2.05 GB/1M-doc index (vs 1.54), and ~3x encode cost; worth a dev A/B if the base teacher caps early. **bge-m3 is disqualified for this architecture**: its 250,002-token XLM-R vocab makes the lookup table 250,002 × 1024 × 2 = 512 MB fp16 — larger than LightRetriever's 466 MB table this project exists to beat — and its strengths (multilingual, multi-function sparse/ColBERT heads) don't show up as an English-dense-retrieval gain over bge-base. Its 568M params also mean ~5x corpus-encode cost per experiment.
