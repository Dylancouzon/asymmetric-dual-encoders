# Related-work notes (web-verified 2026-09-30)

Verification level: "abs" = arXiv/venue abstract page read; "full" = paper HTML read (a fetch tool summarised it, so quotes marked [summary] are as returned by the tool and should be re-checked against the PDF before printing). Items marked UNVERIFIED were not read in full.

## 1. Wang and Lyu, query encoder distillation via embedding alignment
- Yuxuan Wang, Hong Lyu. "Query Encoder Distillation via Embedding Alignment is a Strong Baseline Method to Boost Dense Retriever Online Efficiency." SustaiNLP Workshop at ACL 2023. arXiv:2306.11550. https://arxiv.org/abs/2306.11550
- Overlap: same asymmetric setup and same loss family. The teacher document encoder stays frozen, and the student minimizes Euclidean distance between student and teacher query embeddings (full [summary]). Abstract: "even a 2-layer, BERT-based query encoder can still retain 92.5% of the full DE performance on the BEIR benchmark via unsupervised distillation and proper student initialization."
- Difference: teacher is msmarco-bert-base-dot-v5 (768d); students are truncated-layer or DistilBERT transformers trained on MS MARCO queries. No lookup-table student, no teacher-selection study, no routing or blending.

## 2. EmbedDistill
- Seungyeon Kim, Ankit Singh Rawat, Manzil Zaheer, Sadeep Jayasumana, Veeranjaneyulu Sadhanala, Wittawat Jitkrittum, Aditya Krishna Menon, Rob Fergus, Sanjiv Kumar. "EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval." arXiv:2301.12005 (v1 Jan 2023, revised Jul 2023; no venue on arXiv page). https://arxiv.org/abs/2301.12005
- Overlap: asymmetric students with "a small query encoder and a frozen document encoder inherited from the teacher" (full [summary]). Abstract: "1/10th size asymmetric students that can retain 95-97% of the teacher performance."
- Difference: in-domain MS MARCO and NQ, adds query generation and document-side embedding matching, teacher is DE or cross-encoder. Student is a transformer (11.3M 4-layer to 67.5M 6-layer). Reports recall and MRR on those sets, not BEIR-15 nDCG@10. No lookup table. The "frozen teacher document encoder reused in the student" result is the closest prior to our frozen-index premise.

## 3. LEAF
- Robin Vujanic, Thomas Rueckstiess. "LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations." arXiv:2509.12539 (cs.IR; v1 Sep 2025, revised Apr 2026). https://arxiv.org/abs/2509.12539
- Overlap: abstract: "This alignment enables asymmetric architectures in information retrieval contexts, where documents use the larger teacher while queries employ smaller student models." Headline number [summary of full text]: "it retains 97.7% of the teacher's performance while running a 4.7x smaller model at query time"; 54.8 nDCG@10 asymmetric vs 56.1 teacher (arctic-embed-m-v1.5, 109M) with leaf-ir (23M) on BEIR; symmetric leaf-ir is 53.9, and asymmetric mode is better on 11 of 14 datasets.
- Difference: teacher is 109M, only 4.7x compression, student is a transformer. Paper did not state whether the index is frozen (tool summary said not explicit; the design implies teacher-encoded documents). No lookup-table student, no 10-teacher comparison, no routing. Our transformer at 34.5M against a 400M tower is a larger compression gap with lower retention (90.5%).
- Caveat: LEAF trains the student to mimic teacher embeddings of both queries and documents; check the PDF before claiming the training target matches ours.

## 4. LightRetriever
- Guangyuan Ma, Yongliang Ma, Xuanrui Gou, Zhenpeng Su, Ming Zhou, Songlin Hu. "LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference." ICLR 2026. arXiv:2505.12260. https://arxiv.org/abs/2505.12260
- Overlap: closest prior for the lookup-table query side. Abstract: "reduces the workload of query encoding to no more than an embedding lookup ... over 1000x speedup in query encoding ... maintaining an average of 95% retrieval performance." Full text [summary]: dense query vector is "obtained with a simple lookup-then-average operation" over precomputed token vectors; BEIR 54.4 vs 56.8 nDCG@10 for the Llama3.1-8B version (about 95.8%); Chinese benchmark 63.0 vs 67.6 (about 93.2%).
- Difference: the document side is the full LLM trained jointly with the query lookup by contrastive learning (not a frozen, separately pretrained tower; the paper did not state freezing). Our table is fit in closed form onto an existing frozen tower's query outputs and reaches 81.4% of the tower on BEIR-15, below their 95%. No per-tower screening study, no routing. A fair limitation line: their retention is higher because the document tower co-trains with the lookup.

## 5. NanoVDR
- Zhuchenyang Liu, Yao Zhang, Yu Xiao. "NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval." arXiv:2603.12824 (arXiv abstract page lists EMNLP 2026 Main). https://arxiv.org/abs/2603.12824
- Overlap: abstract: "a frozen 2B VLM teacher indexes documents offline, while a distilled text-only student as small as 69M parameters encodes queries at inference"; "pointwise cosine alignment on query text consistently outperforms ranking-based and contrastive alternatives, while requiring only pre-cached teacher query embeddings"; "retains 95.1% of teacher quality." Same recipe family as ours (regress onto cached teacher query embeddings, frozen index).
- Teacher quality vs student (full [summary]): uses ONE teacher (Qwen3-VL-Embedding-2B) throughout and reports "no ablations compare different teachers." Retention numbers: NanoVDR-S 92.4%, S-Multi 95.1%. It does not compare quality across different teachers. Appendix H does test teacher quality against student retention across 19 datasets for one fixed teacher (Pearson r = +0.607); that differs from our cross-teacher, fixed-dataset comparison. Verified from v3 on 2026-10-01. It also shows students exceed other, larger VLM baselines (ColPali 3B, DSE-Qwen2 2B) on ViDoRe v2/v3, not the teacher itself.
- Difference: visual-document domain, transformer student, no lookup table, no routing.

## 6. Static embeddings: pyNIFE, Model2Vec, sentence-transformers static-retrieval-mrl-en-v1
- pyNIFE. Stephan Tulkens. "pynife: Nearly Inference Free Embeddings." Software, GitHub. https://github.com/stephantul/pynife (no paper found). README [summary]: "compresses large embedding models into static, drop-in replacements with up to 900x faster query embedding"; teacher is unchanged so a teacher-built index is reused; static model initialized by running all tokens through the teacher, trained with cosine loss, two stages (documents, then queries), custom about 100k-vocabulary tokenizers. Reported for gte-modernbert-base: NIFE 59.2 nDCG@10 at 71,400 QPS vs teacher 66.34 at 237 QPS (about 89% retention by my arithmetic; the README does not state a percentage, and the benchmark set was not stated in the summary, so check before citing).
  Closest prior to our lookup-table encoder: same idea (frozen teacher index, static token-vector query side). Difference: gradient-trained, single teacher; ours is a closed-form fit, int8 per-token, tested across 10 towers.
- Model2Vec. Stephan Tulkens, Thomas van Dongen. "Model2Vec: Fast State-of-the-Art Static Embeddings." Zenodo software, 2024, doi:10.5281/zenodo.17270888. https://github.com/MinishLab/model2vec. README: distills a sentence transformer by forward-passing the vocabulary to get token embeddings plus post-processing; "reduces the size of a Sentence Transformer model by a factor of up to 50 and makes models up to 500 times faster, with a small drop in performance." General sentence-embedding models, no frozen-index retrieval claim.
- Sentence-transformers static embeddings. Tom Aarsen. "Train 400x faster Static Embedding Models with Sentence Transformers." Hugging Face blog. https://huggingface.co/blog/static-embeddings (author and 15 January 2025 date verified from the post on 2026-10-01). Model: https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1. Verified: encoder is "as simple as a dictionary lookup" (EmbeddingBag, mean pooling); "100x to 400x faster" on CPU than all-mpnet-base-v2; "87.4%" of all-mpnet-base-v2 on NanoBEIR; 1024d; trained with MultipleNegativesRankingLoss plus MatryoshkaLoss on 13 datasets, "Rather than distillation" (blog). Its own index, not a frozen transformer tower's index. Retention (87.4%) sits between our table (81.4%) and transformer (90.5%), but vs a different, weaker reference model on NanoBEIR.

## 7. Arabzadeh et al.
- Negar Arabzadeh, Xinyi Yan, Charles L. A. Clarke. "Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection." CIKM 2021 (venue verified on 2026-10-01; DOI 10.1145/3459637.3482159). arXiv:2109.10739. https://arxiv.org/abs/2109.10739
- Overlap: per-query routing for cost. Abstract: "a classifier to select a suitable retrieval strategy (i.e., sparse vs. dense vs. hybrid) for individual queries." Full [summary]: BERT cross-encoder over query text (pre-retrieval), routes BM25 vs ANCE; at 50% dense budget recall@1000 0.95 vs 0.91 random; MS MARCO dev only.
- Difference: routes between different retrievers with different indexes; ours routes between two query encoders over one index, reports a per-query oracle upper bound. No retriever-agreement feature (paper has none).

## 8. Query performance prediction
- Faggioli, Formal, Marchesin, Clinchant, Ferro, Piwowarski. "Query Performance Prediction for Neural IR: Are We There Yet?" ECIR 2023. arXiv:2302.09947. https://arxiv.org/abs/2302.09947. Finding quoted: "QPPs perform statistically significantly worse on neural IR systems" (14 systems, 19 QPPs, DL'19 and Robust'04; my tool summary says about 10% drop in passage retrieval, re-check). Use: motivates why QPP signals for dense retrievers are weak, so routing with pre-retrieval features is hard.
- Vlachou, Macdonald. "On Coherence-based Predictors for Dense Query Performance Prediction." arXiv:2310.11405 (Oct 2023). https://arxiv.org/abs/2310.11405. Abstract: dense-embedding coherence predictors for ANCE and TCT-ColBERT; query types account for "up to 35% of predictor instability." Venue not stated on the arXiv page.
- Chifu, Dejean, Mothe, Garouani, Ortiz, Ullah. "Uncovering the Limitations of Query Performance Prediction: Failures, Insights, and Implications for Selective Query Processing." arXiv:2504.01101. https://arxiv.org/abs/2504.01101. Concludes "QPP-driven selective query processing offers only marginal gains" (tool summary). Useful contrast with our oracle-routing upper bound.
- Retriever-agreement signal: I found no peer-reviewed QPP paper whose main signal is cross-retriever top-k agreement for dense models (searched once; not exhaustive). The nearest source is a Qdrant article by Dylan Couzon, "Predicting Weak Retrieval Without an LLM," 2026-06-24, https://qdrant.tech/articles/predicting-weak-retrieval/: `dense_agreement` (overlap across independent dense models) and `retriever_divergence`; "When two retrievers, or two dense models, return different documents, one is usually lost, most often on jargon and out-of-vocabulary terms"; agreement AUC 0.73-0.76 on NFCorpus. It is the author's own prior work; cite as self-citation, not independent evidence. Classic agreement-style signals (for example score-list overlap across systems) exist in older QPP literature; I did not verify a specific paper.

## 9. Compatible representation learning
- Yantao Shen, Yuanjun Xiong, Wei Xia, Stefano Soatto. "Towards Backward-Compatible Representation Learning." CVPR 2020 (oral). arXiv:2003.11942. https://arxiv.org/abs/2003.11942. Trains the NEW model so its features are comparable with the old gallery, avoiding backfilling. Face and image retrieval. Difference: they modify the new model's training with an influence loss; our towers are fixed and only the query side is regressed, with no change to the document model.
- Vivek Ramanujan, Pavan Kumar Anasosalu Vasu, Ali Farhadi, Oncel Tuzel, Hadi Pouransari. "Forward Compatible Training for Large-Scale Embedding Retrieval Systems." CVPR 2022. arXiv:2112.02805. https://arxiv.org/abs/2112.02805. Abstract: BCT "can significantly hinder the performance of the new model"; FCT learns side-information plus a forward transformation from old to new embeddings; gains over BCT of +18.1% (ImageNet-1k), +5.4% (Places-365), +8.3% (VGG-Face2) (these are retrieval accuracy gains vs BCT). Difference: vision, maps old embeddings to new, needs side-information stored at indexing time. Ours maps query text to the existing index with nothing stored per document.

## 10. Stronger teacher is not a better student
- Jang Hyun Cho, Bharath Hariharan. "On the Efficacy of Knowledge Distillation." ICCV 2019. arXiv:1910.01348. https://arxiv.org/abs/1910.01348. Finding: "larger models do not often make better teachers"; small students cannot mimic large teachers; early-stopped teachers help. Image classification (CIFAR/ImageNet).
- IR-specific: Zhenghao Lin et al. "PROD: Progressive Distillation for Dense Retrieval." WWW 2023. arXiv:2209.13335. https://arxiv.org/abs/2209.13335. Abstract frames the problem as "stronger teacher models don't always produce better student models" (tool summary of the abstract) and proposes progressive teacher and data distillation. Differs: score-level distillation into a full dual encoder, not regression onto a frozen index; does not compare teacher retrieval quality against student quality across many towers.
- I found no IR paper that regresses a query-side student onto many different frozen towers and correlates teacher retrieval quality with student quality. Our 10-tower result appears new, but the search was limited to web search, not a systematic survey.

## 11. Blending query vectors from two query encoders over one index
- Searched twice (several phrasings: interpolation, averaging, convex combination, heterogeneous query encoders sharing a document index). Found none. Results were about multi-vector query generation, multi-modal fusion, rank fusion (RRF and score fusion across different indexes or fields), and weight-space model soups. Closest concepts, all different: rank or score fusion across separate retrievers (hybrid search), multi-query retrieval (arXiv:2511.02770 "Beyond Single Embeddings: Capturing Diverse Targets with Multi-Query Retrieval", listed in search results, not read), and Geng et al. heterogeneous-ensemble distillation (arXiv:2411.04403), which blends teacher signals during training, not query vectors at search time.
- Plain statement for the paper: "we are not aware of prior work that blends query vectors from two different query encoders into a single search over one document index." Phrase as "not aware of", since the search was not exhaustive.

## 12. Adaptive or cascaded query encoding
- Closest on query-encoder cost: Nachshon Cohen, Yaron Fairstein, Guy Kushilevitz. "Extremely efficient online query encoding for dense retrieval." Findings of NAACL 2024. https://aclanthology.org/2024.findings-naacl.4/ (code: github.com/amzn/extremely-efficient-query-encoder). Small transformer: about 12x latency drop, MRR@10 38.2 to 36.2 on MS MARCO; RNN that encodes incrementally: about 38x, MRR@10 35.5. Static per-system choice, no per-query routing, not frozen-index BEIR.
- Jurek Leonhardt, Henrik Muller, Koustav Rudra, Megha Khosla, Abhijit Anand, Avishek Anand. "Efficient Neural Ranking using Forward Indexes and Lightweight Encoders." ACM TOIS (2023/2024; arXiv page said "accepted"). arXiv:2311.01263. https://arxiv.org/abs/2311.01263. Fast-Forward indexes: re-ranking with precomputed document vectors plus score interpolation with lexical scores, and reduced-complexity query encoders. Interpolates scores from a lexical and a dense retriever, not two query encoders over one dense index. I did not read the lightweight-encoder details.
- Early exit work found (Early Exit Strategies for Approximate k-NN Search in Dense Retrieval, arXiv:2408.04981, CIKM 2024) concerns ANN cluster visits, not query-encoder depth. I found no paper that early-exits or cascades the query encoder itself. Arabzadeh (item 7) is the nearest per-query routing precedent.
- Also relevant, inference-free sparse lookup query sides: Nardini, Nguyen, Rulli, Venturini, Yates. "Effective Inference-Free Retrieval for Learned Sparse Representations." SIGIR 2025. arXiv:2505.01452 (Li-LSR: table lookup replaces the query encoder; +1.8 BEIR nDCG@10 over Splade-v3-Doc). Zhichao Geng, Yiwen Wang, Dongyu Ru, Yang Yang. "Towards Competitive Search Relevance For Inference-Free Learned Sparse Retrievers." arXiv:2411.04403 (BEIR +3.3 nDCG@10 over prior inference-free model, latency "only 1.1x that of BM25"). Sparse, not dense-frozen-tower.

## Gaps and cautions
- Arabzadeh venue and HF blog author/date were confirmed in the 2026-10-01 source pass; see the dated review below.
- Quotes marked [summary] passed through a fetch tool's summarizer; verify in the PDF before printing.
- A search-only check cannot prove absence for items 10 and 11.

## URLs read (fetched)
https://arxiv.org/abs/2306.11550
https://arxiv.org/html/2306.11550
https://arxiv.org/abs/2301.12005
https://arxiv.org/html/2301.12005
https://arxiv.org/abs/2509.12539
https://arxiv.org/html/2509.12539
https://arxiv.org/abs/2505.12260
https://arxiv.org/html/2505.12260
https://arxiv.org/abs/2603.12824
https://arxiv.org/html/2603.12824
https://arxiv.org/abs/2109.10739
https://arxiv.org/html/2109.10739
https://arxiv.org/abs/2112.02805
https://arxiv.org/abs/2003.11942
https://arxiv.org/abs/1910.01348
https://arxiv.org/abs/2209.13335
https://arxiv.org/abs/2302.09947
https://arxiv.org/abs/2310.11405
https://arxiv.org/abs/2504.01101
https://arxiv.org/abs/2311.01263
https://arxiv.org/abs/2411.04403
https://arxiv.org/abs/2505.01452
https://aclanthology.org/2024.findings-naacl.4/
https://github.com/stephantul/pynife
https://github.com/MinishLab/model2vec
https://huggingface.co/blog/static-embeddings
https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1
https://qdrant.tech/articles/predicting-weak-retrieval/
https://stephantul.github.io/ (no NIFE post found)
Failed or unreadable: https://www.dei.unipd.it/~ferro/papers/2023/ECIR2023-FFMCFP.pdf (binary), https://link.springer.com/chapter/10.1007/978-3-031-28244-7_15 (redirect), https://dl.acm.org/doi/full/10.1145/3631939 (403).
Seen only as search-result titles/snippets (not read): https://arxiv.org/abs/2408.04981, https://arxiv.org/abs/2511.02770, web-search summaries for BCT and Cho & Hariharan venue pages.

## Verification update, 2026-10-01

`REVIEWS/2026-10-01-owner-literature.md` records primary-source checks for all 14 printed references, corrected bibliography entries, and the closest overlaps. NanoVDR's positive teacher-quality association varies datasets for a fixed teacher, not teachers. Cheap training, frozen-index reuse, static/contextual switching, and full-BEIR evaluation already have prior art. The paper claims the cross-teacher table comparison and measured same-index search penalty, with scoped component-cost accounting. No unverified LEAF venue is printed. The bibliography now gives full entries for LightRetriever and QPP instead of placeholders.


## Fixed-index search and precision update, 2026-10-01

V13 adds four primary references, making 18 printed entries. `REVIEWS/2026-10-01-followup-literature.md` records the broader search; `REVIEWS/2026-10-01-quantization-semantics.md` checks the tested release against tagged source. Bibliographic identities were checked on the primary pages below.

- OOD-DiskANN, Jaiswal et al. (2022 preprint): https://arxiv.org/abs/2211.12850. Query-distribution effects on graph-search cost are prior art; its remedy uses queries when constructing the graph.
- RoarGraph, Chen et al. (PVLDB 17(11):2735–2749, 2024): https://www.vldb.org/pvldb/vol17/p2735-chen.pdf. Query-aware graph construction addresses cross-modal query difficulty. Constella measures substituted text query encoders over frozen document representations.
- Beyond Hamming / QA-Cos, Daehun Nyang (ICML 2026, PMLR306:94126–94143): https://proceedings.mlr.press/v306/nyang26a.html. Query-aware binary candidate decoding is established; E18 is a controlled empirical consequence, not a new quantizer or scorer.
- Qdrant v1.19.1: https://github.com/qdrant/qdrant/tree/v1.19.1. Exact file links in the semantics review establish the default sign-coded query, scalar8 collection setting, and native rescoring/segment behavior. Current documentation alone cannot certify the tested version.

The native FiQA interaction weakens at the primary setting on the larger diagnostic. E18 instead holds document sign codes fixed and compares exact sign-query and graded-query candidate coverage. The paper claims separate query-computation and query-precision choices, with no native precision speedup, architectural sensitivity law, or proven geometry mechanism.

## Explanatory source update, 2026-10-01 (v14)

The narration pass adds two primary references, bringing the printed bibliography to 20. The Stella model card identifies the model and its default 1024-dimensional output: https://huggingface.co/NovaSearch/stella_en_400M_v5. The official BEIR repository describes the heterogeneous retrieval benchmark and its dataset/evaluation framework: https://github.com/beir-cellar/beir. Read-only copies were fetched into ignored `.firecrawl/v14-stella-model.md` and `.firecrawl/v14-beir-overview.md`. These clarify names and roles; they supply no new performance comparison. Exact model revision and the study's 15-dataset selection remain pinned in the local evidence.

## Production migration motivation, 2026-10-02

Primary source: Qdrant's [embedding-model migration procedures](https://qdrant.tech/documentation/tutorials-operations/embedding-model-migration/), read via Firecrawl into ignored `.firecrawl/m15-embedding-migration.md`. The documented alternatives retain old and new representations during background re-embedding, maintain incoming upserts under both models, and switch queries when migration is complete. Collection aliases support an atomic collection switch; named-vector migration can retain the original vector for rollback without duplicating payloads. Deletes and partial updates require particular care in the documented two-collection procedure. These support the introduction's availability and transition requirements, not a mandatory outage or a universal twofold storage bill. Added resource demands and validation requirements are engineering implications of the described procedure; no production duration, dollar cost, downtime, or risk probability was measured here.
