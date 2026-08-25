# Literature sweep: asymmetric dual encoders for edge retrieval

Sweep date: 2026-08-24. Anchor paper: LightRetriever (arXiv 2505.12260). Scope: 2023-2026, with a few pre-2023/2024 foundational papers included where the brief leans on them without full citation. Numbers below come from paper abstracts/HTML/PDF extraction via LLM-assisted fetches — treat exact decimal points as approximate and verify against the source table before quoting in the decision report. Anything not directly sourced is marked **[SPECULATION]**.

## What the brief missed — top items

1. **No direct LightRetriever successor exists.** Citation search turned up no 2025-2026 paper that builds on or critiques LightRetriever by name. The two papers citing arXiv 2505.12260 on Semantic Scholar (MIRA, ReSuMe) cite it in passing as related work, not as a baseline they improve on. The closest thing to a "successor line" is the authors' own paper, revised through v5 (30 Jan 2026) — same architecture, no separate follow-up paper found.
2. **LightRetriever ships checkpoints from 1B to 8B**, not just the 8B the brief scoped around: `lightretriever-llama3.2-1b`, `-3b`, `qwen2.5-1.5b`, `-3b`, `-7b`, plus the 8B variants and an MRL variant. This directly loosens the M0 "8B doc encoder is tight" constraint — a 1B-3B backbone gets you running LightRetriever's actual released weights instead of only reproducing the recipe.
3. **The real competitor to the brief's whole idea already exists as a published, code-and-weights-released method: MongoDB LEAF (arXiv 2509.12539, ACL 2026).** It does the exact "tiny adapter into a frozen big-model space" alignment the brief scopes as theme 3, but goes further: a 23M-parameter distilled encoder reaches 96-98% of its teacher's BEIR nDCG@10 when used asymmetrically (query = LEAF, document = frozen teacher), and inherits Matryoshka truncation and int8/binary quantization robustness from the teacher for free. This is a stronger, already-solved baseline for the "spectrum" the brief describes at the "1-2 layer query encoder" rung — it should be a required comparison point, not just background.
4. **DeepMind's LIMIT paper (arXiv 2508.21038)** is a theoretical, not just empirical, ceiling on single-vector embeddings: for any fixed embedding dimension there exist document sets no single-vector query embedding can retrieve correctly, on any architecture. Best dense models get under 20% recall@100 on their adversarial LIMIT benchmark while BM25 gets 93.6%. This generalizes and sharpens the brief's "query compositionality is the weakness" claim into a hard bound worth citing directly, and it argues the failure isn't specific to bag-of-token averaging.
5. **Industry-scale numbers already exist for the query-tower-distillation rung of the spectrum**, at China's Xiaohongshu (RedNote): a 29M-parameter, 4-layer BERT query tower distilled from a 7B LLM query tower keeps 99% of R@50 (54.57 vs 55.15) at 33,810 QPS vs 408 QPS for the 7B tower (arXiv 2411.15766, in production, A/B tested). This is a real, deployed data point for "how much do you retain as query compute drops," which is exactly the brief's stated research question, and it predates LightRetriever.

## 1. LightRetriever: citation search and author follow-ups

**LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference.** Ma, Ma, Gou, Su, Zhou, Hu. arXiv 2505.12260, submitted 18 May 2025, revised through v5 (30 Jan 2026). Under review, OpenReview id vNEY32I8Y8 (review content behind a bot-check wall, not retrievable via fetch tools in this sweep).
- Document tower: full-sized LLM (Llama-3.1-8B, Llama-3.2-1B/3B, Qwen2.5-1.5B/3B/7B, Mistral-0.3-7B variants all released), last-token hidden state, normalized.
- Query tower: token embedding lookup + term-frequency weighting, no transformer forward pass at query time; optional MRL truncation.
- Reported: >1000x query-encoding speedup and >10x end-to-end throughput vs. serving the full LLM on an A800; ~95% average retrieval performance retained across tasks (headline framing figure — the brief's specific 52/47 and 54.4/57 nDCG@10 numbers should be re-pulled from the exact table in the version you build against, since dense vs. hybrid figures did not extract cleanly from the HTML fetch in this sweep and the paper has been revised five times since May 2025).
- Artifacts: **yes** — code at github.com/caskcsg/lightretriever, 8 checkpoints on HF under the `lightretriever` org spanning 1B-8B backbones.
- Citation search (Semantic Scholar graph API on arXiv:2505.12260): 2 citing papers found, both 2026, both peripheral — "MIRA: An LLM-Assisted Benchmark for Multi-Category Integrated Retrieval" and "ReSuMe: Retriever-Summarizer Mutual Enhancement via Reinforcement Learning." Neither builds on or critiques the LightRetriever architecture; both cite it as one line item among related lightweight-retrieval work. **No direct successor or critique paper found.**
- Author follow-up work: no separate 2025-2026 paper from Ma/Hu extending this line was found; the arXiv revisions (v1→v5) are the only visible iteration.

## 2. Asymmetric / heterogeneous dual encoder literature

**Exploring Dual Encoder Architectures for Question Answering.** Dong et al., Google, EMNLP 2022 (arXiv 2204.07120). Foundational, pre-dates the 2023-2026 window but is the paper that names and formalizes the SDE/ADE distinction the brief's `instructions.md` refers to informally — worth citing directly rather than by description.
- Compares Siamese Dual Encoders (SDE, shared weights) against Asymmetric Dual Encoders (ADE, fully separate weights) on MS MARCO, open-domain NQ, MultiReQA.
- Finding that cuts against the brief's framing: **naive ADE underperforms SDE significantly.** The fix isn't just "make the query tower cheap" — it's sharing the final projection layer (ADE-SPL). ADE-SPL closes the gap with SDE, showing the projection head, not the encoder body, is the critical point of embedding-space alignment. Implication for this project: whatever cheap query function is chosen, a shared/aligned final projection matters more than the encoder depth.
- Artifacts: no code/checkpoints found released.

**LEAF: Knowledge Distillation of Text Embedding Models with Teacher-Aligned Representations.** MongoDB. arXiv 2509.12539, to appear ACL 2026.
- Method: pure L2/cosine regression of student embeddings onto frozen teacher embeddings — no contrastive loss, no hard negatives, no model-internals access needed. Directly the "tiny adapter into a frozen space" idea from the brief's theme 3, except the "adapter" here is a full small transformer (23M params), not a linear map.
- `leaf-ir` (23M) distilled from `arctic-embed-m-v1.5` (109M, 4.7x compression): 53.9 nDCG@10 standalone / 54.8 asymmetric (query=leaf, doc=teacher) vs. teacher's 56.1 — 96.1%/97.7% retention.
- `leaf-mt` (23M) distilled from `mxbai-embed-large-v1` (335M, 14.6x compression): 59.4/60.1 aggregate MTEB-v2-English vs teacher's 62.0 — 95.8%/96.9% retention.
- Automatically inherits MRL truncation and int8/binary quantization robustness from the teacher without training for it — nearly identical degradation curve to the teacher's own truncation/quantization behavior.
- Artifacts: **yes** — Apache-2.0 models on HF (`MongoDB/mdbr-leaf-ir`, `mdbr-leaf-ir-asym`, `mdbr-leaf-mt`, `mdbr-leaf-mt-asym`), code and MongoDB engineering blog writeup.
- This is the closest published thing to "beat LightRetriever's zero-compute query with a still-cheap-but-nonzero query encoder" and belongs in the M1 candidate list as a first-class comparison, not a footnote — the brief's CLAUDE.md already lists it as a route to confirm, this sweep confirms it's real, released, and has strong numbers.

**Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders (CARE).** arXiv 2604.10937 (2026).
- Pairs a 305M BERT-style query encoder with a 4B or 8B LLM document encoder. Two-stage training: Stage 1 aligns the frozen-document-encoder query tower via asymmetric contrastive + MSE loss on unlabeled data; Stage 2 unfreezes both and fine-tunes jointly with asymmetric InfoNCE on labeled data.
- CARE-0.3B-4B: 55.91 nDCG@10; CARE-0.3B-8B: 56.75, both on their new CMedTEB benchmark, beating symmetric baselines bge-large-zh-v1.5 (50.32) and gte-Qwen2-1.5B (55.39).
- Domain-specific (Chinese medical) — numbers won't transfer directly, but the two-stage recipe (align frozen, then jointly unfreeze) is a concrete alternative to LightRetriever's "no training on the query side beyond the lookup table" approach.
- Artifacts: code/models promised at github.com/PhilipGAQ/CARE, not verified live in this sweep.

**ScalingNote (Xiaohongshu/RedNote search production system).** arXiv 2411.15766 (Nov 2024).
- Both towers initialized from Qwen2.5-7B, jointly trained, then a "Query-based Knowledge Distillation" (QKD) stage decouples the document tower and distills only the query representation into a much smaller BERT student via MSE + cosine loss.
- Deployed and A/B tested in production (5% of traffic, one week).
- Numbers (student query tower vs. the 7B query tower, R@50 / QPS): BERT-1L (8M params) 49.56% R@50 / 52,205 QPS; BERT-4L (29M) 54.57% / 33,810 QPS; BERT-12L (86M) 54.91% / 19,090 QPS; full 7B tower 55.15% / 408 QPS. The 4-layer/29M student keeps 99% of the 7B tower's R@50 at ~83x the QPS.
- Artifacts: no open code/checkpoints found — internal production system, paper only.
- This is the strongest "industry has already answered the brief's question, partially" data point in the sweep: quality degrades gracefully and mostly saturates well before the query tower goes to zero, which is a useful prior for where your own quality-vs-compute frontier should bend.

**Query Encoder Distillation via Embedding Alignment (Guest et al.).** arXiv 2306.11550 (2023).
- Student initialized from a subset of teacher layers, then trained to minimize Euclidean distance to teacher query embeddings (unsupervised, no labels).
- Document tower stays the full 12-layer teacher (`msmarco-bert-base-dot-v5`); query tower compressed to 1/2/4 layers.
- BEIR retention: 1-layer 86.1%, 2-layer 92.5%, 4-layer 96.2%, vs. a supervised 6-layer baseline's 96.6%. >5x inference speedup reported.
- Artifacts: **yes** — github.com/Guest400123064/distill-retriever.
- Same family as LEAF and ScalingNote but older, smaller-scale, and with layer-truncated-BERT rather than a purpose-built tiny architecture. Useful as the "cheap but still transformer" anchor point on the spectrum, one rung above LightRetriever's lookup table.

**HAKARI-Bench: A Lightweight Benchmark for Comparing Retrieval Architectures and Efficiency Settings under Unified Conditions.** Tateno, arXiv 2606.22778 (2026).
- Directly relevant to M2 of the project plan: a benchmark built to compare BM25, dense, sparse, late-interaction, and reranker families — including their efficiency variants — under identical conditions, using compressed "Nano-set" versions of 35 retrieval benchmarks (551 tasks, 43 languages) for cheap, repeatable comparison.
- Headline finding is explicitly "no universal winner — best architecture depends on scope," i.e., not a single frontier number but a benchmark harness.
- Artifacts: **yes** — github.com/hakari-bench/hakari-bench.
- Worth reusing directly for M2 (benchmark harness + small-model baselines): it may save you from building your own BEIR-subset harness from scratch, since it's designed exactly for "compare architectures under unified efficiency conditions."

## 3. Static-embedding + projection alignment ("tiny adapter, not a model")

**Model2Vec.** MinishLab, github.com/MinishLab/model2vec, ongoing project not a single paper.
- Distills a sentence transformer into a static (lookup-only, no transformer) embedding model by forward-passing the vocabulary once and optionally applying dimensionality reduction (PCA) — data-free distillation.
- `potion-retrieval-32M`: 81.69% of `all-MiniLM-L6-v2`'s retrieval score (35.06 vs. MiniLM's benchmark score) while being "orders of magnitude" faster — this is a same-architecture-family static baseline, not asymmetric (query and doc both use the static model).
- Directly relevant as the brief's own listed "static embedding models used symmetrically" candidate route — confirms it's a real, maintained, benchmarked option, not just a research idea.
- Artifacts: **yes**, actively maintained, MIT-licensed.

**vec2vec — Harnessing the Universal Geometry of Embeddings.** Cornell, arXiv 2505.12540, NeurIPS 2025.
- Unsupervised translation between two embedding spaces with **no paired data and no shared encoder access**, built on the "Strong Platonic Representation Hypothesis" (independently trained encoders converge to a shared latent geometry). Learns an adversarial map into and out of a universal latent space.
- Not retrieval-specific and not evaluated on retrieval quality retention numbers in what this sweep could extract — it's a representation-geometry paper, and the headline claim (high cosine similarity between translated and true embeddings) is a much weaker bar than nDCG@10 parity.
- Artifacts: **yes** — github.com/rjha18/vec2vec.
- **[SPECULATION]** Its practical use for this project would be aligning an off-the-shelf query encoder's space to a document encoder's space post-hoc without training pairs — genuinely novel angle if it works for retrieval, but no retrieval-benchmark evidence was found; treat as an unproven idea, not a validated technique.

**mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations.** arXiv 2510.02348 (2025).
- Follow-up to vec2vec replacing the adversarial neural translator with a linear/orthogonal transformation — much cheaper to fit. This is the closest hit to the brief's literal ask ("train a tiny adapter, not a model," "linear projection align embedding spaces").
- Quantitative retrieval-quality numbers did not extract cleanly from the PDF fetch in this sweep; the paper's own framing is about matching vec2vec's alignment quality at a fraction of the compute, not about beating a supervised baseline. **Needs a direct re-read before citing a number.**
- Artifacts: not confirmed in this sweep — check the paper's repo directly.

**EGA: Adapting Frozen Encoders for Vector Search with Bounded Out-of-Distribution Degradation.** arXiv 2605.05674 (2026).
- Trains a lightweight residual adapter (zero-initialized residual + hypersphere projection + local triplet supervision) on top of a frozen encoder for vector-search-specific adaptation, with a theoretical bound on how much OOD queries can degrade.
- This is the shallow-adapter idea done rigorously, but for adapting one encoder to a new task/domain, not for bridging a cheap query encoder to an expensive document encoder's space — read the exact setup before assuming it transfers to the asymmetric case.
- Artifacts: **yes** — github.com/hpdic/EGA.
- Numbers vs. frozen baseline and full fine-tuning did not extract cleanly from the PDF in this sweep; re-fetch the results table directly before quoting.

**LEAF (arXiv 2509.12539)** — see Section 2. Directly relevant here too: it is evidence that a learned mapping into a frozen teacher's space, trained with pure embedding regression (the "tiny adapter" spirit, even though the model itself is a small transformer rather than a literal linear map), retains 96-98% of quality. **This is the strongest published evidence for theme 3's premise** — a literal linear/MLP-only adapter (vs. LEAF's small transformer) has not been shown at comparable quality in anything found in this sweep.

**Net assessment for theme 3:** the "train a tiny adapter, not a model" framing has real published support at the small-transformer scale (LEAF, query-encoder-distillation), and real but not retrieval-validated support at the literal-linear-map scale (vec2vec/mini-vec2vec, unproven for retrieval; EGA, adapter for a different problem). No paper in this sweep trains a linear or shallow-MLP map from an off-the-shelf static embedding (Model2Vec-style) into a frozen large-model document space and reports retrieval quality retention — **that specific experiment looks like open ground**, not something to expect literature to have already answered.

## 4. On-device / edge retrieval systems (2024-2026)

**MobileRAG.** arXiv 2507.01079 (Jul 2025).
- `EcoVector`, a mobile-friendly ANN search algorithm, plus Selective Content Reduction (SCR) to shrink LLM context after retrieval. Claims to beat conventional vector search and RAG on latency, memory, and power on-device.
- Exact latency/memory/mAh numbers did not extract cleanly from the PDF fetch in this sweep — worth a direct read for the M4/M5 latency-measurement stage since it's the closest published on-device vector-search latency benchmark found.
- Artifacts: unclear from this sweep, check the paper directly.

**Pocket RAG (first-aid, offline mobile).** arXiv 2602.13229 (2026). Notes that dense RAG on 2023-flagship phones had >14s prefill latency, "unsuitable for time-sensitive use," and that Android thermal throttling causes unstable performance over sustained use — a real-world caveat for any on-device benchmark that only measures a single cold query.

**PerCache: Predictive Hierarchical Cache for RAG Applications on Mobile Devices.** arXiv 2601.11553 (2026). Hierarchical query-answer + KV cache with predictive pre-population; 34.4% latency reduction over the best baseline in their tests. Cache-centric, not query-encoder-centric — orthogonal to this project's approach but a reminder that caching may dominate over query-encoder cost in real deployments where query repetition is high.

**EmbeddingGemma (Google, Sept 2025).** 308M-parameter on-device embedding model, MRL-trained (768/512/256/128 dims from one model), quantization-aware trained to sub-200MB RAM, <15ms embedding latency on EdgeTPU for 256-token input. Top-ranked open multilingual embedding model under 500M params on MTEB at time of release.
- This is the strongest "just run a small real transformer on-device" baseline found and should be in the M2 small-model baseline set alongside bge-small/e5-small/MiniLM/gte-small — it's newer and specifically built for on-device use, unlike the brief's listed 2021-2023-era small models.
- Artifacts: **yes** — `google/embeddinggemma-300m` on HF.

**Browser-side inference (transformers.js v4, WebGPU).** Not a paper — an engineering reality check. Client-side embedding via ONNX + WebGPU/WASM is mature and widely used (>1M monthly users of the library), with int8/q4 quantization bringing model download size from ~1.3GB (fp32) to ~336MB (int8). This is the "small conventional model in the browser" alternative the brief needs to benchmark against — it downloads and runs a real transformer client-side (unlike LightRetriever's zero-transformer query path), so the comparison is model-download-size-and-cold-start vs. LightRetriever's near-zero cold start, not just quality-vs-latency.

**Net assessment for theme 4:** no paper found runs a LightRetriever-style zero-transformer lookup query encoder specifically on a phone/browser and reports quality-vs-latency; the on-device RAG literature is entirely built around running small-but-real transformers on-device (EmbeddingGemma, MobileRAG, transformers.js), which makes the brief's zero-compute-query framing more distinctive than the existing on-device literature covers, but also means there's no existing on-device latency/cold-start measurement for the LightRetriever-style approach specifically — M5's Qdrant Edge prototype would be closing a real gap, not confirming known results.

## 5. Query understanding without transformers: measured failure modes

**On the Theoretical Limitations of Embedding-Based Retrieval (LIMIT).** Google DeepMind, arXiv 2508.21038 (Aug 2025).
- Formal result: for any fixed embedding dimension d, there exist combinations of relevant documents that no single-vector embedding can return as top-k for any query — a communication-complexity-style bound, not an empirical training artifact.
- LIMIT benchmark (50,000 docs, 1,000 queries, 2 relevant docs/query) stress-tests this: best dense models score under 20% recall@100 (Promptriever-Llama3-8B 18.9%, GritLM-7B 12.9%, Gemini Embed 10.0%, E5-Mistral-7B 8.3%, Qwen3-Embed 4.8%, Arctic-Embed-Large 3.3%) while **BM25 scores 93.6%** on the same benchmark.
- This generalizes the brief's "bag-of-token-averaging can't distinguish word order" claim into a dimension-independent ceiling that applies to any single-vector encoder, transformer or not — cite this instead of (or alongside) the informal compositionality argument, since it's the sharper and more defensible claim: the limitation is single-vector representation, not specifically the lack of token interaction.
- Artifacts: **yes** — github.com/google-deepmind/limit, LIMIT dataset released.

**Semantic Adapter for Universal Text Embeddings: Diagnosing and Mitigating Negation Blindness.** arXiv 2504.00584 (2025).
- Diagnoses negation blindness on STSB (simple negation) and a new SemAntoNeg benchmark (negation+antonym combinations). Baseline universal embeddings: 77.25% (STSB), 59.22% (SemAntoNeg, 200 samples).
- Fix is a training-free, parameter-free reweighting of embedding dimensions (softmax-weighted by a single hyperparameter) — not a transformer or even a learned linear map — pushing STSB to 81.93% (+4.68 pts) and SemAntoNeg to 73.74-75.08% (+14.5 to +15.9 pts).
- Directly relevant: shows negation failure is partly fixable **without adding any query-side compute**, which cuts against assuming negation requires a heavier query encoder — worth testing this reweighting trick against LightRetriever's lookup-table output directly.
- Artifacts: not stated in what this sweep extracted.

**SparseCL: Sparse Contrastive Learning for Contradiction Retrieval.** Xu, Lin, Sun, Chang, Indyk. arXiv 2406.10746 (Jun 2024).
- Directly uses ArguAna as one of its benchmark datasets — the brief's own example of "long, argumentative queries." Standard similarity search is shown to fail at contradiction retrieval because cosine similarity rewards similarity, not opposition.
- Reports >30% accuracy improvement on synthetic-contradiction MSMARCO/HotpotQA variants over standard dense retrieval, across multiple base architectures, by adding a sparsity-based dissimilarity term to the embedding objective.
- This is close to a direct measurement of the "compositionality/negation breaks bag-of-vectors retrieval, and ArguAna-style queries are where it shows" claim the brief wants characterized — but note it evaluates full dense encoders, not literally query-side token averaging, so it's evidence for the general phenomenon, not a LightRetriever-specific measurement.
- Artifacts: not confirmed live in this sweep, check the paper's repo.

**Training for Compositional Sensitivity Reduces Dense Retrieval Generalization.** arXiv 2604.16351 (2026).
- Empirically shows a tradeoff: training a dense retriever to be sensitive to word order/composition (i.e., to *not* behave like bag-of-words) measurably hurts out-of-domain generalization.
- This complicates the brief's framing that "a conventional small Transformer can do better" at compositionality unconditionally — it suggests small Transformers pay for compositional sensitivity with generalization, so the comparison against LightRetriever should measure both in-domain compositional cases (dog-bites-man style) and out-of-domain generalization, not just the former.
- Artifacts: not checked in this sweep.

**Net assessment for theme 5:** the brief's "compositionality is the weakness" claim holds up and is now backed by a formal result (LIMIT) rather than only an intuitive example. The literature adds two nuances the brief doesn't have: (a) negation specifically may be fixable with a nearly-free reweighting trick, not just a bigger query encoder, and (b) making a small Transformer more compositionally sensitive isn't free either — it trades away generalization. Both are testable cheaply in your own M4 matrix.

## Sources consulted

LightRetriever: arxiv.org/abs/2505.12260, arxiv.org/html/2505.12260v4, github.com/caskcsg/lightretriever, huggingface.co/lightretriever, openreview.net/forum?id=vNEY32I8Y8 (blocked by bot-check), Semantic Scholar Graph API citations endpoint.
Asymmetric encoders: arxiv.org/abs/2204.07120, arxiv.org/abs/2604.10937, arxiv.org/abs/2509.12539 + arxiv.org/html/2509.12539v2, arxiv.org/abs/2411.15766 + html v1, arxiv.org/abs/2306.11550 + html v1, arxiv.org/abs/2606.22778.
Alignment/adapters: github.com/MinishLab/model2vec, huggingface.co/minishlab/potion-retrieval-32M, arxiv.org/abs/2505.12540 (vec2vec), arxiv.org/abs/2510.02348 (mini-vec2vec), arxiv.org/abs/2605.05674 (EGA).
On-device: arxiv.org/abs/2507.01079 (MobileRAG), arxiv.org/abs/2602.13229 (Pocket RAG), arxiv.org/abs/2601.11553 (PerCache), ai.google.dev/gemma/docs/embeddinggemma, developers.googleblog.com EmbeddingGemma post, huggingface.co/docs transformers.js.
Failure modes: arxiv.org/abs/2508.21038 + html (LIMIT), github.com/google-deepmind/limit, arxiv.org/abs/2504.00584, arxiv.org/abs/2406.10746, arxiv.org/abs/2604.16351.

## Caveats on this sweep

- Several numeric extractions came from LLM-summarized PDF/HTML fetches rather than direct table reads (flagged inline above where retrieval was unclean). Before the M1 experiment matrix is finalized, re-pull exact tables for: LightRetriever's dense vs. hybrid BEIR numbers, EGA's OOD bounds, mini-vec2vec's retrieval numbers (if any exist), and MobileRAG's latency table.
- OpenReview reviews for LightRetriever were not retrievable (bot-check page) — worth a manual visit if reviewer critiques matter for the decision report.
- No Google Scholar citation graph was used (not available as a direct tool); Semantic Scholar's graph API was used instead and may undercount citations relative to Google Scholar.
