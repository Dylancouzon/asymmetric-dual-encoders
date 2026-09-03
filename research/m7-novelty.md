# M7 novelty and freshness check (2026-08-25, Sonnet web sweep + Li-LSR full-text read)

Question: is "a dense token→vector lookup table trained end to end against a frozen, off-the-shelf small document encoder" published?

Two distinguishing axes; a candidate has to fail one of them to leave the claim standing. **(a)** the document tower is frozen and off-the-shelf — not co-trained, not a checkpoint then trained further. **(b)** the query-side rows are **dense vectors** read from a trained lookup table — not scalar term weights, not sparse lexical scores, not a live transformer or a per-query optimization.

**VERDICT WITHDRAWN 2026-09-03. pyNIFE matches both axes and predates the sweep by ten months** (§pyNIFE below). The text under the old verdict is kept because the rest of the survey stands.

## pyNIFE — the claim is defeated (found 2026-09-03)

**github.com/stephantul/pynife**, Stephan Tulkens, PyPI 0.1.0 on **2025-11-03**, MIT, Zenodo DOI 10.5281/zenodo.17512919. Axis (a) passes: the teacher (`mxbai-embed-large-v1`, `gte-modernbert-base`) is off-the-shelf, frozen, and the index is reused unchanged. Axis (b) passes: dense per-token rows in a lookup table, initialized by forwarding each vocabulary token through the teacher, then trained end to end by cosine distillation. Two published models, NanoBEIR nDCG@10 59.2 for both against teachers at 65.6 and 66.34. This is `zero`'s construction, published before M7 started. **Every novelty sentence in the report and the whitepaper has to go**; the defensible claims left are the measurement (six BEIR sets, exact search, frozen comparators, reserved sets, pre-registered statistics against NanoBEIR's 50 queries per set), the artifact constraints (30,522 rows int8 at 93.9 MB, no MS MARCO in the lineage against theirs in tokenizer, documents and queries), and the pair with `nano` on one index.

**Why the sweep missed it:** no arXiv paper, no HF model-card language matching the query families, and the searches were phrased for papers and model releases. A GitHub/PyPI/Zenodo release is a publication. Future freshness passes search PyPI and GitHub by construction, not only arXiv and HF.

**Old verdict, 2026-08-28, superseded:** no published construction matching both axes — but the field moved closer, and the defensible phrasing is "we found none", not "this is unprecedented." The re-sweep ran a freshness pass *and* a deliberate falsification pass (nine adversarial query families, listed below). It found one construction that clears axis (a) outright and misses (b) on the representation, not on the ambition.

## Re-sweep 2026-08-28 — what changed

- **KAHM — Kernel Affine Hull Machines, arXiv 2605.02950** (v1 2026-05-01, v2 2026-06-06). **The nearest miss, and absent from the 2026-08-25 file.** Axis (a) passes outright and is the paper's stated premise: "once a strong teacher representation space and corpus index are fixed, repeated neural query encoding can be replaced by a substantially lighter and analytically explicit estimator." Axis (b) fails on two independent grounds — the query is "a noisy mixture of semantic prototypes weighted by posterior cluster probabilities", a mixture over a shared prototype bank rather than per-token dense rows, and the method is explicitly **"backpropagation-free"**, the reverse of trained end to end. It is also not O(1) at query time: it evaluates geometric quantities and forms a weighted mixture per query. **Any future re-check starts with this paper's citations.**
- **ERA / Efficient Retrieval Adapter, arXiv 2604.03403** (v2 **2026-08-26**, inside the freshness window). Axis (a) passes — "No Parameter Access: ERA requires no access to the base embedding model parameters". Axis (b) fails: the query path is a full strong embedder forward pass (up to Qwen3-Embedding-8B) plus a linear adapter. Opposite optimization target — index reuse across embedder upgrades, not cheap queries.
- **LightRetriever is now v5 (2026-01-30), accepted at ICLR 2026 and KDD 2026.** The old file knew only v2. v5's full text was re-read specifically for a frozen-doc-tower ablation added in a later revision: there is none, so axis (a) still fails and now on the current version. Checkpoints (Llama-3.1-8b, 3.2-1b/3b, Qwen2.5-1.5b/3b/7b) and a finetune-data dataset are released.
- **LEAF has a paper**, which the old file was missing: **arXiv 2509.12539, ACL 2026**, plus a new sibling `MongoDB/mdbr-leaf-mt` (23M transformer student distilled from mxbai-embed-large-v1). Still a transformer forward pass per query — axis (b) fails as the existing LEAF rows do. **Relevant to M9** (renumbered from M8, 2026-08-28), whose comparators are LEAF models.
- **No OpenSearch doc-v4.** v3 is still current and its query side is still "a tokenizer and a weight look-up table" producing **scalar** weights — the same axis-(b) failure as before.
- **DistilVDR** re-verified: frozen teacher (axis (a) passes), 70M transformer student (axis (b) fails), different modality.

**Falsification pass, no on-target hits.** Searched as distinct queries: embedding-bag retrieval; bag-of-token-embeddings retriever; token-embedding-lookup query encoder with a frozen doc encoder; training-free query encoder; amortized query encoding; precomputed query embeddings; linear query encoder over a frozen document encoder; static query embeddings for asymmetric retrieval; industry blog posts and model cards. What surfaced instead: ColBERT-style contextualized bags (full transformer per query), VPRF (averages retrieved-document embeddings, nothing trained), search adapters (ERA's family), Test-Time Compute for Frozen Embedding Models (arXiv 2605.11374 — an LLM writes a program per query, the *opposite* of near-zero query compute), TTT-Embed (arXiv 2608.12569 — a per-query optimization loop with no training phase at all).

**Not re-fetched 2026-08-28** (2023 papers, low volatility): arXiv 2306.11550 and arXiv 2304.01016. Both were transformer-student on the 2026-08-25 read, and nothing suggests axis (b) has changed.

## Directly on-target (as of 2026-08-25)

- **LightRetriever** — arXiv 2505.12260 (May 2025, v2). https://arxiv.org/abs/2505.12260 · https://github.com/caskcsg/lightretriever. Trains a token→embedding lookup table end to end; query = tokenize, lookup, average. But the document tower is the same LLM, co-trained in the same run — not a frozen, pre-existing off-the-shelf small encoder.
- **Li-LSR** — arXiv 2505.01452 (April 2025). https://arxiv.org/abs/2505.01452. Full text read to close the caveat:
  1. Doc encoder co-trained, not frozen: "We start our training from the Co-Condenser checkpoint available on HuggingFace" — a pretrained init, then both towers train together for 150,000 steps.
  2. Query-side lookup is scalar, not a vector: "Li-Lsr learns a static relevance score for each token during training by projecting the output of word embeddings into a scalar value using a simple linear layer": si = log(1 + ReLU[wᵀ E_W(xᵢ) + b]). KL-divergence distillation loss (best variant).
  3. BERT/Co-Condenser backbone; BEIR aggregate 51.02 (Big model); no per-dataset breakout for our six.
  4. No code or model release found.
  Both distinguishing axes (frozen off-the-shelf doc side, dense vector rows) fail here.

## Adjacent (frozen doc side, but transformer students — not tables)

- **Query Encoder Distillation via Embedding Alignment** — arXiv 2306.11550 (2023). https://arxiv.org/abs/2306.11550. Closest structural precedent: frozen document encoder, query-side student trained via MSE alignment to teacher query embeddings. Student is a 1/2/4/6-layer BERT slice, not a lookup table. Confirms the frozen-doc/trained-query pattern is established.
- **KALE** — arXiv 2304.01016 (2023). https://arxiv.org/pdf/2304.01016. Post-hoc pruning + KL alignment of a query encoder after bi-encoder training. Architecture details unresolved from the fetch (PDF unreadable); abstract confirms compression/alignment of a query encoder, no lookup table.
- **NanoVDR** (arXiv 2603.12824, 2026) and **DistilVDR** (arXiv 2608.10636, 2026) — visual document retrieval. Frozen large VLM teacher indexes docs; distilled 69–70M text-only transformer student encodes queries. Same asymmetric pattern, different modality, transformer student.
- **SPAR** — dense lexical embedding trained to imitate BM25/uniCOIL, concatenated to a dense embedding. Augments rather than replaces the query-side transformer.

## Checked and empty

- LightRetriever citations (Semantic Scholar page 404'd; search found no citing papers as of 2026-08-25).
- Model2Vec / potion: symmetric static distillation only; no asymmetric or frozen-doc-tower usage in models or docs.
- sentence-transformers static-retrieval-mrl-en-v1: trained symmetrically (MatryoshkaLoss + MNRL on 80M pairs, both sides the same static table).
- "tied embedding retrieval", "embedding bag retriever", DensePhrases-style tricks: no hits on our construction.
- Freshness sweep post-2026-08-24: no LightRetriever v2/successor, no OpenSearch neural-sparse doc-v4, no MongoDB LEAF follow-up, no new tiny query-tower or edge-distillation release. Searched by name and via HF August 2026 release notes.

Sources: [LightRetriever](https://arxiv.org/abs/2505.12260) · [LightRetriever GitHub](https://github.com/caskcsg/lightretriever) · [Li-LSR](https://arxiv.org/abs/2505.01452) · [Query Encoder Distillation](https://arxiv.org/abs/2306.11550) · [KALE](https://arxiv.org/pdf/2304.01016) · [NanoVDR](https://arxiv.org/html/2603.12824v1) · [DistilVDR](https://arxiv.org/html/2608.10636v1) · [Model2Vec](https://github.com/MinishLab/model2vec) · [static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1)
