# M7 novelty and freshness check (2026-08-25, Sonnet web sweep + Li-LSR full-text read)

Question: is "a dense token→vector lookup table trained end to end against a frozen, off-the-shelf small document encoder" published?

**Verdict: not published as of 2026-08-25.** Re-check before the M7 report ships.

## Directly on-target

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
