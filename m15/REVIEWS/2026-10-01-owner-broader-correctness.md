# Broader Constella design claims: correctness review, 2026-10-01

**Conclusion:** The evidence supports foregrounding Constella as one document index with three selectable query-compute tiers. That is a useful positive design capability for search engineers. The paper can make it prominent while attributing the underlying asymmetric/static construction to prior work. Neither “zero retrieval latency” nor a measured production hot-swap guarantee is supported.

## What is demonstrated

**One populated index accepts all three query paths.** E2 starts a native Qdrant server with disk storage, builds a collection from the frozen Stella document vectors, waits until all points are HNSW-indexed, and queries that collection with Zero, Nano, and Stella vectors. It does not rebuild, reload, update, or replace the collection between those encoder batches. This runs on 57,638 FiQA documents and a one-million-passage MS MARCO diagnostic. Each quantization configuration gets its own collection; all three encoders share that collection before the benchmark deletes it. The server remains running across those batches and configurations.

This is stronger evidence than the two-document in-memory README example. The early gap recorded in `m15/HOTSWAP.md` is therefore substantially closed by E2. Its description of an absent large populated-collection measurement is historical, not the current evidence state.

**Per-request encoder choice is supported as a capability.** The serving interface receives an aligned query vector; the document collection has no encoder-specific identity or mutation in the query call. Selecting a loaded encoder and then issuing the same vector query makes the choice available on each request. This enables application-selected quality/compute tiers without re-embedding the corpus, changing the document schema, or maintaining separate dense indexes for these three query encoders. Automatic routing is one possible policy, not a requirement for the capability.

**There is a nearly negligible warmed query-encoding tier.** E1 measures Zero at 0.04396 ms p50 for 5–12-word queries on the M5 Pro, versus 2.254 ms Nano and 31.582 ms Stella, under its stated batch-one CPU protocol. Zero runs tokenization, lookup, pooling, and normalization without a transformer forward pass. “Near-zero query-encoding cost” or “44-microsecond query encoding” is supportable with the measurement context. Cold hydration and first-query costs are separate: E1 records approximately 0.215 s and 0.208 ms for Zero.

**Quality remains useful but changes with the selected tier.** The existing BEIR-15 exact rows give 0.4572 Zero, 0.5081 Nano, and 0.5614 Stella: 81.4% and 90.5% retention for the students, as ratios of macros. Those are descriptive workload aggregates, not a per-query guarantee. A user may select a faster tier even though no learned router can perfectly identify the best tier for every request.

## A literal one-index deployment example already exists in the receipts

For a clear illustration, use rows from one quantization configuration. The independently fastest settings in the existing main text sometimes use different quantization collections, so they should not be presented as one unchanged physical index configuration.

On the **same binary-quantized one-million-passage collection**, choosing the fastest recorded setting for each tier that stays within 1% of that tier's own exact quality gives:

| Query path | Request `ef` | Rescore oversampling | ANN nDCG@10 | Encode + search p50 |
|---|---:|---:|---:|---:|
| Zero | 512 | 2 | 0.6116 | 1.113 ms |
| Nano | 256 | 4 | 0.6893 | 2.484 ms |
| Stella | 128 | 1 | 0.7226 | 21.335 ms |

These rows come directly from `results/m15_e2_ann_msmarco1m.json`; the restriction to binary quantization and this presentation are an exploratory selection from the registered sweep, not a newly registered contrast. The index is shared; `ef` and oversampling are request options. The retained quality targets are relative to each encoder's own exact score, not equal absolute quality. The positive-preserving subset is not full MS MARCO.

This is a concrete positive result: the same index provides an approximately one-millisecond table path, a roughly two-and-a-half-millisecond compact-transformer path, and a roughly twenty-one-millisecond original-teacher path on this measurement. The near-zero encoding tier still pays for retrieval. The exact E2 latency estimand is the sum of per-query encode and search durations measured in the same process but in separate phases; it is not a stopwatch around an interleaved production request handler.

## Necessary claim boundaries

- **No dedicated request-interleaving test.** E2 pre-encodes queries, then searches all Zero queries, all Nano queries, and all Stella queries for each search setting. It demonstrates repeated changes of encoder batches against the same live collection, not a randomized request-by-request switching or concurrency benchmark. Neither result JSON contains an interleaving transcript, collection-content hash before/after switching, or measured switch overhead. Do not call these existing measurements a production hot-swap stress test.
- **Loaded model selection differs from model loading.** The capability assumes the selected encoder is available. E2 creates the encoder objects before timing; it does not promise zero model hydration, memory, cold-start, or network cost. Keeping all tiers resident may have a different memory requirement from serving Zero alone.
- **Compatibility is specific.** These students were aligned to the pinned Stella document space. Equal dimensionality alone does not make arbitrary encoders interchangeable. Prompt, normalization, pooling, and tokenizer handling still matter. The parity numbers in HOTSWAP/PAPER compare each served implementation with its own reference; they do not mean Zero, Nano, and Stella produce identical query vectors.
- **BM25/fusion is an additional retrieval capability.** Dense Zero/Nano/Stella share one dense field. Adding BM25 requires the sparse field/index and fusion configuration; it is not merely another aligned dense query encoder.
- **Prior art limits architectural novelty, not the importance of explaining the design.** The current literature discussion already credits query-side distillation and static/contextual fallback. The empirical contribution can be how much quality, encoding cost, and search cost the three interchangeable tiers actually deliver, and when that flexibility is viable. Do not claim architectural priority or a quantified corpus-migration cost saving.

## Recommended positive framing

A safe central statement is:

> Constella makes query compute a choice that can change from request to request while one document index stays fixed. We implement a lookup-table tier, a compact-transformer tier, and the original teacher query path, and measure their quality and serving costs over the same document vectors. The table makes encoding almost negligible on the measured CPU; retrieval still has a cost.

This gives the reader a capability to understand before the teacher-choice, training, and ANN results explain its consequences. The teacher screen then addresses which spaces support a useful table before indexing; build costs address constructing compatible students; the ANN study measures what the available tiers cost in the engine. It is scientifically useful to foreground this even though the construction is established. A paper does not have to hide its useful design premise because it is not the first to propose it.

No new experiment is necessary for the bounded capability claim above. If the paper insists on a stronger measured interleaving claim, the missing evidence is a small deterministic alternating-request check on an existing collection with unchanged collection state, not another quality evaluation or model-training run. Do not present that optional check as already completed.

## Access and verification

Opened `m15/HOTSWAP.md`, `README.md`, current `m15/PAPER.md`, the E2 section of `m15/MEASUREMENTS.md`, `m15src/e2_ann.py`, `m15src/encoders15.py`, `results/m15_e2_ann_fiqa.json`, `results/m15_e2_ann_msmarco1m.json`, and the summary of `results/m15_e1_latency.json`. A filename-only inventory found no `results/m15_e2_system.json`; the two ANN receipts are the actual outputs. Checked that the SHA-256 of the inspected E2 script exactly matches both committed receipts. Read-only CPU selection of existing rows produced the binary-collection example above. No benchmark, training, raw protected content, `work/` content, or collection operation was run. This review file is the only edit.
