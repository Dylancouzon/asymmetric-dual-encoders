# What is measured about the swap, and what is only asserted

Sub-agent audit 2026-09-17 over a named-file allowlist. The claim under audit: the query encoder is
interchangeable over one frozen document index, and this is operationally real.

## Measured

| Fact | Number | Source |
|---|---|---|
| Zero query path against the numpy reference | 4.470e-08 | `m11/STATUS.md` |
| Nano parity, max cosine error / min cosine | 1.1548399925231934e-07 / 1.0 | `m14/STATUS.md`, `m21/FASTEMBED.md` |
| The published Stella document graph, given the `s2p_query` prompt, reproduces the torch query path | min-cos 1.00000000, max-abs 8.9e-08 over 8 queries | `m11/STATUS.md` addendum 2026-09-04 |
| Document tower CUDA against CPU parity, so an index built on one holds on the other | min-cos 1.000000, max-abs 9.07e-05 | `m11/STATUS.md` |
| A live swap: one in-memory Qdrant collection, Nano then Zero over the same points | runnable quickstart | `README.md` |
| A fourth query-side configuration over the same vectors: Zero + BM25 under Qdrant DBSF at prefetch 100 | see `EVIDENCE_INDEX.md` | `m12/FINDINGS.md` |

Serving cost, common protocol: three fresh processes per model, batch one, four CPU threads, five
warmups, 20 synthetic 20-word queries, medians over three trials.

| Model | Hydration (cold) | First query (cold) | Warm p50, 20 words | Peak RSS | Model assets |
|---|---:|---:|---:|---:|---:|
| constella-zero | 0.2618 s | 0.3529 ms | 0.1119 ms | 275.4 MiB | 90.1 MiB |
| bge-small | 0.6726 s | 8.2401 ms | 6.8400 ms | 291.0 MiB | 127.6 MiB |
| constella-nano | 0.6907 s | 7.6685 ms | 7.2511 ms | 280.9 MiB | 132.3 MiB |

## Asserted, not measured

- BM25 alone and the Stella query tower as swap options are tabulated, but no single live swap
  transcript covers them the way the Nano/Zero pair is covered.
- The plain-English page's edge pairing of 3.4 ms with Zero and 4.5 ms with Nano inside a 256 MB
  container does not appear in `results/edge_prototype.json`, `results/edge_variant.json` or
  `results/ann_sweep.json`. This matches `m21/BENCHMARKS.md` discrepancy item 12. **Do not cite
  these two numbers in the paper until a committed result file is found.**
- `results/costs.json` is a different and older table over a different model set. It does not hold
  the Zero, Nano and bge-small numbers above.
- `results/ann_sweep.json` covers bge-small and lightretriever-qwen2.5-1.5b, not Zero or Nano.

## The gap a skeptic will name

The quality tables come from separate offline exact-search evaluations against frozen per-query
score files. The runtime swap is shown on a two-document in-memory collection. No file measures a
swap against a large, persistent, already-populated Qdrant collection, in one server lifetime,
with no reload and no reindex call. The shared-index property is established by both encoders
hitting the same frozen target vectors to 1e-7, not by one script querying one on-disk index with
both encoders and diffing the returned point identifiers.

**Proposed fix, owner decision required**: one small registered experiment that builds a persistent
Qdrant collection once from the frozen Stella document vectors of a public dataset, then queries it
with Zero, Nano, the Stella query tower and BM25 in one server lifetime, recording returned point
identifiers, scores and wall-clock latency per encoder. It uses committed vectors, needs no GPU and
no protected data, and it converts the paper's central framing from asserted to measured. Scope is
one script and one result JSON with a frozen method and a compact receipt.

## Where the swap is not free

- **Prompt string.** Stella as a query encoder needs its `s2p_query` prompt. Omitting it silently
  drops the vector to cosine 0.80 against the correct one. Zero and Nano need no prefix.
- **Tokenization.** Stella's shipped tokenizer pads to 512 by default. Without `no_padding()` the
  input carries roughly 500 PAD rows and cosine falls to 0.35.
- **Dtype.** FastEmbed's `mean_pooling` promoted output to float64 for Nano-style models, while
  Zero and the Stella document tower pool inside ONNX and stay float32. The preview branch fixed it.
  Two query paths did not share dtype behavior until then.
- **Score scale.** Swapping dense Zero for Zero + BM25 changes the retrieval operator, not just the
  encoder. convex0 is not implementable in Qdrant; DBSF is the runnable substitute, and the two
  split differently across clean-4 and all-six with no equivalence interval.
- **Fusion depth.** DBSF increases with prefetch depth, and at depth 10 the ranking between DBSF,
  convex0 and RRF inverts.
- **Cost.** Two query encoders that search the same index differ by roughly 65x in warm p50.
- **Tooling.** FastEmbed's built-in `query_embed` does not add Stella's prompt, so the normal query
  path returns a wrong-protocol vector for the symmetric graph.
