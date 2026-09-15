# M21 status — research-preview polish

Opened and executed 2026-09-15 for the public research-preview introduction on 2026-09-16.
No new measurement, no new training, no protected access. Reserved four and BEIR-18 remain
**UNSPENT** and belong to M20.

## Outcome

| deliverable | result |
|---|---|
| Nano float64 | Root-caused and fixed in FastEmbed. Nano now returns normalized 1024-d fp32 natively |
| FastEmbed integration | One branch `constella-research-preview` off current upstream main, pushed to the fork |
| Canonical numbers | `m21/BENCHMARKS.md`, every figure traced to a committed file, with a discrepancies audit |
| Model cards | Nano 256→215, Zero 277→205, document tower 162→131 lines; one voice, one install line |
| README | 159→122 lines; quickstart executed end to end |
| Plain-English page | Corrected in place at its original length; five factual errors fixed |
| PROJECT_STATUS | Refreshed to 2026-09-15 |

## The float64 defect

`fastembed/common/utils.py:mean_pooling` expanded an already-`int64` attention mask, cast it to
`int64` again, and multiplied it by the fp32 token embeddings. NumPy promotes `float32 * int64` to
`float64`, and nothing narrowed the result back, so the pooled and normalized vector left FastEmbed
as float64. This was never Nano-specific: it affected all 16 models routed through `mean_pooling`
(10 `PooledNormalizedEmbedding`, 6 `PooledEmbedding`). Zero and the document tower pool inside
their ONNX graphs and were unaffected, which is why one model family returned two dtypes.

Two fixes were implemented and measured against the frozen Torch reference on all ten M14
length-stratified fixtures including the 511/512/513 boundary:

| variant | vs Torch, min cosine | vs Torch, max abs error |
|---|---:|---:|
| A — cast the mask to the input dtype (fp32 accumulation) | 1.0 | 1.1548399925231934e-07 |
| B — keep float64 accumulation, cast the returned array | 0.9999999403953552 | 1.0058283805847168e-07 |

Both pass every registered threshold; they differ by one float32 ULP. **Variant B was selected**:
it is the smaller compatibility change for the 16 affected models, preserves their existing
summation precision, and concedes the strongest objection a maintainer can raise — that float64
accumulation over 512 tokens may be deliberate. A regression test asserts the returned dtype and
fails against the unfixed implementation.

M14's card text documenting the float64 promotion and instructing a manual `.astype(np.float32)`
was correct when written and is now obsolete; it has been removed rather than reworded.

## FastEmbed branch

`constella-research-preview`, based directly on upstream `0dab99c`, pushed to
`Dylancouzon/fastembed`. Four commits: the fixed-padding-length fix, the eager `pad_token` default
fix, the mean-pooling dtype fix, and the three native registrations. All three cards name this one
branch; the earlier split between `add-constella-models` and `m14-constella-preview` is retired.

M20 inherits a three-PR upstream plan (padding fixes / dtype fix / model registrations), kept
separate so that reviewing new models does not require reviewing two behaviour changes.
Detail, measurements and expected reviewer pushback: `m21/FASTEMBED.md`.

## Evidence discipline

`m21/BENCHMARKS.md` is the single public table set. Numbers were copied, never recomputed or
re-partitioned. Absolute per-dataset scores for bge-small and LEAF are **not published**: they
exist only as frozen per-query comparator vectors, and deriving aggregates from them would be
re-deriving a statistic the registration never computed. Nano's absolute rows and the registered
deltas are published instead (owner decision, 2026-09-15).

The audit found eight discrepancies in the public text, all fixed: two operators both called
"fused" without naming convex0 or DBSF, unresolved contrasts described as "statistical ties", the
200,000,000 nominal dose presented as executed against the actual 199,999,721, stale in-progress
build numbers, and Zero cost/size figures mixed across incompatible protocols.

## Verification

- `m14/verify_card.py` **PASSED** against the staged cards and the new branch: `dtype float32`,
  4,096 bytes per query vector, no manual cast, Hub access refused, `add_custom_model` refused,
  both query encoders ranking correctly against one Qdrant collection.
- README quickstart executed end to end; Nano and Zero both queried the same Stella index.
- Zero and document-tower publish-card gates passed after `REPO_ID` substitution.
- FastEmbed focused changed-code tests passed. Full upstream suite results: see below.

## Review

Astra reviewed the M21 plan before execution (NO-GO as written, no P0; two findings folded in:
the README needed executable verification rather than a prose refresh, and the plain-English page
carried two factual errors about Zero's construction and M9's cause). The implementation review is
recorded below.
