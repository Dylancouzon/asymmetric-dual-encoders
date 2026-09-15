# M21 status — research-preview polish

**Closed 2026-09-15.** Opened and executed the same day for the public research-preview
introduction on 2026-09-16. No new measurement, no new training, no protected access. Reserved
four and BEIR-18 remain **UNSPENT** and belong to M20.

M20 inherits: the unspent reserved-four transaction, BEIR-18, the official Nano release, a
complete upstream-suite run on storage with room, and the FastEmbed PR plan in `m21/FASTEMBED.md`
— a registrations-only PR satisfying clause 6, plus two owner-directed bug-fix PRs that M20's
mandate should record explicitly.

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

Both passed every registered threshold, differing by one float32 ULP, and B was selected. Review
then showed that **both were wrong for float16**: either way the narrowing happened before
`normalize()`, and a fully attended 1024-d float16 vector of 10.0 squares to 102400, past
float16's 65504 maximum, so the norm became `inf` and every component returned exactly 0.

The shipped fix therefore narrows at the **post-processing boundary**, after `normalize()`, and
leaves `mean_pooling` byte-identical to upstream. Final M14 parity is tighter than either variant:
minimum true cosine **1.0** across Torch, direct ORT and FastEmbed, maximum absolute error
1.1548399925231934e-07. Tests cover float32, float16 and float64 through both pooled families plus
the overflow case directly.

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
- FastEmbed download-free suites: **23 passed** (`test_common`, `test_preprocessor_utils`,
  `test_custom_models`, `test_attention_embeddings`), including the new dtype coverage.
- Pre-fix versus post-fix on two affected upstream models (`all-MiniLM-L6-v2`,
  `paraphrase-multilingual-MiniLM-L12-v2`): dtype float64 to float32, maximum absolute difference
  **2.384e-08**, norms unchanged. The behaviour change is dtype only.
- The full upstream suite was **not run to completion**. It re-downloads roughly 10 GB of unrelated
  model artifacts into a tmpfs and drove the box into memory pressure; the targeted checks above
  cover the changed code. A complete green suite on a machine with room remains M20's to record
  before the upstream PR.

## Closing review

Sol reviewed the whole milestone at `124155d`: **DO-NOT-CLOSE**, no P0, three P1 and one P2, all
documentation bookkeeping created by things that changed after the documents were written. It
confirmed the substantive claim limits are intact and found no protected access, new measurement,
renumbering, or M15/M20 work in the M21 changes. All four are fixed:

| finding | fix |
|---|---|
| P1 — `m21/FASTEMBED.md` still recommended the superseded float16-breaking fix, called the branch unpushed at four commits, and read as conflicting with M20's registered single-PR clause | Marked the variant decision superseded with the shipped post-normalize fix beside it; updated to the pushed five-commit state; stated the dtype PR must be `a4452ac` **plus** `47a5090`, never the former alone; recorded that the registrations-only PR satisfies clause 6 and that the two bug-fix PRs are an owner-directed addition for M20's mandate to record |
| P1 — README, PROJECT_STATUS and the plain-English page called `6bb167dc` "the published revision" without distinguishing weights from card | All three now label `6bb167dc` as the frozen **weights** revision and note the card has been revised since |
| P1 — `m21/BENCHMARKS.md` still described fixed items in the present tense | Items 3-11 marked RESOLVED with their dispositions; the section states it audits the pre-M21 text and is retained as the record. Items 1 and 12 remain carried limitations |
| P2 — `ROADMAP.md` preamble still called M19 the current successor | Names M20 as the next execution milestone; M19 is a reviewed plan awaiting its own session |

## Publication

The three cards were published 2026-09-15 as card-only commits: `README.md` was the single file
uploaded to each repository, so no model bytes changed and every published weight keeps its
identity. Each upload was verified by re-downloading the card at the new revision and comparing
SHA-256 against the local bytes; all three matched, file counts were unchanged and all three
repositories remained public.

| model | card revision | files |
|---|---|---:|
| `DylanCouzon/constella-nano` | `64c405c3` (weights unchanged at `6bb167dc`) | 8 |
| `DylanCouzon/constella-zero` | `0e9cd89e` | 11 |
| `DylanCouzon/stella-en-400M-v5-doc-onnx` | `244be9e0` | 8 |

## Review

Astra reviewed the M21 plan before execution (NO-GO as written, no P0; two findings folded in:
the README needed executable verification rather than a prose refresh, and the plain-English page
carried two factual errors about Zero's construction and M9's cause).

Astra then reviewed the implementation at `d1b4063`: **NO-GO**, no P0, four P1 and two P2. Its
access log stayed inside the brief's allowlist. Dispositions:

| finding | disposition |
|---|---|
| P2 — dtype fix broke float16 | **Fixed** in fork commit `47a5090`. Reproduced first: a fully attended 1024-d float16 vector of 10.0 squares past float16's 65504 maximum, so the norm became `inf` and every component returned exactly 0. Narrowing moved to the post-processing boundary, after `normalize()`; `mean_pooling` keeps upstream's float64 accumulation untouched. Tests now cover float32/float16/float64 through both pooled families plus the overflow case. M14 parity improved to minimum true cosine 1.0 on all three comparisons |
| P1 — fusion reproduction conditions dropped | **Fixed.** The `bm25s` Lucene-defaults note and the self-exclusion-before-truncation condition are restored beside the fusion table in the Zero card |
| P1 — "1024-d is four times the size of 384-d" | **Fixed.** The claim was wrong (1024/384 ≈ 2.67); the unsupported multiple is removed |
| P1 — document card example needs an undeclared dependency | **Fixed.** The Sentence Transformers example now states that it needs `sentence-transformers torch`, which the FastEmbed install line does not provide |
| P1 — Nano absolutes lack a source trace | **Resolved, not a defect.** The published values are the equal-weight means of the committed per-query rows in `results/m10_final_scores/<dataset>.json` (`system = "nano-dense"`), verified by reproduction: `scifact.json`, n=300, mean 0.721097. Recorded in `m21/BENCHMARKS.md` |
| P1 — per-number pointers for the retained M9/M10/M17/M18 narrative figures | **Recorded as debt**, not remediated. Those figures are sourced to the `FINDINGS.md` files the page already cites. Per-number tracing of historical narrative belongs to M15's paper evidence pass, not to a one-day preview polish |
| P1 (re-review) — fusion caveat over-applied | **Fixed.** The restored wording said "to reproduce either row" and then applied the prefetch-100 self-exclusion to the convex0 row, which uses prefetch 1000. The `bm25s` condition now covers both rows; the self-exclusion is attributed to the DBSF row only |
| P2 — preview branch carries the padding fixes | **Accepted deviation**, owner-endorsed. The branch every card installs should not hand users a FastEmbed that crashes on `thenlper/gte-base` mixed batches. The three-PR split in `m21/FASTEMBED.md` keeps them separable for M20 |

Sol re-reviewed the fixes (alternating reviewers so the model that proposed a fix does not certify
it): the dtype change is **CONFIRMED** — `mean_pooling` is byte-identical to upstream `0dab99c`
(both blob `60d229b`), both pooled families return the graph dtype for float16/float32/float64 and
bfloat16, float32 narrowing stays within 0.5 float32 ULP over a random 37x29x1024 check, single-
token and fully-masked rows behave as upstream, and both new tests fail against the intermediate
implementation rather than being tautological. It raised one P1 on card wording, fixed above, and
confirmed both recorded debts as non-blocking. **GO** after that fix.

Independent of the reviews, pre-fix and post-fix embeddings were compared on two affected upstream
models (`all-MiniLM-L6-v2`, `paraphrase-multilingual-MiniLM-L12-v2`): dtype changes float64 to
float32, values differ by at most 2.384e-08, norms unchanged. The behaviour change is dtype only.
