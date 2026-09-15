# M14 — nano release

Previously M13; moved on 2026-09-10 to make room for cloud execution. Blocked on M13's frozen
candidate, release decision and serving validation. Zero and its document tower already shipped
in M11; they are not republished to close this milestone.

## Deliverables

1. Release `constella-nano` if authorized by the measured outcome: frozen bytes, releasability
   checks, MIT licence, required source attributions and stella-exposure disclosure.
2. ONNX export of the trained artifact with reference/stock-FastEmbed parity and measured costs.
   Reuse `m11/CODEMAP.md`; tokenizer limits, padding and pooling are part of correctness.
3. One clean upstream FastEmbed PR adding all three model entries and reference-derived canonical
   vectors, branched from upstream main. The existing `add-constella-models` branch also carries
   the #703 padding fix and must not be merged wholesale. The PR waits for nano (Dylan's ruling).
4. Model card with measured results and qualified claims: nano shares stella's index at roughly
   bge-small query cost; it is not the near-zero-compute path. Competitive tables belong in M15.

Publishing under another organization or offering the padding fix separately remains an owner
choice at release time. A failure to earn release leaves an honest M13 measurement and M15 paper;
it does not require shipping a model or changing the bars.

Historical release mandate/rulings:
`research/archive/m10-cleanup-2026-09-10/instructions-m13.md`.

Latest user clarification (2026-09-15 UTC): the different LLM handling M14 owns
Hugging Face upload and release verification, as well as the FastEmbed PR. Start
from the completed M13 closure and its release handoff; do not depend on chat history.

## Reserved-four inheritance and Zero comparison — owner ruling 2026-09-15

M13's completed six-set decision triggered the registered descriptive reserved-four stage
(FEVER, DBpedia Entity, CQADupStack Android and CQADupStack English), but the production
executor stopped **before opening any reserved payload**: the required approximately 10M
Stella passage vectors do not exist and the protected-access allowlist/executor is not yet
implemented. The six-set verdict stands. Verify from the M13 closure/handoff that reserved
access is still unspent; never infer this merely from an `INCOMPLETE_RESERVED` label.

If the reserved access remains unspent at M14 execution time, M14 inherits this unfinished
descriptive evaluation. Before opening any of the four datasets, make and push a dated,
pre-observation registry/ledger amendment adding the released dense Zero system to the frozen
roster. The intended row is `constella-zero` as query encoder paired with the same pinned Stella
document tower used by M13 Nano. Evaluate these four systems in the one reserved transaction:

- M13 Nano + pinned Stella document tower
- released Zero + the same pinned Stella document tower
- BGE-small symmetric
- LEAF asymmetric system

Zero must reuse the newly produced Stella document vectors; adding it does not authorize a
second corpus-scale Stella encode. Its incremental work is query encoding and retrieval/scoring.
All reserved-four rows remain descriptive (`alpha = 0`), do not alter the completed six-set
claims, and must disclose FEVER's registered double-contamination caveat. Do not substitute the
released Zero+BM25 fusion for the dense Zero row; a fusion row may be reported only as an
additional, clearly separated descriptive result if its inputs already exist and doing so does
not consume another protected access.

If evidence shows that the reserved payload was already opened before this amendment was pushed,
do not reopen it to add Zero. Preserve the one-access rule and document Zero's omission instead.
