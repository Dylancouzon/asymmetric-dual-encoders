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

Latest user clarification (2026-09-15 UTC): a different session from the one executing M13 owns
M14, including Hugging Face upload and release verification and the FastEmbed PR. Start from the
completed M13 closure and its release handoff; do not depend on chat history.

## Reserved-four inheritance and Zero comparison — owner ruling 2026-09-15

M13's completed six-set decision triggered the registered descriptive reserved-four stage
(FEVER, DBpedia Entity, CQADupStack Android and CQADupStack English). M13 closed **before opening
any reserved payload** and delivered a tested base executor for the originally registered three
systems; the approximately 10M Stella passage vectors still do not exist. The six-set verdict
stands. Verify from the M13 closure/handoff that reserved access is still unspent; never infer
this merely from an `INCOMPLETE_RESERVED` label.

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

Later user clarification (2026-09-15 UTC): M14 also owns the A100 execution of the
already-triggered reserved four. First make the Zero amendment above, extend and test the pushed
M13 base implementation without touching protected data, then complete the one zero-alpha report
before public release. Preserve M13's original `INCOMPLETE_RESERVED` result and the post-trigger
ownership disclosure (R19).

## Owner ruling R20 — research preview rescope, 2026-09-15 (Dylan)

**This ruling supersedes the two clarifications above and the publication gate in
`m14/HANDOFF.md` for M14 only.** It changes release scope and sequencing. It changes no measured
result, registered constant, bar, licence or access rule, and it spends no protected access.

M14 becomes a **research preview release** on Dylan's own Hugging Face account and a custom
FastEmbed branch, targeted for 2026-09-16. Everything that requires protected access, upstream
publication or broad descriptive evaluation moves to M20 (`instructions-m20.md`):

| moved to M20 | stays in M14 |
|---|---|
| Reserved-four A100 transaction (FEVER, DBpedia-entity, cqadup-android/english) | Frozen-byte restore and hash verification |
| Dense-Zero reserved roster amendment and executor extension | Release staging and documented packaging transforms |
| Broad descriptive BEIR-18 validation | Full parity matrix and length-stratified fixtures |
| Official release and public model card | Preview model card with existing M13 evidence |
| Upstream FastEmbed PR | Custom FastEmbed branch, for the preview only |
| Storage-retirement recommendation | Private-first Hub upload, download verification, public transition |

Rationale of record: `m13/RESERVED_EXECUTION.md` caps the reserved transaction at 55.2 hours
because the approximately 10M Stella passage vectors do not exist and the controller runs the
pre-encode and the protected scorer serially. It cannot fit the preview window, and a one-shot
irreversible access must not be compressed to meet a demo date. The reserved rows are zero-alpha
descriptive and cannot change the completed six-set claims, so the preview does not depend on them.

Preview constraints, all mandatory:

- The reserved access stays **unspent**. The Zero amendment window therefore stays open, and M20
  inherits it unchanged. Do not open a reserved payload, create `m8-reserved-spent`, or resume the
  A100 in M14.
- `results/m10_final_run.json` keeps `end_status = INCOMPLETE_RESERVED`. Do not relabel it
  `COMPLETE`; that transition belongs to M20's reserved run.
- The preview card states plainly that it is a research preview, that the reserved four and the
  broad BEIR-18 validation are pending, and that no upstream FastEmbed entry exists yet.
- All existing disclosures survive the preview framing: MIT plus bge-small and Stella attribution,
  the exact 199,999,721 dose with its reconciled shortfall, ArguAna/FiQA Stella contact with the
  clean-four headline, the TREC-COVID loss to LEAF, and synthetic-latency framing for serving costs.
- Publication target is `DylanCouzon/constella-nano` (Dylan, 2026-09-15). The preview is its first
  revision; M20 publishes the official card as a later revision of the same repository. Record the
  preview commit/revision URL so M20 can cite exactly what was shown.
- `DylanCouzon/constella-zero` and `DylanCouzon/stella-en-400M-v5-doc-onnx` are untouched.
- The custom FastEmbed branch carries the three native entries and reference-derived canonical
  vectors only. It must not include the unrelated #703 padding fix, and it is not the upstream PR.

Packaging, parity and upload work are ordinary registered-branch implementation and may proceed
autonomously under this ruling.
