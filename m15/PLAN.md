# M15 paper plan

Form: arXiv-style preprint (Dylan, 2026-09-17). One paper, per the registered default.
Working title (provisional): *Hot-swappable query encoders over a frozen document index*.

## The thesis

A dual-encoder retrieval system has one expensive, sticky asset and one cheap, mobile one. The
document index costs a full corpus encode and cannot be rebuilt casually. The query encoder runs
once per query and can be replaced. This paper treats the query encoder as the replaceable part:
one frozen `stella_en_400M_v5` document index, 1024 dimensions, serving several query encoders that
were never trained together with it, including a lookup table with no query-time neural network.

The contribution is empirical. We measure what a swap costs in retrieval quality and what it buys
in query-time compute, across a benchmark partition registered before the numbers existed.

## Inclusion bar

Every section answers one question: what does a reader now know that they could not have known
before? A paragraph that only restates a known result, narrates project history or defends a choice
nobody would attack gets cut. Concretely:

- A number enters the paper only if it traces to a committed result file and its registered analysis.
- A claim enters only with its caveat attached in the same place, never in a distant footnote.
- A failed approach enters only where it changes how a reader must read a headline number.
- No claim of architectural priority. The lookup-table construction cites pyNIFE as prior art.

## Structure

1. **Abstract.**
2. **Introduction.** The asymmetry: index cost against query cost. Why the query side is the part
   worth moving. What the paper measures.
3. **The hot-swap setting.** A precise statement of what the shared index fixes (dimensionality,
   normalization, score scale, document vectors byte for byte) and what a swap is free to change.
   The operational contract, with the places a swap is *not* free: prompt strings, tokenization,
   score scale under fusion.
4. **Query encoders under test.** Frozen Stella document tower. Zero, the token-vector lookup table.
   Nano, the distilled transformer under a 35M parameter cap. The original Stella query tower and
   BM25 as reference points. Training-data licensing constraints, since they bound the recipe.
5. **Protocol.** Registered clean-4 headline, all six beside it, partition sensitivity as its own
   row, exact search for quality, alpha and multiplicity handling, what was registered when.
6. **Results.** Quality across the roster. The descriptive reserved four and BEIR-15 breadth from
   M20. Fusion with BM25 under Qdrant DBSF@100.
7. **Deployment cost.** Query latency under one common serving protocol, resident size, Edge
   behavior. Cloud fusion kept distinct from measured Edge behavior.
8. **What is new.** The findings that are net-new knowledge, each with the one number that carries
   it. Candidate list is driven by the evidence index, not by what was hardest to produce.
9. **Limitations.** Contamination disclosure, unresolved superiority against equivalence, the
   interval that excludes training-seed variation, the coverage-against-capacity confound.
10. **Reproducibility.** Artifacts, revisions, hashes, harness commands.

## Files

- `EVIDENCE_INDEX.md` — claim to number to source file to caveat. The gate for section 6 onward.
- `NOVELTY.md` — related work and the defended novelty position.
- `HOTSWAP.md` — what is measured and what is only asserted about the swap itself.
- `PAPER.md` — the draft.
- `LOG.md` — decisions and session record.
- `REVIEWS/` — review briefs and returned findings, verbatim.

## Blocked on M20

Section 6's reserved-four and BEIR-15 rows, and any breadth claim that depends on them. Section 7
is complete from M13's common serving protocol. Sections 1 through 5 and 8 through 10 can be
written and reviewed now.
