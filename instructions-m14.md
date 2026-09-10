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
