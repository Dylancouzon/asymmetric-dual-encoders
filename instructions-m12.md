# M12 — the nano half of the pair, and the whitepaper

**Created 2026-09-03 by the renumbering below.** M11 shipped everything that did not depend on
M10; this milestone is what was left, and it is **blocked on M10**, which is blocked on the cloud
GPU budget (`CLAUDE.md` M10 entry). If M10 never runs, M12 does not run either, and the honest
close-out is the zero-only frontier M11 already published.

## Why the split (Dylan, 2026-09-03)

M11's deliverable 1 was "release both models". `constella-zero` and the frozen document tower are
released, ported to ONNX, integrated with FastEmbed and documented. `nano` is not, and cannot be
until M10 produces it. Holding M11 open on a dependency that may never land buys nothing; closing
what shipped and giving the blocked remainder its own number is the accurate record.

## Deliverables

1. **Release `nano`** — M10's distilled query tower, on the same terms as `constella-zero`:
   frozen artifact, `assert_releasable`, MIT, model card carrying measured numbers and the stella
   contamination disclosure. Name is **`constella-nano`** (`m8/LEDGER.md` §6.1).
2. **ONNX port of nano**, parity-verified. `m11/CODEMAP.md` is the checklist — 23 items, each one
   paid for by a real defect. Read it before writing the exporter.
3. **Register nano in FastEmbed**, the same way the pair already is: an entry in
   `supported_onnx_models` plus a canonical vector from the reference implementation. See the
   note on the PR below.
4. **Whitepaper / decision report** — the quality-vs-query-cost frontier with BOTH points, edge
   cost rows, the Qdrant Edge prototype, and the comparator table that was deliberately kept OFF
   the model cards (`instructions-m11.md` Amendment B). This is where
   `LR-dense-pertask 0.4583`, the OpenSearch tie and the missed bar belong.

## The upstream FastEmbed PR is NOT blocked on M10

`constella-zero` and `stella-en-400M-v5-doc-onnx` are both published and both registered on
`Dylancouzon/fastembed@add-constella-models`. A clean, single-concern PR for those two can be
opened whenever Dylan wants, independently of nano. It is listed here only so it is not forgotten;
do not treat it as gated on M10.

Two things that branch is NOT ready to be a PR as it stands:
- it also carries the qdrant/fastembed#703 padding fix (2 commits), deliberately, because it was
  never meant to be merged — Dylan, 2026-09-03: *"this branch won't be merged, we will do a clean
  PR when we're ready"*;
- the models sit on a personal HF account, where upstream hosts under `Qdrant/`. Same ruling.

## Inherited constraints

- **Student cap 35M, hard** (Dylan, 2026-09-01).
- The evaluation protocol is unchanged and still binding: partitions, decontamination, frozen
  comparators, single final run, pre-registered statistics — changeable only BEFORE the numbers
  they affect are observed.
- The reserved four (FEVER, DBpedia-entity, cqadup-android, cqadup-english) still have exactly
  **one unspent confirmatory access**. `results/perquery.json` is irreplaceable.
- M9's registered six-set close-out transaction is still pending Dylan's ratification.
