# M19 stage-5 Astra disposition

Review: `m19/reviews/stage5-astra-2026-09-13.md` at `be034a1`.

## P1 dispositions

1. **Bundle verification — closed.** The verifier now requires an expected build identity and
   reconciles exact manifest files, variant/config identity, model arrays and tokenizer bytes with
   provenance. A rehashed altered-added-row regression is refused.
2. **Measured-input authentication — closed.** Serving v2 runs full inheritance verification,
   validates the development split against its seal, authenticates the candidate build and observed
   bundle, and binds every registry/roster/query/build/bundle/index identity. Confirmation requires
   canonical production bindings and recomputes the exact 10,000-query sequence.
3. **Registered timed route — closed.** Serving v2 uses float32 dense scores, exclusion before
   passage depth, deterministic descending-score/ascending-passage-ID selection at the top-500
   boundary, artifact collapse at 100 and unchanged DBSF. The baseline is the released V1 loader.

The v1 serving evidence remains immutable but is superseded by
`results/m19_serving_benchmark_10000_v2.json` and `results/m19_serving_receipt_t0_v2.json`.

## P2 status

- Released-loader independence is now implemented and measured directly.
- Complete teacher snapshot provenance remains tracked hardening; actual offline regeneration was
  byte-exact and the present vectors/receipts are not disputed.
- Benchmark publication now writes the result first and derives/resumes the receipt from those saved
  bytes.

Verification: 71 M19 tests pass; official v2 serving checks are all true. Re-review must remain
read-only and bounded to these dispositions and directly affected code/evidence.
