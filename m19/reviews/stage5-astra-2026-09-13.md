# M19 stage-5 Astra review

**Decision: `NO-GO` at `be034a1e5177f2a5906d4716f750ea8fc3ae0805`.**

The actual candidate contents reproduce correctly, but stage-5 authentication and serving evidence
have three P1 blockers.

## P1 findings

1. `zero.verify_bundle` accepts altered added rows if the model entry and completion self-hash are
   updated, because it does not reconcile current model/config/tokenizer bytes with their provenance
   digests or an externally expected build identity. Require both and add a negative test.
2. The serving benchmark does not call the full inheritance verifier, validate the development
   bytes against the query seal or authenticate the candidate-build receipt. Its serving receipt
   labels the bundle from that unchecked receipt rather than the observed verified bundle. Bind all
   measured inputs and require the frozen query sequence/host/thread/warmup/repetition contract.
3. The timed shortcut trusts incoming top-k tie order, omits source/copy exclusion and uses float16
   dense scoring, while the registered route requires float32 exact scores, exclusion before depth
   500 and descending-score/ascending-passage-ID ties. Benchmark the registered semantics or prove
   an equivalent fixed-workload route.

## P2 findings

- The official comparison uses `M19QueryEncoder` for both V1 and T0 rather than independently
  loading released V1. On 45 synthetic fixtures its output agrees with the released loader within
  `1.49e-8`, but its diagnostic runtime was about `1.07x` median and `1.09x` p95.
- The teacher receipt binds three large snapshot files but not every consumed tokenizer/module/
  pooling/custom-code file; revision checking also relies on the snapshot directory basename.
- Serving publication is not resumable if interrupted after its receipt is published but before the
  benchmark result.

## Positive verification

- Offline Stella recomputation reproduced all 12 stored teacher vectors exactly (maximum absolute
  difference `0`, identical array identity and runtime receipt).
- Both actual bundle identities, the build/row/algebra receipts and teacher array match independent
  reconstruction. Inherited codes/scales are exact and each bundle holds `31,388,952` compact bytes.
- T0 minimum int8 bare cosine is `0.9999541044`; maximum coordinate error is `0.0005363673`.
- Seventeen bounded Zero/common tests passed. The actual serving receipt satisfies the current
  confirmation parser.

## Access declaration

The reviewer used bounded git inspection, only the named M19 code/metadata/receipts/bundle files,
the explicitly authorized code-only M18/released-loader references, synthetic in-memory probes and
one offline teacher recomputation. It did not read query bodies, qrels, judgments, quality results,
protected historical surfaces or network resources, and made no repository edits. One authorized
`m18src/vocab.py` read included lines 1–180 beyond the intended helper slice; it exposed source code
only and was reported immediately.
