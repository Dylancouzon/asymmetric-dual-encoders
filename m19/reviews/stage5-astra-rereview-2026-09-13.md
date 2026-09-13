# M19 stage-5 Astra closure review

**Decision: `GO` at `9dcddb8fefe4e03f4d1d7aaae45af97248075497` for Stage 5 and the
ten-query pool-cost pilot.** This does not authorize real judging or confirmation.

No P0/P1 remains in the bounded closure scope. Astra accepted all three dispositions:

- production bundle loading binds the expected build identity and current provenance digests;
- serving/confirmation bind inheritance, seal, build, bundle, index, fixed sequence and benchmark
  configuration; and
- the timed route uses float32 scores, exclusion before truncation, deterministic passage-ID ties,
  artifact collapse and the released V1 baseline.

Seventeen targeted tests passed. Seventy synthetic top-500 tie/exclusion cases matched an
independent full sort. The actual T0 bundle matches its build receipt, the serving receipt derives
exactly from the saved v2 benchmark, released-loader parity is `2.24e-8`, and the recorded ratios
recompute exactly: encoder `0.9580` median/`1.0370` p95; end-to-end `0.9956`/`0.9696`.

Tracked P2 qualifications are complete teacher-snapshot provenance, the generic loader's optional
`expected_identity` outside production callers, and the unchanged theoretical degenerate-fallback
difference. Saved-result publication/resume and released-loader comparison are resolved.

Access was limited to the saved disposition, directly affected code/tests, v2 evidence and
necessary unchanged metadata/build/bundle inputs. Probes were synthetic/in-memory. No query bodies,
quality/judgment data, confirmation content, protected surfaces, network or edits were accessed.
