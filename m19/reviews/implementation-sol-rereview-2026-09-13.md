# M19 implementation re-review — Sol — 2026-09-13

Reviewer: `/root/m19_implementation_review`, `gpt-5.6-sol`, xhigh reasoning. Read-only review at
`f04d3ec5308bc797be44f795d942deefdece5a1c`. Decision: **NO-GO** for prospective query
construction. No P0; three P1 classes remained. All 61 allowlisted tests passed.

## Findings and dispositions

1. **P1: production inheritance schema mismatch.** Closed by `1d698e0`: confirmation consumes the
   frozen `identities.released_effective_table` layout, and the synthetic inheritance fixture now
   has that production shape.
2. **P1: candidate and serving gates not authenticated.** Closed by `1d698e0`: exact teacher
   vectors, deterministic row receipt and serving receipt are decision roles; authentication
   reconstructs/quantizes T0 from base rows, roster and teacher vectors, compares full arrays and
   exact extended tokenizer, requires T0 bundle variant, validates serving measurements against
   the registry, binds `candidate`/`versions`, compares every top-level recipe, and parses
   machine-readable role/reviewer/commit/scope GO receipts.
3. **P1: author independence and judged-pool coverage caller-asserted.** Closed by `1d698e0`:
   authors are derived from claimed query bytes; clarification hashes can only come from verified
   judgment-batch receipts; pool/packet/seven-route provenance and scored runs are cross-validated;
   every scored top-ten artifact must have a frozen qrel.
4. **P2: numeric/version slice included non-controls.** Closed by `1d698e0`; it now requires both a
   numeric/version tag and `longer_control`.
5. **P2: query sealing.** Still intentionally open and is the first stage-4 task before real split
   publication.
6. **P2: mutation regression.** Closed for a decision-bound row receipt; the complete suite now has
   63 passing tests.
7. **P3 documentation/schema drift.** STATUS and the new immutable rehearsal v3 result are corrected.

## Re-review access log

The reviewer opened the original request, re-review request, prior report, allowlisted guidance,
registry/locks/status/ledger, M19 source/test files, `m19src/term_inventory.py` and its test, and
`results/m19_rehearsal_v2.json`. It ran path-bounded git status/diffs/revisions, line-numbered reads,
bounded searches of the allowed source/lock files, `sha256sum` over allowed changed files, and all
ten allowlisted test modules with bytecode/cache disabled and a fresh
`/tmp/m19-implementation-rereview-*` base (`61 passed`).

The reviewer declared no protected, excluded, reserved, M18 confirmation, real corpus/index/bundle
or unlisted content was opened, hashed, listed, searched or inferred; no network/real retrieval or
repository writes occurred. The historical v1 rehearsal result was not reopened.
