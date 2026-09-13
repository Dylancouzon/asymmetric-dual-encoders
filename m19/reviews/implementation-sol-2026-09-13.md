# M19 implementation review — Sol — 2026-09-13

Reviewer: `/root/m19_implementation_review`, `gpt-5.6-sol`, xhigh reasoning. Read-only review of
base commit `8a5e63750e818f82aa05a7e71995ef2e42800c71` under the bounded prompt in
`implementation-review-request-2026-09-13.md`. Decision: **NO-GO** for prospective query
construction. No P0 findings.

## Findings and dispositions

1. **P1: decision hashes were shape-checked, not authenticated.** Closed by `502fd7c`: exact
   path/hash bindings are rechecked at construction, every state read/transition and reconciliation;
   registry recipes are compared semantically; bundle/roster/inheritance/review identities are
   verified; development eligibility is independently recomputed from exact bound inputs.
2. **P1: incomplete or inheritance-wrong bundles were loadable.** Closed by `406cde3`:
   `from_bundle` requires complete-manifest verification and mandatory base rows/scales, full base
   tokenizer, selected roster audit, inheritance identity and pooling identity.
3. **P1: qrel freeze trusted a supplied audit report.** Closed by `406cde3` and `502fd7c`: audit
   sampling/agreement/coverage are recomputed, unjudgeables/disagreements require exact
   adjudication, clarification requires complete relabel/fresh seed proof, the primary judge must
   be independent of all query authors, and the transaction itself creates frozen qrels.
4. **P1: metrics could be created before qrels freeze.** Closed by `502fd7c`: scoring accepts only
   an output destination after `qrels-frozen`, reloads bound qrels/runs/queries/support checks,
   computes metrics and eligibility internally, and derives completion from that result. The
   negative test verifies the destination remains absent on an early call.
5. **P1: tokenizer implementation was omitted from the review allowlist.** The re-review request
   explicitly adds `m19src/term_inventory.py` and `m19src/test_term_inventory.py`.
6. **P2: duplicate results/nonbinary qrels and scalar eligibility trust.** Closed by `406cde3` with
   strict inputs and one derived frozen evaluator.
7. **P2: query sealing/near-duplicate/source-exclusion receipt.** Open; must close before sealing
   real splits, not by enlarging the stage-3 transaction.
8. **P2: rehearsal coverage.** Partially closed by `502fd7c`: real synthetic bundle verification,
   exact decision bindings, negative audit sampling, transaction-owned qrels/metrics and immutable
   v2 result publication. The post-interruption path reloads the packet from bound disk bytes.
   Retrieval-collapse and clarification remain covered by focused tests rather than more rehearsal
   branches.
9. **P2: remaining real tokenizer/parity/latency gates.** Open by design until stage 5; no candidate
   construction or quality scoring is authorized by this disposition.
10. **P3: abbreviated commit did not resolve and status was stale.** The full base hash is recorded
    above and STATUS is corrected.

## Original review access log

Workdir was `/home/dylan/asymetric-dual-encoders-m18`. The reviewer opened only the 29 exact files
listed in the request. Commands: `pwd`; bounded `sed` of the request; `git rev-parse HEAD`;
path-bounded `git status`; failed `git rev-parse 8a5e637a` and bounded diff using that abbreviation;
successful `git rev-parse 8a5e637`; path-bounded diff from that commit; `awk`, `nl`, and `sed` reads
of the allowlisted guidance, locks, documents, source and tests; one pytest invocation against the
seven safely importable allowlisted modules in a fresh `/tmp/m19-implementation-review-*` base
directory (`39 passed`); bounded `sha256sum` over allowlisted control/source/result files; and a
final path-bounded clean status check. `test_zero.py` and `test_queries.py` were not run because the
first allowlist omitted their direct dependency `m19src/term_inventory.py`.

The reviewer declared: no protected, excluded or unlisted content was opened, hashed, listed,
searched or inferred; no network or real retrieval was used; no repository file was changed; all
synthetic writes were confined to the fresh `/tmp` directory.
