# M19 fresh Astra end-to-end review

**Decision: `NO-GO` for real development pool freezing/judgments.**

- Reviewed implementation: `b0faad1cea107eb81996736d5aebc054f3d6af9f`
- Scope SHA-256: `a6902092c2ec5f480b9f1648fa9e418b45742ae52f989d407dca44111f625e9e`
- Reviewer: `/root/m19_end_to_end_astra`

## P1

1. The actual pilot packet lacks required parent context: 85/224 items have blank titles and all
   269 displayed passages have empty parent metadata. Resolve artifact titles and thread/repository
   metadata from pinned corpus bytes and require them before judging.
2. Frozen packet validation joins query/artifact IDs but not sealed query text, term and source-
   exclusion identity. Pass authenticated query specs into validation and reject substitutions.
3. Confirmation accepts development qrels/effect gates without authenticating the primary/audit/
   adjudication freeze or Stella headroom. Bind and recompute those prerequisites.
4. Confirmation claim does not reconcile the decision's confirmation digest with the original
   split seal. Require the sealed digest, IDs and counts.

## P2

- Freeze validation does not enforce actual item count at or below the declared cap or per-route
  top-ten length.
- Confirmation binds the review-scope hash but does not reconcile its declared file hashes or rerun
  inherited-byte verification at quality transactions.
- Audit selection exposes repeat metadata rather than producing concealed reviewer-facing repeats.

All 63 corresponding synthetic tests passed. Candidate/bundle/algebra evidence remained valid and
the actual pilot's structural hashes/counts reconciled. The reviewer inspected structural metadata
only, without displaying query/passages or assessing relevance. No confirmation, protected/spent
data, network or edits were accessed.
