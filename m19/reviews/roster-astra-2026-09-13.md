# M19 term-roster Astra review — 2026-09-13

## Identity and brief

- Reviewer: fresh `gpt-6-astra`, reasoning `xhigh`, task `/root/m19_roster_review`.
- Mode: read-only adversarial roster review of commit `1736d93`.
- Result: `NO-GO` for stage 3; no P0, one P1, four P2 and one P3.

The bounded prompt admitted the M19 bootstrap/roster code, locks, result and documentation, the
M18 corpus manifest, exact redacted corpus, and released tokenizer/config. It prohibited edits,
recursive searches, every spent/reserved M7–M13 surface and every raw or tracked M18 confirmation
query, qrel or result.

## Findings and dispositions

| Severity | Finding | Disposition |
|---|---|---|
| P1 | Scanner could consume corpus/tokenizer A then bind B after the initial inheritance check. | Fixed in v2: tokenizer bytes are loaded once; corpus bytes are hashed while parsed; consumed hashes must equal inheritance; inheritance and implementation are reverified after scanning. |
| P2 | ASCII regex support boundaries differed from real AddedToken normalization/boundaries. | Fixed in v2: the jointly extended released tokenizer performs support matching with no truncation. Corrected counts are versioned, while v1 remains in history. |
| P2 | Meaning/demand/intent checks were structural and not grounded to source witnesses. | Fixed: every selected term has five distinct, non-equivalent intended-sense artifact witnesses and artifact references for each of three distinct intent axes. Explicit non-indexable rows are excluded. |
| P2 | Two-file immutable publication could strand an interrupted transaction. | Fixed: each output reconciles an identical existing file and completes the missing peer; differing existing or competing bytes are refused. |
| P2 | “Catalog frozen before scan” lacked an independent pre-scan git receipt. | Disclosed, not retroactively repaired: v2 states catalog and first result entered git together at `1736d93`; runtime order was fixed and no quality input was used. |
| P3 | Status simultaneously claimed no roster and listed it. | Fixed in the v2 status update. |

The reviewer independently reproduced the 12 v1 counts, then measured serving-consistent support.
All terms still exceeded the five-artifact minimum. It also verified all original fragment IDs,
new-token IDs, inherited ID preservation, both existing lock identities, the 79,269 searchable-row
count and absence of retrieval/qrel-driven selection.

## Access log

The reviewer opened only the files and three exact inherited inputs admitted in the brief. It used
bounded line reads, path-restricted git history/diffs, hashes and read-only Python checks for
support, tokenizer boundaries, semantics, input drift and interrupted publication. It reported no
edits, new files, recursive searches, directory listings, network access, raw corpus-text output,
retrieval, query authoring or prohibited evaluation reads.

## Re-review of `255e5cc`

Astra returned `GO` for stage-3 implementation and closed the original P1. It reproduced the v2
inventory/roster, both consumed hashes, the real-tokenizer boundaries and counts, all 60 witness
bindings, all 36 intent mappings, explicit-indexability/duplicate behavior, and interrupted/racing
publication. It found one residual P2: the final 12-term roster copied the full 15-term catalog
scanner audit.

The residual is closed with an immutable v3 roster/result rather than rewriting v2. V3 keeps the
15-term matcher audit as inventory evidence but independently extends exactly the 12 selected terms
for the final audit. It requires the audit term set to equal the roster, `added=12`, base vocabulary
30,522 and final vocabulary 30,534. A driver-level consumed-hash drift regression was also added.

The re-review used only seven named M19 files, bounded diffs for four documentation files, and the
exact corpus/tokenizer inputs. It reported no edit, file creation, directory listing, recursive
search, network, retrieval, query authoring, raw corpus-text output or prohibited evaluation read.
