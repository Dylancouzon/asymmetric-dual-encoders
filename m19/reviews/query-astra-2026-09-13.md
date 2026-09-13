# M19 query pre-seal review — Astra — 2026-09-13

Fresh reviewer `/root/m19_query_review`, `gpt-6-astra`, xhigh, read-only. Scope was the
prospective 60-development/30-confirmation query draft, frozen roster/registry, query validator,
released tokenizer and only draft-cited records from the admitted corpus. This review authorizes
query sealing only, not retrieval, candidate construction, judging, scoring or confirmation claim.

The initial draft (`6a57656e04ca6ddbd3d2b8a4624322a8c8aaf43784bba1a0e02d04ce1b0c233c`)
received **NO-GO**: several confirmation queries reused development feature/fix families and six
numeric/version controls asserted details absent from their citations. The first revision
(`a3a164a78900e9f93468cb10bf659e505569b67af32049d5214a14222ad9b40e`) corrected the unsupported
details and most family leaks but retained three overlaps involving ARM64 cross-builds, asynchronous
HNSW and peer TLS configuration.

Final decision: **GO** for exact draft
`288d2089505aef08ba556b19ef162415c06bb417461077456a12a2a228198b66`.

- Exact counts are 12 bare, 36 short and 12 longer development queries; 24 short and 6 longer
  confirmation queries.
- Every query matches exactly one intended AddedToken and passes its registered lexical shape.
- Normalized text, lexical signature, declared intent family, source family and prospective-support
  artifact intersections are empty across the split.
- The final source-family repairs separate ARM64 page-size/Windows-NEON work, HNSW asynchronous
  integration/parameter overflow, and TLS configuration/certificate rotation.
- Substantive safety coverage is five development controls across five terms and four confirmation
  controls across four terms. Remaining numeric/version details were found directly in their cited
  artifacts.
- No P0–P2 finding remains for sealing.

The reviewer read only `CLAUDE.md`, `instructions-m19.md`, `m19/registry.json`,
`m19/term-roster-lock-v3.json`, `m19src/common.py`, `m19src/term_inventory.py`,
`m19src/queries.py`, `m19src/test_queries.py`, the unsealed draft, the permitted released tokenizer
and programmatically filtered records for exact draft-cited artifact IDs from the admitted corpus.
It used prescribed Python and path-bounded Git commands. It declared no writes, network, retrieval,
quality, judging, scoring, results data, protected evaluation data or M18 confirmation access, and
spawned no subagents.
