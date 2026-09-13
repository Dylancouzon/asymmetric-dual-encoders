# M18 status

**State: EXECUTING — prospective source-only adjudication.** Owner execution authorization arrived
2026-09-12.

Current outcome: the complete pinned source snapshot produced 79,269 indexable document units and
a v4 901-view training protocol, 175-query development set and sealed 75-query confirmation set.
The synthetic end-to-end rehearsal and 41 network-free tests pass. Neither real deliverable
has been decided; no model-quality result or confirmation content has been read.

Two Astra spot-checks rejected the structural qrels even after narrow rule fixes; the v4 sample was
27/40 clear, 8/40 partial and 5/40 wrong. Corpus provenance and measured split leakage passed. A
bounded 344-pair prospective pool is now frozen for source-only Astra adjudication before a final
split exists; only clear, standalone pairs will survive. No retrieval output or model identity is
available to the judge. Full Stella encoding remains paused until the resulting set passes review.

Pointers: `instructions-m18.md`, `m18/registry.json`, `m18/LEDGER.md`, `m18/CODEMAP.md`.
