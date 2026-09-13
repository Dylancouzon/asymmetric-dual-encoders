# M18 status

**State: EXECUTING — realized protocol re-review.** Owner execution authorization arrived
2026-09-12.

Current outcome: the complete pinned source snapshot produced 79,269 indexable document units and
a fresh 1,043-view training protocol, 175-query development set and sealed 75-query confirmation
set. The synthetic end-to-end rehearsal and 33 network-free tests pass. Neither real deliverable
has been decided; no model-quality result or confirmation content has been read.

An Astra spot-check rejected the first realized qrels (24/40 clear, 8/40 partial, 8/40 wrong), while
approving corpus provenance and finding no split leakage. The narrow remediation now excludes each
verbatim query-source document at evaluation, requires substantive answer evidence, rejects
clarification/status-only targets and removes task-style PR titles from the concept/how-to audit
stratum. Next: Astra re-audits the rebuilt development set before the Stella index is encoded.

Pointers: `instructions-m18.md`, `m18/registry.json`, `m18/LEDGER.md`, `m18/CODEMAP.md`.
