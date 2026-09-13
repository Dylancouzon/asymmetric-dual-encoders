# M18 status

**State: EXECUTING — prospective source-only adjudication.** Owner execution authorization arrived
2026-09-12.

Current outcome: the complete pinned source snapshot produced 79,269 indexable document units. The
source-only-adjudicated v5 protocol has 779 training views, 101 development queries and 40 sealed
confirmation queries. The synthetic end-to-end rehearsal and 43 network-free tests pass. Neither real deliverable
has been decided; no model-quality result or confirmation content has been read.

Two Astra spot-checks rejected the structural qrels even after narrow rule fixes. Astra then judged
the 344-pair prospective pool from source text only: 141 clear, 141 partial and 62 bad. Only the
clear pairs entered the final split; smaller realized strata are disclosed rather than filled.
Training uses 297 non-held-out issue/PR titles as teacher-only views without qrels plus 482 alias
views. Full Stella encoding remains paused for one fresh dev-only audit of the final surface.

Pointers: `instructions-m18.md`, `m18/registry.json`, `m18/LEDGER.md`, `m18/CODEMAP.md`.
