# M18 status

**State: EXECUTING — final protocol verification.** Owner execution authorization arrived
2026-09-12.

Current outcome: the complete pinned source snapshot produced 79,269 indexable document units. The
source-only-adjudicated v6 protocol has 780 training views, 100 development queries and 40 sealed
confirmation queries. The synthetic end-to-end rehearsal and 43 network-free tests pass. Neither real deliverable
has been decided; no model-quality result or confirmation content has been read.

Two Astra spot-checks rejected the structural qrels even after narrow rule fixes. Astra then judged
the 344-pair prospective pool from source text only. After an independent correction, 140 are clear,
142 partial and 62 bad. Only the
clear pairs entered the final split; smaller realized strata are disclosed rather than filled.
Training uses 298 non-held-out issue/PR titles as teacher-only views without qrels plus 482 alias
views. A fresh dev-only audit found 39/40 clear and one partial; that pair was removed. Every chunk
of each query-bearing source object is now excluded with compensating overfetch. Full Stella
encoding is next after verification of these two bounded corrections.

Pointers: `instructions-m18.md`, `m18/registry.json`, `m18/LEDGER.md`, `m18/CODEMAP.md`.
