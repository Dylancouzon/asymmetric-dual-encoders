# M18 status

**State: EXECUTING — approved pre-training `k8s` lock amendment.** Owner execution authorization arrived
2026-09-12.

Current outcome: the complete pinned source snapshot produced 79,269 indexable document units. The
source-only-adjudicated v6 protocol has 780 training views, 100 development queries and 40 sealed
confirmation queries. The synthetic end-to-end rehearsal and 45 network-free tests pass. Neither real deliverable
has been decided; no model-quality result or confirmation content has been read.

Two Astra spot-checks rejected the structural qrels even after narrow rule fixes. Astra then judged
the 344-pair prospective pool from source text only. After an independent correction, 140 are clear,
142 partial and 62 bad. Only the
clear pairs entered the final split; smaller realized strata are disclosed rather than filled.
Training uses 298 non-held-out issue/PR titles as teacher-only views without qrels plus 482 alias
views. A fresh dev-only audit found 39/40 clear and one partial; that pair was removed. Every chunk
of each query-bearing source object is now excluded with compensating overfetch. Full Stella
The full 79,269-document Stella/BM25 index is complete. The first baseline read found measurable
Stella headroom and is preserved. Before training, the owner approved a narrow CTO-requested `k8s`
amendment: five deterministic, source-evidenced teacher-only substitutions augment one natural
training occurrence. No qrels or held-out rows changed. Fresh preparation now selects `k8s` as the
sixteenth exact row and yields 148 eligible T0/T2 queries; T1 remains inert. The old lock and
preparation are archived and the changed V0 baseline has been read. A pre-optimizer review retained
batch 256 with the existing deterministic cross-epoch sampler and corrected duplicate alias-pair
handling; the realized alias-consistency loss is inactive because no complete pair survives the
trainable-query filter. T0 passed its step-500 diagnostic with positive dense and fused movement,
so the fixed T0/T2 checkpoint screen is proceeding. T1 is skipped as inert. Confirmation remains
unread.

Pointers: `instructions-m18.md`, `m18/registry.json`, `m18/LEDGER.md`, `m18/CODEMAP.md`.
