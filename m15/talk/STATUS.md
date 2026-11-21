# Talk status: London IR Meetup, Mon 2026-11-23 20:00

State: working. Talk "Zero-Compute Dual Encoders: from Nano to Constella", about 30 min plus Q&A, hybrid (BCS London + Zoom).

## Agreed plan (Claude + gpt-6.2-astra, owner approved 2026-11-20)
Thesis: keep your index, make the query encoder a dial. Takeaways: (1) replace the query encoder without touching the index; (2) a cheaper encoder moves the cost into search, so measure the whole request; (3) choose by testing on your own judged queries.
Chapters: bill that never stops (define zero compute) / one index, three encoders (+ on-device footprint) / how we built them (desktop RTX 3080 for Zero, 57 h A100 for Nano; Paris/Rome) / how much relevance survives / the catch, live (race replay, graph tax, 2.2x and 19x, BM25 callout, live type-off, ef control) / Monday recipe (best model gave worst table; measure ranking not cosine) / close.
Owner notes: high technical bar but not academic; depth in mechanisms, statistics off slides; minimal slide text; narration adds, never reads; one idea per chapter; think big.

## Done
- Word-order probe (exploratory): Stella 4/24 pairs both directions, Nano 2/24, Zero 0/24. `results/m15_talk_word_order.json`.
- Edge demo pipeline `m15src/talk_edge.py` (build, parity, bench). FiQA smoke passed; Edge Zero parity max diff 0.

## Running
- NQ (2,681,468 passages) Stella doc encode on the RTX box: `ssh dylan@100.114.121.66`, `~/constella-talk/encode_nq.log`, output `~/constella-talk/work/nq_stella/`. Copy shards to `work/m15/vecs/nq_stella/` here when done.

## Next
1. Copy NQ vectors, parity-check 256 docs against Mac MPS encode, `talk_edge.py build nq`, `bench nq`.
2. Demo app (local, offline): race replay + live type-off + ef control.
3. New deck (Slides type, Qdrant design system, template look). Narration by gpt-6.2-sol. Gates: andrey-review, Codex, humanizer.
4. E-T1 priced VM throughput (under $10). B3 large-teacher table fit (include only if done by Sunday noon).

## Decisions
- Laptop-only stage demo (owner). No phone demo. VM optional.
- Race uses released in-memory Zero; the Edge-table variant is a backup point.
- NQ encoded on the RTX box (owner).

## Corrections
- None yet.

## Questions for owner
- Paper preview URL for the QR code.
