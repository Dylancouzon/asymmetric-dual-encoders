# Talk status: London IR Meetup, Mon 2026-11-23 20:00

State: working. Talk "Zero-Compute Dual Encoders: from Nano to Constella", about 30 min plus Q&A, hybrid (BCS London + Zoom). Deck: https://claude.ai/artifact/KHUEXngqsxnHt43vh3Xptj (private; sources mirrored in `m15/talk/deck/`).

## Agreed plan (Claude + gpt-6.2-astra, owner approved 2026-11-20)
Thesis: keep your index, make the query encoder a dial. Takeaways: (1) replace the query encoder without touching the index; (2) a cheaper encoder moves the cost into search, so measure the whole request; (3) choose by measuring ranking on your own judged queries. Planning record: `m15/talk/planning/`.
Owner notes: high technical bar but not academic; depth in mechanisms, statistics off slides; minimal slide text; narration adds, never reads; one idea per chapter; think big; Astra reviews, Sol narrates.

## Done
- Word-order probe (exploratory): Stella 4/24 pairs both directions, Nano 2/24, Zero 0/24. `results/m15_talk_word_order.json`.
- Edge demo pipeline `m15src/talk_edge.py`; FiQA receipt `results/m15_talk_edge_fiqa.json`; Edge Zero parity exact.
- Stage demo: traffic-spike room `m15/talk/demo/spike.py` (run: `DATASET=nq .venv/bin/python m15/talk/demo/spike.py`, open http://127.0.0.1:8766; keys and beat sheet in the live slide notes). Old replay app `app.py` kept as backup. Run: `DATASET=nq .venv/bin/python m15/talk/demo/app.py`, open http://127.0.0.1:8765 (keys: 1 race, 2 live, space start, r reset; `?q=` preloads a query).
- NQ encode done 05:05 EST 2026-11-21: 2,681,468 vectors, steady 141 passages/s, 5.3 GPU-hours (T6, C27, L42); box-vs-Mac parity min cosine 0.99994. Vectors in `work/m15/vecs/nq_stella/`. Cost slide updated.
- Deck v1: 18 main slides + 4 hidden backups, Astra storyboard and deck reviews applied, Sol narration in every `<aside>` (29.5 min main).

## Running
- NQ Edge build on the Mac: `bash m15/talk/ops/nq_build.sh` (detached), log `work/m15/logs/nq_build.log`. Then `talk_edge.py bench nq`. Check: bench exact Zero nDCG@10 should match M20's registered 0.5173.

## VM (owner-provided, pay per minute)
- GCP `c4-standard-8`, us-east1-b, `ssh dylan@34.24.205.71`, code in `~/ade` (bundle of m15src, bench/core.py, m11/release, m7/FREEZE.json, NQ M20 pins), venv pinned to local versions. On-demand list price about $0.395/h.
- Encode-only throughput sweep launched 2026-11-21 (`~/encode_run.log`); receipt passes placeholder price 0.40, recompute dollars at 0.395. Single-thread Stella on this CPU: about 271 ms per query vs 49 ms on the M5 Pro.
- Next VM session: end-to-end NQ throughput after the NQ Edge build; upload shard, run `talk_throughput.py --modes e2e`. Ask the owner to stop the VM between sessions.

## Next
1. Finish `talk_edge.py build nq`, then `bench nq`; log T3 NQ (card C26 picks it up).
2. Fill NQ placeholders: bet (encode ratio as bars), race title counts, recovery chart (own-exact top-10 recovery vs hnsw_ef at fixed oversampling), recipe results row.
3. QR code once the paper preview URL exists.
4. Gates: andrey-review, Codex, humanizer on slides and narration. Rehearsal script and recorded demo fallback.
5. Optional: priced VM throughput (runpodctl REST endpoint unresolvable; no other cloud CLI); large-teacher table fit.

## Logging rule (owner, 2026-11-21): log every talk result in m15 as it lands
1. Receipt via a committed `m15src/talk_*.py` script into `results/m15_talk_*.json` (never overwrite).
2. Method in `m15/MEASUREMENTS.md` section "Talk measurements T1-T6" (add T7+ for new ones).
3. Card in `m15/make_evidence.py` (C28+), then `python m15/make_evidence.py`; diff must show only new cards.
4. Lesson in `m15/LEARNINGS.md` (next L-number; last L42), dated entry in `m15/LOG.md`.
5. Commit and push (standing approval).

## Decisions
- Laptop-only stage demo (owner). No phone demo. VM optional.
- Race uses released in-memory Zero; Edge-table variant is backup slide `b-edge`.
- NQ encoded on the RTX box in fp32 (paper document path), not fp16.
- Bet, race and payoff all on the NQ run; paper's MS MARCO benchmark is backup `b-msmarco`.

## Corrections
- None yet.

## Questions for owner
- Paper preview URL for the QR code.
- A cloud account for the priced throughput benchmark, if wanted.
