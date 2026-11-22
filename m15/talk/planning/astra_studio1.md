**Verdict.** The studio delivers live search, simultaneous routing, and a visible latency target. The strongest moment is the fixed split at 600 queries/s. Even Zero waits seconds because all three encoders share the same workers. What is missing is a clear view of that queue. The audience can see latency rise, but the speaker has to explain why. The queries and judgments are real. The arrival patterns and customer tiers are simulated.

**Four changes I would make**

1. **Make the queue visible.** Put measured arrivals/s beside completions/s, then show requests waiting and dropped/s. Keep p95, nDCG@10, and process CPU large. Move CPU/1k into the replay comparison and keep only the latency and traffic charts on stage.
2. **Put workload presets on screen.** For *Agent fan-out*, issue a group of searches together and time the last answer. For *Free/paid search*, attach a tier to each request and show p95 by tier. For *Nightly model swap*, switch new queries from Stella to Nano at constant traffic and keep the same index.
3. **Simplify routing.** Use one split control totaling 100%, with a visible 10/30/60 preset. The current sliders take independent weights but display normalized percentages. Put workers, HNSW `ef`, and oversampling under Advanced.
4. **Make replays comparable under overload.** Lock controls during replay, count every offered request as completed, dropped, or outstanding, and measure CPU time across the trial. The current replay can finalize while slow requests are still queued.

**Four-minute stage path using today’s controls**

| Time | Speaker beat |
|---|---|
| 0:00–0:35 | Press `1`, then `T`. “At 40 queries/s, Stella returned answers in 75 ms at p95, with 0.627 nDCG@10.” |
| 0:35–1:10 | Press `S`, then `T`. “We asked for 600/s. Stella served 81/s, and p95 reached 6.9 seconds.” |
| 1:10–1:50 | Turn the spike off, press `4`, then `T`. “At 40/s, the 10/30/60 split reached 22 ms. Ranking quality fell to 0.592.” |
| 1:50–2:35 | Spike and replay the split. Point to the three encoder rows. “All three waited about 3.2 seconds at p95. Stella’s 60 requests/s occupied the shared workers.” |
| 2:35–3:25 | Keep the spike on. Press `5`, then `T`. “Auto routed everything to Nano in this run. P95 fell to 83 ms, with 0.602 nDCG@10.” |
| 3:25–4:00 | Hold the two 600/s rows on screen. “Same traffic, same laptop, same 2.68 million indexed passages. Changing the query encoder brought p95 below 100 ms.” |

**Honesty traps.** Use those numbers as rehearsal landmarks and read the live result on stage. The current “arriving/s” is a configured rate. A one-second count of 642 completions does not establish sustained capacity above 600/s. Live p95 covers recent completions, and nDCG covers the last 600 completions, so both can hide unserved traffic. Auto follows a fixed preference order with estimated service times. Its 100 ms target is a goal. A 12-second replay also misses the peak of the 60-second daily wave. Keep capacity claims specific to this laptop, its 14 workers, and its single serialized Edge shard.