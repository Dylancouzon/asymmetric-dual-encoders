**READY for the capacity demo.** Say “same target rate” for the fixed trials. Drops now preserve the query sequence, but the timer starts before replay begins. The three 2,000/s trials offered 24,188, 24,204, and 24,210 requests. Pinned p95 and nDCG cover requests completed by pin time. The timely-hit percentage counts every arrival. [Final rehearsal](</private/tmp/claude-501/-Users-dylanc-Documents-GitHub-asymetric-dual-encoders/ffe18849-e8e1-45a2-8851-1392874ab679/scratchpad/final_run2.txt>)

Set **100 ms** before speaking. Check the displayed mode and rate before pressing `T`, then wait for the pin. I reproduced two rapid `↑` presses both selecting 1,500/s. Wait until 1,500 appears before pressing the second time. After Stella’s pin, return to 40/s and press `C` to clear the backlog before the 15-second cooldown. [Rate controls](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/spike.html:89) · [Queue reset](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/spike.py:175)

**Speaker lines.** Use the live pin values on stage. If you cite 6,000/s, identify it as a separate search-throughput measurement.

| Beat | Speaker line |
|---|---|
| Stella at 40/s | “At 40 queries a second, how often does Stella return a relevant passage from this 2.68-million-passage index within 100 milliseconds?” |
| Stella at 1,000/s | “Stella completed 854 of 12,084 requests by the time we took this reading, and almost nobody got a relevant passage in time.” |
| Cooldown at 40/s | “I’ve cleared the queue and brought traffic back to 40 a second to give the laptop 15 seconds to cool.” |
| Nano at 1,000/s | “Nano searched the same index, finished all 12,084 requests, and returned a relevant passage in time for 82% of them.” |
| Nano at 2,000/s | “When I doubled traffic, only 1% of arrivals got a relevant passage in time.” |
| Wide Zero at 2,000/s | “Zero encodes each query with a lookup table, but its wide search still falls behind, with fewer than 1% getting a relevant hit in time.” |
| Narrow Zero at 2,000/s | “Narrowing the search brought timely hits up to 70% with the same lookup table and the same index.” |
| Auto sweep | “Auto starts on Stella, shifts mostly to Nano, splits the peak between Nano and Zero, then returns to Stella as traffic falls.” |