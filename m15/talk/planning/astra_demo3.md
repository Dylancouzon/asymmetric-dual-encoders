**Lead with Zero.** At 800 arrivals/s, wide Zero delivered a relevant top-10 hit within 100 ms for 76% of arrivals. At 1,000/s, it fell below 0.1%. Narrowing the search brought it back to 71% on the same encoder and index. That is the clearest reason for a search engineer to watch. [800/s rehearsal](</private/tmp/claude-501/-Users-dylanc-Documents-GitHub-asymetric-dual-encoders/ffe18849-e8e1-45a2-8851-1392874ab679/scratchpad/choreo_run4.txt>) · [1,000/s rehearsal](</private/tmp/claude-501/-Users-dylanc-Documents-GitHub-asymetric-dual-encoders/ffe18849-e8e1-45a2-8851-1392874ab679/scratchpad/choreo_run5.txt>)

### Three changes before Monday

1. **Fix the trial replay.** A full queue skips the random query draw. That also changes subsequent arrival times. I reproduced this by dropping only the third request. The fourth arrival moved from 2.125 to 4.826 ms and got a different query. Precompute the offered `(arrival offset, query ID)` sequence, then start its clock and trial ID together. Verify identical offered sequences with and without drops. [Traffic generator](</Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/spike.py:98>)
2. **Put the completion count on each pin.** The headline counts every arrival. Pinned p95 and nDCG use only requests completed by pin time. Stella's 0.644 came from 938 completions in the 12-second trial. Show `938 of N completed by pin` beside those metrics. [Pin calculation](</Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/spike.py:193>)
3. **Make 800 and 1,000 adjacent arrow-key rates.** Today `↓` from 1,000 lands at 667. One exact step lets the audience see wide Zero cross its limit before you touch the search settings. [Rate controls](</Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/spike.html:89>)

### Four-minute beat sheet

Set the budget to **100 ms** before speaking. The sequence assumes the arrow change. After each `T`, wait for the pin before changing settings.

| Time | Action | Speaker line |
|---|---|---|
| 0:00–0:25 | `1` Stella at 40/s | “These 2.68 million passages are indexed once in Qdrant Edge on this laptop. How often does a relevant passage arrive within 100 milliseconds?” |
| 0:25–0:50 | `S` to 1,000/s, then `T` | “Stella still ranks passages well when it finishes. Look at how many people get a useful result before their deadline.” |
| 0:50–1:15 | `3` Zero wide, `↓` to 800/s, then `T` | “Zero encodes each query with a lookup table. At 800 arrivals a second, it can afford a wide search.” |
| 1:15–1:40 | `↑` to 1,000/s, then `T` | “Same encoder, same search. I raised traffic by 25%. Watch the queue and the share of relevant hits that arrive in time.” |
| 1:40–2:05 | `4` Zero narrow, then `T` | “I changed only how widely Edge searches. Compare the two 1,000-per-second pins. We recover timely hits and give up some ranking quality.” |
| 2:05–2:35 | `2` Nano, then `T` | “Nano ranks its completed answers better than narrow Zero. Its timely-hit rate ranged from 72% to 83% in rehearsal. Here is today's result.” |
| 2:35–3:00 | `S` to 40/s. Hold Nano for 20 seconds, then `5` Auto. | “This is Nano's ranking score in quiet traffic. Now the service can choose an encoder for each request.” |
| 3:00–3:30 | `S` to 1,000/s. Point to encoder shares. | “Watch which encoder serves the spike. In rehearsal, about 92% of completions in this snapshot came from Nano.” |
| 3:30–4:00 | `S` to 40/s. Leave the queue alone. | “Watch Stella's share as traffic settles. Once 600 fresh answers have arrived, compare the ranking score with quiet Nano. The index stayed in place.” |

**Blocker for the paired-trial claim: `T` changes the offered trace when a mode drops requests.**