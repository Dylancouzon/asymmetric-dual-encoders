**Almost. The surprise is that Nano’s transformer beat Zero’s lookup table on the same 2.68M-passage index.** At 1,000 target arrivals/s, Nano delivered an on-time, judged-relevant hit for 81% of arrivals. Wide-search Zero delivered 0%. What still falls flat is the reveal. The audience sees Zero lose, but the dashboard gives them no way to test *why*. Auto makes a weaker finale on its own. In the run log, it scored 72.8% on-time hits against fixed Nano’s 81%. [Rehearsal log](</private/tmp/claude-501/-Users-dylanc-Documents-GitHub-asymetric-dual-encoders/ffe18849-e8e1-45a2-8851-1392874ab679/scratchpad/choreo_run2.txt>)

### Three changes, in order

1. **Give fixed Zero a wide/narrow search toggle.** Keep the encoder, index, workers, and arrival trace fixed. Switch `ef 1024 / oversampling 8` to `ef 128 / oversampling 4`. Put the two live outcomes side by side: **on-time top-10 hits** and **nDCG@10 on completions**. Underneath, show the isolated search measurements, **1,070/s wide · 6,000/s narrow**, labeled “search only, this laptop.” Wide search has little headroom at 1,000/s. The live trial establishes how much narrowing it helps and what it costs in ranking quality. Both presets already exist. [Search settings](</Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/spike.py:42>)

2. **Make each pin describe one trial.** `C` clears waiting requests, but old completions still feed the next pin. I reproduced a Zero pin showing Stella’s 800 ms latency before Zero completed a single request. Give fixed trials one speaker key that starts the same timed arrival and query trace, tags requests by trial, and pins that trial’s results. Keep Auto continuous across its traffic changes. [Reset and pin](</Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/spike.py:160>)

3. **Give the projector one number and one explanation.** Make **“Relevant top-10 hit within 100 ms”** the largest number. Keep “last 500 arrivals whose deadline has passed” visible after pinning. In the screenshot, the chart clips the 0% line, hiding the whole Stella and Zero failure. Resize the canvas when its height changes. During Auto, show **“Last second: __% Stella · __% Nano · __% Zero.”** The current setting names the most frequent preset, while the stage medians pool all presets. The unlabeled logarithmic bars also make 1 ms and 61 ms look surprisingly similar. [Dashboard rendering](</Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/spike.html:89>) · [Metric windows](</Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/spike.py:197>)

### Four-minute sequence

| Time | On screen | Speaker line |
|---|---|---|
| 0:00 | Stella at 40/s. Set the deadline to 100 ms and leave it there. | “These 2.68 million passages stay indexed on this laptop for the whole demo.” |
| 0:25 | Spike Stella to 1,000/s. Hold on the headline as the queue fills. | “Stella still finds relevant passages. Watch how many arrive in time.” |
| 0:50 | Start wide Zero at 1,000/s. Reveal its isolated encode rate. | “With these 14 workers, Zero encodes 17,000 queries a second. What happens when we also make it search?” |
| 1:20 | Replay with narrow Zero. Put both Zero rows together. | “Same lookup table, same queries. I changed only how widely Edge searches.” |
| 1:50 | Replay with fixed Nano. Pin it beside wide Zero. | “In rehearsal, this transformer got a relevant passage back in time for 81% of arrivals.” |
| 2:15 | Lower traffic to 40/s. Keep Nano fixed and capture its quiet ranking score. | “If we left Nano on all day, this is the ranking quality we would get at ordinary traffic.” |
| 2:40 | Select Auto at 40/s, then spike to 1,000/s. Show the encoder shares. | “Watch which encoder serves the traffic as the queue starts to grow.” |
| 3:20 | Return to 40/s without clearing. Compare with quiet Nano after 600 fresh completions. | “Stella is available again. Does switching back buy us better rankings on the same index?” |

Two claims need care on stage. Zero’s loaded encode timer includes time its workers spend waiting. The isolated **1,070 searches/s** helps explain the bottleneck, but it does not account for every millisecond of the live run. And the screenshot’s dashed line is the **latency budget**. Target arrivals/s appears as a tile. [Stage timing](</Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/spike.py:57>) · [Chart rendering](</Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/spike.html:104>)