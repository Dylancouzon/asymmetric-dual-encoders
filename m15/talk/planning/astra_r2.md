AGREE. The race beats the word-order demo for this audience. It shows why someone would switch encoders. Keep Paris/Rome as the 20-second limit case. Keep the event title and define “zero compute” in minute one: **Zero does no neural inference at query time.**

**1. Give the race a fair clock.** A synchronized stream makes every lane wait for Stella. Three independent streams make the encoders compete for the same laptop CPU. Time each warmed encoder separately on the same fixed batch, then replay the three runs side by side, clearly labeled. Show completed queries and the final nDCG@10 on that batch. Make the type-off live, starting with one judged query where Zero finds the right passage. Free-text queries get results and latency only.

Prepare both search configurations before the talk. Change quantization and `ef` one at a time. Rehearse with Zoom sharing active and keep a recording. The published 2.2× compares sums of separately measured medians at different relevance levels. [Paper §6](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:139)

**2. Keep MS MARCO for the measured comparison.** The fixed judged race fits the project's validation-only rule. The permitted sources do not establish public display rights for either MS MARCO or FiQA. My choice is an aggregate-only race over the existing 1M diagnostic, then a live type-off over speaker-authored passages. Label the corpus switch. The exploratory BM25 receipt confirms the trade on 6,980 queries: Zero added **0.45 ms** at median latency and raised **nDCG@10 from 0.4895 to 0.6094**. [Paper protocol](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:256) · [BM25 receipt](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_bm25_zero_msmarco1m.json)

**3. Keep the revised order and protect four minutes for the Monday recipe.**

| Section | Minutes | The one visual it earns |
|---|---:|---|
| Opening bet | 2 | “Zero encodes 62× faster than Nano. How much faster is the whole search?” |
| Asymmetry and family | 3 | Three query encoders feed one frozen Stella index. |
| Training and screening | 5 | One training diagram marks the desktop GPU work, Nano's final A100 run, and the judged screen that reversed the teacher ranking. |
| Where the speedup went | 5 | Stacked encoding and search times reveal 2.2×. Put each encoder's relevance beside its bar. |
| Demo | 6 | One screen moves from the measured race to a live query across all three encoders. |
| Relevance and limits | 5 | One chart compares Zero, BM25, and hybrid on BEIR-15. Keep the 1M timing comparison in a separate callout. |
| Monday recipe | 4 | Start with judged queries. Measure exact relevance, sweep `ef`, then price the cheapest setting that meets the team's relevance and p95 targets. |