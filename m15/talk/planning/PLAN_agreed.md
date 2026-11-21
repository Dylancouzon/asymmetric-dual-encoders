# Agreed talk plan (Claude + gpt-6.2-astra, four rounds, 2026-11-20)

Event: London IR & AI Meetup, Mon 2026-11-23 20:00, BCS London + Zoom. Title fixed: "Zero-Compute Dual Encoders: from Nano to Constella". About 26 min talk.

Thesis: Keep your index. Make the query encoder a dial.
Takeaways: (1) replace the query encoder without touching the index; (2) a cheaper encoder moves the cost into search, so measure the whole request; (3) choose by testing on your own judged queries, not the leaderboard.

| # | Chapter | Min | The one visual |
|---|---|---|---|
| 1 | The bill that never stops; define zero compute (no neural inference at query time) | 3 | documents once vs queries forever |
| 2 | One index, three encoders | 3 | three encoders into one frozen Stella index (the dial) |
| 3 | How we built them: teacher query vectors as targets; Zero = averaged token table (Paris/Rome = price of pooling); Nano = small transformer; RTX 3080 for Zero + recipe search, one 57 h A100 run (~$95) for Nano | 6 | one training diagram marking desktop GPU vs A100 |
| 4 | How much relevance survives: Nano 90.5%, Zero 81.4% of Stella on BEIR-15 | 3 | one retention picture |
| 5 | The catch, live: bet "Zero encodes ~62x faster than Nano, how much faster is the whole search?"; replay of separately timed runs on the 1M binary index; Zero waits on search (graph connects document vectors; new queries need wider ef); payoff 2.2x vs Nano, 19x vs Stella, nDCG beside each; BM25 callout (0.66 ms / 0.4895 vs Zero 1.11 ms / 0.6094, whole-request timing); live type-off across three encoders | 7 | one demo screen |
| 6 | Monday recipe: judged queries, exact relevance, sweep ef, price cheapest setting meeting floor + p95; best model gave worst table among ten registered; fastembed, HF, paper preview QR; priced-VM $/1M queries if measured | 3 | recipe as four steps + QR |
| 7 | Close: thesis as one image | 1 | the dial |

Cut (paper only): 26-model regression, prospective test, distance-ratio predictor, LightRetriever, routing oracle, layer concatenation, registered contrasts, held-out four, hybrid fusion, cosine-fidelity counterexample, query-generation details, quantization toggle.

Wording guards: "no relevance labels" applies to the initial fit only; best-to-worst is among the ten registered models; ef claim is recovery of each encoder's own exact neighbors; 2.2x and 19x are sums of separately measured component medians at unequal relevance; never "no model server".

Measurements: E-T3 word-order probe (pairs and intended passages fixed before running); E-T1 one priced VM, sustained completed searches/s at a p95 limit, relevance alongside. Drop break-even.

Demo: local on the M5 Pro; race replays separately timed runs (no shared-CPU contention); type-off live; recorded fallback; rehearse with Zoom share. Phone/browser Zero is an optional stretch, off the critical path.
