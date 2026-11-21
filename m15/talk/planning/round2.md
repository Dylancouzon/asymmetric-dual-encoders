Round 2. I accept your corrections on wording (best-to-worst among the ten registered; ef claim is own-exact-neighbor recovery; no "no model server"; verify Zero browser size). The BM25 receipt exists: results/m15_bm25_zero_msmarco1m.json, timed from query text to ranked IDs, three passes, 6,980 queries, 1M MS MARCO passages, Zero binary 1.11 ms vs BM25 0.66 ms, nDCG 0.6094 vs 0.4895. Exploratory.

Where I push back:
A. The title is fixed by the event: "Zero-Compute Dual Encoders: from Nano to Constella". We define it in minute one: "zero compute" means no neural network at query time; a tokenizer, a table lookup, and a mean. We do not hide it.
B. Your demo shows Zero's failure case. The speaker asked for a demo that shows the POWER of the models. A failure case belongs in the talk, but it cannot be the demo. Low-text, high-impact, reliable.

My revised demo, "Three lanes, one index" (all local on the M5 Pro laptop, no venue Wi-Fi dependency; Zoom sees the screen share):
1. Race. One stream of judged queries (MS MARCO 1M diagnostic collection already built locally, or FiQA) feeds three lanes: Stella, Nano, Zero, each encoding then searching the SAME Qdrant collection. Live counters per lane: queries per second, encode ms vs search ms split, running nDCG@10. Zero's lane is visibly bottlenecked by search, not encoding. The speaker flips binary quantization and the per-encoder ef, and the Zero lane jumps. This dramatizes the "where did the 62x go" bet live.
2. Type-off. The speaker (or a Zoom attendee via chat) types a query; three result columns update with per-encoder latency. Include one ordinary win for Zero and the Paris/Rome direction pair as a quick honest beat, which hands off to the limits section.
3. Optional stretch, decided Saturday after a measured browser build: a QR page where phones encode with Zero in-browser and hit a same-origin proxy. Not on the critical path.

Revised arc: bet (62x encoding, guess the end-to-end) -> asymmetry and family -> how we trained on a desktop GPU + the screening surprise -> where the speedup went (graph tax, quantization) -> live demo -> relevance, BM25, hybrid, limits -> Monday recipe.

Measurements: keep E-T3 word-order probe as you specified (pairs and intended passages fixed before running). E-T1 as you specified (one priced VM, sustained completed searches/s at a p95 limit, relevance alongside). Drop E-T2.

Questions for you:
1. Does the race demo beat your word-order demo on the speaker's goal? What would make it fail on stage?
2. Is the MS MARCO diagnostic a fine demo corpus given MS MARCO's non-commercial license (validation-only in this project), or should the race use FiQA or another corpus?
3. Agree or disagree with the revised arc and minute budget? Name the single slide each section must earn.
Answer concisely. If we agree, say "AGREE" and list any remaining changes.
