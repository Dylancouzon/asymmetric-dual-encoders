**Order.** I would use **cover, cost, arch, bench, zero-mech, order, distill, worst, nano, cloud, bet, race, split, recovery, live, recipe, close**.

- Move **bench** after **arch**. Once the audience sees that three encoders share one index, answer its first question: how much retrieval quality does each keep?
- Move **worst** after **distill**. The teacher comparison makes sense once people understand a lookup table. This also lets the live demo lead directly into the recipe and close.

**Blockers**

- **Cover.** “Zero-Compute Dual Encoders” overstates the result for both students. Zero still uses about 28 KFLOPs per query, and Nano runs a transformer. “Three Query Encoders, One Index” states what the talk demonstrates.
- **Worst.** The chart draws an upward line from Stella’s **0.575 to 0.397** because the columns use different scales. Plot both on one nDCG scale, or label the columns by rank. Stella went from second to first among ten while its absolute score fell. :codex-file-citation{path="/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md" purpose="source"}
- **Live.** Change “Same Index, Four Rankers” to **“Four Rankers on One Shard.”** The three dense encoders share Stella’s vector index. BM25 uses a separate sparse index on the same passages.

The headline timings check out. The replay totals give **2.30×** and the median requests give **2.39×**. The cloud prices also reproduce at the slide’s stated $0.395/hour. The supplied aggregate results do not verify the live queries’ named hits or the FastEmbed minimum version. :codex-file-citation{path="/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_talk_edge_nq.json" purpose="source"} :codex-file-citation{path="/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_talk_throughput_nq_gcp-c4-standard-8_encode.json" purpose="source"}

**Three remaining fixes**

1. **Order notes.** Pay off the airport example aloud. Across 24 pairs, Stella ranked the intended passage first in both directions on four, Nano on two, and Zero on zero. That tells the room what the angles meant for retrieval. :codex-file-citation{path="/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/LEARNINGS.md" purpose="source"}
2. **Split.** Put **“Zero missed the 99% target”** beside its bar in large type. The back row should see the quality concession as readily as the 2.4× speedup.
3. **Recipe and close.** Move the optimization formula into the notes and enlarge the decision table. On the close, show only the encoder choice and query against the populated collection. The upload line repeats the corpus work the audience came to avoid.

**BLOCKERS**