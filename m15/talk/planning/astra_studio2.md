**Make the floor change at 2,000 searches/s the centerpiece.** At 40/s, raising the floor to 0.56 leaves Stella eligible and gives the audience little to see. At 2,000/s, changing only the floor reveals what Auto prioritizes.

### Auto policy

- **Fix the impossible floor first.** When no setting qualifies, `or rungs[:1]` silently selects the best available setting. I reproduced a 0.64 floor selecting a synthetic 0.631 setting. Reject infeasible changes, keep the previous floor active, and display the highest attainable value. [studio.py](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/studio.py:192)
- **Limit recovery probes.** After an idle estimate decayed, 14 synthetic picks selected Stella before one completion could update its cost. Allow one probe of a recovering setting, then use its measured latency to decide whether to increase its share. [studio.py](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/studio.py:193)
- **Tie estimates to the actual search settings.** Advanced controls change `ef` and oversampling, but the encoder keeps its startup baseline and timing history. Use exact measured configurations in Auto and key costs by configuration. The lowest quiet-time median is a fallback heuristic. Claiming “highest capacity” requires throughput measurements under contention on the shared shard. [studio.py](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/studio.py:93)

### Four demo changes

1. **Start in Auto and keep it selected.** Quiet currently forces a 100% Stella split, so the apparent recovery can be manual. Give Quiet, Spike, and Overload explicit rates of 40, 600, and 2,000/s. Each should preserve the floor, target, and routing mode. [studio.html](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/studio.html:103)
2. **Replace the encoder table with Auto’s five candidate settings.** Show sweep nDCG, estimated wait plus service, and share of decisions. Give each row a reason such as “below floor,” “over 90 ms planning budget,” or “selected as fallback.” Nano and Nano-fast currently look identical on screen. [studio.py](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/studio.py:78)
3. **Mark control changes on the charts.** A vertical marker labeled “Floor lowered to 0.47” lets the audience connect the first Zero decisions, the draining queue, and the later p95 recovery.
4. **Label what the numbers measure.** Show configured rate and counted arrivals separately. Label quality “last 600 completed searches” and show the sample count for the current floor. At 40/s, the headline can carry about 15 seconds of earlier decisions. [studio.py](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/studio.py:355)

### 3½-minute stage path

- **0:00–0:35** — Start at 40/s, floor 0.50, target 150 ms. Point to Stella and the one index all three encoders search.
- **0:35–1:15** — Spike to 600/s. Follow the shift to Nano and watch p95 through the transition.
- **1:15–2:00** — Raise traffic to 2,000/s. Zero stays ineligible. Show the queue and latency cost of keeping the floor.
- **2:00–2:50** — Hold traffic steady and lower only the floor to 0.47. Your stress run reached 77% Zero, 2,010 served/s, and 100 ms maximum sampled p95, versus 1,410 served/s and 1,093 ms before.
- **2:50–3:30** — Restore 40/s and floor 0.50. Leave Auto on and let the audience see when Stella earns traffic back.

**Honesty line for the stage:** “The floor applies to each setting’s average on the measured NQ sweep. The live score comes from completed searches on that same query set.” The queries, judgments, and searches are real. The arrivals are simulated. The 90 ms planning budget is a heuristic for a 150 ms p95 target. Keep drops visible beside p95, since dropped requests never enter that percentile. [studio.py](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/talk/demo/studio.py:339)