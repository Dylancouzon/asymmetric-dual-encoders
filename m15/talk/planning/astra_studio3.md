1. **Cut the stage view to three traffic presets and one slider.** Keep Quiet, Spike, Overload, and the quality floor. Leave Auto and the 150 ms target visible as labels. Hide the traffic sliders, wave and burst controls, manual routing, machine load, and the optional Agent tasks beat. Keep the encoder throughput and p95 charts. The live quality number can carry the third chart’s job.

2. **Correct three things the audience could misread.**

   - Replace “Never below the quality floor” with “Auto uses settings whose average score on NQ meets the floor.” The live score averages the last 600 completed searches and can dip below it.
   - Say “Lowering the floor admits **Zero fast**.” Regular Zero already qualifies at 0.50. Hold traffic at 2,000/s while making that change. In your run, maximum sampled p95 fell from 1,375 ms to 95 ms.
   - Keep “below floor” in the table and drop “chosen.” I reproduced a case where the table said Stella was chosen while Auto picked Nano. Label the share column “served in the last second.” Label the dashed traffic line “offered” because it shows the configured rate.

3. **Blockers.** None for the rehearsed Auto sequence. The screenshot says “spike off” where traffic rises to 2,000/s. I verified that the current Overload preset emits “2,000/s.”

**READY**