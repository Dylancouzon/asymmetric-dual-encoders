- **Slide 12.** The 2.4× median speedup is correct. Stella’s 20.8 ms bar ends near 2.3 ms on the shared axis. Show Stella as a text row outside the plot. Beside Zero’s 0.510 score, replace the asterisk with **“98.5% of its exact score, below the 99% target.”** That qualification matters to the bet. [NQ results](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/LEARNINGS.md:400)

- **Slides 7–8.** Name Zero in slide 7’s desktop-training title and label the chip **“≈20 min with prepared targets.”** Remove the squared-error formula. Zero’s two stages use alignment and score matching, then contrastive ranking. Slide 8’s **+0.0217** gain still lacks a supporting comparison in the supplied evidence. Cite the two development scores or cut the gain. [Zero training](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:236) · [Nano architecture](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:246)

- **Slide 9.** Plotting Zero + BM25 at 0.044 ms makes the quality gain look free. That timing measures Zero’s dense encoder alone. Move **“87.9% with BM25”** into a callout and name the extra sparse index, search, and merge there. [Paper, Table 2 and fusion result](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:125)

- **Slide 2.** Change “10.3 CPU-hours” to **“10.3 hours for one million sequential encodes.”** The calculation uses 37 ms of elapsed time with four threads. Label **5.3 GPU-hours** as a steady-rate estimate directly in the box. [Corpus measurement](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/LEARNINGS.md:392)

- **Slide 16.** The equation, table, and four steps compete for space. The closing notes then recite the steps. Move the equation to backup and enlarge the winning row. Ask for the choice, then reveal **Nano, `ef=64`, oversampling 4, 0.554 nDCG@10, and 2.12 ms p95**. Keep the spoken follow-up about which constraint the team would change at a 0.62 floor. [NQ sweep](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_talk_edge_nq.json)

- **Slides 1 and 17.** The cover logo appears broken in the contact sheet. The closing QR is an empty rectangle with placeholder text. Fix the logo and give the room a scannable QR with a readable repository URL. [Repository listed in the paper](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:165)

**BLOCKERS**