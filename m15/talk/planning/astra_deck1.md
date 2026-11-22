**Two visual upgrades would make the biggest difference.**

1. **Slides 12–13, bet and race.** Start with Nano and Zero’s encoding bars, take guesses, then extend the same bars with search time. Use the same NQ run throughout. Show median encoding-plus-search time and nDCG@10 beside each lane so the audience can see what the speedup costs. Label the race **Replay**. Save **Live** for the audience query. [Benchmark calculation](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15src/talk_edge.py:218)
2. **Slides 5 and 18, dial and close.** Show a three-position selector with one active path into the Stella index. Change the selection once while the index stays visible. Bring that image back beside the QR code on slide 18. Put the install command and model IDs on the linked page.

**Essential corrections and cuts**

- **Slides 2 and 4.** “Every Search Runs a 400M-Parameter Model” makes the Stella baseline sound universal. Use “Stella takes 31.6 ms to encode an uncached query.” For Zero, say “Zero encodes the query without running a neural network.” [CPU measurement](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:117)
- **Slide 9.** The headline names Zero, so the eye assigns **4/24** to Zero. Reveal it as “Stella got both directions right on four of 24 pairs.” Explain the two-passage test aloud. [Probe results](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/results/m15_talk_word_order.json:4)
- **Slides 14–15.** Title the chart “How wide a search recovers each encoder’s exact top ten?” Hold oversampling fixed and mark one recovery target across the curves. Let the NQ result determine the answer. Go straight from that chart to the audience query and give the three demo instructions aloud.
- **Slides 15–16.** The talk jumps from one shared index to ten models, each with its own index. Bridge it aloud: “What if you are still choosing the document encoder?” Enlarge the scatter, move the equation and variable definitions to notes, and title it “The best of ten retrievers produced the weakest lookup table.” Label the point `gte-large-en-v1.5`. The paper also tests a different `gte-large`. [Model scores](/Users/dylanc/Documents/GitHub/asymetric-dual-encoders/m15/PAPER.md:285)

**Blockers before rehearsal: scope the opening claims, identify the correct model, and make the NQ replay answer the bet at a stated relevance score.**