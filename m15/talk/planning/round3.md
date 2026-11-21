Round 3. New owner direction: a high bar for what is interesting. Do not drown people in information. Attendees leave with a clear picture of the general idea and the takeaways; the paper is hosted in preview for details. The flow must be coherent: one idea per chapter, no mixing.

So I cut hard. One thesis: "Keep your index. Make the query encoder a dial." Three takeaways only:
T1. You can replace the query encoder without touching the index: train a small student on your own model's query vectors. No relevance labels. A desktop GPU did most of it.
T2. A cheaper encoder moves the cost into search. Measure the whole request, not the encoder.
T3. Choose by testing on your own judged queries, not by the leaderboard (the best model gave the worst student).

Cut entirely from the talk (paper only): 26-model regression, prospective test, distance-ratio predictor, LightRetriever, routing oracle, layer concatenation, registered contrasts vs bge-small/LEAF, held-out four, hybrid BM25 fusion, cosine-fidelity counterexample, query-generation details.

Linear flow, one idea per chapter:
1. The bill that never stops (3 min). Documents are encoded once; every query runs a 400M model. Re-indexing to change models is the pain search engineers know. Define "zero compute": no neural network at query time.
2. One index, three encoders (3 min). Stella, Nano, Zero over the same vectors. The dial picture. Nothing else.
3. How we built them (6 min). Teacher vectors as targets. Zero = a table of token vectors you average (show how it works, Paris/Rome falls out here naturally as the price of averaging). Nano = small transformer trained the same way. Desktop RTX 3080 for Zero and the recipe search; one 57-hour A100 run for Nano (about $95). The one surprise of building: the best model made the worst student, so test (T3).
4. What you get (3 min). Retention 90.5% and 81.4% of Stella on BEIR-15; encoding 31.6 ms -> 2.25 ms -> 0.044 ms. Zero vs BM25 on 1M passages: +0.45 ms for nDCG 0.4895 -> 0.6094.
5. The catch, shown live (7 min). Pose the bet: Zero encodes 62x faster than Nano; how much faster is the whole search? Demo race (replayed separately timed runs, labelled) reveals the Zero lane waiting on search. Explain in one picture: the graph was built around Stella's queries, so new queries need a wider search (ef). Flip binary quantization: Zero gets back to 2.2x faster than Nano end to end, 19x faster than Stella. Then the live type-off across the three encoders. (T2)
6. Monday recipe (3 min). Judged queries -> exact relevance -> sweep ef -> price the cheapest setting that meets your floor. pip install fastembed, HF models, paper preview QR.
7. Close (1 min). Thesis restated as one image, not a recap list.
Total about 26 min plus buffer.

Questions:
1. Does each chapter carry one idea, and does the order build without forward references? Where would a listener get lost?
2. Anything in the cut list a search engineer would miss enough to put back? Anything still in that should go?
3. Is folding the "catch" into the demo coherent, or does it overload the demo?
Answer concisely. Say AGREE with any final changes, or state what blocks agreement.
