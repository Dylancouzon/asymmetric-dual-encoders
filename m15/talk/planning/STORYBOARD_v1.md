# Storyboard v1 (slide by slide). About 30 min. Minimal slide text; narration adds.

Numbers: BEIR-15 exact nDCG@10 Stella 0.5614, Nano 0.5081 (90.5%), Zero 0.4572 (81.4%), BM25 0.4006. CPU encode p50 31.6 / 2.25 / 0.044 ms. Assets 1,670 / 132 / 90 MiB. 1M MS MARCO binary: 21.3 / 2.48 / 1.11 ms at nDCG 0.7226 / 0.6893 / 0.6116; BM25 0.66 ms / 0.4895 vs Zero 1.11 / 0.6094 (whole request). Zero encodes ~62x faster than Nano on that diagnostic. Word-order probe: Stella 4/24 pairs, Nano 2/24, Zero 0/24 (exploratory). Nano: 57.3 A100 h (~$95), 3.49M query texts incl. 834k generated, no relevance labels. Zero: trained on desktop RTX 3080. Ridge screen 4-7 min/model (A100). Ten registered models: gte-large best retriever, worst table; Stella best table. e5 tables cosine 0.90/0.89 vs Stella 0.78, yet rank worse. Demo: 2,681,468 NQ Wikipedia passages, Qdrant Edge in-process on the laptop (race numbers measured tonight, exploratory).

## Ch1 The bill that never stops (3 min)
1. Cover. Title, subtitle, name. Visual: three query lines converging on one index.
2. "Every search runs a 400M model." Visual: a query string flows into a big Stella block, out comes a vector, into the index; a 31.6 ms timer ticks. No other text.
3. Documents once, queries forever. Visual: cumulative compute over time: one step for indexing, a line that never stops for queries. Narration: re-indexing is the migration everyone dreads; agents multiply queries.
4. Statement: "Zero compute: no neural network runs when you search." That is the definition for the whole talk.

## Ch2 One index, three encoders (3 min)
5. The dial. Visual: Stella document encoder in the cloud builds one index. Three query encoders plug into it: Stella 400M, Nano 34.5M, Zero (a table). A dial picks one per request.
6. On the device. Visual: three boxes sized by asset size (1,670 / 132 / 90 MiB) inside a laptop/phone outline; documents stay in the cloud box. Encode time under each.

## Ch3 How we built them (6 min)
7. Copy the teacher's query vectors. Visual: same query into Stella (target) and into the student (prediction), an arrow labelled "match". Index icon untouched. Narration: the initial fit needs no relevance labels.
8. Zero, mechanically. Visual: "cheap flights paris to rome" splits into WordPiece tokens; each pulls a row from a 30,522-row table; rows average (sqrt-count weights); normalize. Caption: a term-weight lookup that emits a dense vector. Then the fit in one line: min ||XW - Y||^2 + lambda ||W - W0||^2 (a ridge regression, minutes). Narration: we then trained it further with a ranking loss on a desktop RTX 3080.
9. The price of averaging, and a twist. Visual: "Paris to Rome" and "Rome to Paris" produce the same Zero vector. Click: probe results Stella 4/24, Nano 2/24, Zero 0/24. "Dense retrievers barely read word order either."
10. Nano. Visual: bge-small backbone (34.5M) + projection into Stella's 1024-d space; training feed: 3.5M queries, 834k written by an LLM, no labels. Timeline bar: desktop GPU (Zero, recipe search) vs one cloud run (57 A100 hours, ~$95).

## Ch4 How much relevance survives (3 min)
11. One bar chart: BEIR-15 nDCG@10 Stella, Nano (90.5%), Zero (81.4%), BM25 dashed reference. Narration: averages hide spread; Zero keeps 0.95 on Quora, 0.67 on TREC-COVID and FiQA.

## Ch5 The catch, live (7 min)
12. The bet. Visual: "62x" faster encoding, Zero vs Nano. "How much faster is the whole search?" Room guesses.
13. Live: race on 2.7M Wikipedia passages (switch to demo). Lanes Stella/Nano/Zero + BM25 baseline; replay of measured per-query times; encode (violet) vs search (red) bar shows Zero waiting on search.
14. Why: the graph connects document vectors. Animated HNSW walk: Stella's query lands near the documents it should find; Zero's lands slightly off and stops short unless ef grows. Recovery numbers at the same ef.
15. Binary quantization + rescoring, the paper's 1M benchmark: stacked bars 21.3 / 2.48 / 1.11 ms, nDCG beside each; payoff "19x faster than Stella, 2.2x faster than Nano". BM25 callout.
16. Live: type-off across the four columns; ef control shows recovery of each encoder's own exact top 10. Zoom chat queries welcome.

## Ch6 Monday recipe (4 min)
17. The best model made the worst student. Scatter: ten registered models, teacher score vs table score; gte-large highlighted. Callout: e5 matches its teacher's vectors best (cosine 0.90) and still ranks worse. "Measure ranking, not cosine."
18. The recipe in four steps: judged queries; exact nDCG of each candidate; sweep ef (and oversampling) on your index; pick the cheapest setting that meets your floor and p95. Cost per million queries if measured.
19. Try it tonight: pip install fastembed>=0.9.0; three model IDs; QR to paper preview, repo, Hugging Face.

## Ch7 Close (1 min)
20. "Keep your index. Make the query encoder a dial." The dial image from slide 5. Thanks + QR.

Backup: per-dataset retention, 26-model conditional result, routing oracle, Nano layer concatenation, the encoder as an Edge collection.
