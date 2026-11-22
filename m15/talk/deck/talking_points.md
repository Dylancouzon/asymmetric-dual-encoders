# Talking points

One section per slide, in talk order. `rebuild.py` copies each section into that slide's speaker notes. Lines in [brackets] are stage directions. Every number here has a receipt in `m15/EVIDENCE.md` or the paper.

## cover · ~0.5 min
[Pause on the title. Trace the three lines into the dots.]
- Hook: someone offers you a cheaper query encoder on Monday. How long until you know if it helps?
- If the answer is "re-embed the whole corpus first", the experiment sits in the backlog.
- This talk: how far you can get with the index you already have.
- I'm Dylan Couzon, research in Developer Relations at Qdrant. Welcome, London and Zoom.
Next: I started by timing two jobs that run on very different schedules.

## cost · ~1.5 min
Why it matters: query encoding is the bill that never stops.
- [Grey block] Tonight's index: 2.68M NQ passages, encoded once with Stella. 5.3 GPU-hours on my desktop (6.5 hours real time, including a crash). I made tea.
- [Red blocks] Every question also goes through Stella before Qdrant can search. One question is cheap. They never stop.
- Each red block is one whole index worth of compute. At 100 queries/s you spend one block every 27 hours: 6 more index encodes per week.
- It is also on the latency path: about 20 ms to encode a query on this laptop, a quarter of a millisecond to search.
- Agents make it worse: several searches per answer, each one waits for an encode.
Next: so what else could search the passage vectors I already paid for?

## arch · ~1.5 min
Why it matters: you keep the index and change only the query side.
- [Top row] Documents go through Stella's document encoder once. Those vectors and that HNSW graph never change.
- [Three rows] Each request picks a query encoder. All three write vectors into the same space and search the same index.
  - Stella's own query encoder: 435M parameters, 25 GFLOPs per query.
  - Nano: a small transformer, 34.5M parameters, 0.54 GFLOPs. About 46 times less compute.
  - Zero: no neural network at all, a lookup table. 28 KFLOPs, nearly a million times less than Stella.
- [own ef] Each encoder also gets its own search width. That matters later.
- The family is called Constella.
- Switching costs nothing: no second index, no re-encode. If rankings get worse, switch back on the next request.
Next: how can a lookup table produce a query vector?

## zero-mech · ~1.5 min
Why it matters: this is the whole encoder. No matrix multiply.
- Step 1: Stella's tokenizer turns the query into token IDs.
- Step 2: each token ID picks one row from a table: 30,522 rows, one per token, 1,024 numbers each.
- Step 3: a weighted mean of the rows, then normalize. Weight = square root of the token count, so a repeated word counts a bit more each time.
- The table is int8 with one scale per row: 31 MB. It fits in memory next to the search process.
- [Point to the two vectors] For this query, Zero's vector has cosine 0.88 with Stella's. Not identical, but pointing the same way.
Next: a table of random rows would not do that. Here is how it learned.

## distill · ~2 min
Why it matters: the table learns to speak Stella without touching the index.
- Goal: rows whose average lands where Stella would put the query.
- Start: each row is Stella's encoding of that single token. Rare tokens start with more weight (IDF).
- Phase 1, Align (16,000 steps, batches of 512):
  - Targets: Stella's query vectors, computed once and saved.
  - Loss part one: point the same way as Stella's vector (1 − cosine).
  - Loss part two: for labeled queries, match how Stella spreads its scores over the relevant passage and 31 other candidates (KL divergence).
  - Why both: a vector can look close to Stella's and still rank a different passage first.
- Phase 2, Rank (2,500 steps): InfoNCE on 338,076 labeled query-passage pairs.
  - Push the relevant passage above 32,768 negatives from a frozen bank.
  - Drop candidates Stella scores close to or above the relevant one: they may answer the question too.
- [Formula line] Ê_D, the document side, never changes.
- Cost: about 20 minutes on one RTX 3080, once the targets exist. A new teacher needs 8 to 12 hours to compute targets first.
- All training sources allow commercial use of the trained weights.
Next: I assumed the best retriever would make the best table. It did not.

## worst · ~1.25 min
Why it matters: you cannot pick your teacher by its leaderboard score.
- I fitted a quick lookup table (a few minutes on one A100) to 10 strong models and tested each on its own index.
- [Red line] gte-large was 1st as a retriever and 10th as a table: 0.597 down to 0.246.
- [White line] Stella was 2nd as a retriever and 1st as a table. That is why Constella uses Stella.
- Across all 26 models I tried, the two rankings barely relate (rank correlation +0.09).
- Lesson: if you might want a cheap query encoder later, test it on a small index before you encode the whole corpus.
Next: even with the best teacher, a lookup table has limits you should know.

## order · ~2 min
Why it matters: know which queries Zero will get wrong before your users do.
- Zero does three things: split into tokens, look up one row each, average. Each step loses something.
- [Order] An average has no order. "paris to rome" and "rome to paris" give the same vector and the same results list.
  - Test: 24 query pairs where direction changes the answer. Right passage first in both directions: Zero 0, Nano 2, Stella 4. Hard for everyone, impossible for Zero.
- [Context] One row per token, whatever is next to it. "bank" is the same row in "river bank" and "bank loan". A transformer reads neighbours; Zero only has the other rows in the average.
- [New words] The vocabulary is fixed. "qdrant" becomes q, ##dran, ##t, rows learned from other words. A transformer recombines pieces; Zero averages them.
- Takeaway: Zero fits short keyword queries. Test phrased questions separately.
Next: to get order and context back, you need attention. That is Nano.

## nano · ~2 min
Why it matters: a small transformer that writes vectors into Stella's space.
- Backbone: bge-small, 12 layers, 384 features per token.
- Problem: Stella's vectors have 1,024 numbers. A linear projection of 384 features can only move in 384 directions, however long its output.
- Fix: mean-pool layers 4, 8, and 12, join them (1,152 features), then project to 1,024. Different layers carry different information, so the projection gets more directions to use.
- Training, step 1: freeze the backbone, fit the projection with ridge regression.
- Training, step 2: train backbone and projection together. Loss: squared distance to Stella's vector for the same text.
  - Queries are matched to Stella's query vectors. Passages are matched to Stella's document vectors. Three query batches for every passage batch.
  - No relevance labels needed: Stella supplies every target.
- Data: 3.49M unique queries in 12 forms (questions, product searches, claims, keywords, titles, and seven generated forms). About 200 million examples seen.
- Cost: 57.3 A100-hours, about $95, once. MS MARCO used only for validation, never for training.
Next: so what do the three keep of Stella's ranking?

## bench · ~2 min
Why it matters: is the cheap encoder good enough? Compare it with what you already run.
- [Grey dots] The 10 most-downloaded open embedding models on Hugging Face, each on its own index. Up is better ranking on six BEIR datasets. Right is more compute per query.
- [Nano] Ranks with bge-base and bge-large at the compute of the small models, and searches the index you already have.
- [Zero] Off the transformer scale on the left. No network at all, and still above both MiniLM models, including the most downloaded one.
- Add a BM25 search and fuse the lists, and Zero reaches 0.489, for the price of a sparse search per request.
- Caveat: Stella saw FiQA and ArguAna in training. On the other four datasets Zero trails BM25, 0.410 to 0.441. Judge on your own traffic.
Next: quality is half the story. What does each one cost to run?

## cloud · ~1 min
Why it matters: this is the line item your finance team sees.
- One GCP machine with 8 vCPUs ($0.395/hour), pushed as fast as it goes, one query per batch.
- CPU time to encode one million queries: Stella 123 vCPU-hours ($6.07). Nano 4.8 ($0.24). Zero 2 vCPU-minutes ($0.0016).
- Nano uses 26 times less CPU than Stella. Zero uses 3,700 times less.
- The dollars assume a busy machine. Idle hours still cost money: size for your real rate.
Next: but encoding is not the whole request. Quick bet.

## bet · ~0.75 min
- [Read it] Zero encodes a query 51 times faster than Nano.
- [Click] How much faster is the whole search, from query in to passages out?
- [Take one spoken guess. Give Zoom a moment. Remember the guess.]
Next: let's see where the time goes.

## split · ~1.25 min
Why it matters: optimize the request, not the component.
- [Repeat the guess.] Median request on this laptop: Nano 2.04 ms, Zero 0.86 ms. 2.4 times faster, not 51.
- [Orange] Once encoding is nearly free, search is most of Zero's request. And Zero needs a wider search (coming up).
- [Bottom bar] Stella for scale: 20.8 ms, almost all encoding. It runs off the slide.
- Ranking: Stella 0.631, Nano 0.579, Zero 0.510 nDCG@10 at these settings. Faster has a price.
Next: what does that look like as a bill?

## cloud-e2e · ~1.25 min
Why it matters: the honest saving, with search included.
- Same NQ index on the same cloud machine, now paying for encode plus search.
- Per million queries: Stella $4.91, Nano $0.18, Zero $0.17. About 29 times cheaper.
- Encoding alone said 3,700 times. The search is now the bill.
- Nano and Zero tie: one search process tops out around 600 to 650 queries/s. More processes or shards move that ceiling. A cheaper encoder does not.
Next: why does Zero need more search?

## recovery · ~2.5 min
Why it matters: the search settings are part of the encoder choice.
- Two kinds of HNSW settings in Qdrant:
  - Set when the index is built: `m`, the links per passage (16, the default), and `ef_construct`, the candidates checked when choosing links (100). Change them and you rebuild.
  - Set per request: `hnsw_ef`, the candidates kept during the search walk. Wider finds more true neighbours and takes longer.
  - Oversampling, with quantization: 1-bit binary codes in RAM, full vectors on disk. Fetch 4 times the results with the codes, then rescore with the full vectors.
- [Curves] Each line: of the encoder's own exact top 10, how much does the search find?
- [Circles] Zero at ef 1024 still finds less than Stella at ef 32.
- Why: the graph's links were built from Stella's vectors. Zero's query vectors land a little off, so the walk needs more room. Across 25 models in the paper, tables needed ef 256 to match their teacher at 64.
- So: same graph, but each encoder gets its own ef and oversampling per request.
Next: the encoder and the search width are one decision, and the best answer changes with traffic. So I let a router decide.

## recipe · ~2 min
Why it matters: you can run this policy on your own data.
- [01 Measure] Before traffic: score each encoder at each search width on judged queries. That is this table. At startup, time each encoder and each search on the machine.
- [02 Set] Two rules, in order. Quality first: the average score of what is served stays above the floor. Then latency: stay under the p95 target if the machine can.
- [03 Route] Every 250 ms, look at traffic, and pick the encoder mix plus one search width with the best average quality that fits the machine.
  - Example: at 1,500 searches/s, about three quarters Nano, one quarter Zero, ef 512.
- Advantages: quiet hours get the best ranking, peaks stay served, nothing is dropped, nothing is re-encoded.
- Only possible because all three search one index.
- To do it yourself: a few hundred judged queries from your traffic, quality per encoder and ef, timings on your hardware.
Next: let me show you.

## live · ~4 min
[Before the talk: run DATASET=nq .venv/bin/python m15/talk/demo/studio/app.py, open http://127.0.0.1:8767, wait until the subtitle stops saying "starting", drag traffic to about 1,500 once to warm up, then back to 40. Routing on Auto, quality floor 0.50, p95 target 150 ms. Space pauses; up and down arrows step traffic by 1.5x.]
- [Share the studio.] A search service on this laptop: 2.68M passages, one Stella index, real NQ questions, every answer scored against judged answers.
- Two rules: quality floor first, then the latency target. It never drops a query.
- [40/s] Quiet: everything on Stella with the widest search.
- [Drag to about 600] 15 times the traffic. Within a second it moves to Nano. Latency stays under the line; quality dips a little.
- [Drag to about 1,500] Zero mixes in, and the search narrows by itself. Same index, nothing re-encoded.
- [Drag to about 3,500] Past this laptop's ceiling, about 2,600/s in one process. It keeps the floor and serves every query, so the queue grows. Quality first, as asked.
- [Drag back to 40] The backlog drains in seconds; Stella gets its traffic back.
- [Optional: MORE, agent tasks] Ten searches per task: the gauge now times the slowest of the ten.
- Honest framing: questions and judgments are real, the traffic pattern is simulated, capacity is this laptop. On your hardware the numbers move; that is why you measure.
Next: four things to take home.

## takeaways · ~1 min
- [01] Measure the whole request. 51 times faster encoding became 2.4 times faster search.
- [02] Pick the encoder and its search width together. Zero needs a wider search.
- [03] Set the quality floor before you tune. It decides which encoder ships.
- [04] One index lets every request pick its encoder. That makes routing free to try.
Next: here is how little code it takes.

## close · ~1 min
- [Comment 1] Index once with Stella's document model in FastEmbed.
- [Comment 2] Query the same collection with Zero. To try Nano, change one model name.
- [pip line] FastEmbed 0.9.0 or later.
- Monday: pull a batch of real queries, judge the passages you want back, compare the encoders on your index.
[Advance to the thank-you slide.]

## thanks · ~0.5 min
[Leave this up through Q&A.]
- Thank you. Paper preview, model card, and FastEmbed are on the QR codes. Find me on LinkedIn.
- Happy to take questions.
