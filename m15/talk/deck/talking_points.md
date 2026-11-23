# Talking points

One section per slide, in talk order. `rebuild.py` copies each section into that slide's speaker notes. Lines in [brackets] are stage directions. Every number here has a receipt in `m15/EVIDENCE.md` or the paper.

## cover · ~0.5 min
- Hook: where does the time go in a semantic search request? [Let the room guess.]
- The usual fix, a smaller model, means re-embedding the whole corpus. So most teams never try it.
- This talk: cut the query-side cost and keep your index exactly as it is.
Next: here is the answer, timed on one real request.

## cost · ~1.5 min
Why it matters: one model does two jobs with opposite deadlines. Only one has a user waiting.
- [Top lane] Indexing: every chunk goes through the embedding model once. Here, 2.68M passages from Natural Questions through Stella: 5.3 GPU-hours on my desktop, paid up front.
[Click: the question appears.]
- [Bottom lane] Search: every question goes through the same 435M-parameter model. "Who played Joey's agent Estelle on Friends": 20.5 ms on this laptop.
- [Red walk] Its vector walks the graph to the right passage out of 2.68 million in 0.23 ms. Encoding takes about 90 times longer than the search, on every request.
- It never stops: at 100 queries/s, query encoding burns a whole index worth of compute every 27 hours, and agents fire several searches per answer.
- So the cost worth cutting is the bottom lane, if a cheaper model can still walk the same graph.
Next: what if only the query side changed?

## arch · ~1.5 min
Why it matters: you keep the index and change only the query side.
- [Top row] The index from the last slide, built once with Stella's document encoder. It stays exactly as it is.
- [Three rows] Each request picks one query encoder. All three write into the same vector space and search the same index:
  - Stella's query encoder: 435M parameters, 25 GFLOPs per query (25 billion arithmetic operations).
  - Nano: a 34.5M-parameter transformer, 0.54 GFLOPs, about 46 times less.
  - Zero: a lookup table, no neural network. 28 KFLOPs, nearly a million times less.
- Zero's table is trained for one frozen index: its only job is to put each query where Stella would. That sets it apart from general static embedding models.
- Together they are Constella. Switching is free: no second index, no re-encode, and you can switch back on the next request.
Next: how can a lookup table produce a query vector with no matrix multiplication?

## zero-mech · ~1.5 min
- The table: one learned vector for every token in Stella's vocabulary, 31k tokens. Stored as 8-bit integers so the whole table is about 31mb
- Step 1: We use Stella's tokenizer to get the token for each word
- Step 2: we lookup each matching vector in the table.
- Step 3: average those vectors, weighted by the square root of each token's count (more value to repeated words), then normalize.
- [Two vectors] For this query, Zero's vector has cosine 0.88 with Stella's: not identical, but pointing the same way.
Next: random rows would not do that. Here is how it learned.

## distill · ~1.5 min
- [Open] The main ideas here. At the end I'll share the paper with the full training details
- Stella is the teacher: Zero's goal is to imitate
- Start: each row is Stella's own vector for that single token.
- [Align] Copy Stella: run a query through both, compare the vectors, and nudge Zero's rows closer. Millions of times.
- [Rank] Then learn from label queries: nudge the rows so the correct chunk scores above the wrong ones.
- [Table] The result: the 31 MB table. About 20 minutes on one RTX 3080
Next: I assumed the best retriever would make the best table. It did not.

## worst · ~1.25 min
Why it matters: a leaderboard score does not tell you which teacher to pick.
- Picked Stella first of all for licenses and training dataset disclosure, easy to avoid contamination
- I fitted a quick lookup table on multiple models
- gte-large: 1st as a retriever, 10th as a table. Not sure why, carry lots of context into the vectors
- Tested 26 models in total and rank correlation was pretty poor
- Lesson: The leaderboard tells you how good a model is today, not how cheap its queries can get later.
Next: even with the best teacher, a lookup table has limits you should know.

## order · ~2 min
Why it matters: know which queries Zero gets wrong before your users do.
- Each of Zero's three steps (split into tokens, look up one row each, average) loses something.
- [Order] An average has no order: "paris to rome" and "rome to paris" give the same vector. Hard for most models, impossible for Zero.
- [Context] "bank" is the same row in "river bank" and "bank loan". A transformer reads the neighbouring words through attention; Zero is unable to do that
- [New words] The vocabulary is fixed: "qdrant" becomes q, ##dran, ##t, rows learned from other words. A transformer recombines pieces; Zero averages them.
- Takeaway: Zero is a bag of tokens and acts like one. For queries that needs order and context back, you need attention. That is Nano.

## nano · ~1.5 min
- Start from bge-small, a 12-layer324.5M parameters embedding model. Its vectors have 384 embeddings; Stella's have 1,024.
- One matrix alone cannot fill Stella's dimensions. So Nano reads three of the models layers (4, 8, and 12), joins them into 1,152 embeddings, and maps those to 1,024.
- Training: make Nano's vector match Stella's for the same text, on 3.49M queries. No relevance labels: Stella supplies every target.
- Cost: 57.3 A100-hours, about $95, once.
Next: so what do the three keep of Stella's ranking?

## bench · ~2 min
Why it matters: is the cheap encoder good enough?
- We tested 25 models, same Qdrant configuration, six BEIR dataset, picked diverse ones
- Up: better ranking, Right: more compute per query. Stella is very good but very expensive
- [Nano] Ranks like bge-large with 22 times less compute
- [Zero] No neural network at all, and still above both MiniLM models. Fused with BM25, it reaches 0.489.
- Caveat: Stella saw FiQA and ArguAna in training. On the other four datasets, Zero trails BM25 (0.410 vs 0.441).
Next: quality is half the story. How fast is each one? Quick bet.

## cloud · ~1.5 min
Why it matters: a cheaper encoder only saves what encoding was costing you.
- With Stella, encoding was almost the whole bill. Make it nearly free, and the next cost in line, the search, becomes the bill.
- [Arrow] Zero's encoding is 3,700 times cheaper than Stella's. The whole request is 29 times cheaper, and Nano lands in the same place at 27 times.
- Still a big win: per million queries, Stella $4.91, Nano $0.18, Zero $0.17.
- General lesson: once the encoder is cheap, the next savings come from the search side: more search processes or shards, quantization, and the search width.
Next: and Zero needs more search than the others.

## bet · ~0.75 min
- [Read it] Zero encodes a query about 650 times faster than Stella, and 51 times faster than Nano. Same laptop, same NQ queries, median encode time.
- [Click] How much faster is the whole search, from query in to passages out?
- [Take one spoken guess. Give Zoom a moment. Remember the guess.]
Next: let's see where the time goes.

## split · ~1.25 min
Why it matters: optimize the request, not the component.
- [Repeat the guess.] Median request on this laptop: Nano 2.04 ms, Zero 0.86 ms. 2.4 times faster, not 51.
- [Orange] With encoding nearly free, search is most of Zero's request, and Zero needs a wider search (coming up).
- [Wrapped rows] Stella on the same scale: 20.8 ms, wrapping ten times, almost all encoding. The orange search sliver is at the very end.
- Ranking at these settings: Stella 0.631, Nano 0.579, Zero 0.510 nDCG@10. Faster has a price.
Next: now the bill, starting with encoding alone on a cloud machine.

## cloud-e2e · ~1.25 min
Why it matters: the honest saving, with search included.
- Same NQ index and cloud machine, now paying for encode plus search: Stella $4.91, Nano $0.18, Zero $0.17 per million queries. About 29 times cheaper.
- Encoding alone said 3,700 times. The search is now the bill.
- Nano and Zero tie because one search process tops out around 600 to 650 queries/s. More processes or shards move that ceiling; a cheaper encoder does not.
- So why keep Zero? Not for price at this rate. After the demo, one chart shows where it earns its place.
Next: why does Zero need more search?

## recovery · ~1.5 min
Why it matters: a cheaper encoder can give back part of its savings in search.
- The graph was built from Stella's vectors. Zero's queries land close to where Stella's would, not exactly there, so the search takes more wrong turns.
- The fix is a wider search, set per request with hnsw_ef. Same graph, no rebuild.
- [Chart] With an 8 times wider search, Zero keeps as much of its own quality as Stella does at its narrowest.
- Takeaway: the encoder and the search width are one decision. Pick them together.
- If asked: this is relative to each encoder's own best, and Zero's best is lower. On the stricter top-10 measure, Zero still trails: 90% at ef 1024, against Stella's 91% at ef 32.
Next: and the best decision changes with traffic. Let me show you a router that makes it live.

## recipe · ~2 min
Why it matters: you can run this policy on your own data.
- [01 Measure] Before traffic, score each encoder at each search width on judged queries: this table. At startup, time each encoder and each search on the machine.
- [02 Set] One rule: stay under the p95 target, with the best ranking that fits.
- [03 Route] Every 250 ms, pick the encoder mix with the best average quality that fits the current traffic. Each encoder keeps its own ef. At 1,500 searches/s: about three quarters Nano, one quarter Zero.
- Quiet hours get the best ranking, peaks stay served, nothing is dropped or re-encoded. Only possible because all three search one index.
- To do it yourself: a few hundred judged queries from your traffic, quality per encoder and ef, timings on your hardware.
Next: let me show you.

## live · ~4 min
[Before the talk: run DATASET=nq .venv/bin/python m15/talk/demo/studio/app.py, open http://127.0.0.1:8767, wait until the subtitle stops saying "starting", drag traffic to about 1,500 once to warm up, then back to 40. Routing on Auto, p95 target 150 ms. Space pauses; up and down arrows step traffic by 1.5x.]
- [Share the studio.] A search service on this laptop: 2.68M passages, one Stella index, real NQ questions scored against judged answers.
- One rule: the best ranking that still meets the latency target. It never drops a query.
- Each encoder keeps its own search settings, like any model. Every 250 ms the router only moves traffic between them: the best average quality that fits the machine. Only possible because all three search one index.
- [40/s] Quiet: everything on Stella.
- [Drag to about 600] 15 times the traffic. Within a second it moves to Nano. Latency holds; quality dips a little.
- [Drag to about 1,500] Zero takes the traffic Nano cannot encode in time. Same index, nothing re-encoded.
- [Drag to about 2,500] Past this laptop's limit, about 2,000 searches/s, it still serves every query and the queue grows. More traffic needs another shard, not a cheaper encoder.
- [Optional, at 2,000: Zero's ef to 1024] A wider search costs search time. The laptop now carries less, and the router moves traffic back to Nano.
- [Drag back to 40] The backlog drains in seconds; Stella gets its traffic back within about 20 seconds.
- [Optional: MORE, agent tasks] Ten searches per task: the gauge now times the slowest of the ten.
- Honest framing: questions and judgments are real, the traffic is simulated, the capacity is this laptop's. On your hardware the numbers move; that is why you measure.
- To run it yourself: a few hundred judged queries from your traffic, the quality of each encoder at each search width, and timings on your hardware.
Next: the whole range on one chart.

## plan · ~0.75 min
If Zero cost the same as Nano, why would you use it?
- Every encoder has a throughput ceiling: how many queries one machine can encode per second. On my laptop, Nano does about 1,400, Zero about 17,000.
- Below that ceiling, use Nano: it ranks better than Zero and costs about the same per query. Quiet hours can even afford Stella.
- Above it, Zero keeps every query served, instead of queueing or adding machines. The price is quality: the average NDCG goes from 0.62 when quiet to 0.52 at 2,500 searches/s.
- In general: pick Zero wherever encoding capacity is the bottleneck: traffic peaks, CPU-only hosts, small devices. And because all three share one index, you never have to pick just one.
Next: how to use it.

## close · ~1 min
Why it matters: you can try this on Monday with a few lines of code.
- [Comment 1] Index once with Stella's document model in FastEmbed.
- [Comment 2] Query the same collection with Zero. For Nano, change one model name.
- [hnsw_ef=512] Zero needs the wider search from the recovery slide. Set it per request.
- [pip line] FastEmbed 0.9.0 or later.
- Then judge the three encoders on real queries against your own index.
Next: and if your index was built with a different model?

## byo · ~1 min
Why it matters: this works for any frozen index.
- [01] Keep your index. Nothing is re-encoded.
- [02] Your current query model is the teacher: encode real queries with it, and its vectors become the targets.
- [03] Train Zero, Nano, or both. First fit a quick table to check your model is easy to copy (remember gte-large).
- [04] Measure on your own queries, then pick each encoder's search width.
- [QR] Everything else, data, losses, and what failed, is in the paper.
[Advance to the thank-you slide.]

## thanks · ~0.5 min
[Leave this up through Q&A.]
- Thank you. Paper preview, model card, and FastEmbed are on the QR codes. Find me on LinkedIn.
- Happy to take questions.
