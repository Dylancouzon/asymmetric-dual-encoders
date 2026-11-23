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
- Zero's table is trained for one frozen index: its only job is to put each query where Stella would. That sets it apart from general static embedding tables.
- Together they are Constella. Switching is free: no second index, no re-encode, and you can switch back on the next request.
Next: how can a lookup table produce a query vector?

## zero-mech · ~1.5 min
- The table: one learned vector of 1,024 numbers for every token in Stella's vocabulary, 30,522 tokens (words and word pieces). Stored as 8-bit integers with one scaling factor per row: 31 MB, small enough to sit next to the search process.
- Step 1: Stella's tokenizer splits the query into tokens from that vocabulary.
- Step 2: each token looks up its vector in the table.
- Step 3: average those vectors, weighted by the square root of each token's count (a repeated word counts a bit more), then normalize.
- [Two vectors] For this query, Zero's vector has cosine 0.88 with Stella's: not identical, but pointing the same way.
Next: random rows would not do that. Here is how it learned.

## distill · ~2 min
Why it matters: the table learns to imitate Stella without touching the index. In distillation terms, Stella is the teacher: the model being imitated.
- Start: each row is Stella's encoding of that single token. Rare tokens start with more weight (IDF).
- Phase 1, Align (16,000 steps of 512 examples), against Stella's query vectors, computed once and saved:
  - Point the same way as Stella's vector (loss: 1 − cosine).
  - For labeled queries, give the relevant passage and 31 other candidates the same relative scores Stella gives them (KL divergence). A vector can look close to Stella's and still rank a different passage first.
- Phase 2, Rank (2,500 steps): contrastive training (InfoNCE) on 338,076 labeled query-passage pairs. The relevant passage must outscore 32,768 wrong ones from a fixed pool. Candidates Stella scores close to or above it are dropped: they may answer the question too.
- [Formula line] Ê_D, Stella's document vectors, never changes.
- Cost: about 20 minutes on one RTX 3080 once Stella's vectors are saved. A different teacher needs 8 to 12 hours to compute them first. All training sources allow commercial use of the trained weights.
Next: I assumed the best retriever would make the best table. It did not.

## worst · ~1.25 min
Why it matters: a leaderboard score does not tell you which teacher to pick.
- I fitted a quick lookup table (a few minutes on one A100) to 10 strong models, each tested on its own index.
- [Red line] gte-large: 1st as a retriever, 10th as a table (0.597 down to 0.246).
- [White line] Stella: 2nd as a retriever, 1st as a table. That is why Constella uses Stella.
- Across all 26 models I tried, the two rankings barely relate (rank correlation +0.09).
- Lesson: if you might want a cheap query encoder later, test it on a small index before you encode the whole corpus.
Next: even with the best teacher, a lookup table has limits you should know.

## order · ~2 min
Why it matters: know which queries Zero gets wrong before your users do.
- Each of Zero's three steps (split into tokens, look up one row each, average) loses something.
- [Order] An average has no order: "paris to rome" and "rome to paris" give the same vector. On 24 query pairs where direction changes the answer, right passage first in both directions: Zero 0, Nano 2, Stella 4. Hard for everyone, impossible for Zero.
- [Context] "bank" is the same row in "river bank" and "bank loan". A transformer reads the neighbouring words through attention; Zero only has the other rows in the average.
- [New words] The vocabulary is fixed: "qdrant" becomes q, ##dran, ##t, rows learned from other words. A transformer recombines pieces; Zero averages them.
- Takeaway: Zero fits short keyword queries. Test phrased questions separately.
Next: to get order and context back, you need attention. That is Nano.

## nano · ~2 min
Why it matters: a small transformer whose vectors land in Stella's space, so they search Stella's index.
- Start: bge-small, a 12-layer embedding model with 384-dimensional vectors. Stella's index holds 1,024 dimensions.
- Problem: one matrix can stretch 384 numbers to 1,024 but cannot add information. The output still spans only a 384-dimensional slice of Stella's space.
- Fix: average the token vectors of layers 4, 8, and 12 and join them (3 × 384 = 1,152), then map to 1,024 with one matrix. Different layers capture different things, so the matrix has more to work with.
- Training:
  - Step 1: keep bge-small fixed and fit only the matrix, by ridge regression (least squares with a penalty on large weights).
  - Step 2: train everything on the squared distance to Stella's vector for the same text. Queries match Stella's query vectors, passages its document vectors, three query batches per passage batch. No relevance labels: Stella supplies every target.
- Data: 3.49M unique queries in 12 forms (questions, product searches, claims, keywords, titles, and seven generated forms). About 200 million examples seen.
- Cost: 57.3 A100-hours, about $95, once. MS MARCO only for validation, never for training.
Next: so what do the three keep of Stella's ranking?

## bench · ~2 min
Why it matters: is the cheap encoder good enough compared with what you run today?
- [Grey dots] The 10 most-downloaded open embedding models on Hugging Face, each on its own index. Up: better ranking on six BEIR datasets. Right: more compute per query.
- [Nano] Ranks with bge-base and bge-large at small-model compute, on the index you already have. Against bge-large: 0.532 vs 0.533 nDCG@10, at 0.54 vs 12.1 GFLOPs per query, about 22 times less.
- [Zero] Left of the transformer scale, with no network, and still above both MiniLM models, including the most downloaded one.
- Fuse Zero with a BM25 search and it reaches 0.489, for the price of one sparse search per request.
- Caveat: Stella saw FiQA and ArguAna in training. On the other four datasets Zero trails BM25, 0.410 to 0.441. Judge on your own traffic.
Next: quality is half the story. How fast is each one? Quick bet.

## cloud · ~1 min
Why it matters: CPU time is what you pay for.
- One GCP machine with 8 vCPUs, pushed as fast as it goes, one query per batch.
- CPU time to encode one million queries: Stella 123 vCPU-hours. Nano 4.8. Zero 2 vCPU-minutes.
- Nano uses 26 times less CPU than Stella. Zero uses 3,700 times less.
- This assumes a busy machine. Idle hours still cost money: size for your real rate.
Next: encoding is not the whole request. Same machine, encode plus search.

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

## recovery · ~2.5 min
Why it matters: the search settings are part of the encoder choice.
- Qdrant's HNSW settings come in two kinds:
  - Set when the index is built: `m`, the links per passage (16, the default), and `ef_construct`, the candidates checked when choosing links (100). Change them and you rebuild.
  - Set per request: `hnsw_ef`, the candidates kept during the search walk. Wider finds more true neighbours and takes longer.
  - Oversampling, with quantization: 1-bit binary codes in RAM, full vectors on disk. Fetch 4 times the results with the codes, then rescore with the full vectors.
- [Curves] Each line: the share of the encoder's own exact top 10 that the search finds.
- [Circles] Zero at ef 1024 still finds less than Stella at ef 32.
- Why: the graph's links come from Stella's vectors, and Zero's queries land slightly off, so the search must keep more candidates to find the same neighbours. Across 25 models in the paper, lookup tables needed ef 256 to match their teacher at ef 64.
- So: one graph, with ef and oversampling set per encoder, per request.
Next: encoder and search width are one decision, and the best one changes with traffic. So I let a router decide.

## recipe · ~2 min
Why it matters: you can run this policy on your own data.
- [01 Measure] Before traffic, score each encoder at each search width on judged queries: this table. At startup, time each encoder and each search on the machine.
- [02 Set] Two rules, in order. Quality first: the average score served stays above the floor. Then latency: stay under the p95 target if the machine can.
- [03 Route] Every 250 ms, pick the encoder mix and one search width with the best average quality that fits the current traffic. At 1,500 searches/s: about three quarters Nano, one quarter Zero, ef 512.
- Quiet hours get the best ranking, peaks stay served, nothing is dropped or re-encoded. Only possible because all three search one index.
- To do it yourself: a few hundred judged queries from your traffic, quality per encoder and ef, timings on your hardware.
Next: let me show you.

## live · ~4 min
[Before the talk: run DATASET=nq .venv/bin/python m15/talk/demo/studio/app.py, open http://127.0.0.1:8767, wait until the subtitle stops saying "starting", drag traffic to about 1,500 once to warm up, then back to 40. Routing on Auto, quality floor 0.50, p95 target 150 ms. Space pauses; up and down arrows step traffic by 1.5x.]
- [Share the studio.] A search service on this laptop: 2.68M passages, one Stella index, real NQ questions scored against judged answers.
- Two rules: quality floor first, then the latency target. It never drops a query.
- [40/s] Quiet: everything on Stella, widest search.
- [Drag to about 600] 15 times the traffic. Within a second it moves to Nano. Latency holds; quality dips a little.
- [Drag to about 1,500] Zero mixes in and the search narrows by itself. Same index, nothing re-encoded.
- [Drag to about 3,500] Past this laptop's ceiling, about 2,600/s in one process. It keeps the floor and serves every query, so the queue grows: quality first, as asked.
- [Drag back to 40] The backlog drains in seconds; Stella gets its traffic back.
- [Optional: MORE, agent tasks] Ten searches per task: the gauge now times the slowest of the ten.
- Honest framing: questions and judgments are real, the traffic is simulated, the capacity is this laptop's. On your hardware the numbers move; that is why you measure.
Next: the whole range on one chart.

## plan · ~0.5 min
Why it matters: this is when to pick Zero over Nano.
- [Chart] The router's plan at every traffic level on this laptop. The demo showed a few points; this is the whole range.
- [Stella, then Nano] Quiet hours go to Stella. From about 100 searches/s Nano takes a growing share; by 1,000 it carries everything.
- [Dashed line] Around 1,000 searches/s Nano reaches its encoding capacity: about 1,400 queries/s on this laptop, against Zero's 17,000.
- [Zero] Past the line, Zero takes the traffic Nano cannot encode in time, and the search narrows to hold latency (ef along the top).
- So pick Zero when encoding capacity runs out: traffic peaks, CPU-only hosts, small devices. Below that, Nano ranks better for about the same cost.
- Average nDCG@10 falls from 0.636 at 40/s to 0.524 at 2,500/s: the price of serving every query.
Next: how to use it.

## close · ~1 min
Why it matters: you can try this on Monday with a few lines of code.
- [Comment 1] Index once with Stella's document model in FastEmbed.
- [Comment 2] Query the same collection with Zero. For Nano, change one model name.
- [hnsw_ef=512] Zero needs the wider search from the recovery slide. Set it per request.
- [pip line] FastEmbed 0.9.0 or later.
- Then judge the three encoders on real queries against your own index.
Next: and if your index was built with a different model?

## byo · ~1.25 min
Why it matters: the method works for any frozen index. You can train cheap query encoders for yours.
- [01] Keep your document encoder and your index. Nothing is re-encoded, ever.
- [02] Encode many real queries with your current query model, including its prompt and preprocessing: those vectors are the targets. For Nano, also encode passages with your document model.
  - A hosted model works if it returns the same vectors.
  - First fit a quick lookup table to check that your model is easy to imitate (remember gte-large).
- [03] Train:
  - Zero: one row per token of your tokenizer, each starting from your model's vector for that token, then the same two phases. About 20 minutes on one GPU once the targets exist; for Stella, computing them took 8 to 12 hours.
  - Nano: a small embedding model, three averaged layers, one matrix to your vector size, trained to reproduce your model's vectors. No relevance labels. Ours cost about $95 of A100 time.
- [04] Judge on your own queries: exact nDCG@10 per encoder, then sweep ef and oversampling. Set your floor and let a router pick.
- [QR] The paper has every detail: data, losses, steps, and what failed.
[Advance to the thank-you slide.]

## thanks · ~0.5 min
[Leave this up through Q&A.]
- Thank you. Paper preview, model card, and FastEmbed are on the QR codes. Find me on LinkedIn.
- Happy to take questions.
