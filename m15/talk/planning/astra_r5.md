**Make offline retrieval the chapter 5 demo.** Let someone in the room supply a query. Run it through Zero and Nano on the laptop, with the same Qdrant Edge shard open. Keep the shard ID fixed on screen as the ranked passages change. That delivers the abstract’s on-device promise and makes “keep your index” visible.

| Rank | Addition | Decision and placement |
|---|---|---|
| 1 | **B1. Qdrant Edge** | Build first. In chapter 2, draw one boundary around the laptop and put document encoding outside it. Label the encoder files accurately: Zero 90 MiB, Nano 132 MiB, Stella 1,670 MiB. In chapter 5, use a separate Zoom host so the demo laptop can go offline. |
| 2 | **B4. Throughput and cost** | Run one VM experiment under $10. Chapter 6 earns one row per encoder showing sustained **completed searches/s**, p95, nDCG@10, and cost per million searches. A 1,000-QPS core estimate from encoding alone misses the bottleneck the talk just demonstrated. |
| 3 | **B2. Fit during the talk** | Stretch only. Rehearse the second teacher’s ridge fit and exact evaluation on the M5 Pro. Disclose cached teacher vectors at launch. Reveal elapsed time and judged retrieval score in chapter 6 only if the job finishes before the live demo. The paper’s four-to-seven-minute fits used an A100. |
| 4 | **B5. Agentic workload** | Give it one sentence in chapter 1. “One user task can trigger several uncached searches. Each one pays for query encoding.” Cut the slide and measurement. |
| 5 | **B3. Large teacher** | Defer. A new tokenizer, teacher targets, and a separately encoded document index would consume rehearsal time to answer a different question. |
| 6 | **B6. Hosted demo** | Cut. Use the chapter 6 QR for the released encoders and local example. |

**The better addition is a second control on the demo screen: `ef`.** Hold the query and encoder fixed, widen `ef`, and show how many of that encoder’s exact top ten the graph recovers. Then switch encoders on the same graph. Keep the existing million-passage timing replay labeled separately from live Edge measurements.

That sets up a stronger chapter 6 payoff. Apply **one shared relevance floor and p95 limit** to the VM rows, then reveal the cheapest qualifying setting. If Zero misses the floor, leave that result visible. The audience gets a decision it can repeat on its own index.