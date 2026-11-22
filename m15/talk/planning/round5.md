Round 5. Owner approved the agreed plan (PLAN_agreed.md) with two notes: (a) the room is search engineers, so a very high technical bar, but interesting rather than academic: depth goes into mechanisms (Zero as a term-weight lookup that emits a dense vector, the one-line ridge fit, why the HNSW walk needs wider ef, binary quantization + rescoring, "measure ranking not cosine"), statistics stay off slides; (b) "think big": we may run new experiments and bigger demos; we are not limited to existing assets. Budget: about 2.5 working days, cloud spend under $10 per run without review (ceiling $1,000 total).

Also: the submitted abstract promises edge retrieval with on-device query encoders ("documents by large cloud-side models, queries by substantially smaller on-device models"). The plan underplays that.

Candidate big additions. Rank them by (attendee value x feasibility in 2.5 days), and kill the weak ones:
B1. Edge chapter beat + demo lane: Qdrant Edge (in-process shard) on the laptop with Wi-Fi off, Zero encoding, same Stella vectors. Footprint dial: Zero 90 MiB, Nano 132 MiB, Stella 1.7 GiB assets.
B2. "Built during this talk": start a ridge table fit for a second open model on the M5 Pro when the talk starts; reveal its time and its retrieval score in the recipe chapter. Pre-measured as backup.
B3. Zero for a large modern open teacher with its own tokenizer (e.g., Qwen3-Embedding-0.6B or 4B, Apache-2.0), ridge-fit on a rented GPU: "a lookup-table query encoder for a 4B-parameter index". Uncertain result; reported whatever happens.
B4. Throughput-at-scale slide: cores (and $/month) to serve 1,000 QPS of query encoding per encoder on one priced cloud VM, measured sustained throughput at a p95 limit.
B5. Agentic workload framing: agents issue many queries per task, so query-side cost multiplies. A slide only, or a tiny measured scenario?
B6. Hosted try-it page on a VM + Qdrant Cloud after the talk (QR).
Also propose anything bigger and better that serves the thesis "Keep your index. Make the query encoder a dial." Be concrete about the visual each addition earns and where it sits in the seven chapters without breaking one-idea-per-chapter. Concise.
