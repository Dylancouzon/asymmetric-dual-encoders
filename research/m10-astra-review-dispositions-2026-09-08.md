# gpt-6-astra whole-plan review, 2026-09-08 — findings and dispositions

Brief: `research/m10-whole-plan-brief-2026-09-08.md` (discovery-framed, our interpretation stripped
out, W14 and the frozen-tower question deliberately omitted so the reviewer could not be anchored).
Verbatim output: `research/m10-astra-whole-plan-2026-09-08.log`. 6,314 tokens.
Read-exclusion honoured; log audited before findings were read — clean.

**It did not rediscover W14 or the frozen-tower debate.** It went somewhere else entirely, which is
what the brief was for.

---

## 1. VERIFIED, and it undercuts a headline comparison: there is no small-model + BM25 row

*"A hybrid beating a dense-only baseline cannot establish the architectural advantage."*

`zero` fused with BM25 is **0.4911**, reported against `LR-dense-pertask` 0.4583, OpenSearch
0.4868 and LightRetriever's own hybrid. Checked `results/FINAL_MATRIX.md`: it carries
`bge-small-en-v1.5` (dense alone) and `bm25` (alone, 0.4174) and LR's hybrid — **no
bge-small + BM25 fusion anywhere**. The release bar is bge-small at 0.5042 *dense*; its hybrid is
unmeasured and would presumably be higher. So our fused headline is compared against everything
except the obvious hybrid of the model we must beat.

**Not cheaply recoverable.** `results/perquery.json` does hold `bge-small-en-v1.5` and `bm25` on
identical qids for all six datasets — but as per-query **nDCG@10 values**, not ranked lists or raw
scores, and you cannot fuse two systems from their final metric. It needs fresh retrieval, and
CLAUDE.md records that those caches no longer exist.

**Disposition: DYLAN.** Adding a comparator to a frozen matrix after seeing our own numbers needs
his ruling. The argument for doing it: the omission *flatters us*, so adding it can only weaken our
claim, which makes it conservative rather than opportunistic — the same shape as M12's fusion
re-scoring. Cost is bge-small + BM25 over the six, no GPU-heavy work.

## 2. VERIFIED from our own numbers: nano buys no query-compute saving at all

*"nano using bge-small's backbone does not automatically buy cheaper query inference than
conventional bge-small."*

nano (F's winner) is **34,540,672** params on bge-small's backbone with a 1152-wide head.
`bge-small-en-v1.5` is **~33.4M**. nano is **larger** and runs the same backbone forward pass.

So nano's claim cannot be "near-zero query compute" — it is **"better quality at equal edge cost,
sharing stella's 1024d index"**. The north-star question, *how much quality survives as query
compute approaches zero*, is answered by **zero**, not by nano. nano is a different point on the
frontier: same cost as the baseline, higher quality, no re-indexing.

**Disposition: a CLAIM fix, not a research change** — and it must reach `CLAUDE.md`'s north star,
the M14 paper framing and nano's model card, because "near-zero-compute query encoder" applied to
nano is simply wrong. Recorded here; the wording change is owed before M13/M14.

## 3. VERIFIED gap, cheapest high-information item, and it needs no GPU

*"You may be minimizing arithmetic while paying in memory... benchmark the actual packaged table
and small model across cold starts and warm queries."*

What exists: `results/m9_edge_cost_Apple_M5_Pro.json` is a proper profile — `cold_load_ms`,
`peak_rss_mb`, `p50/p95_ms` per query-length bucket, artifact bytes — but only for
**nano-bge-small, nano-minilm-l6, mdbr-leaf-ir**. `results/costs.json` has bge-small's coarse
`latency_ms`/`load_s` with no peak RSS, cold start or p95. `results/m7_costs.json` has a `/table`
field with none of them.

So the frontier's **cost axis is unmeasured, under one harness, for the two points that matter
most: `zero` itself (the actual near-zero-compute point) and conventional `bge-small` (the bar).**
A lookup table can lose on download bytes, resident memory, cold page faults or tokenization while
winning on FLOPs, and nothing here would show it.

**Disposition: DO IT, session, no ruling needed.** Runs on the Mac (M5 Pro, free — the box is busy
with arms), needs no relevance labels, touches no protected evaluation, and can invalidate or
confirm the deployment premise in an afternoon. It extends an existing harness rather than
inventing one.

## 4. NEW AVENUE, unconsidered: compile the queries into the INDEX, not into a student

*"Where your generated queries retain source-document IDs, index those query strings as lexical
aliases for their documents."*

doc2query-shaped, and we already own the expensive input: **0.83M generated queries with source doc
IDs**, plus 1.25M harvested real forms. BM25 over aliases, aggregate to doc IDs, optionally fuse.
Near-trivial query path, **no student and no distillation at all**. The tradeoff nobody here has
priced: **index expansion versus query-model complexity** — and index size is an edge cost we do
track.

**Disposition: register as an avenue (M16-shaped), DYLAN to scope.** Two cautions on the record:
the generated queries were screened as *training* data, so using them as *index content* needs the
contamination rule re-applied (the reviewer flagged excluding evaluation-derived aliases), and it
does not serve stella's dense index, so it is a different product, not a drop-in.

## 5. THE DEEPEST REFRAME: Spearman ≈ 0 may mean compressibility is TRAINABLE

*"Your strongest table teacher may be one trained to be compressible... train a document
representation jointly with a lookup-only query encoder, then freeze it."*

We treat **Spearman ≈ 0 over eleven teachers** as a finding about teacher *selection* — "tower
quality does not predict table quality" — and it is currently M14's headline. The reviewer's
reading: it is a symptom that **no independently-trained tower is compressible into an additive
token representation, because none was trained to be**, and compressibility is a property you train
for rather than shop for. Under that reading, searching teachers was always going to return noise,
which is exactly what Spearman ≈ 0 says.

This connects to M8's `E14-LORA` (document-side co-adaptation, **authorised and never run**) and to
M16, which parks it because it costs the drop-in property.

**Disposition: FRAMING, and possibly M14's headline.** No compute now. It does not invalidate the
Spearman result; it changes what the result *means*, which is worth more than the number.

## 6. WHERE THE REVIEWER IS WRONG — its #1, on a misreading

It argues the least-questioned premise is *"query encoding belongs on the edge"*, because
*"your index already requires a server"*. **It does not.** The architecture is: the server indexes
documents **once** at ingestion; the **edge client holds the document index** plus the query path
(that is what the Qdrant Edge prototype is). There is no query-time server, so server-side query
encoding would defeat the offline premise rather than being a cheaper alternative to it.

The salvageable half is a fair demand for precision: say explicitly whether the requirement is
near-zero **edge** compute or near-zero **total** compute. Given finding 2, that distinction is
load-bearing.

## 7. NOTED, low value here

Query-result caching and amortized cost (`E[C] = h·C_hit + (1-h)·C_miss`). Deployment-dependent, we
have no traffic logs, and the reviewer itself says not to pretend BEIR supplies a hit rate. Worth
one line in the report as a deployment consideration, not an experiment.

---

## What the review is actually saying, in one line

*"Gate further training spend on finding an operational region where asymmetry wins."* Findings 1,
2 and 3 all point the same way: we have been measuring **quality** carefully and **cost** loosely,
and the architectural claim lives or dies on the cost axis we have not finished measuring.
