OpenAI Codex v0.153.4
--------
workdir: /home/dylan/asymetric-dual-encoders
model: gpt-6-astra
provider: openai
approval: never
sandbox: read-only
reasoning effort: high
reasoning summaries: none
session id: 01a08357-a9ed-7651-acfe-35defbbc8cfe
--------
user
# What are we not seeing?

One-shot review, tight credit budget. **Do not validate our reasoning — we have three other
reviewers for that.** Your job is to find what nobody on this project has considered at all:
missing options, wrong premises nobody questioned, cheaper paths to the same goal, or a better goal.

## READ-EXCLUSION (mandatory)
Never read: `results/frozen_eval/untouched-*`, any reserved qrels cache, `work/m9reserve`,
`results/perquery.json`. No repo-wide grep. Read at most `CLAUDE.md` and `m10/STATUS.md` if you
want grounding; everything you need is below. Name any other file you want instead of reading it.

## Facts, without our interpretation

**Goal.** Decide, with defensible numbers, whether an asymmetric dual encoder (large frozen
document encoder server-side, near-zero-compute query path on the edge) beats the obvious
alternative: running a small 20–100M conventional embedding model on the edge. Research question:
how much retrieval quality survives as query-side compute approaches zero.

**Architecture.** Documents indexed once by frozen `stella_en_400M_v5` (1024d). Two query paths
serve that same index: **zero** = token→vector lookup table, no transformer (shipped);
**nano** = ≤35M distilled transformer (in progress).

**Measured, on 6 BEIR-family datasets (nDCG@10), exact search:**
- zero: 0.4339 alone; **0.4911 fused with BM25** (convex), 0.4887 with a Qdrant-implementable
  operator — a tie. Comparator `LR-dense-pertask` 0.4583; OpenSearch 0.4868.
- Released artifact is vocab×dim int8; competing LightRetriever table is 466 MB.
- nano attempt 1: 82.2% teacher retention overall, **93.8% on NQ vs 50–71% on two CQADupStack
  components**, stopped at 3.74B tokens. Release bar is bge-small at 0.5042; aim 0.5155.
- nano attempt 2 (running): 1152-wide head pooled from 3 layers, cyclic LR, 1.25M harvested real
  query forms + 0.83M generated, ~16.8B tokens. Student selected today: bge-small (34.5M) beat
  MiniLM-L6 (23.9M) by 0.0116, resolved.
- **Eleven teachers were distilled into tables: teacher tower quality vs distilled table quality
  Spearman ≈ 0.** mxbai matches bge-base's tower (0.4433 vs 0.4484) with a table 0.057 worse.
- A doc-side linear map, document centering, whitening, top-PC removal and per-token scalar
  weights are all provably absorbable into the table — verified to machine precision. Only n-gram
  rows and multiplicity-dependent pooling add capacity.

**Hard constraints.** Student ≤35M params. Licence must permit commercial release of derived
weights. MS MARCO excluded from training (priced at +0.0058 on the six). No component from a
vendor whose core business is vector search. Evaluation protocol changes only BEFORE the affected
numbers exist. Budget ceiling $1,000, ~$90–245 committed. One RTX 3080 + rented cloud GPU.

**Organisational facts.** ~35 registered arms/contrasts, a pre-registered selection surface of
13,416 queries in 4 families, Bonferroni denominator 13, MDE 0.0056. Most contrasts were expected
to be statistically unresolved before we started. Four reviewers have passed over the plan.

## What we want from you

1. **What is missing entirely** — an option, a measurement, a lever, a failure mode, or a framing
   that appears nowhere above.
2. **Which premise nobody has questioned** is the one most likely to be wrong. Not the ones we
   already debate (we debate the frozen tower, the teacher, the dose); the one that reads as
   background assumption.
3. **If you had this goal, these constraints and this budget and could ignore what we have built**,
   what would you do — and what does that imply we should change now?
4. **The cheapest experiment with the highest information value** that is not in the plan.
5. Anything else you consider more important than the above.

Rank by expected value. Be blunt and terse — dense over long, no restating our facts back to us,
no praise. If something above is internally inconsistent or a number looks wrong, say so first.

codex
Comparison gap first: **small-model + BM25 is missing.** A hybrid beating a dense-only baseline cannot establish the architectural advantage. And nano using bge-small’s backbone does not automatically buy cheaper query inference than conventional bge-small.

Ranked by expected value:

1. **The least-questioned premise: query encoding belongs on the edge.** Your index already requires a server. If hybrid retrieval sends query text there, server-side query encoding is an obvious alternative: negligible edge model footprint, potentially smaller requests, shared batching and caching. The missing measurement is its latency and marginal serving cost. If the scientific requirement is *near-zero total query compute*, say that explicitly; “near-zero edge compute” permits a much cheaper solution to the engineering problem.

2. **You may be minimizing arithmetic while paying in memory.** A lookup table can lose on download size, resident memory, cold page faults, tokenization, or energy despite winning on transformer FLOPs. Parameter count does not settle this. **Cheapest high-information experiment:** on one intended edge device, benchmark the actual packaged table and small model across cold starts and warm queries. Record artifact bytes, peak memory, and p50/p95 latency, including tokenization and vector construction. This needs no relevance labels and touches no protected evaluation. It could invalidate the deployment premise in an afternoon.

3. **Compile likely queries into the document index instead of into a student.** Where your generated queries retain source-document IDs, index those query strings as lexical aliases for their documents. Search aliases with BM25, aggregate to document IDs, optionally fuse with existing retrieval. This shifts learning and computation to ingestion while preserving a nearly trivial query path. You already possess much of the expensive input. The overlooked tradeoff is **index expansion versus query-model complexity**. Register a small, separate diagnostic before running it; use disjoint natural queries and exclude evaluation-derived aliases. No additional transformer training is needed.

4. **The unit of analysis may be wrong: unique benchmark queries versus repeated requests.** A cache of popular queries’ teacher vectors—or retrieval results—could deliver teacher quality with almost no amortized query encoding. Its value depends on traffic repetition, personalization and index churn, none of which appears here. Measure
   \[
   E[C]=hC_{\text{hit}}+(1-h)C_{\text{miss}}
   \]
   alongside request-weighted quality. Sweep cache hit rate if representative logs are unavailable, but do not pretend BEIR supplies a deployment hit rate. This alternative changes the answer if the goal concerns average serving cost rather than mandatory computation limits on every unseen query.

5. **Your strongest table teacher may be one trained to be compressible.** Distilling independently trained towers assumes their useful rankings admit an additive token representation. Tower quality need not measure that property. An omitted research direction is to train a document representation jointly with a lookup-only query encoder, then freeze it for deployment. That tests an asymmetric architecture directly. It is a different experiment from selecting another pretrained teacher; your present setup cannot establish the ceiling of near-zero-compute retrieval generally.

If starting over, I would build a measured quality–latency–memory–serving-cost frontier around conventional small + BM25, lookup + BM25, server encoding, and lexical query aliases. **Gate further training spend on finding an operational region where asymmetry wins.** A small nDCG improvement without such a region answers a narrower question than your stated goal.
tokens used
6,314
Comparison gap first: **small-model + BM25 is missing.** A hybrid beating a dense-only baseline cannot establish the architectural advantage. And nano using bge-small’s backbone does not automatically buy cheaper query inference than conventional bge-small.

Ranked by expected value:

1. **The least-questioned premise: query encoding belongs on the edge.** Your index already requires a server. If hybrid retrieval sends query text there, server-side query encoding is an obvious alternative: negligible edge model footprint, potentially smaller requests, shared batching and caching. The missing measurement is its latency and marginal serving cost. If the scientific requirement is *near-zero total query compute*, say that explicitly; “near-zero edge compute” permits a much cheaper solution to the engineering problem.

2. **You may be minimizing arithmetic while paying in memory.** A lookup table can lose on download size, resident memory, cold page faults, tokenization, or energy despite winning on transformer FLOPs. Parameter count does not settle this. **Cheapest high-information experiment:** on one intended edge device, benchmark the actual packaged table and small model across cold starts and warm queries. Record artifact bytes, peak memory, and p50/p95 latency, including tokenization and vector construction. This needs no relevance labels and touches no protected evaluation. It could invalidate the deployment premise in an afternoon.

3. **Compile likely queries into the document index instead of into a student.** Where your generated queries retain source-document IDs, index those query strings as lexical aliases for their documents. Search aliases with BM25, aggregate to document IDs, optionally fuse with existing retrieval. This shifts learning and computation to ingestion while preserving a nearly trivial query path. You already possess much of the expensive input. The overlooked tradeoff is **index expansion versus query-model complexity**. Register a small, separate diagnostic before running it; use disjoint natural queries and exclude evaluation-derived aliases. No additional transformer training is needed.

4. **The unit of analysis may be wrong: unique benchmark queries versus repeated requests.** A cache of popular queries’ teacher vectors—or retrieval results—could deliver teacher quality with almost no amortized query encoding. Its value depends on traffic repetition, personalization and index churn, none of which appears here. Measure
   \[
   E[C]=hC_{\text{hit}}+(1-h)C_{\text{miss}}
   \]
   alongside request-weighted quality. Sweep cache hit rate if representative logs are unavailable, but do not pretend BEIR supplies a deployment hit rate. This alternative changes the answer if the goal concerns average serving cost rather than mandatory computation limits on every unseen query.

5. **Your strongest table teacher may be one trained to be compressible.** Distilling independently trained towers assumes their useful rankings admit an additive token representation. Tower quality need not measure that property. An omitted research direction is to train a document representation jointly with a lookup-only query encoder, then freeze it for deployment. That tests an asymmetric architecture directly. It is a different experiment from selecting another pretrained teacher; your present setup cannot establish the ceiling of near-zero-compute retrieval generally.

If starting over, I would build a measured quality–latency–memory–serving-cost frontier around conventional small + BM25, lookup + BM25, server encoding, and lexical query aliases. **Gate further training spend on finding an operational region where asymmetry wins.** A small nDCG improvement without such a region answers a narrower question than your stated goal.
