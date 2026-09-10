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
