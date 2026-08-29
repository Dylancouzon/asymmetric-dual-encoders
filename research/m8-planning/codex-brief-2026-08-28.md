# Brief: attack M7's decisions and propose M8 directions

You are an adversarial reviewer and idea generator for a retrieval research project. You have
read-only access to the repo at the current working directory. We just closed milestone M7 with a
MISS and are planning M8, the v2. Your job: (1) find the decisions, assumptions, and blind spots in
M7 that most plausibly cost quality, (2) propose concrete M8 directions with arithmetic, (3) break
the framing where it deserves breaking. We want you to push us to the best obtainable system, not
to confirm what we did.

## What the system is

A zero-query-compute retriever: documents are encoded by a frozen off-the-shelf teacher
(`NovaSearch/stella_en_400M_v5`, 1024-d); the query side is a 30,522 x 1024 lookup table (one row
per BERT WordPiece token) — tokenize, gather rows, pool with sqrt-count saturation, L2-normalize.
No transformer at query time. Optionally fused with BM25 (convex, w=0.8, selected on dev). The
released artifact is 31.3 MB int8.

## The M7 outcome (final confirmatory run, six BEIR datasets, one-shot access)

- int8 table avg-6 nDCG@10 **0.4339**; release bar was LightRetriever-dense-pertask **0.4583**:
  missed by −0.0243 [−0.0405, −0.0086], CI-resolved.
- vs BM25 0.4174: +0.0165, does not survive the registered Holm rule.
- fused system 0.4911 vs OpenSearch inference-free sparse 0.4868: statistical tie.
- Retention vs the teacher-symmetric 0.5744: **0.755** on the six.
- Clean-4 (datasets with no teacher-training overlap): the table is BELOW BM25 (−0.0311).
- The clean-stack tax was measured: adding decontaminated MS MARCO gains only +0.0058 — the miss
  is architectural, not data-licensing.
- Fusion vs dense +0.057 descriptive is the bright spot (trec-covid +0.153, scifact +0.097).

## Key constraint arithmetic you should exploit

The comparator LightRetriever table is 151K vocab x 1536 fp16 = 466 MB (233 MB int8). Ours is
31.3 MB int8 — **the size budget is ~15x unspent**. Established algebra
(`results/m7_absorb_check.json`): query-side centering/whitening/per-token scalar weights are all
absorbable into the table (not capacity); only **n-gram rows** and **multiplicity-dependent
pooling** add capacity to a bag-of-tokens encoder.

## Read these repo files before answering

- `m7/STATUS.md` (final numbers), `m7/RECIPE.md` (full shipped recipe), `m7/FINDINGS.md`
  (transferable lessons), `m7/EXPLORED.md` (closed avenues — challenge closes you think premature),
  `instructions-m8.md` (M8 mandate and pre-registrations), `CLAUDE.md` (project history, standing
  directives, vendor/licence rules).
- Optional depth: `m7/LEDGER.md` (protocol), `results/FINAL_MATRIX.md`.

## Binding constraints (not revisitable by you, only flaggable to Dylan)

- Licence: released weights must be Apache-2.0-compatible; MS MARCO stays excluded from release.
- Vendor rule: no components from direct vector-search competitors (see CLAUDE.md relaxed rule).
- Evaluation integrity: M8's confirmatory sets are the reserved four (FEVER, DBpedia-entity,
  CQADupStack-android/english), one-shot, pre-registered bars. Dev-only selection.
- Hardware: RTX 3080 10 GB VRAM, 25 GB RAM box.
- Replicable and defendable on benchmarks — no protocol relaxation after numbers are seen.

## What we want back, in priority order

1. **Attack the closed avenues**: which rows of `m7/EXPLORED.md` are premature closes under the
   standing directive in CLAUDE.md? Which diagnoses are confounded?
2. **Attack the architecture framing**: is the 30K-WordPiece-vocab, single-vector-per-token,
   linear-bag design leaving obvious capacity on the table given a 233-466 MB budget? What would
   YOU build? Concrete options with size/compute arithmetic (e.g. bigger tokenizer vocab, n-gram
   rows trained through the forward, multiple vectors per token, phrase tables, subword-to-word
   composition, learned sparse arm replacing BM25 in the fusion, doc-side changes).
3. **Attack the teacher/doc-side premise**: doc tower is frozen off-the-shelf. Would fine-tuning
   or adapting the doc tower to be *table-approximable* (co-training doc encoder with the table)
   plausibly dominate? What does that cost on this hardware, and does it break any constraint?
4. **Attack the training recipe**: objective (KL + cos + InfoNCE), phase structure (B 16k steps
   then A 2.5k), the never-run phase3_hparams sweep (temp=0.02, n_neg=32768 fixed untested),
   pseudo-query pool design, the bare-target open item.
5. **Attack the evaluation/selection design for M8**: the dev suite over-rewarded in-distribution
   gains (retention 0.915 dev vs 0.755 confirmatory). How should M8's dev suite and bars be built
   so selection transfers?
6. **Anything else** — genuinely outside the box, as long as it stays zero-transformer-at-query-time
   (or argues precisely why a near-zero-compute exception, e.g. a single tiny matmul, should be
   put to Dylan as a scope question rather than assumed).

Format: numbered findings/proposals, each with severity or expected-value, the evidence or
arithmetic, and what measurement would confirm/kill it. Be specific and quantitative. Do not
flatter. A review told "confirm this" returns nothing — break things.
