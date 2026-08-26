# M7 status

**Stage:** Stage 0 done, gate GO. Research session 2026-08-26 done: teacher chosen by measurement,
retention plan re-ranked, four prior claims corrected. **No training has restarted yet.**
**Updated:** 2026-08-26

## The one thing that changed the plan

Last status said "stella × 85% clears Tier 1 with no fusion at all". **Withdrawn.** It projected
from a *single* calibration model (bge-small, ratio 0.976) — the 3rd-highest of the nine we have
measured both ways, so every teacher estimate was biased high. Refit on all nine
(`results/m7_calibration.json`): ratio spread 0.926–1.001, affine r=0.950, **residual sd 0.0102**.

Also separated, because the mandate does and the last status did not: **Tier 2 (release bar 0.4583)
must be cleared by the dense int8 table alone; only Tier 1 (0.4868) may use fusion.** Retention
needed, against today's **78.5%**:

| teacher | six est | Tier 2, dense only | Tier 1, with fusion |
|---|---|---|---|
| bge-base (current) | 0.5082 | 90.2% (+11.7) | misses at any plausible retention |
| gte-large-en-v1.5 | 0.5473 | 83.7% (+5.2) | clears at ~88% |
| **stella_en_400M_v5** | **0.5562** | **82.4% (+3.9)** | **clears at ~85%** |

**So a teacher swap is required for the release bar itself, not just for the stretch aim.** That is
the headline change. Tier 1 is still reachable — but as teacher × retention × fusion, not by any
one of them.

## Teacher: shortlist re-run, decided by measurement not projection

`research/m7-teacher-shortlist-2026-08-26.md`. Two front-runners pass licence + vendor + vocab ≤50K
+ dim ≤1024 + a 10 GB 3080: **stella_en_400M_v5** (58.97, MIT, vendor CLEAN — NovaSearch is a
3-person org with no product) and **gte-large-en-v1.5** (57.91, Apache, Alibaba = admissible with
justification). Their 1.06-MTEB gap is *inside* the calibration residual, so
`m7src/teacher_probe.py` ranks them by measured ceiling on the two CQADupStack dev components — the
only dev components on no candidate's disclosed training list. Closed: the **Snowflake tier is
moot** (best model 2 points back, so Dylan never rules on it), **Qwen3-0.6B is dominated** (not a
vocab casualty), and the 2025–26 discovery sweep found **nothing new**.

**Live risk on stella:** MTEB's registry records ArguAna and FiQA2018 as its in-domain training
data — 2 of our 6 final eval datasets, and our comparators carry no such flag. Community metadata,
not an author disclosure. If stella wins the probe this must be labelled at the dataset row.

## Retention: what is real, after the algebra

`results/m7_absorb_check.json` settles it to machine precision, and it re-ranks the plan:

- **Centering / whitening / top-PC removal / SIF weighting add NO capacity** — all absorbable into
  the table (`mean(W−mu) = mean(W)−mu`; a per-token scalar by scaling that row). Last status had
  this as the top lever *because* it was "genuine new capacity". It is not, and p1-objB's learned
  weights **already are** IDF-like (spearman −0.44 vs row update count, [CLS]/[SEP] down to 0.61×
  median). Demoted to a cheap initialisation experiment.
- **N-gram / phrase rows are the only structurally new lever** on the old list — and where an
  original claim lives (nobody has published a bag encoder beating BM25 on multi-hop).
- **New lever nobody listed:** multiplicity-dependent pooling (count saturation). Not absorbable,
  costs nothing, untried.
- **A no-op, so nobody wastes a run:** length scaling (1/√|T|), removed by the final L2 normalize.
- **Contrastive is diagnosed, not dead.** Two of three suspects died *by measurement*
  (`results/m7_diag_scores.json`): fn_margin=0.02 removes only 4.3% of the top-100 hardest
  negatives; random negatives are not separable. Only ~29 of 32,768 negatives carry gradient at
  τ=0.02, so negative *quality* dominates pool size. The lr is the survivor — 3e-3 against a
  published 1e-5–3e-4 — and the one with an analytic mechanism (arXiv 2110.09348). train.py now has
  warmup and collapse diagnostics; the phase-2 screen was rebuilt around the lr, and its kill
  criterion can no longer fire on a misconfigured arm.
- Also refuted: "learned weights buy nothing" was a proxy-3 artifact — on the full suite the
  trained table beats the closed-form flat optimum by **+0.021**.

## Fusion is measured, and it inverts the two bars

`results/m7_fusion_report_p1-objB.json` — selected on dev, convex w=0.5, depth 1000:
**dense 0.4795 · BM25 0.4525 · fused 0.5520 (+0.0725, CI [0.0665,0.0784])**. The gain is **broad,
not one component wide**: hotpotqa +0.142, cqadup-programmers +0.073, cqadup-physics +0.044,
nq-250k +0.031, every one CI-resolved. It is *largest* on hotpotqa, the component where the dense
table loses to BM25 — fusion converts a −0.031 loss into a +0.110 win, repairing exactly the
diagnosed multi-hop failure. Both CQADupStack rows gain too, and those are the nearest dev analogue
to FiQA.

**If that +0.0725 transfers to the six, the bars are mis-ordered.** Tier 1 (0.4868, fusion allowed)
then needs dense ≥ **0.4143**; Tier 2 (0.4583, dense table alone) needs dense ≥ **0.4583**. So the
stretch aim is *easier* than the release bar:

| | bar | dense needed | retention needed on bge-base | on gte-large |
|---|---|---|---|---|
| **Tier 1** (aim, fused) | 0.4868 | 0.4143 | **81.5%** (+3.0 over today) | 75.7% — already there |
| **Tier 2** (release, dense only) | 0.4583 | 0.4583 | 90.2% (+11.7) | 83.7% (+5.2) |

Tier 1 is close to in hand on the *current* teacher. **Tier 2 is the binding constraint, and it is
what forces the teacher swap.** Caveat to carry: the gain is measured on dev, and the six contain no
NQ-like component, so transfer is an assumption until the final run.

## Order of work

1. **Teacher probe** — minutes, no GPU-hours. Picks stella vs gte-large on a measured number
   rather than a projection whose residual is bigger than the gap. Teacher is the one decision that
   is expensive to redo (full corpus re-encode), so it goes first.
2. **Document-side instruction** — the cheapest structural lever we have and, per round-2 research,
   **unpublished for the bge/e5/gte/stella families**: the "no doc prompt" convention is an
   index-prebuilding convenience, not a measured result. It changes doc vectors *non-linearly*, so
   unlike a doc-side linear map it is NOT absorbable into the table — genuine new capacity, testable
   by re-encoding with one fixed string.
3. **Count saturation** — near-zero cost, provably non-absorbable, and now with cross-family
   support: BM25, SPLADE and NUMEN (arXiv 2601.15205) independently converge on sublinear count
   damping, NUMEN in a train-free *dense* bag.
4. **Contrastive refit at a published lr**, warmup, teacher-mined hard negatives from the frozen
   index (permanently valid — our tower never moves).
5. **Teacher swap + corpus re-encode** (~4× compute, 1024d storage). After 3–4, because capacity-gap
   literature expects retention to *fall* as the teacher strengthens; measure it on the cheap
   teacher first. The re-encode is the costliest step and should happen once.
6. **N-grams — demoted.** Still the only structurally new lever by algebra, but Sent2Vec's own
   ablation (the founding reference) says bigrams help supervised classification and *not*
   unsupervised similarity, which is the regime retrieval lives in, and it reports no retrieval
   numbers at all. Arbitrary bigrams over a 30K vocab is ~10^9 candidate rows. The original-claim
   argument survives; the expected-value argument does not.

**Closed by round-2 research, so nobody spends a re-encode on them:** doc2query-style expansion
appended before dense encoding (controlled evidence says it *harms* strong encoders, and helps less
the stronger they get); multi-vector documents (ME-BERT's own table has single-vector winning at
short passage lengths, and our six are titles+abstracts); document chunking (nothing to chunk);
order-binding via HRR/VSA (zero retrieval evidence anywhere).

## The honest strategic note

**Two transformer layers are worth 92.5% retention** on a frozen doc tower (arXiv 2306.11550, BEIR
nDCG@10; four layers 96.2%) against our table's 78.5%. Priced through our calibration that is 0.470
on the six with the current teacher and 0.506 with gte-large — clearing Tier 1 *dense-only, before
fusion*, at ~0.1–0.5 ms instead of 0.023 ms. **That is exactly M8's mandate.** So M7's contribution
is the zero-transformer point on the cost frontier, not the quality winner, and we should stop short
of over-investing to close a gap the next milestone closes by design.

Related: our multi-hop ceiling is now backed by a theorem, not just our own measurement — DeepMind's
arXiv 2508.21038 proves realizable top-k document subsets are bounded by embedding dimension. The
framing shifts from "our bag is weak" to "single-vector retrieval is provably limited here, and a
bag is the cheapest way to reach that limit."

## Open for Dylan

1. **Nothing is blocking.** The stella-provenance call from last status is resolved without you:
   NovaSearch ships no vector product, and Alibaba (stella's lineage, and gte-large itself) is
   admissible under the relaxed rule.
2. **Host:** stop Windows Update auto-rebooting (Event 1074 took the box at 05:52 on 2026-08-26;
   nothing lost).
3. HF release go, later. Plus: if stella wins the probe, you may want a view on shipping a teacher
   whose training data is undisclosed and whose registry lists 2 of our 6 eval sets.
