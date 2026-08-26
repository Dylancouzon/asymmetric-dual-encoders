# M7 status

**Stage:** Stage 0 done, gate = **GO**. Strategy re-planned after research. **Tier 1 is reachable.**
**Updated:** 2026-08-26

## Where we actually are

Gate passed on all four conditions (`results/m7_gate_p1-objB.json`): G1 +0.0994, G3 +0.0270 (both
p<1e-4), G4 int8 bound 0.00053 vs a 0.005 bar, G2 pass (near-vacuous, logged as such).

**But read the per-component row, not the macro.** The +0.0270 win over BM25 is one component wide
and it is the training-adjacent one: nq-250k +0.1445, cqa-physics +0.0152, cqa-programmers −0.0203,
**hotpotqa −0.0316 CI-resolved loss**. Diagnosis: a normalised bag of token vectors cannot express
word order or composition, so we win on single-hop and lose on multi-hop and duplicate-question.
Current projection to the six: **~0.41 = Tier 4.**

## Tier 1 is reachable — the arithmetic

Calibrating MTEB-Retrieval to our six-set via bge-small (the one model measured both ways,
ratio 0.976):

| teacher | licence | vocab/dim | MTEB-Ret | six est | x78% (today) | x85% | x88% |
|---|---|---|---|---|---|---|---|
| bge-base (current) | MIT | 30,522 / 768 | 53.25 | 0.520 | **0.406** | 0.442 | 0.457 |
| bge-large | MIT | 30,522 / 1024 | 54.29 | 0.530 | 0.413 | 0.450 | 0.466 |
| **stella_en_400M_v5** | **MIT** | 30,528 / 1024 | **58.97** | **0.575** | 0.449 | **0.489** | **0.506** |

Bars: BM25 0.4174 · Tier 2 release 0.4583 · **Tier 1 aim 0.4868**.
**stella x 85% clears Tier 1 with no fusion at all**; fusion (+0.02–0.04 in our favourable regime)
sits on top. So the job is: **raise the teacher, and move retention 78% -> 85%.**

## The four levers on retention, all now literature-backed

Detail and citations: `research/m7-research-2026-08-26.md`.

1. **Query-side centering / top-PC removal (SIF).** Cheapest untried lever, and the algebra says
   it is genuine new capacity: pure whitening is absorbable into the table
   (`normalize(W·mean) = normalize(mean(W·))`) but the centering offset is not. Evidence is for
   exactly our setup — WhiteningBERT +4.80 Spearman on *mean-pooled* BERT; RepBERT (mean-pooled)
   +6.9–22.8% nDCG in-domain, up to +25% OOD, and the six are OOD for us.
2. **Fix contrastive.** It was never broken — **our lr was 30–300x above every published recipe**
   (3e-3 vs 1e-5..3e-4), and arXiv 2110.09348 proves analytically that high lr shifts the embedding
   mean into dimensional collapse: loss falls while representation degenerates, our exact symptom.
   Temperature was NOT the problem (BGE uses 0.01). `fn_margin=0.02` is tighter than NV-Retriever's
   tuned 0.05 which they already call accuracy-penalising.
3. **N-gram / phrase rows.** The only structural fix for the diagnosed cause. No modern numbers
   exist, so magnitude is unknown until built — and nobody has reported a bag encoder beating BM25
   on multi-hop, so this is where an original claim lives.
4. **Stronger teacher**, with the caveat that capacity-gap literature expects retention to fall
   somewhat as the teacher strengthens — re-measure, never assume.

## Next session = research/re-plan, then rebuild

See the handoff prompt. Order: cheap diagnostics and the centering lever first (hours), then the
contrastive refit at a sane lr, then teacher swap, then n-grams, then fusion. Preserve the eval
protocol, partition ledger, decontamination, pinned dev suite, frozen comparators and
freeze/final-run machinery — those are twice-adversarially-reviewed and are not what is wrong.

## Open for Dylan

1. **stella_en_400M_v5 provenance call.** MIT, NovaSearch ships no vector product — but the weights
   are initialised from Alibaba's `gte-large-en-v1.5`, and Alibaba ships OpenSearch Vector Search /
   AnalyticDB. The releasing party is clean; the lineage touches a competing vendor. Your call.
   Conservative fallback: bge-large (+1.04 MTEB-Ret, Tier 2 only).
2. **Host:** stop Windows Update auto-rebooting (Event 1074 took the box at 05:52; nothing lost).
3. HF release go, later.

## Guardrails now in CLAUDE.md

A standing directive (added after this session declared Tier 1 unreachable and was wrong within the
hour): exhaust the angles before calling any bar unreachable — redo the arithmetic with the best
component, diagnose every failure mechanistically, sweep the literature, check capability claims
algebraically. Plus: past decisions are revisitable with evidence and Dylan's sign-off, the goal
supersedes them — except that eval-protocol changes must precede the numbers they affect.
