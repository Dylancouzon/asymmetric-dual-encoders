# Whole-plan review: asymmetric dual encoders, at the M10 screen's midpoint

You are the strongest reviewer we have access to, and this is a **one-shot, expensive** call. Do
not spend it re-checking arithmetic we can check ourselves. Spend it on the questions below, where
being smarter actually changes the answer. **Disagree with the framing if the framing is wrong.**

## READ-EXCLUSION — mandatory, and the log is audited afterwards
Do NOT read, grep, open, list or glob, at any depth:
- `results/frozen_eval/untouched-*` · any reserved qrels cache · `work/m9reserve` ·
  `results/perquery.json`
Do NOT run a repo-wide or recursive grep/find. Read ONLY:
- `CLAUDE.md` (the whole file — north star, constraints, standing directives, decision log)
- `m10/STATUS.md`, `m10/LEDGER.md`
- `m9/FINDINGS.md` (why M9 failed), `m10/PLANNING.md` §9 and §11–12
- `instructions-m10.md` (mandate + the 2026-09-04/05 amendment blocks)
- `research/m10-paired-row-registration-draft.md`
- `results/m10_contrast_F1.json`, `results/m10_arm_F-bge-small.json`
If you need any other file, NAME it in your report and stop.

## The project in one paragraph
Server indexes documents once with a large FROZEN encoder (`stella_en_400M_v5`, 1024d). The edge
client holds that index plus a near-zero-compute query path. Two query paths, both serving the SAME
index: **zero** (M7, shipped — a token→vector lookup table, no transformer, avg-6 0.4339, fused
with BM25 0.4911) and **nano** (a ≤35M distilled transformer, M9 failed, M10 is the retry). The
deliverable is a quality-vs-query-cost frontier plus the released models. Hard constraints: student
≤35M params; licence must permit commercial release of derived weights; MS MARCO excluded from
training; no component from a vendor whose main business is vector search; the evaluation protocol
may change only BEFORE the numbers it affects are observed.

## Where we actually are
- **M9 failed on COVERAGE, not parameters:** 82.2% teacher retention overall but **93.8% on NQ
  against 50–71% on two CQADupStack components**. Diagnosis on record: a 384-wide linear head that
  L2 regression cannot push past ~90–93% once queries are diverse.
- **M10's answer:** 1152-wide output pooled from three layers, LEAF small-batch cyclic schedule,
  warm start from a closed-form ridge head, ≈1.25M harvested real query forms + ≈0.83M generated
  for the forms no corpus contains.
- **The screen is running.** Family F resolved today: **bge-small beats MiniLM-L6 by +0.011595**
  (lower bound +0.007221 at α/24 two-sided, 13,416 paired queries, sign-stable). 3 of the
  remaining 11 arms done. So the student is bge-small and every later family is conditional on it.
- **A measured surprise:** the registered resolution distance was 0.0086 (measured between
  UNRELATED models); F's actual paired distance came in at **0.004374**, the optimistic end of the
  pre-priced range. The screen has more power than registered — on one contrast.

## What I believe, so you can attack it
1. The coverage diagnosis is right and M10's recipe addresses it.
2. The screen is worth its ~52 box hours because it prevents building the wrong recipe.
3. The pair (zero + nano) as a frontier is the right deliverable and the right paper.
4. If nano misses its bars again, the publishable finding is that **teacher quality does not
   predict distilled-table quality** (Spearman ≈ 0 over eleven teachers; mxbai matches bge-base's
   tower at 0.4433 vs 0.4484 while its table is 0.057 worse).

## The questions. Answer the ones where you can say something we could not.

**Q1 — Is the coverage diagnosis causally right, or are we treating a symptom?**
M9's retention was 93.8% on NQ and 50–71% on CQADupStack. We concluded "the head is too narrow and
the query forms too few". An alternative we have NOT excluded: distilling into a frozen teacher's
query space caps at how much of that space is *linearly reachable* from a small encoder, and
widening the head buys geometry, not coverage. If that is the real limit, what measurement
distinguishes the two, and does M10's recipe move the right variable?

**Q2 — W14, the structural defect we could not resolve.**
No decision-bearing surface sees the headline forms. The screen selects on COV (forum posts,
medical, legal contracts, templated finance). The release bar is the pre-registered **clean-4**
(scientific, biomedical). So the screen can optimise the recipe AWAY from the release bar and
nothing registered would notice until the final evaluation. Codex framed the root question as:
*is family A a causal experiment, a catastrophe veto, or diagnostics? It cannot be all three.*
Constraint: COV is now OBSERVED, so no surface can be added to it, and any new surface must be
licensed, decontaminated, qrel-bearing and declared before it is read. **Is there an option we
have not seen — or is one of the three readings clearly correct?**

**Q3 — Is the screen buying information, or spending time?**
MDE is fixed at 0.0056. Most B–G contrasts were expected UNRESOLVED before we started. The
counterfactual is: cut the screen, spend the compute on a longer build. Given F's distance came in
at 0.0044 rather than 0.0086, does that change the calculus — and which of the remaining contrasts
would you cut *now* on expected information alone?

**Q4 — The frozen-tower premise itself.**
The architecture assumes a frozen off-the-shelf document tower, which buys the drop-in-to-stella's
-index property. M8 authorised and never ran document-side co-adaptation (`E14-LORA`), and M16
parks it because it costs that property. Is the premise still the right one given everything
measured since, or is the evidence now pointing at the document side?

**Q5 — What would a hostile reviewer of the paper attack first?**
The teacher (stella) discloses ArguAna, FiQA and FEVER in its training data, and every stella-based
claim carries that qualification. Headline is clean-4, all six reported beside it. The reserved four
have one unspent confirmatory access. Where is the hole?

**Q6 — Budget.** ~$90–245 committed of a $1,000 ceiling, the box does the screens and generation,
the cloud buys the build. What is the highest-value use of the remaining budget that we are not
currently planning to do?

**Q7 — The question we should have asked you and did not.**
Name it, and answer it.

Be concrete and be blunt. If a belief above is wrong, say which and why. If a question is
malformed, reformulate it before answering.
