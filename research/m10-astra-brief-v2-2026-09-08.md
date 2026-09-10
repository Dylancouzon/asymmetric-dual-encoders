# Improve this plan. Read it first — do not answer from this brief.

Your previous pass reviewed a one-page summary because the brief said everything needed was
inline. That was a mistake on our side: you never saw the plan. **This time, read the files.** Do
not answer from this page alone; it only tells you where to look and what we already know.

**No terseness limit.** Depth is what we are paying for. Long is fine if it is load-bearing.

## READ-EXCLUSION (mandatory, audited afterwards)
Never read, at any depth: `results/frozen_eval/untouched-*`, any reserved qrels cache,
`work/m9reserve`, `results/perquery.json`. No repo-wide or recursive grep.

## READ THESE
- `instructions-m10.md` — the mandate, plus the 2026-09-04 and 2026-09-05 amendment blocks
- `m10/screen_registry.json` — the ~35 arms and contrasts, MDE, quantiles, family rules,
  `outcome_to_action`, `serve_cost_order`
- `m10/PLANNING.md` — §9 (why we diagnosed the head), §11 (rates), §12 (the synthetic-data question)
- `m9/FINDINGS.md` — why attempt 1 failed
- `CLAUDE.md` — north star, hard constraints, standing directives, decision log
- `m10/STATUS.md` — current state
Name any other file you want rather than grepping for it.

## What we already decided — do NOT re-propose these
Re-deriving a settled item wastes the call. All of these are ruled and on the record:
- Teacher = `stella_en_400M_v5`, chosen on the distilled TABLE not the tower. Eleven teachers swept.
- Doc-side linear maps, document centering, whitening, top-PC removal, per-token scalar weights:
  **provably absorbable** into the table, verified to machine precision. Cosine-space distillation
  on a normalized output is likewise a no-op. Only n-gram rows and multiplicity-dependent pooling
  add capacity.
- MS MARCO excluded from training (priced +0.0058). Non-commercial licences are validation-only.
- Student ≤35M hard. Family F selected bge-small over MiniLM-L6, resolved.
- C-M9init cut; L12 cut; family A's STOP removed; MDE fixed at 0.0056 and the COV surface is now
  OBSERVED, so it cannot take a new admission.
- CUREv1 is a reported diagnostic, never selection-bearing.

## What your last pass already gave us (verified, being actioned)
1. No small-model+BM25 comparator exists — our fused headline dodges the release bar's own hybrid.
2. nano (34.5M) is LARGER than bge-small (~33.4M) and runs the same forward pass, so it buys no
   compute saving; only the lookup table does.
3. The edge cost axis (cold start, peak RSS, p95) is unmeasured for the table and for bge-small.
4. Lexical query aliases (doc2query over our generated queries) as a no-student alternative.
5. Spearman≈0 over teachers may mean compressibility is trainable, not that teachers are alike.
Do not repeat these. Build past them or contradict them.

## What we want

**A. Attack the screen you have now read.** ~35 arms and contrasts, Bonferroni denominator 13,
MDE 0.0056, a 13,416-query family-weighted selection surface, an `outcome_to_action` map, and
~52 box hours. Most contrasts were expected unresolved before we started. Name specifically:
which arms or contrasts would you **cut**, which are **mis-specified** (measuring something other
than what their rule claims), and which **decision the map gets wrong**. Quote the registry.

**B. Name the plan's biggest structural error.** One thing. The one that, if we are wrong about it,
makes the rest of the effort answer a question nobody asked.

**C. Concrete additions.** For each: the exact arm, contrast, measurement or decision to add, what
it would cost, and what it would change. Rank by (information gained) / (cost).

**D. Where the pre-registration machinery is doing harm.** We have a lot of it. Say where it is
buying rigour and where it is buying paperwork — and what to drop.

**E. The M14 paper.** Given everything you have now read, what is the strongest defensible claim
this project can make, and what should it stop claiming?

**F. Anything above these in value.**

If a number or an argument in the files is wrong, lead with that. Quote file and line.
