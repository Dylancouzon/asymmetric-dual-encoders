# Transferable findings

What someone **not** building this exact system should take away. Deliberately separate from
`LEDGER.md` (protocol), `RESULTS.md` (runs), `EXPLORED.md` (dead ends) and `CODEMAP.md` (code):
this is the article/whitepaper skeleton, and the test for an entry is *"would this still be true,
and useful, for a different team distilling a different query encoder?"* Each entry names its
evidence; none restates a number an artifact already holds without pointing at it.

**The headline quality answer is not here yet** — the single confirmatory six-set access has not
happened. Everything below is methodology and component-level, and none of it depends on how that
number lands.

## On choosing a teacher

1. **Select a teacher on the distilled artifact, never on the teacher's own retrieval quality.**
   Spearman(teacher's symmetric ceiling, its distilled table's score) = **0.000** over ten
   candidates; the best-ceiling teacher ranks fifth on the thing that ships. A teacher approved on
   the ceiling produced a table 0.0480 *below* the incumbent's and was withdrawn the same day.
   `m7_learnability_report.json`, `m7_teacher_probe.json`.
2. **Within a family, the base model out-approximates the large one — every time, by +0.04 to
   +0.07.** arctic-m > arctic-l, gte-base > gte-large, bge-base > bge-large, e5-base > e5-large.
   Useful as a *screening rule*: a family whose large variant scores below ~0.28 on this probe
   cannot reach the incumbent by shrinking, and is not worth a probe. Closes a shortlist by
   arithmetic instead of by exhaustion.
3. **Cosine agreement with the teacher's query vector is not the metric.** It rises with the ridge
   λ while nDCG falls, and it mis-ranks: the highest-cosine candidate is sixth of ten on retrieval.
   Imitating a query vector is not reproducing a ranking.
4. **The closed-form probe is cheap and hardware-robust**: CUDA vs Apple MPS agree to **7e-4**
   across the whole λ grid with the same argmax, two orders of magnitude below the effects it
   resolves. So this criterion transfers to whatever hardware a team has.

## On what a lookup-table query encoder can and cannot gain

5. **Do the algebra before buying the GPU time.** Query-side centering, whitening, top-PC removal
   and any per-token scalar weight are all **absorbable** into a freely-parameterised table — they
   cannot raise the ceiling, only act as a prior. Proved to machine precision.
   Only **n-gram rows** and **multiplicity-dependent pooling** add anything. `m7_absorb_check.json`.
6. **A doc-side linear map is absorbable only if the mapped document is not renormalized** —
   `q·(Md) = (Mᵀq)·d` exactly. Retrieval renormalizes, so it is *not* absorbable in practice: rank
   agreement with the absorbed form is 1.000 without renormalization and 0.000 with it. The
   half-right version of this claim sat in our own notes for a day.
7. **Length sensitivity was an artifact of how it was measured.** A probe that draws a fresh
   document sample per length bucket confounds length with document population; it showed
   agreement falling 0.3443 → 0.2997 from 8 to 256 words. Re-measured as nested prefixes of *the
   same* documents, the curve is **flat from 16 to 256 words** and slightly rising. A bag-of-token
   query encoder did not degrade with query length the way everyone expected.

## On evaluating a system you are also selecting on

8. **The load-bearing methodological finding: a nuisance parameter moved our dev macro more than
   any effect we adopted.** Changing only the A-phase step count — a number nobody reports — moved
   the macro by **0.0027–0.0078** across three arms. Every adopted or adjudicated effect
   (+0.0040, +0.0065, +0.0038, +0.0023, −0.0048) sits inside that band.
   `m7_compare_full_stepspread.json`.
9. **Every confidence interval in a project like this is a query-sampling interval.** With
   deterministic training there is no replication term at all — ours is ~5e-6 — so the CIs answer
   "would another sample of queries agree", not "would another equally defensible recipe agree".
   Only the second question is the one a reader has. **Measure the recipe-perturbation band and
   report effects against it.**
10. **This does NOT deflate a confirmatory comparison** where the recipe is fixed first and scored
    against frozen comparator vectors on datasets never used for selection. Dev reuse contaminates
    the *selection*, not the *measurement*: the cost of over-fitting is a worse true recipe, not a
    biased final number. Keep the two claims apart.
11. **A dev suite assembled from training-adjacent sources systematically over-rewards
    in-distribution gains.** 90% of one adoption's gain landed on the three components that are in
    the training mix; the out-of-domain pair moved +0.0002 and +0.0036. Across the whole
    late-stage lever programme the out-of-domain subset spans **0.0040** while the macro spans
    0.0128. **Report an out-of-domain subset next to every macro**, and retention against both:
    ours is 0.915 all-six and **0.764** out-of-domain.
12. **A cheap in-training proxy can be anti-correlated with the instrument you decide on.** Our
    three-component proxy ranked three arms *exactly backwards* from the full suite, and a proxy
    peak did not reproduce when re-run (0.5130 → 0.5126). Use a proxy to pick a step, never a
    winner — and check that the peak replicates.
13. **Mined hard negatives sharpened memorization, not retrieval.** The apparent +0.0072 was
    `heldout-train` +0.0297 and `hotpotqa` +0.0187 — a seen-document/unseen-query slice and a
    component whose train split is a training source — while the out-of-domain components moved
    −0.0009 and +0.0013. A gain concentrated on the components closest to training is a signature,
    not a result.
14. **Count the dev reuse and publish the number.** Ours: **58 trained arms, 322 in-training
    evaluations, 90 eval-only variants**, with multiplicity control applied inside named families
    only. "We were careful" is not checkable; a count is. `m7_dev_reuse_count.json`.

## On running this kind of project

15. **Make the decision rules executable.** A pre-registered rule a session can re-read in its own
    favour is not a pre-registration. Ours run as code (`negatives_decide.py`,
    `simplify_decide.py`) and write their own verdict artifacts.
16. **Audit rule compliance; do not discover it.** We found a pre-registered step rule unapplied
    *by accident*, after it had governed four arms and a promoted adoption. `rule_audit.py` now
    checks every mechanically-checkable rule against every family it binds, and lists what it
    cannot check as unverifiable rather than as passing.
17. **Never read a rounded confidence bound.** A true lower endpoint of +4e-5 displays as 0.0000.
    We wrote the rule down, then broke it twice — once in the ledger, and once in `final_run.py`,
    the single irreversible decision in the project. Both found by review, not by us.
18. **Name evidence files after the artifact they describe.** A fixed `..._full.json` re-pointed
    when a lever was re-adjudicated, leaving the shipping artifact's metadata citing a *different*
    artifact's failure as the justification for its own rule.
19. **Do the launch arithmetic, and take the rate in the slow region.** One job would have needed
    10.4 GB of token ids where a flat int32 layout needs 1.2 GB. Another was estimated at 36
    minutes from its first shards and was really ~4 hours, because the data was ordered by source
    and the expensive source came later.
20. **A second machine is a cheap replication instrument.** Requiring the second box to re-measure
    the *incumbent*, not just the new candidates, turned a convenience into a cross-platform
    reproducibility check — and it cost one extra row.
