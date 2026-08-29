# Adversarial review: M8's re-route onto D2

You are reviewing a decision that is about to become expensive. Be adversarial. I will state what
I believe; your job is to break it, and then to tell me what I have not thought of. A review that
confirms my plan is worthless to me.

Repo: `/home/dylan/asymetric-dual-encoders`, branch `m8-planning`, HEAD is pushed. Read at least:
`CLAUDE.md`, `m8/STATUS.md`, `m8/NEXT-SESSION.md`, `m8/LEDGER.md` (§5, §7, §8, §13, §17b, §23, and
the top entry of §15 dated 2026-08-29 "MILESTONE AUDIT AND RE-ROUTE"), `m8/registry.json` (the `D2`
and `NF-CROSSED-FUSED` rows), `m8/EXPLORED.md`, `m7/STATUS.md`.

## Context in one paragraph

M7 built a zero-query-compute lookup-table query encoder (tokenize → look up rows → pool →
normalize → ANN). Its confirmatory run MISSED its release bar CI-resolved (int8 table 0.4339 vs
LR-dense-pertask 0.4583, −0.0243 [−0.0405, −0.0086]); on the four datasets with no disclosed teacher
overlap the table is BELOW BM25 (−0.0311). M7 measured the gap as architectural, not licensing (the
clean-stack tax arm: +0.0058 unresolved, miss survives). M8 is the "learnings v2" of the same
artifact class. Its confirmatory sets are four RESERVED, never-scored datasets (FEVER,
DBpedia-entity, CQADupStack-android, CQADupStack-english) with ONE one-shot access.

## What has been measured in M8 so far — all nine probes

- `B3` data volume: 4× dose moves dense +0.00135 / fused +0.00369 vs a 0.0040 bar. Fitted slope
  +0.00097 per doubling → reaching the bar needs ~17.6× the pool (~5.9M pairs). M7's entire MS
  MARCO addition was 490K.
- `E14-HEAD` doc-side head (= menu item D1): LIN (1024×1024, 1.05M) −0.0244 dense, MLP (4.2M)
  −0.0293, vs a +0.0040 bar. All six arms agree in sign. The patch stack measured as a clean null
  (R0N vs R0: −0.00001). LIN's registered label is OPTIMIZATION-INADEQUATE at 2,500 steps.
- `T1` teacher swap: granite-r2 −0.052 [−0.066,−0.039], gte-modernbert −0.109. NO SWAP.
- `B2`: the KL term is degenerate (teacher target median entropy 4.73e-07 nats).
- `B7`: block-CG solves 131,072 rows in 17 s / 5.7 GB. `B6-pre`: doc-side ONNX fuse PASSES.
- Noise floors: σ_A 0.00106; crossed B×A gives σ_chain 0.00153, and the registered formula's answer
  for a chain-varying arm is 0.00519 (recorded NOT ADOPTED at the time).
- Power/ship rule: MDE at 80% power 0.0068. P(ship) by scenario: structural_target 0.84, modest
  0.80, recipe_only 0.21, m7_repeat 0.002, dense_lags_fused 0.57.

## What I did, and what I believe

I audited the milestone and found that `D2` — a self-trained multi-word tokenizer, the only
remaining lever with a measured mechanism pointing up — had **no registry row, no schedule, and no
place on the worklist**, while the worklist's "next capacity lever" was bigram rows, which §13 had
retired as superseded by D2 with "no auto-revival". I registered D2, adopted the chain floor
(so D2's bar is 0.00519, not 0.0040), deferred the recipe/data class, and wrote a pre-committed
exit: if D2 and its alternate `B10`/`pool_mode` both miss, M8 does not spend its access.

**The mechanism I am betting on (§17b):** the table falls 0.050 nDCG further behind the teacher per
+1.0 subwords-per-word, t=4.61, and this survives every single-dataset exclusion at t ≥ 3.28.
Mechanism direction: the *teacher* gets BETTER with fragmentation (+0.038, t=2.72) while the table
is flat (−0.012, t=−0.85).

**Prior evidence against, which I have written into the registration:** the only published
vocabulary-size ablation on a static/bag-of-words retriever is VDR (arXiv 2212.07699, ICLR 2024):
30K → 110K rows moved BEIR nDCG@10 **44.5 → 42.6**, a regression — confounded (the vocabularies also
swapped English BERT for multilingual BERT). M7 shipped with 5.71% of rows never trained at a 30,522
vocabulary; 128K makes coverage the first question.

## Attack these specifically

1. **Is D2 actually the right bet, or am I pattern-matching a correlation?** §17b's slope is a
   between-query association of the table-vs-teacher GAP whose mechanism is the teacher improving.
   I wrote into the registration that it "sizes an upper bound, not an expected gain". Is even that
   too generous? Construct the strongest case that a multi-word vocabulary CANNOT convert this
   slope into table quality — and separately, the strongest case that it can, with a magnitude
   estimate. What would the fertility reduction have to be for +0.005 dense, and is that reachable?
2. **Is the D2 registration exploitable?** Nested vocabulary selection on wikipedia+heldout while
   the bar reads out-of-domain — does that actually protect the bar, given the groups are not
   independent and both come from one dev suite that has been read many times (G8 tracks a dev-reuse
   count)? Is the >20% never-updated-dev-reachable-row coverage gate the right gate, right
   threshold, and can it be gamed by the pool? Is the compositional init floor (mean of constituent
   unigram rows) actually a floor, or could it be WORSE than a random init for a multi-word row
   whose meaning is non-compositional ("hot dog", "New York")?
3. **Is bar 0.00519 correct for D2?** D2 retrains the B leg AND changes the tokenizer, which changes
   the bag composition everywhere. Is it in fact POOL-varying too (§23 explicitly does not bound a
   pool-varying lever), which would make even 0.00519 anti-conservative? The pseudo-query pool text
   is fixed but its tokenization is not.
4. **Dense-only bar.** I dropped the fused endpoint to descriptive because no chain-level fused
   floor exists and E14 measured fusion damping dense changes ~10×, so a 3-seed fused sign condition
   would veto real wins. Is that reasoning right, or did I just remove a guard because it was
   inconvenient? What is the actual type-I cost of losing intersection-union here?
5. **The pre-committed exit.** CLAUDE.md carries a standing directive that "before writing that any
   bar is unreachable" the arithmetic must be redone with the best component, every failing
   component diagnosed, the literature swept, and capability claims checked algebraically. Does my
   exit violate that directive, or satisfy it? Is not spending the access the right call, or is it
   premature abandonment dressed up as discipline — given that a MISS is pre-registered as a
   publishable outcome and the reserved four are otherwise just sitting there?
6. **The deferral.** I deferred 10 recipe/data probes plus D-FINEWEB behind D2 on the strength of
   one M7 sentence (post-gate lever programme transferred at 0.000 ± 0.005). Is that sentence strong
   enough to carry 11 deferrals? Is there a recipe probe among them that is cheap enough and
   independent enough that deferring it is a false economy?

## The question I most want answered: what have we NOT tried?

This project has closed a lot of doors: post-hoc linear projection (dead), doc-side heads (harm),
teacher swaps (dead), data volume (dead), KL distillation (degenerate), ICT augmentation (retired),
higher table dims (algebra), absorbable transforms — centering, whitening, top-PC removal, per-token
scalar weights — (provably no-ops, `results/m7_absorb_check.json`), full late interaction (out of
scope), doc2query full dose (compute), MS MARCO (licence). The project's own algebra says **only
n-gram rows and multiplicity-dependent pooling add capacity** to a bag-of-token-vectors query
encoder against a frozen document tower.

**Is that algebra complete?** Name capacity directions this project has not enumerated. I am
explicitly interested in things that sit outside the current frame, including ones that would need
Dylan to reopen a structural premise (the frozen off-the-shelf document tower, "no transformer at
query time", single-vector documents, the dense+BM25 fusion operator). For each: the mechanism, a
rough magnitude, the cost, and what would kill it fast. Cite real systems or papers where they
exist. Do not pad the list — three well-argued directions beat ten speculative ones.

Also: is there a cheaper, faster experiment that would tell us whether D2 can work BEFORE we spend
5 full training chains on it? A closed-form or diagnostic probe on existing artifacts that would
predict D2's sign?

## Constraints on your answer

- Read the files; do not rely on this brief alone. Quote file:line for anything you claim the repo
  says.
- Distinguish BLOCKER (this is wrong and will produce a bad decision) / MAJOR / MINOR, and say for
  each whether you verified it in the repo or inferred it.
- If you think the whole re-route is wrong, say so plainly and say what you would do instead.
- Do not write to any file.
