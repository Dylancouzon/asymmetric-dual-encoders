# M7 status

**Stage:** Stage 0 done, gate GO. Two research rounds done 2026-08-26. **No training restarted.**
Detail: `research/m7-research-2026-08-26b.md`, `research/m7-teacher-shortlist-2026-08-26.md`.
Closed avenues: `EXPLORED.md`. Numbers: `results/m7_*.json`.

## Fusion is measured, and it inverts the two bars

`results/m7_fusion_report_p1-objB.json` (dev, convex w=0.5, depth 1000):
**dense 0.4795 · BM25 0.4525 · fused 0.5520 (+0.0725, CI [0.0665,0.0784])**. Broad, not one
component wide — all four components gain, CI-resolved, and the gain is *largest* on hotpotqa
(+0.142), turning a −0.031 loss to BM25 into a +0.110 win. That repairs the diagnosed multi-hop
failure, and both CQADupStack rows gain too (nearest dev analogue to FiQA).

If that transfers to the six, **Tier 1 is easier than Tier 2**, because only Tier 1 may fuse:

| | bar | dense needed | retention on bge-base | on gte-large |
|---|---|---|---|---|
| **Tier 1** (aim, fused) | 0.4868 | 0.4143 | **81.5%** (+3.0 over today's 78.5%) | already there |
| **Tier 2** (release, dense alone) | 0.4583 | 0.4583 | 90.2% (+11.7) | 83.7% (+5.2) |

**Tier 2 is the binding constraint and it is what forces the teacher swap.** Transfer of the fusion
gain is an assumption until the final run — the six have no NQ-like component.

## Teacher

Shortlist re-run under the relaxed vendor rule. Two front-runners, both clearing licence + vendor +
vocab ≤50K + dim ≤1024 + a 10 GB 3080: **stella_en_400M_v5** (MTEB v1 Ret 58.97, MIT, vendor CLEAN)
and **gte-large-en-v1.5** (57.91, Apache, Alibaba — admissible with justification). Their gap is
*inside* the calibration residual (sd 0.0102), so `teacher_probe.py` decides it by measurement on
the two CQADupStack dev components — the only dev components on no candidate's training list.

Closed: the Snowflake tier is **moot** (best model 2 points back — Dylan never rules on it),
Qwen3-0.6B is **dominated**, granite-r2 **ties** the current teacher, and the 2025–26 sweep found
**nothing new**. Live risk: stella's registry lists **ArguAna and FiQA2018** — 2 of our 6 eval sets.

## What is real on retention, after the algebra and round 2

- **Centering / whitening / top-PC removal / SIF weighting add NO capacity** — all absorbable
  (`results/m7_absorb_check.json`, machine precision). Demoted from top lever to an init experiment.
  p1-objB's learned weights already *are* IDF-like (spearman −0.44 vs update count).
- **Count saturation** — promoted. Non-absorbable, ~free, and BM25/SPLADE/NUMEN independently
  converge on sublinear count damping.
- **A document-side instruction** — new, cheap, and *unpublished* for these encoder families; the
  "no doc prompt" convention is index-prebuilding convenience, not a measured result. Non-linear on
  the doc side, so not absorbable.
- **N-grams — demoted.** Sent2Vec's own ablation says bigrams help supervised classification, not
  unsupervised similarity, and reports no retrieval numbers.
- **Contrastive is diagnosed, not dead.** Two of three suspects died by measurement
  (`m7_diag_scores.json`); the lr is the survivor (3e-3 vs a published 1e-5–3e-4, analytic
  mechanism in arXiv 2110.09348). train.py now has warmup and collapse diagnostics, and the phase-2
  screen was rebuilt around the lr so its kill criterion cannot fire on a misconfigured arm.
- Refuted: "learned weights buy nothing" was a proxy-3 artifact — on the full suite the trained
  table beats the closed-form flat optimum by **+0.021**.

## Order of work

1. **Teacher probe** (minutes) — teacher is the one expensive-to-redo decision, so it goes first.
2. **Doc-side instruction** — one re-encode to test, genuine new capacity.
3. **Count saturation** — near-zero cost.
4. **Contrastive refit** at a published lr with warmup and teacher-mined hard negatives.
5. **Teacher swap + re-encode** — after 3–4, because retention is expected to *fall* as the teacher
   strengthens and that must be measured on the cheap teacher first.
6. N-grams, if anything is still missing.

## The honest strategic note

**Two transformer layers are worth 92.5% retention** against a frozen doc tower (arXiv 2306.11550;
four layers 96.2%) versus our table's 78.5% — which clears Tier 1 *dense-only, before fusion*, at
~0.1–0.5 ms instead of 0.023 ms. **That is exactly M8's mandate.** So M7's contribution is the
zero-transformer point on the cost frontier, not the quality winner, and we should stop short of
over-investing to close a gap the next milestone closes by design.

Our multi-hop ceiling is also now a theorem rather than just our measurement: DeepMind's
arXiv 2508.21038 bounds realizable top-k document subsets by embedding dimension.

## Open for Dylan

1. **Nothing is blocking.** The stella-provenance call resolves without you.
2. **Host:** stop Windows Update auto-rebooting (Event 1074 took the box at 05:52, nothing lost).
3. Later: HF release go; and a view on shipping a teacher whose training data is undisclosed and
   whose registry lists 2 of our 6 eval sets, if stella wins the probe.
