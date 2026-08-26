# M7 status

**Stage:** Stage 0 done, gate GO. Two research rounds + a Fable adversarial review done 2026-08-26
(1 BLOCKER / 8 MAJOR / 7 MINOR, all actioned). **No training restarted — next session is compute.**
Detail: `research/m7-research-2026-08-26b.md`, `research/m7-teacher-shortlist-2026-08-26.md`,
`EXPLORED.md`, `results/m7_*.json`.

## Run these first, in this order

1. **`m7src/validate_encoder.py` does not exist yet — write it and run it before the probe.**
   Every new `Spec` needs the M2/M4 loader-validation step: encode the model card's own example and
   reproduce its printed similarities. This session added stella's post-pooling Dense head only
   after review caught that it was missing; the loader is now *plausible*, not *validated*, and a
   bad loader would silently decide the teacher.
2. `m7src/teacher_probe.py` — picks stella vs gte-large on measurement. ~70K docs/candidate.
3. Phase-2 screen (`program.phase2_screen`) — the decisive contrastive test at a published lr.
4. Doc-side instruction test (§below), count saturation, then the teacher swap.

## Bars, with honest intervals

`results/m7_calibration.json` now carries a real 95% **prediction** interval (regression sigma on
n−2 df, t(7), and the extrapolation widening term). The old ±2·resid_sd band understated the
half-width at stella by ~68%. **Consequence: no candidate clears Tier 1 CI-resolved on the dense
arm at any plausible retention**; stella clears Tier 2 lower-bound at ≥0.88 retention, gte-large at
≥0.91.

| teacher | six est | 95% PI | Tier 2 lower-bound clears | Tier 1 lower-bound clears |
|---|---|---|---|---|
| bge-base (current) | 0.5082 | ±0.024 | no, at any retention | no |
| gte-large-en-v1.5 | 0.5473 | ±0.030 | at ≥0.91 | no |
| stella_en_400M_v5 | 0.5562 | ±0.035 | at ≥0.88 | no |

Two caveats the interval does **not** contain, both pushing the same way: it carries no uncertainty
from the retention factor, and **0.7853 is itself optimistic** — it is dev-macro retention whose
only CI-resolved win is nq-250k, the training-adjacent component, while per-component retention runs
64% (cqadup-programmers) to 89% (nq-250k) and the six lean toward the 64% end. Also, stella's
registry lists **ArguAna and FiQA2018** — 2/6 of our target suite but only 2/15 of the MTEB
predictor — so the affine fit cannot express that upward bias.

## Fusion: measured, real, and narrower than first reported

`results/m7_fusion_report_p1-objB.json` (dev, convex w=0.5, depth 1000, in-sample over a 12-point
grid): **dense 0.4795 · BM25 0.4525 · fused 0.5520 (+0.0725)**. All four components gain,
CI-resolved. But **+0.1418 of it is hotpotqa**, and the six contain no multi-hop component. The
hotpotqa-free mean is **+0.049**, which is the defensible transfer estimate:

| | bar | dense needed | retention on bge-base | on gte-large |
|---|---|---|---|---|
| Tier 1 (fused, +0.049) | 0.4868 | 0.4374 | **86.1%** | **79.9%** |
| Tier 2 (dense alone) | 0.4583 | 0.4583 | 90.2% | 83.7% |

So Tier 1 is still the easier bar, but **not "already there" for gte-large** — 79.9% is above
today's 78.5%. An earlier version of this file said otherwise off the additive +0.0725; withdrawn.
Also unexamined: six-set BM25 (0.4174) is weaker than dev BM25 (0.4525), and weakest exactly on
FiQA, the row fusion is meant to rescue.

## What is real on retention

- **Centering / whitening / top-PC removal / SIF weighting add NO capacity** — absorbable
  (`m7_absorb_check.json`, machine precision at fp32/fp16; the released **int8** artifact could
  differ slightly since absorbing μ changes per-row absmax, and G4's 0.005 bar would catch it).
  Demoted to an init experiment. p1-objB's learned weights already *are* IDF-like.
- **Count saturation** — promoted. Non-absorbable, ~free; BM25, SPLADE and NUMEN converge on it.
- **A document-side instruction** — cheap and *unpublished* for these families. Non-linear on the
  doc side, so not absorbable. NB: doc2query-style expansion is **demoted, not closed** — see
  EXPLORED for why the cited evidence does not reach our regime.
- **N-grams — demoted.** Sent2Vec's own ablation: bigrams help supervised classification, not
  unsupervised similarity, and it reports no retrieval numbers.
- **Contrastive: lr is the leading untested hypothesis, not a diagnosis.** Two named suspects are
  *bounded small* (`m7_diag_scores.json`: fn_margin removes 4.3% of the top-100 hardest; 84% of
  queries have no random negative outscoring the positive) — but measured in the **teacher's**
  geometry, not the student's, and the suspect list never included Adam-on-sparse-rows dynamics or
  cross-query row interference. The decisive test is the phase-2 `sane-5e5` vs `warmup-only` arms.
  Watch `fn_masked_frac`: at margin 0.05 with *mined* negatives the filter will bite far harder.
- The proxy-3 near-tie with the closed-form flat table **did not replicate** on the full suite
  (+0.021) — but that is uncontrolled (objective, weights and optimizer all differ) and has no CI,
  so it refutes "training buys nothing over closed-form flat distillation", **not** "learned weights
  buy nothing". p4-weights remains the clean test.

## The strategic note, properly hedged

A 2-layer distilled query tower reportedly retains **92.5%** on a frozen doc tower (arXiv 2306.11550,
4-layer 96.2%). That is retention *of its own teacher on its own BEIR selection* — not comparable
denominators to our 78.5%, and second-hand. **Verify what the 92.5% is a percentage of before
letting it steer effort.** If it holds, it clears Tier 1 dense-only at ~0.1–0.5 ms, and it is
exactly M8's mandate — which would make M7 the zero-transformer cost-frontier point rather than the
quality winner. Directionally important, numerically unverified.

DeepMind's arXiv 2508.21038 bounds realizable top-k subsets by embedding dimension. It is a
**citation that our ceiling is a known class of limitation**, not a theorem about our table — our
own teacher does not share the hotpotqa deficit (0.667 dev).

## Open for Dylan

1. **Nothing is blocking.** The stella-provenance call resolves without you.
2. **Host:** stop Windows Update auto-rebooting (Event 1074 took the box at 05:52; nothing lost).
3. Later: HF release go; and a view on shipping a teacher whose training data is undisclosed and
   whose registry lists 2 of our 6 eval sets, if stella wins the probe.
