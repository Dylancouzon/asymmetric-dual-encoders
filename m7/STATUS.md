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

## Order of work

1. **Fusion (dense + BM25) measured on dev** — in flight. Our profile is ideally complementary
   (we beat BM25 on NQ by +0.145 and lose HotpotQA by −0.032), and it decides whether Tier 1 needs
   anything beyond teacher + retention. Sets the Tier-1 candidate.
2. **Teacher probe** — minutes, no GPU-hours. Picks stella vs gte-large on a measured number.
3. **Contrastive refit at a published lr**, warmup, teacher-mined hard negatives from the frozen
   index (permanently valid — our tower never moves). The main retention lever with real evidence.
4. **Count saturation** — near-zero cost, genuinely new capacity.
5. **Teacher swap + corpus re-encode** (~4× compute, 1024d storage). Deliberately *after* 3–4:
   capacity-gap literature expects retention to *fall* as the teacher strengthens, so measure it on
   the cheap teacher first. The re-encode is the costliest step and should happen once.
6. **N-grams** — days, unknown magnitude, and it adds parameters when our problem is
   generalisation: the capacity probe hits ~1.0, so the frozen-tower tax is a *generalisation* gap,
   not an expressivity limit. Needs a bounded frequency-selected phrase vocabulary, priced.

## Open for Dylan

1. **Nothing is blocking.** The stella-provenance call from last status is resolved without you:
   NovaSearch ships no vector product, and Alibaba (stella's lineage, and gte-large itself) is
   admissible under the relaxed rule.
2. **Host:** stop Windows Update auto-rebooting (Event 1074 took the box at 05:52 on 2026-08-26;
   nothing lost).
3. HF release go, later. Plus: if stella wins the probe, you may want a view on shipping a teacher
   whose training data is undisclosed and whose registry lists 2 of our 6 eval sets.
