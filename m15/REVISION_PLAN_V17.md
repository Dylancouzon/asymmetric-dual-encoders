# Proposed v17 plan: from article to research paper (2026-10-01, owner round 4)

**Status: proposed, awaiting owner decision on the title and on about $25 of pod time.**
Inputs: the owner's verdict on v16 ("reads like an article, not scientifically exciting; Stella
retaining 100% of itself is absurd"), two independent reviews on one brief
(`REVIEWS/2026-10-01-fable-register-v16.md`, `REVIEWS/2026-10-01-astra-register-v16.md`), and the
E21 conditional analysis run on existing data (`results/m15_e21_width_model.json`, method E21).

## What both reviewers said, and where they differ

Agreed diagnosis. Section titles are slogans; no research question or hypothesis precedes any
number; second-person advice and prose addressed to an imagined referee sit inside results; the
26-teacher roster, the paper's primary dataset, never appears as a table; Spearman is the only
effect size and no model is fitted to the 26 points; the contribution is stated by negation.

Agreed scientific weight. Finding A as written is a pooled correlation with an interval that admits
a moderate positive association. The exciting result is in the data already: conditional
predictability hidden by pooled rankings, with width as the competing explanation. Finding B is a
credible measurement with a null predictor; it becomes a finding only with a predictor, and the
divergence between neighbour recovery and relevance on TREC-COVID is itself worth explaining.

Agreed decisions to reverse. Width becomes the result, not a qualifier. The geometry null becomes a
table of tried predictors with intervals. Fusion and routing go to one paragraph or future work.
The precision control goes to one line. Both propose a conventional paper structure with research
questions, related work, method, hypotheses before numbers, analysis, and limitations.

Difference. Fable would narrow to one paper on teacher selection and make the Constella release a
worked instance, with a science-first title. Astra keeps two questions provisionally, construction
first, and would keep the title. Both rank the width-conditioned model first among next steps.

## E21 result (existing data, run 2026-10-01)

Student six-set score modelled on teacher score and log2 width over the 26 pooled checkpoints:

| Recipe | Model | Standardized betas (95% checkpoint bootstrap) | In-sample R2 | Leave-one-family-out Spearman |
|---|---|---|---:|---:|
| head | teacher only | teacher +0.34 [−0.16, +0.65] | 0.12 | −0.17 |
| head | teacher + width | teacher +0.69 [+0.40, +0.97], width −0.83 [−1.14, −0.60] | 0.70 | +0.69 |
| head | + dev screen | dev +0.71 [+0.36, +0.99]; teacher +0.28, width −0.22 | 0.86 | +0.77 |
| table | teacher only | teacher +0.37 [−0.30, +0.72] | 0.14 | −0.15 |
| table | teacher + width | teacher +0.63 [+0.06, +0.89], width −0.64 [−0.95, −0.32] | 0.48 | +0.40 |
| table | + dev screen | dev +0.70 [+0.31, +1.01]; teacher +0.23, width −0.24 | 0.78 | +0.80 |

Held-out test using the project's own split: the teacher + width model fitted on the ten
registered teachers ranks the 16 later exploratory checkpoints at Spearman 0.85 [0.53, 0.99] for
the head and 0.70 [0.20, 0.96] for the table, where teacher score alone gives 0.37 and 0.16.
Partial Spearman student~teacher given width: +0.58 (head), +0.45 (table). Teacher score and
width correlate +0.61, which is why the pooled correlation is near zero.

Selection regret (six-set nDCG below the best student): strongest-teacher rule 0.285 (head) and
0.152 (table); the model's leave-one-out pick 0.249 and 0.107; the student's own dev screen 0.035
and 0.000. The model explains the structure; the dev screen makes the pick. Both facts go in.

Disclosure for the paper: the hypothesis was formed after seeing all 26; the ten-to-16 split is a
held-out fit, not a prospective test. E24 below supplies the prospective test.

## v17 shape (about 3,800 main-text words)

| Section | Words | Content |
|---|---:|---|
| Abstract | 180 | Two research questions, two answers with effect sizes, the Constella instance in one sentence |
| 1 Introduction | 450 | Problem: frozen index, replaceable query encoder. RQ1: what predicts a cheap student's quality, and does the strongest teacher give the strongest student. RQ2: does query substitution change graph-search effort over the unchanged index, and is the effort predictable. Contributions as claims with numbers. |
| 2 Related work | 350 | Query-side distillation and frozen-tower alignment; stronger-teacher weaker-student; OOD queries in graph ANN; query-aware binary scoring. |
| 3 Experimental setup | 650 | Table A: the 26 teachers (family, width, readout, six-set score, registered or exploratory). Two student recipes. Fit list, dev forums, six-set exact scoring, exposure. ANN protocol: engine, graph, ef grid, two builds, tie-aware recovery, censoring. |
| 4 RQ1 results | 900 | H1 stated: the strongest teacher gives the strongest student. Table B: per-teacher table, head, retention. Figure 1: student vs teacher, colour by width, both recipes. Table C: nested models with standardized betas and intervals, leave-one-family-out, the ten-to-16 held-out prediction, E24 prospective test if run. Regret table. |
| 5 RQ2 results | 800 | H2 stated: table queries need more graph effort than teacher queries over the same index. Table D per workload: spaces above 1, censored, multiplier quantiles, recovery gap with across-space interval, relevance-loss result in full. Figure 2: recovery gap by space grouped by family. Table E: predictors tried with intervals (E22), size deconfound (E23). |
| 6 The Constella instance | 300 | Table 1 and the registered contrasts as the worked example of both questions; Zero's ef 512 and the 2.2-fold figure in one paragraph. |
| 7 Limitations | 250 | Dependence among checkpoints, three width levels, closed-form students, one engine, size-domain confound, exploratory status. |
| Appendices | | A registered evaluations in full; B construction details and costs; C search and precision controls; D fusion and routing. |

Register rules for the rewrite: every results paragraph opens with the hypothesis or question,
then the table, then the reading. No second person. No reviewer-directed sentences. Every effect
with an interval. Section titles name functions. Hedges live in Limitations, one caveat sentence
per result in the body.

## Title

Option 1 (Fable, recommended): "Teacher Selection and Search Cost for Closed-Form Query Students
over a Frozen Document Index", with Constella named in the abstract and Section 6.
Option 2 (Astra): keep "Constella: Teacher Selection and Search Cost in Query-Side Distillation".

## Experiments

| Id | Claim | Required outcome | Plausible negative | Cost | Status |
|---|---|---|---|---|---|
| E21 | Conditional on width, teacher quality predicts student quality; width is the larger effect | Done: betas above, held-out 0.85 / 0.70 | n/a | $0 | complete |
| E22 | The recovery gap across 75 space-by-workload points is predictable from declared query-placement features computed on cached vectors | At least one pre-declared feature beats the constant predictor leave-family-out, with an interval excluding zero | Nothing beats the null; RQ2 stays a measurement and the paper leans on RQ1 | about $10 pod time, one to two days | pre-specified, awaiting approval |
| E23 | The SCIDOCS attenuation is corpus size, not domain: FiQA subsampled to 25,657 documents shows a smaller gap | Gap shrinks toward the SCIDOCS value at 25k on the same queries | Gap persists at 25k, so domain or query form explains it | about $5, two to three hours | pre-specified, awaiting approval |
| E24 | The E21 rule generalises prospectively: fitted on the 26, it ranks eight new, never-scored checkpoints spanning widths | Spearman between prediction and actual above teacher-only, with an interval excluding zero; dev screen regret below the strongest-teacher rule | Prediction fails outside the roster; RQ1 keeps the held-out fit and loses the prospective claim | about $10 to $20, half a day | pre-specified, awaiting approval |

Pre-spend essentials-only Codex review precedes the first pod minute, per the owner's rule.

## Decisions for the owner

1. Title option 1 or 2.
2. Approve E22, E23, and E24 (about $25 to $35 total within the $150).
3. Confirm the register rules above, then the v17 rewrite proceeds.
