# Brief: does v16 read as science, and what would make it exciting?

Repository: /Users/dylanc/Documents/GitHub/asymetric-dual-encoders, branch m15-whitepaper, read-only. Read-only shell commands (cat, sed -n, head, grep on named files) are allowed. Change no files, run nothing else. Never read results/frozen_eval/untouched-*, reserved qrels caches, work/m9reserve, raw M18 confirmation or sealed M19 confirmation. No repo-wide content search across results/ or work/.

## The owner's verdict on v16

"This kind of reads like an article/blog to me, not scientifically exciting enough. Also we say that Stella retains 100% in the beginning, compared to Stella, which is absurd. Feel free to revisit past decisions."

The goal has not changed: credibility with IR researchers and seasoned search engineers; findings they would cite and argue about; a research paper, not a benchmark report, a tutorial, or a blog. Constraints: no full-model retraining; about $150 of A100 time is available; the six public BEIR sets and BEIR-15 may be used freely for evaluation; the four held-out sets (FEVER, DBpedia-entity, CQADupStack android and english) stay untouched; existing results are immutable.

## Read

1. m15/PAPER.md (the v16 draft, whole thing)
2. m15/REVISION_PLAN_V16.md (the plan and the agreed claim wording; these decisions are open to challenge)
3. m15/EVIDENCE.md cards C1, C3, C3b, C10, C13, C15, C16, C17 (what the numbers actually are)
4. m15/LEARNINGS.md (33 catalogued findings across the project; material not in the paper)
5. m15/FOLLOWUP.md (experiment designs and what was rejected or deferred)
6. m15/REVIEWS/2026-10-01-sol-reader-v16.md and 2026-10-01-astra-correctness-v16.md (today's reviews, already applied)
7. m15/RELATED_WORK.md (what prior work established)

## Answer, under 1,000 words, in this order

1. **Register diagnosis.** Name the specific features that make v16 read as an article rather than a paper. Be concrete: quote sentences. Consider structure (no problem statement, hypotheses, or method section as such; findings as section titles; a "what to take from this" paragraph), voice (second-person advice, "you"), evidence presentation (numbers in prose instead of tables with intervals; no effect sizes beyond Spearman; no model of the data), and claims (does the paper pose a research question and answer it, or describe what was observed?).
2. **Scientific weight.** For each of the two findings, say what would make an IR researcher sit up. Is the teacher-selection result a correlation anecdote over 26 related checkpoints, or can it be made a tested hypothesis (for example: a regression of student quality on teacher quality, width, and family; a pre-registered prediction on held-out checkpoints; a mechanism)? Is the search-cost result a measurement, or can it be tied to a quantity the reader can compute before building (for example: a predictor of the ef multiplier from properties of the query distribution, tested across 25 spaces)? Where the current evidence cannot support more, say so.
3. **Decisions to reopen.** The current paper leads with construction (Finding A) then serving (Finding B), keeps the family as setup, demotes fusion and routing, treats the width effect as a qualifier, reports the geometry null in one sentence, and did not run the native query-precision experiment. Which of these would you reverse, and why? Would a narrower paper (one finding, done deeply) be stronger than two?
4. **Proposed shape.** A section outline for a version that reads as a paper: problem statement and research questions, related work, method, results with hypotheses stated before the numbers, analysis, limitations. Word budgets. Which figures and tables carry the argument.
5. **Three experiments or analyses, ranked**, each with claim, required outcome, plausible negative, and rough cost, that would most raise the scientific weight within the constraints. Analyses on existing data count and are cheaper.
6. **The abstract's "100%" sentence** and any other sentence you would not sign.

Disagree with the owner where the evidence warrants; he asked for that. Do not soften.
