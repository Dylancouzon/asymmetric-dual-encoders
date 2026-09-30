You are an adversarial reviewer (read-only) of the agreed plan for an arXiv cs.IR research preprint, before the project hands it to a fresh session that will execute it. The plan is m15/PLAN.md in the repository below (branch m15-whitepaper, checked out). Read it in full, then CLAUDE.md and instructions-m15.md (both carry uncommitted 2026-09-30 amendments you should read as current).

Repository (read-only): /Users/dylanc/Documents/GitHub/asymetric-dual-encoders.
HARD ACCESS RULES:
- Never read or list `results/frozen_eval/untouched-*`, anything under `work/`, any reserved qrels cache, or raw queries/qrels of FEVER, DBpedia-entity, cqadupstack-android, cqadupstack-english.
- No recursive content search (grep -r, rg over a directory, find) across `results/` or `work/`. You may open by exact name any file under `results/` that PLAN.md, m15/EVIDENCE_INDEX.md or a STATUS/FINDINGS file names explicitly. Recursive search is fine in source dirs (m7src m8src m9src m10src m12src m13src m20src bench scripts) and the markdown of m7/ m8/ m9/ m10/ m12/ m13/ m15/ m20/ research/ (not research/archive).
- No web searches naming the four reserved datasets. No edits, no commits, no training or evaluation runs.

Goal the findings are measured against: a research paper that creates new knowledge for the IR industry and is unimpeachable to a hostile IR reviewer, executed with this project's evidence and access discipline, with measurement scaffolding proportional to each decision.

Essential-only (Dylan, 2026-09-15): report only what is essential to that goal: a claim in the plan the evidence does not support as written, a wrong number, a planned measurement that is invalid, leaky, cannot answer its question, or violates a registration or access rule, a missing disclosure that would sink the paper, a real execution blocker, and scope that is over-engineered or under-specified for the decision. No wording, style or formatting notes, and no hypothetical failures the documents already guard against. If you find nothing essential, say so plainly.

Check in particular:
1. Each measurement E1-E8: validity of the design for the stated question; leakage between fit and evaluation (e.g. E6's threshold data vs its 12 evaluation datasets; E8's lambda and fit list vs the six BEIR sets); whether E2's 1M MS MARCO subset (all dev-judged positives plus random fill) biases nDCG under ANN relative to exact and how it must be reported; whether recall@10 vs exact plus nDCG under ANN can answer "does the saving survive the search"; E4's controls; E8's tower set (shared BERT WordPiece vocab assumption, arctic-l mean readout), contamination per tower on the six, and what E8 can and cannot establish about "distillability belongs to the tower"; whether any measurement touches a reserved or known-test dataset in any role (the reserved four are spent and known-test and may never decide anything).
2. Every number in PLAN.md against its source file.
3. Whether the findings as worded in PLAN.md overclaim relative to their evidence rating.
4. Execution blockers the handoff must name (data not on this Mac, e.g. the TRAIN list `work/trainq_texts.json` referenced by scripts/teacher_learnability.py; MS MARCO licence role; Mac memory limits).
5. The single most likely reason this plan fails to produce a paper that creates new knowledge, and the minimal change.

Output: numbered findings, each with severity P0 (blocks execution), P1 (fix before the method file), P2 (fix during execution), file:line evidence, and the minimal fix. Then a one-line verdict. Under 1,200 words.
