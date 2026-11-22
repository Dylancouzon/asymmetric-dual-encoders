# Common review brief, M15 paper draft

**Goal the findings are measured against:** `m15/PAPER.md` must be an arXiv-style preprint whose
every claim is true, traceable to a committed result file, and defensible against a hostile reader
who knows the retrieval literature. Its focus is net-new knowledge.

**Essential-only rule.** Report what is essential to that goal: a claim that is wrong or
unsupported, a number that does not match its source, a statistical reading that overstates what
was established, a novelty claim the literature already covers, missing evidence or access
discipline damage, and scope deviation. Wording, style, formatting and preference restructuring are
out of scope at this stage. If you find nothing essential, say so plainly. Do not manufacture
findings.

**Access rules, hard.** Never read `results/frozen_eval/untouched-*`, reserved qrels caches or
anything under `work/m9reserve`. Never run repo-wide or recursive content searches across
`results/` or `work/`. Read only named files. Report every file you opened.

**Files in scope:** `m15/PAPER.md`, `m15/EVIDENCE_INDEX.md`, `m15/NOVELTY.md`, `m15/HOTSWAP.md`,
`m15/PLAN.md`, `instructions-m15.md`, and as sources: `m21/BENCHMARKS.md`, `m7/STATUS.md`,
`m8/FINDINGS.md`, `m9/FINDINGS.md`, `m9/RESULTS.md`, `m10/FINDINGS.md`, `m12/FINDINGS.md`,
`m13/STATUS.md`, `m13/FINDINGS.md`, `m14/STATUS.md`, `m14/MODEL_CARD.md`, `m11/STATUS.md`,
`research/m1-m6-findings.md`, `research/m7-novelty.md`, `PROJECT_STATUS.md`, `README.md`.

**Context the reviewer needs:** the reserved four and BEIR-15 are unspent and running under M20, so
section 5.4 is deliberately empty. The owner ruled that limitations stay in full while most failed
approaches drop to one appendix line each.
