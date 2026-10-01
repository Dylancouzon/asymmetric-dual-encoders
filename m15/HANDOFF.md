# M15 handoff (2026-10-01)

Branch `m15-whitepaper`. Read **OWNER_PREFERENCES.md**, **LEARNINGS.md**, the last LOG.md entry,
and **PAPER.md** before revising. The owner keeps challenging purpose, readability, and value;
previous agent passes are not owner acceptance.

For coworker review, start at **REVIEW_GUIDE.md**. **BRANCH_INVENTORY.md / BRANCH_FILES.tsv** list
the baseline tracked content; **PROVENANCE_AUDIT.md** resolves exact source versions and states
which inputs are external/unavailable. A fresh clone supports substantive review, not every exact
rerun. EVIDENCE_INDEX.md is explicitly superseded historical discovery material.
**EDITORIAL_RATIONALE.md** explains the current choice, its strongest objection, the revision
history, and eight alternative directions with evidence, limitations, and what could change the
choice. It is an agent recommendation for coauthor discussion, not owner acceptance.

## Current deliverables

- **PAPER.md v15:** “Constella: Lower-Cost Queries over a Fixed Document Index.” An empirical
  whitepaper, not a tutorial. Roughly 3,300 main words versus roughly 5,900 in v14; two main figures.
  The PDF has 11 pages including methods, original contrasts, held-out aggregates, and references.
- **LEARNINGS.md:** 33 candidate lessons from the whole history (M0–M23), with milestone coverage,
  evidence, production decision, strength, limits, editorial selection, and justified research gaps.
  Unscheduled M16 and incomplete release/archive work are identified as lacking new retrieval
  evidence. Keep this catalog broader than the paper.
- **PAPER_EVIDENCE_MAP.md:** file-level provenance for the current manuscript. Internal repository
  paths are removed from the manuscript; normal literature references and artifact availability
  remain. EVIDENCE.md retains all numerical cards, including material omitted editorially.
- **AUTHORS.json:** provisional owner-supplied author order, Qdrant affiliation, and all four emails.
  The PDF build reads this roster. Query/encoding/loading latency uses ms consistently.
- **latex/paper.tex / paper.pdf:** build with `sh m15/latex/build.sh` (pandoc/tectonic installed).
  Figures use scientific Matplotlib, not brand diagrams. Regenerate only changed figure functions
  with `.venv-mac/bin/python`; the main figures are f4_system and f3_towers, printed as Figures 1/2.

## Direction and owner corrections

The paper's central evidence is the distinction between encoding savings and retrieval savings
with the same document vectors. The teacher/table reversal is a separate construction finding
before choosing a teacher/index. The fixed-index search and fusion results precede that section.
Fusion-depth reversal and actual runtime failures bring useful findings from older milestones into
view. Router/blend failures remain concise. Query precision is a bounded control, not an already
measured native remedy.

Owner rejects “search service,” repository filenames used as paper citations, a verbose inventory,
and a goal unclear to a search engineer. The ultimate audience outcome is interesting, applicable,
worth sharing, and production value. Owner explicitly reiterates **whitepaper, not tutorial**.
The final draft presents findings and interpretation, removing repeated instructional prompts.
A worked absolute-quality/time example is explicitly hypothetical and limited to the measured
rows; it is not a prospectively selected routing policy or recommended SLO.

The full-history audits and Astra synthesis review are in REVIEWS/2026-10-01-learning-*.md.
V15 reader and correctness reviews plus their focused closures are in REVIEWS/2026-10-01-v15-*.md.
Root dispositions are in REVIEWS/2026-10-01-v15-synthesis.md. They resolve identified issues;
none certifies owner acceptance, publication readiness, or a breakthrough.

## Scientific boundaries

- Nano/Zero quality budgets are different, with mismatched training data. BEIR-15 aggregates are
  descriptive. Original registered clean-four/all-six tests and reserved aggregate results remain
  visible, including Zero's failed Holm BM25 test and Nano's unfavorable broader references.
- E2 measures sequential warmed queries on FiQA and a positive-preserving 1M validation diagnostic.
  Encode-plus-search is a sum of separately measured per-query phases. It is not a request stopwatch,
  throughput, concurrent failover, or equal-absolute-quality speedup.
- E8/E8x screens one ridge-table recipe over 26 checkpoints (10 registered, 16 exploratory), each
  with its own teacher index. It does not select trained Zero/Nano or permit unrelated teacher swaps.
- E16's strong native FiQA interaction has a weaker primary E17 replica. E18 fixes document sign
  codes and tests sign versus full-query candidate scoring, without a graph or native latency.
  Larger absolute Zero gains have more miss headroom; mechanism/proportional sensitivity unproven.
- $95 is final Nano optimization at a historical rental rate, not complete reproduction cost.
- M12 fusion-depth curve is descriptive development evidence, bm25s with Lucene settings and
  fixed convex weight .8. The DBSF recommendation changed on implementability grounds after prior
  evaluation. Native BM25 substitution and equivalence are not established.
- Runtime failures are historical candidates/defects, not claims that current released fp32 paths
  fail. Small internal specialization/patch studies do not establish an improved encoder.

## Next work

1. Continue editorial selection against LEARNINGS.md and the owner's latest feedback. Do not fill
   the paper with every candidate or default back to a methods inventory/checklist.
2. For stronger research, choose a specific claim first. Native magnitude-preserving query scoring
   at matched recovery/relevance and measured cost has the clearest deployment value. Account for
   collection rebuild/graph variation. Another teacher space would test reach beyond Stella.
   FOLLOWUP.md preserves these options. No full retraining is authorized or needed by this plan.
3. Before actual publication, reconfirm the provisional roster only if it changes, and verify release/licence state. No public paper
   publication happened during this rewrite. Twenty literature references retain their existing
   primary-source audits in RELATED_WORK.md.

## Execution and access

All existing result payloads remain immutable. No new experiment, teacher/document encoding,
model training, rental, or protected-data access occurred in the full-history rewrite. M15 historical
cloud spend remains $22.79; Runpod pod `k3aee2m68765em` is stopped. Container `/opt/m15-venv` is lost
on restart; rebuild from `m13/cloud_requirements.txt` with uv at
`/home/dylan/.local/share/m13-uv-0.12.5/bin/uv`. Remote worktree `/home/dylan/m15run`.

Never read `results/frozen_eval/untouched-*`, reserved qrels caches, `work/m9reserve`, raw M18
confirmation, or sealed M19 confirmation. Never overwrite `results/perquery.json`. No repo-wide
content searches over results/work. Published aggregate receipts are the allowed reserved evidence.
Commit and push coherent batches frequently; no force-push. Models: Sol writer/reader, Astra
adversarial correctness reviewer, per mandate and owner permission.

## Mac pitfalls learned this milestone

- One MPS job at a time; an MPS encode hung for six hours in a Metal wait (process state `UN`). Watch
  shard timestamps, not only the log. Identify Mac jobs by pid and start time: `ps` truncates the
  command line, which once made a live job look dead and caused a duplicate run.
- Latency runs need an idle machine; background load moved FiQA encode latencies by up to 1.7x.
- `pgrep -f` over SSH matches its own command line; use a bracket pattern (`[e]8x_towers`).
