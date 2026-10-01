# Branch inventory

Tracked baseline snapshot: [41d436c2](https://github.com/Dylancouzon/asymmetric-dual-encoders/tree/41d436c2a7508204f5dd64bce251cd07fd93a112) on `m15-whitepaper` before this reviewer package was added. **1,841 tracked files.** [BRANCH_FILES.tsv](BRANCH_FILES.tsv) lists every baseline path, Git blob identity, and byte size from tree metadata; it does not read file contents.

The new reviewer package adds README.md, REVIEW_GUIDE.md, BRANCH_INVENTORY.md, BRANCH_FILES.tsv,
PROVENANCE_AUDIT.md, EDITORIAL_RATIONALE.md, and REVIEW_APPENDIX.md under m15/. Later collaboration
receipts/links are recorded in HANDOFF.md and LOG.md. Use `git ls-files` for the checkout you are
reviewing if it differs from the pinned snapshot.

## Material by purpose

| Area | What it contains |
|---|---|
| `m15/PAPER.md`, `m15/latex/` | Current manuscript, LaTeX build script/source, rendered PDF |
| `m15/AUTHORS.json` | Provisional author order, affiliation, emails |
| `m15/PAPER_EVIDENCE_MAP.md`, `EVIDENCE.md` | Current manuscript provenance and readable numerical cards |
| `m15/LEARNINGS.md` | 33 potential lessons from the entire history, including omitted/negative findings |
| `m15/MEASUREMENTS.md`, `PLAN.md`, `FOLLOWUP.md` | Methods/amendments, historical planning, bounded future research options |
| `m15/RELATED_WORK.md`, `REVIEWS/` | Primary literature notes, review briefs/findings, dispositions and historical iterations |
| `m15/OWNER_PREFERENCES.md`, `HANDOFF.md`, `LOG.md` | Owner direction, current handoff, revision/experiment history |
| `m15/figures/`, `make_evidence.py` | Figure sources/assets (including unused figures), evidence-card generator |
| `m15src/` | 29 experiment, encoder, input, analysis, and check files |
| `results/m15*.json` | 26 M15 result/freeze/diagnostic records; exact source versions linked in the provenance audit |
| `m7/`–`m14/`, `m17/`–`m23/`, corresponding `*src/` | Milestone findings, decisions, methods, code and release/integration work where present; navigate with LEARNINGS.md |
| `results/` | Historical registered/exploratory receipts, published aggregates and score rows; some paths are protected |
| `research/`, `audit/` | Earlier study notes, licensing evidence, archived guidance and access/protocol records |
| Root documentation | CLAUDE.md, instructions-m15.md, ROADMAP.md, PROJECT_STATUS.md, HARNESS.md, README.md and historical mandates |
| `bench/`, `scripts/`, `vendor/` | Harnesses, exporters/integration checks, supported dependency components |

A missing directory in the table does not mean a milestone is missing: numbering changed, some mandates are root files, and unscheduled M16 produced no experiment. LEARNINGS.md explains milestone coverage.

## Baseline counts by top-level location

| Location | Tracked files |
|---|---:|
| `(root)` | 36 |
| `.claude` | 7 |
| `audit` | 13 |
| `bench` | 18 |
| `m10` | 18 |
| `m10src` | 76 |
| `m11` | 16 |
| `m12` | 6 |
| `m12src` | 6 |
| `m13` | 36 |
| `m13src` | 29 |
| `m14` | 7 |
| `m15` | 113 |
| `m15src` | 29 |
| `m17` | 9 |
| `m17src` | 29 |
| `m18` | 10 |
| `m18src` | 26 |
| `m19` | 47 |
| `m19src` | 33 |
| `m20` | 9 |
| `m20src` | 7 |
| `m21` | 3 |
| `m22` | 5 |
| `m7` | 10 |
| `m7src` | 80 |
| `m8` | 7 |
| `m8src` | 43 |
| `m9` | 17 |
| `m9src` | 29 |
| `research` | 166 |
| `results` | 834 |
| `scripts` | 52 |
| `vendor` | 3 |
| `work` | 12 |

## Exact reviewer-relevant paths

### M15 paper and reviews

- [AUTHORS.json](AUTHORS.json)
- [EVIDENCE.md](EVIDENCE.md)
- [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md)
- [FOLLOWUP.md](FOLLOWUP.md)
- [HANDOFF.md](HANDOFF.md)
- [HOTSWAP.md](HOTSWAP.md)
- [LEARNINGS.md](LEARNINGS.md)
- [LOG.md](LOG.md)
- [MEASUREMENTS.md](MEASUREMENTS.md)
- [NOVELTY.md](NOVELTY.md)
- [OWNER_PREFERENCES.md](OWNER_PREFERENCES.md)
- [PAPER.md](PAPER.md)
- [PAPER_EVIDENCE_MAP.md](PAPER_EVIDENCE_MAP.md)
- [PLAN.md](PLAN.md)
- [PROOF_ABSORB.md](PROOF_ABSORB.md)
- [RELATED_WORK.md](RELATED_WORK.md)
- [REVIEWS/2026-09-30-andrey-review-v7.md](REVIEWS/2026-09-30-andrey-review-v7.md)
- [REVIEWS/2026-09-30-astra-e8-driver.md](REVIEWS/2026-09-30-astra-e8-driver.md)
- [REVIEWS/2026-09-30-astra-e8-rereview.md](REVIEWS/2026-09-30-astra-e8-rereview.md)
- [REVIEWS/2026-09-30-astra-e9e10.md](REVIEWS/2026-09-30-astra-e9e10.md)
- [REVIEWS/2026-09-30-astra-evidence-ratings.md](REVIEWS/2026-09-30-astra-evidence-ratings.md)
- [REVIEWS/2026-09-30-astra-measurements-rereview.md](REVIEWS/2026-09-30-astra-measurements-rereview.md)
- [REVIEWS/2026-09-30-astra-measurements-review.md](REVIEWS/2026-09-30-astra-measurements-review.md)
- [REVIEWS/2026-09-30-astra-novelty.md](REVIEWS/2026-09-30-astra-novelty.md)
- [REVIEWS/2026-09-30-astra-paper-v4.md](REVIEWS/2026-09-30-astra-paper-v4.md)
- [REVIEWS/2026-09-30-astra-paper-v6.md](REVIEWS/2026-09-30-astra-paper-v6.md)
- [REVIEWS/2026-09-30-astra-plan-review.md](REVIEWS/2026-09-30-astra-plan-review.md)
- [REVIEWS/2026-09-30-astra-proposal-v3.md](REVIEWS/2026-09-30-astra-proposal-v3.md)
- [REVIEWS/2026-09-30-astra_paper_v5.md](REVIEWS/2026-09-30-astra_paper_v5.md)
- [REVIEWS/2026-09-30-fable-novelty.md](REVIEWS/2026-09-30-fable-novelty.md)
- [REVIEWS/2026-09-30-fable-research-shape.md](REVIEWS/2026-09-30-fable-research-shape.md)
- [REVIEWS/2026-09-30-fable-shape-v3.md](REVIEWS/2026-09-30-fable-shape-v3.md)
- [REVIEWS/2026-09-30-opus-e8-driver.md](REVIEWS/2026-09-30-opus-e8-driver.md)
- [REVIEWS/2026-09-30-sol-e8x.md](REVIEWS/2026-09-30-sol-e8x.md)
- [REVIEWS/2026-09-30-sol-reader-v3.md](REVIEWS/2026-09-30-sol-reader-v3.md)
- [REVIEWS/2026-09-30-sol-unreviewed-code.md](REVIEWS/2026-09-30-sol-unreviewed-code.md)
- [REVIEWS/2026-09-30-sol_reader_v5.md](REVIEWS/2026-09-30-sol_reader_v5.md)
- [REVIEWS/2026-09-30-sonnet-literature.md](REVIEWS/2026-09-30-sonnet-literature.md)
- [REVIEWS/2026-09-30-sonnet-repo-survey.md](REVIEWS/2026-09-30-sonnet-repo-survey.md)
- [REVIEWS/2026-10-01-astra-v10-rereview.md](REVIEWS/2026-10-01-astra-v10-rereview.md)
- [REVIEWS/2026-10-01-astra_paper_v9.md](REVIEWS/2026-10-01-astra_paper_v9.md)
- [REVIEWS/2026-10-01-e16-pilot-review.md](REVIEWS/2026-10-01-e16-pilot-review.md)
- [REVIEWS/2026-10-01-e17-replication-review.md](REVIEWS/2026-10-01-e17-replication-review.md)
- [REVIEWS/2026-10-01-e18-precision-review.md](REVIEWS/2026-10-01-e18-precision-review.md)
- [REVIEWS/2026-10-01-followup-design.md](REVIEWS/2026-10-01-followup-design.md)
- [REVIEWS/2026-10-01-followup-literature.md](REVIEWS/2026-10-01-followup-literature.md)
- [REVIEWS/2026-10-01-humanizer-v8.md](REVIEWS/2026-10-01-humanizer-v8.md)
- [REVIEWS/2026-10-01-learning-audit-early.md](REVIEWS/2026-10-01-learning-audit-early.md)
- [REVIEWS/2026-10-01-learning-audit-late.md](REVIEWS/2026-10-01-learning-audit-late.md)
- [REVIEWS/2026-10-01-learning-audit-middle.md](REVIEWS/2026-10-01-learning-audit-middle.md)
- [REVIEWS/2026-10-01-learning-synthesis-review.md](REVIEWS/2026-10-01-learning-synthesis-review.md)
- [REVIEWS/2026-10-01-owner-broader-correctness.md](REVIEWS/2026-10-01-owner-broader-correctness.md)
- [REVIEWS/2026-10-01-owner-broader-reader.md](REVIEWS/2026-10-01-owner-broader-reader.md)
- [REVIEWS/2026-10-01-owner-broader-synthesis.md](REVIEWS/2026-10-01-owner-broader-synthesis.md)
- [REVIEWS/2026-10-01-owner-correctness.md](REVIEWS/2026-10-01-owner-correctness.md)
- [REVIEWS/2026-10-01-owner-literature.md](REVIEWS/2026-10-01-owner-literature.md)
- [REVIEWS/2026-10-01-owner-reader.md](REVIEWS/2026-10-01-owner-reader.md)
- [REVIEWS/2026-10-01-owner-synthesis.md](REVIEWS/2026-10-01-owner-synthesis.md)
- [REVIEWS/2026-10-01-owner-v11-correctness.md](REVIEWS/2026-10-01-owner-v11-correctness.md)
- [REVIEWS/2026-10-01-owner-v11-reader.md](REVIEWS/2026-10-01-owner-v11-reader.md)
- [REVIEWS/2026-10-01-quantization-semantics.md](REVIEWS/2026-10-01-quantization-semantics.md)
- [REVIEWS/2026-10-01-sol_reader_v9.md](REVIEWS/2026-10-01-sol_reader_v9.md)
- [REVIEWS/2026-10-01-v13-correctness.md](REVIEWS/2026-10-01-v13-correctness.md)
- [REVIEWS/2026-10-01-v13-reader.md](REVIEWS/2026-10-01-v13-reader.md)
- [REVIEWS/2026-10-01-v13-synthesis.md](REVIEWS/2026-10-01-v13-synthesis.md)
- [REVIEWS/2026-10-01-v14-correctness.md](REVIEWS/2026-10-01-v14-correctness.md)
- [REVIEWS/2026-10-01-v14-reader.md](REVIEWS/2026-10-01-v14-reader.md)
- [REVIEWS/2026-10-01-v14-synthesis.md](REVIEWS/2026-10-01-v14-synthesis.md)
- [REVIEWS/2026-10-01-v15-correctness.md](REVIEWS/2026-10-01-v15-correctness.md)
- [REVIEWS/2026-10-01-v15-reader.md](REVIEWS/2026-10-01-v15-reader.md)
- [REVIEWS/2026-10-01-v15-synthesis.md](REVIEWS/2026-10-01-v15-synthesis.md)
- [REVIEWS/BRIEF-common.md](REVIEWS/BRIEF-common.md)
- [REVIEWS/andrey-2026-09-17.md](REVIEWS/andrey-2026-09-17.md)
- [REVIEWS/briefs-2026-09-30/astra_brief.md](REVIEWS/briefs-2026-09-30/astra_brief.md)
- [REVIEWS/briefs-2026-09-30/astra_brief2.md](REVIEWS/briefs-2026-09-30/astra_brief2.md)
- [REVIEWS/briefs-2026-09-30/astra_brief3.md](REVIEWS/briefs-2026-09-30/astra_brief3.md)
- [REVIEWS/briefs-2026-09-30/astra_brief4.md](REVIEWS/briefs-2026-09-30/astra_brief4.md)
- [REVIEWS/briefs-2026-09-30/astra_e9e10_brief.md](REVIEWS/briefs-2026-09-30/astra_e9e10_brief.md)
- [REVIEWS/briefs-2026-09-30/astra_measurements_brief.md](REVIEWS/briefs-2026-09-30/astra_measurements_brief.md)
- [REVIEWS/briefs-2026-09-30/astra_measurements_rereview.md](REVIEWS/briefs-2026-09-30/astra_measurements_rereview.md)
- [REVIEWS/briefs-2026-09-30/astra_paper_v4.md](REVIEWS/briefs-2026-09-30/astra_paper_v4.md)
- [REVIEWS/briefs-2026-09-30/astra_paper_v5.md](REVIEWS/briefs-2026-09-30/astra_paper_v5.md)
- [REVIEWS/briefs-2026-09-30/e8_driver_brief.md](REVIEWS/briefs-2026-09-30/e8_driver_brief.md)
- [REVIEWS/briefs-2026-09-30/e8_rereview_brief.md](REVIEWS/briefs-2026-09-30/e8_rereview_brief.md)
- [REVIEWS/briefs-2026-09-30/plan_v4.md](REVIEWS/briefs-2026-09-30/plan_v4.md)
- [REVIEWS/briefs-2026-09-30/proposal.md](REVIEWS/briefs-2026-09-30/proposal.md)
- [REVIEWS/briefs-2026-09-30/sol_reader_v3.md](REVIEWS/briefs-2026-09-30/sol_reader_v3.md)
- [REVIEWS/briefs-2026-09-30/sol_reader_v5.md](REVIEWS/briefs-2026-09-30/sol_reader_v5.md)
- [REVIEWS/briefs-2026-09-30/sol_unreviewed_code.md](REVIEWS/briefs-2026-09-30/sol_unreviewed_code.md)
- [REVIEWS/briefs-2026-09-30/spine.md](REVIEWS/briefs-2026-09-30/spine.md)
- [REVIEWS/codex-sol-2026-09-17.md](REVIEWS/codex-sol-2026-09-17.md)
- [REVIEWS/codex-sol-rereview-2026-09-17.md](REVIEWS/codex-sol-rereview-2026-09-17.md)
- [REVIEWS/fable-hostile-2026-09-17.md](REVIEWS/fable-hostile-2026-09-17.md)
- [REVIEWS/factcheck-2026-09-17.md](REVIEWS/factcheck-2026-09-17.md)
- [e8_exposure.json](e8_exposure.json)
- [figures/f1_frontier.pdf](figures/f1_frontier.pdf)
- [figures/f1_frontier.png](figures/f1_frontier.png)
- [figures/f2_per_dataset.pdf](figures/f2_per_dataset.pdf)
- [figures/f2_per_dataset.png](figures/f2_per_dataset.png)
- [figures/f3_towers.pdf](figures/f3_towers.pdf)
- [figures/f3_towers.png](figures/f3_towers.png)
- [figures/f4_system.pdf](figures/f4_system.pdf)
- [figures/f4_system.png](figures/f4_system.png)
- [figures/f5_routing.pdf](figures/f5_routing.pdf)
- [figures/f5_routing.png](figures/f5_routing.png)
- [figures/f6_precision.pdf](figures/f6_precision.pdf)
- [figures/f6_precision.png](figures/f6_precision.png)
- [figures/make_figures.py](figures/make_figures.py)
- [latex/.gitignore](latex/.gitignore)
- [latex/build.sh](latex/build.sh)
- [latex/paper.pdf](latex/paper.pdf)
- [latex/paper.tex](latex/paper.tex)
- [make_evidence.py](make_evidence.py)

### M15 experiment code

- [m15src/chain2.sh.txt](../m15src/chain2.sh.txt)
- [m15src/common.py](../m15src/common.py)
- [m15src/e10_router.py](../m15src/e10_router.py)
- [m15src/e11_mechanism.py](../m15src/e11_mechanism.py)
- [m15src/e11b_margin.py](../m15src/e11b_margin.py)
- [m15src/e12_failure_modes.py](../m15src/e12_failure_modes.py)
- [m15src/e12b_fragility.py](../m15src/e12b_fragility.py)
- [m15src/e13_robustness.py](../m15src/e13_robustness.py)
- [m15src/e14_screen_checks.py](../m15src/e14_screen_checks.py)
- [m15src/e15_decision_audit.py](../m15src/e15_decision_audit.py)
- [m15src/e16_quantization_pilot.py](../m15src/e16_quantization_pilot.py)
- [m15src/e17_quantization_replication.py](../m15src/e17_quantization_replication.py)
- [m15src/e18_query_precision.py](../m15src/e18_query_precision.py)
- [m15src/e1_latency.py](../m15src/e1_latency.py)
- [m15src/e2_ann.py](../m15src/e2_ann.py)
- [m15src/e2_encode.py](../m15src/e2_encode.py)
- [m15src/e4_prefix.py](../m15src/e4_prefix.py)
- [m15src/e5_oracle.py](../m15src/e5_oracle.py)
- [m15src/e6_router.py](../m15src/e6_router.py)
- [m15src/e8_towers.py](../m15src/e8_towers.py)
- [m15src/e8x_towers.py](../m15src/e8x_towers.py)
- [m15src/e9_blend.py](../m15src/e9_blend.py)
- [m15src/encode_more.py](../m15src/encode_more.py)
- [m15src/encoders15.py](../m15src/encoders15.py)
- [m15src/examples.py](../m15src/examples.py)
- [m15src/pod_encode.py](../m15src/pod_encode.py)
- [m15src/roster_ids.py](../m15src/roster_ids.py)
- [m15src/test_m15.py](../m15src/test_m15.py)
- [m15src/vectors15.py](../m15src/vectors15.py)

### M15 results and frozen settings

- [results/m15_e10_frozen.json](../results/m15_e10_frozen.json)
- [results/m15_e10_router.json](../results/m15_e10_router.json)
- [results/m15_e11_mechanism.json](../results/m15_e11_mechanism.json)
- [results/m15_e11b_margin.json](../results/m15_e11b_margin.json)
- [results/m15_e12_failure_modes.json](../results/m15_e12_failure_modes.json)
- [results/m15_e12b_fragility.json](../results/m15_e12b_fragility.json)
- [results/m15_e13_robustness.json](../results/m15_e13_robustness.json)
- [results/m15_e14_screen_checks.json](../results/m15_e14_screen_checks.json)
- [results/m15_e15_decision_audit.json](../results/m15_e15_decision_audit.json)
- [results/m15_e16_quantization_pilot.json](../results/m15_e16_quantization_pilot.json)
- [results/m15_e17_quantization_replication.json](../results/m15_e17_quantization_replication.json)
- [results/m15_e18_query_precision.json](../results/m15_e18_query_precision.json)
- [results/m15_e1_latency.json](../results/m15_e1_latency.json)
- [results/m15_e2_ann_fiqa.json](../results/m15_e2_ann_fiqa.json)
- [results/m15_e2_ann_fiqa_run1.json](../results/m15_e2_ann_fiqa_run1.json)
- [results/m15_e2_ann_msmarco1m.json](../results/m15_e2_ann_msmarco1m.json)
- [results/m15_e4_prefix.json](../results/m15_e4_prefix.json)
- [results/m15_e5_oracle.json](../results/m15_e5_oracle.json)
- [results/m15_e6_router.json](../results/m15_e6_router.json)
- [results/m15_e6_thresholds.json](../results/m15_e6_thresholds.json)
- [results/m15_e8_frozen_lambdas.json](../results/m15_e8_frozen_lambdas.json)
- [results/m15_e8_towers.json](../results/m15_e8_towers.json)
- [results/m15_e8x_towers.json](../results/m15_e8x_towers.json)
- [results/m15_e9_blend.json](../results/m15_e9_blend.json)
- [results/m15_e9_frozen.json](../results/m15_e9_frozen.json)
- [results/m15_examples.json](../results/m15_examples.json)

## Exclusions and access rules

Untracked/ignored assets do not travel with the branch. See [PROVENANCE_AUDIT.md](PROVENANCE_AUDIT.md) for concrete unavailable inputs and archive status.

Never use this inventory as authorization to read protected files: `results/frozen_eval/untouched-*`, reserved qrels caches, `work/m9reserve`, raw closed M18 confirmation, and sealed M19 confirmation remain excluded. No repo-wide content searches over results/work. Use a named-file allowlist selected from the current evidence map. Do not overwrite existing result payloads or `results/perquery.json`.
