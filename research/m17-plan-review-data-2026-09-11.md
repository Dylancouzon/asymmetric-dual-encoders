# M17 planning draft: data and training review (2026-09-11)

Review only. The plan remains explicitly `DRAFT_NOT_EXECUTABLE`; no plan files or experiments
were changed. Protected six-set, reserved and LoTTE payloads were not read.

## Findings

### F1 — The 72-hour budget is a ceiling, but the proposed workload is not yet priced

The allocation spends 10 hours on four screen arms, 16 hours on two longer arms across two
seeds, then 10 hours on a newly built six-domain panel and 6 hours on export/parity/serving
costs ([`m17/registry.json`](../m17/registry.json), lines 29–80; [`m17/PLANNING.md`](../m17/PLANNING.md),
lines 210–231). This is coherent as a stop-point outline, but no real-path rate prices the
300,000-query cap, teacher-target preparation, 131,072-document bank mining, checkpoint I/O,
exact retrieval, or independently judged panel. The P0 artifact explicitly excludes those
stages and uses synthetic resident tensors ([`results/m17_planning_probe.json`](../results/m17_planning_probe.json),
`gpu.limitations`).

Concrete fix before execution: after the real smoke, replace the screen/final dose with measured
examples-per-second and one measured target/cache/mining pass at two sizes. If those timings do
not fit, reduce query dose or arm count before locking; do not multiply P0's 26k–50k synthetic
examples/s into a 72-hour forecast. This is already the plan's stated dependency, but it must be
an actual budget entry before an expensive run.

### F2 — The data plan supports technical vocabulary only conditionally

The plan correctly keeps existing admitted commercial-training sources as the default and treats
Kubernetes as optional pending approval ([`m17/PLANNING.md`](../m17/PLANNING.md), lines 100–120).
However, support minima of 20 distinct documents and 50 distinct contexts per new term, plus a
25% targeted slice within a 300,000-query cap, are not yet tied to observed counts
([`m17/registry.json`](../m17/registry.json), lines 15–27). P0/P0b are hand-authored fixtures and
explicitly do not estimate corpus coverage ([`m17/FINDINGS.md`](../m17/FINDINGS.md), lines 1–17).
Thus “broad domains” is a plan objective, not evidence that the admitted sources contain enough
S3/k8s, medicine, finance, and legal contexts.

Concrete fix: make the admitted-source support/provenance manifest the first data exit, and
record per-domain counts after source/document/query-family deduplication. Allow the proposed
extension to shrink to supported terms/domains when counts fail; do not fill slots from the
unapproved Kubernetes/AWS material by implication.

### F3 — The baseline wording is mostly sound, but the selected controls need an explicit v1
serving comparison

The frozen int8 v1 is correctly the product baseline, while C is a matched continuation control
and the four screen contrasts are exploratory ([`m17/PLANNING.md`](../m17/PLANNING.md), lines
142–172). The implementation must preserve the historical unfolded checkpoint, tokenizer,
teacher, and M13 scope; the CODEMAP calls out the folded/unfolded and legacy-driver hazards
([`m17/CODEMAP.md`](../m17/CODEMAP.md), lines 19–42). Before any screen, verify serialized v1
parity and retain the original v1 artifact as the rollback comparator. Report both dense and
fixed M12 DBSF@100 results; do not turn exploratory targets in the registry into release bars.

### F4 — Candidate-bank storage is well bounded, but target construction still needs a measurable
cache contract

The plan correctly stores candidate IDs and teacher scores rather than per-query vectors and
caps the bank at 131,072 documents ([`m17/PLANNING.md`](../m17/PLANNING.md), lines 122–140).
The registry's candidate mixture totals 64 candidates (1 known positive, 31 teacher neighbors,
16 v1 neighbors, 16 uniform), with deterministic dedup/fill ([`m17/registry.json`](../m17/registry.json),
lines 46–50). For unlabeled queries, the “known-positive slot” becomes another teacher hit;
this is stated but should be represented as a distinct candidate kind in the cache schema so
downstream code cannot mistake a teacher hit for a label. Require cache keys to include teacher
revision, query preprocessing, document-bank identity, candidate mix/K, and dtype, and measure
cache hit/miss and mining time in the smoke.

### F5 — The implementation surface is larger than the stated “small” update

The plan asks for a new tokenizer bundle, warm-start table, four losses/arms, candidate mining,
resume/checkpointing, panel construction and judgments, exact dense plus DBSF evaluation, numpy
and ONNX parity, latency/RSS, and two independent reviews. CODEMAP correctly warns that the old
M7 launcher invokes dev scoring and has incompatible assumptions, and says no M17 driver exists
([`m17/CODEMAP.md`](../m17/CODEMAP.md), lines 19–24). This is a substantial implementation
program, not merely a recipe tweak.

Concrete fix: keep the first implementation bounded to one new M17 driver, one cache schema,
the four declared screen arms, and reuse existing table/fusion primitives. Defer optional
bundle polish if the real-path smoke or data manifest slips, while preserving the DRAFT status
and recording incomplete outcomes. Do not broaden into a framework rewrite or M13 changes.

## Outcome

The draft is technically coherent and preserves historical/M13 rules, but it is not execution
ready until the support manifest, real end-to-end timings, cache identity contract, and panel
availability are established. The main risk is schedule overcommitment caused by treating the
synthetic P0 rate and an unbuilt six-domain panel as if they were priced resources. No quality or
coverage claim is supported yet.

Exact files read: `m17/PLANNING.md`, `m17/registry.json`, `m17/LEDGER.md`, `m17/STATUS.md`,
`m17/CODEMAP.md`, `m17/FINDINGS.md`, `results/m17_planning_probe.json`,
`results/m17_tokenizer_followup.json`, `instructions-m17.md`, `CLAUDE.md`,
`research/m17-data-training-2026-09-11.md`, and `m7/RECIPE.md`. No recursive searches were used;
no protected payloads were accessed.

## Final disposition after draft fixes

The follow-up review finds the major planning findings addressed: source support is the first
data exit; labeled and query-only candidate mixtures are separate with explicit `has_label` and
full cache identity; positive-bank membership is bounded; incomplete phase matrices cannot drive
selection; and implementation is bounded to one driver/cache schema. Document-encoder metadata
checks are now required without binding the generic encoder space to a production corpus or
adding a caller API.

These are planning fixes, not execution evidence. Real admitted inputs and panel, end-to-end
rates at two sizes, measured dose fit, owner protocol ratification, a working driver, and two
independent implementation reviews remain execution dependencies. The registry correctly stays
`DRAFT_NOT_EXECUTABLE`; no additional finding or plan edit is required from this review.
