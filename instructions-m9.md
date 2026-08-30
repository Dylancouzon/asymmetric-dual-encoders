# M9 — nano: a LEAF-style distilled query tower (mandate)

*Rewritten 2026-08-30 by the planning session on Dylan's direction ("goal is to beat LEAF
asymmetric… build the best model we can"; pair with zero preferred — NB the M7 model is named
**zero**, "zeo" in older files was a typo). Evidence: `m9/PLANNING.md`. Adversarial reviews of
this plan: gpt-5.6-sol pass 1 (`research/m9-codex-plan-2026-08-30.md`, 23 findings) and pass 2 on
the committed text (`research/m9-codex-mandate-2026-08-30.md`, 19 findings) — all actioned;
dispositions in PLANNING.md §7.*

## What binds from M7 — exhaustive, not residual

Only these carry forward from `instructions-m7.md`: decision authority; vendor and source-licence
rules; frozen-comparator validation and pairing; decontamination requirements as re-specified
here; pushed-freeze, ledger, and crash/abort-disclosure mechanics; Sonnet-only research subagents;
the headless reporting-file convention translated to `m9/` (STATUS/RESULTS/EXPLORED/LEDGER);
CLAUDE.md's tightness rule. **M9 explicitly supersedes** M7's mission, teacher-selection text,
architecture, objectives A/B/C, Stage 0, go/no-go gate, tier definitions, confirmatory-contrast
family, untouched-final composition and sequencing, fusion rule, table ablations, ANN sweep,
branch authorization, and deliverables. Where M9 changes a carried-forward mechanism, M9 controls.
Read `m8/CODEMAP.md` before writing code. Watch-long-runs checklist for anything >10 min.

## Owner approvals on record (Dylan, 2026-08-30, this planning session)

1. **FineWeb approved** as a seed corpus for nano's regression text — doc-side texts only, ODC-By,
   wikipedia-domain URL exclusion, fingerprint screening per §Data. Recorded here; copy into
   `m9/LEDGER.md` at M9.0.
2. **Git: M9 execution work happens on branch `m9-work`**, frequent commit+push under the headless
   contract; merges to main at stage boundaries need Dylan's go. The M7 branch grant did not
   transfer; this planning session's commits to main predate this ruling and are disclosed.
3. **Query-asset size: quality first.** Target 70 MB fp16; exceeding it requires a logged,
   measured quality justification ("some more if proven useful is okay"). int8 reported as a
   descriptive row. This is the recorded cap the student-selection gate reads.

## Goal, bars, and the permitted claim

nano = a ≤35M transformer query encoder distilled into the selected teacher's query-vector space.
Default and preferred: teacher = stella-400M, serving against the SAME frozen 1024d index as zero —
one document index, two query paths. Bars, paired on `results/perquery.json` (irreplaceable —
never overwrite):

| | avg-6 | NDO-4* |
|---|---|---|
| teacher ceiling (stella symmetric, M7 final run) | 0.5744 | 0.5640 |
| arctic-m-v1.5 (best row in the M4 matrix) | 0.5264 | 0.5348 |
| **AIM (heavily preferred): leaf-ir-asym** | **0.5155** | **0.5233** |
| **RELEASE BAR: bge-small symmetric** | **0.5042** | **0.5046** |

*NDO-4 = no-disclosed-overlap-4 (SciFact, NFCorpus, SciDocs, TREC-COVID). Disclosed data bounds
KNOWN contamination only.*

- **Release** = C1 passes. **Aim** = C2 passes (does not gate the ship; a nano between the bars
  ships as the pair's low-compute point, reported as "did not resolve above the LEAF system").
- **If C2 passes, the only permitted performance headline is:** "the selected-teacher-document +
  nano-query asymmetric system outperformed the arctic-document + leaf-query system on the six
  development-informed datasets in our exact-search harness" — always carrying the
  disclosed-overlap qualification. Never "nano beats LEAF". **NDO-4 and reserved NDO-3 are
  descriptive robustness checks only** — they cannot grant, remove, or weaken the qualification;
  report each preregistered nano−leaf estimate with CI, and state prominently if the point is
  non-positive or the CI includes zero. Do not use "resolved/confirmed/unrestricted" for either.
- Mandatory disclosures beside the headline: teacher sizes/dims (400M/1024d vs 109M/768d), index
  bytes, doc-encode cost, retention vs each system's own teacher **on this same six-set harness**
  (LEAF: 97.9%; its 97.7% BEIR-14 figure only as a differently-surfaced literature number),
  per-dataset rows incl. the expected TREC-COVID loss (ceiling 0.8234 vs arctic 0.8461),
  disclosed-overlap table.
- Feasibility (a prior, NOT a forecast): the aim needs ≥89.7% avg-6 / ≥92.8% NDO-4 retention;
  the literature band 96–98.6% assumed LEAF-scale dose. Achievable retention comes from M9's own
  pilot curves.

## Stage plan — no evaluation before its lock

- **M9.0 SCREEN LOCK.** Before any target encoding, training, retrieval evaluation, or LoTTE
  access: commit and push `m9/LEDGER.md` entries fixing every M9.1 arm and its order, defaults,
  data snapshots + hashes, seeds, doses, optimizer settings, evaluation surfaces, macro weights,
  bootstrap method, all numeric decision rules and tie rules, outcome→action mappings, validation
  samples and bin edges (length/fragmentation bins, fp16-cache sample, parity sample, LoTTE
  confirmation threshold), R1/R2/R3 defined verbatim, and the owner approvals above. Screen
  results produced without this lock are diagnostic only and cannot select M9.2's recipe.
- **M9.1 PILOTS + SCREENS.** Bridge-tolerance dry run on dev · ONNX skeleton exports (both
  finalists, real weights, opset 17, parity min-cos ≥1−1e-4 / max-abs ≤1e-3 on every example of
  the locked sample) · fastembed `add_custom_model()` registration + serving parity for any
  student that can win · throughput pilot (10–50K real texts) pricing target-encode and steps/hour
  · the six sequential screen arms (§Teacher/student screen). Ends with a written screen report.
- **M9.2 RECIPE LOCK.** One pushed commit containing: selected teacher/student/prompt/mix; corpus
  snapshots, sampling seeds, licences, decontamination evidence; objective formula (L2 form,
  normalization location, reduction, epsilon, autocast; loss in fp32); optimizer + precision;
  dose in examples / non-pad tokens / steps / seq-length mix; checkpoint schedule; shipping seed;
  extension slope (dev-retention change per million non-pad tokens over named checkpoints),
  threshold, and **a fixed integer extension cap**; numeric kill envelope and phase-2 trigger
  (vector-error and top-k-margin-concentration thresholds, named surfaces) plus **one fully
  specified phase-2 loss** and its hyperparameters; complete serving path + the size ruling;
  parity gates; LoTTE batch manifests and outcome rules; fusion grid + tie rule; bridge statistic
  + tolerance; complete six-set and reserved-batch system manifests (every descriptive row marked
  INCLUDED or OMITTED — nano-symmetric only if its document path is trained/exported/frozen; no
  system added after any final output is revealed); model/code/data hashes; the statistics
  implementation (§Final run) with replicate counts, seeds, strata, weights, tails; the claim
  decision table; disk/RAM budget; crash/restart semantics. **No field may read TBD. Codex and
  Fable review the pushed lock before the main run.**
- **M9.3 BUILD.** Main run under the kill rule; registered extensions only; **"diagnosed defect"
  means demonstrated divergence from the locked implementation or infrastructure — never poor
  quality or an unfavourable curve** — and repairing one requires a pushed amendment, diff
  classification, repeat adversarial review, and the inherited access disclosure. Final checkpoint
  exported and parity-tested; artifact frozen; pre-freeze review.
- **M9.4 FINAL.** The single six-set transaction (§Final run) → decision → the reserved batch
  runs only via the already-registered conditional `if C1 then execute`.

## Teacher/student screen — six sequential, reusable arms (locked at M9.0)

Arms, in this order, identical texts / steps / non-pad-token budget / seed / optimizer:
1–3. The preregistered anchor student (default: bge-small-en-v1.5) under the baseline prompt/mix
against **stella-400M**, **stella_en_1.5B_v5**, **Qwen3-Embedding-0.6B**.
4. The other finalist (all-MiniLM-L6-v2) against the selected teacher.
5. The alternate prompt policy, selected teacher/student.
6. The alternate data mix, selected teacher/student/prompt.
Order and baselines may not change after arm 1 starts. Student, prompt, and mix picks use the
tuning-dev decision rules registered at M9.0. **Teacher swap rule:** a challenger replaces stella
only if its asymmetric system beats the stella-anchor arm by ≥0.010 point AND its one-sided 97.5%
stratified paired-bootstrap lower bound (B=10,000, fixed equal-component tuning-dev macro;
Bonferroni over the two challenger contrasts) exceeds zero. If a challenger wins, arm 4 runs
against it before student selection.

**Conditional teacher branch:** if no challenger fires, all stella-specific numbers here apply.
If one fires: nano means that teacher's query space and its separately frozen document index; the
"one index / two query paths" claim is forbidden; zero is unchanged; every ceiling, retention
denominator, symmetric row, disclosure, reserved system list, and storage/encode estimate is
recomputed before M9.2. No stella-specific number carries into that branch.

Student facts: MiniLM-L6 22.7M Apache (LEAF's init; orig fine-tune @128 → long-query risk),
bge-small 33.4M MIT (retrieval-tuned @512; the release-bar comparator). Both serve at 512.
Report tokenizer fertility, [UNK] rate, truncation rate, retained-token fraction, and retrieval by
the locked length/fragmentation bins; first-512 truncation is stated as a limitation unless the
locked head+tail probe ran. Excluded with reasons in PLANNING.md §4: Ettin, arctic-xs/s,
granite-30m, e5-small, gte-small, mdbr-leaf-ir (vendor).

## Recipe

- **Phase 1: plain L2 regression on teacher vectors** (LEAF, arXiv 2509.12539 — its ablation
  rejected auxiliary losses). Phase 2 (the single locked loss, Jasper-family margin/Gram terms are
  the candidates to pick from AT LOCK) triggers only on its locked symptom: low vector error with
  disagreement concentrated at teacher top-k margins. Broadly high vector error → dose/coverage.
  MSE+cosine is dropped: affine-redundant under normalized outputs.
- **Role byte-templates (locked at M9.0):** register literal templates for QUERY and DOC. Prompt
  arm (a): student and teacher both get the stella s2p query template. Arm (b): student gets raw
  query bytes, teacher gets the s2p template. In the 70/30 mix arm, teacher DOC targets are
  encoded from raw document bytes and student DOC inputs carry one fixed explicit document-role
  marker. The same raw student input never maps to two different teacher targets.
- **fp16 target cache** accepted only if, vs live fp32 on the locked 10K stratified sample:
  min-cos ≥0.9999, max-abs coordinate error ≤1e-3, query retrieval macro shift ≤1e-4; else fp32.
- **Seeds:** ship the preregistered seed regardless of replicas. After the final dose is fixed,
  train two additional preregistered seeds at the identical full recipe and dose — reporting-only
  (six-free DEV/LoTTE variability, reported separately from query-bootstrap CIs). Dylan may waive
  by logged ruling if the final dose makes replicas multi-day.

## Data

Pool: M7 TRAIN stack **minus fever-train** (FEVER is reserved and stella-disclosed; drop ≈98K of
~560K queries; TriviaQA/ESCI/nqopen already in-stack) + FineWeb docs per the owner approval.
LoTTE-clean is NOT training data. MS MARCO stays excluded.
**Reserved-set decontamination without reading reserved bytes:** before the reserved batch,
decontamination may use only pre-existing, non-reversible reserved-set fingerprint artifacts whose
paths, hashes, construction dates, algorithms, and thresholds are recorded at M9.0 (M7's R3
machinery produced such artifacts — verify and pin them); it may not open reserved corpora,
queries, qrels, or caches. FineWeb must pass exact + near-document checks against DBpedia and both
reserved CQA slices through those fingerprints, in addition to the wikipedia-domain exclusion and
the R2-style screen vs the six. If suitable pre-existing fingerprints do not exist, FineWeb is
excluded — creating fingerprints now would read reserved bytes and is forbidden.

## Selection surfaces

- **Tuning dev** = M7's six pinned components (heldout-longq = the long-query canary). 494 reads
  deep, stays labelled DEV; continue the reuse counter.
- **LoTTE-clean** (7 slices, 20,122 q, macro over slices, never pooled) is fresh only until its
  first atomic batch. Before that batch: commit the complete model list, checkpoint hashes, fixed
  fusion grid (including the dense endpoint), metrics, confirmation threshold with both outcomes'
  actions, and output-suppression code. One read = one execution of that committed batch; nothing
  added after any output is revealed. **Read #1** may select the fusion weight and apply the
  locked screen-confirmation rule; afterwards LoTTE is DEV and is never again called fresh.
  **Read #2** is audit-only pre-freeze: it may stop M9 or remove a claim, never change recipe,
  checkpoint, teacher, student, prompt, mix, grid, or weight. No third read. Forum-heaviness
  (same family as reserved CQA) is disclosed.

## Final run — one six-set transaction, then the conditional reserved batch

- **The bridge is phase 1 of the sole six-set transaction, not a separate access.** The final
  scorer validates frozen qids/comparators, scores the bge-small anchor, requires zero
  missing/extra/reordered qids and max per-query |Δ nDCG| ≤ 3e-4 vs frozen, verifies hashes
  (qrels, preprocessing, model revisions, dtype, exact-search code, tie-breaking) — and only on
  success proceeds, in the same process, to score the manifest. Bridge failure consumes the
  attempted access under the inherited abort/disclosure rule.
- **Gates:** C1 nano-dense > bge-small; C2 nano-dense > leaf-ir-asym. Statistics, per contrast:
  align identical qids per dataset; per-query nDCG@10 differences; B=10,000 bootstrap replicates
  resampling n_d queries with replacement within each dataset, dataset means averaged at weight
  1/6; the empirical 0.0125-quantile lower bound must exceed zero (one fixed logged seed, shared
  resample indices across C1/C2). Separately B=100,000 one-sided dependent sign-flip replicates on
  the same equal-weight statistic, independent Rademacher signs per paired query shared across
  C1/C2, p=(1+#(T*≥Tobs))/(B+1), Holm step-down over the two p-values at family α=0.025. A
  contrast passes only if BOTH the bootstrap bound and Holm reject; the sign-flip is a required
  sensitivity conjunct, not evidence its weak-null assumptions hold.
- **Reserved batch:** manifest, estimands, comparator directions, confidence procedure, model
  revisions, and crash semantics live in the SAME pushed freeze commit that precedes the six-set
  transaction; after any six-set output is revealed only the registered conditional may run,
  unmodified. Reserved reads: family-weighted NDO-3 macro (0.50·DBpedia + 0.25·cqadup-android +
  0.25·cqadup-english) and the other descriptive cuts (dataset macro, pooled, per-dataset,
  leave-one-out); **FEVER is a labelled double-contaminated sensitivity row, zero alpha, never
  gate-relevant.** Resource rehearsal first: ~30M doc encodes across three towers, ~44 GB fp16
  vectors — disk audit, full-pipeline rehearsal on open sets, pinned hashes, restart semantics,
  intermediate scores suppressed until every system completes.

## Costs

Three M7 rows (query asset ≠ doc index ≠ hydration; index shared with zero — say so), PLUS:
document-encode docs/s, dim, raw-vector bytes, built-index bytes, exact-search latency, separately
from query encoding. Latency protocol: ONNX Runtime batch-1, fixed threads, tokenizer included,
length buckets, warm p50/p95, cold load, peak RSS, model bytes; this box's CPU named as the proxy;
same protocol for MiniLM-L6, bge-small, and mdbr-leaf-ir.

## Deliverables

Frozen candidate + `m9/FREEZE.json` (assert_releasable), frontier table update, section in the M7
report artifact, decisions logged in CLAUDE.md. HF push is M10's, on Dylan's go. Every review
brief carries the reserved read-exclusion; the log is grepped afterwards.

## Out of scope (reopening conditions in PLANNING.md §6)

E14-LORA / doc-side co-adaptation (breaks the pair; post-M10 with a real budget). MRL /
smaller-dim index (separate learned heads → separate system; M10+ if ever). Re-deriving zero
against a stronger teacher (T1: tower does not predict table). >35M student (M10+ scoping,
Dylan's call). Teacher-layer-pruned or zero-warm-started inits. Any change to zero — frozen.
