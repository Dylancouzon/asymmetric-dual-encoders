# M9 — nano: a LEAF-style distilled query tower (mandate)

*Rewritten 2026-08-30 by the planning session on Dylan's direction ("goal is to beat LEAF
asymmetric… build the best model we can"; pair with zeo on the same teacher preferred). Evidence
and arithmetic: `m9/PLANNING.md`. Adversarial review of this plan: gpt-5.6-sol,
`research/m9-codex-plan-2026-08-30.md` — 9 BLOCKER / 12 MAJOR, all actioned; dispositions in
PLANNING.md §7.*

**SUPERSESSION (of the 2026-08-28 tiny mandate), explicit per review finding 1:** the teacher
screen is restored but small and margin-gated (was: mandatory, then informally waived); "no new
data / dev suite" is DROPPED (FineWeb enters, LoTTE-clean becomes the fresh surface, fever-train
LEAVES the pool); "embedding alignment + ranking preservation" becomes phase-1 L2 with
symptom-gated phase-2; the E14-head/LoRA pair-cost paragraphs are moot (M8 shipped no candidate —
the document side is exactly M7's frozen stella index). Everything else binds from
`instructions-m7.md` unchanged: decision authority, licensing, R1/R2/R3 decontamination, dev-only
selection, frozen-comparator pairing, freeze/ledger one-final-run protocol, Sonnet research
subagents, headless git contract — working files under `m9/`, four-file split, CLAUDE.md tightness
rule. Read `m8/CODEMAP.md` before writing code. Watch-long-runs checklist for anything >10 min.

## Goal, bars, and permitted claims

nano = a ≤35M transformer query encoder distilled into stella-400M's query-vector space, serving
against the SAME frozen 1024d stella document index as zeo. **The product is the pair: one
document index, two query paths.** Bars, paired on `results/perquery.json` (irreplaceable — never
overwrite):

| | avg-6 | NDO-4* |
|---|---|---|
| teacher ceiling (stella symmetric, M7 final run) | 0.5744 | 0.5640 |
| arctic-m-v1.5 (best row in the M4 matrix) | 0.5264 | 0.5348 |
| **AIM (heavily preferred): leaf-ir-asym** | **0.5155** | **0.5233** |
| **RELEASE BAR: bge-small symmetric** | **0.5042** | **0.5046** |

*NDO-4 = "no-disclosed-overlap-4" (SciFact, NFCorpus, SciDocs, TREC-COVID) — renamed from
"clean-4": disclosed data bounds KNOWN contamination only.*

- **Release bar** (gates the ship): resolved above bge-small on avg-6 under §Final-run rule.
- **Aim** (does not gate the ship): resolved above leaf-ir-asym on avg-6. The UNRESTRICTED
  headline additionally needs NDO-4 resolved; if NDO-4 is only point-positive, the headline
  carries the contamination restriction. A nano between the bars still ships as the pair's
  low-compute point, reported as "did not resolve above the LEAF system".
- **The headline is a SYSTEM claim, never a student claim** (finding 22): "the stella-document +
  nano-query system outperformed the arctic-document + leaf-query system on this harness." Never
  "nano beats LEAF" — our document tower is 400M/1024d vs their 109M/768d. Mandatory disclosures
  next to it: teacher sizes and dims, index bytes, doc-encode cost, retention vs own teacher
  (theirs 97.7%), per-dataset rows incl. the expected TREC-COVID loss (ceiling 0.8234 vs arctic
  0.8461), disclosed-overlap table.
- Feasibility (prior, NOT a forecast — finding 2): ceiling makes the aim possible at ≥89.7% avg-6
  / ≥92.8% NDO-4 retention; the 96–98.6% literature band assumed LEAF-scale dose. Achievable
  retention is estimated from OUR pilot curves only.

## Stage plan — no long run before its lock

- **M9.1 PILOTS + SCREENS** (next session): everything below marked (S). Ends with a written
  screen report in `m9/RESULTS.md`.
- **M9.2 RECIPE LOCK**: read S-results, lock recipe + dose + kill envelope + confirmatory
  registration into `m9/LEDGER.md`; **adversarial review (Codex + Fable) of the lock before any
  long run** — briefed to break it, with the read-exclusion, log audited after.
- **M9.3 BUILD**: main run under the kill rule; artifact export + parity; pre-freeze review.
- **M9.4 FINAL**: bridge check → one final run on the six → decision → (only if release bar
  passed) the reserved-four batch.

## Teacher: stella-400M default; screen restored, margin-gated

Pair economics + measured ceiling make stella the registered default. The screen (finding 5)
protects against "stella's query space is unusually hard for a small student":
- (S) Arms at identical registered text set / steps / token budget: {stella-400M × both student
  finalists} + {stella_en_1.5B_v5, Qwen3-Embedding-0.6B × the fixed better student}. Challenger
  target encodes cost ~3.5×/1.5× — include in the pilot budget.
- **Swap rule (registered before the screen runs):** leave stella-400M only if a challenger's
  asymmetric student beats the incumbent's on the selection surface with lower-CI > 0 AND point
  gain ≥ 0.010. A +0.001 CI-resolved gain does NOT break the pair. A swap = two document indexes;
  the lock records the cost explicitly.
- Challenger facts: stella-1.5B MTEB-6 0.584 vs incumbent 0.561 (MIT, same ArguAna/FiQA
  disclosure); Qwen3-0.6B 0.565 (Apache, zero six-overlap, FEVER disclosed). gte-Qwen2-1.5B dead:
  undisclosed training data (harrier precedent).

## Student: two finalists; the shipping artifact is frozen in M9

1. **all-MiniLM-L6-v2** (22.7M, Apache-2.0, community; LEAF's init; in fastembed). Risk: original
   fine-tune @seq128 — long-query bins are a registered screen metric.
2. **bge-small-en-v1.5** (33.4M, MIT, BAAI clean; retrieval-tuned @512; fastembed default; also
   the release-bar comparator).

Excluded, recorded (PLANNING.md §4): Ettin (ModernBERT ONNX friction), arctic-xs/-s (vendor:
strongest-justification AND LEAF's teacher vendor), granite-30m/e5-small/gte-small (justify-tier,
no edge). Serving at max_len 512 for BOTH finalists; report fertility, truncation rate, and
retrieval by query-length bin (finding 16).

**Artifact discipline (finding 14):** the confirmed thing is the EXPORTED artifact. (S) Export
both finalists' skeletons to ONNX opset 17 (real weights, no custom ops; parity ≥1−1e-4
min-cosine, ≤1e-3 max-abs). Register the full serving path at lock: tokenizer input policy
(literal bytes), max length, pooling, normalization, head, weight+output precision (fp16 ONNX
primary; int8 = descriptive cost row), runtime preprocessing. Final checkpoint exported and
parity-tested BEFORE the final run; **the final run scores the ONNX artifact**. M10 packages, it
does not change numerics.

## Recipe

- **Phase-1 objective: plain L2 regression on teacher vectors.** Specify at lock (finding 17):
  L2 vs MSE form, normalization location, reduction, epsilon, autocast; loss/normalization in
  fp32. Targets cached fp16 only after (S) validating angular + retrieval error vs fp32 on a 10K
  sample. MSE+cosine is DROPPED — affine-redundant under normalized outputs.
- **Prompt policy (S, finding 8):** two arms — (a) promptful student → prompted teacher target
  (LEAF-faithful), (b) raw query text → prompted teacher target (promptless serving). Freeze the
  winner's literal tokenizer input as part of the artifact.
- **Data mix (S, finding 7):** equal-token arms — 100/0 query-only (query-role targets) vs 70/30
  query/doc (doc texts with DOC-role, promptless targets, LEAF-faithful). No implicit swamping of
  ~460K queries by a 3M-doc pool.
- **Pool** (licence-clean; MS MARCO stays excluded): M7 TRAIN stack **minus fever-train** (FEVER
  is a reserved set — finding 9; drop costs ~98K of ~560K queries; TriviaQA/ESCI already
  in-stack), **LoTTE-clean is NOT training data** (it is the fresh surface — finding 6), FineWeb
  docs sampled with wikipedia-domain URL exclusion (DBpedia is Wikipedia abstracts) + R2-style
  near-dup screen vs the six corpora; R1/R3 reserved-set protections re-verified at build (R3
  already shows cqadup-untouched overlap = 1 doc of 854,921).
- **Dose in units, not wall-clock (finding 4):** register examples, non-pad tokens, optimizer
  steps, seq-length mix, checkpoint schedule. (S) throughput pilot (10–50K real texts) prices
  target-encoding and steps/hour BEFORE the lock. Initial main dose ≈ what the box affords in
  ~2–3 days; **registered extension rule**: while the dev retention curve's improvement over the
  last 25% of dose exceeds a lock-registered slope, extend in 50% increments (compute is cheap,
  dead ends are not — a rising curve is not a dead end).
- **Seeds (finding 19):** ship the preregistered seed; 2 screen-scale seed replicates of the
  winning config to report training variance (query-bootstrap CIs do not contain it).

## Selection surfaces and reads

- **Tuning dev** = M7's six pinned components (incl. heldout-longq as the long-query canary).
  It is 494 reads deep — it may trigger recipe changes, so it stays labelled DEV, never "held-out".
  Continue the reuse counter (adapt `m8src/dev_reuse_m8.py`).
- **Fresh surface** = LoTTE-clean, 7 slices, 20,122 q, macro over slices (never pooled): at most
  TWO reads — (1) screen winner confirmation at M9.2, (2) pre-freeze check at M9.3. Its
  forum-heaviness (same family as reserved CQA) is disclosed in the report. Fusion-weight
  selection for nano reads LoTTE read #1's grid, fixed grid incl. the dense endpoint (finding 15).
- Kill rule (findings 3, 18): log-spaced checkpoints; early reads are diagnostic. Kill/route only
  by the lock-registered envelope calibrated on screen curves, or a diagnosed defect. Phase-2
  routing by symptom: low vector error + disagreement concentrated at teacher top-k margins →
  ranking-preservation loss (Jasper-style margin/Gram terms are the candidates); broadly high
  vector error → more dose/coverage, not a new objective.

## Final run and the one confirmatory access

- **Bridge check first (finding 23):** re-encode bge-small on the six with the current stack;
  require per-query agreement with its frozen vectors within a lock-registered tolerance (M7
  conformance precedent |Δ| ≤ 3e-4); hash qids, qrels, preprocessing, revisions, dtype. Only then
  score nano.
- **Gates, on the six, paired on frozen vectors (finding 12):** exactly two confirmatory
  contrasts — C1 nano-dense > bge-small (release), C2 nano-dense > leaf-ir-asym (aim). Rule per
  contrast, both conjuncts required: one-sided stratified paired-bootstrap **98.75% lower bound
  > 0** (Bonferroni over the 2-family) AND dependent sign-flip Holm-rejected at family α = 0.025.
  Everything else is descriptive: nano-fused, nano-symmetric (LEAF's asym-vs-sym comparison),
  zeo+nano fusion (optional), NDO-4 status for the headline rule.
- **Reserved-four batch (spends the only access; run ONLY if C1 passed):** ONE batch scoring all
  pre-registered frontier systems (nano-dense/fused, zeo-int8/fused, bge-small, leaf pair
  fresh-encoded, BM25, stella-symmetric) so the M10 whitepaper gets an untouched frontier.
  Pre-registered reads: **primary = family-weighted NDO-3 macro (0.50·DBpedia +
  0.25·cqadup-android + 0.25·cqadup-english)** as a DIRECTION check for the aim headline (point >
  0 required to keep it; CI reported); dataset macro, pooled, per-dataset, leave-one-out all
  descriptive. **FEVER: labelled double-contaminated sensitivity row, zero alpha, never
  gate-relevant** (findings 10, 11). Registration text committed to `m9/LEDGER.md` BEFORE the
  first reserved byte is read.
- **Resource rehearsal before the batch (finding 13):** ~30M doc encodes across three towers,
  ~44 GB fp16 vectors — disk audit, full pipeline rehearsal on open sets, pinned model hashes,
  crash/restart semantics, intermediate scores suppressed until every system completes.

## Costs

Same three rows as M7 (query asset ≠ doc index ≠ hydration; doc index SHARED with zeo — say so).
Latency protocol (finding 20): ONNX Runtime batch-1, fixed thread count, tokenizer included,
length buckets, warm p50/p95, cold load, peak RSS, model bytes — this box's CPU named as the
proxy; same protocol run for MiniLM-L6, bge-small, and mdbr-leaf-ir for comparability.

## Deliverables

Frozen candidate + `m9/FREEZE.json` (assert_releasable), frontier table update, section in the M7
report artifact, decisions logged in CLAUDE.md. HF push is M10's, on Dylan's go. Adversarial
reviews: M9.2 lock and pre-freeze (standing grant; read-exclusion in every brief; log audited).

## Out of scope (reopening conditions in PLANNING.md §6)

E14-LORA / doc-side co-adaptation (breaks the pair; post-M10 with a real budget). MRL /
smaller-dim index (separate learned heads → separate system + re-encode; M10+ if ever — finding
21). Re-deriving zeo against a stronger teacher (T1: tower does not predict table). >35M student
(a third frontier point; M10+ scoping, Dylan's call). Teacher-layer-pruned or zeo-warm-started
inits. Any change to zeo — frozen.
