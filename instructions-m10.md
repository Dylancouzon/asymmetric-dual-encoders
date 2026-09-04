# M10 — nano v2: the retry, built around coverage (mandate)

*Written 2026-09-01 by the planning session on Dylan's direction ("M9 failed to achieve our goals
… make M10 a retry … build something unimpeachable by competitors"). Evidence `m10/PLANNING.md`;
M9's record `m9/FINDINGS.md`. Adversarial review: gpt-5.6-terra, read-only, six passes
(`research/m10-codex-plan-2026-09-01.md`, `-plan2-` … `-plan6-`; full logs are gitignored `.log`
files beside them); every finding and its disposition is in PLANNING §8, including two reviewer
dissents the owner can overrule. **Amended 2026-09-01 after pass 6 on Dylan's compute ruling:** M10
runs on a rented GPU budget or not at all (§Owner rulings, §Compute); the review amendments taken
with it are decision 9 and PLANNING §8's amendment block. **Amended 2026-09-04 after a full plan
review on Dylan's direction ("is this the best way to go about it? Is it the most efficient?
… I'm not sure why we need to generate synthetic data?"): see §Amendment 2026-09-04, which is
authoritative wherever an older sentence in this file disagrees.** The M9 model is **nano**; M7's
table is **zero**; the product is still the pair on one stella index.*

## What binds from M7 and M9 — exhaustive

From `instructions-m7.md`: decision authority; vendor and source-licence rules (as relaxed in
CLAUDE.md); frozen-comparator validation and pairing; pushed-freeze, ledger, crash/abort
disclosure; Sonnet-only research subagents; the headless reporting files under `m10/`
(STATUS/RESULTS/EXPLORED/LEDGER); the eval-use licence standard (source-level evidence before a
selection set is used); CLAUDE.md's tightness rule. From `instructions-m9.md`: the bars, the
permitted claim and its qualification, the six-set transaction design (`m9/FINAL_LOCK.md`,
re-registered for M10 with new hashes and the one change in §Final run), the LoTTE atomic-read
protocol, R1/R2/R3, the stage discipline (no evaluation before its lock; "diagnosed defect" =
implementation divergence only), the 70 MB fp16 query-asset target, screen-in-closed-form first,
the watch-long-runs checklist. **M10 supersedes** M9's data pool, mix, optimizer, schedule, plateau
rule, phase-2 handling, screen arm list, and selection surfaces. Where M10 changes a carried
mechanism, M10 controls. Read `m8/CODEMAP.md` and `m9/CODEMAP.md` before writing code.

## Why M9 missed, and what M10 changes

M9 retained **93.8%** of the teacher on NQ and **50–71%** on the two CQADupStack dev components
(`m9/FINDINGS.md` §1): its 463K Wikipedia-QA and product queries did not cover the forms the model
is tested on, and its 384-wide linear head is the subspace L2 regression cannot push past once
queries are diverse (PLANNING §9). Parameter count is not the constraint. Both causes are tested
before anything is built (§Screen, families A and G). M10 changes, in order of evidence (PLANNING §3):

1. **Coverage** — ~4.0M query texts across 12 forms, most of them REAL text (§Data).
2. **Optimizer regime** — LEAF's batch 32 and three linear 1e-4→1e-5 cycles; plateau read on
   annealed checkpoints, best-to-best.
3. **A wider linear output** — three mean-pooled layers concatenated (1152-d) before the head.
4. **Warm start from the M9 candidate** as a screen arm.
5. **LEAF's own loss** — the L2 norm ‖e‖₂, not M9's squared L2 — as the one objective arm.

**The student cap is 35M and it is hard** (Dylan, 2026-09-01: "109M is not an option. This isn't
low compute anymore. 33M was already in the upper bound of what I think is acceptable"). bge-small
with the three-layer head is 34.5M; MiniLM-L6 with it is 23.9M. The cheaper student gets the
benefit of a tie (§Screen, family F).

## Owner rulings already made (Dylan, 2026-09-01)

- **Git:** M9 was merged to `main` on 2026-09-01 after the repo cleanup (its six-set close-out is
  still pending and runs from `m9-work`); M10 execution work happens on branch **`m10-work`**
  under the headless commit-and-push contract; merges to main at stage boundaries
  need Dylan's go. M9's registered six-set close-out still runs from `m9-work`, because `guard9`
  pins that branch (`m9src/guard9.py:35`); the branch is kept until then.
- **Compute (Dylan, 2026-09-01): "M10 won't be done on a 3080. M10 will be done on a GPU budget, if
  allowed, or not at all."** Every M10 stage from M10.0-c on runs on a rented GPU under the budget in
  §Compute, or M10 does not run. The RTX 3080 is not an execution target for any M10 step; it only
  supplies M9's frozen checkpoint once (§Compute). Every dose and duration here is set from the
  budget, not from the box.

## Amendment 2026-09-04 — plan review (authoritative over any older sentence here)

Dylan asked whether the plan was the best and most efficient route, questioned the synthetic data,
and said *"don't over-engineer"* and *"keeping the same teacher (Stella) is the goal"*. Nothing
below touches stella, the 35M cap, the frozen tower, the pair, or any observed number — no M10
number exists. Evidence: PLANNING §11 (measured rates), §12 (the synthetic-data question),
`m10/LEDGER.md` §3. Full disposition of what was cut: `m10/EXPLORED.md`.

| # | change | why |
|---|---|---|
| A1 | **Family D cut from three ranking-aware arms to one arm: LEAF's L2-norm loss `L=‖e‖₂`.** Deletes the 1M-document candidate bank, the exact-mining pass and its HNSW fallback, the τ entropy rule, the 129-way D-NCE spec and the seed-rank provenance field | LEAF's own Appendix B added distillation terms to plain regression and got **no improvement**; LEAF (97.7% asym), EmbedDistill, arXiv 2306.11550, mxbai-edge-colbert and DistilVDR all use pure embedding regression. The class is the least-supported and most machinery-heavy part of the plan. LEAF's loss is the **norm**, not the square — a one-line arm we had silently diverged from |
| A2 | **Generation cut from 3.0M to ≈1.0M, and only for the forms no corpus contains.** The harvestable forms come from **real** text mined out of the licensed pool (titles, headings, lead claim sentences, extracted interrogatives) — new arm A3 | Three independent saturation curves put diminishing returns under ~1–1.5M (DistilVDR saturates above 75% of a 1.49M pool; SPEED log-linear to ~920K; doc2query 50–75% coverage ≈ 90–95% of max gain). And **three of the four clean-4 headline datasets have a real-text counterpart** (scidocs↔titles, scifact↔claim sentences, trec-covid/nfcorpus↔headings), so the headline can rest on real text plus the teacher instead of on the generator's prior |
| A3 | **C1/C2 registered on clean-4 as well as avg-6** (§Goal, §Final run). clean-4 bars: bge-small **0.5046**, leaf-ir-asym **0.5233** | M14 registered clean-4 as the headline partition for both zero and nano on 2026-09-04 (`instructions-m14.md`), while M10 had C1/C2 on avg-6 with NDO-4 *descriptive*. Left alone, the paper's headline would carry no pre-registered pass/fail for nano. Fixed now, before any M10 number exists |
| A4 | **The COV resolution number is measured first and SIZES the screen** (arms, contrast count, MDE), instead of being measured after the lock and deciding nothing | The registered MDE 0.0056 sits **below the surface's own resolution**: at a one-sided 0.025/16 bound (z≈2.96) on a family-weighted macro whose paired SE is ~0.003 (BRIGHT ~100 queries/slice, CorporateLobbying 340), a contrast needs ≈0.008–0.009 to resolve, so a contrast landing at the MDE can never resolve. Codex pass 7's objection was to comparators drawn from family F; a direction-free power quantity on non-candidates (e5-small-v2, gte-small) is a power calculation, not a selection |
| A5 | **Decision 8 (second build seed, ≈100 GPU-hours) withdrawn; confirmations capped at two decisions** | Seed 1 is descriptive by construction and can trigger no action; the same hours buy ~3 extension cycles that can move the number. A screen-dose seed pair gives a replication band for ~5% of the cost |
| A6 | **Family F runs SECOND (right after A), and the remaining families run on its winner** | It was seventh, which made every other verdict transfer to the build student by assumption. Reordering costs nothing and removes the assumption. Order is now A → F → G → B → E → C → D |
| A7 | **The box is an execution target again for everything that is not generation.** The 2026-09-01 ruling stands where it bites — the *dose* is not set by the box — but the box is where the screens and (optionally) the build run, and it holds ~200 GB of M9 caches the plan had budgeted 12 GPU-hours and a day of network to re-derive | Measured on the box 2026-09-04 (PLANNING §11): the M10 recipe shape runs at **400 examples/s** in M9's two-chunk collate and **745** in one padded chunk, against the plan's imported LEAF planning rate of 560. The 3080 meets or beats the assumed A100 rate because at batch 32 with 35-token queries the job is launch-bound, not FLOP-bound. Generation still needs the cloud: Qwen3-8B bf16 is 16 GB on a 10 GB card |
| A8 | **Two pre-training data quality gates added** (§Data): near-duplicate and dispersion metrics per form, and a **distribution-overlap check against real MS MARCO dev queries in stella's own space** | The synthetic risk here is not wrong labels — the teacher's embedding of any text is a correct target by construction — it is distribution shift and diversity collapse, both measurable before a training step. MS MARCO is permitted for validation by Dylan's 2026-09-04 rule. FORMS-12 cannot serve this purpose: it scores student-teacher agreement on the same synthetic queries and is circular for it |

**Withdrawn from the review's own recommendations, and why** (kept so it is not re-proposed):
dropping family F to anchor on MiniLM-L6 was recommended and then **withdrawn** — it would have
killed family C too (M9's candidate is bge-small, so there is no MiniLM warm start), and arXiv
2306.11550's depth curve (1/2/4 layers → 86.1/92.5/96.2% retention) disagrees with LEAF's success
on 6 layers. One 5M arm ≈ 2 GPU-hours is not the place to economise when it picks the build student.
F stays, C stays, bge-small stays the anchor; A6 fixes the ordering problem instead.

## Owner decisions (defaults apply until Dylan rules; each is recorded in `m10/LEDGER.md`)

| # | decision | default while open |
|---|---|---|
| 1 | Ratify M9's final-lock amendment **together with the close-out amendment that strikes M9's reserved conditional** (§Stage plan, M10.2): M9's close-out is six-only and cannot spend the reserved access | blocks the close-out only |
| 2 | **GPU budget — VALIDATED by Dylan 2026-09-04.** Re-priced at the measured rates after the plan review: hybrid (screens on the box, cloud for generation and the build) **≈ 56–101 cloud GPU-hours ≈ $85–250**; all-cloud ≈ 130–190 GPU-hours ≈ $200–475. Ceiling **$1,000** unchanged and unspent. PLANNING §6 (re-derived) and §11 | approved; the day-one benchmark still re-derives §6 before anything scales |
| 3 | FineWeb as a seed — **ruled out 2026-09-01** (delegated): Wikipedia and the approved corpora carry the topics; FineWeb adds a rights review and a blocklist for no measured gain. Reopening condition in `m10/EXPLORED.md` | closed |
| 4 | PAQ (machine-generated questions over Wikipedia; data CC BY-SA, generation code CC BY-NC) as query text | include, from Facebook's official release (never the unofficial HF mirror); 1.0M uniform sample in the build (seed 0, file hashes pinned), 4.037M in the volume-control screen arm A2 only; attribution recorded |
| 5 | FineWeb documents — **excluded 2026-09-01**: no reserved-set document fingerprints exist and creating them would open reserved corpora (`m9/LEDGER.md` §1.3) | closed |
| 6 | A >35M tier — **ruled out by Dylan 2026-09-01**; 35M is a hard cap | closed |
| 7 | Confirm: LoTTE read #1 withdrawn unexecuted in M9; renumbering M10/M11/M12 | as recorded in `m9/STATUS.md` and CLAUDE.md |
| 8 | Second build seed at full dose — **WITHDRAWN 2026-09-04** (amendment A5): descriptive by construction, could trigger no action, and cost ≈100 GPU-hours. Replaced by a screen-dose seed pair on the selected recipe as the replication band; the freed hours go to extension cycles | closed |
| 9 | The 2026-09-01 review amendments taken with the compute ruling: dose 200M examples, screen dose 5M, G-1536, the COV resolution number, extension capped by budget (PLANNING §8, amendment block and pass 7). **D-KL1, D-NCE and the seed-rank field are struck by amendment A1; the resolution number is repurposed by A4** | adopted as amended |
| 10 | The **2026-09-04 plan-review amendments A1–A8** (§Amendment 2026-09-04) | adopted on Dylan's "make your changes to the plan"; any item he strikes reverts to the 2026-09-01 text |

## Goal, bars, and the permitted claim — unchanged from M9

nano = a ≤35M transformer query encoder serving stella-400M's frozen 1024d index (the SAME index
as zero). Bars, paired on `results/perquery.json` (sha `6b18e3dd…`, irreplaceable):

| | avg-6 | NDO-4 |
|---|---|---|
| teacher ceiling (stella symmetric) | 0.5744 | 0.5640 |
| **AIM: leaf-ir-asym** | **0.5155** (89.7% retention) | 0.5233 |
| **RELEASE BAR: bge-small** | **0.5042** (87.8%) | 0.5046 |

**NDO-4 is clean-4** (`nfcorpus`, `scidocs`, `scifact`, `trec-covid`), the partition M14 registered
on 2026-09-04 as the headline for both halves of the pair.

C1 nano-dense > bge-small (release); C2 nano-dense > leaf-ir-asym (aim). Both dense-vs-dense;
fused rows descriptive only. **Amendment A3, 2026-09-04: C1 and C2 are each registered on BOTH
partitions** — `C1a`/`C2a` on avg-6 (bars 0.5042 / 0.5155) and `C1b`/`C2b` on clean-4 (bars
**0.5046 / 0.5233**, recomputed from the frozen comparator rows of `results/perquery.json`, a
comparator-only read taken 2026-09-04 before any nano number existed). The four are one family:
Holm across the four conjuncts inside the existing 0.025 family alpha, sequence fixed in the M10.2
lock. clean-4's release bar is 0.0004 above avg-6's and its aim bar 0.0078 above, so the aim is
strictly harder on the headline partition — registered knowing that. **C2 is a whole-system comparison** (stella documents + nano queries
vs arctic documents + LEAF queries): different document towers, index sizes, encode costs and
disclosed teacher overlap. It supports exactly M9's verbatim headline sentence and **no statement
about nano versus LEAF's query tower**; the report carries both systems' retention against their
own teachers, index bytes, document-encode cost and query latency beside the number. NDO-4 and
reserved NDO-3 stay descriptive. Additional mandatory disclosures: per-dataset retention, the
synthetic-data provenance table (generator revision and terms, prompts, seed sources and ids,
counts per form, removal counts per screen), the selection-surface table (licence evidence,
revisions, sizes, removals from training data), dose in examples / tokens / GPU-hours beside
LEAF's ~100 A100-hours, and the dev-reuse count.

## Stage plan

- **M10.0 DIAGNOSIS + SCREEN LOCK** (no six-set, reserved or LoTTE output exists during this stage).
  (a) Mac diagnostics — **done 2026-09-01**, `m10/RESULTS.md`, PLANNING §9–9c: the rank probe
  (a 384-d subspace keeps 99.5% of one query distribution and 90–93% of three; 98–100% at 640 —
  evidence about the class under L2 regression, not a bound), the head-width probe (a frozen
  bge-small retains more with each pooled layer), and the serving-parity check (fastembed 0.8.0
  reproduces the three-layer per-token head to 2e-7; 34.5M parameters). **Action taken:** the head
  stays linear; the pooled feature widens to three layers (§Recipe); family G decides. Repeat the
  parity check on MiniLM's head before family F may select it, on the four-layer head before G's
  1536 arm runs, and on the trained artifact with M9's locked parity sample before the freeze.
  (b) Capacity probe (`m9src/capacity_probe.py`, unchanged) — **optional, report-only**: no outcome
  changes an M10 action under the hard cap; runs only on an idle GPU; its 109M student is
  768-hidden, so any gain is partly width.
  (c) Per-component DEV-6 read of the M9 candidate incl. `heldout-longq` (the baseline row).
  **Runs on the box** (amendment A7): it holds the checkpoint and the encode caches.
  (d) **COV admission** (§Surfaces): for every candidate component, record in `m10/LEDGER.md` §2
  its primary-source licence URL and terms, HF repo and revision, corpus size, query count, qrels
  format and metric, its corpus-level contamination check and its fingerprint screen against the
  six and the reserved four. A component is named COV only after that record is pushed. **COV
  admits only surfaces no M10 decision has read**: the two CQADupStack dev components were scored
  by the Mac diagnostics (PLANNING §9, 86 raw reads) and stay in DEV-6. Then every admitted COV
  corpus, query set and document set joins the protected index (`m8src/protected_filter`) before
  any seed is drawn or any harvested, PAQ or synthetic text is constructed. **Re-run admission under
  Dylan's 2026-09-04 licence rule before judging the family floor: ConsumerContractsQA (CC BY-NC) is
  re-admissible, giving four families without LEDGER.** The **COV resolution number** (§Surfaces) is
  measured on the admitted surface and pushed before (e), and by amendment A4 it **sizes the screen**.
  (e) **Screen lock**: `m10/LEDGER.md` §0 (skeleton committed 2026-09-01) fixes every arm of
  §Screen (thirteen arms), order, doses, seeds, surfaces, the eleven contrasts, the MDE and
  multiplicity control **as sized by the resolution number**, confirmation design and
  outcome→action maps.
- **M10.1 DATA.** Harvesting of the real query-like text (§Data), generation for the non-harvestable
  forms under the §Data contract (200-query smoke per form first), PAQ samples, decontamination
  against the protected index (now including COV) and the six's documents, the FORMS-12 hold-out,
  teacher targets, `results/m10_data_manifest.json` with hashes and the provenance table.
  **Before any arm and recorded in the manifest**, the two A8 quality gates: per-form
  near-duplicate and dispersion metrics, and the stella-space distribution-overlap check against
  real MS MARCO dev queries (validation only).
- **M10.2 SCREEN + RECIPE LOCK.** The arms of §Screen, the confirmation runs, then one pushed lock
  commit with every field of M9's M9.2 list filled — including the objective, the best-to-best
  plateau/extension rule on annealed checkpoints, the GPU-hour allocation under the ceiling (build,
  extensions), the final-run registry with the four C-conjuncts and their Holm sequence, and
  LoTTE read #1's manifest. Codex and Fable review the pushed lock. **Then, and only then,** M9's close-out runs:
  its registered six-set transaction **amended before execution to six-only** (the `if C1 then
  execute` reserved conditional is struck; disclosed and ratified under decision 1), so M9 cannot
  spend the reserved access. Its rows are a forecasting calibration and a whitepaper frontier
  point; nothing in M10 may change on them. **Wherever those rows are published — frontier table,
  whitepaper, model card — they carry M9's build-provenance disclosure with them:** the M9
  candidate's provenance spans TWO build lock states (mid-build infrastructure repair, session
  force-reopened; `m9/LEDGER.md` "M9.3 BUILD PROVENANCE DISCLOSURE" and `m9/BUILD_LOG.md`), and it
  may never be described as an uninterrupted single-lock run. LoTTE read #1 then runs as the
  registered veto on the selected recipe.
- **M10.3 BUILD.** One candidate under the kill and extension rules; export + parity; freeze;
  pre-freeze review; LoTTE read #2 (audit only). The screen-dose seed pair (amendment A5) is the
  replication band; no full-dose replica runs.
- **M10.4 FINAL.** The six-set transaction → decision → the registered reserved conditional.

## Data

**Query corpus (~4.0M unique texts, query role) — amendment A2: most of the form breadth is REAL text.**

| source | build count | licence | role |
|---|---|---|---|
| M9 real queries (hotpotqa, squad, esci, mrtydi, nqopen, triviaqa; fever out) | 463K | CC BY-SA / Apache | real forms |
| PAQ sample (decision 4) | 1.0M | CC BY-SA (data) | factoid volume — capped so it cannot dominate |
| **Harvested real query-like text** from the licensed document pool (§Harvest) | ≈1.5M | inherits the pool's licences; no new rights surface | the harvestable forms, with **no generator prior** |
| **Synthetic, Qwen/Qwen3-8B (Apache-2.0; revision `b968826d…`), bf16 via vLLM on the rented GPU; registered fallback: hosted open-weights inference of the same revision if the smoke's end-to-end projection exceeds 20 GPU-hours** | ≈1.0M | generated under the generator's terms; provenance pinned; **not redistributed** without review | ONLY the non-harvestable forms |

**Form taxonomy — 12 forms**, quotas locked at M10.1 (±10% realized), and each form is assigned to
exactly one of the two sources. **Harvested (real text, ~250K each):** paper-title query ·
scientific claim (a statement) · 2–4-word keyword query · factoid question · consumer-health
question · product-search query. **Generated (~165K each):** how-to / troubleshooting question with
title and body · long counter-argument paragraph (120–220 words) · finance / personal-economics
question · comparison question · yes/no verification question · conversational multi-sentence
request. **The assignment is not arbitrary: three of the four clean-4 headline datasets fall in the
harvested half** (scidocs↔titles, scifact↔claim sentences, trec-covid and nfcorpus↔headings and
consumer-health), so the headline partition rests on real text plus the teacher rather than on the
generator's prior. The generated half covers the interactive forms no corpus contains, which map to
the non-headline sets and to COV's forum-style families.

**§Harvest — the real-text pipeline (M10.1, deterministic, no model in the loop).** From the 6.15M
pool documents and the Wikipedia seed corpus, all under licences already approved for training:
titles and headings as-is (title and keyword forms); declarative lead sentences of abstract-like
passages, filtered to 8–40 words with a finite verb and no first person (claim form); sentences
ending in `?` extracted with their preceding sentence as optional body (factoid, consumer-health,
product forms, routed by the source corpus). Every harvested string goes through the same screens,
quotas and hold-out as a generated one, and the manifest records the extraction rule, source
document id and per-rule yield. **Seeds for the generated half:** Wikipedia stratified by
top-level category (CC BY-SA) and the approved pool corpora. FineWeb is out of M10 (decisions 3
and 5). MS MARCO may never seed either half. **Contaminating source families — never seeds, never regression text, and
excluded from COV** (`research/m7-data-licensing.md` map): S2ORC / Semantic Scholar; PubMed;
NutritionFacts.org and its mirrors; CORD-19; StackExchange personal finance (money.SE) and Reddit
finance; args.me / idebate; every six-set and reserved corpus.

**Generation contract (M10.1):** generator pinned by HF repo + revision in the manifest (bf16
weights; the hosted provider and its served revision recorded if the fallback fires); vLLM sampling temperature
0.8, top-p 0.95, `max_new_tokens` per form (60, or 400 for the argument and conversational
forms), `seed = blake2b-64(seed_passage_id)`; the reply must parse as one JSON list of exactly n
strings (`m10src/forms.parse`, strict — no preamble); one retry on a contract failure, then the
seed is dropped; exact-duplicate queries removed. **Smoke:** 200 queries per form, read by Dylan,
who is the approver; a form passes when ≥ 90% of replies meet the contract and ≥ 80% of a
50-query sample are judged on-form; a failing form's prompt may be revised at most twice, each
revision recorded in `m10/LEDGER.md` §1 before the next smoke. **Seeds are pre-filtered:** a seed
passage that exact- or near-matches the protected index is never used.

**Screens on every generated or PAQ query, thresholds fixed here (M7's fingerprints,
`m7src/decontam.py`):** exact `blake2b-64` match or word-8-gram bottom-32 sketch ≥ 8/32 against
(i) the protected index (six + dev + reserved + LoTTE **+ admitted COV queries and documents**),
(ii) the six's documents; (iii) any word-5-gram shared with the query's own seed passage (a
copied span is not a query) — **for harvested text this screen is against the source document
minus the harvested span itself, since a harvested title IS a span of its document**;
word-4-gram containment for 4–7-word queries. The M9 real-query pool
and the document pool are re-screened against the COV additions (R1 removes matching queries;
matching pool documents are
removed too). Removal counts per screen, per form and per COV component are recorded **before any
COV component is scored**. **FORMS-12 hold-out:** 500 seed documents per form are set aside first;
queries generated or harvested from them are never trained on.

**Quality gates (amendment A8), executed on the immutable manifest before any arm, recorded in
`m10/LEDGER.md` §1; each is a REPORTED number with one registered action.** The risk this addresses
is not label noise — there are no labels, and the teacher's embedding of any text is a correct
target by construction — but distribution shift and diversity collapse.
1. **Diversity, per form:** near-duplicate rate under the existing word-8-gram bottom-32 sketch at
   ≥ 16/32 within the form, and mean pairwise stella-space cosine. **Action:** a form whose
   near-duplicate rate exceeds 25% has its quota cut to its post-dedup unique count rather than
   being topped up by more generation.
2. **Distribution overlap against real queries:** encode a 50,000-query sample of **MS MARCO dev**
   (validation only — never a seed, target, negative or gradient; no cache under
   `work/train/sources/`) and each form's sample with stella in the query role, and report each
   form's mean cosine to its nearest real-query neighbours plus the two-sample energy distance.
   **Action: none — this is a disclosed diagnostic**, because MS MARCO is web-search-shaped and a
   form legitimately unlike it (paper titles, arguments) must not be penalised for that. It exists
   so the report can state how far the training distribution sits from a real one, which is the
   one thing M9's coverage failure had no outside measurement of. FORMS-12 cannot serve this
   purpose: it scores student-teacher agreement on the same synthetic queries and is circular for it.

**Document corpus (document role):** the M9 pool, 6.15M documents, re-screened as above. FineWeb
documents are excluded (decision 5).

**Mix:** by *example*, decided by screen family B; default **75% query-role / 25% document-role**
examples per step. Query-role examples get raw bytes (prompt policy (b)); document-role examples
carry M9's fixed document-role marker; teacher targets use the s2p template for queries and raw
bytes for documents. The same student input never maps to two teacher targets.

**Hard candidates, the mining pass, the 1M bank, the HNSW fallback and the seed-rank field are all
struck by amendment A1** — nothing in M10 needs a candidate list once family D is a single
loss-form arm. Reopening condition in `m10/EXPLORED.md`.

## Recipe (defaults; screen families decide the marked items)

- **Student:** bge-small-en-v1.5 is the **screen anchor**; the **build student is decided by family
  F**, MiniLM-L6-v2 by default. **Amendment A6: F runs SECOND, right after A, and every later family
  is screened on F's winner** — it used to run seventh, which made every other verdict transfer to
  the build student by assumption. Before F can select MiniLM, its three-layer head passes the same
  export and fastembed parity check as bge-small's (M10.0-a; the Mac or the box can run it). **Feature [family G]:** masked mean-pooled hidden states of
  layers 12, 8 and 4 concatenated (1152-d; MiniLM-L6: layers 6, 4, 2) → Linear(1152→1024) → L2
  normalize; head 1.18M parameters, 34.5M total for bge-small. G's fourth arm adds layer 2
  (MiniLM-L6: layer 1) → 1536-d, head 1.57M, 34.9M total, under the cap; its head passes the same
  parity check first. Warm-started in closed form (ridge)
  for the bge-small init; the M9-candidate init keeps its 384-d head and zero-initializes the two
  extra layers' columns. Exported per token so fastembed's mean pooling reproduces it exactly.
- **Phase 1:** squared L2 on unit-norm teacher vectors, fp32 loss, bf16 autocast (M9 form). A
  cosine-space variant is closed by algebra, not measurement (`m10/EXPLORED.md`): the head ends in
  L2 normalize and the targets are unit-norm, so ‖a−b‖² = 2−2cos and the gradients agree up to a
  factor of 2.
- **Objective [family D], amendment A1 — one arm, not three.** The single arm is **D-NORM**:
  LEAF's own loss, the L2 **norm** `L = ‖t − s‖₂` on unit-norm vectors, in place of the squared L2,
  from the first step. This is what the 97.7%-retention system minimised and M9 had silently
  diverged from it; it is a one-line change and adds no data structure. Default is the squared L2.
  **The ranking-aware class (KL over mined candidates, InfoNCE on the seed passage) is CUT** — with
  it the τ entropy rule, the 1M-document bank, the mining pass and the 129-way spec. Evidence:
  LEAF's Appendix B added distillation terms to plain regression and found no improvement, and
  every comparable asymmetric system (LEAF, EmbedDistill, arXiv 2306.11550, mxbai-edge-colbert,
  DistilVDR) trains by pure embedding regression. **Reopening condition, registered here so it is
  not a symptom-gated decision later:** the class reopens only as its own milestone, never mid-M10,
  and only if families A and G both land and the annealed plateau still misses the release bar by
  less than 0.01 — a gap that size is the only regime where a ranking term is worth its machinery.
- **Optimizer:** AdamW β=(0.9, 0.999), eps 1e-8, wd 0.01 on dim>1, clip 1.0. **Batch 32 examples**
  [family E].
- **Schedule:** 3 cycles of equal example count, each linear 1e-4→1e-5; 2,000 warmup steps in
  cycle 1. Evaluation at every cycle end (annealed) and at cycle midpoints (curve watch only).
- **Init [family C]:** bge-small (default) or the M9 candidate.
- **Dose:** **200M examples** registered — LEAF's dose (6.7M texts × 30 epochs ≈ 201M; PLANNING §5),
  three cycles of 66.7M. Tokens follow the mix: at 75/25, 150M × ~35 + 50M × ~230 ≈ **16.8B**; at
  50/50 ≈ 26.5B. Query epochs ≈ 37 over 4.0M texts, document epochs ≈ 8 over the 6.15M pool. At the
  **measured** 745 examples/s in one padded chunk (PLANNING §11, on the box) that is ≈ **75
  GPU-hours**, and ≈ 139 h in M9's two-chunk collate — so **length-bucketed single-chunk batching is
  part of the build, not an optimisation** (amendment A7). The day-one benchmark's rate replaces
  both, and family E's batch-32 penalty is recorded in the lock as the build's GPU-hour line. **Extension:** let m_k be the COV macro (full precision, the locked formula and
  evaluation hashes) at the end of cycle k. After every cycle k ≥ 3, one more cycle of 66.7M examples
  (linear 1e-4→1e-5, as cycle 3) starts iff m_k − max(m₁, …, m_{k−1}) ≥ 0.003 **and** the extension
  cycles already run are fewer than `max_extension_cycles`, an integer the lock fixes from the
  approved dollars minus every mandatory line of PLANNING §6 at the measured rates and the billed
  price (§Compute). Whole cycles only; a cycle whose projected cost plus billed spend to date would
  exceed the ceiling does not start. The lock records m_k's formula, the evaluation hashes,
  `max_extension_cycles`, and the spend source.
- **Kill:** non-finite loss/grad; two consecutive cycle-end evaluations more than 0.0056 below the
  best. **Plateau** is read best-to-best on annealed checkpoints only.
- **Seeds:** one shipping seed for the build (seed 0); confirmation seeds at screen dose per §Screen.
  **No full-dose replica runs** (decision 8 withdrawn, amendment A5); the replication band is the
  selected recipe re-trained at screen dose under two further seeds, reported descriptively.

## Screen — seven families, thirteen arms, eleven contrasts, locked at M10.0-e

**Screen dose = 5M examples** (2.5% of the build; ≈ 420M tokens at 75/25; ≈ 1.9 GPU-hours per arm at
the measured 745 examples/s), full 3-cycle schedule compressed to that dose, one seed, identical
evaluation: **COV at every cycle end; DEV-6 once, at the final checkpoint** (its 5.2M-document
hotpotqa and 6.17M-row heldout components cost ~13 GB of reads per pass — M9's practice).
Throughput is recorded for every arm and decides nothing except family E (below).
**Order (amendment A6): A → F → G → B → E → C → D.** Data first because it is the thesis, then the
student, then the two architecture/regime questions, then init and objective; every family after F
runs on F's winner, so no verdict transfers to the build student by assumption.
**Anchor** = the full M10 corpus (A4's data), mix 75/25, bge-small init, squared L2, bs 32, 1152-d
feature. Screens run **on the box** (amendment A7).

| family | arms | contrasts | rule and default |
|---|---|---|---|
| **A — data (the thesis)** | A1: M9 pool (463,314 queries) · A2: M9 pool + PAQ (factoid forms only — the volume control) · A3: A2 + the **harvested real** query-like text · A4: A3 + the **generated** forms (the full M10 corpus, = anchor). **A2, A3 and A4 are cut to the identical post-screen unique-text count** (the smallest of the three after decontamination, the larger two downsampled with seed 0) and all hashes are locked before any arm | **A3−A2** (forms from real text, at equal volume) · **A4−A3** (what generation adds over harvesting) · A4−A2 and A2−A1 descriptive | **Three registered outcomes on A3−A2** (the forms contrast, now carried by the real-text arm): corrected lower bound > MDE → coverage **resolved on the COV families**, build proceeds; point ≥ MDE and lower bound > 0 but ≤ MDE → **positive, not resolved**, build proceeds and the report says so; otherwise → **M10 stops before any build and returns to Dylan with all four rows**. **A4−A3 decides whether the generated half is in the build at all**: if it does not resolve, the build uses A3's corpus and the ≈1.0M generated queries are dropped from the build (they stay in the report as a measured null). A2−A1 is the volume effect; if it resolves, the build keeps volume as well as forms |
| **F — student** | bge-small (34.5M with the head) · MiniLM-L6-v2 (23.9M) at equal examples | 1 | bge-small only if it wins **resolved**; **default MiniLM-L6-v2** — owner preference for the low-compute point (2026-09-01); MiniLM is 2× cheaper to train and serve (`m9/RESULTS.md`), and M9's screen had it −0.0026 unresolved. Runs second so the rest of the screen uses the winner |
| **G — output width** | feature = last layer only (384, M9's head) · last two of the three layers (768) · three layers (1152, = anchor) · four layers (1536, §Recipe) | 1152−384 · 1152−768 · 1536−1152 | resolved winner; **default 1152** (the probe's evidence that width binds under L2 and is still rising at three layers, PLANNING §9–9b; the screen, not the probe, decides). The 384 and 768 arms are also the paper's evidence for the M9 diagnosis |
| **B — mix** | 100/0 · 50/50 query/document (75/25 = anchor), **matched query presentations** (3.75M query examples in every arm; document examples 0 / 1.25M / 3.75M on top; totals 3.75M / 5M / 7.5M; the document cost in tokens and GPU-hours is reported) | 100/0−75/25 · 50/50−75/25 | resolved winner; default 75/25 |
| **E — batch** | 32 · 128 at equal examples and identical schedule | 1 | resolved winner; default 32 (LEAF). **Amendment A7: E is the one family whose throughput is read** — bs128 measured 1,331 examples/s against bs32's 745 (PLANNING §11), so a bs32 win must also be worth its 1.8× build cost; the lock records both the quality contrast and the GPU-hour delta, and a bs32 win that does not resolve reverts to bs128 |
| **C — init** | bge-small (or MiniLM, per F) · the M9 candidate | 1 | resolved winner; default the off-the-shelf backbone. Available only if F selects bge-small: M9's candidate is a bge-small student, so a MiniLM build has no warm start and C is skipped and reported as skipped. The closed-form ridge head warm start (M9's `m9s1c`, +0.0272) is retained in every arm regardless |
| **D — objective** | anchor (squared L2) · **D-NORM** (LEAF's ‖e‖₂) | 1 | resolved winner; default squared L2. The ranking-aware class is cut (amendment A1, §Recipe) |

A2 exists only as a control; the build never uses more than 1.0M PAQ. **Equal examples** holds for
every family except B, which is matched on query presentations by design. **Definitions:** a
decision's *margin* is the COV macro difference between the winner's and the default's final
checkpoints in the original screen; an arm's *seed range* is max minus min of its COV macro over
its three seeds.

**Rule, per contrast (families B–G):** the difference in COV macro (family-weighted, §Surfaces)
between the two arms' final checkpoints; paired stratified bootstrap over queries within component,
B = 20,000, seed 0; a contrast **resolves** when the point estimate ≥ the MDE **and** the one-sided
lower bound at the **0.025/11 quantile** (Bonferroni over the eleven contrasts) is > 0, and the sign
is stable across the last two cycle-end checkpoints. **Amendment A4: the MDE is not 0.0056 by
assertion — it is set at the lock from the measured COV resolution number** (§Surfaces), as
`MDE = max(0.0056, the measured point-to-lower-bound distance at 0.025/11)`, so the registered
detectable effect cannot be smaller than the surface can resolve. If that distance exceeds 0.010 the
lock must first enlarge the surface or cut contrasts, and record which; a screen whose MDE its own
surface cannot support is not run. **Family A's contrasts A3−A2 and A4−A3 are exempt from the
generic rule** and use the three-outcome rule in the table (resolved requires the corrected lower
bound > MDE); A4−A2 and A2−A1 are descriptive. **Confirmation:** for every decision whose
non-default option won, both the winner and the default are re-trained with two more seeds at
screen dose — **at most two such decisions, largest margins first** (amendment A5; it was four);
the rest revert to default. The decision stands only if the winner's margin exceeds the largest
seed range observed in either arm. Worst-case confirmation cost is now 45M examples ≈ 17
GPU-hours, plus the synthesized selected-recipe arm (5M) and the replication seed pair (10M);
PLANNING §5 has the arithmetic. Every screen verdict is artifact-specific at screen dose; never
"resolved" in the report's sense.

## Surfaces

- **COV** — the primary selection surface: **qrel-bearing** retrieval components with an
  affirmative licence at the dataset's primary source (M7's eval-use standard), decontaminated,
  and **read by no M10 decision**, admitted at M10.0-d, **weighted equally per family**
  (slices within a family averaged first). Candidates, draft records in `m10/COV_CANDIDATES.md`
  (primary-source licences checked 2026-09-01): **consumer-health** MTEB MedicalQARetrieval (CC BY
  4.0 at MedQuAD; NIH sources); **BRIGHT**, one family, six slices (biology, earth-science,
  economics, psychology, robotics, sustainable-living; CC BY 4.0 on the benchmark at its primary
  source — the same dataset-level standard that admitted CQADupStack and the six, whose documents
  are also third-party text; the document-rights caveat is disclosed and the data is evaluation-only,
  never redistributed); **legal** MTEB LegalBenchCorporateLobbying (CC BY 4.0; ConsumerContractsQA
  refused, CC BY-NC); **finance** LEDGER (CC BY 4.0 annual-report QA) once its structure and a
  100K-chunk cap are verified. Climate-FEVER refused (no licence at its primary source, as in M7).

  **Admission reopened 2026-09-04 (Dylan: non-commercial licences are admissible for validation, not
  training — `research/m7-data-licensing.md` §Rule change 2026-09-04).** A set refused *for its licence
  alone* is re-admissible, and the resuming session must re-run admission before the family floor is
  judged: **MTEB ConsumerContractsQA (CC BY-NC) is admissible**, restoring a second legal set and
  taking the count to four **without** LEDGER, whose structure was never verified. Any other CC BY-NC
  or research-only eval set is now in scope. **Still refused:** Climate-FEVER (no affirmative grant —
  a different class), and anything excluded for contamination. MS MARCO itself is admissible but is a
  **poor COV member** — every comparator trains on it and neither of ours does, so it is biased
  against us; prefer it for the within-system read under FORMS-12.
  **At least three families must survive admission** or M10 returns to Dylan; the report names the
  family count (Codex pass 6 preferred four — with the CQADupStack pair demoted, four needs LEDGER;
  Dylan may raise the floor). The two CQADupStack components are **DEV**, reported beside every COV
  read, never in the macro. COV contains no
  scientific-claim, paper-title or argument retrieval (no licensed, non-contaminating set exists),
  so **family A's verdict is a verdict about coverage on the COV families**; those three forms are
  tested only by the six-set transaction (FORMS-12 reports them descriptively before that).
  **Resolution number (M10.0-d, before the lock) — amendment A4: it now SIZES the screen.** With the
  contrast rule's own bootstrap (paired, stratified within component, B = 20,000, seed 0), measure the
  distance between the point estimate and the one-sided 0.025/11 lower bound for the COV-macro
  difference between two frozen comparators that are candidates in no M10 family — **e5-small-v2 and
  gte-small** (both `results/FINAL_MATRIX.md` rows) — each scored symmetrically on the admitted
  surface. Only the distance is recorded, never which comparator led. It is the first disclosed COV
  read (`m10/LEDGER.md` §4) and is published beside every contrast, so a reader can tell an unresolved
  verdict from an invisible one. **The lock then sets MDE = max(0.0056, that distance)**, and if the
  distance exceeds 0.010 the lock must enlarge the surface or cut contrasts before any arm runs.
  Codex pass 7's objection was that the earlier version scored family F's own backbones and made a
  selection; this version scores non-candidates, records no direction, and feeds a power calculation
  — which is what a pre-registration is supposed to do before fixing an MDE, not after.
- **DEV-6** (incl. the two CQADupStack components) secondary, reported beside every COV read.
  SCREEN-3 is retired.
- **FORMS-12**: 12 × 500 held-out synthetic queries, overlap@10 between student and teacher
  rankings over the 1M bank, per form. **Descriptive only** — teacher agreement on generated
  queries is a coverage diagnostic, not retrieval quality.
  **Now also permitted (2026-09-04, optional, descriptive):** the same overlap@10 diagnostic on a
  sample of **real MS MARCO dev queries**, as an external check that the 12 forms cover the natural
  query distribution. Validation only: MS MARCO text may **never** seed generation or enter the pool,
  and no MS MARCO-derived cache may be written under `work/train/sources/` (`m7src/mix.py:22-25`).
  **The stella-space distribution-overlap gate of amendment A8 (§Data) is the primary use of this
  permission** and runs on the manifest before any arm; this overlap@10 row is the post-hoc version
  of the same idea and stays optional. Neither is a substitute for COV: both are teacher-agreement
  or distributional diagnostics, not retrieval quality.
- **LoTTE-clean** (7 slices, macro over slices; its corpora — ~2.8M passages — are encoded with
  stella once, ≈ 1.3 GPU-hours at the assumed 600 docs/s, budgeted in PLANNING §6): **read #1**
  after the recipe lock.
  Before LoTTE opens, the **selected recipe is trained once as a single synthesized arm at screen
  dose** (its checkpoint hash committed); read #1 scores that checkpoint and the anchor's in one atomic batch.
  Veto rule (M9 §7, unchanged): the selection is vetoed if the selected recipe's 7-slice macro is
  worse than the anchor's by more than 0.004 AND the one-sided 97.5% paired-bootstrap upper bound
  (B = 10,000, seed 903, paired within slice) on (selected − anchor) is below −0.004; a veto means
  **the anchor recipe builds**. No other action may follow from read #1. **Read #2** pre-freeze,
  audit only. No third read.
- **Dev→six calibration**: M9's close-out rows (available only after M10.2) and M7's; forecasting
  only, never gating.
- Dev reuse is counted (`m8src/dev_reuse_m8.py`) and published.

## Final run

`m9/FINAL_LOCK.md` re-registered verbatim for M10 in the M10.2 lock with new BOUND-AT-FREEZE hashes
and an `m10-six-spent` tag — bridge as phase 1, the C-conjuncts with the empirical 0.0125-quantile
bootstrap bound and the Holm sign-flip conjunct, FEVER labelled — **with two changes. (1) The
reserved conditional is `if any C-conjunct passes then execute`**, so an aim claim never stands
without its descriptive reserved rows. **(2) Amendment A3: four conjuncts, not two** — `C1a`/`C2a`
on avg-6 and `C1b`/`C2b` on clean-4 (bars 0.5042 / 0.5155 / 0.5046 / 0.5233), Holm across the four
inside the unchanged 0.025 family alpha, the sequence fixed in the M10.2 lock before any six-set
output exists. clean-4 is M14's registered headline partition for both halves of the pair, so it
carries a pre-registered pass/fail here rather than arriving as a descriptive row after the fact. Zero alpha on the reserved batch is unchanged. `m9src/final9.py`'s
scoring path is written and reviewed before M9's close-out and reused.

## Compute and costs — re-priced 2026-09-04 on measured rates (amendment A7)

**Split execution.** The 2026-09-01 ruling stands where it bites — no dose or screen size is set by
the box's wall-clock — but the box is an execution target again for everything that is not
generation, because the rate it was withdrawn on was wrong:

| configuration, M10 recipe shape, **measured on the RTX 3080 2026-09-04** (PLANNING §11) | examples/s | 200M examples |
|---|---|---|
| batch 32, 75/25, M9's two-chunk collate | 400 | 139 h |
| batch 32, one padded chunk | **745** | **75 h** |
| batch 128, 75/25 | 1,331 | 42 h |

The plan's imported LEAF planning rate was 560 examples/s on a rented A100; the box meets or beats
it, because at batch 32 with ~35-token queries the step is launch-bound, not FLOP-bound, so a
bigger card buys ~1.5–2× rather than 10×. Caveat, disclosed: these are fixed-shape random-token
microbenchmarks with no data loading and no evaluation, so the real trainer is slower; they bound
the *hardware*, not the pipeline. **Consequences, all registered here:** length-bucketed
single-chunk batching is part of the build (§Recipe); the screens, confirmations, COV admission and
encodes, and M10.0-c run **on the box**, which already holds ~200 GB of M9 caches the plan had
budgeted 12 GPU-hours and a day of network to re-derive; the day-one benchmark still re-derives
PLANNING §6 before anything scales.

**What the money buys.** Generation needs the cloud — Qwen3-8B bf16 is ~16.4 GB on a 10 GB card —
and the build is rented for wall-clock, not because the box cannot do it (the box would take
≈ 3–6 days at the measured rates). One rented **A100 80 GB** (H100 if its cost per example measures
lower on the smoke), ≥ 500 GB persistent disk that survives stopping the instance, SSH, and a GitHub
deploy key so the headless commit-and-push contract holds. Provider is Dylan's choice; $1.5–2.5/h
assumed, unverified Sept 2026. The instance is stopped between stages.

| line | GPU-hours | $ at 1.5–2.5/h |
|---|---|---|
| generation, ≈1.0M queries (was 3.0M) | 10–20 | 15–50 |
| build, 200M examples at the measured rate | 40–75 | 60–190 |
| cloud-side encodes the build needs + export, parity, final run | 6 | 9–15 |
| day-one rate benchmark | 1 | 2–3 |
| persistent disk, egress | — | ≈ 25 |
| **cloud total, hybrid (screens on the box)** | **57–102** | **≈ $110–280** |
| all-cloud variant (screens and confirmations rented too) | 130–190 | ≈ $220–500 |
| optional: extension cycle, 66.7M examples, each | 13–25 | 20–63 |

Hard ceiling **$1,000**, unchanged and now far from binding. **Allocation order at the lock:** every
mandatory line first at the measured rates and the billed price, then whole extension cycles from the
remainder (the second build seed is withdrawn, amendment A5). A cycle whose projected cost plus
billed spend to date would exceed the ceiling does not start.

**Day-one rate benchmark (≈ 1 GPU-hour, before any scale-up):** stella docs/s on 20K pool documents;
training examples/s at batch 32 for 2,000 steps on the 75/25 anchor mix, the 50/50 mix and MiniLM-L6;
generation requests/s per form on the 200-query smoke, end to end (prefill, decode, retries, JSON
failures); the provider's billed price. Wall-clock ≈ 1.5–2 weeks; screen arms are independent and the
box runs them while the cloud generates.

On the instance, `results/perquery.json` (tracked) is sha-verified against `6b18e3dd…` before any
scoring, and M9's parity sample is regenerated from `m9/registry.json` (tracked; seed and sha256
pinned) — `m9src/port.py` refuses a sample that does not hash to the lock, in which case the 512
texts are transferred from the box. Only what training needs travels to the instance (tokenized
corpora and teacher targets, or re-derived there from the repo and HF); nothing that only the
evaluation stages need. Reserved-set encodes run only if the reserved conditional fires.

M9's cost protocol (`instructions-m9.md` §Costs) unchanged; the frontier is reported per index
configuration, naming the one measured (`m9/RESULTS.md` rounds 1–4). TurboQuant 4-bit (Qdrant 1.19)
joins the M13 all-in quantization comparison.

## Unimpeachable by competitors — what the report must carry

1. Comparators byte-verified against the official artifacts, scored in the same exact-search
   harness, frozen as per-query vectors before any M10 number existed; the bridge check proves
   the harness unchanged.
2. Pre-registration with pushed commits as the external witness; one six-set transaction per
   milestone, each disclosed (M7, M9 close-out, M10); the reserved four's single access; dev-reuse
   count and the selection-surface table published.
3. Contamination handled three ways: stella's disclosed overlap (ArguAna, FiQA) at every headline;
   NDO-4 rows; reserved NDO-3 and LoTTE as surfaces no decision touched.
4. Training data affirmatively licensed, no MS MARCO, attribution recorded; synthetic queries from
   an open-weights generator with pinned revision, per-query provenance and removal counts.
5. Compute disclosed in examples, tokens and GPU-hours beside LEAF's ~100 A100-hours.
6. System-level framing only (§Goal): no claim isolates nano from its document tower.
7. Reproducibility: code, corpus manifests with hashes, model revisions, seeds, statistics code
   with tests, and the full screen table including losing arms and confirmation seeds.

## Deliverables

Frozen candidate + `m10/FREEZE.json` (`assert_releasable` with a proper run record — it walks
`work/runs/<id>.json` for `cfg.sources` and `cfg.init` across the whole lineage and fails CLOSED on
a missing ancestor, which is why M9's freeze was refused; the record must be **derived from the
run's own hash-bound `manifest.json`, never asserted by hand**, because a post-hoc record that
happens to satisfy a licence guard is precisely the artifact that must not be fabricated; see
`m9/BUILD_LOG.md`), the frontier
update, the M10 section of the report artifact, decisions logged in CLAUDE.md, handoff to M13.

## Out of scope (reopening conditions in PLANNING §7)

Document-side co-adaptation (inside M10 it breaks the pair; a tower co-trained against both query paths at once keeps it and is recommended 2026-09-01 as the next-milestone candidate (that slot was M12 then; **M16** after the 2026-09-04 renumbering), Dylan's call, PLANNING §7) · any student above 35M
(hard cap, Dylan 2026-09-01) · teacher change (stella-1.5B measured worse; Qwen3-0.6B never
screened and not the pair) · a nonlinear head (no fastembed path; width comes from linear
multi-layer pooling) · a cosine-space phase-1 loss (closed by algebra, `m10/EXPLORED.md`) · a
ranking-aware phase-2 loss (cut by amendment A1; reopening condition in §Recipe) · MS MARCO **in any training role** (unchanged; **validation/diagnostic use was
permitted 2026-09-04**, see the admission note in §COV) · FineWeb in any role (decisions 3 and 5) · any
change to zero.
