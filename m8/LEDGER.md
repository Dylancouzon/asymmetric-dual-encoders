# M8 protocol ledger

**The protocol authority for M8.** Partitions, licence evidence, every pre-registered bar and
decision rule, gate results, freeze record, incidents. Detail lives in `results/m8_*.json` and is
pointed at, never restated. Transcribed 2026-08-29 from `m8/PLAN-DRAFT.md` v5 (itself gated by
three Codex passes and one Opus scientific-judgment review; archived at
`research/m8-planning/`). Nothing here is new decision-making: every rule below traces to the plan,
to Dylan's twelve rulings (§12), or to M7's ledger, which binds unchanged except where amended in
writing here.

**Reading order for a cold session:** `m8/STATUS.md` → this file. `m7/LEDGER.md` is the inherited
protocol and is authoritative for anything this file does not override. `m7/CODEMAP.md` is the
module map and pitfall list — read it before writing code.

---

## 0. Inheritance: frozen / amendable / ruled

| class | items |
|---|---|
| **FROZEN** (registered in `instructions-m8.md` on 2026-08-28, *before* M7's six-set number existed; not amendable by anyone but Dylan, and not by him retroactively) | The four confirmatory sets (FEVER, DBpedia-entity, CQADupStack-android, CQADupStack-english; hash-pinned, un-scored); paired frozen-M7-vs-frozen-M8 in ONE access; the statistics family shape (Holm + raw CI + simultaneous bound, dependence-preserving); six-set scoring descriptive-only and labelled development-informed; comparator BARS from frozen M7 + frozen `fusion.bm25_run` + published numbers as context; minimum release bar = beats frozen M7 CI-resolved on the reserved sets; licensing and decontamination rules; dev-only selection; the one-access freeze/ledger protocol. |
| **AMENDABLE, in writing, only before the first M8 number the change would affect** | Macro weighting; exact hypotheses / α / legs; dev-suite composition; probe designs; the E12 descriptive-comparator addition (registered as such, §5). Every amendment gets a dated entry in §15 with its reasoning, and may only move a bar in the harder direction once its numbers exist. |
| **RULED by Dylan 2026-08-28/29** | E1–E13, §12. Reopening any of them is Dylan's call, never inferred. |

**The one class of exception to "past decisions are revisitable" (CLAUDE.md) applies here in
full:** the evaluation protocol exists to make a good number believable. Protocol changes are legal
only *before* the numbers they would affect are observed, and only written down with reasoning.
Never retroactively, never silently.

---

## 1. Environment

Inherited from `m7/LEDGER.md` § Environment, unchanged: RTX 3080 / 10 GB VRAM, 25 GB RAM (18 GB
peak budget), 16 cores, ext4, nvcc 12.6; Python 3.12.14, torch 2.8.0+cu126, transformers 4.57.6,
datasets 5.0.1, pytrec-eval-terrier 0.5.10, qdrant-edge-py 0.8.0, Qdrant server v1.19.0; lock
`m7/requirements.lock.txt`; training dataset revisions `results/m7_trainmix_revisions.json`.
Disk at transcription: 781 GB free.

**Teacher (incumbent, the registered default): `NovaSearch/stella_en_400M_v5` @
`ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`**, dim 1024, WordPiece 30,522, `trust_remote_code` with
the 14-file sha256 pin in `results/m7_teacher_code_pin.json` and vendored modules under `vendor/`.
Workstream T (§10) may challenge it; only Dylan may swap it.

**Doc-encode dtype**: fp16 for dev and training, fp32 compute for the confirmatory access, fp16 at
rest — the M4 convention the frozen comparators were produced under.

**Code lives in `m8src/`.** `m7src/` is frozen: M8 reads it, imports from it, and does not edit it
(guardrail G3, §14). `bench/` and `scripts/` are the shared M1–M6 harness; do not fork them.

---

## 2. Partitions

### 2.1 CONFIRMATORY — the reserved four (FROZEN, un-scored, one access)

Pinned by `results/eval_manifest.json` (`m7_untouched_final`) + `results/frozen_eval/untouched-*`,
including `qtexts_sha256`. Never opened by any M8 code path outside the guarded access
(guardrail G2, §14).

| set | docs | queries |
|---|---|---|
| FEVER | 5,416,568 | 6,666 |
| DBpedia-entity | 4,635,922 | 400 |
| CQADupStack-android | 22,998 | 699 |
| CQADupStack-english | 40,221 | 1,570 |
| **total** | **10,115,709** | **9,335** |

**No clean member, disclosed at the rows.** M7-mix TRAIN-document overlap: FEVER 11.3%,
DBpedia 9.32% — these are **M7-mix placeholders and are recomputed for M8's own final mix**
(§3). android/english are ~0% overlap but are the same *family* as two dev components, so they
measure within-family transfer to unseen subforums, never "untouched generalization". FEVER
additionally carries the E9 proxy-provenance caveat, and every confirmatory leg is reported a
second time FEVER-excluded as a registered sensitivity read.

**Cost, and why the pre-encode is separated from the access:** 10.12M documents ≈ **20.6 GB fp16
per teacher-dim system**, tens of hours of teacher encode. Document vectors for every scored
system are pre-encoded and hash-pinned **after the freeze and before the access**, by a named
script that is physically unable to open the untouched query/qrel payloads (§6, §14 G2). Doc
encoding reads no queries and no qrels and produces no ranking: it is the same contact class as
the mandated decontamination. The guarded access is then a minutes-long gather/rank/metric step.

### 2.2 DEVELOPMENT — M7's pinned suite, inherited verbatim

Six components, hash-pinned in `results/m7_dev_manifest.json` (sha256
`4f991015db4407080ed9c8d1d2a85541d34b1e676aa593e510296eedca77e2ea`): nq-250k 250,000 docs /
3,452 q · hotpotqa 5,233,329 / 7,405 · cqadup-programmers 32,176 / 876 · cqadup-physics 38,316 /
1,039 · heldout-train (corpus = the 6,169,142-doc pool) / 7,325 · heldout-longq (same corpus) / 55.

Inherited disclosures that still bind: heldout-longq is a **55-query SUBSET of heldout-train**
(identical per-query nDCG, weighted 1/6 as a component and 55/7,325 inside another) → every dev
comparison uses the dependence-preserving statistics (§4.3). heldout-train is a
**seen-document / unseen-query** slice and rewards document-anchored memorisation. BM25 and potion
have **no row on the two held-out slices** (pool row indices carry no document text); that is
stated at every call site, never absorbed silently by an intersection.

**Selection rule (§4.5): median / worst-group gain over the registered groups, never the
arithmetic-mean macro.** M7's clean-4 are burned diagnostics and are not a dev instrument.

**M7's six are DEVELOPMENT-INFORMED for M8** and may be scored descriptively for continuity; every
six-set claim carries the label "development-informed at milestone level". They also carry a ship
consequence, one-directional only: the six-set **no-regression guard** (§5).

### 2.3 SHADOW — LoTTE (E10, mandatory, STOP-on-failure)

Adopted under the written "not literally CQADupStack" reading; CC BY-SA, pre-clickwrap 2021 dump.
Hash-frozen at the very start of Phase 0, **after** the overlap measurement:

- **Overlap measurement, run before the pin**: community-list intersection AND document-hash
  overlap versus reserved android/english and dev physics/programmers.
- Any hit → **drop the offending slice** and record it in the wake-up note.
- **Material overlap → E10 reopens** and goes to Dylan; the session does not decide it.

**Registered coverage statement, stated up front rather than discovered:** the shadow reads the
**CQA half of the estimand only**. No clean Wikipedia/entity-shaped shadow exists — candidates were
swept and failed on licence. The FEVER/DBpedia half is guarded instead by the six-set
no-regression guard and the worst-group guard (§5). Touché-2020 is banned and stays banned
(args.me is ArguAna's source family).

**One crossing, ever.** GO threshold = the minimum detectable effect from the joint power
simulation (§4.4), **not zero**. Registered branches: GO → the access; NO-GO → "defer to M9, panel
preserved" (default) or report-only. Never a second crossing; never a fallback candidate.

### 2.4 M9 RESERVE (E4: both sets)

EUR-Lex and USPTO retrieval sets. **Overnight freezes ONLY the corpus and query-text
inventories** — what the contamination filter needs. Qrel construction and the final hash-pin
follow a reviewed procedure and are accompanied by a **PROVISIONAL sanity report**. Never scored
in M8; covered by the protected-query filter (§3) and by guardrail G2.

### 2.5 TRAIN

Approved sources only (`research/m7-data-licensing.md`, `results/m7_field_table.md`). M8 re-derives
its own mix and its own counts (§3); M7's 340,850-pair figure is the inherited starting point, not
an M8 number. Source-level licence evidence (NQ, HotpotQA, CQADupStack, FEVER, ESCI, MIRACL,
Mr. TyDi, DBpedia) is in `m7/LEDGER.md` § Source-level licence evidence and binds unchanged.
**MS MARCO stays excluded from the release stack, permanently** (non-commercial-research terms;
IBM Granite is the precedent). `freeze.assert_releasable` enforces it and is ported to M8 paths.

---

## 3. DATA: decontamination and enforcement (precedes every probe)

**Rules, inherited:** R1 **remove** on query overlap (all partitions) · R2 **remove** on
positive-document overlap with protected evaluation sets · R3 **measure and disclose, do not
remove** for protected *documents* — removal there would forbid training on Wikipedia while
evaluating on Wikipedia benchmarks; what removal protects (test queries and qrels) is enforced by
R1, and every comparator has the same property, so the comparison stays like-for-like.

**Method, inherited:** blake2b-64 word hashes, polynomial rolling word-8-grams, bottom-32 sketch,
≥8/32 shared (est. Jaccard ≥ 0.25); word-4-grams additionally for 4–7-word queries on query paths
only. Index built over the TRAIN side; protected corpora streamed against it.

**M8 extensions, binding and ordered before any probe:**

1. **The protected-query filter covers six + reserved-4 + LoTTE shadow + M9-reserve inventories.**
   Built, hashed and committed BEFORE any teacher download or probe (ordering interlock, §14 G7).
2. **R1/R2 run over every corpus in the M8 mix, Wikipedia ICT included.** Post-filter hashes
   frozen.
3. **The closed-form fit list is regenerated through the filter.** M7's
   `work/trainq_texts.json` (349,934 queries, dumped pre-filter) carries **4,582 R1 hits (1.31%)**
   and is unusable for any M8 absolute number. Its *ranking* use in M7 was sound (shared fit set);
   M8 does not inherit that excuse because M8's teacher screens may change the frame.
4. **M8-mix overlap rates are recomputed and replace the M7-era figures** at the reserved rows.
5. **Near-duplicate screening vs ALL protected partitions** for every new corpus, in addition to
   R1/R2.

**Cleared sources** (primary-source verified, zero eval overlap): USPTO (37 CFR), EUR-Lex
(2011/833/EU; TDM-silence noted), US federal. **OUT**: bulk arXiv, SEC EDGAR, HackerNews,
post-2024 StackOverflow. **PMC-OA: EXCLUDED** (E8); the PMID measurement is deferred to the single
condition that would revisit it — a biomedical-specific gap shown by the genre probe.

**Genre probe (registered):** ONE frozen bundle (USPTO + EUR-Lex + federal at registered shares,
capped total technical share), matched examples and updates against a Wikipedia-only arm.
Endpoint = the registered OOD groups plus a technical non-protected exploratory group built from
held-out cleared-corpus pseudo-queries. **Whole-bundle in or out; no post-hoc cherry-pick.**

**Synthetic queries (E2):** a dosed, registered component of the pool spec with its own bar.
Qwen3-line prompted, deduplicated against all benchmarks, per-query provenance retained.
Downsides recorded at approval: style bias; self-limited by the OOD bar; ~half a day of GPU.

**FineWeb arm (E13):** `Qdrant/FineWeb-10B` span sampler (~1–2M spans) → the full
contamination/near-dup filters vs ALL protected partitions → teacher-encode the survivors. The arm
joins the **registered data probe** under the clean-stack-tax design: same bar, matched exposure,
**never released**, refused by `freeze.assert_releasable`. If it clears the bar by a margin worth
shipping, the number goes to Dylan and the **licensing ruling is his** — never inferred. Recorded
so it is not re-derived later: Qdrant redistributing FineWeb is evidence of the company's posture,
but **a wrapper tag — including our own — is not a licence.**

---

## 4. Statistics (pre-registered; executable before the first M8 number)

### 4.1 Inherited machinery

`m7src/boot.py`, unchanged: `signflip` is THE p-value (paired sign-flip randomization on the
macro, R = 100,000, seed 0, valid at any n) and Holm consumes only these; `paired` gives intervals
(B = 10,000, seed 0) and its tail mass is named `boot_tail` because it is **not** a p-value;
`_align(strict=True)` on every confirmatory path — abort on a missing dataset or qid, never score
the intersection.

**Decisions read `ci95_raw` / `one_sided_lower_raw`, NEVER the rounded `ci95`.** A true lower
endpoint of +4e-5 displays as 0.0000. This rule has been broken twice in this project's history,
once in `final_run.py`, on the one irreversible decision.

### 4.2 The estimand and the family

**Estimand: the equal-weight macro over the four reserved sets.** The grouped variant (CQA pair
as one group, FEVER and DBpedia as their own) is a **registered sensitivity read only**, never the
primary.

One-sided paired hypotheses; **family α = 0.025**; **m = 3**. A leg is resolved only with all three
of:

1. Holm-corrected `signflip` rejection at family α = 0.025;
2. the raw two-sided 95% paired CI lower endpoint > 0 (unrounded);
3. the raw one-sided lower bound at the Bonferroni level **α/3 = 0.008333** > 0, from the **same**
   bootstrap draws.

| leg | hypothesis |
|---|---|
| **C1** (primary) | fused-M8 > fused-M7 |
| **C2** (strict, per E11) | **dense** released-M8 system > frozen **dense** M7 system |
| **C3** (absolute floor) | fused-M8 > BM25, frozen `fusion.bm25_run` builder |

C2 is presented as a **system comparison, not table causality**: both endpoints are fully frozen
systems and the claim is about them, not about which component moved.

### 4.3 Dependence

`signflip_dep` / `paired_dep` (`m7src/test_dep_stats.py` is the guard): one shared sign per
underlying qid; stratified bootstrap resampling each membership stratum once and reusing the draw
in every component it feeds. Reported three ways so the effects separate (ordinary →
fixed-stratum-independent isolates conditioning; → shared-draw isolates covariance). The four
reserved sets are disjoint, so the *confirmatory* macro needs only dataset-stratified paired
resampling — but the dev suite's nesting (heldout-longq ⊂ heldout-train) requires the dependence-
preserving path throughout selection, and the code takes the same route on both so the two cannot
diverge.

### 4.4 Registration deliverables (before Phase 0 spends its week)

1. **Executable confirmatory decision code** (`m8src/decide.py`): draws, seed, stratified paired
   resampling, strict qid alignment, Holm ordering and tie handling, the α/3 bound, unrounded
   endpoints. A rule a session can re-read in its own favour is not a pre-registration.
2. **The joint power simulation of the full ship rule** — every leg, both guards, the qualifying-
   table requirement — publishing **minimum detectable effects AND P(ship)** under the surviving
   levers' expected effects. This goes in front of Dylan **before** Phase 0 spends its week: a
   knowing report-only choice beats a discovered one.
3. **`rule_audit.py` ported to M8**, so rule compliance is audited rather than discovered. M7 found
   its step-selection rule unapplied *by accident*, after it had governed four arms and a promoted
   adoption.

### 4.5 Weak-null calibration caveat (carried verbatim, may never be dropped)

Sharp-null Holm validity is exact under per-query exchangeability. Under two empirical weak-null
constructions the complete three-leg rule rejected at **0.0198 and 0.0283** against a nominal
0.025 (`results/m7_tier_rule_calibration.json`, S = 4,000); these simulations are **sensitivity
evidence, not a demonstration of uniform weak-null FWER control**; a union bound over the marginal
legs alone would say 0.039. The claim "the family is bounded at 0.025" is **withdrawn
permanently** and the nominal figure may never be quoted without this alongside.

### 4.6 Dev statistics are SELECTION evidence

Every dev p-value and CI is selection evidence. The only confirmatory claims are the three frozen
comparisons in the single access. **No text may say a lever was "statistically confirmed".**
The measured **recipe-perturbation band is 0.0027–0.0078** — a dev-selection fact about which
recipe you happen to pick, which does **not** deflate a frozen confirmatory comparison, and which
every dev-scale adoption claim must be read against. Replay noise is ~5e-6.

**Dev-reuse counter runs from evaluation #1** → `results/m8_dev_reuse_count.json`.

### 4.7 Noise floor (measured before any bar is frozen)

**Two matched null replicates per endpoint a bar will read**: a seed change, and ±10% A-steps.
Published to `results/m8_noise_floor.json`. **Every bar is then set at ≥ 2× the measured floor**,
or uses the **B3 template** (sign-consistency on both OOD components + a seed replicate). Bars
below carrying `TBD-noise-floor` are frozen by an amendment entry in §15 once the floor exists.
**No bar, no run** — enforced by `m8src/probe_guard.py` (§14 G1), not by prose.

---

## 5. The ship rule

**Ship requires C1 ∧ C2 ∧ C3, all three legs each, PLUS all four of:**

1. **A qualifying v2 table.** Enumerated as qualifying: objective-family change, data-construction
   change, feature/tokenizer change, row-init-construction change, structural riders, doc-side
   head. **NOT qualifying**: seed, step count, temperatures, negative counts, learning rate, pool
   sizes, ordinary tuning. A distinct int8 payload is **necessary but not sufficient**.
   **D1 (doc-side head) alone does NOT satisfy this** — per strict C2 (E11) and gate-4's G4-4, a
   qualifying *table* change (R1 or D2) must also survive for the release path to open.
2. **The +0.005 point guard on C1** — a product margin, not a hypothesis. A statistically resolved
   win that is not worth a version bump is not a v2.
3. **The worst-group guard** — no reserved group regresses versus fused-M7 by more than 0.01 at
   the point estimate.
4. **The six-set no-regression guard** — descriptive, frozen per-query vectors, **zero new
   access**: M8 must not fall below M7 on the six by more than a registered margin.
   *Why it exists:* all four reserved sets are training-adjacent, and M7's FINDINGS #13 signature
   (gains concentrated in-distribution) would otherwise read as a win there. This is the
   anti-memorization ship-blocker.

**Descriptive context inside the same access (E12, registered here before any M8 number):**
`bge-small-en-v1.5` and `LR-dense-websearch` scored on the reserved four — **outside the Holm
family, no ship consequence**, the external anchor for the report. Their document vectors are
pre-encoded with everything else (§6).

**FEVER handling (E9):** FEVER rows carry the proxy-provenance caveat; **all legs are additionally
reported FEVER-excluded**; the cancellation argument is stated **conditional on a stella-lineage
teacher** and is void if the teacher leaves that lineage.

**A miss is a publishable outcome, written down before the number exists** — inherited from M7's
REPORT FRAMING and re-registered here. Nothing about the system may change after the reserved-set
numbers are seen.

---

## 6. Pipeline (order is binding)

```
protected-partition freezes (LoTTE + M9-reserve inventories)
  → protected-query filter build
  → teacher freeze (workstream T)
  → noise-floor measurement
  → Stage R  (one assembly + ONE validation gate)
  → Stage S  (one finalist, by executable rule)
  → three-seed aggregation
  → int8 export
  → ONNX parity
  → fusion instantiation
  → immutable candidate manifest
  → ONE mandatory LoTTE shadow crossing
  → freeze
  → reserved-4 doc pre-encode (all scored systems)
  → THE SINGLE ACCESS
```

- **Any post-manifest mutation invalidates the shadow crossing.**
- **A teacher change after Stage R begins = a full R/S restart.**
- **One-shot mechanics inherited verbatim** from `m7/LEDGER.md` § Provenance / "The one-shot path,
  hardened 2026-08-28": durable spent-receipt tag pushed to origin the moment the confirmatory
  block is on disk; **peeled** tag check (`refs/tags/X^{}` — an annotated tag resolves to the tag
  OBJECT and would otherwise never match); the access counts as SPENT when the RESULT FILE holds
  the confirmatory block, not when the ledger says so; exclusive pid lock; strict hashes; atomic
  write before any secondary block; table snapshot after verification; the guard runs *before*
  preflight; a single `--infra-retry` requiring the same commit. `test_final_guard.py` and
  `test_freeze_binding.py` are ported to M8 paths and must pass.

---

## 7. Stage R — degrees of freedom, each with its M7 fallback

**R0 :=** the registered M7 recipe settings instantiated under the selected teacher, current
filters, and M8's data volume / precision / seed policy. R0 is the fallback for everything below.

| # | degree of freedom | probe | fallback |
|---|---|---|---|
| 1 | ICT pair fraction | B3 | M7 (no ICT component) |
| 2 | listwise distillation arm — candidate sampler + split temperatures | B2-triggered performance arm | M7 objective |
| 3 | phase structure — ONE registered three-arm test: sequential / mixed-replay / listwise-only, equal updates, B-target retention tracked | — | M7 sequential B→A |
| 4 | negatives | B13, matched steps | M7 (`hard_neg_k=0`) |
| 5 | hyperparameters | B13 | M7 |
| 6 | target design | B8 | M7 |
| 7 | row init | B15 | M7 (`teacher`) |
| 8 | pool spec — ONE frozen composition: per-source quotas ≤25%, multi-span/doc, Wikipedia ICT, the genre bundle (§3), and a dosed synthetic-query component (E2) with its own bar | genre probe + E2 bar | M7 pseudo-query pool |
| 9 | riders: B9 low-rank, B10 pooling, B14 doc instruction | B9/B10/B14 | M7 |

**Probe outputs are tri-state**: adopt the named setting / keep the named fallback / stop the
direction. Diagnostics may only trigger **separately registered** performance arms. Cross-frame
conclusions (closed-form → trained) are reconfirmed on the assembled candidate.

**B13 confirms ONE complete named configuration jointly** — a single confirm arm against the
complete fallback — never per-axis adoptions.

**The fused-objective lever (recipe P3) is consciously EXCLUDED per E11** (strict C2: the dense
table must stand alone, because hybrid is not everyone's deployment). Recorded so no future
session re-derives it as an oversight.

**Assembly and the one validation gate.** Adopted settings form one bundle. **ONE** common-frame
validation: assembled-R1 vs R0, matched updates / data / seed policy, **dense AND fused
endpoints**, bar sized from the confirmatory arithmetic (registration deliverable §4.4). The
outcome is binary: R1, or **wholesale fallback to R0**. No component-by-component back-off —
that is adaptive dev search and is forbidden (the M7 simplification precedent).

**Fusion operator** — family grid, depth, dev components, frozen `bm25_run` — is frozen **before**
Stage R and applied identically at every fused read; the final invocation instantiates parameters
only. It may be amended only if D4' BM25F re-registers the lexical function, and only before
Stage R. Inherited mechanics: one family, one parameter, no per-dataset weights or routing,
`fusion.DEPTH` = 1000 for selection and application alike, fitted against the **int8 release**
artifact; the zero-score-padding drop and the self-hit drop are part of the frozen function; on an
exact tie the **simpler** system wins (dense-only first, then the first grid point in order);
`n_tied_at_best` is written into the spec.

---

## 8. Stage S — the menu and the selection rule

**Selection**: fixed within-family rules first (D2's vocabulary by its own nested dev split), then
family finalists versus **R1-alone** on one named group vector — registered groups, precision,
aggregation and budget, with **worst-group as an explicit formula** — under a practical-equivalence
band. **Tie-break**: total downloadable bytes, then doc-index delta. Registered outcomes: **no
survivor → the candidate is R1-alone; multiple → the rule picks.**

**Release-format rule: int8, always.** It is the C2 identity, the proven-quality-free format
(M7 G4 int8 upper bound 0.00013 against a 0.005 bar), and what the ONNX graph embeds. All sizing,
eligibility and tie-breaks are computed at int8. A 4-bit variant is **research-only in M8 and never
ships**.

**Seeds**: three. Aggregation pre-declared per architecture class — table-average only for
identically-parameterized aligned tables, otherwise mechanical median. **Never best-seed.**

### The menu

- **D2 — compositional capacity (in scope).** Self-trained tokenizer (64–128K, multi-word merges),
  rows initialized per B15's winner, **trained through the forward** under R1. Sized at int8 under
  the 233 MB cap — 128K × 1024 int8 = 131.6 MB, so it fits with no quantization experiment.
  Gated by B7. **Registered coverage spec:** a minimum-updates-per-reachable-row criterion;
  targeted rare-row span sampling with the pool expansion needed to meet it; a
  coverage-vs-capacity diagnosis rule for any failure; and the "bag mass on cold rows vs per-query
  retention" diagnostic run **on existing artifacts first**.
  *Context:* M7 shipped with **1,743 rows (5.71%) never trained by either phase**
  (`results/m7_cold_rows_p4n-teacher16-a.json`); 994 were `[unusedN]` placeholders and the
  reachable 749 contributed at 0.143× a trained row. A 128K vocabulary makes coverage the first
  question, not an afterthought.
- **D1 — doc-side head (in scope per E3, conditional).** Linear 1024→1024 / 2-layer MLP / →512
  variants over cached teacher vectors, jointly trained. **Preconditions:** (i) fuses into ONE
  doc-side ONNX file as plain MatMul/activation nodes — E3's hard condition, tested at B6 entry;
  (ii) per E11 + G4-4, **a D1 win alone does not make a qualifying v2 table** (§5.1).
  *Why it is live at all:* the "absorbable" dismissal was **half wrong**. `q·(Md) = (Mᵀq)·d` holds
  only if the mapped document is not renormalized; retrieval uses L2-normalized document vectors,
  so the per-document factor `1/|Md|` cannot move into a shared table — rank agreement with the
  absorbed form is 1.000 without renormalization and **0.000 with it**
  (`results/m7_absorb_check.json`).
- **D4' — lexical arm, bounded (in scope, auxiliary).** BM25F title/text, weights dev-fitted and
  frozen **before** Stage R (re-registers the fusion function). **May never be the sole qualifying
  change.** Full-dose dual-index question expansion stays dead on compute (263–702 days); a
  bounded ≤50K-doc research probe may run under E2 and may **never** be extrapolated to the
  reserved system.
- **R1-only** — legitimate **iff** at least one qualifying change survived (§5.1).

**Research-only, never candidates:** D3 index-time adaptation (E5 — one labelled measurement
allowed AFTER the final access); D5 nonlinear query-side head (E1 — out entirely; that niche is
`constella-nano`'s).

**Kill-list, standing (algebra and arithmetic, not preference):**
higher table dims (identity-linear MRL heads off a 1024-d hidden state);
absorbable transforms as capacity — query-side centering, whitening, top-PC removal, any per-token
scalar weight (`results/m7_absorb_check.json`); allowed only as registered training priors, killed
on cross-domain validation;
full late interaction;
another 31 MB unigram table with better hyperparameters as the sole change.

---

## 9. Probe registrations

Every probe entry point calls `m8src/probe_guard.py`, which reads this section at the current
commit and **refuses any probe id lacking bar, endpoint, comparator, multiplicity and
no-survivor outcome**. Bars marked `TBD-noise-floor` are frozen by an amendment in §15 after
`results/m8_noise_floor.json` exists, at ≥ 2× the measured floor or the B3 template.

**Before any probe:** (i) benchmark every new code path on a 10K-doc / 1K-query slice and publish
the serial GPU/RAM/disk schedule — including the reserved pre-encode line and the timed B+A chain
that settles the ~30-min-vs-3-h discrepancy; (ii) measure the noise floor (§4.7).

### Wave 1

| id | question | endpoint | comparator | bar | multiplicity | no survivor |
|---|---|---|---|---|---|---|
| **B2** | Is the KL term degenerate? (H2: one-hot to ~1e-4 nats under 31 uniform distractors at temp 0.02) | candidate-set entropy quantiles, uniform vs top-200 | — (diagnostic) | descriptive; **may not adopt anything** | none | triggers the separately registered listwise arm, or not |
| **B3** | Is Phase A pair-starved? (H1) | dense + fused, registered OOD groups | equal updates and equal exposure across ICT fractions {0, .25, .5, .75} | **B3 template**: sign-consistency on both OOD components + a seed replicate | within-probe, 4 arms | keep R0's ICT fraction |
| **B17** | Does the class cap in-domain? | 50/50 query split on the dev CQA components; oracle table fitted on one half, scored on the other, against the 0.481 teacher ceiling | the teacher ceiling | **registered routing rule** (below) | none | routing rule is exhaustive by construction |
| **B9** | Is the table low-rank? | SVD rank truncation curve | full-rank R1 rows | `TBD-noise-floor` | within-probe over ranks | keep full rank |
| **B10** | Which scoring family? | sum / max / top-k / LSE, **exact search only** | R1's pooling rule | `TBD-noise-floor`, Holm within the family, raw CI > 0 in fp16 **and** int8 | within-family | keep R1's rule |

**B17's registered routing rule** (fixed before its number): held-out ≥ ~0.45 ⇒ supervision and
objective are the story, **R1 is the milestone's center of gravity**; stalls ≤ ~0.40 ⇒ the class
caps in-domain and **D2/D1/D4' carry the milestone**; between ⇒ both, budget split as registered.

### Wave 2

| id | question | gates | notes |
|---|---|---|---|
| **B7** | block-CG vocabulary curve, 30.5K control / 64K / 128K | **D2** | the arithmetic that closed granite-r2 and gte-modernbert in M7 was a 50,368-vocab fp64 Gram at 20.3 GB; the CG solver is what reopens that class |
| **B6** | doc-side map on a frozen table | **D1** | **precondition: a demonstrated fused one-file doc ONNX graph**, then the quality bar |
| **B8** | target design: bare + doc-centroid targets | R-DoF 6 | closed form |
| **B13** | A-grid + matched-steps negatives + riders | R-DoF 4/5/9 | **ONE complete configuration confirmed jointly** |
| **B14** | doc-side instruction refit | R-DoF 9 | the two OOD dev corpora only |
| **B15** | context-averaged row init (Wada) | R-DoF 7, D2 | |

**Inside workstream T: B16** (MEV / self-similarity) — **descriptive only; may not prune a
candidate** unless separately validated on fresh clean-screen artifacts.

### Removed from M8's calendar, with reasons

- **B1' / B4** — E1 makes them decision-irrelevant here; recorded as **M9 planning diagnostics**.
- **B5** — E5: index-time adaptation is research-only, AFTER the final access.
- **B12** — superseded by the int8-always rule; a 4-bit sweep may run post-finalist as research.
- **B11** (fusion complementarity) — moot under E11's strict C2; the fusion operator is frozen
  mechanics, not a lever.

---

## 10. Workstream T — the teacher (opens Phase 0)

**Order: protected freezes → filter → screens.** No teacher download or probe before the filter is
built, hashed and committed (§14 G7).

- **Fixed student frame per screen.** The fit list is regenerated through the current filter;
  M7's list had protected hits and is unusable (§3.3).
- **Provenance rows per candidate** (registry-proxy convention), so a disclosure liability is
  priced before adoption, not after.
- **Candidates**: the **incumbent re-probed in the same frame** (mandatory — a screen without the
  incumbent is not a paired comparison); `granite-embedding-english-r2` and `gte-modernbert-base`
  as CG-frame controls; `stella_en_1.5B_v5`; `harrier-oss-v1-0.6b`.
- **stella-1.5B breaks WordPiece compatibility** — fingerprints get rebuilt if it wins.
- **harrier-0.6b's training data is undisclosed** — a contamination black box. **It needs a
  ruling from Dylan before adoption**, and the session does not make that ruling.

**SWAP BAR (all conditions, none waivable by a session):**

1. The challenger's **closed-form distilled table** beats the incumbent's on the probe components,
   CI-resolved. *(The criterion is the table, not the tower: Spearman(symmetric ceiling, distilled
   table) = 0.000 over eight candidates, and arctic-embed-l, approved on the ceiling, produced a
   table 0.0480 BELOW the incumbent's.)*
2. **The margin exceeds the swap's CI-widening penalty** — near-sibling ≈ 0.005 half-width,
   dissimilar ≈ 0.0096, stated numerically from the power simulation. A swap that wins by less
   than it costs in resolution is a loss.
3. A widened off-family read (nq-250k and hotpotqa) does not reverse the sign.
4. **Dylan signs off.**

**Costs a swap charges, written down so they cannot later be discovered as reasons to avoid it:**
double reserved-4 pre-encode (~20.6 GB and tens of hours *per system*); the FEVER-cancellation
argument is lost if the teacher leaves the stella lineage (E9); WordPiece/fingerprint rebuild;
re-encode of the 6.17M-doc pool, dev corpora and TRAIN targets; fusion re-selection; gate re-run.
**Same-teacher is the registered default.** Naming reopens with Dylan if the teacher leaves the
stella lineage (`constella` = constellation + stella).

**Tie-break** if two are within noise: prefer no disclosed overlap with the protected sets, then
the smaller dimension.

**ONNX feasibility evidence is assessed for every finalist BEFORE the freeze** — a successful
local export or clear family precedent. **Absence of a published artifact is not failure**;
demonstrated infeasibility is the only ONNX-based exclusion.

**Reusable arithmetic bound (inherited):** base out-approximates large in every family by +0.04 to
+0.07, so a family whose *large* variant scores below ~0.28 on the table criterion cannot reach
stella by shrinking. That closes a shortlist by arithmetic rather than exhaustion.

**Second-machine protocol** (`m7/LEDGER.md`) stays in force if any probe runs on Dylan's Mac:
the second machine must also produce an incumbent row; any Mac winner is re-probed on the RTX box
before it moves anything; `validate_encoder.py` must pass there first; work lands on its own
branch.

---

## 11. ONNX / fastembed (scope approved)

Feasibility verified: `research/m8-planning/onnx-feasibility-2026-08-29.md` — stella's export
blocker is two config flags (`unpad_inputs`, `use_memory_efficient_attention`), and
gte-large-en-v1.5 (same architecture) ships first-party ONNX. Verdict: days, not weeks.

1. **`constella-zero` ships AS an ONNX graph** — Gather → sqrt-count pool → normalize, int8
   initializer — fastembed-native, with the BM25-bespoke-class fallback.
2. A **parity-verified export of the SELECTED teacher** is an M8 task; M10 is the fallback landing
   zone. Demonstrated ONNX-infeasibility (not absence of an artifact) is the only ONNX-based
   teacher exclusion.
3. **D1, if it survives, ships fused into the doc graph — one file** (E3's condition).
4. **Parity runs on the final aggregated int8 artifacts BEFORE the shadow crossing**: full
   conformance fixture suite, vector and cosine tolerances, top-k tie policy, an nDCG delta bound,
   all pinned in the manifest.
5. Index-side tooling is **offline, not served**.

---

## 12. Rulings (Dylan, 2026-08-28/29 — authoritative; prior wording superseded)

| # | ruling |
|---|---|
| **E1** | **Pure lookup is the product.** No query-side neural head in M8, not even as research — "if people have some compute capability, there's no reason to not use M9". |
| **E2** | **Synthetic Qwen3 training queries approved** ("green light if no downsides"; downsides recorded at §3). |
| **E3** | **Doc-side head approved CONDITIONALLY**: must fuse into the doc ONNX graph as plain nodes — one served file, no custom pipeline — and clear its probe. |
| **E4** | **Reserve BOTH M9 sets** (EUR-Lex + USPTO), frozen construction procedure, never scored in M8. |
| **E5** | **Index-time adaptation: research-only, end of project** ("seems over engineered… afraid of the accusations"). Never in the confirmatory candidate. |
| **E6** | **Training-only second teacher allowed if licence-clean**; the vendor rule binds *shipped* components; documented on the model card. |
| **E7** | **Byte cap 233 MB int8** ("storage can be fairly cheap"); cold-start, latency and optics matter more than bytes. |
| **E8** | **PMC-OA excluded** (delegated "include if it moves the needle, otherwise exclude" → excluded: its unique value is duplicated by cleaner sources; the cost is the NFCorpus/TREC-COVID honesty read). |
| **E9** | **FEVER: label + sensitivity read** — proxy-provenance caveat at the rows; all legs also reported FEVER-excluded; cancellation stated conditional on a stella-lineage teacher. |
| **E10** | **LoTTE adopted as the mandatory shadow** under the written "not literally CQADupStack" reading — pending the overlap measurement (§2.3); STOP-on-failure. |
| **E11** | **STRICT C2** ("we want something that looks good on benchmarks too; hybrid should be the default but isn't to everyone") — the dense table must beat M7's dense table; the fused-objective lever stays consciously excluded. |
| **E12** | **Comparators inside the access: YES — bge-small + LR-dense-websearch**, descriptive only, outside the Holm family, registered before any M8 number. |
| **E13** | **FineWeb (`Qdrant/FineWeb-10B`): measure first, ship-decide later.** The affirmative-licence standard stays in force for the RELEASED stack; the arm runs under the clean-stack-tax design (filtered, matched exposure, never released, refused by the release guard). If it clears the bar by a shippable margin, the licensing ruling comes back to Dylan **with the number**. A wrapper tag — including our own — is not a licence. |

---

## 13. Inherited-obligation matrix

| item | disposition |
|---|---|
| sqrt full-chain arm | Own registration slot at R1-assembly time — run, or formally deferred with owner-visible reasoning. B10/B13 inform it; they do not falsify it. **Never revived at M7's arm (a).** |
| n-gram rows (carried lever) | **Superseded by D2** — a no-whitespace multi-word tokenizer IS the n-gram direction in non-overlapping form. If D2 dies, additive rows need their own registration; **no auto-revival.** |
| negatives / step-count confound | B13 matched-steps; disposition from its single joint confirm arm. The honest M7 statement stands: "the dev suite cannot separate the negatives source from the step count", never "mined negatives do not help". |
| doc2query full dose | Dead on compute for anything confirmatory; bounded research probe only (E2). |
| teacher revisit | Workstream T, §10. Swap bar includes the CI-widening penalty; same-teacher default; Dylan signs. |
| M7 mandatory ablations (flat-vs-learned weights, prefix variants, init controls, dense/BM25/fusion decomposition, int8) | Mapped per eligible architecture; each **adopted or not-applicable WITH a reason** recorded here before the freeze. |
| ANN sweep + cost reporting | `ann_sweep.py` on the final candidate; cost rows split **payload / container / doc-index / hydration** — M7's frozen container was 93,886,950 bytes carrying *both* fp16 and int8 payloads; M8 ships int8-only and reports both numbers. |
| one-shot mechanics | Copied verbatim; guard and freeze-binding suites run against M8 paths. |
| M7 report addendum | LR-websearch row as a labelled **exploratory TIE** (+0.0019 [−0.0153, +0.0195]) — the honest sentence is "matches LR's single-table system at 1/15 the bytes". |
| exporter fix | int8-only artifact + §11 parity. |
| Wada context-averaged init / MEV | B15 / B16 (descriptive). |
| Touché-2020 | Banned, stays banned. |

---

## 14. Session guardrails (hard; a violation stops work and goes to the wake-up note)

| id | guardrail |
|---|---|
| **G1** | **No probe run before its bar is committed and pushed.** `m8src/probe_guard.py` reads this ledger at the current commit and refuses any probe id lacking bar / endpoint / comparator / multiplicity / no-survivor outcome. Every probe entry point calls it. |
| **G2** | **Path guard, allowlist form.** The protected paths are `results/frozen_eval/untouched-*`, the LoTTE payloads and the M9-reserve payloads. **No code under `m8src/` may open them except the modules on the explicit allowlist in `m8src/paths_guard.py`**, each named with the contact class that justifies it: (a) the freeze/inventory scripts (they must hash what they pin), (b) the decontamination/filter scripts (they must read protected *text* to protect against it), (c) the reserved-4 doc pre-encode script — corpora only, and physically unable to open a query or qrel payload, (d) the confirmatory-access script, only after the freeze. Everything else — every probe, every training path, every dev evaluation — is refused at open time. Enforced by a runtime shared-import guard **and** a static grep test over `m8src/`, both in the test suite. Adding a name to the allowlist is a §15 amendment, not an edit. |
| **G3** | **Nothing irreversible without Dylan.** No freeze, tag, or access; no HF upload; no writes to `results/perquery.json`, `results/eval_manifest.json`, `results/frozen_eval/`, or `m7/`; no final M9-reserve hash-pin (inventories only); no amendment to any §0 FROZEN item. AMENDABLE changes require a dated, reasoned §15 entry. |
| **G4** | **Noise floor before any bar is frozen** (§4.7). |
| **G5** | **Long-run discipline, mechanically.** Smoke every new path (~90 steps) — prefer a path with **no execution history**. `setsid nohup` for anything > 10 min (harness interrupts kill background tasks; M7 lost a final run to exactly this). Monitors grep `Traceback\|Error\|FAILED\|OOM\|Killed\|assert` **alongside** the progress marker — a monitor that only matches the happy path cannot tell success from a crashloop. Write the wall-clock estimate BEFORE launch and kill any job exceeding it 2×. Take the rate check **in the slow region**, not on the first batches. |
| **G6** | **Publish the serial GPU/RAM/disk schedule before the first probe**, including the reserved-4 pre-encode line (~10.12M docs ≈ 20.6 GB fp16 per system — scheduled, not run early). One memory-heavy job at a time; 18 GB peak RAM budget; `flock`. |
| **G7** | **Ordering interlock**: no teacher download or probe until the protected-query filter covering six + reserved + shadow + M9-reserve inventories is built, hashed and committed. |
| **G8** | **Dev-reuse counter runs from evaluation #1** (`results/m8_dev_reuse_count.json`). |
| **G9** | **Wake-up-note discipline**: anything needing Dylan goes to the top of `m8/STATUS.md`; it is never decided alone. |

---

## 15. Amendments and incidents

*(Dated entries only. An amendment to an AMENDABLE item is legal here only before the numbers it
would affect exist; it states what changed, why, and that the numbers did not exist yet.)*

- **2026-08-29 — ledger opened.** Transcribed from `m8/PLAN-DRAFT.md` v5 at commit `f8b67f3`.
  No M8 number of any kind exists at this entry. No protected partition has been touched.
