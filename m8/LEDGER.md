# M8 ledger — CLOSED 2026-08-30

**M8 closed as a MEASUREMENT.** `m8/FINDINGS.md` is what this milestone produced; `m8/EXPLORED.md`
is the closed-avenue register. **This file is now the PROTOCOL AND PROVENANCE record**: the rules a
future milestone inherits, every ruling Dylan made, every claim this project withdrew, and the
measured numbers a bar reads.

**Compressed 2026-08-30** from 1,619 lines. Removed: the stage pipeline, the Stage-R and Stage-S
menus, the prose probe registrations (`registry.json` is the authoritative pre-registration record)
and the gate-findings index — all M8-specific machinery for work that will not run. **Kept verbatim:
every Dylan ruling, every withdrawn claim, the guardrails, the statistics and the floors.** The rule
that governs this file is §14 G10 and the lesson behind it: *a cut is not safe because prose was
removed — it is safe only when every RULE survives.*

## 0. Inheritance: frozen / amendable / ruled

| class | items |
|---|---|
| **FROZEN** (registered in `instructions-m8.md` on 2026-08-28, *before* M7's six-set number existed) | The four confirmatory sets (FEVER, DBpedia-entity, CQADupStack-android, CQADupStack-english; hash-pinned, un-scored); paired frozen-M7-vs-frozen-M8 in ONE access; the statistics family shape (Holm + raw CI + simultaneous bound, dependence-preserving); six-set scoring descriptive-only and labelled development-informed; comparator BARS from frozen M7 + frozen `fusion.bm25_run` + published numbers as context; minimum release bar = beats frozen M7 CI-resolved on the reserved sets; licensing and decontamination rules; dev-only selection; the one-access freeze/ledger protocol. |
| **AMENDABLE, in writing, only before the first M8 number the change would affect** | Macro weighting; exact hypotheses / α / legs; dev-suite composition; probe designs; the E12 descriptive-comparator addition (registered as such, §5). |
| **RULED by Dylan 2026-08-28/29** | E1–E13, §12. Reopening any of them is Dylan's call, never inferred. |

**The amendment rule, stated once and without an exception.** An AMENDABLE item may be changed
only **before any raw number it would affect exists**. Once such a number exists the rule is
closed — **in both directions, easier or harder**. Every amendment gets a dated §15 entry stating
what changed, why, and that the dependent numbers did not yet exist. Never retroactively, never
silently.

*(v1 of this file carried a clause permitting a bar to move "in the harder direction" after its
numbers existed. That is wrong and is deleted: post-hoc tightening is still selection on an
observed number, and it contradicts `CLAUDE.md` and M7's practice. Codex gate MAJOR 1.)*

---

## 1. Environment and code

Inherited from `m7/LEDGER.md` § Environment, unchanged: RTX 3080 / 10 GB VRAM, 25 GB RAM (18 GB
peak budget), 16 cores, ext4, nvcc 12.6; Python 3.12.14, torch 2.8.0+cu126, transformers 4.57.6,
datasets 5.0.1, pytrec-eval-terrier 0.5.10, qdrant-edge-py 0.8.0, Qdrant server v1.19.0; lock
`m7/requirements.lock.txt`; training dataset revisions `results/m7_trainmix_revisions.json`.
Disk at transcription: 781 GB free (766 GB after the LoTTE fetch).

**Teacher (incumbent, the registered default): `NovaSearch/stella_en_400M_v5` @
`ffeb2b7ee715c226d4ffe5e4619f7dbb48624c20`**, dim 1024, WordPiece 30,522, `trust_remote_code` with
the 14-file sha256 pin in `results/m7_teacher_code_pin.json` and vendored modules under `vendor/`.
Workstream T (§10) may challenge it; only Dylan may swap it.

**Doc-encode dtype**: fp16 for dev and training, fp32 compute for the confirmatory access, fp16 at
rest — the M4 convention the frozen comparators were produced under.

**Code lives in `m8src/`.** `m7src/` is frozen: M8 reads it, imports from it, and does not edit it
(G3). `bench/` and `scripts/` are the shared M1–M6 harness; do not fork them. Module names in
`m8src/` must not shadow `m7src/` names — `m8src/_paths.py` did, and silently broke every m7src
import until it was renamed `m8base.py`.

---

## 2. Partitions

### 2.1 CONFIRMATORY — the reserved four (FROZEN, un-scored, one access)

Pinned by `results/eval_manifest.json` (`m7_untouched_final`) + `results/frozen_eval/untouched-*`,
including `qtexts_sha256`.

| set | docs | queries |
|---|---|---|
| FEVER | 5,416,568 | 6,666 |
| DBpedia-entity | 4,635,922 | 400 |
| CQADupStack-android | 22,998 | 699 |
| CQADupStack-english | 40,221 | 1,570 |
| **total** | **10,115,709** | **9,335** |

**Pin provenance, disclosed (inherited from `m7/LEDGER.md`, dropped from v1 of this file and
restored — Codex gate MAJOR 5).** `qtexts_sha256` was **computed from the already-committed
`results/frozen_eval/` payloads**, whose `qids_sha256` was re-verified at the same time — NOT
re-derived from a fresh download, because regenerating would itself have been an access. It
therefore proves **forward immutability**, not independent provenance; git history is what
supports how the payloads got there.

**No clean member, disclosed at the rows.** M7-mix TRAIN-document overlap: FEVER 11.3%,
DBpedia 9.32% — **M7-mix placeholders, recomputed for M8's own final mix** (§3). android/english
are ~0% overlap but are the same *family* as two dev components, so they measure within-family
transfer to unseen subforums, never "untouched generalization". FEVER additionally carries the E9
proxy-provenance caveat.

**Cost, and why the pre-encode is separated from the access.** 10.12M documents ≈ **20.7 GB fp16
per teacher-dim system** (`results/m8_schedule.json`). Document vectors for every scored system
are pre-encoded and hash-pinned **after the freeze and before the access**, by a named script that
cannot open the untouched query/qrel payloads (§14 G2). Doc encoding reads no queries and no
qrels and produces no ranking: it is the same contact class as the mandated decontamination. The
guarded access is then a minutes-long gather/rank/metric step.

### 2.2 DEVELOPMENT — M7's pinned suite, inherited verbatim

Six components, hash-pinned in `results/m7_dev_manifest.json` (sha256
`4f991015db4407080ed9c8d1d2a85541d34b1e676aa593e510296eedca77e2ea`): nq-250k 250,000 docs /
3,452 q · hotpotqa 5,233,329 / 7,405 · cqadup-programmers 32,176 / 876 · cqadup-physics 38,316 /
1,039 · heldout-train (corpus = the 6,169,142-doc pool) / 7,325 · heldout-longq (same corpus) / 55.

Inherited disclosures that still bind: heldout-longq is a **55-query SUBSET of heldout-train**
(identical per-query nDCG) → every dev comparison uses the dependence-preserving statistics (§4.3).
heldout-train is a **seen-document / unseen-query** slice and rewards document-anchored
memorisation. BM25 and potion have **no row on the two held-out slices**; that is stated at every
call site, never absorbed silently by an intersection.

**THE MANDATORY DEV DISCLOSURE (inherited from `m7/LEDGER.md` § "THE DEV MACRO IS A BIASED
ESTIMATOR"; dropped from v1 and restored — Codex gate MAJOR 6).** Every adoption reports the full
dev macro **AND** the out-of-domain subset (cqadup-programmers + cqadup-physics). An adoption
whose gain is concentrated in-distribution is **labelled "in-distribution only"** and may not be
offered as evidence of reserved-set improvement. **The out-of-domain subset's resolution is
≈0.005** at n = 1,915 on a StackExchange-only proxy: a per-arm out-of-domain difference below
about 0.005 is *unresolved*, so "the out-of-domain effect is zero" is a claim this instrument
cannot make. "Nothing detectable, on a narrow proxy" is the required wording.

**Selection rule: median / worst-group gain over the registered groups (§8), never the
arithmetic-mean macro.** M7's clean-4 are burned diagnostics and are not a dev instrument.

**M7's six are DEVELOPMENT-INFORMED for M8.** They may be scored descriptively for continuity;
every six-set claim carries the label "development-informed at milestone level". Their one ship
consequence is the six-set no-regression guard (§5), and **when they are scored is registered**
(§5.4) precisely so "frozen, zero new access" cannot become false.

### 2.3 SHADOW — LoTTE (E10, mandatory, STOP-on-failure)

**Acquired 2026-08-29**: `https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz`,
3,576,167,599 bytes, sha256
`37c0f39af23a6e3464f63395a4d04a22b91fe59c1aa64ea1773a8aff113c7ab5`. Inventory and provenance:
`work/lotte/inventory.json`, `work/lotte/PROVENANCE.md`.

**LICENCE, measured rather than assumed — and it splits.** Santhanam et al. 2022, Appendix D,
verbatim: *"The original StackExchange post archive is licensed under a Creative Commons BY-SA 4.0
license… We will also release LoTTE with a CC BY-SA 4.0 license. The search queries can be used
for non-commercial research purposes only as per the GooAQ license."* So the passages and the
package are CC BY-SA 4.0, but **the `search` queries are non-commercial-research-only**.

**RULING FOR THIS SESSION (conservative, and it improves the instrument): the shadow uses the
`forum` queries ONLY.** They are CC BY-SA 4.0, and they are also the better analogue — forum
questions are what CQADupStack is made of, which is the half of the estimand the shadow can see
at all. The `search` split is not used, not encoded and not scored. Flagged to Dylan (§ wake-up
note) because it is a licence nuance on a Dylan-owned axis, but no ruling is needed for the
session to proceed: the restricted choice is legal under both readings.

**Dump date, evidenced not assumed:** no exact date exists in the primary source (the paper cites
the rolling `archive.org/details/stackexchange` item). Hard upper bound: arXiv submission
2021-12-02, corroborated by internal tarball mtimes 2022-01-15..22 and the server's
`Last-Modified: 2022-02-01`. So "a 2021 dump, predating StackExchange's 2024 no-LLM-training
clickwrap" is **supported**, with the precision limited to "before December 2021".

**Structural finding that changes the overlap measurement:** within a topic, LoTTE's dev and test
splits draw from **entirely disjoint** StackExchange communities (e.g. `science/dev` =
{chemistry, stats, academia, astronomy, earthscience, engineering, datascience, philosophy};
`science/test` = {biology, math, physics}; zero overlap). **Dev and test must therefore be
screened separately** — a clean result on one says nothing about the other.

**OVERLAP MEASUREMENT — executable, run before the pin (Codex gate MAJOR 3).** Probe id `S0`,
registered in `m8/registry.json`. For every (topic, split) slice, against reserved
android/english and dev physics/programmers:

1. **Community intersection** — `metadata.jsonl`'s `dataset` field vs the CQADupStack subforum
   names. Any shared community ⇒ **drop that slice**.
2. **Document near-duplicate rate** — the R1 fingerprint machinery (blake2b-64 word hashes,
   rolling word-8-grams, bottom-32 sketch, ≥8/32 shared ⇒ est. Jaccard ≥ 0.25), not exact hashes.
   Exact hashing alone was v1's method and misses paraphrase; it is not used.
3. **Query leakage** — the same fingerprint over LoTTE forum queries vs the protected query
   inventories, plus the word-4-gram path for 4–7-word queries.

**Materiality, numeric:** a slice is **dropped** if community intersection is non-empty, or if
document near-dup rate > **0.5%**, or if any query leakage hit is found (query leakage has zero
tolerance — a leaked query is a scored query). **E10 REOPENS and goes to Dylan** if, after drops,
either the surviving document near-dup rate exceeds **2%** or fewer than **two topics survive in
the split used for the crossing**. The session does not decide those.

### S0 OUTCOME, 2026-08-29: ALL TEN SLICES REJECT. E10 REOPENS WITH DYLAN.

`results/m8_lotte_overlap.json`, 5.25M documents screened in 19 minutes. The registered branch
fired: fewer than two topics survive in either split, so **the session does not substitute a
shadow** and the ruling returns to the owner. The failures split cleanly in two:

| | slices | reason |
|---|---|---|
| hard reject | `writing/test`, `science/test`, `technology/test` | their StackExchange communities **literally include protected sets**: english, physics, android + softwareengineering |
| query-leakage only | the other seven | 2–15 fingerprint matches per ~2,000 forum questions (0.1–0.75%); document near-duplicate rates 0.001–0.16%, all under the 0.5% bar |

**Exact** query matches concentrate almost entirely in the three community-overlapping slices
(111, 13, 34); the seven clean-community slices are nearly all fingerprint-**near** (2–12 each,
one exact). Two identical question titles across different StackExchange sites are not by
themselves evidence of contamination.

**The bar was not relaxed after it bit.** But the tension is real and goes to Dylan: §3's standing
rule **R1 removes the ITEM** on query overlap, while S0's newer, narrower bar drops the whole
**SLICE**. Under a per-question remedy seven slices survive with ~2,000 questions each. That
alternative is computed and stored in the artifact, labelled DESCRIPTIVE / NOT ADOPTED, and its
`would_keep` flag embeds an **unregistered ≥500-question threshold** which would itself need
registering if the remedy were adopted. A third option was measured and is on the table:
`results/m8_shadow_alternatives.json` — the **eight unused CQADupStack subforums** (gaming, gis,
mathematica, stats, tex, unix, webmasters, wordpress: 323,488 documents, 8,961 queries), already
licence-cleared, no contamination against reserved android/english, but the same benchmark family
as two reserved sets.

**Registered coverage statement, stated up front rather than discovered:** the shadow reads the
**CQA/forum half of the estimand only**. No clean Wikipedia/entity-shaped shadow exists —
candidates were swept and failed on licence. The FEVER/DBpedia half is guarded instead by the
six-set no-regression guard and the worst-group guard (§5). Touché-2020 is banned and stays
banned (args.me is ArguAna's source family).

**THE CROSSING — one, ever. GO rule, two-legged (Fable D1 + Codex gate MAJOR 3).**
The v1 rule was "LoTTE macro gain ≥ MDE". That is mis-specified: the shadow is structurally blind
to the Wikipedia/entity half of the estimand, which is exactly where the plan's flagship data
lever (Wikipedia ICT) should land, so a genuinely-above-bar candidate could NO-GO on an
instrument that cannot see its gain. Registered instead, before any crossing:

> **GO iff BOTH**
> (a) **LoTTE non-regression**: the surviving-slice macro of the M8 candidate is **≥ −0.0068**
>     against the frozen M7 system (the MDE from `results/m8_power.json`, used here as a
>     two-sided noise scale, i.e. the shadow may not show a real *loss*); **AND**
> (b) **dev worst-group ≥ +0.0068** over the registered dev groups (§8), the same MDE as a
>     positive requirement on the instrument that *can* see the whole estimand.

Endpoint: equal-weight macro of nDCG@10 over surviving LoTTE slices, forum queries, exact search.
Comparator: the frozen M7 released system on the identical slices. Statistic: point estimate for
(a) — it is a guard, not a hypothesis — and the registered dev selection statistic for (b).

**What this rule gives up, stated rather than glossed (2026-08-29 review).** v1 demanded an
affirmative gain on data never used for selection. v2 demands only **non-regression** there and
moves the affirmative requirement onto **dev**, an instrument the candidate was selected on and
which §2.2 itself labels a biased estimator. Leg (b) is numerically demanding, but its pass
probability is inflated by selection in a way v1's leg was not. **The crossing's protective
function is therefore catastrophe detection, not confirmation**, and it may not be described as
external validation. A middle rule existed and is recorded as considered: require the affirmative
gain on the shadow's *own* half of the estimand and non-regression only on what it cannot see.
It was not adopted because the shadow's CQA half is exactly where selection pressure is highest
(two dev components are CQADupStack), so that rule buys less independence than it appears to.
Second, leg (a)'s −0.0068 is the **reserved-4** MDE borrowed as a noise scale; at ~2,000 queries
per slice the shadow's own macro SE is roughly 0.001–0.003, so −0.0068 is a 2–6 SE allowance —
loose, and stated as loose.
Registered branches: **GO → the access; NO-GO → "defer to M9, panel preserved" (default) or
report-only.** Never a second crossing; never a fallback candidate; no re-selection after seeing it.

### 2.4 M9 RESERVE (E4: both sets)

EUR-Lex and USPTO. **Overnight built INVENTORIES ONLY** — what the contamination filter needs —
plus a provisional sanity report (`work/m9reserve/PROVISIONAL_SANITY.md`). Never scored in M8;
covered by the protected-query filter (§3) and by G2. Final construction, qrels and hash-pin are
M9's, following a reviewed procedure.

- **EUR-Lex**: corpus `NLP-AUEB/eurlex` (EURLEX57K, 57,000 docs, CC BY-SA per the loader and
  homepage), queries `dennlinger/eur-lex-sum` English (1,504 pairs, CC BY-SA 4.0 by its own card).
  Both COMPLETE. Underlying rights: Decision 2011/833/EU permits commercial and non-commercial
  reuse with attribution; **silent on TDM**, carried forward unchanged.
  **Finding:** only **216 of 1,504** eur-lex-sum celex ids exist in EURLEX57K — the two draw from
  different EU curation programmes, so the gold-doc pool must come from eur-lex-sum's own
  `reference` text, not from an assumed EURLEX57K membership.
- **USPTO**: buildable-now option is HUPD's pre-built `sample-jan-2016` (26,808 applications,
  **explicitly a SAMPLE**; full HUPD is ~125 GB and was correctly not pulled). Underlying text is
  public domain under 37 CFR 1.71(d)&(e), 1.84(s). **Licence correction, for Dylan:** HUPD's HF
  card tag is **CC-BY-NC-SA-4.0**, more restrictive than the "CC-BY" previously recorded. Under
  this project's standing rule a wrapper tag is not a licence and cannot restrict public-domain
  text — but that is a legal interpretation, so it goes to Dylan before it is relied on.
  The stronger citation-based construction (PatentsView, CC BY-4.0) needs an API key Dylan must
  request.

### 2.5 TRAIN

Approved sources only (`research/m7-data-licensing.md`, `results/m7_field_table.md`). M8 re-derives
its own mix and its own counts (§3). Source-level licence evidence (NQ, HotpotQA, CQADupStack,
FEVER, ESCI, MIRACL, Mr. TyDi, DBpedia) is in `m7/LEDGER.md` and binds unchanged.
**MS MARCO stays excluded from the release stack, permanently** (non-commercial-research terms;
IBM Granite is the precedent). `freeze.assert_releasable` enforces it and is ported to M8 paths.

---

## 3. DATA: decontamination and enforcement (precedes every probe)

**Rules, inherited:** R1 **remove** on query overlap (all partitions) · R2 **remove** on
positive-document overlap with protected evaluation sets · R3 **measure and disclose, do not
remove** for protected *documents*.

**Method, inherited:** blake2b-64 word hashes, polynomial rolling word-8-grams, bottom-32 sketch,
≥8/32 shared (est. Jaccard ≥ 0.25); word-4-grams additionally for 4–7-word queries on query paths
only. Index built over the TRAIN side; protected corpora streamed against it.

**M8 extensions, binding and ordered before any probe:**

1. **The protected-query filter covers six + reserved-4 + LoTTE shadow + M9-reserve inventories.**
   Built, hashed and committed BEFORE any teacher download or probe (G7). It emits a
   **query-only hash inventory** — fingerprints, no labels — so downstream filtering never needs a
   label-bearing process (Codex gate BLOCKER 6).
2. **R1/R2 run over every corpus in the M8 mix, Wikipedia ICT included.** Post-filter hashes frozen.
3. **The closed-form fit list is regenerated through the filter.** M7's `work/trainq_texts.json`
   (349,934 queries, dumped pre-filter) carries **4,582 R1 hits (1.31%)** and is unusable for any
   M8 absolute number.
4. **REQUIRED COUNT ARTIFACT (Codex gate MAJOR 7).** Before any training run,
   `results/m8_decontam.json` records, **per rule × per source × per protected partition**: the
   denominator, the number removed, and the rate — R1, R2, R3 and the near-duplicate screen
   separately. M7 published these per source; a filter without its counts is unauditable.
5. **Near-duplicate screening vs ALL protected partitions** for every new corpus, in addition to
   R1/R2.
6. **M8-mix overlap rates replace the M7-era figures** at the reserved rows.

**Cleared sources** (primary-source verified): USPTO (37 CFR), EUR-Lex (2011/833/EU, TDM-silence
noted), US federal. **OUT**: bulk arXiv, SEC EDGAR, HackerNews, post-2024 StackOverflow.
**PMC-OA: EXCLUDED** (E8); the PMID measurement is deferred to the one condition that would
revisit it — a biomedical-specific gap shown by the genre probe.

**Genre probe (registered, id `D-GENRE`):** ONE frozen bundle (USPTO + EUR-Lex + federal at
registered shares, capped total technical share), matched examples and updates against a
Wikipedia-only arm. Endpoint = the registered OOD groups plus a technical non-protected
exploratory group from held-out cleared-corpus pseudo-queries. **Whole-bundle in or out; no
post-hoc cherry-pick.**

**Synthetic queries (E2, id `D-SYNTH`):** a dosed, registered component of the pool spec with its
own bar. Qwen3-line prompted, deduplicated against all benchmarks, per-query provenance retained.
Downsides recorded at approval: style bias; self-limited by the OOD bar; ~half a day of GPU.

**FineWeb arm (E13, id `D-FINEWEB`):** `Qdrant/FineWeb-10B` span sampler (~1–2M spans) → the full
contamination/near-dup filters vs ALL protected partitions → teacher-encode the survivors. Joins
the registered data probe under the clean-stack-tax design: same bar, matched exposure, **never
released**, refused by `freeze.assert_releasable`. If it clears the bar by a margin worth
shipping, the number goes to Dylan and **the licensing ruling is his**. Recorded so it is not
re-derived later: Qdrant redistributing FineWeb is evidence of the company's posture, but **a
wrapper tag — including our own — is not a licence.**

**H1's prior, stated honestly (Fable A).** H1 says Phase A is pair-starved and the 924,704
discarded ICT pairs are the fix. The nearest measurement this project owns points the other way:
the clean-stack-tax arm added **490,241 real, high-quality MS MARCO pairs** to the frozen recipe
and moved the six-set int8 macro **+0.0058 [−0.0015, +0.0131] — unresolved**
(`results/m7_cleanstack_tax.json`). ICT spans are synthetic and should be worth *less* per pair
than MS MARCO pairs. So B3's honest prior is "a rounding lever, not a route to the bar", and the
budget allocation in §7 reflects that. The one caveat that keeps B3 worth running: the tax arm
held A-steps fixed at 2,500, so exposure per pair fell as pairs were added, and B3's
equal-updates-and-equal-exposure design is what separates those.

---

## 4. Statistics (pre-registered; executable in `m8src/decide.py`)

### 4.1 Inherited machinery, and which route each decision takes

`m7src/boot.py`, unchanged: `signflip` is THE p-value (paired sign-flip randomization on the
macro, R = 100,000, seed 0, valid at any n) and Holm consumes only these; `paired` gives intervals
(B = 10,000, seed 0) and its tail mass is named `boot_tail` because it is **not** a p-value;
`_align(strict=True)` on every confirmatory path — abort on a missing dataset or qid, never score
the intersection.

**Which route, explicitly (Codex gate BLOCKER 8).** v1 said both dev and confirmatory take the
dependence-preserving route. That is wrong and would have made leg 3 uncomputable: `paired_dep`
returns neither `one_sided_lower_raw` nor the α/3 bound.

- **CONFIRMATORY** (the reserved four) uses `boot.signflip` + `boot.paired`. The four sets are
  **disjoint** — no shared underlying query, no nesting — so dataset-stratified paired resampling
  *is* the dependence-preserving estimator here and `paired_dep` reduces to it exactly.
  **ASSERTED IN CODE, not merely argued** (`m8src/test_decide.py`, 2026-08-29): the ordinary and
  dependence-preserving point estimates agree to 1e-12 and their interval half-widths to within
  5% on the four reserved sets, which is the degeneracy the argument claims.
- **DEV** (the six components, two of them nested) uses `signflip_dep` / `paired_dep` throughout.
- **The α/3 bound is computed at the exact level.** `boot.paired` hardcodes the percentile string
  `"0.8333"`; M8 computes `100 × 0.025/3 = 0.8333333…` in `m8src/decide.py` from the same draws
  and **asserts the two agree to < 1e-6**, so a rounded percentile is never the authority.

**Decisions read `ci95_raw` / `one_sided_lower_raw`, NEVER the rounded `ci95`.** A true lower
endpoint of +4e-5 displays as 0.0000. This rule has been broken twice in this project's history,
once in `final_run.py`, on the one irreversible decision.

### 4.2 The estimand and the family

**Estimand: the equal-weight macro over the four reserved sets** — equal weight per DATASET, not
per query. The grouped variant (CQA pair as one group, FEVER and DBpedia as their own) and every
FEVER-excluded read are **registered sensitivity reads only**: descriptive, outside the family,
and **they may not rescue, replace or soften any C1–C3 verdict** (Codex gate MAJOR 8).

One-sided paired hypotheses; **family α = 0.025**; **m = 3**. A leg is resolved only with all
three of:

1. Holm-corrected `signflip` rejection at family α = 0.025;
2. the raw two-sided 95% paired CI lower endpoint **> 0** (unrounded);
3. the raw one-sided lower bound at **α/3 = 0.00833333** **> 0**, from the **same** draws.

| leg | hypothesis, stated as the endpoints it compares |
|---|---|
| **C1** (primary) | **fused-M8 system > fused-M7 system**, both complete released systems. |
| **C2** (strict, per E11) | **M8 int8 query TABLE > M7 int8 query TABLE**, both scored against the **same frozen incumbent document vectors, with any doc-side head (D1) DISABLED.** |
| **C3** (absolute floor) | **fused-M8 > BM25**, frozen `fusion.bm25_run` builder. |

**C2 IS ONLY COMPUTABLE UNDER THE SAME TEACHER, and the alternative is registered now rather
than discovered later (self-review, 2026-08-29, before any M8 number).** "The same frozen
incumbent document vectors" is satisfiable only while M8 and M7 share a document tower. If the
teacher swaps, M7's table is bound to stella's document space and M8's to another, and a
table-versus-table comparison across two different document spaces is not a table comparison at
all — C2 as written would be *unsatisfiable*, and a post-swap session would have to invent a
replacement after the fact. So both forms are fixed here, selected by a fact settled at pipeline
step 5, long before any M8 number exists:

- **Same teacher (the registered default): C2 = table vs table on identical document vectors,
  D1 disabled.** One encode serves both endpoints.
- **Teacher swapped (which needs Dylan's sign-off in any case): C2 = the dense SYSTEM comparison**
  — M8's released dense system against M7's released dense system, each on its own teacher's
  document vectors, D1 still disabled. The "same document vectors" clause is void because it is
  unsatisfiable, not because it was inconvenient.

This is a further, previously unstated cost of a swap, and it belongs in §10's list: **a swap
converts E11's strict table claim into a system claim.** That is a reason to prefer the incumbent
that has nothing to do with GPU hours.

*And a cost a swap does NOT carry, so it is not over-budgeted:* D1-disabled document vectors are
just the base teacher vectors, and D1-enabled is one GEMM away over the cached ones. Reporting C2
with D1 disabled needs **no second document encode**.

**Why C2 is written that way (Codex gate BLOCKER 2).** v1 said "dense released-M8 *system*", which
would include D1 if D1 ships — so a table that LOSES to M7 could be rescued by a document-side
head while the ledger claimed strict C2 had passed. E11's words are "the dense **table** must beat
M7's dense table". C2 is therefore table-versus-table on identical document vectors. **There is no
C4**: if D1 ships, its contribution appears inside C1's system and is otherwise **descriptive
only**. m stays 3.

### 4.3 Dependence

`signflip_dep` / `paired_dep` (`m7src/test_dep_stats.py` is the guard): one shared sign per
underlying qid; stratified bootstrap resampling each membership stratum once and reusing the draw
in every component it feeds. Reported three ways so the effects separate (ordinary →
fixed-stratum-independent isolates conditioning; → shared-draw isolates covariance). Under full
duplication the dependence-blind interval is 1.43× too narrow.

### 4.4 Registration deliverables — two done, one outstanding

1. **`m8src/decide.py`** — the executable confirmatory rule and the complete ship predicate:
   draws, seed, stratified paired resampling, strict qid alignment, Holm ordering and ties, the
   exact α/3 bound, unrounded endpoints, and every §5 guard as a literal inequality. Its
   `self_test()` runs the whole rule end to end on synthetic data, so the registration is
   demonstrably executable before any real number exists.
2. **`m8src/power.py` → `results/m8_power.json`** — the joint power simulation of the FULL ship
   rule. Headline numbers, calibrated from real paired per-query vectors:

   | quantity | value |
   |---|---|
   | reserved-4 equal-weight macro SE (near-sibling reference class) | **0.00209** |
   | 95% half-width | **0.0041** (the plan's prior estimate was 0.005 — agreement) |
   | **MDE** at power 0.8 (the binding leg is the α/3 bound, z = 2.394) | **0.0068** |
   | P(ship) — structural target δ=0.020 | **0.84** |
   | P(ship) — modest δ=0.010 | **0.80** |
   | P(ship) — recipe-only δ=0.005 | **0.21** |
   | P(ship) — M7 repeat δ=0.000 | **0.002** |
   | P(ship) — dense lags fused (C2 binds) | **0.57** |

   Calibration extrapolates each reserved set's per-query variance from its nearest dev analogue
   (fever←hotpotqa, dbpedia←nq-250k, android←programmers, english←physics), because the reserved
   sets have never been scored. **DBpedia is the weak link**: n=400 against its analogue's 3,452.
   Sensitivity is reported at ±25% / −20% on the calibrated sd. **The P(ship) figures go to Dylan
   before Phase 0 spends its week** — a knowing report-only choice beats a discovered one.
3. **`m8src/rule_audit.py`** — ported 2026-08-29, and M8-specific rather than a copy. M7's version
   checked M7's rules; M8's strongest check is one M7 did not have: for every stamped result, it
   fetches the registry blob **from git at that result's own commit** and diffs the probe's bar,
   endpoint, comparator, multiplicity and no-survivor against today's. **A registration that moved
   after a number existed is a BLOCKER**, in either direction. It also verifies the stamped commit
   is an ancestor of HEAD, checks registry hygiene, and checks that this gap list is still true —
   which is what caught the gap list naming two files that had already landed. Four things it
   cannot check are listed as unverifiable rather than reported as passes.

**GAP LIST — obligations this ledger states that are NOT yet implemented.** *(`m8src/test_decide.py` and
`m8src/rule_audit.py` landed 2026-08-29 and are struck from this list.)* Written here because
a protocol document asserting guards that do not exist is the same failure class as code producing
a wrong number, pointed the other way: a future session (or the owner reading GitHub) trusts a
"DONE" heading. Each line is a blocker for the stage named.

| missing | what it must do | blocks |
|---|---|---|
| `m8src/final_run.py` + `m8src/test_final_guard.py` | the one-shot access path and the 14-line checklist in §6, each with an acceptance test. **The test cannot precede the module** — there is nothing yet to test, and writing acceptance tests against a module that does not exist is how a suite ends up asserting its author's intentions. | the access |
| `m8src/freeze.py` + `m8src/test_freeze_binding.py` | the freeze path and the refusals it must make on M8 paths. Same dependency: **the module first.** M7's `freeze.py` is 34,659 bytes of accumulated refusals; the port is a real job, not a rename. | the freeze |
| B-leg noise floor | a null pair varying the B leg, for arms that restructure it (R-PHASE and any pool/init change flowing through B) | those probes' bars |

`./run_m8_tests.sh` names the unported suites in its output for the same reason.

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
recipe you happen to pick, which does **not** deflate a frozen confirmatory comparison. Replay
noise is ~5e-6. **Dev-reuse counter runs from evaluation #1** → `results/m8_dev_reuse_count.json`.

### 4.7 Noise floor (measured before any bar is frozen) — FORMULA REGISTERED

*(Codex gate MAJOR 2: v1's "±10% A-steps" is a treatment change, not a null, and "the measured
floor" did not say which statistic.)*

- **The floor is measured from TRUE NULLS ONLY**: `K = 3` replicates of the *identical* recipe
  differing only in **training seed**, on every endpoint a bar will read.
- **Floor statistic**: `floor(endpoint) = max over the 3 pairwise |Δ|` on that endpoint. Using the
  max, not the mean, because a bar must survive the unlucky pair.
- **Bar**: `bar(endpoint) = max(planning_minimum, 2 × floor(endpoint))`, where `planning_minimum`
  is whatever the probe's own registration names (never lower than 0.0040, the smallest effect
  this project has ever adopted).
- **Multi-endpoint probes** take the max over their endpoints' bars.
- **The ±10% A-steps perturbation is still run and published**, but as a **recipe-sensitivity**
  number reported beside the floor — never as the floor itself. It is the same quantity as M7's
  0.0027–0.0078 perturbation band.
- Published to `results/m8_noise_floor.json`. Bars marked `TBD-noise-floor` in `m8/registry.json`
  are frozen by a §15 amendment once it exists. **No bar, no run** — enforced by
  `m8src/probe_guard.py` (G1), not by prose.

**MEASURED, 2026-08-29** (`results/m8_noise_floor.json`; five arms, `m8nf-*`, one process each).
Three arms differing ONLY in training seed, scored on the full pinned dev suite through the
released `QueryTable` path, at both precisions and both pooling rules:

| endpoint | floor (max pairwise \|Δ\| over 3 seeds) | bar = max(0.0040, 2×floor) |
|---|---|---|
| group-vector median | 0.00095 – 0.00190 | **0.0040** |
| worst group | 0.00162 – 0.00227 | **0.0040**, except fp16·mean where 2×floor = **0.00454** |
| out-of-domain macro | 0.00162 – 0.00227 | as above |
| all-component macro | 0.00102 – 0.00141 | **0.0040** |

**The planning minimum binds nearly everywhere**: seed-to-seed variation is smaller than the
smallest effect this project has ever adopted, so 0.0040 is the operative bar on every dense
endpoint but one. That is the outcome that keeps the wave-1 and wave-2 probes runnable; had the
floor come in at 0.003, every bar would have doubled and most levers would have been unresolvable
by construction.

**Recipe sensitivity, reported BESIDE the floor and never as it** (§4.7): the ±10% A-step arms
move the all-component macro by **−0.0009 and +0.0015** (span 0.0024) — at the low end of M7's
0.0027–0.0078 recipe-perturbation band, as a smaller perturbation should be. The two numbers
measure different things and both stay on the record.

**ONE replication, seen through three rank-invariant lenses — corrected 2026-08-29.** The seed-0
arm reproduces M7's shipped candidate's dev proxy macro to all sixteen digits
(0.5105689103506673), its full-suite macro at `sqrt` (0.6153), and its fused macro (0.57266,
§4.7b). The first version of this file called that "two exact replications" and said the harness
"reproduced a 2,500-step run exactly". **Both claims were wrong**, and a byte comparison shows it:
`m8nf-seed0.npz` and `p35w-2m-s2500.npz` **differ** — 0.066% of `rows_fp16` elements, max
|Δ| = 4.88e-4 (one fp16 ULP), and `rows_int8` and `token_weights` differ too. M7's stated
non-bit-identical GPU reductions held exactly as M7 said. What the three matching numbers actually
demonstrate is that **nDCG is rank-based and quantizes ~1e-7 of weight noise away**: no ranking on
the dev suite flipped. That is a real and useful fact — the floor's frame is the released
artifact's frame for every purpose a bar reads — but it is ONE replay observed three ways, not
three independent validations, and the mechanism is metric insensitivity, not bitwise determinism.

**What this floor does NOT cover** (§4.4 gap list): it holds the B checkpoint fixed, so an arm that
restructures the B leg (R-PHASE, and any pool or init change flowing through B) has a larger floor
that is not yet measured. **A B-leg null pair remains the one outstanding floor.**

### 4.7b The FUSED floor — MEASURED 2026-08-29

`results/m8_noise_floor_fused.json` (`m8src/fused_floor.py`). B3, B13, R1-ASSEMBLY, D-SYNTH and
D-FINEWEB register "dense AND fused" endpoints, so without this their bars were not computable as
registered and `probe_guard` refused them. The frozen `convex0` w=0.8 operator is APPLIED, never
re-fitted — re-fitting per arm would measure the floor of a fitting procedure, a different and
much larger quantity, and would let each arm choose its own operator (§7).

| endpoint | floor (max pairwise \|Δ\| over 3 seeds) | bar |
|---|---|---|
| fused macro, int8 · sqrt | **0.00066** | **0.0040** |
| fused macro, int8 · mean | 0.00059 | 0.0040 |
| fused macro, fp16 · sqrt | 0.00060 | 0.0040 |
| fused macro, fp16 · mean | 0.00066 | 0.0040 |

**The fused floor is tighter than the dense one** (0.0006 against 0.0010–0.0023), which fusing
with a deterministic BM25 run should do: part of the fused score comes from a component with no
seed at all, so seed variation is damped. **Two corrections to how that was first stated**
(2026-08-29 review): `convex0` at w = 0.8 weights BM25 at **20%**, not "half"
(`m7src/fusion.py`); and the fused macro is over **four** components while the dense endpoints are
over **six**, so "three times tighter" compares two different estimands and is not a like-for-like
ratio. The directional point stands and is the one that matters: **a fused endpoint resolves
smaller effects than a dense one, so a lever that clears the fused bar and not the dense one has
NOT shown a table improvement.**

**A third exact replication.** The seed-0 arm served at `sqrt` gives a fused dev macro of
**0.57266**, against the frozen fusion spec's own recorded `dev_macro` of 0.5726634997854769
(`results/m7_fusion_p35w-2m-s2500.json`). The floor's frame reproduces the released system on the
dense proxy, on the full dense suite, and now under fusion.

Components: `nq-250k`, `hotpotqa`, `cqadup-programmers`, `cqadup-physics` — the frozen spec's own
four. The two held-out dev slices carry pool row indices rather than document text, so BM25, and
therefore any fused read, does not exist for them by construction.
- **Exempt from noise calibration, named explicitly**: purely descriptive diagnostics that adopt
  nothing (B2, B16, `retention_decomp`) and arithmetic/feasibility gates (B7's memory curve, the
  ONNX export precondition). Everything else is calibrated.

---

## 5. The ship rule — one literal predicate

Implemented in `m8src/decide.py::ship()`. **Ship iff all seven conditions hold.** Every threshold
below is a number, not a word (Codex gate BLOCKER 1).

1. **C1 resolved** (all three legs, §4.2).
2. **C2 resolved** (all three legs; table-vs-table, D1 disabled).
3. **C3 resolved** (all three legs).
4. **Qualifying v2 table.** Mechanical, from the immutable manifest's declared config diff against
   R0, keyed on **config keys, not labels**. **The FULL key space is enumerated in
   `m8/registry.json` — all 35 `train.Cfg` fields plus the artifact-level fields a release
   records — and classified NOW, while the amendment is legal.** An unclassified key fails by
   design, but leaving foreseeable keys unclassified would turn the condition into a coin decided
   by naming conventions at manifest time, which is the opposite of a pre-registration.
   Four classes: `qualifying_table` (27 keys), `qualifying_non_table` (`doc_side_head` alone),
   `not_qualifying` (23 keys), and `neutral` — frame descriptors like `teacher_id`, `dim`,
   `precision`, `fusion_param`, which are neither a lever nor a disqualifier.

   **The teacher-swap side effect, registered before any swap exists.** A swap flips `teacher_id`
   (neutral) and, if the tokenizer changes, `tokenizer_id` and `vocab` — which *are*
   qualifying-table keys. That is a consequence of the swap, not an E11-sense v2 lever. So
   whenever a teacher swap appears in the diff, **`tokenizer_id` and `vocab` do not count toward
   condition 4**: the manifest must declare at least one other qualifying-table key. Without this,
   swapping the teacher would have satisfied the qualifying-table requirement by itself, and the
   registered swap branch of C2 (§4.2) would have been a release path with no lever in it.
   - **THE LISTS BELOW ARE A PARTIAL RENDER. `m8/registry.json` is the authority** (§9), and it
     classifies **27** qualifying-table keys, **1** qualifying-non-table, **23** not-qualifying and
     **9** neutral. *Corrected 2026-08-29 in the milestone audit: this prose named seven qualifying
     keys and read as exhaustive, which understates the condition — a session reading §5 alone
     would wrongly conclude the recipe/data class cannot satisfy condition 4. It can; what it
     cannot do is carry the bar (§7).* Read the registry before writing a manifest.
   - **QUALIFYING_TABLE keys** — at least one must appear in the diff. Seven of the twenty-seven,
     as illustration only: `objective_family`, `tokenizer_id`, `vocab`, `row_init_construction`,
     `pool_composition`, `feature_set`, `structural_rider`. The registry also classifies
     `ict_fraction`, `sources`, `phase_structure`, `pool_mode`, `init`, `learned_weights`,
     `low_rank_delta`, `doc_instruction`, `ngram_rows`, `synthetic_query_dose`, `genre_bundle`
     and others as qualifying.
   - **NOT_QUALIFYING keys** (may appear, never sufficient). Illustrative subset:
     `seed`, `steps_a`, `steps_b`, `temperature`, `hard_neg_k`, `lr`, `b_pseudo_queries`,
     `batch_size`, and any key matching `*_tuning`.
   - **AMENDED 2026-08-29 (Dylan): a DOCUMENT-SIDE win now satisfies condition 4.** *"M8 can ship a
     better system. If that system makes sense and is defensible. I'm okay having a custom document
     encoder if this works well."* The rule that `doc_side_head` alone could not open the release
     path (E11 + G4-4) is **superseded**; condition 4 takes one key from `qualifying_table_keys`
     **or** `qualifying_system_keys`. **What still bounds a v2** (registry
     `invariants_that_survive_the_amendment`): the query side stays a **pure lookup table** (E1,
     non-negotiable); the document tower must be **derived from stella** — LoRA/adapter/last-block/
     head, never trained from scratch (*"I don't want to go all the way to train a dense embedding
     model from scratch"*); the win must clear C1/C2/C3 with its cost rows; and **the report must
     decompose the win**, never presenting a document-side gain as the table having improved.
   - **Any key in the diff that is on neither list ⇒ the condition FAILS.** Classification happens
     at manifest time, before the access; a key cannot be argued into a category after a number.
   - A distinct int8 payload is **necessary but not sufficient**.
5. **Point guard on C1**: `delta_raw(C1) > 0.005`, strict, unrounded. A product margin, not a
   hypothesis — a resolved win too small to be worth a version bump is not a v2.
6. **Worst-group guard**: **groups = the four reserved datasets individually**, no grouping choice
   left open. `min over the four of (mean_ds(M8) − mean_ds(fused-M7)) ≥ −0.010`, point estimates,
   unrounded. *Known cost, stated rather than discovered:* DBpedia has n = 400, so at truly-equal
   performance this guard vetoes about **5%** of the time on DBpedia alone. That is accepted:
   for a release guard, rejecting a real improvement is the safe error and accepting a real
   regression is not — the same stance M7 took on lever #7's guardrail, and it is not loosened
   after seeing whether it bites.
7. **Six-set no-regression guard**: `macro_six(M8) − macro_six(frozen M7) ≥ −0.0075`.
   - **Margin provenance, measured (not chosen).** The equal-weight six-macro paired SE for a
     *near-sibling* system pair, computed from the frozen comparator vectors, is **0.0026–0.0032**
     (leaf-ir-asym vs mdbr-leaf-ir 0.00263; LR-dense-pertask vs LR-dense-websearch 0.00316; the
     variance is dominated by trec-covid's n = 50). **0.0075 ≈ 2.4 × SE**, giving
     P(false veto | truly equal) ≈ 0.99 and ~80% power against a true regression of ≈0.0135. A
     memorization signature — the thing this guard exists to catch — would show a six-set drop far
     larger than that.
   - **WHEN the six are scored, registered (Codex gate BLOCKER 1).** M8's six-set vectors do not
     exist yet, so "frozen vectors, zero new access" is only true if the timing is fixed. It is:
     the six are scored **after the immutable candidate manifest and after the freeze, inside the
     same guarded process as the confirmatory access, before the reserved payloads are opened**,
     and the result is written atomically. It is logged as a known-test access in
     `m8/SIX_ACCESS.log`. No model, fusion parameter or fallback may change after it.
   - One-directional: it can only block, never license.
   - **AND IT IS NOT A DECISION POINT (self-review, 2026-08-29, before any M8 number).** The six
     are scored *before* the reserved payloads are opened, in the same process, which creates an
     obvious temptation: see a six-set regression, stop, "fix" the candidate, re-run. The freeze
     receipt does not protect against that, because the six-set read does not spend the reserved
     access. So it is registered here instead: **the six-set result is computed, recorded, and the
     reserved access proceeds regardless of what it says.** No candidate, fusion parameter,
     fallback or artifact may change after it, and no run may be abandoned because of it. It
     governs SHIPPING, not MEASURING — and the measurement is what this milestone exists to
     produce, since a miss is the pre-registered publishable outcome. The only thing the guard
     does is decide whether a v2 replaces v1 on the hub.

**Descriptive context inside the same access (E12, registered before any M8 number):**
`bge-small-en-v1.5` and `LR-dense-websearch` scored on the reserved four — **outside the Holm
family, no ship consequence**, the external anchor for the report.

**FEVER handling (E9):** FEVER rows carry the proxy-provenance caveat; all legs are additionally
reported FEVER-excluded as a **descriptive sensitivity read with no ship consequence** (§4.2); the
cancellation argument is stated **conditional on a stella-lineage teacher** and is void if the
teacher leaves that lineage.

**A miss is a publishable outcome**, written down before the number exists and inherited from M7's
REPORT FRAMING. Nothing about the system may change after the reserved-set numbers are seen.

---


## 6–8. Pipeline, Stage R, Stage S — REMOVED 2026-08-30 (binding rules restored in §6 below)

**Cross-reference note.** Surviving text still cites "§6", "§7" and "§8". Every rule those citations
depended on is restored in **§6.1–6.5** below; what was removed is the M8 stage plan, the Stage-R
degree-of-freedom table and the Stage-S menu, which describe work that will not happen. A citation
you cannot resolve in §6 refers to removed planning, not to a lost rule — and the diff that proves
it is `research/m8-planning/` plus this file's own git history.

M8 assembled no candidate, so the binding stage ordering, the Stage-R degrees of freedom
and the Stage-S menu describe work that will not happen. The prose probe registrations are
superseded by **`m8/registry.json`**, which is machine-readable, is what `probe_guard`
actually reads, and retains every row including those that never ran. The one finding
those sections carried is in `m8/EXPLORED.md`: **the recipe/data class is hygiene, not a
route to the bar** — M7's entire post-gate lever programme transferred to the six at
0.000 ± 0.005.

## 6. RULES THAT SURVIVE THE COMPRESSION — restored 2026-08-30

*Sections 6–8 were removed as M8-specific planning. An adversarial diff (Codex, 2026-08-30) found
that they also carried BINDING rules with no other home — including the one-access safety
checklist, while the confirmatory access is still UNSPENT and inherited by M9. Those rules are
restored here VERBATIM. This is the incident the §14 G10 rule exists to prevent, caught by the
mandated diff rather than by luck.*

### 6.1 Inheritance and naming

**NAMING DISCREPANCY — RULED 2026-09-03 (Dylan): the milestone suffix is dropped.** The released
names are **`constella-zero`** and **`constella-nano`**; the org is Dylan's own account for the
PoC releases (Amendment A), not `qdrant/`. The `-m8` suffix was wrong because the zero artifact is
M7's. **Not honoured at the first push**: `zero` went out as `DylanCouzon/zero-query-encoder-v1`
without this ruling being sought, which the paragraph below had required before anything shipped.
Rename tracked in `m11/PLANNING.md` §T6.


**Reading order for a cold session:** `m8/STATUS.md` → this file → `m8/registry.json`.
`m7/LEDGER.md` is the inherited protocol and is authoritative for anything this file does not
override. `m7/CODEMAP.md` is the module map and pitfall list — read it before writing code.

**Release names, LOCKED by Dylan:** **`qdrant/constella-zero-m8`** (this milestone's table) and
**`qdrant/constella-nano-m9`** (M9's distilled tower). Constella = constellation + stella:
navigate by fixed stars, no engine. **If the teacher leaves the stella lineage, naming reopens
with Dylan.**

### 6.2 The one-access safety checklist — BINDING, and the access is UNSPENT

**M9 inherits the reserved four and their single access. Every item below must be ported to M9's
paths with its own acceptance test before that access is spent.** M8 never built
`test_final_guard.py` / `test_freeze_binding.py` (§4.4 gap list); this is the specification they
must satisfy, not a record that they do.


| # | obligation |
|---|---|
| 1 | Durable **spent receipt**: an annotated `m8-reserved-spent` tag pushed to origin the moment the confirmatory block is on disk. |
| 2 | **Peeled** tag check (`refs/tags/X^{}`) — an annotated tag resolves to the tag OBJECT and would otherwise never match. |
| 3 | The access counts as SPENT when the **result file** holds the confirmatory block, not when the ledger says so. |
| 4 | **Exclusive pid lock**; two concurrent launches cannot both pass a read-only guard. |
| 5 | Guard runs **before** preflight; preflight takes the kinds it may verify and logs its own payload-hash reads. |
| 6 | Table **snapshotted** after verification, not reopened by path per dataset. |
| 7 | Confirmatory result written **atomically**, before any secondary block. |
| 8 | Per-query values persisted as **raw floats** (rounding to 1e-6 made a close decision unreproducible). |
| 9 | Corpora load **corpus-only**; labels come from the frozen payloads and nowhere else. |
| 10 | **BM25 package/config verification** against the fusion spec's own cache keys — a `bm25s`/`PyStemmer` upgrade between freeze and access would silently change the fused system C3 judges. |
| 11 | `load_and_verify` **re-runs** `assert_releasable` and `assert_gate_passed` rather than trusting recorded verdicts. |
| 12 | `assert_gate_passed` requires the exact registered condition set, the pinned component list, a real Stage-0 checkpoint, the per-query dump's bytes, and a clean committed **evaluator identity**. |
| 13 | Exactly **one** `--infra-retry`, requiring the same commit. |

### 6.3 Stage-R invariants that still bind

**R0 :=** the registered M7 recipe settings instantiated under the selected teacher, current
filters, and M8's data volume / precision / seed policy.

direction. Diagnostics may only trigger **separately registered** performance arms — and a
diagnostic may not trigger an arm that does not yet have a complete registry row.


### 6.4 The frozen fusion contract

**Fusion operator** — family grid, depth, dev components, frozen `bm25_run` — is frozen **before**
Stage R and applied identically at every fused read; the final invocation instantiates parameters
only. Amendable only if D4' BM25F re-registers the lexical function, and only before Stage R.
Inherited mechanics: one family, one parameter, no per-dataset weights or routing,
`fusion.DEPTH` = 1000 for selection and application alike, fitted against the **int8 release**
artifact; the zero-score-padding and self-hit drops are part of the frozen function; on an exact
tie the **simpler** system wins (dense-only first, then the first grid point in order);
`n_tied_at_best` written into the spec.

### 6.5 Release-format, seeds, and the standing kill-list

→ the candidate is R1-alone; multiple → the rule picks.**

**Release-format rule: int8, always.** It is the C2 identity, the proven-quality-free format
(M7 G4 int8 upper bound 0.00013 against a 0.005 bar), and what the ONNX graph embeds. All sizing,
eligibility and tie-breaks are computed at int8. A 4-bit variant is **research-only and never
ships**.

identically-parameterized aligned tables, otherwise mechanical median. **Never best-seed.**


  **before** Stage R. **May never be the sole qualifying change.** Full-dose dual-index question
  expansion stays dead on compute (263–702 days); a bounded ≤50K-doc research probe may run and
  may **never** be extrapolated to the reserved system.
- **R1-only** — legitimate **iff** at least one qualifying table change survived (§5.4).

**Research-only, never candidates:** D3 index-time adaptation (E5 — one labelled measurement
allowed AFTER the final access); D5 nonlinear query-side head (E1 — out entirely).

**Kill-list, standing (algebra and arithmetic, not preference):** higher table dims
(identity-linear MRL heads off a 1024-d hidden state); absorbable transforms as capacity —
query-side centering, whitening, top-PC removal, any per-token scalar weight — allowed only as
registered training priors and killed on cross-domain validation; full late interaction; another
31 MB unigram table with better hyperparameters as the sole change.

**Considered and recorded, not adopted (so it is not rediscovered as an oversight):** doc-side
small-k multi-vector (k = 2–4 facets, max-over-facets). It is not the kill-list's token-level late
interaction, the query side is untouched, and it attacks the short-query ambiguity floor nothing
else on the menu reaches — but it multiplies the document index by k, which is an E7/product
conversation Dylan has not had. It is in the wake-up note beside E14, not in the menu.

- **B12** — superseded by the int8-always rule; a 4-bit sweep may run post-finalist as research.

## 9. Probe index — `registry.json` is authoritative

The prose registrations were removed 2026-08-30; **`m8/registry.json` is the pre-registration record** and is what `probe_guard` reads. Every row is kept, including those that never ran — deleting them would hide what was committed to in advance, which is the one thing a pre-registration exists to prevent. This index exists so no registry row is orphaned from the ledger.

| probe | question | outcome |
|---|---|---|
| `S0` | LoTTE overlap screen vs the protected partitions | ran |
| `T1` | teacher screen (workstream T) | ran |
| `B2` | candidate-set entropy (is the KL term degenerate?) | ran |
| `B3` | real-pair pool scaling at fixed compute (is Phase A pair-starved?) | ran |
| `B7` | block-CG vocabulary curve (gates D2) | ran |
| `B6-pre` | doc-side head ONNX fuse precondition (gates D1) | ran |
| `B17` | in-domain oracle generalization ceiling | ran |
| `B8` | target design (bare + doc-centroid targets) | ran |
| `B9` | SVD rank truncation | registered, not run |
| `B10` | scoring/pooling family | registered, not run |
| `B13` | joint configuration confirm (A-grid + matched-steps negatives + ri | registered, not run |
| `B14` | doc-side instruction refit | registered, not run |
| `B15` | context-averaged row init (Wada) | registered, not run |
| `B16` | MEV / self-similarity | registered, not run |
| `B6` | doc-side map quality arm (D1) | registered, not run |
| `D-GENRE` | genre bundle (USPTO + EUR-Lex + federal) | registered, not run |
| `D-SYNTH` | synthetic Qwen3 query component (E2) | registered, not run |
| `D-FINEWEB` | FineWeb arm (E13, clean-stack-tax design) | registered, not run |
| `R-LIST` | listwise distillation arm | registered, not run |
| `R-PHASE` | phase structure (sequential / mixed-replay / listwise-only) | registered, not run |
| `R1-ASSEMBLY` | the one Stage-R validation gate | registered, not run |
| `S-SELECT` | Stage-S family selection | registered, not run |
| `NF` | noise-floor measurement (the instrument every bar is set against) | ran |
| `B3-ICT` | ICT (Inverse Cloze Task) pair augmentation | registered, not run |
| `E14-HEAD` | doc-side re-shaping: RENORMALIZED heads on cached document vectors | ran |
| `E14-LORA` | doc-side co-adaptation proper: LoRA the document tower jointly wit | registered, not run |
| `E10-REMEDY` | the LoTTE shadow, remediated per-question and re-screened to zero | prior remedy RAN and was INVALIDATED; corrected remedy never ran |
| `D2` | D2 -- compositional capacity: a multi-word tokenizer, trained thro | NOT run — `D2-PRE` routed against authorising its five chains |
| `NF-CROSSED-FUSED` | NF-CROSSED-FUSED -- the chain-level FUSED floor, from cells alread | not run — no chain-level fused floor exists |
| `D2-PRE` | D2-PRE -- closed-form preflight: does ANY new-row class carry resi | ran |
| `VECTOR-PRF` | VECTOR-PRF -- train-free dense pseudo-relevance feedback on the fr | ran |
| `E14-PRE` | E14-PRE -- does the document tower hold reachable structure its re | registered, launched, CANCELLED mid-run |

## 10. Workstream T — the teacher

**Executable swap rule (Codex gate BLOCKER 4). Probe id `T1`, governed by G1.** Every degree of
freedom below was a word in v1 and is a value here.

- **Frame, clarified (§15, 2026-08-29).** "Fixed student frame" is fixed **within a tokenizer
  family**. A challenger with a different tokenizer is screened in its own natural frame — its
  tokenizer, its vocabulary — because that is the frame it would actually ship in. The comparison
  is then explicitly a **teacher-plus-tokenizer** comparison, not a teacher comparison, and the
  vocabulary size is reported at every row so the confound is visible rather than hidden. Any
  swap argued on such a screen must say which of the two factors it is buying.
- **Frame.** Within a family: the closed-form distilled table
  (`scripts/teacher_learnability.py` + `learnability_report.py`), fit list **regenerated through
  the current protected-query filter** (M7's list had 1.31% R1 hits and is unusable), λ selected
  on the same interior grid for every candidate, identical bag matrix.
- **Components, named:** `cqadup-programmers` and `cqadup-physics`, equal weight. These two and no
  others — they are the dev suite's only out-of-domain members and the criterion M7 adopted.
- **Statistic:** dependence-preserving paired bootstrap (`boot.paired_dep`, B = 10,000, seed 0) +
  `boot.signflip_dep`, on per-query nDCG@10, **strict alignment**. "CI-resolved" means the **raw**
  two-sided 95% CI excludes 0 **and** `signflip_dep` p < 0.05.
- **The incumbent is re-probed in the identical frame**, mandatory — a screen without the
  incumbent is not a paired comparison.
- **Multiplicity:** Holm across the challenger set at α = 0.05.

**SWAP BAR — all four, none waivable by a session:**

1. The challenger's closed-form table beats the incumbent's, **CI-resolved** as defined above.
   *(The criterion is the table, not the tower: Spearman(symmetric ceiling, distilled table) =
   0.000 over eight candidates, and arctic-embed-l, approved on the ceiling, produced a table
   0.0480 BELOW the incumbent's.)*
2. **The point-estimate margin exceeds the swap's CI-widening penalty**, where the penalty is
   **fixed per challenger BEFORE its result is read**, from a rule that leaves no classification
   choice: a challenger is **near-sibling** iff it shares the incumbent's tokenizer identity
   (`bert-wordpiece-30522`) AND its dimension, else **dissimilar**. Penalty = **0.0050**
   (near-sibling) or **0.0096** (dissimilar), from `results/m8_power.json`. The classification of
   every candidate in §10's list is written into `m8/registry.json` now, before any screen runs.
3. **Off-family condition, exact:** on `nq-250k` and `hotpotqa`, the challenger's per-component
   point-estimate delta versus the incumbent must be **≥ 0 on BOTH components**. One negative
   component is a reversal and fails the condition — not "the pooled macro is positive".
4. **Dylan signs off.**

**Tie-break interval, exact:** two candidates are tied iff the raw 95% CI of their difference lies
entirely within **±0.0040** (the project's practical-equivalence band). Among tied candidates:
prefer no disclosed overlap with the protected sets, then the smaller dimension.

**Candidates, established from primary sources 2026-08-29**
(`research/m8-planning/challenger-specs-2026-08-29.md`). Every challenger is **dissimilar** —
none shares the incumbent's `bert-wordpiece-30522` tokenizer — so every penalty is **0.0096**,
fixed here before any screen runs.

| candidate | vocab × dim | int8 MB | trust_remote_code | notes |
|---|---|---|---|---|
| `stella_en_400M_v5` (incumbent control) | 30,522 × 1024 | 31.3 | yes, same-repo | the frame everything is compared against |
| `ibm-granite/granite-embedding-english-r2` | 50,368 × 768 | 38.7 | **no** (native ModernBert) | headroom either precision |
| `Alibaba-NLP/gte-modernbert-base` | 50,368 × 768 | 38.7 | **no** (native ModernBert) | **byte-identical tokenizer to granite-r2**, so the two share a bag matrix and are in one frame |
| `NovaSearch/stella_en_1.5B_v5` | 151,646 × 1024 | 155.3 | yes, **same-repo** (revision pins the code) | fits the 233 MB cap at int8; **busts it at fp16** (310.6 MB) — an int8-only survivor |
| `microsoft/harrier-oss-v1-0.6b` | 151,669 × 1024 | 155.3 | no (native Qwen3) | see the three blockers below |

**Arithmetic verdict: none of the four busts the 233 MB int8 cap.** The two Qwen-family
candidates are int8-only survivors — at fp16 they would be 310.6 MB — which is consistent with
the int8-always release rule (§8) but worth stating, because it removes a fallback.

**harrier carries THREE separate blockers, only one of which was known:**
1. **Training data undisclosed** — a contamination black box against four hash-pinned reserved
   sets. **Needs Dylan's ruling before adoption**; the session does not make it. *(known)*
2. **It uses LAST-TOKEN pooling**, and `m7src/teacher.py`'s `pool_project_normalize` implements
   only `cls` and `mean` and RAISES on anything else. Screening it needs real new code, not a new
   `Spec` row — and `m7src` is frozen for M8 (G3), so that code would have to live in `m8src`.
   *(new, 2026-08-29)*
3. **No published retrieval-only number** — only a mixed-task MTEB v2 overall of 69.0. There is no
   BEIR or MTEB-Retrieval figure to sanity-check a screen result against. *(new)*
   Its vendor position is **Microsoft**, which the relaxed vendor rule places in
   "OK WITH JUSTIFICATION" (Azure AI Search is one service among hundreds) — so it is not
   disqualified on vendor grounds, and that is worth knowing before spending the ruling.

**And a fallback-row problem the plan had not seen.** `QueryTable`'s degenerate-empty-query
fallback needs a sequence-start row. **Neither Qwen-family candidate has a usable one**:
stella-1.5B's `config.json` and `tokenizer_config.json` disagree with each other
(`bos_token_id` 151643 versus `bos_token: null`), and harrier's entire embedding lives at the LAST
token, so a start-token fallback is close to meaningless there. Any screen of those two must
register what the fallback row IS before it runs — `m8src/teacher_screen.py` passes
`spec.cls_id` explicitly rather than inheriting `table.py`'s BERT default, which makes the choice
visible instead of silently wrong.

**A padding-gap reminder, larger than before:** harrier's true tokenizer size is **151,669** while
its `config.json` reports `vocab_size` 151,936 — a 267-row gap. `Spec.vocab` is the TOKENIZER
size; taking the config figure would ship 267 dead rows. The existing gte-large/stella-400M case
was six rows.

**Costs a swap charges, written down so they cannot later be discovered as reasons to avoid it:**
**it converts E11's strict table claim into a system claim** — C2's table-vs-table form is
unsatisfiable across two document spaces (§4.2), which is a protocol cost, not a compute one;
double reserved-4 pre-encode (20.7 GB and tens of hours *per system*); the FEVER-cancellation
argument is lost if the teacher leaves the stella lineage (E9); the release NAME reopens with
Dylan; WordPiece/fingerprint rebuild; re-encode of the 6.17M-doc pool, dev corpora and TRAIN
targets; fusion re-selection; gate re-run; and **every noise floor and bar measured in the
incumbent frame is invalidated** (§6 step 5). **Same-teacher is the registered default.**

**ONNX feasibility evidence is assessed for every finalist BEFORE the freeze** — a successful
local export or clear family precedent. **Absence of a published artifact is not failure**;
demonstrated infeasibility is the only ONNX-based exclusion.

**Reusable arithmetic bound (inherited):** base out-approximates large in every family by +0.04 to
+0.07, so a family whose *large* variant scores below ~0.28 on the table criterion cannot reach
stella by shrinking.

**Second-machine protocol** (`m7/LEDGER.md`) stays in force if any probe runs on Dylan's Mac.

---

## 11. ONNX / fastembed (scope approved)

Feasibility verified: `research/m8-planning/onnx-feasibility-2026-08-29.md` — stella's export
blocker is two config flags (`unpad_inputs`, `use_memory_efficient_attention`), and
gte-large-en-v1.5 (same architecture) ships first-party ONNX. Verdict: days, not weeks.

1. **`constella-zero` ships AS an ONNX graph** — Gather → **the selected frozen pooling
   operator** → normalize, int8 initializer — fastembed-native, with the BM25-bespoke-class
   fallback.
   **The graph is defined by the SELECTED operator, not hardcoded to `sqrt` (Codex gate
   BLOCKER 9).** v1 fixed the graph as sqrt while B10 was free to adopt sum/max/top-k/LSE, which
   would have let one function be evaluated and another exported. Registered instead: **any B10
   alternative must demonstrate ONNX export and pass the §11.4 parity fixtures as an adoption
   PRECONDITION**; an operator that cannot be exported and verified is not adoptable, and the
   default remains M7's `sqrt`.
2. A **parity-verified export of the SELECTED teacher** is an M8 task; M10 is the fallback landing
   zone. Demonstrated infeasibility (not absence of an artifact) is the only ONNX-based teacher
   exclusion.
3. **D1, if it survives, ships fused into the doc graph — one file** (E3's condition).
4. **Parity runs on the final aggregated int8 artifacts BEFORE the shadow crossing**: full
   conformance fixture suite, vector and cosine tolerances, top-k tie policy, an nDCG delta bound,
   all pinned in the manifest.
5. Index-side tooling is **offline, not served**.

---

## 12. Rulings (Dylan, 2026-08-28/29 — authoritative; prior wording superseded)

| # | ruling |
|---|---|
| **E1** | **Pure lookup is the product.** No query-side neural head in M8, not even as research — *"if people have some compute capability, there's no reason to not use M9"*. |
| **E2** | **Synthetic Qwen3 training queries approved** — *"green light if no downsides"*; downsides recorded at §3. |
| **E3** | **Doc-side head approved CONDITIONALLY**: must fuse into the doc ONNX graph as plain nodes — one served file, no custom pipeline — and clear its probe. |
| **E4** | **Reserve BOTH M9 sets** (EUR-Lex + USPTO), frozen construction procedure, never scored in M8. |
| **E5** | **Index-time adaptation: research-only, end of project** — *"seems over engineered… afraid of the accusations"*. Never in the confirmatory candidate. |
| **E6** | **Training-only second teacher allowed if licence-clean**; the vendor rule binds *shipped* components; documented on the model card. |
| **E7** | **Byte cap 233 MB int8** — *"storage can be fairly cheap"*; cold-start, latency and optics matter more than bytes. |
| **E8** | **PMC-OA excluded** (delegated *"include if it moves the needle, otherwise exclude"* → excluded: its unique value is duplicated by cleaner sources; the cost is the NFCorpus/TREC-COVID honesty read). |
| **E9** | **FEVER: label + sensitivity read** — proxy-provenance caveat at the rows; all legs also reported FEVER-excluded; cancellation stated conditional on a stella-lineage teacher. |
| **E10** | **LoTTE adopted as the mandatory shadow** under the written *"not literally CQADupStack"* reading — pending the overlap measurement (§2.3); STOP-on-failure. |
| **E11** | **STRICT C2** — *"we want something that looks good on benchmarks too; hybrid should be the default but isn't to everyone"* — the dense table must beat M7's dense table; the fused-objective lever stays consciously excluded. |
| **E12** | **Comparators inside the access: YES — bge-small + LR-dense-websearch**, descriptive only, outside the Holm family, registered before any M8 number. |
| **E13** | **FineWeb (`Qdrant/FineWeb-10B`): measure first, ship-decide later.** The affirmative-licence standard stays in force for the RELEASED stack; the arm runs under the clean-stack-tax design. If it clears the bar by a shippable margin the licensing ruling comes back to Dylan **with the number**. A wrapper tag — including our own — is not a licence. |

---

## 13. Inherited-obligation matrix

| item | disposition |
|---|---|
| sqrt full-chain arm | Own registration slot at R1-assembly time — run, or formally deferred with owner-visible reasoning. B10/B13 inform it; they do not falsify it. **Never revived at M7's arm (a).** |
| n-gram rows | **REOPENED 2026-08-29** (§15, review response). The former ruling — "superseded by D2, no auto-revival" — collapsed two hypothesis classes and is **withdrawn on algebra**: D2 picks one segmentation and removes constituent activations, while additive overlapping rows keep the incumbent unigrams and can fire several phrases at once, and an additive row with **zero residual recovers R0 exactly** where a segmentation change does not. Character n-grams additionally reach rare strings a frequency tokenizer never tokenizes. The classes are compared head to head at equal row budget in **`D2-PRE`**; whichever wins is the registered lever. |
| negatives / step-count confound | B13 matched-steps. The honest M7 statement stands: "the dev suite cannot separate the negatives source from the step count", never "mined negatives do not help". |
| doc2query full dose | Dead on compute for anything confirmatory; bounded research probe only. |
| teacher revisit | §10. Swap bar with the CI-widening penalty; same-teacher default; Dylan signs. |
| M7 mandatory ablations (flat-vs-learned weights, prefix variants, init controls, dense/BM25/fusion decomposition, int8) | Mapped per eligible architecture; each **adopted or not-applicable WITH a reason** recorded here before the freeze. |
| ANN sweep + cost reporting | `ann_sweep.py` on the final candidate; cost rows split **payload / container / doc-index / hydration** — M7's frozen container was 93,886,950 bytes carrying *both* fp16 and int8 payloads; M8 ships int8-only and reports both numbers. |
| one-shot mechanics | The 14-line checklist in §6, each with an M8 acceptance test. |
| M7 report addendum | LR-websearch row as a labelled **exploratory TIE** (+0.0019 [−0.0153, +0.0195]) — the honest sentence is "matches LR's single-table system at 1/15 the bytes". |
| exporter fix | int8-only artifact + §11 parity. |
| Wada context-averaged init / MEV | B15 / B16 (descriptive). |
| Touché-2020 | Banned, stays banned. |

---

## 14. Session guardrails (hard; a violation stops work and goes to the wake-up note)

| id | guardrail |
|---|---|
| **G1** | **No probe run before its bar is committed and pushed.** `m8src/probe_guard.py` reads `m8/registry.json` at the current commit and refuses any probe id lacking bar / endpoint / comparator / multiplicity / no-survivor outcome, any `TBD` bar, an uncommitted ledger or registry, or a HEAD not on the remote. **The registry sha256 is stamped into every result file before any metric is written**, so gating does not depend on an entry point remembering to ask. |
| **G2** | **Protected-path guard, allowlist form** (`m8src/paths_guard.py`). Protected kinds and every route to them: `results/frozen_eval/untouched-*`; **`work/dev/cqadup-{android,english}.json` — the reserved android/english corpora WITH their qrels, materialized 2026-08-26 and found by review, not by accident**; the HF caches `BeIR___fever-qrels` and `BeIR___dbpedia-entity-qrels`; `datasets.load_dataset` by reserved dataset id/config (the network route touches no guarded path); `work/lotte`; `work/m9reserve`. Only the modules on the allowlist may open their own kinds: **`m8src.freeze_lotte`**, **`m8src.freeze_m9reserve`**, **`m8src.protected_filter`** (which emits a query-only hash inventory, never a label), **`m8src.shadow_cross`**, **`m8src.final_run`**, and **`m8src.pre_encode`** — which holds NO
protected kind at all and carries only a corpus-only dataset-loader exemption, since the reserved
CORPORA are ordinary public downloads while their query and qrel payloads stay guarded by path. `claim()` verifies the call site is physically inside the module it claims; there is no public uninstall; paths are RESOLVED before classification so a symlink alias cannot dodge it. Adding a name is a §15 amendment, not an edit. Enforced by a runtime guard **and** a static grep test, both in `m8src/test_guards.py`. **What it is not:** a sandbox. It is a mistake bulkhead — "an ordinary mistake cannot silently burn the access", not "the access is sealed". |
| **G3** | **Nothing irreversible without Dylan.** No freeze, tag, or access; no HF upload; no writes to `results/perquery.json`, `results/eval_manifest.json`, `results/frozen_eval/`, or `m7/`; no final M9-reserve hash-pin (inventories only); no amendment to any §0 FROZEN item. **Git contract, inherited from `instructions-m7.md` and given a home here (Codex gate MAJOR 9):** work lands on the M8 work branch (`m8-planning`, then the M8 work branch), **never on `main`**; **no force-push, with no de-minimis exception** (M7 logged one violation); commit and push after every completed item; never `git add -A` without checking `.gitignore` — one did, committed a multi-GB encode cache, and the push hung. |
| **G4** | **Noise floor before any bar is frozen** (§4.7), with its formula and its named exemptions. |
| **G5** | **Long-run discipline, mechanically.** Smoke every new path (~90 steps) — prefer a path with **no execution history**. `setsid nohup` for anything > 10 min (a harness interrupt killed M7's first final run 40 minutes in). Monitors grep `Traceback\|Error\|FAILED\|OOM\|Killed\|assert` **alongside** the progress marker. Write the wall-clock estimate BEFORE launch and kill any job exceeding it 2×. Take the rate check **in the slow region**, not on the first batches. Never `pkill` from a shell whose own command line contains the pattern — it kills itself first and exits 144. |
| **G6** | **Publish the serial GPU/RAM/disk schedule before the first probe** (`results/m8_schedule.json`), including the reserved-4 pre-encode line and the E12 comparator line. One memory-heavy job at a time; 18 GB peak RAM; `flock`. |
| **G7** | **Ordering interlock**: no teacher download or probe until the protected-query filter covering six + reserved + shadow + M9-reserve inventories is built, hashed and committed. |
| **G8** | **Dev-reuse counter runs from evaluation #1** (`results/m8_dev_reuse_count.json`). |
| **G9** | **Wake-up-note discipline**: anything needing Dylan goes to the top of `m8/STATUS.md`; it is never decided alone. |
| **G10** | **Markdown stays TIGHT** (Dylan, 2026-08-29: *"we're diluting context for next session"*). Every line in a file a session must read before deciding is a tax on every session that follows. Write the decision, the number a rule reads, and the pointer — nothing else. One fact, one home: numbers in the result JSON, bars in `registry.json`, runs in `RESULTS.md`, closed avenues in `EXPLORED.md`, long-form in `research/m8-planning/`. An amendment is *what changed, why, the pointer*, not the reasoning that produced it. **Always keep withdrawn claims and owner rulings** — re-deriving a withdrawn claim costs more than the lines — but keep them short. Adding a long entry means compressing an old one. `wc -l` what you edit; past ~1,500 lines, compressing is part of the task. Full rule in CLAUDE.md. |

---

## 15. Amendments and incidents

*Newest first. An amendment is legal only before any raw number it would affect exists. Entries are
SHORT by rule (§14 G10): what changed, why, and the pointer. Long-form lives in the registry row,
the result JSON, or `research/m8-planning/`.*

**RULINGS BY DYLAN**

- **M8 CLOSES NOW, and the registered exit precondition is SUPERSEDED (2026-08-30).** *"M8 should be
  closed now. Since it's mostly a loss, we should really sanitize the files and only keep the
  learnings."* **This is an OWNER OVERRIDE, not the registered exit firing.** `D2.exit_precondition`
  required `D2`, `B10`, `B8` **and** `R-LIST` to have run and missed, plus a fresh standing-directive
  audit. Only `B8` did (NO SURVIVOR, §25); `D2`'s chains were never authorised (`D2-PRE` routed
  against them, §24) and `B10` and `R-LIST` were never run. **So M8's closure is a decision not to
  spend, and no document may present it as proof that the route is exhausted.** The exit clause
  always reserved this override to Dylan; it is exercised here explicitly and dated so a future
  session does not mistake a budget decision for a measurement. Untested and inherited: `R-LIST`
  (mechanistically triggered by `B2`), `B10`, a trained `D2`, `E14-LORA`.
- **The release is the PAIR (2026-08-30).** M7 is releasable *paired with a good low-compute model* —
  `constella-zero` + `constella-nano` as two points on a quality-vs-query-cost frontier, not a
  leaderboard claim. M10 carries the release, the ONNX port **including the document model**, the
  fastembed integration and the whitepaper (`instructions-m10.md`; renumbered M11 on 2026-09-01, `instructions-m11.md`). M11 (now M15) is noted as an **image**
  model, on the ground that most edge workloads are vision.

- **M8 MAY SHIP A BETTER SYSTEM (2026-08-29).** *"M8 can ship a better system. If that system makes
  sense and is defensible. I'm okay having a custom document encoder if this works well. Obviously I
  don't want to go all the way to train a dense embedding model from scratch."* **Ship-rule
  condition 4 amended**: satisfied by a `qualifying_table` **or** `qualifying_system` key, so a
  document-side win can carry a v2; E11/G4-4's "a D1 win alone does not open the release path" is
  superseded. `C2` selects its **system** form whenever the tower is modified. Legal only because no
  M8 candidate or number exists. **Bounds that survive, so this is not "anything that raises a
  number ships":** query side stays a **pure lookup table** (E1, non-negotiable); the tower must be
  **derived from stella** (LoRA/adapter/last-block/head — a from-scratch encoder is out, and a
  different base model is a §10 teacher swap); the win clears C1/C2/C3 with cost rows; and **the
  report decomposes the win** rather than presenting a document-side gain as a better table.
  Unblocks `E14-LORA` to be registered and measured. Registry: `ship_rule.condition4_policy`,
  `invariants_that_survive_the_amendment`.

- **E14-LORA REOPENED + stella licence CLOSED (2026-08-29).** *"We wouldn't say keep your normal
  document encoder. Since most people are not currently using stella. I'm not against LoRA on the
  document tower"* + *"stella is good for derived weights, no license blocker."* The refusal rested
  on "it costs us the keep-your-own-encoder story" — **that story never existed**: adopting this
  system already means adopting stella, so the user re-indexes either way and co-adaptation has
  **zero marginal product cost**. Any argument leaning on "frozen off-the-shelf document tower" as a
  *user-facing* virtue is void; the premise survives only where defended on evidence or protocol.
  **Reframe:** LightRetriever **trains its document encoder** (`research/lightretriever.md:19,23,382`),
  so `LR-dense-pertask 0.4583` — M7's missed bar — was set with a co-adapted document side while
  M7/M8 fit a table to a frozen tower. Best available explanation for the flat table-side levers;
  unmeasured. `E14-HEAD` does not settle it (head on a *finished* vector; scope limit registered
  before any arm ran). **Still needs a ruling before it can SHIP, not before it is measured:** E11/§5.4
  say a document-side win is not a qualifying v2 *table*, and C2 falls to its `teacher_swapped`
  branch — **does M8 ship a better SYSTEM or must it ship a better TABLE?** Staging binding: dev-scale
  on the two OOD components against their own re-encoded corpora; only a clearing result buys the
  10.12M pre-encode. Row: `E14-LORA`.
- **D-FINEWEB defaults to EXCLUSION (2026-08-29).** *"The Qdrant dataset should really prove its
  value. The webcrawl data could be okay to use, but it's better if we don't."* Measurable, but
  web-crawl enters training only on a **clearly-resolved** gain; being Qdrant's own dataset earns no
  discount. Still pool-varying, so its bar remains uncomputable — that, not the patent
  question, is what blocks it (the patent trigger was REMOVED; see the ruling below).
- **Reserved sets are fine (2026-08-29)** — *"the sets are fine, continue with the codex review."*
  Closes the incident below; no quarantine.
- **M9 picks its own document tower on measurement (2026-08-29)**, defaulting to a re-probed stella;
  T1's NO SWAP does not transfer, being a fact about distilled TABLES not towers. **Refined minutes
  later: PREFER THE PAIR** — a shared document side is the default, broken only on CI-resolved loss.
- **E14: measure it small first (2026-08-29).** Dev-scale only; the doubled pre-encode and any C2
  redefinition were NOT authorised. Partly superseded by the E14-LORA reopening above.
- **E10: seven clean-community LoTTE slices, per-question remedy (2026-08-29).** CQADupStack
  subforums rejected **not on contamination — they passed — but on correlation with the exam**: two
  of the reserved four ARE CQADupStack. The shadow is a check, never a selection surface.
- **harrier CLOSED (2026-08-29)** on undisclosed training data: no design repairs not knowing what a
  teacher has read. stella stands. The vendor rule was never the obstacle.
- **E12: LightRetriever-dense is published numbers only, labelled (2026-08-29).** No LR encode is
  bought; the report may never state or imply a head-to-head on our data.
- **Patents CLOSED (2026-08-29):** *"patents do not have to stay available as a clean held-out for
  M9."* We are not preserving patents as an evaluation domain, so HUPD's CC-BY-NC-SA tag, the
  37 CFR 1.71 counter-argument and the PatentsView alternative are all moot. **`D-FINEWEB`'s patent
  trigger is REMOVED** — web-crawl data may enter the training mix without settling a patent
  question first. *(Restored 2026-08-29 after the compression dropped it and left text that
  resurrected the trigger.)* What still blocks `D-FINEWEB` is not paperwork: it is pool-varying and
  its floor does not exist.

**PROTOCOL AMENDMENTS**

- **Stale bars unfrozen `B8` / `R-LIST` / `B10` (2026-08-29), after `D2-PRE`.** All three still read
  `TBD-noise-floor` although the floor was measured on 2026-08-29, so `probe_guard` refused them —
  and since `B10` is `D2`'s NAMED ALTERNATE and `B8` + `R-LIST` are both required by
  `D2.exit_precondition`, **the milestone's entire fallback path was unreachable and the
  pre-committed exit could never fire.** D2 was the only runnable route by accident of the guard, not
  by evidence. Bars frozen from the MEASURED floor by §4.7's formula, arm type per §23: `B8`
  **0.0040** (closed-form, deterministic, adopts nothing), `R-LIST` **0.0040** (A-leg-only), `B10`
  **0.00519** (trains through the served rule → chain-varying). No number is new; nothing is
  loosened. **Standing lesson: a measured floor does not freeze a bar — an amendment does. When a
  floor lands, sweep every row whose bar reads TBD in the same commit.** (The first attempt was
  itself refused because the provenance note inside the `bar` string contained the literal "TBD" —
  the guard was right, and provenance now lives in `bar_frozen`, not in the rule.)

- **`D2-PRE` review response #3 (2026-08-29), before any solve or retrieval score existed.** A Codex
  pass on the IMPLEMENTATION design returned **6 BLOCKERs / 8 MAJORs, all adopted**; two BLOCKERs
  were contradictions *inside the frozen row* that no implementation could repair. The row is the
  record (`registry.json` `D2-PRE.amended_2`, `frozen_definitions`, `cross_fitting_design`); the
  four that change what the probe *means*: **stage 1 is descriptive and may not stop the probe** (it
  gated on §17b's slope as a one-directional bound, which §17b forbids, and would have killed the
  additive arms on a fertility test irrelevant to them); **the five folds now score disjoint OOD
  query folds**, because scoring every fold's table on the identical 1,915 queries made "positive in
  4/5 folds" nearly redundant with a positive mean; **one shared comparator** (arm (a)'s sum-init
  compile) for all four arms; and **arm (d)'s coldness frozen at <20 fit activations**, without
  which it is algebraically identical to arm (a) (a ridge column with no activations has normal
  equation `λΔ = 0`). Also: the ridge denominator is R0's for every arm — it cancels at serve time
  but **not inside the least-squares objective**, so an arm-specific one silently changes each arm's
  target and effective λ. **Disclosure:** stage 1's number was measured in a prototype *before* this
  amendment (OOD fertility reduction 0.229/0.245 against the ~0.104 the old text named). It clears
  under both readings, so the amendment cannot have chosen this probe's route.
  **Standing lesson: the pooling canary must carry CONTEXT TOKENS.** A phrase that is a query's only
  content normalizes sum-init and mean-init to the *same* vector, so the two-token fixture the first
  self-test used scored 0.0 for both and could never have failed — CODEMAP pitfall 19's family,
  caught by printing the control rather than by the assertion. With context tokens: sum 1.5e-8,
  mean 0.040.

- **Review response #2 (2026-08-29), before any arm ran.** A second adversarial pass on the handoff
  returned 5 findings; all adopted. **The compression had dropped three BINDING items** — restored
  above and marked: `D-FINEWEB`'s patent trigger REMOVED (the compressed text had resurrected it);
  **the E14 lr rule** (pre-registered at 1e-3, *the ladder selects nothing*) — the registry had
  drifted back to a selecting ladder, a live protocol regression, now fixed and extended to bind
  `E14-LORA`; and T1's withdrawn "independent reproduction" wording. **Lesson, and it is the cost of
  compressing a protocol file: a cut is not safe because prose was removed — it is safe only when
  every RULE survives. Diff old against new for rules, bars, withdrawn claims and rulings before
  committing a compression.** Also fixed: the handoff files contradicted each other and the registry
  (STATUS still asked the ruled system/table question; NEXT-SESSION said do not start `E14-LORA`
  while its row said authorised), and **`D2-PRE`'s router was undefined** — "clearly positive",
  "plausibly above", "adequate", "no material" are judgement, not a rule, and `probe_guard` checks
  presence, not executability. Now numeric and frozen, including the additive-over-D2 reversal
  margin. **Recorded for Dylan, not resolved:** condition 4 can now ship a system whose table
  regressed behind a tower gain; `decide.ship()` has no table guard and `freeze.py` does not exist,
  so the artifact invariants are prose. Registered default: **the table must not regress in a common
  frame**, reported decomposed. Review: `research/m8-planning/codex-handoff-review-2026-08-29.md`.

- **MILESTONE AUDIT AND RE-ROUTE (2026-08-29).** All nine probes returned nulls, negatives or
  instrument reads, while `D2` — the only lever with a mechanism pointing up — had no registry row
  and no place on the worklist. Registered `D2`; deferred the recipe/data class; wrote the
  pre-committed exit. **ADOPTED: the chain floor.** §23 measured σ_chain 0.00153 and computed
  **0.00519**, recorded NOT ADOPTED pending its own amendment before any affected arm ran. This is
  it; no such arm has run. **A B-leg-varying arm now reads 0.00519**, not 0.0040; A-leg-only bars
  (`B3`, `E14-HEAD`) are untouched. **Corrections made here:** §5's key list read as exhaustive and
  is a partial render of the registry's 27 — which also **withdraws the audit's own first claim**
  that the recipe class could not satisfy condition 4 (it can; it just cannot carry the bar, §7).
  The `E10` amendment was living outside §15 (moved). The LEDGER carried two `E14-HEAD` claims that
  `RESULTS.md` had already recorded as withdrawn.
- **`D2` AMENDED after adversarial review, before any arm ran; `D2-PRE` REGISTERED; §13 REOPENED
  (2026-08-29).** Three BLOCKERs, two against the audit's own decisions hours earlier.
  (1) **The "compositional init floor" was not a floor.** It specified the **mean** of constituent
  rows; the served query is `normalize(Σ_types g(count)·w_t)`, so the **sum** leaves the summand
  identical and the query exactly unchanged, while the mean silently downweights the phrase ~2×.
  Re-derived independently. Rows are now a residual on the sum init.
  (2) **The coverage gate permitted "expand the pool and re-measure"**, which invalidates the
  existing R0 chains as controls and makes the arm pool-varying — 0.00519 would not be calibrated.
  Removed; gate rebuilt on dev token occurrence mass plus a performance condition.
  (3) **The exit fired too early.** `B2` did NOT close the KL class — its `teacher_top200` arm is
  0.777 nats and B2's own artifact names `R-LIST`; and §15 forbids `E14-HEAD` closing `E14-LORA`.
  Exit now gated on `D2`, `B10`, `B8`, `R-LIST` all missing plus a re-run of the standing directive.
  **§13's "n-gram rows superseded by D2, no auto-revival" is WITHDRAWN on algebra** — D2 removes
  constituent activations while additive overlapping rows keep them, and an additive row with zero
  residual recovers R0 exactly. Classes compared at equal row budget in `D2-PRE`.
  Also: §17b downgraded to **correlated headroom, not a bound in either direction**;
  `NF-CROSSED-FUSED` made mandatory with its undefined "plausibly clear" escape removed; tokenizer
  training pinned deterministic. **Kept against the review:** 0.00519 stands (lowering a registered
  bar on an argument available beforehand is what the protocol forbids), and retokenization is not
  pool-varying (the review agreed). **Recorded, unfixed:** G8's dev-reuse counter is absent from
  HEAD. Review: `research/m8-planning/codex-d2-reroute-review-2026-08-29.md`.
- **`E10` REOPENED (2026-08-29). The remedy artifact is NOT decontaminated; its zero re-screen was
  tautological. `results/m8_lotte_remedy.json` may not be pinned, served or grandfathered.**
  (a) The screen compared **roles, not protected content** — LoTTE queries were never compared to
  protected *documents*; two verbatim survivors confirmed, and 36 retained queries share an 8-word
  run with protected documents. (b) The re-screen **cannot fail**: remediation builds the exact
  complement of its own hits and re-reads in-memory survivors, not the serialized files. (c) The
  100× query/document asymmetry measures the **detector**, not the corpus (documents need 8 shared
  bottom-32 entries, so a document under 15 words can never hit). (d) The exact 14,034 match is
  **circular**, not validation. Needs: union screen across roles and families, an INDEPENDENT
  acceptance detector, a canary proving acceptance can fail, length-adaptive matching, and a
  measured correlation with the exam. Review: `research/m8-planning/codex-e10-remedy-review-2026-08-29.md`.
- **`E14-HEAD` AMENDED after a design review, then CUT BACK after an implementation review
  (2026-08-29, both before any arm ran).** The registration required a nonlinear head on a premise
  this ledger had already refuted: a **renormalized** linear map is NOT absorbable (the per-document
  `1/|Md|` cannot move into a shared row; §6 D1 recorded rank agreement 1.000 without renormalization
  and 0.000 with it). So **LIN became primary, MLP its nonlinearity control**. Zero-init gives
  `normalize(d)` not `d` → new comparator **`R0N`**. The lr ladder would have observed the endpoint
  before selecting → made dev-blind on a disjoint tuning seed. Second review returned 5 BLOCKERs,
  two in machinery the probe did not need. **THE LEARNING RATE IS PRE-REGISTERED AT 1e-3 AND THE
  LADDER SELECTS NOTHING** *(restored 2026-08-29 — the compression dropped this and the registry
  had drifted back to "the ladder selects")*: the `lin` ladder came back FLAT across a 10× range
  (−0.2434 / −0.2404 / −0.2430), so a dev-blind selection apparatus, a bespoke holdout statistic and
  a plateau continuation were deciding something that does not appear to matter, and every part of
  it was a way to be wrong. 1e-3 is the registered grid's midpoint, the standard rate for a small
  zero-init adapter head, and where that flat curve peaked. The ladder arms survive **only as a
  descriptive sensitivity band, run if the primary reports a null**. Nothing is chosen after seeing
  a number. **This rule binds any future adapter probe, `E14-LORA` included.**
- **Registration hygiene (2026-08-29).** Full manifest key schema fixed before any manifest existed
  (35 `train.Cfg` fields + artifact-level; unknown key ⇒ condition 4 FAILS). T1's ordering, frame and
  `spec_name` fixed before any T1 number. Ledger opened v1 from `PLAN-DRAFT.md` (since **DELETED** —
  it had become a second source of truth disagreeing with this one after 17 amendments); v2 after
  Codex (BLOCK 9/9/3) and Fable gates; further amendments from a second Fable pass and a fourth
  results-review pass, all before the numbers they bind.

**RESULTS THAT CHANGED THE PROTOCOL**

- **`E14-HEAD` REPORTED — NO SURVIVOR (2026-08-29).** Dense −0.0244 (LIN) / −0.0293 (MLP) vs a
  +0.0040 bar, all six arms agreeing in sign; fused −0.0024 / −0.0042. **The patch stack measured as
  a null** (`R0N` vs `R0`: −0.00001 dense), which is what licenses reading the rest.
  **CORRECTED after review — two claims WITHDRAWN.** (1) The mechanism control shows only that the
  head reduced bag-query nDCG LESS than teacher-query nDCG (LIN +0.0091, MLP +0.0075, all twelve
  values positive) — descriptive evidence of **relative** alignment. It does **not** show an absolute
  bag benefit (both absolute gains are negative); "genuinely makes documents more reachable by a bag"
  and "buys bag-reachability only by destroying information" are withdrawn. At n=3 the smallest
  one-sided sign-test p is 0.125 and the cells are not independent. (2) "The evidence disqualifies
  the OPTIMIZATION-INADEQUATE reading" is withdrawn: LIN is a strong negative for the registered
  2,500-step config but remains inadequate for the method-level question; MLP is a different
  architecture and 0.350 vs 0.220 is one unreplicated arm per budget. The endpoint also did not move
  "away" monotonically — early persistent harm, not divergence. **Registered labels stand as the rule
  produced them; contrary evidence sits beside them.** **Every future probe row must carry a negative
  branch.** Bearing on M9: teacher-style queries lose MORE than bag queries (−0.031 vs −0.022), so a
  shared document side is not free for the pair. `results/m8_e14_head.json`.
- **`B3` RAN — UNINFORMATIVE, the useful kind (2026-08-29).** A 4× dose moves dense +0.00135 / fused
  +0.00369 vs 0.0040; slope +0.00097 per doubling ⇒ ~17.6× the pool (~5.9M pairs). Phase A is not
  meaningfully pair-starved. B3's lever had been replaced (ICT → real-pair pool scaling) before any
  B3 number existed; the retired 1.00-vs-0.75 primary came out NEGATIVE on dense.
  `results/m8_b3_decision.json`.
- **The B-leg floor, and a WITHDRAWN claim (2026-08-29).** The B-leg artifact's "aliasing understates
  the floor" claim is **withdrawn**: at K=3 the floor statistic is the sample range (CV 0.525), so
  `P(R_B ≤ R_A)` is a coin flip and "the B leg adds nothing" had no content. The crossed B×A design
  was registered anyway for a different reason and measured (§23).

**INCIDENTS**

- **An external reviewer read two RESERVED sets in full (2026-08-29).** A repo-wide grep dumped
  `untouched-cqadup-english` (164,742 B) and `untouched-dbpedia-entity` (67,801 B) — queries AND
  qrels — into a Codex review's context. **`paths_guard`/G2 is an in-process bulkhead and is
  structurally incapable of constraining a separate process**, so the routine-review grant reopens
  this hole every time. Nothing was scored; no model or decision read them. **Dylan ruled the sets
  fine**; quarantine lifted. **Fix kept regardless:** briefs carry a read-exclusion for
  `results/frozen_eval/untouched-*`, the reserved qrels caches and `work/m9reserve`; briefs name
  files rather than inviting repo-wide searches; review logs are audited before findings are read.
  In CLAUDE.md so it binds outside M8.
- **Reserved corpora materialized on disk, found by review (2026-08-29).** `work/dev/cqadup-{android,
  english}.json` held the complete corpora AND qrels of two reserved sets. **Nothing scored them.**
  Now a protected kind under G2.
- **A registration edit REVERTED because the audit refused it (2026-08-29).** Disclosed rather than
  quietly kept.


## 16. Gate findings — REMOVED 2026-08-30

An index of where each pre-milestone gate finding was discharged. Every one was discharged before any probe ran. **Correction (Codex diff, 2026-08-30): two dispositions lived in the removed §6 — gate BLOCKER 5 in its stage ordering and MAJOR 4 in its port checklist. The checklist is restored verbatim in §6.2; the stage ordering is gone with the stage plan, and its disposition now exists only in `research/m8-planning/`.** All others live in the sections they changed.

## 17–23. Measured results that bars or plans read

*One block per probe: the verdict, the numbers a rule reads, and the withdrawn claims. Everything
else — full tables, per-arm detail, provenance — is in the result JSON and `m8/RESULTS.md`.*

### §17 / §17b — what the query-side loss is made of (`results/m8_retention_decomposition.json`)

**Fragmentation is the channel, and it survives the attack the length claim fails.** Pooled
within-dataset, the table falls **0.050 nDCG further behind the teacher per +1.0 subwords/word**
(t = 4.61), and **every single-dataset exclusion leaves the slope positive at t ≥ 3.28** (worst case
scifact +0.0369, t = 3.28). Mechanism: the **teacher improves** with fragmentation (+0.038, t = 2.72)
while the table is flat (−0.012, t = −0.85) — the gap widens because the teacher pulls away.

**Read it as CORRELATED HEADROOM, not a bound in either direction** (downgraded 2026-08-29): it is
uncontrolled between-query OLS, so the slope may be driven by specificity, rarity or domain and D2
could recover none of it — while a phrase feature could equally exceed it via conjunction effects.
No write-up may quote `0.050 × Δfertility` as a forecast.

**WITHDRAWN:** H3's short-query framing is unsupported and may not be quoted (removing ArguAna
collapses its t from 3.01 to 1.51; ArguAna is 97% single-arm and has no contrast to measure). The
binary present/absent contrast is **not resolved** (4/5 informative datasets positive, one-sided
p = 0.19) — **the continuous slope is the instrument to quote**. An earlier "6/6, p = 0.016" counted
ArguAna's coin flip and tokenized with punctuation attached. Also withdrawn: the "post-2018 drift"
story — the costly words are a mix of compounds, domain terms, entities and ordinary English.
Because of that mix, D2's tokenizer training corpus matters **less** than first claimed, and the
argument this section seemed to give the FineWeb arm is withdrawn.

### §18 — `B7`, the solver gate: **PASSED** (`results/m8_b7_solver.json`, `m8_b7_realdata.json`)

30,522: 26 its / 5.2 s / 3.77 GB · **65,536: 51 its / 10.4 s / 4.42 GB** · 131,072: 68 its / 16.6 s /
5.72 GB. A dense fp64 Gram would be 34.4 GB and 137.4 GB. Agrees with the direct solve to 4.6e-7.
**Unpreconditioned CG does not converge in 1,500 iterations on Zipfian data (5.9e-4); Jacobi
converges in 61** — a uniform synthetic converges in 131 and would have produced a wrong PASS.
Real-data precondition discharged: identical dev macro at every λ, argmax λ=1e-2 at 0.343924,
reproducing M7's 0.3439 for stella. **Unblocks D2 (64–128K vocabularies) and T1.**

### §19 — `B2`, the KL term (`results/m8_b2_entropy.json`)

**H2 CONFIRMED for the recipe's UNIFORM bank, and only that.** Teacher target median entropy
**4.73e-07 nats**, p_max median 1.000000, 84.3% below 1e-4, entropy 0.0052 of ceiling; the shipped
table already ranks the positive first in **99.75%** of queries, so the KL term's own median value is
1.08e-07 nats. **It does NOT close the KL class:** the `teacher_top200` arm measures **0.777 nats
mean / 0.369 median**, and this artifact names **`R-LIST`** as the consequence. Hard-candidate
listwise distillation stays live.

### §20 — `B17`: the branch fired and was **DISOWNED** (`results/m8_b17_oracle.json`)

Held-out **0.1999** (41.6% of the 0.4806 teacher) fired the registered ≤0.40 branch — but the same
class on 350K general queries scores **0.3439**, so B17 measured its **own 957-query fit set**,
exactly as its pre-registered caveat warned. **The rule was NOT amended**; the branch is disowned in
the direction that costs us something. Standing lesson: a capacity probe whose fit set is the binding
constraint measures the fit set. (This is why `D2-PRE` must cross-fit.)

### §21 — `T1`, the teacher screen: **NO SWAP** (`results/m8_t1_decision.json`)

stella 0.3438 · granite-r2 0.2915 (**−0.052** [−0.066, −0.039]) · gte-modernbert 0.2349 (**−0.109**
[−0.123, −0.094]). All optima interior; condition 1 fails for both, so conditions 2–4 never arise.
Gaps are 5–11× the 0.0096 swap penalty. **The tower again fails to order the table**: gte-modernbert
has the HIGHER published score of the two challengers (55.33 vs granite's 53.1) and the far LOWER
distilled table. **WITHDRAWN, and stated at its true weight:** calling that an "independent
reproduction" of M7's result oversold it — it is an n=2 sign anecdote resting on two self-reported
model-card BEIR figures from different harnesses. M7's eight-candidate Spearman(ceiling, table)=0.000
remains the evidence; this is corroboration. **Holm family pinned:** the family is the challenger
set, so if stella-1.5B or harrier is ever screened, Holm re-runs over the UNION of all challengers
ever screened against this incumbent, not the newcomers alone. Screened in a shared
student frame only within a tokenizer family; cross-family screens are labelled teacher-plus-tokenizer.

### §22 — `B6-pre`: E3's hard condition is **MET** (`results/m8_b6_pre.json`, `m8_b6_pre_mlp.json`)

One file, opset 17, **zero custom-domain ops**; 3,415 nodes linear / 3,426 with GELU (exports as a
plain `Erf`); parity min-cosine 0.99999994 / max-abs 2.05e-07 against §11.4 tolerances of 1e-4 and
1e-3. **This cleared D1's EXPORT gate only** — `E14-HEAD` later measured D1's quality at −0.0244 /
−0.0293. It still matters because **`D2`'s output must clear the same gate**, and because both passes
used **near-identity weights**: the actual trained artifact must be re-exported before anything is
called shippable.

### §23 — the crossed B × A floor (`results/m8_noise_floor_crossed.json`)

Nine cells, (B-checkpoint seed) × (A seed), int8/sqrt. On the out-of-domain macro and worst group:
**σ_B 0.00103, σ_A 0.00106, σ_resid 0.00039, σ_chain 0.00153**, SD(fresh null Δ) 0.00217, and the
standing 0.0040 carries **~6.5%** type-I against a fresh null difference between two chains. Nothing
detectable on the other two endpoints. σ_chain is **a floor on the floor** (moment estimators clipped
at zero, `sqrt` concave, ~8% low at this design).

**Consequences in force.** A-leg-only arms (`B3`, `E14-HEAD`) read σ_A alone → `2 × 1.693 × 0.00106
= 0.0036`, under the planning minimum, so **their bars stand unchanged at 0.0040**. A **B-leg-varying
arm reads 0.00519**, adopted by the 2026-08-29 audit amendment — `D2` is the first arm it governs.
**Still not bounded: a POOL-varying lever** — all nine cells share one pseudo-query pool — which is
why `D-FINEWEB`'s bar remains uncomputable. **No chain-level FUSED floor exists** (cells were scored
dense only); `NF-CROSSED-FUSED` measures it.

**WITHDRAWN:** the "aliasing understates the floor" claim that motivated this design. The aliased
diagonal is `B_s + A_s + e`, already carrying the chain variance — unbiased-but-noisy, not
anti-conservative. Confirmed on this data: the diagonal's range sits at 0.43× its own expectation,
inside the [0.25, 1.96] interval a K=3 range spans.

### §24 — `D2-PRE`: **DO NOT AUTHORISE** (`results/m8_d2_pre.json`)

**All four new-row classes are NEGATIVE out-of-fold** against the +0.00519 bar, so `D2`'s five
chains are not authorised: `add_word` **−0.0028** (best) · `seg` **−0.0052** · `seg_cold` −0.0055 ·
`add_char` −0.0137. One positive fold out of five for three arms, zero for `add_char`. 96 minutes.

**The negative is not a failed measurement — the three artifact explanations are all excluded.**
(a) The sum-init zero-residual compile reproduces R0 to **−2.8e-06** against a 0.001 tolerance, so
the registry's corrected compositional floor (**sum**, not mean) is confirmed at full scale and
condition 5 passes. (b) Coverage was never the constraint: zero-update occurrence mass **0.0001 –
0.001** against the 20% gate. (c) No leakage — **zero** OOD query texts appear in the fit list — and
λ was **interior** for three of four arms, so the descent-and-stop amendment cannot be blamed.

**The sharpest evidence.** `seg_cold` forces **23,601 of 35,014** rows to zero residual and moves the
result by 0.0002 against `seg`. Two-thirds of the added vocabulary is inert: the residual capacity is
not thinly spread, it is absent.

**BOTH classes close, and that was the design.** `reversal_margin_met` fired — additive beats
segmentation by 0.0024 ≥ the registered 0.0020 margin, as the algebra predicted (an additive row at
zero residual recovers R0 exactly; a segmentation change does not). Had anything cleared, the chains
would have gone additive. Since both are negative, a `D2` miss **cannot** be re-read as the wrong
parameterisation — which is exactly why the additive arms were registered alongside it.

**Scope, stated:** this is a CLOSED-FORM screen on the CQADupStack dev pair. It says the added rows
carry no closed-form residual capacity; it does not prove trained rows could not. That is what the
0.00519 routing threshold was for, and it was not met by any arm in any direction.

**Descriptive only, never a gate** (§15 review #3): stage-1 fertility reduction 0.164 / 0.176 on the
OOD pair. It cleared the old text's ~0.104 comfortably and the arms still lost — which is itself
evidence for §17b's downgrade to correlated headroom, since the mechanism moved and the metric did
not follow.

### §25 — `B8`, target design: **NO SURVIVOR** (`results/m8_b8_target.json`)

Closed form, 340,850 decontaminated pairs (**0 dropped** — every query's positives resolve in the
pool, so no selection bias), three targets sharing one X / W0 / λ grid.

| target | group vector | OOD | Δ vs bare | λ |
|---|---|---|---|---|
| `bare` (R0's) | **0.6388** | 0.3474 | — | 1e-2 interior |
| `mix50` | 0.6356 | 0.3064 | **−0.0032** | 1e-2 interior |
| `centroid` | 0.4719 | 0.1645 | **−0.1669** | 1e-1 interior |

Bar 0.0040; neither clears. **The comparator is sound**: `bare` independently reproduces B7's
closed-form 0.3439 (§18) at 0.3474 on a differently-built fit list, and every optimum is interior so
nothing is grid-limited.

**The mechanism, stated because the intuition was reasonable and is now refuted.** Retrieval scores
`q·d` and never `q·q`, so fitting the table to the teacher's QUERY point looks like aiming at the
wrong manifold. It is not: aiming at the positives' centroid costs **0.167**. The teacher's query
encoder already performs the cross-manifold mapping, and a centroid target instead fits *which*
documents happen to be positive. `mix50`'s −0.0032 is inside the noise band — read it as "bare is at
least as good", not as "blending hurts".

**Consequence:** one of `D2.exit_precondition`'s three remaining probes is discharged. `R-LIST` and
`B10` remain.

### §26 — `VECTOR-PRF`: **NO SURVIVOR** (`results/m8_vector_prf.json`)

Train-free dense pseudo-relevance feedback on R0's frozen table at the PUBLISHED config
alpha=0.4, beta=0.6, k=3 (arXiv 2205.00235), used verbatim with no grid. **Dev group vector
−0.0510, fused macro −0.0207, and negative on ALL SIX components** — nq-250k 0.801→0.767,
hotpotqa 0.613→0.545, cqadup-programmers 0.339→0.328, cqadup-physics 0.396→0.390,
heldout-train 0.632→0.584, heldout-longq 0.912→0.865. Bar 0.0040. Query drift cos(q,q') 0.80–0.92.

**Why it was worth running and why the negative is informative.** It was surfaced by an external
review as the cheapest mechanistically-live lever the plan had missed, and it attacks the
bag-vs-context mismatch from the opposite side to `D2`: instead of making the query's *vocabulary*
finer, it moves the query *vector* onto the document manifold using the documents themselves. Both
fail, which is a stronger joint statement than either alone — **the gap is not a query-representation
resolution problem and not a query-placement problem.**

**Scope, honestly.** This closes TRAIN-FREE post-hoc refinement. It does NOT close a LEARNED feedback
encoder (ANCE-PRF and family), which trains a query encoder and is out of scope under E1 anyway. The
uniformly negative sign across six components with a single frozen config is also consistent with
PRF being a poor fit for a bag query specifically: feedback assumes the first pass is good enough to
trust, and the table's first pass is precisely what is weak.

**Consequence:** a `qualifying_system` route closes. `R-LIST` and `B10` remain before the exit.
