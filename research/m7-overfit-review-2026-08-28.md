# Pre-freeze audit: dev-suite overfitting, over-engineering, reproducibility

Audited 2026-08-28, read-only, against the repo at `3b8dcdf` plus uncommitted results.
Files cited by absolute path; line numbers from the current working tree.

## Verdict (one paragraph)

**The confirmatory number will mean something — freeze once the queued in-protocol steps are done,
not today.** The thing 322 dev evaluations contaminate is the *selection*, not the *measurement*:
the six datasets were never used to choose anything (three benign disclosed deviations,
`m7/SIX_ACCESS.log` verified), the comparators are frozen per-query vectors, the recipe is fixed
before the access, and the three tier comparisons are pre-registered at family α=0.025. Dev reuse
cannot bias that number; it can only mean the frozen recipe is less good than dev suggests — and
the arithmetic below says exactly that: roughly +0.01 of the post-gate +0.017 dev gain is expected
selection/nuisance optimism, and the expected six-set contribution of the entire post-gate lever
programme is **0.000 ± 0.005**. So the number is trustworthy; the *story around it* must shrink
(findings B1, B2). Do not freeze until: fusion is re-selected on `p35w-2m-s2500` (M4), the teacher
probes / lever #7 queue is resolved or explicitly abandoned in the ledger (M6), RECIPE.md is made
truthful and complete (B1), and today's result artifacts are committed. Nothing found here breaks
the protocol; everything found here is documentation, framing, or a queued step.

---

## Q1. Selection risk, quantified

**The measurement.** The gate-era winner `s2w-1e3-s1000` scored 0.5987 on the full dev suite; the
shipping candidate scores 0.6153 (`results/m7_compare_full_simplify.json`). Banked post-gate gain:
**+0.0166** (including sqrt pooling +0.0040).

**Per-look noise.** Query-sampling SE of a paired full-suite delta ≈ 0.0018 (raw CI half-widths
~0.0036 across `m7_compare_full_steprule.json`, `m7_lever4_pooling_*.json`). Recipe-nuisance spread
from `m7_compare_full_stepspread.json`: a step count alone moves the macro +0.0027..+0.0078
(mean +0.0052). Combined effective sd per selection look σ ≈ 0.003, plus a systematic ~+0.005
available to any "train a bit more"-shaped choice.

**Selection events.** Four banked accept decisions (500k adoption, 2m cross-arm pick, s2500
extension, sqrt pooling), each a best-of-3-to-8 comparison, each arm additionally carrying a
best-of-~5 proxy step pick — a pick the project itself proved selects noise (the 0.51300 proxy peak
reproduced as 0.51262, and the proxy ranked three arms exactly backwards from the full suite).
Expected banked noise if every true effect were zero: 4 × E[max of k~5 draws of N(0, 0.003)] ≈
4 × 0.0035 ≈ **+0.014**. That is the winner's-curse ceiling; the matched-control floor is:

**Matched-control attribution** (`results/m7_compare_full_postabl.json`): the whole 925K-span
pseudo pool buys **+0.0046** (0.6153 vs nopseudo 0.6107); the 2M-vs-500k dose buys **+0.0002**;
the A-step extension **+0.0023**; sqrt **+0.0040** (which then failed to replicate on the next
artifact, p=0.063). Every one of these is inside the 0.0027–0.0078 nuisance band. Attributable
total ≈ 0.011 of the 0.0166; each attributed term is itself one in-band draw.

**The transfer check.** Over the same programme the out-of-domain pair moved ~0.000 (all arms
0.3648–0.3688; the eight mandated-ablation arms span 0.0009), and `heldout-longq` got *worse* under
several adopted-adjacent arms.

**Defensible estimate: the dev-selected candidate's 0.6153 carries ≈ +0.008 to +0.015 of
selection/nuisance optimism, i.e. half to nearly all of the post-gate gain; expected six-set
contribution of everything adopted since gate #2: 0.000 ± 0.005.**

**Does the planned framing survive?** The tier framing survives — it is measured on the six
directly, against frozen vectors, and needs nothing from dev. Two things must shrink:

1. The report may not present the lever programme as having improved the released system. The
   honest sentence is: *the released artifact is the gate-era distillation recipe (B16k + A) plus a
   pseudo-query pool worth ≈ +0.005 in-distribution dev and a set of in-band tweaks with no
   demonstrated out-of-domain effect.* Lead with the matched-control attribution, never the chained
   +0.0126.
2. Expectation-setting: the honest pre-run projections straddle the release bar —
   0.5562 (stella six_est, `m7_bars_after_swap.json`) × 0.764 OOD retention = **0.425** (miss);
   BM25-ratio anchor 0.4174 × (0.3672/0.3223) = **0.476** (clear release 0.4583, miss tier-1 dense
   0.4868); dev-macro retention × 0.915 = 0.509 (clear). The report skeleton should be written to
   accommodate a miss *before* the final run, per the ledger's own one-access logic.

## Q2. Adoptions that do not survive whole-search multiplicity

Primary bar-facing hypotheses across the search ≈ 40–60 (lever2 ×3, lever4 ×6 twice, lever5 ×6,
lever6 ×2, negatives ×6, ablations ×8, simplify, doc2query, bigram, plus the dev-audit
re-judgments). Holm rank-1 threshold at m=50: **p ≈ 0.001**.

- **Does NOT survive: lever #4 sqrt pooling** — p=0.0113/0.0128 (`m7_lever4_pooling_p35w-2m-s2500.json`).
  Empirically corroborated: it failed outright on the next artifact (p=0.063/0.067). It stays only
  because its bar was pre-registered and the change is byte-free; the report must label it exactly
  as LEDGER.md:499-504 already requires and additionally note it fails any search-wide correction.
- **Does NOT survive: lever #2's middle decision** (the 2m cross-arm pick), p=9.7e-3. Its matched
  dose control is +0.0002.
- **Survives numerically but is hollow: lever #2's step extension** (p=3e-5). It is *literally the
  nuisance parameter* the stepspread pass priced at +0.0027..+0.0078; a tiny p here certifies
  query-sampling stability of a step-count artifact, not a recipe effect. Say so.
- **Survives: lever #2's first decision** (500k adoption, p=1.2e-4) and, as a cumulative claim
  only, the +0.0126 chain (outside the band) — with the ledger's own "adaptive search found a
  better dev artifact, causal attribution not established" wording (LEDGER.md:476-486).
- The only component-level effect that survives everything: **learned per-token weights**
  (removal −0.0062, CI [−0.0094, −0.0032]) — and even it moves the out-of-domain pair by 0.0001.

## Q3. Over-engineering: every shipping component and its evidence

| component | evidence it earns its place | verdict |
|---|---|---|
| learned per-token weights | flat ablation −0.0062, CI-resolved | **earned** (the only one) |
| no query prefix | prefix −0.0019, CI-resolved | earned (absence of a component) |
| pseudo-query pool (925K spans) | matched control +0.0046 full-suite; in-band | plausible, in-band; keep, label |
| 2M-vs-500k dose | +0.0002, unresolved | inertia; kept only by the joint simplify failure |
| teacher-context init (30,522 fwd passes) | +0.0004/+0.0002 single-knob; cold-rows says untrained rows contribute at 0.143x anyway | inertia; kept only by the joint failure |
| IDF weight seeding | removal *helps* +0.0013 (p=0.0124, fails Holm-8) | inertia; point estimate disfavours it |
| reg_init 1e-3 | −0.0000, CI [−0.0001, +0.0001] | pure inertia; kept only by the joint failure |
| sqrt pool_mode | +0.0040, in-band, non-replicating | free at serve time; keep with the ledger's label |
| steps_a 2500 | +0.0023; is the nuisance parameter | in-band |
| hard_neg_k=0 | closure sound (OOD flat + rule honoured against interest) | earned |
| **fn_margin=0.02, use_provided_hardneg=True, temp=0.02, n_neg=32768, kl_k=32, b_pseudo_frac=0.5** | **none — `phase3_hparams` never ran** (m7/EXPLORED.md:74) | untested defaults, live in the shipping loss (`m7src/train.py:625-627` folds ESCI/Mr.TyDi provided negatives into every A step) — worse-evidenced than the four "inert" components, and two of them are absent from RECIPE.md |

**Was the simplification test the right shape?** As a freeze-adjacent guard against adaptive
search, yes — one pre-registered joint arm, no ladder, is the correct shape *for that constraint*.
As *evidence about the components*, no: n=1 joint arm cannot attribute, and the ledger's own
correction (LEDGER.md:411-417) concedes the −0.0063 main-effects gap is the same size as the
perturbation band, so "one bad draw from the recipe distribution" and "interaction" are not
separable. **The margin is defensible-but-lucky rather than principled**: it is anchored to the
smallest adopted effect (+0.0040 sqrt), and that anchor later failed to replicate — i.e. the margin
is anchored to what is probably noise. What rescues the outcome is that it does not hinge on the
margin: the raw CI lower bound is −0.0102, which fails any margin up to 0.0102, including one
anchored at the top of the perturbation band (0.0078). State that in the report; it converts a
margin-provenance objection into a non-issue. What is NOT rescued is RECIPE.md's framing — see B1.

## Q4. Reproducibility walk (clone → artifact)

An outside engineer with the repo, licences, and a 10 GB GPU:

1. **Env** — `m7/requirements.lock.txt` ✓.
2. **Training data** — `m7src/trainmix.py:56-145`: every `load_dataset` call is **unpinned** (no
   `revision=`) for BeIR/*, `rajpurkar/squad`, `tasksource/esci`, `nq_open`, `trivia_qa`; only
   Mr. TyDi is path-pinned (not hash-pinned). HF datasets are mutable. Counts in
   `results/m7_field_table.md` detect drift but do not repair it. → M1.
3. **Teacher** — weights revision pinned; the `trust_remote_code` modeling code is **not** —
   CODEMAP.md:105 admits it ("Pinning weights does not pin trust_remote_code code, which comes
   from a separate repo at HEAD"). Different code → different document vectors → nothing downstream
   reproduces. → M2.
4. **Derived state** — `work/` (decontam `kept.json`, `banned_pool_rows.npy`, the 9.5 GB pool,
   encode caches) is all rebuildable via `run_stage0*.sh`, self-checking via the pool `id_sha256`
   interlock (`m7src/train.py:137-159`). ✓, at an 8–12 h cost.
5. **The chain** — the exact Cfg of both legs is committed
   (`results/m7_run_p35b-2m.json`, `results/m7_run_p35w-2m-s2500.json`), including the knobs
   RECIPE.md omits. So the repo reproduces; **RECIPE.md alone does not**: it omits `fn_margin=0.02`
   and `use_provided_hardneg=True` (dataset-provided hard negatives enter every A-phase step), and
   its stated purpose is to spare a third party reading `program.py` plus four drivers. → B1.
6. **Serving rule** — `adopt_pool_mode.py` + committed `m7_lever4_pooling_p35w-2m-s2500.json` ✓.
   Export via `table.save_release` ✓ (folding conformance-tested).
7. **Fusion** — absent from RECIPE.md entirely, and **not yet selected for the shipping
   candidate**: the only committed fusion artifacts are `m7_fusion_p1-objB.json` (superseded era).
   `run_freeze_prep.sh` step 1 will produce it; RECIPE.md must then document w (and whether the
   dense-only endpoint won), because the tier-1 claim is made by the fused system. → M4.
8. **Determinism envelope** — a re-run on this same box differs from the shipped baseline by
   4.5e-6 macro / 2.7e-5 on one component (`m7_compare_full_ablations.json`, p4-base-a raw CI
   [0.0, 1.34e-5]); a different GPU will differ more. Harmless — but see m1, because the ledger
   claims exactly zero.

## Q5. Was correcting the missed step rule itself selection? (the attack)

The strongest attack: *rule-compliance discovery is event-driven, therefore selective — you audit
the rule you happen to notice, and noticing is not exchangeable.* Three sub-attacks, and where they
land:

1. **"The correction was written after the favourable numbers existed"** — true (the 2500-step
   +0.0072 adoption was on the books) — **but the fix was committed before the corrected numbers
   existed** (commit `5cabf45` pre-registers, `24ba6c6` records the outcome), including the clause
   "the full-suite number gets no vote", and the outcome cost the project its adoption. Selection
   optimises in your favour; this ran against it. The attack fails on the direction of the payoff.
2. **"The non-retroactivity of the amendment was chosen after seeing both versions pay"** — the
   amendment (match the baseline's steps) applied retroactively would have *kept* the 2500-step
   arms, which cleared the bar (+0.0112/+0.0111). Choosing non-retroactivity kept the worse result.
   Fails the same way.
3. **"Only the family you noticed was audited"** — this one lands. Nobody enumerated every
   pre-registered rule against every family it binds. The mandated ablations and attribution
   controls also inherited `steps_a=2500` without per-arm proxy selection; their null conclusions
   happen to be blessed by the later amendment, but that blessing is retroactive-in-effect and
   undocumented as such. **Action (M5): before freeze, one pass — list every pre-registered rule in
   LEDGER.md and tick every arm family it applies to, so compliance stops being discovered by
   accident.** ~1 hour, and it is the difference between "we honoured the rules we noticed" and
   "we honoured the rules".

Residual, already conceded by the ledger and non-negotiable for the report: the negatives closure's
reason #1 rests on an instrument the same session proved broken; the reportable claim is "the dev
suite cannot separate the negatives source from the step count" (LEDGER.md:258-263), and
RECIPE.md's closed-avenues row currently says the stronger "closed; the apparent gain was
memorisation" — align it.

## Q6. What the project is fooling itself about

**The load-bearing, comfortable, unchecked belief: "the out-of-domain pair is an oracle, and its
flatness proves the levers do nothing out-of-domain."** Three unchecked parts:

- **Resolution.** Per-arm OOD deltas carry ~±0.005 raw CIs (n=1,915 queries). Every lever effect in
  play is 0.002–0.011 — at or below the instrument's resolution. "Whichever step count you choose,
  the out-of-domain effect of mined negatives is zero" (LEDGER.md:268) is a within-noise reading
  presented as a measured zero. The eight-ablation-arm span of 0.0009 is genuinely strong (eight
  consistent point estimates); the negatives-era "zero" (span 0.0030 over four arms) is not. The
  honest form everywhere: "unresolved below ~0.005".
- **Representativeness.** The "out-of-domain" pair is two subforums of ONE family — StackExchange
  duplicate-question retrieval. Of the six confirmatory sets only FiQA is StackExchange-adjacent;
  ArguAna, TREC-COVID, SciFact, NFCorpus, SciDocs resemble no dev component. 0.764 OOD retention is
  the *more* honest figure, but it is a StackExchange number wearing a "generalisation" label — it
  neither bounds nor predicts scientific-domain retention. The report must scope it.
- **And the next concrete self-caught-class error, found:** STATUS.md:36 and RECIPE.md:112 say
  retention is **0.926** on the dev macro. 0.926 × teacher 0.6724 = 0.6225 — that is
  **`p4n-teacher16-a`, the reverted candidate**. The shipping candidate's is 0.6153/0.6724 =
  **0.915**. The headline honesty figure was computed on the abandoned arm and survived the revert.
  (The 0.764 OOD figure survives coincidentally — both arms score ~0.367 there.)

Second confirmed error of the same class: LEDGER.md:395-396 ("deterministic to the last digit",
"raw CI of exactly [0.0000, 0.0000]") and LEDGER.md:349 ("p4-base-a and p4n-bank-a agree to 16
digits") are contradicted by the committed artifact: the replay's raw delta is 4.47e-6, raw CI
[0.0, 1.34e-5], with a 2.7e-5 component delta (`m7_compare_full_ablations.json`) — the ledger read
the *rounded display CI*, the exact reading LEDGER.md:127-128 forbids. (The proxy-path macros do
agree to 16 digits — `m7_dev_reuse_count.json` — so determinism holds where it matters; the claim
is over-stated, not wrong in substance.)

---

## Findings

### BLOCKER

- **B1 — RECIPE.md is the permanent release document and it is wrong in three places.**
  `/home/dylan/asymetric-dual-encoders/m7/RECIPE.md:79` ("Four things that look removable and are
  not" + "the cheaper alternative that fails") asserts what LEDGER.md:411-417 explicitly withdrew —
  the correct claim is "non-inferiority not demonstrated; cause not established". Same file omits
  `fn_margin=0.02` and `use_provided_hardneg=True` (live in the shipping loss,
  `m7src/train.py:625-627`, present in the committed cfg `results/m7_run_p35w-2m-s2500.json`) and
  omits fusion entirely. RECIPE.md:112 carries the stale 0.926 retention (see Q6). **Action:**
  rewrite the section header to "Four removals whose joint non-inferiority failed its bar", add the
  two knobs to the phase blocks, add a fusion section after freeze-prep step 1, correct 0.926 →
  0.915. One hour, before the freeze commit makes it permanent.
- **B2 — Report framing: the lever programme may not be presented as having improved the released
  system.** Quantified in Q1: expected optimism +0.008..+0.015 of the +0.0166 post-gate dev gain;
  matched-control attribution +0.005 in-band; OOD movement ~0.000; expected six-set transfer
  0.000 ± 0.005. **Action:** the report leads with the matched-control attribution
  (`results/m7_compare_full_postabl.json`) and the perturbation band, never the chained +0.0126;
  and it states before the final run that a release-bar miss is a publishable outcome (projections
  0.425–0.509 straddle 0.4583).

### MAJOR

- **M1 — Training data is not revision-pinned.** `m7src/trainmix.py:56,59,62,91,103,142,145` — bare
  `load_dataset` calls. **Action:** before freeze, record the HF commit SHA of each source repo as
  currently cached (a lookup against the local `datasets` cache, no re-download) into
  `results/m7_field_table.md`; add `revision=` to the calls in the same commit.
- **M2 — Teacher `trust_remote_code` code is unpinned** (CODEMAP.md:105). **Action:** vendor the
  stella modeling files (or pin the code repo SHA) into the freeze; the doc side of every
  third-party reproduction depends on it.
- **M3 — "OOD effect is zero" overclaims exceed the instrument.** `m7/LEDGER.md:268` (and echoes in
  STATUS.md:19-21, RECIPE.md:99-101). Per-arm OOD resolution is ~±0.005 at n=1,915, one dataset
  family. **Action:** re-word to "unresolved below ~0.005 on a StackExchange-only proxy" everywhere
  except the eight-arm ablation span, which may stand as written.
- **M4 — Fusion not yet selected for the shipping candidate**; only `p1-objB`-era fusion artifacts
  exist, and every one is declared superseded by LEDGER.md:852-854. **Action:** run
  `run_freeze_prep.sh p35w-2m-s2500` step 1, commit `m7_fusion_*<candidate>*.json`, document the
  selected w (or the dense-only endpoint) in RECIPE.md before freeze.
- **M5 — Rule-compliance is discovered, not audited** (Q5.3). **Action:** one pre-freeze pass:
  enumerate every pre-registered rule in LEDGER.md against every arm family it binds; record the
  matrix in the ledger. The step-rule was found unapplied by luck; the next one should be found by
  procedure.
- **M6 — The pre-freeze queue must be closed out, not left dangling.** STATUS.md:56-66 still queues
  teacher probes (arctic-m, gte-base — a swap is pre-registered as before-freeze-or-never,
  LEDGER.md:774-780) and lever #7 (pre-registered, unrun). **Action:** either run them or write
  one-line closures in the ledger *before* freeze; a pre-registered, unrun, unaccounted lever is
  exactly what an external reviewer will read as selective abandonment.

### MINOR

- **m1 — Determinism overclaim.** LEDGER.md:349, 383-384, 395-396 vs raw CI [0.0, 1.34e-5] in
  `results/m7_compare_full_ablations.json`. Fix the wording to "replay agrees to ≤1.3e-5 raw
  (display-rounds to zero); proxy path bit-identical"; the ledger's own raw-endpoint rule demands it.
- **m2 — Stale reuse counts.** STATUS.md:35-36 and CODEMAP.md:84 say 53/299/74;
  `results/m7_dev_reuse_count.json` says **58 / 322 / 90**. The report must quote the JSON.
- **m3 — Committed decision artifact carries refuted text.** `results/m7_simplify_decision.json`
  ("4x less pseudo-query generation" — corrected to 2.85x in LEDGER.md:465-470; and "evidence that
  individually-inert components interact" — withdrawn LEDGER.md:411). Don't edit the artifact; add
  the correction pointer to its ledger entry and ensure the report quotes the corrected values.
- **m4 — Number drift.** `m7src/program.py:203` says p1-objA declined "to 0.3366"; every other
  record says 0.3248 (`m7_dev_reuse_count.json`, RESULTS.md:25). Fix the docstring.
- **m5 — Uncommitted evidence.** ~40 `results/m7_*.json` from today are untracked (git status),
  including `m7_compare_full_steprule.json` and `m7_compare_full_simplify.json` — the inputs the
  committed decision artifacts cite. `final_run.py` refuses a dirty tree, so this self-heals, but
  commit them now so the decision executors' provenance chain is inspectable.

## The direct answers, one line each

1. **Optimism ≈ +0.01 (of +0.017) on dev; expected six-set transfer of the post-gate search
   0.000 ± 0.005; the confirmatory number itself is unbiased and the tier framing survives — the
   lever narrative does not.**
2. **sqrt pooling and the 2m dose pick fail any whole-search correction; the step extension
   survives it numerically but is the nuisance parameter itself; only learned weights and
   the 500k adoption survive on their own terms.**
3. **One component is evidence-earned (learned weights); four ship on a joint test that cannot
   attribute; five loss hyperparameters ship with no test at all; the margin was anchored to noise
   but the failure is margin-robust (CI lower −0.0102 fails any margin ≤ 0.0102).**
4. **Reproducible from the repo (committed run cfgs), NOT from RECIPE.md alone; unpinned HF data
   revisions and unpinned trust_remote_code are the two real gaps; fusion is undocumented.**
5. **The correction was legitimate (pre-registered before its numbers, honoured against interest,
   commit-order verified); the exposed weakness is event-driven rule auditing — fix with M5.**
6. **The unchecked belief: the two-component StackExchange pair as a generalisation oracle read
   below its resolution; and the found errors: retention 0.926 belongs to the reverted candidate
   (real figure 0.915), and "deterministic to 16 digits" is contradicted by its own committed
   artifact.**
