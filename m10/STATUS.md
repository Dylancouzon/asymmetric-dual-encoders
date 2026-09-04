# M10 status — PLANNED, budget VALIDATED 2026-09-04, plan re-cut the same day; nothing has trained

Mandate `instructions-m10.md` (read **§Amendment 2026-09-04** and **§Amendment 2026-09-04b** first — they are authoritative over any older sentence there; the **Weekend runbook** below is the execution order) · evidence `m10/PLANNING.md` (§11 measured rates, §12 the synthetic-data
question) · runs `m10/RESULTS.md` · closed avenues `m10/EXPLORED.md` · selection-surface drafts
`m10/COV_CANDIDATES.md` · code `m10/CODEMAP.md` · M9's record `m9/FINDINGS.md`.
Owner report: https://claude.ai/code/artifact/fce61c94-5444-4c78-bb2e-46112cb7547a

**Where things stand.** The 2026-09-01 plan went through seven Codex passes and Dylan's compute
ruling. On 2026-09-04 he validated the budget and asked for a full efficiency review before any
money was spent ("is this the best way … is it the most efficient?", "I'm not sure why we need to
generate synthetic data?", "don't over-engineer", "keeping the same teacher (Stella) is the goal").
The review produced amendments **A1–A8**. What changed, shortest form:

- **Measured, not assumed:** the M10 recipe runs at **683 examples/s blended (718 query-bucket / 596 document-bucket) on the box**
  against the plan's imported 560 on a rented A100 — the step is launch-bound at batch 32
  (`results/m10_rate_bench_box.json`). The box is an execution target again for everything except
  generation; the build is ≈81 GPU-hours at the hardware bound, not 100; cloud spend re-prices to **≈$110–280** hybrid.
- **Less machinery:** family D's three ranking-aware arms cut (LEAF's ‖e‖₂ loss plus **D-COV**, a
  document-covariance-weighted regression, in their place), which deletes the candidate bank, the
  mining pass, the HNSW fallback, the τ rule and the seed-rank field. Fifteen arms, fourteen
  contrasts. Confirmations capped at two decisions. The full-dose seed-1 replica is withdrawn.
  A **registered plateau response** replaces the cut class, so a flat curve still has an answer.
- **Mostly real text, not generated:** ≈1.5M harvested titles / headings / claim sentences from the
  licensed pool as new arm A3, generation cut to ≈1.0M for the six forms no corpus contains. Three
  of the four clean-4 headline datasets fall in the harvested half.
- **Two protocol fixes:** C1/C2 are registered on **clean-4 as well as avg-6** under
  fixed-sequence gatekeeping (bars 0.5046 / 0.5233), because M14 made clean-4 the headline for both
  halves of the pair; and the COV resolution number is reported as the screen's power disclosure (its sizing role was
struck the same day by the Codex pass).

**Then Fable reviewed the amendments and returned 3 BLOCKER / 8 MAJOR / 7 MINOR — all actioned**
(`research/m10-fable-plan-2026-09-04.md`, dispositions in the mandate; read-exclusion audit clean).
The four that changed the plan most: the A1 cut had cited LEAF's Appendix B, which is about
intermediate-layer KD and not about a ranking term on regression, so the justification was corrected
and a plateau response registered; "Holm" and "fixed sequence" were named together and are
incompatible, so it is now gatekeeping and avg-6 loses no alpha; **the document pool is Wikipedia
plus ESCI and contains no scientific text, so "harvest paper titles and scientific claims" was
unfounded** — arXiv metadata is added as a licence-gated source and every harvest yield is measured
before quotas lock; and the benchmark was re-run because the shape 25% of build steps will use had
never been timed.

**What the review means for the numbers above:** the 683 examples/s is a *hardware* bound. M9's
realized pipeline ran at ~10% of the comparable roof, so **the build is priced as a range and the
real-data re-measure gates every dollar** (PLANNING §11). Do not commit to a box-versus-cloud split
on the optimistic end.

**Second review the same day (feasibility; B1–B6, `instructions-m10.md` §Amendment 2026-09-04b;
`research/m10-feasibility-review-2026-09-04.md`).** Verdict: **C1a reachable if coverage works; C1b
and C2a are the contest at ~91–92% uniform retention; C2b (~95%) is a low-prior stretch aim with no
pure-regression precedent at our teacher gap.** Weakness found and fixed: every clean-4 set is scientific/biomedical and the COV screen
had no surface that could see those forms (B4, `arxiv-title`). G-MLP, a per-token nonlinear head,
is proven servable and replaces the 768 arm (B3). Decision 12 (CUREv1 as a validation-only diagnostic) is Dylan's. **Then a Codex pass and an Opus pass
(`research/m10-codex-feasibility-2026-09-04.md`, `research/m10-opus-review-2026-09-04.md`), every finding
actioned — read the mandate, not this paragraph, for the rules.**

## Dylan — open decisions (defaults apply meanwhile)

| # | decision | default |
|---|---|---|
| 1 | Ratify M9's final lock plus the six-only amendment (one sentence: "run M9's six-set scoring as registered, six only, no reserved batch") | blocks only the M9 close-out |
| 4 | PAQ as query text (CC BY-SA data, official release) | include |
| 7 | Confirm LoTTE read #1 withdrawal and the renumbering | as recorded |
| 10 | The 2026-09-04 amendments A1–A8 | adopted; strike any item and it reverts to the 2026-09-01 text |
| 11 | Release rule: does C1a-pass / C1b-fail ship as "nano"? | **ruled 2026-09-04:** default stands (release needs C1b); "make sure we win enough so this isn't a question" |
| 12 | CUREv1 as a validation-only biomedical read | **adopted 2026-09-04** as a reported diagnostic, never selection-bearing |
| 13 | The 2026-09-04b amendments B1–B6 | adopted; strike any item and it reverts |
| 14 | **Generation on the box** with Qwen's official `Qwen3-8B-AWQ` via vLLM, smoke pushed as `m10/SMOKE.md` for remote approval; hosted bf16 as fallback | **adopted 2026-09-04 ("Go on 14")** |
| A7 | the box runs the screens | **confirmed 2026-09-04**; three uninterrupted box days over the weekend |
| — | **Generation smoke approval** — you are the approver (200 queries × 12 forms, 90% contract / 80% on-form). It runs on the box (decision 14); the sample is pushed as `m10/SMOKE.md` and a GitHub issue "M10 smoke approval" is opened — reply `approved: <forms>` there, or name the forms to redraft | **the one thing the weekend needs from you**; blocks generation and therefore family F |

Decisions 2 (budget), 3, 5, 6, 8 and 9 are closed — see `instructions-m10.md` §Owner decisions.

## Weekend runbook — unsupervised, 2026-09-05 → 09-08 (Dylan follows on GitHub; decision 14 adopted)

Standing rules: commit-and-push after every completed step; smoke every code path at 90 steps before
a long run; arm the failure-signature monitor; read the first rate line; `setsid nohup` for anything
long; zero cloud spend; no six / reserved / LoTTE read; dev reads counted; **no lock edit after an arm
starts and no protocol change after a number is observed**. Anything not covered below: stop, record
here, wait. Every step names its branch, so nothing below needs a judgement call.

| step | what | branch |
|---|---|---|
| **0a** (tonight) | vLLM in its own venv (`.venv-gen`, not the trainer's), `Qwen/Qwen3-8B-AWQ` rev `4da05a8e…` served on the card; a 50-prompt throughput smoke | works → 1; fails after 3 h of setup → **generation falls back to hosted bf16 after Dylan returns**; the weekend runs 0b–7 only and **no arm starts** (the anchor needs the generated half) |
| **0b** (tonight) | rate re-measure on real tokenized corpora with `num_alloc_retries` logged; `torch.compile` on the fixed buckets | informational; pushed to `m10/RESULTS.md`; the box-vs-cloud build decision is the lock's (§0b), not the window's |
| **1** | 12-form smoke, 200 per form, with the AWQ artifact → push **`m10/SMOKE.md`** (contract rate per form; a 50-query on-form sample per form) and open a GitHub issue "M10 smoke approval"; poll it hourly with `gh` | Dylan approves per form by commenting `approved: <forms>`; a form under 90% contract gets one prompt revision (≤2, each recorded in LEDGER §1) and a re-smoke; **no approval by Sunday noon → generation waits, F cannot start, the weekend delivers 2–7**; `gh` unavailable → same, approval arrives with Dylan's next session |
| **2** | COV admission (M10.0-d): MedicalQA, BRIGHT, CorporateLobbying, ConsumerContractsQA; LEDGER if its structure verifies; **CUREv1 as a diagnostic** (decision 12); **`arxiv-title` diagnostic** drawn by id-without-version, seed 0; every admitted corpus into the protected index; stella encodes; the resolution number pushed | **fewer than three families admit → STOP and return to Dylan** (registered) |
| **3** | harvest pipeline and yields (arXiv Kaggle CC0 — record artifact and revision; Wikipedia titles / headings / lead sentences; ESCI), post-dedup and post-screen counts pushed to LEDGER §1 | a form under 100K → reverts to generation (registered); quotas fixed only after the yields are pushed |
| **4** | PAQ from Facebook's official release; the 1.0M build sample and the 4.037M A2 control (seed 0, hashes pinned) | — |
| **5** | trainer port: per-token heads with pooling after the head, token-output export wrapper and their parity test; 4-step mix window; cyclic schedule; ‖e‖₂ and D-COV arms; `test_resume.py`; examples/s counter; 90-step smoke of every arm shape | any smoke failure is fixed before any arm; nothing else changes |
| **6** | M10.0-c: per-component DEV-6 read of the M9 candidate incl. `heldout-longq` | descriptive baseline row |
| **7** | parity checks for MiniLM-L6 / L12 three- and four-layer heads (CPU) | a failing head disqualifies that arm, reported |
| **8** (after approval) | generation ≈1.0M under the §Data contract (seeds pre-filtered; strict JSON; one retry; dedup) → decontamination against the protected index and the six's documents → **A8 gates** → FORMS-12 hold-out → teacher targets → `results/m10_data_manifest.json` | a form over 25% near-duplicates → quota cut to its unique count (registered); the MS MARCO overlap row is disclosed, no action |
| **9** | push **LEDGER §0a** (design) — it must precede any arm — then **§0b** (counts, hashes, measured rates, allocation) | no arm before both are on origin |
| **10** | **family F**: anchor bge-small 20M (read 5 / 10 / 20M), MiniLM-L6 20M, L12 5M probe (extended iff within the MDE of the better 5M reading); COV at every cycle end, DEV-6 once | rule registered in §Screen; the winner is the build student; ties → cheaper to serve, labelled a product preference |
| **11** | **family A** on the winner: A1, A2, A3 at 5M (A4 = the winner's 5M checkpoint) | three-outcome rule on A3−A2 (COV macro); A4−A3 decides the generated half; **A3−A2 fails → M10 STOPS before any build and returns to Dylan with all four rows** (registered) |
| window ends | a running arm finishes under the watchdog; nothing new starts; this file records where things stand | — |

**Needs Dylan during the window:** the smoke approval only. **Cannot happen:** a cloud instance, a
protected read, an arm before §0a/§0b, a build. Realistic yield: steps 0–9 and family F; family A if
the smoke is approved by Saturday.

## Then, in order

Families G → B → E → C → D on F's winner; ≤2 confirmations; the synthesized selected-recipe arm;
the recipe lock (Codex and Fable review it); M9's six-only close-out from `m9-work`; LoTTE read #1
→ cloud instance only if §0b puts the build there → M10.3 build (200M, whole extension cycles) →
export, parity, freeze, LoTTE read #2 → M10.4 final: the six-set transaction in the order C1b → C1a
→ C2a → C2b, then the reserved conditional.

## Guardrails that bite here

No six/reserved/LoTTE access outside the registered transactions. `results/perquery.json` is never
rewritten. Never edit a `guard9` protocol-scope file before M9's close-out runs. Every review brief
carries the reserved read-exclusion; audit the log after. Long runs: smoke, arm the
failure-signature monitor, check the rate, watch the machine. Stella on the Mac runs only in
`.venv-mac`. A stopped cloud instance costs disk only; an idle running one costs the budget.
