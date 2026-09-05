# M10 status — weekend window RUNNING. Steps 0a, 0b and 1 are DONE; nothing has trained; the runbook below is the execution order

## Weekend progress (2026-09-04 evening, updated live)

| step | state | outcome |
|---|---|---|
| **0a** vLLM + generator | **DONE, PASSES** | vLLM 0.28.0, attempt 1, no fallback. Health: contract 93.75% (gate 90%), **1027–1173 aggregate output tok/s** (gate 700). Generation stays on the box; the bf16 rented-GPU fallback does NOT fire. ≈1.0M queries ≈ 10 box-hours. `results/m10_gen_health_box.json` |
| **0b** rate re-measure | **DONE** | Real corpora, real memmapped targets: bucketed + prefetched + `torch.compile` gives **914–960 ex/s** query / **792** document, blended ≈890 → **~62 GPU-h for the 200M build on the box** (vs §11's 683 bound, M9's realized 226). `results/m10_rate_bench_real_box.json`, RESULTS.md |
| **1** generation smoke | **DONE — all 7 forms pass both gates**, but 3 are HELD | yesno 100 · conversational 96 · argument 88 · finance 86 · comparison 84 · health 84 · howto 80.0. `m10/SMOKE.md`, GitHub issue **#1**. Four forms auto-approving; **`howto`, `argument`, `conversational` held for Dylan** — three procedural defects found after the numbers were observed (LEDGER §1, §3 W1–W2) |
| **2** COV admission | in progress | structures probed (`work/m10cov/structure.json`); licence drafts in `m10/COV_CANDIDATES.md` |
| 3–11 | not started | — |

**Reaching Dylan:** `PushNotification` is **disabled in /config**, so the phone route does not work.
**GitHub issue #1 "M10 smoke approval" is the live channel**; he can also reply via Remote Control.

**Three questions are open for Dylan and logged as `m10/LEDGER.md` §3 W1–W3.** W1 binds the rest of
M10: §Data registers three different terminal rules for a form that fails twice (drop · ≤2
revisions · bf16 re-smoke). None is resolvable here — the numbers they govern are already observed,
which is Tier 3. **Nothing generates before COV admission and the build seed draw, so waiting costs
nothing** and every other branch continues.

**Serving facts a resuming session needs:** `work/m10gen/serve.sh` (port **8001**);
`VLLM_WSL2_ENABLE_PIN_MEMORY=1` and `VLLM_USE_FLASHINFER_SAMPLER=0` are both required on this box;
`--gpu-memory-utilization 0.88` (only 8.86 of 10 GiB is free); **never `--enforce-eager`** — it
costs 1.69× and is the difference between failing and passing the 700 floor. vLLM lives in
`.venv-gen`, the trainer in `.venv`. vLLM holds 8.7 GB, so it and a training arm cannot share the
card — stop one before starting the other.

Mandate `instructions-m10.md` — §Amendment 2026-09-04, §Amendment 2026-09-04b and §Delegated
authority are authoritative over any older sentence there · evidence `m10/PLANNING.md` (§11 measured
rates, §12 the synthetic-data question; §5–§6 superseded) · lock `m10/LEDGER.md` §0a/§0b · runs
`m10/RESULTS.md` · closed avenues `m10/EXPLORED.md` · surfaces `m10/COV_CANDIDATES.md` · code
`m10/CODEMAP.md` · M9's record `m9/FINDINGS.md`.
Owner report: https://claude.ai/code/artifact/fce61c94-5444-4c78-bb2e-46112cb7547a

**Where things stand.** Budget validated 2026-09-04. The same day the plan was re-cut (A1–A8), given a
feasibility review (B1–B6), then attacked by Codex, Opus and Codex again; every finding is actioned and
the records are `research/m10-fable-plan-2026-09-04.md`, `research/m10-feasibility-review-2026-09-04.md`,
`research/m10-codex-feasibility-2026-09-04.md`, `research/m10-opus-review-2026-09-04.md`,
`research/m10-codex-soundness-2026-09-04.md`. **Read the mandate for rules, not those files.**
Feasibility verdict: **C1a reachable if coverage works; C1b and C2a are the contest at ~91–92% uniform
retention; C2b (~95%) is a low-prior stretch aim** (`results/m10_conjunct_arithmetic.json`). Measured
trainer rate on the box: **683 examples/s blended** (718 query-bucket / 596 document-bucket), a hardware
bound; M9's pipeline ran at ~10% of its roof, so the real-data re-measure (step 0b) gates every plan.

## Dylan — decisions (all ruled 2026-09-04 unless marked)

| # | decision | state |
|---|---|---|
| 1 | Ratify M9's final lock plus the six-only amendment (one sentence: "run M9's six-set scoring as registered, six only, no reserved batch") | open; blocks only the M9 close-out, which runs after M10.2 |
| 4 | PAQ as query text (CC BY-SA data, official release) | default: include |
| 7 | Confirm LoTTE read #1 withdrawal and the renumbering | default: as recorded |
| 10, 13 | Amendments A1–A8 and B1–B6 | adopted; strike an item to revert it |
| 11 | Release rule: release needs C1b (the headline) | default confirmed: "make sure we win enough so this isn't a question" |
| 12 | CUREv1 as a validation-only biomedical **diagnostic** | adopted ("yes") |
| A7 | the box runs the screens | confirmed; three uninterrupted box days over the weekend |
| 14 | generation on the box with Qwen's official `Qwen3-8B-AWQ` via vLLM; self-hosted bf16 via the same contract as the only fallback | adopted ("Go on 14") |
| 15 | conditional pre-approval of the generation smoke (contract rate by the session, on-form rate by an independent Fable subagent, six-hour veto window) | adopted ("Yes to decision 15") |
| — | delegated authority for unsupervised windows: Tier 1 alone, Tier 2 after a Fable consultation logged in LEDGER §3, Tier 3 never | granted (mandate §Delegated authority) |

Decisions 2 (budget), 3, 5, 6, 8 and 9 are closed — `instructions-m10.md` §Owner decisions.

## Reaching Dylan

`PushNotification` reaches his phone **only if Remote Control is connected to the running session**;
otherwise it is a terminal notification. Ping at exactly these moments, one line, under 200
characters, leading with the action: the smoke sample is ready; a registered STOP fires; a fallback
fires; a Tier-2 decision was taken; family F's verdict lands. Never for progress. The GitHub issue
"M10 smoke approval" (opened by the session with `gh`; account `Dylancouzon` is the approver) is the
channel that works without Remote Control; he can also reply through Remote Control.

## Weekend runbook — unsupervised, 2026-09-05 → 09-08 (decisions 14 and 15; §Delegated authority applies)

Standing rules: commit-and-push after every completed step; smoke every code path at 90 steps before
a long run; arm the failure-signature monitor; read the first rate line; `setsid nohup` for anything
long; zero cloud spend; no six / reserved / LoTTE read; dev reads counted; **no lock edit after an arm
starts and no protocol change after a number is observed.** A situation no row covers is a Tier-2
decision (mandate §Delegated authority): consult a Fable subagent, decide, log, push, ping.

| step | what | branch |
|---|---|---|
| **0a** (tonight) | **vLLM setup**, timer starts at the first `pip install`: `python3.12 -m venv .venv-gen && .venv-gen/bin/pip install vllm==0.28.0` (attempt 1); attempt 2 = `vllm==0.27.1`; no third. Serve `Qwen/Qwen3-8B-AWQ --revision 4da05a8edb55c6046cce958586c33b61da07bb79 --max-model-len 4096 --gpu-memory-utilization 0.85 --reasoning-parser qwen3`, thinking disabled per request (`chat_template_kwargs={"enable_thinking": false}`). **Health assertion:** `/v1/models` lists the model, and a 64-prompt batch (8 seeds × 8 forms, n=5) returns ≥ 90% contract-valid replies at **≥ 700 aggregate output tok/s** | pass → step 1; either attempt fails, or the health assertion fails, or 3 h elapse → **fallback: self-hosted bf16 via the same vLLM contract on a rented GPU after Dylan returns**; the weekend runs 0b–7 only, **no arm starts** (the anchor needs the generated half); ping |
| **0b** (tonight) | rate re-measure on real tokenized corpora with `num_alloc_retries` logged; `torch.compile` on the fixed buckets | informational, pushed to `m10/RESULTS.md`; the box-vs-cloud build decision belongs to the M10.2 lock |
| **1** | **Smoke state machine (decision 15).** For each of the **seven generated forms** (keys in `m10src/forms.FORMS`: `howto argument finance comparison yesno conversational health`) generate 200 queries; state per form = `(form, prompt-hash, smoke-commit, state)`. Gates: contract ≥ 90% (session); on-form ≥ 80% on a 50-query sample judged by an **independent Fable subagent** given only the form's registered description (verdicts pushed). Push `m10/SMOKE.md`; open the GitHub issue; ping. A form that passes both gates is **auto-approved at push time + 6 h** unless a comment `redraft: <form>: <note>` from `Dylancouzon` (or a Remote Control message) arrives; `approved: <form>[, <form>]` ends the window early for those forms. Approval binds to the prompt-hash | a form failing a gate or vetoed → one prompt revision (≤ 2 total per form, each hash recorded in LEDGER §1) → re-smoke → new window; **two failures → the form is dropped from the build**, quota not redistributed, reported; a veto after generation started → that form's queries leave the build and are regenerated under a redraft if time permits. Forms are independent: approved forms generate while others redraft |
| **2** | **COV admission (M10.0-d):** MedicalQA, BRIGHT, CorporateLobbying, ConsumerContractsQA; LEDGER if its structure verifies (chunk rule, 100K cap); **CUREv1 as a diagnostic** (decision 12); every admitted corpus into the protected index; stella encodes; the resolution number pushed | **fewer than three family IDs** (`consumer-health`, `BRIGHT`, `legal`, `finance` iff LEDGER) **→ STOP, ping, wait for Dylan** (registered) |
| **3** | **Push LEDGER §0a** — the design lock: arms, order F → A → G → B → E → C → D, doses (B: 3.75M / 5M / 7.5M), seeds, surfaces, the thirteen contrasts by name, MDE 0.0056, 0.025/13, B = 200,000 `inverted_cdf`, confirmation design, warm-start record, outcome→action maps | must be on origin **before any harvest** (Codex B4) |
| **4** | **Harvest.** First: download the Kaggle arXiv metadata artifact, record version and sha256, **draw `arxiv-title`** (sorted version-stripped ids, `default_rng(0).choice(N, 100_000, replace=False)`, first 2,000 = queries), protect and encode it. Then extraction rules over arXiv, Wikipedia and ESCI; post-dedup, post-screen yields pushed to LEDGER §1 | a form under 100K → reverts to generation at ≈143K (registered); quotas fixed only after the yields are pushed |
| **5** | PAQ from Facebook's official release; the 1.0M build sample and the 4.037M A2 control (seed 0, hashes pinned) | — |
| **6** | trainer port: per-token heads with pooling after the head, token-output export wrapper and their parity test; the 4-step mix window (4Q · 3Q+1D · 2Q+2D); cyclic schedule; ‖e‖₂ and D-COV arms; the kill and plateau rules as registered; `test_resume.py`; examples/s counter; 90-step smoke of every arm shape | any smoke failure is fixed before any arm; nothing else changes |
| **7** | M10.0-c: per-component DEV-6 read of the M9 candidate incl. `heldout-longq`; parity checks for MiniLM-L6 / L12 heads (CPU) | descriptive baseline; a failing head disqualifies that arm, reported |
| **8** (as forms clear their windows) | generation ≈1.0M under the §Data contract → decontamination against the protected index and the six's documents → **A8 gates** (near-duplicate rate > 25% → keep representatives only; < 50,000 retained → form dropped) → FORMS-12 hold-out → teacher targets → `results/m10_data_manifest.json` | the MS MARCO overlap row is a diagnostic, no action |
| **9** | **Push LEDGER §0b** — corpus counts and hashes, arXiv artifact sha256, served generator revision and vLLM version, local measured rates, the screen's box allocation | no arm before §0a and §0b are both on origin; cloud price and build allocation are NOT here (M10.2) |
| **10** | **family F:** anchor bge-small 20M (read 5 / 10 / 20M), MiniLM-L6 20M, L12 5M probe (extended iff `m_L12 ≥ max(m_L6, m_bge) − 0.0056`); COV at every cycle end, DEV-6 once | winner = highest 20M macro if its margin resolves, else cheapest to serve (labelled a product preference); ping the verdict |
| **11** | **family A** on the winner: A1, A2, A3 at 5M (A4 = the winner's 5M checkpoint) | three-outcome rule on A3−A2 (COV macro); A4−A3 decides the generated half; **A3−A2 fails → M10 STOPS before any build; ping; wait for Dylan** (registered). A's verdict is labelled conditional on F's winner |
| window ends | a running arm finishes under the watchdog; nothing new starts; this file records where things stand | — |

**Needs Dylan during the window:** nothing, unless he vetoes a form or a STOP fires. **Cannot happen:**
a cloud instance, a protected read, an arm before §0a and §0b, a build, any Tier-3 decision.
Realistic yield: steps 0–9 and family F; family A if every form clears its window by Saturday.

## Then, in order

Families G → B → E → C → D on F's winner; ≤ 2 confirmations (stability checks, labelled); the
synthesized selected-recipe arm; **the M10 `decide()` for four conjuncts under gatekeeping, written
and unit-tested**; the recipe lock with cloud price, build allocation and `max_extension_cycles`
(Codex and Fable review it); M9's six-only close-out from `m9-work`; LoTTE read #1 (decision-bearing:
the veto) → cloud instance only if the lock puts the build there → M10.3 build (200M, whole extension
cycles) → export, parity, freeze, LoTTE read #2 (audit) → M10.4 final: the six-set transaction in the
order C1b → C1a → C2a → C2b, then the reserved conditional.

## Guardrails that bite here

No six/reserved/LoTTE access outside the registered transactions. `results/perquery.json` is never
rewritten. Never edit a `guard9` protocol-scope file before M9's close-out runs. Every review brief
carries the reserved read-exclusion; audit the log after. Long runs: smoke, arm the
failure-signature monitor, check the rate, watch the machine. Stella on the Mac runs only in
`.venv-mac`; vLLM lives in `.venv-gen`, never in the trainer's `.venv`. A stopped cloud instance costs
disk only; an idle running one costs the budget.
