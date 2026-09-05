# M10 status — weekend window RUNNING. Steps 0a, 0b, 1, 2 and 3 are DONE; nothing has trained; the runbook below is the execution order

## Weekend progress — 2026-09-05. READ THIS FIRST, then the mandate.

| step | state | outcome |
|---|---|---|
| **0a** vLLM + generator | **DONE, PASSES** | vLLM 0.28.0, attempt 1, **no fallback** — generation stays on the box, no cloud spend. Contract 93.75% (gate 90%), **1,027–1,173 aggregate output tok/s** (gate 700). ≈1.0M queries ≈ 10 box-hours. `results/m10_gen_health_box.json` |
| **0b** rate re-measure | **DONE** | bucketed + prefetched + `torch.compile` on fixed buckets gives **914–960 ex/s** query-bucket, **792** document-bucket, blended ≈890 → **~62 GPU-h for the 200M build on the box**. `results/m10_rate_bench_real_box.json` |
| **1** generation smoke | **DONE — all 7 forms APPROVED by Dylan** | Contract 100% everywhere after 7 prompt revisions across 4 forms. `m10/SMOKE.md`, GitHub issue **#1**. `argument` ships with **67%** (full-output) as its honest rate |
| **2** COV admission | **DONE — four families** | MedicalQA · BRIGHT (6 slices) · CorporateLobbying · ConsumerContractsQA · **LEDGER**. 13,416 queries, 4 family IDs. All screened clean. Teacher-encoded. LEDGER §2 |
| **2b** COV **resolution number** | **DONE — W5 ANSWERED** | **0.008619**, paired SD 0.00302. The registered **MDE 0.0056 sits BELOW it**, so a contrast landing at the MDE cannot resolve — the bottom of §Surfaces' predicted band. **More data will not fix it:** BRIGHT carries 50.0% of the macro variance and legal 32.5%, against LEDGER's 2.7% for 10,000 of the 13,416 queries — family-equal weighting means LEDGER bought dilution, not power. Dylan's three options stand; the registered default (report unresolved as unresolved) holds unless he moves the MDE, which is Tier 3 and must happen before any arm. `results/m10_cov_resolution.json`, LEDGER §3 W5 and §4 |
| **3** **§0a design lock** | **DONE, PUSHED, then AMENDED S1–S10** | The lock is **`m10/screen_registry.json`**, not prose; `m10src/screen_lock.py` validates it (18 tests). **S1 was a structural confound that would have shipped:** the anchor was "F's winner at its 5M checkpoint" — ~75% through cycle 1 of a 20M schedule, un-annealed — against genuine three-cycle 5M arms, on the `b` side of **ten** contrasts. It is now **its own trained 5M arm**: 16 trained arms, +2 GPU-h. LEDGER §0a |
| **protected10** | **DONE (was runbook item 3 of the next-three)** | `seeds.draw` now screens against the M10 protected index — six + dev + reserved queries **plus admitted COV queries AND documents**. 464,757 COV fingerprints cached. `SCREEN_VERSION` bumped; `_key` no longer hardcodes `ROUTE` |
| **7 (part)** family-F parity | **DONE — all six heads PASS** | bge-small · MiniLM-L6 · MiniLM-L12 at 3 and 4 layers, min-cos ≥ **0.99999988** through fastembed, zero custom ops, all under the 35M cap. A first run read 0.93–0.95 and would have disqualified both MiniLM students; diagnosed, see the hard-won facts. `results/m10_student_parity_box.json` |
| **W3 seed supply** | **SOLVED in principle; the precision gate is the open item** | New store **`wikipedia-body`** (T2-5, Tier 2, Fable-reviewed, registered before its scan): body paragraphs (**lead excluded**) of `wikimedia/wikipedia@20231101.en`, **unchanged** `ROUTE` and `min_score ≥ 4`, cap 3 seeds/article/form, health and finance only. Full-dump scan running at ~2,000 articles/s. The shortfall is closed by corpus size, not by lowering any bar |
| **4–11** | not started | next: finish the scan → draw → the judged precision gate; then step 4 harvest |

## THE NEXT THREE THINGS, in order

1. **Finish `wikipedia-body` and gate it.** `m10src/wikibody.py`: `scan()` (running) → `draw(per_form)`
   → **the judged precision gate**. The gate is the whole decision and it is what killed
   `ROUTE_WIDE`: **200 uniform-random seeds per form from the population `draw()` actually
   returns** (top-score-first), judged by an independent Fable subagent against the frozen
   `m10src/forms.RUBRIC`, **≥ 80% hard**, plus a **200-seed control from the incumbent `ROUTE`
   intro seeds judged in the same run** (reported, never an escape clause). Fail → not adopted;
   the registered next lever is **category-membership routing** over the `categorylinks` dump,
   never a relaxed floor. **Do not adopt on the counts alone** — asserting precision instead of
   measuring it is what sank `ROUTE_WIDE` v1.
2. **Step 4, harvest.** §0a is on origin, so the harvest is unblocked. Download the Kaggle arXiv
   metadata artifact, record version and sha256, **draw `arxiv-title`** (sorted version-stripped
   ids, `default_rng(0).choice(N, 100_000, replace=False)`, first 2,000 = queries), protect and
   encode it; then the extraction rules over arXiv, Wikipedia and ESCI, yields to LEDGER §1.
3. **Step 6, the trainer port** — per-token heads with pooling after the head, the token-output
   export wrapper and its parity test, the 4-step mix window, the cyclic schedule, ‖e‖₂ and D-COV,
   the kill and plateau rules, `test_resume.py`, the examples/s counter, and a **90-step smoke of
   every arm shape** before any arm.

## Hard-won facts a resuming session must not rediscover

- **Serving:** `work/m10gen/serve.sh`, **port 8001**. `VLLM_WSL2_ENABLE_PIN_MEMORY=1` and
  `VLLM_USE_FLASHINFER_SAMPLER=0` are both **required** on this box (WSL kills pinning → vLLM fails
  on UVA; flashinfer JITs kernels and CUDA 12.6's nvcc rejects gcc 15). `--gpu-memory-utilization
  0.88` — only 8.86 of 10 GiB is free. **Never `--enforce-eager`**: it costs 1.69× and is the
  difference between failing and passing the 700 tok/s floor. vLLM in `.venv-gen`, trainer in
  `.venv`; vLLM holds 8.7 GB so it and a training arm cannot share the card.
- **`load_dataset_builder(...).info.features` can be STALE.** It cost the LEDGER admission a wrong
  refusal (8 columns listed, 13 actually present, `qrels` among the missing). Load the rows.
- **Sub-8-word boilerplate fakes contamination.** BRIGHT's 6,123 "exact" hits and LEDGER's 710 are
  near-empty pages; above the 8-word fingerprint floor both are ~0. Always re-screen length-filtered.
- **fastembed serves `min(model_max_length, max_length)` from `tokenizer_config.json`.**
  `all-MiniLM-*-v2` ships `max_length` 128 beside `model_max_length` 512, so fastembed truncated at
  128 against a 512-token torch reference and the parity check read 0.93–0.95 min-cos — a
  mis-specified check that would have disqualified both MiniLM students and gutted family F. Every
  text under the limit was bit-exact. The export writes the tokenizer we intend to ship. bge-small
  ships no `max_length` key, which is why it alone read 1.0.
- **The validator and the tests run only under `.venv/bin/python`** — the system python has no
  numpy, and that ImportError is not a lock failure.
- **`m10src/forms.RUBRIC` is the frozen gate standard; `forms.FORMS` is the revisable prompt.**
  Never judge against the prompt — `argument` reads 8% that way and 88% against the rubric.
- **`torch.compile` is training-only** (rules in `m10/HEADROOM.md` §T): eager `state_dict`, eager
  export/eval/encode, parity test on a compiled-run checkpoint.

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
