# Codex soundness review of the executable plan — 2026-09-04, evening (gpt-5.6-sol, read-only, high effort)

Brief: can the weekend runbook be executed unsupervised from the text alone; do the four earlier
review passes cohere. **Read-exclusion audit: CLEAN** — only the named files plus three permitted
extras (`m9/FINAL_LOCK.md`, `m9/registry.json`, `m9src/final9.py`); every grep file-scoped; no
`work/`, `untouched-*`, reserved or LoTTE path. Full log: the gitignored `.log` beside this file.
Verdict: **"No — not sound or executable unsupervised"**; 9 BLOCKER / 9 MAJOR / 7 MINOR. **All actioned**
in the same evening, together with Dylan's decision 15 and the delegated-authority section he asked for.

## Dispositions

| # | finding (short) | disposition |
|---|---|---|
| B1 | vLLM setup undefined (no version, command, revision, health check, throughput floor, timer) | **adopted** — runbook step 0a pins vLLM version, venv, full 40-char revisions, launch flags (thinking off), health assertion, a 700 output-tok/s floor, two bounded attempts, timer start |
| B2 | approval state machine undefined | **adopted** — decision 15 state machine: canonical form keys from `m10src/forms.FORMS`; only the generated forms are smoked; `(form, prompt-hash, smoke-commit, state)`; grammar `approved:` / `redraft:` from Dylan's GitHub account or Remote Control; at most two redrafts; late veto handled |
| B3 | fallback branch incomplete; hosted API may not honour seeds | **adopted** — fallback = **self-hosted bf16 via the same vLLM contract on a rented GPU after Dylan returns**, never a third-party API; triggers enumerated; A8 has no "fail", only quota actions, stated |
| B4 | §0a pushed too late (after data) | **adopted** — §0a pushed right after COV admission, before any harvest |
| B5 | §0b needs a cloud price the weekend cannot have | **adopted** — cloud price and build allocation move to the M10.2 recipe lock; §0b = corpus hashes, local rates, screen allocation |
| B6 | consumer-health listed as harvested and generated | **adopted** — five harvested / seven generated; quotas ≈1.25M harvested, ≈1.0M generated (7 × ≈143K) |
| B7 | "four families without LEDGER" is three | **adopted** — family IDs enumerated: consumer-health, BRIGHT, legal, finance iff LEDGER; floor counted on IDs |
| B8 | arXiv draw underspecified | **adopted** — artifact, sha256 recorded before the draw, ID universe (version stripped, sorted), `default_rng(0).choice(N, 100000, replace=False)`, first 2,000 = queries |
| B9 | A8 gates not numeric | **adopted** — near-duplicate rate defined (8-gram bottom-32 ≥16/32, first-occurrence representative); >25% → cut to representatives; <50K retained → form dropped; cosine and energy distance are diagnostics |
| M1 | contrasts are thirteen, not fourteen | **adopted** — thirteen everywhere, 0.025/13 |
| M2 | L12 rule not an inequality; winner function incomplete | **adopted** — `m_L12 ≥ max(m_L6, m_bge) − 0.0056` extends; winner = highest 20M macro if its margin over the runner-up resolves, else cheapest to serve (labelled) |
| M3 | B arms' doses not recorded | **adopted** — 3.75M / 5M / 7.5M with patterns 4Q, 3Q+1D, 2Q+2D |
| M4 | plateau tied to cap exhaustion | **adopted** — plateau = improvement < 0.003 only; top-up needs remaining capacity; weights revert after it |
| M5 | G-MLP warm-start record incomplete | **adopted** — seed 21, n_fit 60,000, λ reselected by the registry's grid rule on the M10 fit set, one shared sample for all three solves |
| M6 | F then A on the same COV is conditional reuse | **adopted as labelling** — A's verdict is conditional on F's winner and says so; confirmation seeds are stability checks. A crossed student × data design was considered and not adopted (≈6 GPU-h for a labelling problem) |
| M7 | LoTTE read #1 is decision-bearing | **adopted** — stated; read #2 is audit, not independent confirmation |
| M8 | release gate is development-informed | **adopted** — the released artifact is labelled "selected on development-informed datasets; reserved four descriptive" |
| M9 | `final9.decide()` is hard-coded to two conjuncts | **adopted** — an M10 `decide()` for four conjuncts under gatekeeping, unit-tested before the recipe lock |
| m | stale 745 / bf16 / "box not a target"; two cost tables; "F runs SECOND"; CODEMAP pitfall 3; `forms.py` quotas; "Decision 12 (open)"; history prose | **all adopted** — one cost table (generation on the box); history compressed to pointers |

## Verbatim findings

## Verdict

**No.** The runbook is not sound or executable unsupervised. The first undefined judgment occurs at **step 0a**: “fails after 3 h of setup” has no clock start, bounded attempt list, executable launch command, or machine-checkable success condition.

Before execution, replace steps 0a–1 with a deterministic generation/approval state machine, resolve the data taxonomy and COV-family count, and move §0a/§0b to points where their fields are both available and immutable.

## BLOCKER

- `m10/STATUS.md:83`; `m10/CODEMAP.md:5-14` — “fails after 3 h of setup” — No vLLM version, install/launch command, full model commit, API invocation, health criterion, minimum throughput, or definition of when “setup” starts exists; CODEMAP contains no generation driver. An agent must invent all of them. — Pin the full 40-character revision, environment lock, commands, chat-template/thinking mode, success assertions, throughput floor, allowed retries, and timer start.

- `m10/STATUS.md:85`; `instructions-m10.md:388-398` — “Dylan approves per form by commenting `approved: <forms>`” — The approval state machine does not define canonical form names, whether approval is required for all 12 smoke forms or only the six generated forms, whose GitHub identity counts, whether previous approvals survive another form’s redraft, or what happens to an approval arriving after Sunday noon. “one prompt revision (≤2)” is internally inconsistent. — Register the exact accepted grammar and approver account; track `(form, prompt-hash, smoke-commit, state)`; require approval against that hash; define partial approval, two failures, and late-arrival transitions.

- `m10/STATUS.md:83-85`; `instructions-m10.md:173,413-429` — “hosted bf16 … fallback if … the smoke / A8 gates fail” — The runbook invokes fallback only when vLLM setup fails. It has no branch after a form fails twice. A8 also has no “fail”: overlap has no action, and diversity merely cuts quota. Hosted inference may not expose the pinned served revision or per-request seed required by the contract. — Specify when AWQ switches to bf16, whether bf16 requires a fresh smoke/approval, and an approved provider/API that implements every sampling and seed field; otherwise define cloud self-hosted bf16 via the same vLLM contract.

- `m10/STATUS.md:86,93`; `m10/LEDGER.md:7-24`; `instructions-m10.md:283-305` — “push LEDGER §0a … then §0b” at step 9 — §0a is registered for M10.0-e, after step 2’s resolution read but before M10.1 data; the runbook postpones it until after harvesting, generation, A8, and the manifest. That destroys its function as a pre-data witness. — Push §0a immediately after step 2 and before step 3. Push only genuinely data-dependent §0b fields after step 8.

- `m10/LEDGER.md:22-23`; `instructions-m10.md:735-758`; `m10/STATUS.md:77,93` — §0b requires “billed $/h,” a re-derived cloud table, and box/cloud allocation while the weekend mandates zero cloud spend and says the provider is Dylan’s choice. Those fields cannot be filled, so step 9 forbids family F even after successful local generation. — Move cloud price/build allocation to the post-screen recipe lock, or pre-register a quote source and an automatic allocation algorithm. Keep §0b limited to corpus hashes, local rates, and screen allocation.

- `instructions-m10.md:336-347,362-386` — Consumer health is in the six-form **harvested** list, then declared **generated** from Wikipedia-medical seeds. The first assignment produces six harvested/six generated forms; the second produces five/seven, invalidating the stated ≈1.5M/≈1.0M quotas. — Choose one role, revise both lists and totals, and state the per-form quota after every possible `<100K harvest → generation` transition.

- `m10/STATUS.md:86`; `m10/COV_CANDIDATES.md:9-21`; `instructions-m10.md:604-629` — “four … families … without LEDGER” — MedicalQA is consumer-health, BRIGHT is one family, and both CorporateLobbying and ConsumerContractsQA are one legal family: **three**, not four. CUREv1 and `arxiv-title` are diagnostics. A component-based count could therefore pass the three-family gate with only two actual families. — Enumerate family IDs explicitly: `{consumer-health, BRIGHT, legal}` plus `{finance iff LEDGER admits}`; compute the floor from distinct IDs only.

- `instructions-m10.md:355-361,631-644`; `m10/LEDGER.md:14`; `m10/STATUS.md:86-87` — The held-out arXiv draw happens in step 2, but the Kaggle artifact/revision is not selected until step 3. Seed 0 lacks a universe, ordering, PRNG, and sampling algorithm. LEDGER additionally says “100K held-out papers,” contradicting the registered 2,000 held-out queries among 100K abstracts. — Pin artifact URL, full revision/hash and normalized ID universe before the draw; specify sort, PRNG, sampling operation, and 2,000/100,000 roles.

- `instructions-m10.md:413-429`; `m10/STATUS.md:92` — “near-duplicate rate exceeds 25%” — The denominator and clustering rule are undefined. Cutting to the “post-dedup unique count” is a no-op if exact dedup already occurred, and there is no minimum acceptable count or bf16 fallback trigger. Mean cosine and energy distance have no numeric gates. — Define near-duplicate rate and representative selection exactly, a minimum retained quota, and deterministic regenerate/fallback/drop actions. Either give the other metrics thresholds or call them diagnostics, not gates.

## MAJOR

- `m10/LEDGER.md:11-12`; `instructions-m10.md:565-585` — “fourteen contrasts” — The table names **13**: F2 + A2 + G3 + B2 + E1 + C1 + D2. The conditional F contrast is already included in F2. — Register a fourteenth contrast or change the count and multiplicity denominator consistently.

- `instructions-m10.md:568`; `m10/STATUS.md:94` — L12 extends if “within the MDE” of the better arm — Absolute “within” would eliminate an L12 probe that is more than 0.0056 **better**. “Statistically indistinguishable” also lacks a two-sided/symmetric rule; the directional `L6−bge` contrast cannot establish a bge win. — Use an explicit inequality such as `m_L12 ≥ max(m_L6,m_bge)−0.0056`, then specify the complete winner function for every sign and CI outcome.

- `m10/LEDGER.md:10`; `instructions-m10.md:441-447,570,575-576` — §0a assigns every arm a “5M screen dose,” while B explicitly uses 3.75M, 5M, and 7.5M totals to match 3.75M query presentations. — Record each B arm’s actual total examples/tokens and exact four-step patterns: 4Q, 3Q+1D, and 2Q+2D.

- `instructions-m10.md:492-502,533-545` — Plateau is “the extension condition” failing, including failure because `max_extension_cycles` is exhausted; the response then mandates another top-up counted against that exhausted cap. The schedule after a successful top-up also does not say whether doubled form weights persist. — Make plateau depend only on the `<0.003` improvement test, require remaining capacity before top-up, and define the subsequent schedule and weighting.

- `instructions-m10.md:557-569`; `m9/registry.json:476-488` — G-MLP is called “exact and deterministic,” but only λ and `n_fit` are inherited; registry seed 21, sampling universe/order, split, and whether λ is fixed at 0.0001 or reselected by the grid are omitted. — Copy the complete warm-start record into §0a and define the shared samples used by the linear ridge, PCA, and residual ridge.

- `instructions-m10.md:557-568` — F selects the student on A4 using COV, then A judges A4/A3/A2 on that already selected student using the same COV. Confirmation seeds reuse COV and the two largest observed margins are selected after observation. — This is conditional reuse, not independent confirmation, and favors data effects compatible with the A4-selected student. Pre-register a crossed student×data comparison or split COV for student selection versus data judgment; label seed reruns as stability checks.

- `instructions-m10.md:631-640,681-685,780-791` — LoTTE vetoes the selected recipe in favor of the anchor, yet the report contract calls LoTTE a surface “no decision touched.” — State that LoTTE read #1 is decision-bearing and therefore no longer untouched; do not present read #2 as independent confirmation.

- `instructions-m10.md:236-253,788-791` — The release gate is the clean-4 partition explicitly targeted by the data taxonomy, while the only untouched reserved surface is descriptive. — A hostile reviewer can accurately call “release” development-informed rather than confirmatory. Either make an untouched surface a release conjunct or label the artifact a development-selected candidate pending external confirmation.

- `instructions-m10.md:692-708`; `m9src/final9.py:259-278`; `m9/FINAL_LOCK.md:74-107` — The mandate says to reuse M9’s scoring path after changing two contrasts/Holm into four contrasts/fixed sequence. The actual path is hard-coded to `C1`, `C2`, and the deleted “C1 fail/C2 pass → aim” outcome. — Require and test an M10-specific decision implementation before the recipe lock; do not reuse `decide()` unchanged.

## MINOR

- `m10/EXPLORED.md:17-19`; `m10/RESULTS.md:35-38`; `instructions-m10.md:732-747`; `CLAUDE.md:75` — These still say 745 ex/s, bf16/cloud generation, or that the box is not an execution target, contradicting decision 14 and the 683/1,517 artifacts. — Replace or explicitly mark historical text; update cloud costs for local AWQ.

- `instructions-m10.md:161,739-747` — Hybrid cost is `$85–250 / 56–101 h` in the decision table but `$110–280 / 57–102 h` in §Compute. — Keep one current cost table.

- `instructions-m10.md:455-458` — “F runs SECOND, right after A” contradicts the immediately binding `F → A → …` order. — Replace with “F runs first, immediately before A.”

- `m10/CODEMAP.md:20` — “A nonlinear head is not” servable contradicts its own line 11 and the parity artifact for per-token nonlinear heads. — Qualify it as post-pooling only.

- `m10src/forms.py:1-6`; `instructions-m10.md:336-342` — “Quotas (250K per form)” is stale: generated forms are specified at ≈165K and harvested forms at ≈250K. — Remove quotas from the module docstring or source them from the manifest.

- `instructions-m10.md:646` — “Decision 12 (open)” contradicts its adopted state at lines 154-155 and 171-173. — Mark adopted.

- `m10/STATUS.md:8-54`; `instructions-m10.md:64-150` — Review history and disposition prose duplicate the operative contract and already contain stale execution claims. — Move history to the research records and keep only consolidated current rules plus pointers.

## Files opened

Required: `m10/STATUS.md`, `instructions-m10.md`, `m10/LEDGER.md`, `m10/EXPLORED.md`, `m10/COV_CANDIDATES.md`, `m10/CODEMAP.md`, `m10/RESULTS.md`, both requested review records’ disposition tables, the four requested result JSONs, `m10src/forms.py`, `m10src/head_mlp_parity.py`, `scripts/clean4_bars.py`, the requested portions of `m9src/longrun.py`, `m9src/nano.py`, `m9/M92_LOCK.md`, and the requested `CLAUDE.md` sections.

Three additional files: `m9/FINAL_LOCK.md`, `m9/registry.json`, `m9src/final9.py`. No excluded path was opened.
