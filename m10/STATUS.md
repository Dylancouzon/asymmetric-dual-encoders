# M10 status — 2026-09-05. Steps 0a, 0b, 1, 2, 3, 4 and the PAQ artifact are DONE; nothing registered has trained

## IN FLIGHT right now (check these first on a cold start)

| job | how to check | ETA |
|---|---|---|
| **M10.0-e calibration**, arms P0/P1/P2 at 5M each | `work/m10calib_run.log`, `grep "ex/s"`; artifacts `work/m10calib/P*.json` and `P*_cov.json` | P0 ~70% through; **slowed to ~875 ex/s** by CPU contention with the draw (GPU 65% → 40%) |
| ↳ **when all three land** | run **`.venv/bin/python m10src/calib_report.py`** → `results/m10_calib_report.json`. It refuses any P arm whose `total_steps != 156250` (the smoke shares the path) or that stopped early. Then record it in LEDGER §M10.0-e and RESULTS, and log **COV read #2** in §4 | ~2 min |
| **PAQ draw** — 4.037M for A2, 1.0M nested for the build | `work/m10paq/draw.log`; artifacts `work/m10paq/paq_draw.json`, `paq_a2.jsonl`, `paq_build.jsonl` | read 4.24M lines in 25s; the protected screen over 4.24M queries is the cost, ~25 min |
| **arm-shape smoke** re-run, `nice 19` | `work/m10arm_smoke_full.log` | ~15 min; this run also exercises the two warm starts |

**Two stale-artifact hazards.** (1) The 90-step smoke of `calib.run_arm` writes to the same
`work/m10calib/P0.json` path the real run uses — check `total_steps == 156250` before believing any
P arm's JSON. The stale record is moved aside to `_SMOKE_STALE_P0.json.bak`, but `calib_report.py`
still enforces the check. (2) `calib.py:run_arm` saves a **fresh** `AdamW` beside the model, so
`work/m10calib/P*.pt` carry **no usable optimizer state** — never warm-start an arm from them.

**`lr_at` off-by-one — the LAST step of every arm trains at PEAK LR, and it must NOT be fixed
while the calibration runs.** `per = total_steps // cycles` leaves a remainder, so for 156,250
steps the final step 156,249 wraps to `within = 0` and the LR jumps 1.0e-05 → **1.0e-04**.
Cycle-end COV evals are taken at step 156,248 and are **clean**, so every screen verdict is
unaffected; but the final checkpoint and any post-loop encode carry one peak-LR update, which is
what the BUILD would export. The three P arms all carry it identically, so the calibration's
paired widths stand. **Fix after M10.0-e completes and before any registered arm** — changing it
now leaves P1/P2 incomparable to P0. LEDGER §`lr_at` off-by-one.

**CPU contention is real on this box (16 cores).** Three concurrent jobs took the calibration's GPU
from 65% to 35% and its rate from 945 to 874 ex/s; and `E-bs128` at 512 tokens **on CPU** took the
box from 10 GB free to 2 GB, because `output_hidden_states=True` keeps every layer's states for
128x512 positions. `nice 19` on the least urgent job fixes the first, `--max-len 128` the second.
Check `free -g` and `nvidia-smi` before adding a fourth job.

## Where the milestone actually stands

| step | state | outcome |
|---|---|---|
| **0a / 0b / 1 / 2** | DONE | vLLM + generator pass; rates re-measured; all seven generated forms approved; four COV families admitted (13,416 queries) |
| **2b** COV resolution | **DONE — W5 answered** | **0.008619**; MDE 0.0056 sits below it. Variance: BRIGHT 50.0%, legal 32.5%, consumer-health 14.7%, finance 2.7% |
| **teacher ceiling** | **DONE** | stella's own COV macro **0.5567** — legal 0.8845, consumer-health 0.7507, finance 0.3726, **BRIGHT 0.2191**. `results/m10_cov_teacher_ceiling.json` |
| **3** §0a design lock | DONE, amended S1–S10 | `m10/screen_registry.json` + `screen_lock.py`, 18 tests. The anchor is its own trained arm (16 arms) |
| **4** harvest | **DONE, including the draw. The A3 corpus is 1,250,000 rows** (189 MB, 1,785 s, peak 9.06 GB) | title 417,000 · keyword 417,000 · claim 416,000, **no form short**. Rubric range removed 1,943,222 `claim` rows and 0 title/keyword; exact-dup removed 3,200,570 `keyword` (35%). Screens dropped 103,197 of 1,875,000 (5.5%), of which **99,544 came from `dev:hotpotqa` alone** — a Wikipedia lead sentence IS a hotpotqa document, so `claim` survived 83.8% vs title 99.7%. **Realized mix: `title` 73.6% arXiv, but `claim` only 28.4% — see the §1 DISCLOSURE before reading A3−A2 as evidence about scientific claim text.** LEDGER §1 |
| **4b** the `ask` rule | **contributes ZERO rows, and that IS the registration** | The mandate registers **five** harvested forms at ~250K (`instructions-m10.md`:366); the quota table registers three. So `factoid` (5,605) and `product` (40,977) have no quota row — **and both fall under the mandate's own "under 100K reverts to generation at ≈143K"**, which is +286K over the 1.0M generation cap = Tier 3. **Default excluded; W7 is Dylan's.** LEDGER §Harvest amendment 2026-09-05 |
| **5** PAQ | **artifact DOWNLOADED and VERIFIED; the draw still needs a margin** | `PAQ.tar.gz` sha256 `177eefb2…`, 1,447,064,073 bytes, **64,875,601 pairs**, from `dl.fbaipublicfiles.com` — never an HF mirror. The tarball ships `PAQ/LICENSE` = the CC BY-SA 3.0 legal code, which is the primary-source grant the mandate asks for. Only the `question` field is read. `m10src/paq.py` |
| **6** trainer port | **DONE. The arm-shape smoke is COMPLETE — 12/12 pass, all under the 35M cap** (`results/m10_arm_smoke.json`, CPU, `max_len 128`) | `nano10` · `data10` · `trainer10` · `qfilter` · **`arm_smoke`**. **`G-384` could not be CONSTRUCTED** — `KeyError: 1`, `LAYERS` had no 1-layer key, so a registered arm of the locked design was unbuildable; fixed as "last layer only" per `instructions-m10.md`:616. **Two registered warm starts are UNIMPLEMENTED**: G-MLP's three-solve recipe (`warm_start_linear` raises on an MLP head) and C-M9init's zero-padded 384-d head. Both arms would train from a fresh head, biasing G3 and C1 *against* the non-default — **implement before families G and C** |
| **seeds** | **RESOLVED FOR ALL SEVEN generated forms — generation's only blocker is the GPU** | `wikipedia-body` scanned (6,407,814 articles → 22,243,221 body chunks → 238,823 kept) and drawn after T2-8 rung 1: **health 33,000 · finance 33,000, short 0** (`work/m10gen/wikibody_draw.json`). `howto` is topical from the incumbent at **37,927** ≥ 33,000 — the marginal form on BOTH supply (≈12% margin after the FORMS-12 500 and the 400 gate seeds) and fidelity (80.0%, the threshold exactly, held for Dylan). `argument`, `comparison`, `yesno`, `conversational` route as `"general"` in `seeds.ROUTE`, so they draw from the 2,615,015 length-eligible passages — ample. **Rungs 2 and 3 do NOT run** (§W6 RESOLVED demotes the ladder to a diagnostic); there is **no post-generation admission test**, only A8's manifest gates and the FORMS-12 hold-out |
| **W6 seed store** | **RESOLVED by Dylan** | Use `wikipedia-body`, report the measured numbers, **invent no standard**. There is no admission bar. Revisitable with reviewer approval |
| **word-range filter** | **DONE — the largest on-form win found** | Enforcing each form's own frozen-rubric range: health **0.780 → 0.857**, finance **0.790 → 0.806**. Out-of-range queries score ~0 on-form. `results/m10_qfilter_effect.json` |
| **5, 8–11** | not started | PAQ · generation (~10 box-h) · §0b · family F (~17 GPU-h) · family A |

## THE NEXT FIVE THINGS, in order

1. **`.venv/bin/python m10src/paq.py --pilot 200000`** for the dedup and protected-screen loss
   rates (dedup is ~0.002% — `PAQ.filtered` is already deduped), then set the margin and run the
   real draw: 4.037M for A2, 1.0M **nested inside it** for the build.
2. **Generation**, ~10 box-hours on the GPU once the calibration arms free it. Apply `qfilter`
   to the output — it is the single largest quality lever measured, and it enforces a range the
   frozen rubric already specifies.
3. **Implement the two missing warm starts** before families G and C — G-MLP's three-solve
   recipe (`instructions-m10.md`:616 specifies it exactly) and C-M9init's zero-padded 384-d head
   (`instructions-m10.md`:510). Until then both arms would start from a fresh head, which biases
   G3 and C1 *against* adopting the non-default.
4. **§0b**, then **family F**.

**What the calibration result licenses, and what it does not.** It produces two numbers:
`lr_pair.distance_raw` — a same-init contrast's paired width, which **B, D, G and C** are read
against — and `seed_effect_point_estimate` = |macro(P0) − macro(P1)|, the seed effect, which the
bootstrap **cannot see** because it resamples queries and not seeds. A contrast that "resolves" a
difference smaller than that seed effect has resolved noise. **F and E are read against the
0.008619 unrelated-models number and no LR pair speaks for them** — two students sharing only the
teacher target ARE the unrelated case. It closes W5 for no family on its own, changes **no**
constant, and any MDE decision is Dylan's, before family F starts, never after.

## Three things a resuming session must NOT redo

- **Do not move the seed-precision bar.** W6 is resolved: there is no bar, the numbers are
  reported. Two reviewers already withdrew one replacement rule; the post-mortem is LEDGER §W6.
- **Do not re-weight or drop BRIGHT.** Considered and rejected 2026-09-05 with the arithmetic in
  LEDGER §5: BRIGHT is 46% of the signal for 50% of the variance, all four family deltas are
  same-signed, and the uninformative family is **legal** (9.8% / 32.5%). Dropping BRIGHT moves the
  distance 0.0086 → 0.0081 and loses 28% of the signal.
- **Do not treat the on-form diagnostic as an admission instrument.** It admits nothing (T2-7).

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
