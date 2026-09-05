# M10 ledger — protocol, rulings, and the numbers a rule reads

Skeleton committed 2026-09-01 (Codex pass 5). Every section is filled by the GPU session, pushed before the
step it governs, and never edited after that step's output exists. Numbers live in the JSON the
row points at; this file records the decision, the number a rule reads, and the pointer.

## §0a Screen lock — design, fill at M10.0-e, before any data exists (Opus B2 split §0 in two)

- COV resolution number (mandate §Surfaces; power disclosure only — A4's sizing struck by the Codex pass): the measured distance for e5-small-v2 vs gte-small at 0.025/13, beside the fixed MDE 0.0056; no direction, no verdict, no effect on α.
- Arms (fifteen) in the order **F → A → G → B → E → C → D** (A6 as amended by the Opus pass): F-bge-small (= anchor, 20M, read at 5/10/20M) F-MiniLM-L6 (20M) F-MiniLM-L12 (5M elimination probe; extended to 20M iff within the MDE of the better 5M reading) · A1 A2 A3-harvested A4-full (= the winner's 5M checkpoint) · G-384 G-1536 G-MLP (per-token residual `W_lin·x + W₂·GELU(W₁·x+b₁)`, W₁ 1152→192, B3) · B 100/0 (3.75M, pattern 4Q) B 50/50 (7.5M, 2Q+2D) (B 75/25 = anchor, 5M, 3Q+1D) · E-bs128 · C-M9init (skipped and reported skipped if F does not select bge-small) · D-NORM D-COV: per arm the mix, init, objective, batch, student, feature layers, dose in examples and tokens (5M screen dose), seed. Corpus hashes are §0b.
- **The thirteen contrasts, by name:** F: L6−bge-small@20M · L12−winner@20M (if extended) · A: A3−A2 · A4−A3 (three-outcome rule; A4−A2 and A2−A1 descriptive) · G: 1152−384 · 1536−1152 · MLP−1152 · B: 100/0−75/25 · 50/50−75/25 · E: bs128−bs32 · C: M9init−fresh · D: NORM−anchor · COV−anchor. Thirteen, counting F's conditional second whether or not it runs (Codex 2026-09-04 corrected a count of fourteen).
- Bound 0.025/13, **B = 200,000, `inverted_cdf`**, **MDE 0.0056 fixed**; rank-stability rule; the A4−A3 drop rule. `arxiv-title` is descriptive (no action).
- Confirmation design: which decisions (**at most two**, A5), seeds, the margin and seed-range definitions; the replication seed pair on the selected recipe.
- COV macro formula (family IDs `consumer-health`, `BRIGHT`, `legal`, `finance` iff LEDGER; slice averaging; equal weights); the `arxiv-title` diagnostic (100,000 papers drawn by id-without-version with `default_rng(0)` from the sorted universe, 2,000 of them queries; every version excluded from training; in the protected index; teacher denominator reported; artifact sha256 in §0b); DEV-6-once evaluation rule.
- G-MLP warm start: one shared fit sample (n_fit 60,000, seed 21), λ reselected by the registry's locked grid on a training-only holdout, all three solves; per-token PCA via a streamed Gram matrix.
- Outcome → action map for every family; the synthesized selected-recipe arm; LoTTE read #1 manifest and veto rule.
- Release rule under four conjuncts (decision 11) and decision 12's ruling, copied verbatim; the gatekeeping order that follows from decision 11.

## §0b Screen lock — data-dependent constants, fill at the close of M10.1, before any arm

- A2, A3 and A4 post-screen unique-text counts (identical) and corpus hashes; the arXiv artifact version and sha256; the generator revision actually served and vLLM version.
- Local measured rates: stella docs/s; examples/s at batch 32 on the 75/25 window, the 50/50 window and each F student, on real tokenized data with `num_alloc_retries` logged; generation output tok/s.
- The screen's box allocation (arms, doses, order, expected hours).
- **Cloud price, build allocation and `max_extension_cycles` are NOT §0b** (the weekend spends nothing and the provider is Dylan's choice): they are fixed at the M10.2 recipe lock (Codex B5).
- Nothing in §0b is chosen after seeing an arm.

## §1 Data manifest — fill at M10.1

- **Prompt revisions (2026-09-04, weekend window) — seven across four forms; full record and the judged samples in `m10/SMOKE.md`.** The gate's rubric is `m10src/forms.RUBRIC`, **frozen at commit 7fff677**; `forms.FORMS` is the revisable generator prompt. One dict served both until this was caught: a revision then moved the bar the gate measures, and `argument` read 8% against its own revised prompt but 88% against the registered description.

| form | rev | hash | trigger | change |
|---|---|---|---|---|
| `howto` | 1 | `6fdbab62`→`c7dc27b0` | contract 45% | title and body were emitted as two list items (2n strings); output shape restated |
| `howto` | 2 | `c7dc27b0`→`a7dae4a2` | on-form 40% | body must be exactly one or two sentences; newline is the separator |
| `argument` | 1 | `a1a9ed46`→`84f30dfb` | on-form 22% | 120-word floor restated as a hard requirement |
| `argument` | 2 | `84f30dfb`→`8bb9258b` | **word-count proxy, NOT a gate result** | 8+ sentences; prompt asks 160–210, inside the registered 120–220 |
| `conversational` | 1 | `d38b66fe`→`d16f3212` | on-form 22% | 30-word floor restated |
| `conversational` | 2 | `d16f3212`→`be4fa0ff` | **a VOID verdict (moving rubric), not a valid trigger** | four-sentence structure, the want only in sentence 4 |
| `health` | 1 | `933894ec`→`8c30d65e` | on-form 62% | 8-word floor and patient framing restated |

  Final gates, all against the frozen rubric: yesno 100 · conversational 96 · argument 88 · finance 86 · comparison 84 · health 84 · **howto 80.0 (the threshold exactly, zero margin)**; contract 100% everywhere. **Three procedural defects disclosed in `m10/SMOKE.md` and unresolved on purpose** — resolving them after observing the numbers is Tier 3: (i) `argument` r1→r2 skipped the gate; (ii) under §Data's "two failures" wording `howto` should not have reached r2, and §Data gives three different terminal rules (drop / ≤2 revisions / bf16 re-smoke); (iii) the round-robin judge sample takes position-1 items first and is lenient — `argument` is 88% judged but **67% over its full output**, every other form within a few points. `argument`'s r1 sample was overwritten and is lost. **Auto-approval runs only for `finance`, `comparison`, `yesno`, `health`; `howto`, `argument`, `conversational` are HELD for Dylan.**
- Generator (repo and revision- Generator (repo and revision — `Qwen/Qwen3-8B-AWQ` `4da05a8e…` under decision 14; hosted provider and served revision if the bf16 fallback fired), sampling parameters, seed rule, retry/dedup policy; per-form smoke results (contract %, on-form %), approver, prompt revisions (≤2 per form, each recorded here with the diff).
- Seed sources and revisions; seed pre-filter removals; per-form quotas realized, per source (harvested vs generated, amendment A2).
- **§Harvest:** per extraction rule, the rule text, source corpus, yield, and the form it feeds; the span-exclusion form of the seed-passage screen.
- **A8 quality gates:** per-form near-duplicate rate and mean pairwise cosine with any quota cut taken; the stella-space distribution-overlap table against the MS MARCO dev sample (disclosed diagnostic, no action).
- PAQ release files and hashes; build sample (1.0M) and A2 sample; attribution.
- Decontamination removals per screen, per form, per COV component; FORMS-12 hold-out seed ids.
- Teacher-target cache keys. ~~bank, mining method, recall@64 audit~~ — struck with the ranking-aware class (amendment A1).
- `results/m10_data_manifest.json` sha256.

## §2 COV admission records — fill at M10.0-d, one row per component

| component | family | repo · revision | licence at primary source (URL) | corpus / queries / qrels / metric | corpus-level contamination check | fingerprint screen vs six + reserved | verdict |
|---|---|---|---|---|---|---|---|

## §3 Owner rulings and decisions

Copied from `instructions-m10.md` §Owner decisions as each is taken, with date and wording.

- 2026-09-01 — Student cap: "109M is not an option. This isn't low compute anymore. 33M was already in the upper bound of what I think is acceptable." 35M hard.
- 2026-09-01 — FineWeb: out of M10 in every role (delegated ruling; `m10/EXPLORED.md`).
- 2026-09-01 — Compute (decision 2 reframed): "M10 won't be done on a 3080. M10 will be done on a GPU budget, if allowed, or not at all." Box withdrawn as an execution target; budget request expected $400–715 with generation on the GPU or $465–895 hosted, ceiling $1,000 (PLANNING §6).
- 2026-09-04 — **Budget VALIDATED** by Dylan, together with "do a full review yourself … is this the most efficient?", "I'm not sure why we need to generate synthetic data?", "isn't synthetic data lower quality? I wanna make sure we have the best chances on our side", "don't over-engineer", "keeping the same teacher (Stella) is the goal", and "make your changes to the plan, then have Fable do an adversarial review". Re-priced on measured rates to ≈$110–280 hybrid; ceiling $1,000 unchanged. Amendments **A1–A8** adopted (`instructions-m10.md` §Amendment 2026-09-04, decision 10); cuts and their reopening conditions in `m10/EXPLORED.md`; evidence PLANNING §11–12.
- 2026-09-04 — Three days of box compute offered before the cloud instance ("I will be leaving for 3 days tonight"). Used for the no-approval-needed stages only; generation cannot start in that window (Dylan is the smoke approver and Qwen3-8B bf16 does not fit 10 GB) — **unless decision 14 (official AWQ on the box, remote smoke approval) is adopted.**
- 2026-09-04, evening — **Decision 11:** "Let's make sure we win enough so this isn't a question" — default stands, release needs C1b. **Decision 12:** "yes" — CUREv1 as a reported diagnostic. **A7:** "Yes, you will have 3 days of uninterrupted access to the 3080 during the weekend. The more we can prove before spending the better." **Asked:** whether synthetic generation is still needed and whether it could run locally in the window → decision 14, proposed, then **adopted the same evening ("Go on 14")**: `Qwen/Qwen3-8B-AWQ` on the box via vLLM, smoke approved remotely via a GitHub issue; self-hosted bf16 via the same vLLM contract the fallback. **Decision 15 ("Yes to decision 15"):** conditional pre-approval — contract rate by the session, on-form rate by an independent Fable subagent, six-hour veto window, late veto drops the form. **Delegated authority** granted for unsupervised windows (mandate §Delegated authority): Tier 1 alone, Tier 2 after a Fable consultation logged below, Tier 3 never. Generator pinned: `Qwen/Qwen3-8B-AWQ` `4da05a8edb55c6046cce958586c33b61da07bb79`; fallback base `b968826d9c46dd6066d109eabc6255188de91218`.

### Delegated decisions (unsupervised windows) — append-only

| when (UTC) | commit | decision | options considered | Fable subagent's verbatim recommendation | taken | why |
|---|---|---|---|---|---|---|
| 2026-09-05 00:2x | 028d234 | **T2-1 `max_tokens` reading.** §Data registers "60, or 400 for the argument and conversational forms". Read as **PER QUERY × n** (howto n=5 → 300, argument n=5 → 2000) | (a) per query × n; (b) per request, smoke at n=1 for long forms; (c) per request as written, let the forms fail | *"Reading (a) — 60/400 is a per-query budget, multiplied by n. Tier 2, not Tier 3. Log it in LEDGER §3 before the first generation call, and capture `finish_reason` so a truncation failure is never mistaken for a prompt failure."* Reasoning: the per-form length spec and the token cap contradict each other under (b)/(c); step 0a's own registered health assertion is unsatisfiable under (c); (c) burns a prompt revision on a known budget failure and converts it into a dropped form — "the Tier-3 outcome (a quota change) by the back door". Wrong if Dylan meant 60 as a hard length cap overriding the form lengths | (a) | Two registered constants contradict; (a) is the only reading under which both hold. `finish_reason` now recorded, so truncation is separable from wording — it is how the `howto` failure was diagnosed. **Dylan may strike this**; if 60 was meant as a hard cap, say so and the three long forms are re-cut |
| 2026-09-05 00:2x | 028d234 | **T2-2 smoke seeds drawn before COV admission.** Runbook step 1 precedes step 2; the mandate says COV joins the protected index "before any seed is drawn". Resolved: the smoke is a **gate artifact** — its seeds and its 1,400 queries never enter any build corpus; the build's seed draw at step 8 is a fresh draw against the post-COV index | run the smoke pre-COV as a gate artifact; or block the smoke until after COV admission (costs the weekend) | *"Your resolution is sound"* with three hardenings: smoke output never enters a build corpus; fix `seeds.cached()` (its key ignored store, seed, min_score, pool size and the screen); add the six's document fingerprints only if that index already exists — "do not build a new document index tonight". Reasoning: the smoke produces a contract rate, an on-form verdict and a prompt hash, none of which is data, and prompt wording contains no passage content. Wrong if a redraft could add TOPIC words drawn from what the judge saw | resolution + all three hardenings | The cache was the real leak and is closed (key now covers the draw and a `SCREEN_VERSION`). **Registered now: a prompt redraft may change form WORDING only, never topic words.** The `howto` revision below complies — it restates output shape only |
| 2026-09-05 00:2x | 028d234 | **T2-3 is the smoke rigged?** Topical seeds are the top keyword scorers of 400K candidates; the build cannot be that selective | (i) keep, disclose; (ii) draw as the build will; (iii) both, gate on the build-representative one | *"(i) with two additions, not (ii) or (iii)"* — the gate asks whether a PROMPT is well worded, and "a health prompt on a sports biography fails for a reason unrelated to wording"; (ii) is undefined tonight because the build's seed rule is not locked; a harder unregistered gate that drops a form "would be a quota change on a measurement the gate was never registered to make". Additions: register that the build's topical draw uses the same `ROUTE` patterns and `min_score ≥ 4`; add a report-only build on-form diagnostic. Also: the judge sample must spread across all 40 seeds — `queries[:50]` was "the ~10 strongest seeds" | (i) + both additions + the judge-sample fix | Disclosed in `m10/SMOKE.md`. The judge sample is now round-robin over all 40 seeds |

### Open questions for Dylan — raised by the weekend window, NOT resolved here

| # | question | why it is not mine to answer | cost of waiting |
|---|---|---|---|
| W1 | **Is the terminal counter two GATE FAILURES or two REVISIONS, and does a second failure trigger a form-only bf16 re-smoke (§Data:428) or a drop (§Data:436, `m10/STATUS.md`:58)?** Three rules are registered for one situation | The numbers it governs are already observed (`howto` 80.0%, `argument` 88%), so choosing a reading now is "changing a protocol after a number it affects is observed" — Tier 3 | none: nothing generates before COV admission and the build seed draw |
| W2 | Do `howto` (80.0%, at the threshold), `argument` (88% judged / 67% full output) and `conversational` (96%, but its r2 was triggered by a void verdict) stand as passes? | Their gate results are sound; their *procedure* was not, and I found that only after seeing them | none, as W1 |
| W3 | At `min_score ≥ 4` the full `hotpotqa-corpus` projects **health 8.8K** and **finance 21.0K** topical seeds against the ~28.6K a 143K-query form needs at 5 queries/seed. Widen the seed pool, lower the floor, or raise queries-per-seed (which the A8 near-duplicate gate then bites)? | Touches the registered data recipe | decide before step 8 |

## §4 Dev-reuse log

| date | surface | raw score reads | artifact |
|---|---|---|---|
| 2026-09-01 | cqadup-programmers, cqadup-physics (Mac diagnostics) | 43 + 43 | `results/m10_rank_probe_mac.json`, `results/m10_head_width_probe_mac.json` |
| 2026-09-04 | frozen comparator rows of `results/perquery.json` (bge-small, leaf-ir-asym, lr-dense-pertask, opensearch, bm25) on all-6 and clean-4 | comparator-only, no nano existed | amendment A3's clean-4 bars 0.5046 / 0.5233; not a dev-surface read |

## §5 Amendments and withdrawn claims (never compressed away)

**2026-09-04 amendments A1–A8 and B1–B6, and the Codex and Opus passes on them:** one home only —
`instructions-m10.md` §Amendment 2026-09-04 / §Amendment 2026-09-04b (Opus: the copies here had already
drifted). Dispositions: `research/m10-fable-plan-2026-09-04.md`, `research/m10-codex-feasibility-2026-09-04.md`,
`research/m10-opus-review-2026-09-04.md`. **Decision 12 (CUREv1 as a validation-only diagnostic) is open.**

**Withdrawn in the same review, kept so it is not re-proposed:** dropping family F to anchor on
MiniLM-L6. It would have killed family C as well (M9's candidate is a bge-small student) and arXiv
2306.11550's depth curve disagrees with LEAF's 6-layer success. F and C stay; A6 fixes the ordering
problem instead.

**Corrected in the same review:** `instructions-m16.md`'s "if only one thing here ever runs, run
B3" — B3 (cosine-space distillation) is a no-op on a normalized output, closed by algebra. The
pyNIFE retention gap must be attributed to B4/B5.

