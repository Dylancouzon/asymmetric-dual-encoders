# M10 ledger — protocol, rulings, and the numbers a rule reads

Skeleton committed 2026-09-01 (Codex pass 5). Every section is filled by the GPU session, pushed before the
step it governs, and never edited after that step's output exists. Numbers live in the JSON the
row points at; this file records the decision, the number a rule reads, and the pointer.

## §0a Screen lock — design, LOCKED 2026-09-05, before any arm and before any build seed draw

**The lock is `m10/screen_registry.json`, not this section.** Prose is not authoritative; the
registry is (M9's rule, `m9src/final_stats.py`). `m10src/screen_lock.py` validates it and is the
only reader a rule may use; `m10src/test_screen_lock.py` shows it refusing a fourteenth contrast,
a contrast naming an arm that does not exist, an arm no contrast reads, a family with no
outcome→action entry, a quantile that is not α/13, numpy's default quantile method, a families
list that disagrees with `m10src/cov_macro`, and a registry resolution number that has drifted
from its artifact.

| what | value | where |
|---|---|---|
| arms | **16 trained**, 17 entries — A4 is not trained separately, it IS the ANCHOR arm | `arms` |
| **anchor** | **its own trained 5M arm** (S1 amendment). It had been "F's winner at its 5M checkpoint" — ~75% through cycle 1 of a 20M schedule, un-annealed, against genuine three-cycle 5M arms: a handicap on the `b` side of **ten** contrasts, in the direction that adopts a non-default, and it left the anchor with no second cycle end (sign-stability) and no DEV-6 read. **+2 GPU-hours** | `anchor`, `arms.ANCHOR` |
| data cut | A2, A3 and A4 cut to the identical post-screen unique-text count, min-of-three, seed-0 downsample; **the anchor trains on the CUT A4**, so it and A3 differ in which text, never how much | `data_cut` |
| rules | every decision path numerically registered: A's two corrected bars, F's post-hoc orientation and its serve-cost ordering, E's cost rule, arm failure, confirmation eligibility and failure, the C-skip denominator, the D tie, and A2−A1 as **descriptive with no build action** | `rules` |
| parity | all six family-F heads PASS (bge-small · MiniLM-L6 · MiniLM-L12, 3 and 4 layers), min-cos ≥ 0.99999988, zero custom ops, all under the 35M cap | `parity_preconditions`, `results/m10_student_parity_box.json` |
| order | F → A → G → B → E → C → D; every family after F runs on F's winner and every later verdict is labelled conditional on it | `order` |
| contrasts | **13 decisive** (F2 counted whether or not L12 is extended) + 3 descriptive | `contrasts`, `descriptive_contrasts` |
| statistics | family-weighted COV macro over the four family IDs, paired stratified bootstrap over queries within unit, **B = 200,000, seed 0, `inverted_cdf`, one-sided 0.025/13**, **MDE 0.0056 fixed** | `statistics`, implemented in `m10src/cov_macro.py` |
| resolve | point ≥ MDE **and** the lower bound > 0 **and** the sign stable across the last two cycle-end checkpoints | `statistics.resolve_rule` |
| power | measured distance **0.008619** — the MDE is BELOW it, so a contrast landing at the MDE cannot resolve. It sizes nothing; α does not move (A4's sizing struck) | `results/m10_cov_resolution.json`, §3 W5 |
| outcome → action | one entry per family, including A3−A2's three outcomes and its **registered STOP**, and the default that an unresolved contrast reverts to and is REPORTED as unresolved | `outcome_to_action` |
| confirmation | at most **2** decisions, largest margin first, seeds 1 and 2; stands iff the margin exceeds the largest seed range in either arm | `confirmation` |
| evaluation | COV every cycle end; **DEV-6 once**, at the final checkpoint, never selection-bearing; FORMS-12, `arxiv-title` and CUREv1 descriptive; LoTTE read #1 after the recipe lock | `evaluation` |
| warm start | G-MLP: n_fit 60,000, seed 21, λ reselected on the registry's locked grid on a training-only holdout; per-token PCA by a streamed Gram matrix, direction signs fixed | `warm_start` |

**Amended 2026-09-05 on a Fable adversarial pass (S1–S10), before any arm ran and with nothing
trained.** S1 above is the one that mattered. Also taken: the measured 2.2× bs128/bs32 throughput
ratio moved OUT of §0a (it is data, so it is §0b); `bootstrap.chunk = 5000` added, since the chunk
size is part of the draw-plan definition; C-M9init's head init registered with its confound
disclosed; the validator hardened against twelve mutations that passed the first version, each
with a test. **The validator runs only under `.venv/bin/python`** — the system python has no
numpy, and that ImportError is not a lock failure.

Corpus counts, hashes, measured rates and the box allocation are **§0b**, and cloud price, build
allocation and `max_extension_cycles` are neither — they are fixed at the M10.2 recipe lock.

## §0b Screen lock — data-dependent constants, fill at the close of M10.1, before any arm

- A2, A3 and A4 post-screen unique-text counts (identical) and corpus hashes — **STILL OPEN: A4
  needs the generated half, so `data_cut` cannot be computed until generation runs.** Determinate
  now: **A3's harvested component is 1,250,000 rows** (§1), **PAQ's A2 sample 4,037,000 with the
  build's 1,000,000 nested inside it** (`work/m10paq/paq_draw.json`, hashes there), and the M9 pool
  is 463,314. **Generator and server, pinned:** `Qwen/Qwen3-8B-AWQ` revision
  `4da05a8edb55c6046cce958586c33b61da07bb79`, served by **vLLM 0.28.0** in `.venv-gen`
  (`work/m10gen/serve.sh`, port 8001; `VLLM_WSL2_ENABLE_PIN_MEMORY=1` and
  `VLLM_USE_FLASHINFER_SAMPLER=0` both required on this box, never `--enforce-eager`). The bf16
  fallback did NOT fire.
- **arXiv artifact, DRAWN 2026-09-05** (`work/m10arxiv/arxiv_draw.json`): the registered Kaggle
  `Cornell-University/arxiv` · `arxiv-metadata-oai-snapshot.json`, zip sha256
  `47cec120969d4238d67be52b960b7b851c993dc039a64f582cec97ec114443d9`, 1,820,571,144 bytes.
  **3,148,882 records → 3,148,792 unique version-stripped ids**, sorted
  lexicographically; `default_rng(0).choice(N, 100,000, replace=False)`; the first 2,000 drawn are
  the queries (title → own abstract), 0 empty titles, 0 empty abstracts. All 100,000 base ids are
  excluded from every training role (`arxiv_excluded_base_ids.json`); the 2,000 queries and their
  abstracts are **in the protected index** (`protected10.VERSION` … `+arxiv-title`), which is what
  §Surfaces requires before any extraction. `arxiv-title` is DESCRIPTIVE and triggers no action.
  Credential: Dylan's Kaggle token, 2026-09-05, stored at `~/.kaggle/access_token` **outside the
  repo**; nothing in the tree contains it.
- **Local measured rates, MEASURED 2026-09-04 on real tokenized corpora** — RTX 3080, torch
  2.8.0+cu126, batch 32, 200 steps, **`alloc_retries = 0` on every row**
  (`results/m10_rate_bench_real_box.json`). Not the random-token microbenchmark, which bounds the
  hardware and not the pipeline (`results/m10_rate_bench_box.json`).

| corpus (role, mean tokens) | M9's two-chunk path | length buckets | fixed buckets + `torch.compile` |
|---|---|---|---|
| `nqopen` (query, 11.7) | 639.7 | 732.4 | **959.5** |
| `triviaqa` (query, 19.9) | 643.3 | 729.2 | **913.9** |
| `pseudoq` (doc span, 41.4) | 527.5 | 646.4 | **949.9** |
| `documents` (doc, 94.6) | 249.1 | 465.0 | **792.2** |

  **Blended 75/25 with compile ≈ 910 examples/s**, against the 683 the plan was re-priced on and
  the 560 it originally imported. **Independently confirmed in flight:** the M10.0-e calibration's
  first arm logged **962 ex/s** on the real M9 query pool at batch 32, matching `nqopen`'s 959.5 —
  so the compiled fixed-bucket path is what the trainer actually runs. Under three-way CPU
  contention on this 16-core box the same arm fell to **842–852 ex/s** (GPU utilisation 65% → 33%),
  which is the number a schedule should use if anything else is running.
- **The screen's box allocation** — arms and doses from `m10/screen_registry.json`, order
  **F → A → G → B → E → C → D**. **This presumes the screen runs as registered; W8 and W9 are
  open and may cut it.**

| family | examples |
|---|---|
| F (bge-small 20M · MiniLM-L6 20M · L12 5M, +15M iff extended) | 45M–60M |
| A (A1 · A2 · A3 at 5M; A4 = ANCHOR) | 15M |
| ANCHOR | 5M |
| G (384 · 1536 · MLP at 5M) | 15M |
| B (3.75M + 7.5M) | 11.25M |
| E (bs128, 5M) | 5M |
| C (M9init 5M, skipped iff F does not pick bge-small) | 0M–5M |
| D (NORM · COV at 5M) | 10M |
| **total before confirmations** | **106.25M–126.25M** |

  At the measured blended 910 ex/s that is **32–39 GPU-hours**; at 683 it is 43–51. Confirmations
  plus the synthesized selected-recipe arm and the replication seed pair add up to ~60M more,
  putting the pre-build training stage at **166M–186M examples — comparable to a whole 200M build**
  (Codex 2026-09-05). `E-bs128` is cheaper than its dose suggests: batch 128 measured 1,517 ex/s.
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
| `conversational` | 2 | `d16f3212`→`be4fa0ff` | **a VOID verdict (moving rubric), not a valid trigger — and UNNECESSARY: r1 re-judged against the frozen rubric scores 100%** | four-sentence structure, the want only in sentence 4 |
| `health` | 1 | `933894ec`→`8c30d65e` | on-form 62% | 8-word floor and patient framing restated |

  Final gates, all against the frozen rubric: yesno 100 · conversational 96 · argument 88 · finance 86 · comparison 84 · health 84 · **howto 80.0 (the threshold exactly, zero margin)**; contract 100% everywhere. **Three procedural defects disclosed in `m10/SMOKE.md` and unresolved on purpose** — resolving them after observing the numbers is Tier 3: (i) `argument` r1→r2 skipped the gate; (ii) under §Data's "two failures" wording `howto` should not have reached r2, and §Data gives three different terminal rules (drop / ≤2 revisions / bf16 re-smoke); (iii) the round-robin judge sample takes position-1 items first and is lenient — `argument` is 88% judged but **67% over its full output**, every other form within a few points. `argument`'s r1 sample was overwritten and is lost. **Auto-approval runs only for `finance`, `comparison`, `yesno`, `health`; `howto`, `argument`, `conversational` are HELD for Dylan.**
- Generator (repo and revision- Generator (repo and revision — `Qwen/Qwen3-8B-AWQ` `4da05a8e…` under decision 14; hosted provider and served revision if the bf16 fallback fired), sampling parameters, seed rule, retry/dedup policy; per-form smoke results (contract %, on-form %), approver, prompt revisions (≤2 per form, each recorded here with the diff).
- Seed sources and revisions; seed pre-filter removals; per-form quotas realized, per source (harvested vs generated, amendment A2).
- **§Harvest:** per extraction rule, the rule text, source corpus, yield, and the form it feeds; the span-exclusion form of the seed-passage screen.
  **EXECUTED 2026-09-05** — `work/m10harvest/harvest_draw.json`, `harvest_drawn.jsonl`
  (**1,250,000 rows**, 189,060,152 bytes), 1,785 s wall, VmHWM **9.06 GB**.

| rule | source | form fed | raw rows | in rubric range | after dedup | drawn | survived | **taken** |
|---|---|---|---|---|---|---|---|---|
| `title` | Wikipedia, arXiv | `title` / `keyword` by the rubric's own ranges | 4,690,821 | all (routed BY the range) | — | — | — | — |
| `heading` | Wikipedia | `title` / `keyword` | 4,833,120 | all | — | — | — | — |
| `claim` | Wikipedia 6,020,221 · arXiv 2,239,838 | `claim` | 8,260,059 | 6,316,837 | 6,270,443 | 624,000 | 522,747 | **416,000** |
| `ask` | the pool corpora | `factoid` 5,605 / `product` 40,977 | 46,582 | — | — | **not drawn** | — | **0** (no quota row — §Harvest amendment item 1) |
| — | *`title` form total* | — | — | — | 3,559,910 | 625,500 | 623,882 | **417,000** |
| — | *`keyword` form total* | — | — | — | 5,924,646 | 625,500 | 625,174 | **417,000** |

  **Rubric-range removals** (amendment item 3): `claim` **1,943,222**; `title` and `keyword`
  **0** each, as expected — `route_by_length` assigns them BY that range. **Exact-dup removals:**
  `keyword` **3,200,570** (35% — short strings repeat heavily across Wikipedia), `title` 95,276,
  `claim` 46,394.

  **Screens, 1,875,000 candidates, matches REMOVED:** query-side (protected index) **4,092**;
  document-side **99,960**; total **103,197 (5.5%)**. Every document-side drop came from four
  streams — `dev:hotpotqa` **99,544** of 5,233,329 (822 s), `cov:BRIGHT` 251, `dev:nq-250k` 118,
  `six-docs` 47; cqadup ×2, MedicalQA, both LegalBench and LEDGER dropped **0**. This is why
  `claim` survived only **83.8%** against `title` 99.7% and `keyword` 99.9%: a harvested Wikipedia
  lead sentence and a `hotpotqa-corpus` document are the same text. Margin 1.5 covered it and no
  form came up short.

  **Realized source mix — REPORTED, never fixed in advance, and one row deserves attention:**

| form | arXiv | Wikipedia | arXiv share |
|---|---|---|---|
| `title` | 307,065 | 109,935 | **73.6%** |
| `claim` | 118,039 | 297,961 | **28.4%** |
| `keyword` | 8,668 | 408,332 | 2.1% |

  **DISCLOSURE.** arXiv was added because "harvested from [the pool], *paper title* would be
  Wikipedia article titles and *scientific claim* would be Wikipedia lead sentences, neither of
  which is the form scidocs and scifact test" (`instructions-m10.md`:383). The `title` form is now
  genuinely scientific (73.6% arXiv), so the **scidocs↔titles** leg of A3's hypothesis is
  well-served. **The `claim` form is 71.6% Wikipedia lead sentences**, so the **scifact↔claim**
  leg is only partly served — the uniform draw preserves the corpus's own distribution, and the
  corpus has 2.7x more Wikipedia claims than arXiv claims. The hotpotqa screen pushed the arXiv
  share *up* (it removes Wikipedia claims preferentially). Nothing here is a protocol deviation —
  §Harvest registers the mix as reported, not fixed — but **A3−A2 must not be read as evidence
  about scientific claim text**, and if that leg matters the lever is a per-source quota on
  `claim`, which is a quota change and therefore Dylan's.

  **Two measured quality disclosures on the SHIPPED corpus** (spot-checked after the draw, and
  neither is a rule deviation — both are what the registered rules produce):

  1. **`claim`: 6,827 of 416,000 rows (1.64%) are sentence-splitter artifacts**, truncated on an
     abbreviation or initial — *"A Tight Corner is a 1932 British comedy film directed by Leslie
     S."*, *"…first described by J."*, *"…a goalkeeper for club S.C."*, *"…produced by the U.S."*.
     `harvest.sentences` splits on `.` and does not protect single-letter initials or `U.S.`-style
     abbreviations. **Not re-drawn**, and the reasoning is on the record: these still satisfy the
     registered `claim` rule (declarative, ≥ 8 words, finite verb, no first person), a truncated
     sentence is still real text whose teacher embedding is a correct target by construction, and
     they are **0.55% of the 1.25M corpus**. Re-drawing would cost a re-run of both harvest passes
     plus the draw and change every number in this section for a cosmetic gain. **The fix for any
     future harvest** is an abbreviation guard in `sentences()` — protect `[A-Z].` and the
     `U.S.`/`S.C.` pattern before splitting.
  2. **`keyword`: 273,189 of 417,000 rows (65.5%) are entity-name shaped** — every token
     capitalised — and only 12.6% contain a function word. The form is registered as a "2–4-word
     keyword query" harvested from "titles as-is", and a Wikipedia title *is* a real keyword query
     ("Hurricane Wilma"), so this is the rule working. But **a third of A3 is proper nouns**, which
     is narrower than "keyword query" may suggest to a reader, and it should be stated that way in
     the report rather than left to be discovered.

  **Artifact-collision fix.** `draw()` wrote its outputs to the module-level `OUT` whatever
  `paths` said, so the scaled-down integration smoke produced a `harvest_draw.json` and
  `harvest_drawn.jsonl` at exactly the real paths — 900 rows that read like a 1.25M corpus. Same
  trap as `calib.run_arm`'s 90-step smoke writing the real `P0.json`. `draw()` now takes
  `out_dir` and records it in the report; a smoke must pass one.
- **A8 quality gates:** per-form near-duplicate rate and mean pairwise cosine with any quota cut taken; the stella-space distribution-overlap table against the MS MARCO dev sample (disclosed diagnostic, no action).
- **PAQ — DRAWN 2026-09-05** (`work/m10paq/paq_draw.json`), decision 4, from Facebook's official
  release and never an HF mirror. **Release file:** `https://dl.fbaipublicfiles.com/paq/v1/PAQ.tar.gz`
  → `PAQ/PAQ.filtered.jsonl`, tarball sha256
  `177eefb2ddf8ab46a8d2248c058d5be52a4f2ce7614e55c1696f69fd0fe051c3`, 1,447,064,073 bytes,
  **population 64,875,601 pairs** (the release table says 64.9M). **Licence, primary source:** the
  tarball ships `PAQ/LICENSE` = the CC BY-SA 3.0 Unported legal code, and the release table states
  "The PAQ QA-pairs and metadata is licensed under CC-BY-SA"; the repo's CC-BY-NC covers the
  GENERATION CODE, which is not used. **Attribution required on the model card.** Only the
  `question` field is read — the answers are never used, and no objective could consume them.
  **Draw:** uniform without replacement, seed 0, margin 1.05 (set from the pilot's measured
  99.656% survival, `paq_pilot.json`), 4,238,850 drawn → exact-dedup **−2,172** → protected-index
  screen **−14,677 (0.35%: near 14,503, exact 23, contains 151)** → **4,222,001 survivors**, then
  shuffled before truncation. 767 s.

| sample | n | sha256 | bytes |
|---|---|---|---|
| A2 (volume control, screen arm A2 only) | **4,037,000** | `8f32bcdf602ff982…` | 275,051,180 |
| build (nested inside A2) | **1,000,000** | `5bd7b7283360caf1…` | 68,125,371 |

  **The build sample is a SUBSET of the A2 sample** — A2 is the volume control, so "the build with
  less PAQ volume" is the coherent nesting. Nothing registered specifies it; stated because it is
  an implementation choice. **PAQ's protected overlap being only 0.35% is worth knowing on its
  own:** PAQ is machine-generated over Wikipedia passages and NQ/TriviaQA/HotpotQA are Wikipedia
  questions, so a large overlap was the expectation — it does not reproduce their phrasing.
- Decontamination removals per screen, per form, per COV component; FORMS-12 hold-out seed ids.
- Teacher-target cache keys. ~~bank, mining method, recall@64 audit~~ — struck with the ranking-aware class (amendment A1).
- `results/m10_data_manifest.json` sha256.

### §Harvest — the A3 real-text pipeline, REGISTERED 2026-09-05 before its draw ran

Four extraction rules, exactly the registered set, deterministic and with no model in the loop
(`m10src/harvest.py`, `m10src/test_harvest.py`):

| rule | text | form | source |
|---|---|---|---|
| `title` | titles as-is | `title` / `keyword` by the frozen rubric's own ranges | Wikipedia, arXiv |
| `heading` | section headings as-is, apparatus sections dropped | `title` / `keyword` | Wikipedia |
| `claim` | declarative LEAD sentences, 8–40 words, a finite verb, no first person | `claim` | Wikipedia, arXiv |
| `ask` | sentences ending in `?`, preceding sentence kept as optional body | `factoid` / `product`, routed by the SOURCE store | the pool corpora |

**Draw rule, fixed before the numbers:** per form, a **uniform reservoir** (seed 0) over the union
of every source's rows after exact-text dedup — not weighted, not balanced, not scored. Harvested
text has no score to sort by, and uniform preserves the corpus's own distribution, which is the
whole point of an arm called "real text". **The realized source mix is REPORTED, never fixed in
advance.** Quota 417K / 417K / 416K ≈ 1.25M, margin 1.5.

**Screens are a generated string's, verbatim:** the M10 protected index on the query side; the
six's documents, all four DEV components' documents and every admitted COV component's documents
streamed against a candidate-side `Inverted` on the document side. Matches REMOVED. Running out of
margin raises rather than returning a short draw.

**Disclosures.** The `claim` rule **under-fires by design**: its finite-verb test is a closed,
explicit list after a unit case caught a regular-inflection branch firing on the plural noun in
"nine plain nouns", admitting a bare noun phrase as a claim — English plurals and third-person
verbs share a suffix and no regex separates them, and a parser would be the model the rule is
registered not to have. Supply is millions and never binding; a non-sentence in the `claim` form
is a real defect. **The reserved-DOCUMENT gap (W4) applies here exactly as to seeds**: a harvested
Wikipedia lead sentence can be near-identical to a DBpedia-entity abstract and no fingerprint
exists to catch it. **ESCI queries are NOT harvested** — they are real user queries already in the
M9 pool through `esci-us`, so harvesting them would double-count arm A1's data into A3. The
rubric's ranges leave a **gap between 4 and 6 words** (keyword ends at 4, title starts at 6), so
5-word titles are dropped; a property of the frozen rubric, disclosed rather than patched.

### §Harvest amendment 2026-09-05 — the quota table omits two registered forms; four `draw()` defects fixed

All PRE-DRAW: `draw()` had never run, no number downstream of it observed. Tier-2 consultation
logged in §3 (Fable, two passes). Yields: wiki 16,057,076 · arXiv 4,983,385 · pool 46,582 = **21,087,043 rows**
(`work/m10harvest/*.report.json`); by form keyword 9,125,216 · claim 8,260,059 · title 3,655,186 ·
product 40,977 · factoid 5,605.

**1. `factoid` and `product` have no quota row, so they are NOT DRAWN — the reason is the missing
registration, not their content.** `instructions-m10.md`:366 registers **five** harvested forms at
~250K each (paper-title · claim · keyword · **factoid** · **product**); the 2026-09-05 quota table
here registered only three (417K/417K/416K). The mandate also registers "a harvested form that
falls under 100K reverts to generation at ≈143K" (:372, :395) — at 5,605 and 40,977 rows **both
fall under it**. Executing that revert is a quota decision (two prompts, two smoke windows, +286K
over the registered 1.0M generation cap) and is **Dylan's at the M10.2 lock; default excluded** →
§Open questions. The `harvest.py` comment "plus whatever the `ask` rule returns" was never
registered anywhere and is **struck**. Rows are still harvested and reported, and the report now
carries `skipped_no_quota` so the exclusion is visible in the artifact.

**2. Measured `ask` quality, reported as the §1 template requires — a disclosure, and NOT the
reason for (1).** Post-rubric-range, post-dedup the rule yields ~15.2K rows (~1.2% of 1.25M), 84%
esci-prod. esci-prod: 24.2% exact-dup, 33.9% of rows in a repeated group, top in-range repeats
"what are you waiting for?" ×306, "want to make dad look like a super star?" ×142 — seller
marketing copy. `factoid`: ~50% open with a wh-word/auxiliary, top in-range repeats "who wants to
be a millionaire?" ×16, "where in the world is carmen sandiego?" ×7 — media titles ending in `?`,
plus splitter artifacts ('Cobb, Jr., whose book Is It Too Late?'). A future admission reads this first.

**3. The frozen rubric's word range is now enforced at the draw.** Binds `claim` only: the rule's
extraction window is 8–40 (`CLAIM_MIN/CLAIM_MAX`, kept, so the constants still match the rows on
disk and their reports) against the frozen `RUBRIC` range **(8, 25)** — the LEDGER's "8–40" was the
code constant, not the rubric, and the rubric is the standard. `title`/`keyword` are in range by
construction (`route_by_length` routes BY that range). Direction-safe (removes only), no
claim-length number observed, supply after the range ~5.9M (71.6% of 8.26M) against a 416K quota.
Per-form `off_rubric_range` reported.

**4. Four `draw()` defects, all Tier 1** (`m10src/harvest.py`; five new tests in `test_harvest.py`,
12 green):
- **Truncating the reservoir to its first `n` slots is NOT uniform.** Algorithm R overwrites slot
  *j* with later items only, so slot *j* can hold initial item *j* and no other; first-*n*-of-*want*
  over-represents stream positions [0, n) by exactly the margin (**1.50×**, simulated 75.0 vs 50.0
  expected) and draws positions [n, want) **zero** times (0.0 vs 25.0). The stream is Wikipedia dump
  order, whose prefix §T2-5 measured as a **5× distortion** — the bias would land on the
  known-skewed population. Fixed by shuffling with the draw rng before truncation, which is what
  "uniform reservoir" already promised. Regression test asserts the [n, want) band is drawn >60
  times; it reads exactly 0 without the fix.
- `_iter_rows` **silently skipped a missing path** — a draw started before a pass finished would
  have reported a clean corpus with a whole source absent. Now raises on missing, unreported, or
  `complete != true`.
- `seen_txt` (~21M normalized strings, 3–4 GB) was **held through pass 2**; freed after pass 1.
- Rows dropped for having no quota row were **invisible**; now `skipped_no_quota`.

Disclosed, not fixed: dedup is **case-folded and whitespace-normalized**, not literally "exact
text", and its scope is **cross-form** (the one collision class is a 6+ word title ending in `?`;
routing is by length/punctuation so the forms are near-disjoint by construction). The `ask` rule's
3–40 word window lives in code, not here. `body` is screened on `text` only — moot while `ask` is
undrawn, and a blocker for any future admission of it.

Also found, unrelated and cosmetic: `m10src/calib.py:run_arm` saves a **fresh** `AdamW` beside the
trained model, so `work/m10calib/P*.pt` carry no usable optimizer state. Harmless (the P arms are
never resumed; the build warm-starts from the M9 candidate) but never warm-start an arm from them.

## §2 COV admission records — fill at M10.0-d, one row per component

Structures verified and screened 2026-09-04 (`work/m10cov/structure.json`, `work/m10cov/screen.json`,
`m10src/cov_admit.py`, `m10src/cov_screen.py`). Licence column is the `m10/COV_CANDIDATES.md` finding,
unchanged. Screen = M7 fingerprints, near-match ≥ 8/32; candidate-side `Inverted`, the six's documents
and the full protected query index streamed against it. **Reserved DOCUMENT side not screened — W4.**

| component | family | repo · revision | licence at primary source | corpus / queries / qrels / metric | fingerprint screen vs the six's docs + protected queries | verdict |
|---|---|---|---|---|---|---|
| MedicalQARetrieval | `consumer-health` | `mteb/medical_qa` · `a77efe81` | CC BY 4.0 at MedQuAD; HF card tags CC0, mismatch disclosed | 2,048 / 2,048 / 2,048 binary; nDCG@10 | **0 exact, 0 near**, both sides | **ADMIT** |
| BRIGHT (6 slices) | `BRIGHT` | `xlangai/BRIGHT` · `3066d29c` | CC BY 4.0 at the primary source; documents are third-party pages, caveat disclosed (Codex dissent recorded in COV_CANDIDATES) | 404,416 docs / 632 queries across biology 57,359·103, earth-science 121,249·116, economics 50,220·103, psychology 52,835·101, robotics 61,961·101, sustainable-living 60,792·108; graded; slices averaged into one family macro | queries **0/0**. Documents: the raw count is 6,123 exact, and **it is an artefact** — BRIGHT ships 91,626 documents under 8 words (23% of the corpus; 4,606 are the literal `".\n"`, 676 `"copy link"`). Restricted to the 312,790 documents at or above the 8-word fingerprint floor: **0 exact, 23 near (0.008%)** (`work/m10cov/bright_len_filtered.json`) | **ADMIT** as one family, unfiltered so it stays the published benchmark; the boilerplate share is a disclosed corpus property, not a contamination finding |
| LegalBenchCorporateLobbying | `legal` | `mteb/legalbench_corporate_lobbying` · `f4343695` | CC BY 4.0 (LegalBench README) | 319 / 340 / 340 binary | **0 exact, 0 near** | **ADMIT** (weak component, tiny corpus) |
| LegalBenchConsumerContractsQA | `legal` | `mteb/legalbench_consumer_contracts_qa` · `f9eafd45` | CC BY-NC 4.0 — admissible for validation under Dylan's 2026-09-04 rule; COV never enters training | 154 / 396 / 396 binary | **0 exact, 0 near** | **ADMIT** |
| **LEDGER** | **`finance`** | `artefactory/ledger-long-context-KPI-QA` · `7881df568382` (the collection's only QA member; there is no `artefactory/LEDGER` repo) | data CC BY 4.0, code MIT | **10,000 queries · 494 reports → 47,820 pages · 116,912 graded qrels (0/1/2)**; page-level retrieval, nDCG@10 | queries **0 exact / 0 near**. Pages: raw 710 exact, again the sub-8-word artefact (1,444 pages, 3.0%); above the 8-word floor **0 exact / 1 near (0.002%)** | **ADMIT.** Chunk rule = the dataset's own page split on the literal marker `<--- Page Split --->`, **verified**: every one of the 116,912 qrel page ids resolves in the split (0 missing), 96.8 pages/report, 47,820 pages — under the 100K cap with no cap applied. Disclosed: template-generated queries with DBpedia aliases, LLM-judged qrels, some queries judged across adjacent years, 1,444 near-empty pages |

**Family floor: PASSED with margin. FOUR family IDs** — `consumer-health`, `BRIGHT`, `legal`,
`finance` — against a registered STOP of "fewer than three". No component shows contamination.

**WITHDRAWN, and the error is kept because it will recur.** LEDGER was **refused on 2026-09-04 and
the refusal was wrong on fact.** The claim "no qrels and no corpus, only report-level relevance"
came from `load_dataset_builder(...).info.features`, which returns a **stale 8-column list**; the
loaded dataset has 13 columns including `qrels`. Every downstream inference — no passage-level
relevance, a degenerate company-name task, a near-saturated component — was sound reasoning on a
false premise, and it was asserted confidently in a commit. **Read the artefact, not the metadata
about the artefact.** Found by the Fable post-execution review, which loaded the rows.

**Consequence for the screen's power, and it is not small.** §Surfaces expects the resolution
distance at 0.009–0.0135 against the fixed MDE 0.0056, so most B–G contrasts were already expected
unresolved, and LEDGER was named there as "the one candidate large enough to move the surface's
power". That remedy does not exist. The admitted surface carries **3,416 queries** in total
(MedicalQA 2,048 · BRIGHT 632 · CorporateLobbying 340 · ConsumerContractsQA 396). CUREv1 cannot help:
decision 12 makes it a reported diagnostic, never selection-bearing. **The registered resolution
number is therefore measured, not estimated, and reported to Dylan as the power disclosure before
the screen runs** — §3 W5.

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
| 2026-09-05 14:xx | (this commit) | **T2-9 the `ask` rule's forms have no quota row.** `instructions-m10.md`:366 registers five harvested forms at ~250K; the §Harvest quota table registers three. Measured yield: `factoid` 5,605, `product` 40,977 — both under the mandate's 100K revert-to-generation threshold (:372) | (a) include uncapped on top of 1.25M; (b) register quotas for both; (c) five forms share 1.25M; (d) delete the rule, skip `pool_pass`; (e) `factoid` only; (f) execute the registration as written | *"Your framing is wrong in one place … the registered state is a sixth option you have not written down"* — (a)/(b)/(c)/(e) each change a registered number after observing yields, so they are **Tier 3 in disguise**; (d) is **not** conservative, since its premise ("yields nothing usable") is false and skipping `pool_pass` deletes a yield measurement the §1 template requires. (f) changes nothing = Tier 1. On the second pass, given the measured quality: *"they lower the temptation to deviate; they are not the reason the rows are out … The moment the LEDGER says 'excluded **because** the content is marketing boilerplate', you have made a quota decision on observed data and dressed it as a disclosure. That is the laundering, and it is one word away."* Projected effect of admitting them **< 0.001 against MDE 0.0056** | (f), with the exclusion attributed to the missing quota row and the quality measurement logged as a separate finding | The rows could not have been included even had the content been excellent — there is no quota row and adding one is Tier 3 — so the exclusion is registration-driven and not data-dependent. Yield + quality reported per the §1 template; admission is Dylan's at M10.2, default excluded (§Open questions W7) |
| 2026-09-05 15:xx | (this commit) | **T2-10 `qfilter` — the frozen rubric's word range is ENFORCED, on generated output and at the harvest draw.** It was implemented and measured but **never logged as a decision** (a Fable pass flagged the gap): it changes what enters the build manifest, so it needed a §3 row before generation, not only a `STATUS.md` line | (a) enforce each form's own `RUBRIC` range; (b) leave out-of-range strings in and report the rate; (c) invent a single global length window | (a). The ranges are parsed out of the **frozen** `RUBRIC` at import, so the filter cannot drift from the text the judges score against and a rubric edit moves both together — it enforces an already-registered spec rather than adding a standard. Direction-safe: it only removes | (a) | Largest on-form lever measured in M10 (`results/m10_qfilter_effect.json`): health **0.780 → 0.857**, finance **0.790 → 0.806**, and out-of-range strings score **0.000** (wikipedia-body) and **0.0385** (incumbent) on-form — they are almost pure noise. Drop rates are small (health 8.7%, finance 3.6%) and supply pays for it (51,633 health seeds against a 33,000 need). Per-form drop counts go to the manifest, never silently absorbed. Also applied at `harvest.draw` where it binds `claim` (§Harvest amendment item 3) |
| 2026-09-05 15:xx | (this commit) | **T2-11 does T2-8's rung 2 still have to run?** The rung-1 artifact's `_registered_outcome` says *"rung 2 — LLM classification of the lead sentence — is next as registered"* after rung 1 missed the 0.80 gate (health 0.655, finance 0.535). Generation is ~10 GPU-hours on one busy card, so the GPU goes to rung 2 or to generation | (i) run rung 2 first; (ii) rung 2 retired for admission but run as a quality lever; (iii) rung 2 does not run, generation proceeds | *"Your **conclusion** is right … your **reasoning** is wrong in a way that will bite you at the next step, because Record B is itself a withdrawn record."* I had argued from the W6 draft at `:377`; the operative section is **§W6 RESOLVED (`:280`)**, which states *"there is now no admission bar for a seed store"* and *"T2-8's rung ladder is DEMOTED to a diagnostic: rungs 2 and 3 no longer gate anything, and rung 1 stays applied because it measurably raised health precision by 0.110."* Against (ii): Dylan ruled **with rung 2 explicitly on the table** (`:312`), T2-7 ⑦ already stopped at three instruments for one store question, finance moved **0.000** under rung 1, and rung 2 cannot see off-topic chunks inside on-subject articles | (iii) | **And one of my own bullets was WRONG and is withdrawn:** I wrote that "admission is decided after generation, on the generated queries". There is **no** post-generation admission test — applying the withdrawn conjunct 3 (floor 0.754) after seeing the corpus would re-instantiate a killed rule on observed numbers, Tier 3 twice over. What gates the output is only what was already registered: the **A8 manifest gates** (near-dup > 25% → representatives only; < 50,000 retained → form dropped) and the **FORMS-12 hold-out**. Two stale pointers fixed in this commit |

- **2026-09-04, W2 RULED by Dylan: "approve both".** `howto` (80.0%, at the threshold) and `argument` (88% on the registered 50-query sample, **67% across its full output**) are **approved into the build**, notwithstanding the procedural defects disclosed in `m10/SMOKE.md`. With `conversational` cleared on evidence, **all seven generated forms are approved**. `argument` ships with 67% reported as its honest on-form rate, and with the conditions the Fable pass attached: the build-time on-form diagnostic drawn as a **uniform random** sample (not round-robin), the A8 near-duplicate rate and mean pairwise cosine reported prominently because word-8-gram sketches cannot see template collapse, and the 5 exact duplicates in 200 long paragraphs treated as a 4-bit repetition warning. **W1 (the terminal-rule reading) remains open** and is not needed for this ruling.

| 2026-09-05 02:1x | (this commit) | **T2-4 — amends T2-3.** T2-3 registered the build's topical seed draw as `ROUTE` + `min_score ≥ 4`. The build draw now uses **`ROUTE_WIDE` v2** and `SCREEN_VERSION` is bumped so cached draws under the old routing are invalid | keep T2-3's routing and accept `health` at ~10.4K seeds; widen the keyword lists (recall, precision held); relax `min_score` (precision cost); raise queries-per-seed | *"the first lever is recall at held precision… lowering `min_score` is a precision lever that admits passages with two mentions of 'blood'. Widening dominates and should be a rung before relaxing the floor. It must be registered now, as a fixed list, before P1 counts — not tuned until the number lands."* | widen first, registered before the scan | **v1 was registered before its scan and then WITHDRAWN as defective**: it kept the `\b(alt|…)\w*\b` shape, so `pain`→"paints", `nurse`→"nursery", `chronic`→"chronicle", `capital`→"capital city", `credit`→"credited". v1's counts are withdrawn with it. **v2 lists explicit word forms**, and — because v1's failure was precision *asserted* rather than measured — a judged-precision gate on v2's newly-admitted passages is registered before the widening is used |

| 2026-09-05 | (this commit) | **T2-5 — `wikipedia-body` admitted as a seed store for `health` and `finance`, closing W3.** Body paragraphs (lead section EXCLUDED) of `wikimedia/wikipedia` `20231101.en` @ `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, chunked in the registered 40–220 word window, routed by the registered `ROUTE` at the registered `min_score >= 4`, **cap 3 seeds per article per form** | (a) the registered next lever, a lead-sentence subject filter — it CANNOT work: it raises precision on a marginal whose usable ceiling is ~17.6K against a ~33K need, so it cannot close a supply gap; (b) rung 4, queries-per-seed 5 -> 8 — cuts distinct seed topics ~45% for the same quota and needs a re-smoke and a fresh veto window; (c) do nothing and report `health` at ~60% of quota | *"Adopt with named changes"* — Wikipedia is a **registered** seed source (`instructions-m10.md`:412; `research/m7-data-licensing.md`:42,61, CC BY-SA position confirmed by Dylan 2026-08-25), so this is not a new admission and the FineWeb closure does not reach it (FineWeb was unregistered and its overlap with reserved text unbounded; body-only overlap is bounded by construction). Lead exclusion judged sound, not a fig leaf: DBpedia-entity is abstracts and FEVER is introductory sections, so a body-only store is disjoint from both **by construction** and is strictly cleaner than `hotpotqa-corpus`, which is already at 9.3% / 11.2% R3 near-dup against them. Named changes, all adopted: verify the lead rule by hand before the scan; scan the FULL dump or a hash-random subset, never a prefix; per-article cap; scope to health and finance only; screen and REMOVE against the protected query index, the six's documents, dev documents and admitted COV documents; gate 200 seeds per form (not 50) drawn from the population the BUILD will use, with a 200-seed control from the incumbent `ROUTE` intro seeds judged in the same run; pin the HF revision; model-card attribution line | adopted with every named change | **The prefix-bias warning was right and is why the pilot number is not the registered one.** The first 20,000 articles in dump order are 2001–02 core articles at 13.1 chunks each and projected 92,000 health seeds/1M; a **shuffled** 5,000-article sample gives 4.57 chunks/article and **18,200 health / 38,000 finance per 1M after the cap** — a 5x correction, still **3.5x the ~33K need** over the 6.41M-article dump. Lead rule hand-verified on 60 prefix and 22 shuffled articles including stubs: boundary correct in every case, 0/60 articles with no heading. **The claim that body text is HIGHER precision than intros is NOT adopted** — the judge's counter-cases (biography "Death"/"Illness" sections, settlement "Economy" sections) are exactly the failure mode that sank `ROUTE_WIDE`, so precision is decided by the gate and by nothing else. Divergence disclosed: the mandate says "stratified by top-level category" and this routes by keyword, as T2-3 already does; category-membership routing over the `categorylinks` dump is the registered next lever if the gate fails — **never a relaxed floor** |

| 2026-09-05 | (this commit) | **T2-6 — `wikipedia-body` FAILS its registered gate and is NOT ADOPTED; the control is the finding.** Blinded, interleaved, 200 candidate + 200 incumbent passages per form, four independent Fable judges, frozen `forms.RUBRIC` standard | adopt on the counts; adopt on the incumbent comparison; refuse and report | (the gate itself is the ruling; no consultation) | **not adopted** | `results/m10_wikibody_precision.json`. **health 0.545 vs the incumbent's 0.590** (−0.045, z = −0.91, indistinguishable); **finance 0.535 vs the incumbent's 0.315** (+0.220, z = 4.56). Gate is ≥ 0.80: **both stores fail it.** The incumbent `ROUTE` intro store — the one the approved smoke prompts were built on and the one the build would otherwise use — had **never had its base precision measured**; only the MARGINAL of the `ROUTE_WIDE` widening ever was, against this same bar. So "not adopted" does not return the build to a store that meets the bar. Whether an 80% bar written for a widening's marginal is the right bar for a whole seed population is **Tier 3 and Dylan's** — and must not be answered by moving the bar now the numbers exist (W6). Every failure class the four judges named is an **article-subject** error, so the registered next lever — a subject-level filter, never a relaxed floor — is aimed exactly right |

| 2026-09-05 | (this commit) | **T2-7 — the build on-form diagnostic, REGISTERED BEFORE ITS FIRST GENERATION CALL, and it admits nothing.** T2-3 already registered a "report-only build on-form diagnostic"; this executes it early, on two forms and two stores | (a) run it and let a good on-form rate admit `wikipedia-body`; (b) run it report-only and build the subject filter regardless; (c) skip it | **APPROVED WITH CONDITIONS, and (a) REFUSED**: *"Attaching admission to a different, kinder-looking instrument after the registered one returned FAIL is a protocol change after observation — Tier 3 by the same logic you applied in W6."* My 59%→84% argument was **killed**: the smoke's seeds were the top-40 keyword scorers of 400K, the gate's control was uniform over the whole 8,663/19,466 store, so the pair is (top tail, uniform population) and says nothing about absorption. Fable also corrected the failure model: off-subject seeds most plausibly hurt via **A8 diversity collapse**, not off-form queries, so the diagnostic must carry A8-style metrics or it measures the wrong thing | (b), with all eight conditions | **Registered design, fixed here:** ① seeds = the exact 400 already-judged passages per form (`work/m10gen/gate/key.json`), so every query links to a seed-precision verdict and P(on-form \| off-subject seed) is measurable — a fresh draw cannot separate absorption from "the on-subject half did all the work"; ② generator contract unchanged, `Qwen/Qwen3-8B-AWQ` `4da05a8e…`, approved prompt hashes, **`n = 5` unchanged** (changing `n` moves the approved prompt basis and needs a re-smoke); ③ judged sample = **one query per seed, uniform-random among its five**, 200 per arm per form, 800 total — not 50, which cannot tell 80 from 72; uniform-random, not round-robin (Dylan's W2 ruling); ④ blinded and interleaved in one permutation per form, arms and seed-verdict classes mixed, key held outside the judge's files, independent Fable judges, `forms.RUBRIC` the only standard; ⑤ reported per arm per form: contract rate, on-form rate with SE, **on-form conditional on the seed's precision verdict**, exact duplicates, A8 near-duplicate rate at 16/32 and mean pairwise stella cosine (both underpowered at this n, disclosed); ⑥ **outcome map: the diagnostic ADMITS NOTHING.** It informs W6; the only action it licenses is that `wikipedia-body` reading WORSE than the incumbent stops the subject-filter line on that store; the subject filter proceeds regardless and is re-gated against the **unchanged 0.80** seed-precision bar; ⑦ **stop rule:** this is the third instrument aimed at one store question, and the session brings no fourth — anything further is Dylan's; ⑧ gate artefacts never enter a build corpus (T2-2) |
| 2026-09-05 | (this commit) | **T2-8 — the subject-filter LADDER, fixed before rung 1 runs.** Two different "registered next levers" existed in two places — category-membership routing (T2-5) and lead-sentence patterns (W3/HEADROOM) — and they are not the same lever | pick one; register a ladder | *"Register the ladder explicitly … Stop at the first rung that clears 0.80, with the same no-peeking rule"*; regex first is *"the right first rung and is not over-engineering; the LLM classifier as a first move would be"* | the ladder | **Rung 1** lead-sentence head-noun regex · **rung 2** LLM classification of the lead sentence (pinned, seeded, disclosed; the judge is a different model, so the gate is not circular) · **rung 3** `categorylinks`. **Stop at the first rung that clears 0.80; no trying the next one "to see".** The lead is used for a boolean route decision only and is never stored or emitted — asserted by test, since the lead is what the store exists to exclude. The re-gate samples the **new** drawn pool: the top 33K of a filtered store is a different population from the one already judged. **Supply after filtering is TIGHT for health and must be checked, not assumed:** 72,626 × 0.545 ≈ 39.6K on-subject, less the ~4% screen and any false negatives, against a 33K need. Finance is comfortable (≈88K). The incumbent cannot be rescued by any filter (8,663 × 0.59 ≈ 5.1K), so `wikipedia-body` + a filter that clears the re-gate is the ONLY route to 33K on-subject health seeds; if it does not clear, health's quota is Dylan's |

**T2-7 RESULT (2026-09-05, `results/m10_onform_diag_{health,finance}.json`) — report-only, admits nothing.** 400 already-judged seeds per form, one uniform-random query of five per seed, 200 per arm per form, blinded and interleaved, four independent judges, frozen `RUBRIC`.

| form | `wikipedia-body` on-form | incumbent on-form | diff | z |
|---|---|---|---|---|
| health | **0.780** | 0.735 | +0.045 | +1.05 |
| finance | **0.790** | 0.635 | +0.155 | **+3.48** |

**The mechanism, measured instead of argued.** On-form given an ON-subject seed: 0.881 / 0.831 (health), 0.925 / 0.936 (finance). Given an OFF-subject seed: 0.659 / 0.598 (health), 0.634 / 0.496 (finance). **Seed subject precision propagates but does not determine** — an off-subject seed roughly halves the odds of an on-form query without destroying them. My absorption claim was too strong; the flat dismissal of it was also too strong; neither had to be guessed. **Diversity: 0 exact duplicates and a 0.0 near-duplicate rate at 16/32 in every arm** — A8 collapse, the failure Fable predicted as the real cost of off-subject seeds, does not appear at this n (underpowered; the stella-cosine half is the manifest-time gate and is not computed here).

**Two things Dylan needs from this, neither of them a decision I may take.** ① The forms were approved on a seed draw the mandate itself flags as non-representative (T2-3's top-40-of-400K), and on the ACTUAL build population the approved prompts read **0.735 health / 0.635 finance on-form — below the 0.80 bar the forms were approved at.** ② The registered outcome map licenses exactly one action and it is not admission: `wikipedia-body` reading WORSE would have stopped the subject-filter line, and it reads better on both, so the filter proceeds and is re-gated against the **unchanged** 0.80 seed-precision bar.

| 2026-09-05 | (this commit) | **T2-9 — §Kill's "two consecutive scheduled evaluations" has two readings.** Consecutive **in the schedule** (a midpoint and the cycle end after it, each below its own kind's best) or consecutive **within a kind** (two successive midpoints) | the schedule reading; the per-kind reading | (no consultation — a plain-sense reading of a registered constant, logged rather than chosen silently, as T2-1 was) | the **schedule** reading | It is the plain sense of the words, and it is the safer of the two: it demands both kinds be failing at once, where the per-kind reading kills an arm on two bad midpoints alone and a false kill costs a whole arm. The rule's stated purpose — *"so the rule can fire inside the build and not only at its end"* — is served either way, because it is satisfied by midpoints entering the comparison at all. Implemented in `m10src/nano10.kill_fires`; `test_nano10` asserts BOTH that the taken reading fires and that the reading not taken does not, so the choice is visible in the tests rather than buried. **Dylan may strike this**; if he meant per-kind, one line changes and no number is affected — nothing has trained |

| 2026-09-05 | (this commit) | **T2-10 — the RE-GATE's design, fixed before the filtered pool exists.** Three arms, not two: `wikipedia-body` **+ rung 1** (the candidate, gated at the unchanged 0.80) · the incumbent **unfiltered** (the same control as the first gate, so the two gates are comparable) · the incumbent **+ rung 1** (reported) | two arms as before; three arms | (no consultation — this adds a CONTROL to the existing instrument, not a fourth instrument, so Fable's stop rule is not touched) | three arms | Without the third arm a pass is ambiguous between "the filter works" and "`wikipedia-body` is better", and those license different next steps. An intro passage IS its article's lead, so the subject patterns apply to the incumbent's passages directly. Reported only: Fable's note stands that filtering cannot rescue the incumbent's SUPPLY (8,663 × 0.59 ≈ 5.1K against a 33K need), so this arm is about precision alone. Sampling is uniform within each arm's own pool, blinded and interleaved in one permutation, 200 per arm per form, batches of 200, the frozen `RUBRIC` the only standard — every constant unchanged from the first gate |

**T2-8 rung 1 RESULT (2026-09-05, `results/m10_wikibody_precision-r1.json`).** Three arms, 600 judged items per form, six independent judges, gate unchanged at 0.80.

| form | `wikipedia-body` + rung 1 | incumbent | incumbent + rung 1 | gate |
|---|---|---|---|---|
| health | **0.655** (was 0.545, **+0.110**) | 0.595 | **0.770** | 0.80 |
| finance | **0.535** (was 0.535, **+0.000**) | 0.350 | 0.490 | 0.80 |

**Rung 1 works on health and not on finance, and nothing reaches 0.80.** It rejected 89,215 of 238,823 rows (37%) — person 38,000 · organisation 21,249 · place 17,323 · work 6,098 · taxon 4,679 · nonhuman 1,866 — and health still fills 33,000 from 51,633 (1.56×). **`wikipedia-body` is STILL NOT ADOPTED.**

**The trade-off, stated plainly because it is Dylan's to make:** the highest health precision measured anywhere in this project is the **filtered incumbent at 0.770**, one SE below the bar — and its screened pool is **6,625 against a 33,000 need**. The best-precision option cannot supply the build; `wikipedia-body` + rung 1 supplies 51,633 at 0.655. W6 decides whether either is admissible at all.

**Rung 2 is next as registered.** — **SUPERSEDED the same day by §W6 RESOLVED (`:280`), which
DEMOTES the whole T2-8 ladder to a diagnostic: "rungs 2 and 3 no longer gate anything, and rung 1
stays applied because it measurably raised health precision by 0.110." Rungs 2 and 3 do NOT run.**
`results/m10_wikibody_precision-r1.json` `_registered_outcome` carries the same stale sentence and
is deliberately left unedited — a results artifact is a frozen record and the pointer belongs here.
The residual failures the six judges named are what a definitional-pattern regex cannot reach: institutions whose lead is not definitional, bare lists of journal names and JEL codes, and word-sense errors inside a genuine-looking lead ("market" as a physical marketplace, "bank" as a riverbank, "trade" as a craft, "Banker" as a mycologist's surname). **Note for whoever runs it:** some `wikipedia-body` failures are chunks off-topic inside an ON-subject article, which a LEAD-sentence classifier of any kind cannot see. Classifying per CHUNK would reach them — but that is a different lever from the one T2-8 registered, and choosing it now, after these numbers, is an amendment that needs a Fable pass first.

### W6 — RESOLVED by Dylan, 2026-09-05, AFTER two adversarial passes withdrew the rule I drafted

**The ruling: use `wikipedia-body`, report the measured numbers, and invent no standard to bless
it.** There is now **no admission bar for a seed store.** The store is used because it is the only
one that can supply either form's quota; its quality is REPORTED, never certified. Revisitable
later "if necessary and with reviewer approval" (Dylan).

**What the report will say, unvarnished:** seed subject precision **0.655 health / 0.535 finance**
after T2-8 rung 1; generated-query on-form **0.780 / 0.790**; the incumbent alternative reads
0.595 / 0.350 and 0.735 / 0.635 and **cannot supply either quota** (8,663 and 19,466 screened seeds
against a 33,000 need). No statistic is constructed to turn any of that into a pass.

**Why the practical stakes are small, with the arithmetic so a reader can check it.** The two forms
are ≈286K of a ≈3.71M-query corpus (M9 pool 463,314 + PAQ 1.0M + harvested ≈1.25M + generated
≈1.0M), i.e. **≈8%**. Within health, 34.5% of seeds are off-subject and an off-subject seed yields
an on-form query 0.659 of the time against 0.881 — so the seed problem costs **7.7 points of
on-form on that form**, ≈11K of 143K queries. Net: **under 1% of training examples are wrong-form,
and none of them is mislabelled** — the teacher's embedding of any text is a correct target by
construction (§Data, amendment A8's own rationale). The synthetic risk is distribution shift and
diversity collapse, not wrong labels, and those are A8's job.

**The drafted admission rule is WITHDRAWN, and why is kept because the failure mode will recur.**
A Fable pass and a Codex statistical pass, briefed independently, agreed:

1. **It was a weaker bar dressed as the old one.** "Not significantly below 0.80" at n = 200 fails
   only at ≤ 150/200, so the operative floor was **0.755, not 0.80**. A store whose true rate is
   0.78 passes **83%** of the time under it against **45%** under the approval rule it claimed to
   preserve. Codex: failure-to-reject is the wrong instrument outright; the correct form is
   one-sided **non-inferiority**, which at margin 0.05 needs **X ≥ 161 (0.805)** — and the measured
   156/200 and 158/200 **fail it**. The framing turned a fail into a pass.
2. **The option set put to Dylan was SKEWED — every option led to admitting the store.** Never
   offered: reduce the quotas (his lever, already named in W3 and T2-8, and the only option that
   buys precision instead of supply); drop the two forms; finish the rung-2 ladder first; per-chunk
   classification. Dylan was told this and ruled anyway, on the arithmetic above.
3. **Corrections of fact in the withdrawn draft:** the incumbent fails supply on BOTH forms, so the
   three-conjunct test had exactly one possible admittee; "the rule precedes the numbers it governs"
   was a fig leaf, since 0.655·0.881 + 0.345·0.659 = **0.804** was already predictable from measured
   conditionals; conjunct 2 could not fail; and `argument` already ships at 0.67 by Dylan's W2
   ruling, so invoking 0.80 as a fixed standard was inconsistent on its face.

**What survives and is not withdrawn:** the owner's AUTHORITY over a seed-store gate — it is a
data-recipe constant, not one of `CLAUDE.md`'s protocol exceptions, and contamination exposure does
not grow (body-only is disjoint from DBpedia and FEVER by construction; the incumbent is itself
Wikipedia). **T2-8's rung ladder is DEMOTED to a diagnostic**: rungs 2 and 3 no longer gate
anything, and rung 1 stays applied because it measurably raised health precision by 0.110.
**A4−A3's VALUE is now disclosed as conditional on the store**: that contrast reads "what generation
from a 0.655 / 0.535 on-subject Wikipedia-body store adds", and the report says so.

### M10.0-e — SAME-INIT CALIBRATION, re-registered 2026-09-05 after Codex refused the first framing

**It is not a power study and it does not close W5.** Codex: one LR pair "cannot close W5 or
establish global screen power"; runnable "only if relabeled as a descriptive, conditional
calibration for that exact LR pair". Dylan approved it on that basis. Registered scope:

- **What it measures:** the paired width of a contrast whose two arms share backbone, tokenizer,
  init, seed, data order and warm-start head. That brackets **B, D, G and C**, and nothing else.
- **What it does NOT measure:** **F and E.** bge-small against MiniLM-L6 shares only the teacher
  target, which in per-query-correlation terms IS the unrelated-models case — so **F and E are read
  against the 0.008619 already measured**, and no LR pair is allowed to speak for them. Both
  reviewers flagged the first framing as an instrument built to return the favourable answer.
- **Two statistics, not one** (Fable): the bootstrap distance is QUERY-sampling noise for a fixed
  pair; the SEED effect is the point estimate `|macro(P0) − macro(P1)|`, which the bootstrap cannot
  see because it resamples queries, not seeds. Both are reported. A screen that "resolves" a
  difference smaller than the seed effect has resolved noise.
- **Corpus bias, with its sign** (Fable): two M9-pool models miss finance and legal the same way, so
  they agree more than two M10-corpus models would — another under-estimate, compounding the LR
  proxy's. Disclosed with its direction, not corrected.
- **P0's relation to A1** (Fable): P0 is anchor-shaped on the M9 pool at 5M seed 0, which is arm A1
  with bge-small. A1 carries no decisive contrast (A2−A1 is descriptive), so nothing leaks; the read
  is recorded as COV read #2 and **A1 is retrained in the screen regardless**.
- **No DEV-6, FORMS-12 or CUREv1 read happens on any P arm.** COV only.
- Changes no constant: MDE 0.0056 and α 0.025/13 are untouched, no arm is added, removed or
  reordered.

**Why.** The resolution number 0.008619 was measured between UNRELATED models, and the mandate
records that this over-estimates a same-init contrast's width without saying by how much. Family F
costs ≈17 GPU-hours and every later verdict is taken on its winner, so the size of that
over-estimate is worth ≈5 GPU-hours before the screen starts. Dylan approved 2026-09-05.

**Design.** Three arms on the **M9 pool** (A1's corpus — its teacher targets already exist), at the
full **5M screen dose** so the variance is estimated at the dose the screen actually uses:

| arm | shape | what its pairing measures |
|---|---|---|
| P0 | anchor shape, seed 0 | — |
| P1 | anchor shape, **seed 1** | `SD(P0−P1)` = pure seed noise, the lower bound on a contrast's width |
| P2 | anchor shape, **peak LR 8e-5** instead of 1e-4, seed 0 | `SD(P0−P2)` = a representative CONTRAST's width, the number wanted |

**The lever is peak LR precisely because it is NOT one of the thirteen registered contrasts**, so
nothing here can leak a hint about a screen verdict. The corpus is the M9 pool and not M10's, so
these models sit at a different quality level; disclosed, not corrected.

**What it changes: nothing.** MDE 0.0056 and α 0.025/13 stay fixed, no arm is added, removed or
reordered. It is a power disclosure exactly as the first one was, reported beside every contrast.
**Outcome map:** a same-init distance below 0.0056 closes W5 — the screen is adequately powered as
registered. At or above it, Dylan decides on the MDE **before family F starts**, never after.

### W6 — RULED … — **SUPERSEDED AND WITHDRAWN. See §W6 RESOLVED above (`m10/LEDGER.md:280`).**

> **Do not read the section below as operative.** Its three-conjunct test (supply · A8 diversity ·
> a fidelity floor of 0.754 on generated queries) is the rule two adversarial passes withdrew, and
> §W6 RESOLVED replaced it with **no admission bar for a seed store**. There is therefore **no
> post-generation admission test of any kind**; re-applying conjunct 3 after seeing the generated
> corpus would re-instantiate a killed rule on observed numbers. Kept verbatim because the failure
> mode — a session inventing a standard to bless its own store — is the lesson.

#### (withdrawn) a seed store is admitted on the QUERIES it produces

**The ruling.** Seed subject precision stops being the admission instrument. A store is admitted on
what the build actually consumes: the generated queries' fidelity and diversity, plus supply.

**The test, three conjuncts, ALL required.** Measured at M10.1 on the build manifest, per form, per
store:

1. **Supply** — the store fills the form's registered quota (≥ 33,000 screened seeds at the
   registered 5 queries per seed for a 143K quota).
2. **Diversity** — A8's registered gates on the generated queries: near-duplicate rate below 25% at
   the registered 16/32 threshold, mean pairwise stella cosine disclosed.
3. **Fidelity** — the generated queries' on-form rate, judged blinded against the frozen
   `forms.RUBRIC` at n ≥ 200 per form per store, is **not significantly below the 0.80 at which the
   forms were approved** (one-sided binomial, α = 0.05; at n = 200 that is a floor of **0.754**).

Among stores passing all three the build uses the highest on-form rate; a tie inside one SE goes to
the larger supply.

**Why this is not the forbidden change, stated plainly rather than assumed.** The evaluation
protocol — partitions, decontamination, the frozen comparator vectors, the single final run, the
pre-registered statistics — is untouched, and no reported number depends on which store the seeds
came from. This is a **data-recipe** gate, which `CLAUDE.md` puts explicitly on the list that is
"fair game to reopen with evidence and Dylan's sign-off". The session raised it; the owner ruled it.

**Four disclosures that ride with the ruling.**
- **0.80 is not invented.** It is the same on-form bar the forms were approved at (decision 15).
  Only the TEST is new — "cannot be ruled out at 0.80" rather than "point estimate ≥ 0.80" — and
  the reason is on the record: the approval sample was T2-3's top-tail draw, which the mandate
  itself flags as non-representative, so a build-population estimate is expected to sit lower.
- **The decision reads numbers that do not yet exist.** Every on-form figure measured so far
  (0.780 / 0.790 vs 0.735 / 0.635) is on the PRE-filter pool and is a diagnostic. Conjunct 3 is
  re-measured on the actual post-rung-1 build corpus, so the rule genuinely precedes the numbers
  it governs.
- **The seed-precision gate is not deleted.** It is reported beside every store as a diagnostic.
  It stops being the instrument that admits.
- **Supply already decides health on its own:** the incumbent store holds 8,663 screened health
  seeds against a 33,000 need and fails conjunct 1 outright, under any fidelity standard.

### Open questions for Dylan — raised by the weekend window, NOT resolved here

| # | question | why it is not mine to answer | cost of waiting |
|---|---|---|---|
| W1a | **A reading I proposed and then WITHDREW (2026-09-04, same evening).** I argued the three texts harmonise: that "revised at most twice" and "fails twice" *must* resolve in favour of the revision counter, because counting a form's first failure as failure #1 would drop it after one revision and contradict "at most twice". **Wrong, and withdrawn on an independent Fable pass.** "At most twice" is an upper bound, and an upper bound not reached is not violated; the same sentence carries a second trigger I skipped — "A failing **or vetoed** form's prompt is revised" — so *pass → veto → r1 → fail → r2 → fail → drop* uses both revisions and both failures consistently. The two-counter reading holds. Fable also noted the shape of the error: my reading rescues exactly ONE form, `howto` at 80.0%, since `argument` and `conversational` had unauthorised r2 transitions under any reading. **Kept because the failure mode — harmonising a rule in the direction of my own results, after seeing them — is the lesson.** Under the two-counter reading `howto` was terminal after its second failure and its r2 is an off-protocol measurement | — | — | — |
| W1 | **Is the terminal counter two GATE FAILURES or two REVISIONS, and does a second failure trigger a form-only bf16 re-smoke (§Data:428) or a drop (§Data:436, `m10/STATUS.md`:58)?** Three rules are registered for one situation | The numbers it governs are already observed (`howto` 80.0%, `argument` 88%), so choosing a reading now is "changing a protocol after a number it affects is observed" — Tier 3 | none: nothing generates before COV admission and the build seed draw |
| W2 | Do `howto` (80.0%, at the threshold) and `argument` (88% judged / 67% full output) stand as passes? **`conversational` is RESOLVED on evidence, not interpretation:** its r1 output survived, and re-judged against the frozen rubric — executing the registered measurement, not changing it — it scores **50/50, 100%**. Its r1 transition was triggered by a genuine gate failure (22% on the frozen rubric), so `conversational` has **one** authorised revision and a clean pass. Its earlier 50% was purely the moving-rubric artefact. **Recommend reverting the active prompt to r1 `d16f3212`** (one revision, authorised, 100%) rather than keeping r2 `be4fa0ff` (two revisions, void trigger, 96%); both pass, r1 is strictly cleaner | Their gate results are sound; their *procedure* was not, and I found that only after seeing them | none, as W1 |
| W5 | **ANSWERED 2026-09-05: the resolution distance is 0.0086, the MDE is BELOW it, and no available admission moves it.** `results/m10_cov_resolution.json`: family-weighted COV macro over the four admitted families, paired stratified bootstrap at the registered B = 200,000 / seed 0 / `inverted_cdf` / one-sided 0.025/13. Distance **0.008619**, paired SD 0.00302, implied z 2.93. The rule needs point > distance STRICTLY as well as >= MDE, so **a contrast landing at the MDE 0.0056 cannot resolve**; it lands at the bottom of §Surfaces' predicted 0.009–0.0135 band, which is what admitting LEDGER bought. **Where the width comes from, and why more data will not fix it:** BRIGHT carries 50.0% of the macro variance and legal 32.5% — 83% between them — against LEDGER's 2.7% for 10,000 of the 13,416 queries. Family-equal weighting is why: LEDGER helped by diluting the weights 1/3 -> 1/4, not by adding power, and any further admission does the same at a smaller margin. Disclosures on the record: the distance is measured between UNRELATED models, whose per-query differences are far less correlated than two same-init arms', so the mandate's expectation is that it over-estimates a real contrast's width — an expectation, not a guarantee; the artifact prices it as arithmetic (`distance_if_paired_sd_scaled`: 0.75x -> 0.0065, 0.5x -> 0.0043, so the paired SD must fall to ~0.65x before the MDE binds). §Screen's two namings of the quantile (`inverted_cdf` vs "the 384th order statistic") differ by one observation and by **1e-6** here; `inverted_cdf` runs, both are recorded. **Dylan's three options are unchanged and the registered default (accept, report unresolved contrasts as unresolved) stands unless he moves the MDE — which is Tier 3 and must happen before any arm.** Original entry: | **The screen's power — remedy RECOVERED, size to be measured.** The LEDGER refusal is withdrawn (§2): the surface is **four families and 13,416 queries** (was 3,416), LEDGER supplying 10,000 over 47,820 pages. §Surfaces already expected most B–G contrasts to be unresolved at MDE 0.0056. Options if the measured resolution number confirms it: accept and report the unresolved contrasts as unresolved (the registered default); admit a further surface; or revisit the MDE — **the last is Tier 3 and must be decided before any arm runs, not after** | Admitting a new surface or moving the MDE is Tier 3, and MDE-after-observation is forbidden outright | decide before family F starts |
| W4 | §Surfaces requires a COV fingerprint screen "against the six **and the reserved four**". Reserved-set DOCUMENT fingerprints do not exist and creating them opens the reserved corpora — the reasoning that ruled FineWeb out on 2026-09-01 (`m9/LEDGER.md` §1.3). `m10src/cov_screen.py` therefore screens against the six's documents and the full protected QUERY index (which already covers reserved queries), and the reserved DOCUMENT side is **not** screened | Building it would open a reserved surface outside a registered transaction — Tier 3 | none: the query-side screen runs, and results so far are 0 hits |
| ~~W6~~ | **RULED 2026-09-05 — see the section above.** Original question: **Four things to weigh, from the Fable pass.** (i) **The registered default is the worst measured option on finance** — "keep the bar, no store passes, revert to the incumbent" leaves finance at 0.315 when a store measured at 0.535 (z = 4.56) is available; a default is not neutral when it is the measured minimum. (ii) **The approved `finance` prompt was accepted at 86% on the top-tail seeds and its build-population behaviour has never been observed** — its prompt carries no subject steer, unlike `health`'s, which is consistent with its 5-of-50 entity-trivia leak even on the cleanest seeds. (iii) **Seed precision is a proxy one step removed from anything the build consumes:** the build consumes QUERIES, and the registered quality gates on queries are A8 diversity and on-form rate — so the right admission standard for a seed store may be the queries it yields, but re-registering that is yours, after the diagnostic exists. (iv) **Two "next levers" were registered in two places** and T2-8 picks a ladder. **Cost of waiting is ~zero:** nothing generates before the build seed draw at step 8. Original entry: It was registered for the MARGINAL of a keyword widening — "of the passages this widening newly admits, what share are on topic" — and a Fable pass transplanted it to a whole-population gate on a new store. Measured 2026-09-05 on a blinded, interleaved sample: the **incumbent store itself reads 0.590 (health) and 0.315 (finance)**, so the bar has never been met by anything in this project and "revert to the incumbent" is not a safe default. Three options, none taken here: keep the bar and accept that no store passes (the registered default, and it leaves finance on 0.315); re-register the bar as a RELATIVE test (a new store must beat the incumbent, which `wikipedia-body` does on finance by z = 4.56 and ties on health); or fix precision at the source with the subject filter and re-gate against 0.80 unchanged — the only option that needs no protocol change, and the one being executed | The bar is a registered constant and the numbers it governs are now observed, so re-reading it is exactly the change the protocol forbids me to make | **none if the subject filter clears 0.80.** If it does not, the build's seed precision is ~0.55/0.53 at best and Dylan chooses |
| W7 | **Do `factoid` and `product` revert to generation at ≈143K each, or stay dropped?** The mandate registers both as harvested forms (~250K each) AND registers "a harvested form that falls under 100K reverts to generation"; they yield 5,605 and 40,977. Reverting costs two new prompts, two smoke gates, two veto windows and **+286K over the registered 1.0M generation cap**; not reverting means the 1.25M is carried by three forms at 417K/417K/416K rather than five at ~250K. **Registered default: excluded**, and that is what `draw()` executes. Also for ratification: the three-form quota table itself is a reallocation of the mandate's five-form split. Measured `ask` quality is in §Harvest amendment 2026-09-05 item 2 — read it before admitting anything | Every option changes a registered quota number; quotas are Tier 3 | none for the harvest (the draw runs on the registered three), but it must be settled **before the M10.2 recipe lock** |
| W8 | **Can the screen earn its keep, and is "MDE 0.0056" the rule you want?** The registered `resolve_rule` requires the POINT estimate to reach the MDE, which caps power at 50% at a true effect of 0.0056 whatever the interval does; F and E need a true **0.0112** for ~80% power (16% at 0.0056), and **A's rule is `lower > MDE`, so A3−A2 — the contrast M10 exists to run — must read above 0.0142.** The screen costs **106–126M examples (43–51 h), 166–186M with confirmations**. Codex's recommendation, recorded not adopted: freeze the recipe on stated priors, spend the compute on the build and on seed replication, and report the recipe as prior-selected. **Three ways to go: (a) accept and report unresolved contrasts as unresolved — the registered default; (b) move the MDE or the rule; (c) cut families and spend on the build.** A 5M contrast cannot show a data advantage persists at 200M, so a causal coverage headline needs a matched near-full-dose control either way | The MDE and the resolve rule are pre-registered statistics; the screen's existence is the milestone's shape. Tier 3, and **before family F**, never after | **none for the data path** — the corpus is needed under every option and is being built now |
| W9 | **Eleven registry decision-logic defects; five BLOCK any arm.** `E-bs32` and `G-384` **cannot win their own contrasts** (one-sided rule + orientation: their win is a negative point, and `point >= MDE` refuses it), F's post-hoc orientation makes the familywise α **0.02885 not 0.025**, F2's comparator is adaptively selected with no selection-aware bootstrap, and the **L12 5M probe is schedule-confounded** — L6/bge read at 5M inside a 20M schedule are mid-cycle and un-annealed, which is the defect S1 created the standalone anchor to fix. Plus six specification gaps (multi-arm tie-breaks in G/B/D, the confirmation revert rule only in prose, `trained_arms_expected` 16 vs 15 when C is skipped, prose-DSL "machine-readable" fields, registry 16 vs mandate 15 arms, the "MDE" wording). Full table in §Codex review 2026-09-05 | Reorienting a contrast or reallocating α changes pre-registered statistics = Tier 3. **I have deliberately not edited the design lock**, since a partial repair would leave it inconsistent and the α item is yours regardless | **none for the data path**; blocks family F |
| W10 | **MEASURED, and the blind spot is hiding a real number.** `results/m10_a8_blindspot.json` (`m10src/a8_blindspot.py`) puts the registered gate beside a sensitive **unregistered** diagnostic — word-4-grams, near-dup at ≥ 50% of the smaller gram set — on real data. **On the 200-query smoke from the pinned generator: `health` reads 0.00% on the registered gate and 20.50% on the diagnostic**, against a 25% action threshold, because its (8,30) range puts most queries under the 23-word floor. The gate works where it *can* fire (`argument`, 120–220 words, reads 3.59%). **On the shipped harvested corpus the inert gate costs nothing observable** — `keyword` 0.05%, `title` 3.28%, `claim` 8.06% on the diagnostic — so real text is genuinely not collapsed and the exposure is the GENERATED half. Caveats that must ride with these numbers: the smoke is 200 queries from T2-3's top-tail seeds and is **not build-representative**, 20.50% is still under 25%, and the 4-gram rule is a diagnostic establishing direction, not a verdict. Original entry: **A8's diversity gate cannot fire for five of twelve forms, including the generated form `yesno`.** An N-word query has N−7 word-8-grams, so the registered **16/32** threshold is unreachable below **23 words**; `factoid`, `keyword`, `product`, `title` and `yesno` have their whole range below it, and `claim`/`comparison`/`finance`/`health` only reach it at their longest. Template collapse with one varying slot passes untouched. **Decision 14 accepted a 4-bit generator on the stated ground that this gate guards against 4-bit repetition** — for `yesno` it does not. Options: (a) accept and report the gate as inert for short forms, leaning on exact dedup and the registered mean-pairwise-cosine diagnostic; (b) scale the threshold with sketch size (e.g. ≥ half the available grams) so short forms are covered; (c) use a shorter n-gram for short forms; (d) promote the stella cosine diagnostic to the guard for forms the sketch cannot reach. **Nothing is changed here** — the implementation faithfully implements the registration | The threshold, the n-gram and the gate's ACTION (cut to representatives, drop below 50,000) are a registered quality gate. Tier 3 | **DOES NOT BLOCK GENERATION** — corrected: §Data:457 executes the A8 gates "on the immutable manifest **before any arm**", i.e. AFTER the corpus is assembled. So generation runs under the unchanged contract, both measures are taken on the real build-representative output (which removes the top-tail-smoke caveat), and the ruling is needed before the manifest is frozen and before any arm. If a rate crosses the cut, regenerating one form is ≈1.4 GPU-hours, not 10 |
| W11 | **HALF DONE.** Codex finding 2: two registered screens had never been applied to the shipped harvest corpus. **The FORMS-12 hold-out is now EXECUTED** as a post-pass (`corpus10.harvest_holdout`, no re-draw): **1,500 documents held → 1,614 hold-out rows** (claim 546 · keyword 546 · title 522, each ≥ the 500 FORMS-12 needs) and **1,248,386 training rows**, in `work/m10harvest/harvest_{train,forms12}.jsonl` + `harvest_holdout.json`. It holds out by **DOCUMENT across every form**, because one Wikipedia article yields a title, headings and a claim, and holding it out for `title` alone would train on that article's `claim`. **Still owed: the own-source word-5-gram screen.** Note it is close to vacuous for harvested text once the span is excluded — the string IS a span of its document by construction, so what remains is "does this string also occur elsewhere in its own document" — and it needs a pass over every source document. Cost/benefit is worth stating before spending it | the hold-out was execution, not a decision; the remaining screen is too | the hold-out is done, so the corpus is trainable; the residual screen is a disclosure if not run |
| W3 | **Seed supply is OPEN — the widening was tried and REJECTED by its own gate.** Full-store, `min_score ≥ 4`: `health` 10,399, `finance` 22,375, `howto` 37,927 against a ~32–33K need. Widening the keyword lists raised the raw counts (health 36,284) but the registered judged-precision gate reads **28% on-topic on health's marginal and 38% on finance's**, against ≥ 80% — the router selects on the presence of "medic\*"/"hospital"/"financial", not on subject, so the marginal is mostly biographies and organisations. Estimated usable: **health ~17.6K, finance ~22.8K — both still short.** `ROUTE_WIDE` is NOT adopted; `draw()` defaults back to T2-3's `ROUTE`. **Next lever (registered, not yet tried): a subject-level filter on lead-sentence patterns**, since `hotpotqa-corpus` is entity intros — reject "X (born …) was a …" and "X is a company/hospital/journal …". Same judged gate before adoption. If that fails too, the levers left are relaxing `min_score` (worse precision, so unlikely to help), raising queries-per-seed against the A8 gate, or Dylan lowering the `health`/`finance` quotas | Touches the registered data recipe; quotas are Tier 3 | decide before step 8 |

### `lr_at` off-by-one — the LAST step of every arm trains at PEAK LR, 2026-09-05. **Do not fix while the calibration is running.**

`nano10.lr_at` computes `per = total_steps // cycles` and `within = step % per`. For
`total_steps = 156_250`, `per = 52_083`, so three cycles cover **156,249** steps and the loop's
final step **156,249** has `within = 0` → **LR jumps back to peak 1e-4**, immediately after
annealing to 1e-5. §Recipe registers "3 cycles of equal example count, **each linear 1e-4 → 1e-5**",
so this contradicts the registration; it is a defect, not a choice. Verified:

| step | 156,246 | 156,247 | 156,248 | **156,249** |
|---|---|---|---|---|
| LR | 1.0003e-05 | 1.0002e-05 | 1.0000e-05 | **1.0000e-04** |

**What it does and does not touch** (`trainer10.train_arm` evaluates *after* `opt.step()`, and
`cycle_ends(156250,3) = [52082, 104165, 156248]`):

- **Cycle-end COV evals are CLEAN** — the last one is taken at step 156,248, before the peak-LR
  step. So **every screen verdict and the sign-stability clause are unaffected.**
- **The final checkpoint and any post-loop encode are NOT.** `ckpt_every = 15,625` and
  `156250 % 15625 = 0`, so a checkpoint *is* written after step 156,249. The build's exported model
  would therefore carry one full-peak-LR AdamW update on an annealed model.
- **The M10.0-e calibration's own COV is measured post-loop** (`calib.py` calls `cov_of(m)` after
  `train_arm` returns), so all three P arms carry the step. Identical treatment across the three,
  which is the same argument `calib.py` already makes for its fixed lambda, so **the paired widths
  and the seed effect stand**; the absolute macros are of a model one peak-LR step past annealing.

**Two neighbouring trainer semantics, verified while tracing this, that a screen-arm runner must
not get wrong.** (i) The mix window is exact — `100/0` → `QQQQ`, `75/25` → `QQQD`, `50/50` → `QQDD`,
q_share 1.0 / 0.75 / 0.50 — so families B and the anchor are what they claim. (ii) **`stopped =
"plateau at cycle 3"` on a 3-cycle screen arm does NOT mean the arm failed.** `PLATEAU_FROM_CYCLE`
is 3 and screen arms run 3 cycles, so the plateau rule can only ever fire at the FINAL cycle end —
the arm has completed its dose and stops one step short. Anything that reads a non-empty `stopped`
as "unusable arm" (as `calib_report.py` correctly does for the P arms, which pass no `eval_fn` and
so can never trip either rule) would wrongly discard a complete screen arm. Read `stopped` together
with the cycle index.

**Fix, and its timing constraint.** Clamp the last cycle to the end of training, e.g. carry the
remainder into the final cycle so `within` never wraps, or clamp `within` to `per - 1` on the last
cycle. **It must NOT be applied while the P arms are running** — P0 trained under the current
function, so changing it mid-flight would leave P1/P2 incomparable to P0 and destroy the only
thing the calibration exists to measure. Apply after M10.0-e completes and **before any registered
arm**, with a test asserting the final step's LR equals `final` for a `total_steps` not divisible
by `cycles`.

### Codex CODE review, 2026-09-05 — nine findings; A8's gate is inert for five of twelve forms

Log audited before reading: `frozen_eval` and `m9reserve` appear **only** in the brief's own
exclusion text (lines 20–21), no reserved read. Every finding below I reproduced myself.

**FIXED (mine, unambiguous).**

| # | defect | fix |
|---|---|---|
| 3 | **`arm_smoke` reported PASS after a registered warm start threw** — `passed` read only steps, `stopped` and the cap, so a random MLP head running 90 finite steps printed PASS and exited 0 | `warm_start_implemented` is now part of `passed`, and so of `all_shapes_pass` and the exit code. It also hardcoded `lam=1e-4` and never exercised `select_lambda`; it now selects |
| 4 | **PAQ accepted a truncated extraction** — `read_rows` never checked it found every requested index, and the 5% margin would absorb a short read into a quota-filling, positionally-biased sample. The tarball hash does not prove `SRC` is a complete extraction OF it | asserts the line count equals the pinned population AND that every requested index was found, before anything is written |
| 5 | **`warm_start_from_m9` accepted a missing backbone** — `strict=False` reported `missing_keys` and never rejected them, so M9's head on a FRESH pretrained backbone would be reported as an implemented M9 warm start | refuses an empty or partial backbone key set |
| 6b | **the A8 action read the ROUNDED rate** — 50001/200001 = 0.25000375 displays as 0.25 and escaped the `> 0.25` cut | `near_dup_rate_raw` drives the action; the rounded value is display only |
| 8 | **`copied_span` excluded gram VALUES globally**, so a query copied from another occurrence of the same five words passed | positional exclusion: only windows lying entirely inside an occurrence are dropped, so boundary windows still catch. **NOT fully closed and cannot be from here** — with the span occurring twice, both are excluded, and telling them apart needs the offset `harvest` does not record. Asserted as a known limitation by test |
| 9 | `build_form` reported `final` before the quota cut; its end-to-end test passed vacuously on an empty result | `final` describes what is returned; `before_quota_cut` added |

**FINDING 1, CRITICAL, NOT a code bug — the registered A8 gate is structurally inert for short
forms.** A query of N words has N−7 word-8-grams, so a bottom-32 sketch reaches the registered
**16/32** threshold only at **N ≥ 23**. Verified across the registered ranges:

| A8 gate can fire | forms |
|---|---|
| **never** (whole range < 23 words) | `factoid` (5,15) · `keyword` (2,4) · `product` (3,12) · `title` (6,16) · **`yesno` (6,20)** |
| partly (only the longest queries) | `claim` · `comparison` · `finance` · `health` (upper bounds 25–30) |
| always | `howto` (25,60) · `conversational` (30,80) · `argument` (120,220) |

Codex reproduced 40 heavily templated 22-word queries: `near_duplicates=0`, `representatives=40`.
Exact-identical strings are still removed by exact dedup, but **changing one slot defeats the
gate entirely**. This matters beyond bookkeeping: **decision 14 accepted a 4-bit generator on the
stated rationale that "the A8 diversity gate guards against 4-bit repetition", and for `yesno` —
a generated form — it cannot.** Registered-gate territory, so not changed here → **W10**.

**FINDING 2 — the SHIPPED harvest corpus has not had two registered screens applied.** §Data:
"Every harvested string goes through the same screens, quotas and **hold-out** as a generated
one", and screen (iii) is the own-source word-5-gram copy check with the harvested span excluded.
`harvest.draw` applies the protected-query and protected-document screens and **neither of those**.
It needs no re-draw: `harvest_drawn.jsonl` carries each row's `doc` id, so both are a post-pass
over the shipped file. **Owed before the corpus is used** → §Open questions, and it is mine to
execute, not to decide.

**FINDING 6a — "an EARLIER query" is ambiguous and the reading changes what gets cut.** Indexing
only representatives misses chains: with A~B and B~C but A!~C, B is dropped and C survives even
though C matches an earlier query. `near_dup_gate` now takes `against=`, **defaulting to the
literal `"earlier_query"`** — more faithful to the text and better at the gate's stated purpose —
with `"representative"` available and both covered by a discriminating test. Logged as a Tier-2
reading; if Dylan reads it the other way, one keyword flips it.

**WITHDRAWN — my own claim, again.** I wrote that G-MLP's warm start "cannot start worse than the
linear head". **False.** The second ridge solve minimises the UNNORMALISED residual, and lower raw
error does not imply a better normalised objective; Codex produced a counterexample worsening
0.82380 → 0.84793. The pooling algebra is exact and unaffected — G-MLP does start *at* the
anchor's fitted head plus a fitted correction — but the fairness rationale for contrast G3 is
weaker than I stated. The test asserted a favourable random case as if it were a theorem.

**Confirmed correct, and worth not re-deriving:** the rubric-range filter before dedup and before
`seen_n` keeps Algorithm R uniform over the eligible stream; sharing one deterministic RNG between
reservoir and shuffle is not a defect; **the shuffle genuinely fixes the truncation bias rather
than disguising it**; the MLP pooling identity holds exactly, `up.bias` included; `G2 − outer(mu,
mu)` is the correct token-weighted centred covariance with padding excluded; `features[:, :384]`
is the last layer.

### Codex adversarial review of the screen, 2026-09-05 — ELEVEN decision-logic defects, and one of my claims withdrawn

Brief `codex_out.txt` (scratchpad, high effort, read-only). Read-exclusion carried; **log audited,
no reserved read** — the only `untouched-*`/`m9reserve` strings are the brief's own exclusion text.
It read `m10/{screen_registry.json,LEDGER,PLANNING,STATUS,SMOKE,EXPLORED}`, `instructions-m10.md`,
`m9/{registry.json,FINAL_LOCK.md}`, `results/m10_cov_{resolution,teacher_ceiling}.json`,
`m10src/{calib_report,head_width_parity}.py`. **Every checkable claim below I re-verified against
the artifacts myself.**

**WITHDRAWN — my own claim, and it was wrong.** I put it to the reviewer that selecting on BRIGHT
is near-incoherent because the TEACHER scores only 0.2191 there. Rejected, and the arithmetic is
against me: 0.2191 nDCG@10 over tens of thousands of documents per slice is not near-random, and no
random baseline or qrel density was ever measured to support the word. The unrelated-model probe
shows BRIGHT **can** express model differences — `abs_family_delta` **BRIGHT 0.0547 vs legal
0.0116** at variance shares 0.4998 / 0.3254 — so **legal** is the family with 32.5% of the variance
for 12% of the signal. This is what §5 and `STATUS.md` already said; I re-derived a worse version of
a settled question. *"Selecting on BRIGHT is not incoherent, and point 5 overstates the case
substantially."* The narrow criticism that survives: BRIGHT's low ceiling may weaken the link
between distillation fidelity and qrel performance, and **that link has never been calibrated.**

**Factual correction to this file and `STATUS.md`:** the implied z is **2.8516**
(`results/m10_cov_resolution.json` `z_implied`), not the 2.93 recorded in §W5.

**The power arithmetic, sharper than "0.0056 < 0.008619".** `resolve_rule` is `point >= MDE AND
lower > 0 AND sign stable`, so requiring the POINT estimate to reach the MDE **caps power at 50% at
a true effect of exactly 0.0056, however narrow the interval** — "MDE" is the wrong word for it.
Consequences, all arithmetic on recorded numbers:

| contrast set | read against | minimum true effect for ~80% power | power at a true 0.0056 |
|---|---|---|---|
| F, E | 0.008619 (unrelated) | **0.0112** = 0.008619 + 0.842(0.003022) | **~16%** |
| B, D, G, C | the P-arm same-init pair | needs `distance_raw ≲ 0.0043`, not merely < 0.0056 | — |
| **A (the thesis)** | rule is `lower > MDE`, **not** `lower > 0` | needs an observed point **> 0.014219** | — |

So **A3−A2, the contrast M10 exists to run, must read above 0.0142 to resolve.** The calibration
now running must land near **`distance_raw` ≤ 0.0043 and `seed_effect` ≤ 0.002–0.003** for B/D/G/C
to work at 0.007-scale effects; it cannot rescue F/E/A, and one LR contrast on the M9 pool says
nothing about the variance of architecture, data, objective or init changes. **This file contradicts
itself** — it calls the calibration descriptive (§M10.0-e) and then treats `< 0.0056` as
"adequately powered" (§W5 area). The descriptive reading is the correct one.

**Eleven registry defects. Five are BLOCKING and the reviewer's recommendation is that no
registered arm starts until they are repaired.**

| # | defect | verified |
|---|---|---|
| 1 | **Family E has no implementable verdict.** E1 is `E-bs128 − E-bs32`; a bs32 win is NEGATIVE and can never satisfy `point >= MDE`. One field reverts an unresolved contrast to bs32, `E_cost` reverts an unresolved bs32 win to bs128 | ✅ orientation and `resolve_rule` read from the JSON |
| 2 | **`G-384` cannot win its own contrast.** G1 is `G-1152 − G-384`, same one-sided problem — only 1152 can ever win, yet the action says "resolved winner" | ✅ |
| 3 | **F's post-hoc orientation breaks the familywise α.** 11 ordinary tails + 4 F tails = 15 × 0.025/13 = **0.02885**, not 0.025. Needs 0.0125/13 per F tail or a max/pairwise procedure | arithmetic |
| 4 | **F2's comparator `F-winner` is adaptively selected** with no selection-aware bootstrap: holding the observed winner fixed ignores the selection step | ✅ alias is prose |
| 5 | **The L12 5M elimination probe is schedule-confounded.** L6/bge are READ at 5M inside a 20M three-cycle schedule (cycle ends ≈6.67M/13.33M/20M) so they are mid-cycle and un-annealed there, while a genuine 5M L12 arm completes three compressed annealed cycles — **the exact defect the standalone anchor was created to fix (S1)**. And the registry never says whether an extended L12 restarts on a 20M schedule or continues | ✅ doses/schedule |
| 6 | Multi-arm winner selection is undefined in G, B and D: two alternatives can both resolve against the default and nothing says whether the higher point wins, whether they must resolve against each other, or what a tie does. D never tests D-NORM against D-COV | ✅ |
| 7 | The confirmation cap's consequence — non-default winners beyond the two confirmed **revert to default** — exists only in prose, in a file that declares prose non-authoritative | ✅ |
| 8 | `trained_arms_expected: 16` is false on an allowed path: `C-M9init` is `trained: true` **and** `skipped_iff` F does not pick bge-small, so the count is 15 | ✅ |
| 9 | `F-winner`, `runs_iff`, `skipped_iff` and the L12 `iff` are free-text prose, not machine-readable, contradicting the file's claim to be the executable rule source | ✅ |
| 10 | Registry says 16 trained arms; `instructions-m10.md`:597 still says fifteen | ✅ |
| 11 | "MDE" is internally misleading — `point >= MDE` caps power at 50%, and A's `lower > MDE` is stronger still | ✅ |

**My cost premise was stale** (PLANNING §5 labels itself superseded). The current registry is
**106.25M–126.25M screen examples** before confirmations (F 45–60M, the rest 61.25–66.25M) ≈ **43–51
raw training hours** at 683 ex/s; with confirmations and the synthesized recipe/seed pair,
**166M–186M — almost another build.**

**The reviewer's recommendation, recorded not adopted:** drop the claim that the screen selected an
empirically superior recipe; freeze the recipe on stated engineering/literature priors, spend the
compute on the build and on training-seed replication, and report that the recipe was
prior-selected because the pilot surface lacked decision resolution. *"If the headline must be
causal — 'coverage caused the gain' — the cheap solution does not exist"*: a 5M contrast cannot show
that a data advantage persists at 200M. **Not mine to take** → W8, W9.

**Also recorded: a reweighting is no longer pre-registration, even though no arm has trained.**
*"'No registered arm has trained' is a weaker criterion than 'the numbers affected have not been
observed.'"* The resolution contrast and the teacher ceiling were both measured on this exact
surface, so any weighting change is now pilot-informed and must be labelled as such. Query-weighting
would put finance at 74.5% and make COV mostly a finance benchmark; it would cut the distance to
≈0.00497, and inverse-variance weights give ≈0.00496 with ~75% on finance — power bought by
abandoning the coverage estimand. **The registered equal-family weighting stands.**

## §4 Dev-reuse log

| date | surface | raw score reads | artifact |
|---|---|---|---|
| 2026-09-01 | cqadup-programmers, cqadup-physics (Mac diagnostics) | 43 + 43 | `results/m10_rank_probe_mac.json`, `results/m10_head_width_probe_mac.json` |
| 2026-09-04 | frozen comparator rows of `results/perquery.json` (bge-small, leaf-ir-asym, lr-dense-pertask, opensearch, bm25) on all-6 and clean-4 | comparator-only, no nano existed | amendment A3's clean-4 bars 0.5046 / 0.5233; not a dev-surface read |
| 2026-09-05 | **COV read #3 — the teacher ceiling.** stella scoring its own documents on the admitted surface | no candidate, no selection; the denominator retention is read against | `results/m10_cov_teacher_ceiling.json` |
| 2026-09-05 | **COV read #2 — the M10.0-e calibration arms** (P0/P1/P2) | not registered arms, no contrast verdict; COV only, no DEV-6 / FORMS-12 / CUREv1 | `work/m10calib/P*_cov.json` |
| 2026-09-05 | **COV read #1 — the resolution number** (§Surfaces). Two non-candidate probes on the admitted surface, 13,416 queries x 2 | direction discarded by construction; no candidate, no selection | `results/m10_cov_resolution.json` |

## §5 Amendments and withdrawn claims (never compressed away)

**BRIGHT re-weighting — considered 2026-09-05 after the teacher-ceiling read, and REJECTED. Not
raised to Dylan.** The teacher scores 0.219 on BRIGHT while BRIGHT carries 50% of the macro's
variance, and I took that as evidence the family was buying noise. **Both halves were wrong.**

- **It is Tier 3 and it is post-observation.** The motive is the *observed* resolution distance
  0.008619, not the ceiling — the same adaptation the Codex pass struck in amendment A4 ("an α that
  adapts to an observed width"), with the sign flipped. The "measured on the teacher, so untunable"
  defence fails for the reason this morning's did: the measurement is fixed, the RULE would be
  drafted knowing which family it hits.
- **The data says the opposite, and it was already in my own artifact.** Verified independently in
  `results/m10_cov_resolution.json`: the four weighted family deltas sum EXACTLY to
  `abs_macro_delta` (0.029752), so all four are **same-signed** — BRIGHT is directionally
  concordant, not a coin flip. Signal / variance share: **BRIGHT 46% / 50%** (SNR on par with
  consumer-health), consumer-health 24% / 15%, finance 20% / 3%, **legal 9.8% / 32.5%**. The
  uninformative family is **legal**, not BRIGHT. That I arrived at BRIGHT — where the *teacher*
  looked bad — and not at legal is the tell.
- **It would not even work.** Dropping BRIGHT moves the distance 0.0086 → 0.0081, still above the
  MDE, while losing 28% of the signal (SNR 9.8 → 7.5). It could only change which arm wins.
- **BRIGHT is the most on-thesis family:** M10 exists because M9 failed on diverse, long,
  non-factoid queries, and BRIGHT's long StackExchange posts are the closest COV comes to that.
  Removing it would bias selection toward the factoid families that resemble the release surface.

**Adopted instead, Tier 1, no rule touched:** every macro is reported with its per-family deltas
(`contrast()` already returns `per_unit_delta_raw`) and, once available, the P-arm per-family seed
noise. Descriptive rows, no action attached. **This entry was committed before any P-arm COV file
was opened**, because reading one makes every later surface change post-observation of student
results.

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

