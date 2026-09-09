# M10 status — 2026-09-09. **SCREEN READ. Ten contrasts computed, the recipe is selected but one family is open.** Box idle. **NEXT ACTION: two decisions for Dylan (1a SYNTH-20M, 1b the seed arm), then push the recipe lock. E is PENDING until the A100 runs both E arms.** Both adversarial reviews have landed and every finding is applied; the next rotation slot is astra.

**Selected recipe:** student **bge-small** · corpus **A4** (harvest + the 834,463 generated queries) ·
head **1152-wide linear** · mix **75/25** · objective **squared L2** · batch **PENDING**.
Verdicts and the three non-optional readings: `m10/RESULTS.md` §M10.2 SCREEN VERDICTS;
selection `results/m10_screen_verdicts.json`; driver `m10src/contrasts.py`.

**Read this, then `m10/LEDGER.md`.** The box is **preparation for the cloud GPU run, not a
measurement target** (Dylan, 2026-09-05): run as much as it can here first, no re-shaping; the
remainder moves to the A100 under the same registry. **The weekend timeline is not binding.**
Working model (Dylan, 2026-09-05): the session is the ML lead, Opus/Sonnet subagents do the build
work, Codex is the adversarial reviewer.

## The screen, done (2026-09-09)

13 arms on the box, all `exit 0`, zero failures (`work/m10arms/rest_chain.log`, `F_chain.log`);
`E-bs128` is the 14th and is CLOUD_ONLY. The staged amendment was applied AFTER the chain read
COMPLETE and BEFORE any contrast was computed — `m10/AMENDMENT_STAGED_2026-09-08.md`, applied
verbatim, `screen_registry.json._amended_2026_09_09` carries the precise pre-observation claim
(no contrast computed; per-arm COV macros were visible, and that is stated rather than denied).

**The lesson the re-stamp taught:** the registry is frozen the moment a contrast is computed
against it, not the moment the arms finish. Every contrast record pins `registry_sha256`, so a
later registry edit — even a prose one — invalidates all of them and costs a full recompute.

Memory: RESOLVED and VALIDATED 2026-09-07 (`DocTargetView` + `release_arena()`); full-dose
pre-flight at `n_docs=5,000,000` passes at 8,228 MB peak arm RSS, min host free 3,297 MB. If an
arm ever crashes again, the peak is dominated by `m9src/data.row_texts` materializing a whole
store per chunk — LEDGER has the numbers and four withdrawn claims of mine.

## What is DONE and verified

| step | outcome |
|---|---|
| **0a/0b/1/2/3** | vLLM + generator smoke; all seven generated forms approved; four COV families admitted (13,416 queries); §0a design lock |
| **4 harvest** | **A3 corpus = 1,250,000 rows** from 21,087,043 harvested (wiki 16.06M · arXiv 4.98M · pool 46.6K). title/keyword/claim at 417K/417K/416K, no form short. FORMS-12 hold-out applied by DOCUMENT across all forms: **1,500 docs held → 1,614 eval rows, 1,248,386 train rows** (`work/m10harvest/harvest_{train,forms12}.jsonl`) |
| **5 PAQ** | **A2 4,037,000** (`8f32bcdf…`) + **build 1,000,000** nested inside it, from `dl.fbaipublicfiles.com`, CC BY-SA 3.0 shipped in the tarball. Protected screen removed only 0.35% |
| **6 trainer** | `nano10` · `data10` · `trainer10` · `qfilter` · `corpus10` · `arm_smoke` · `screen_lock` · **`corpus_loader` · `targets10` (built 2026-09-05 evening, Codex review pending)**. **190 tests green.** `assemble_arm` is the only launcher path. M9 pools re-screened against the M10 protected index (709 queries / 79,630 docs removed). Arm-shape smoke **12/12 on CUDA at max_len 512** |
| **M10.0-e** | **COMPLETE.** P0 0.477528 · P1 0.476141 · P2 0.473892. **Same-init distance 0.00288**, seed effect 0.00139 (n=1). `results/m10_calib_report.json` |
| **seeds** | resolved for **all seven** generated forms; health/finance 33,000 each from `wikipedia-body`, howto 37,927, the other four route `"general"` |

## OPEN — needs Dylan (none blocks the next step; W12/W13 ruled 2026-09-05, kept one line each)

| # | what |
|---|---|
| ~~W12~~ | **RULED: STOP removed.** Family A reports three labels and its DEV-6 CQADupStack + FORMS-12 reads; C1b is the failure condition (`instructions-m10.md` §Amendment 2026-09-05 C1) |
| ~~W13~~ | **RULED: L12 CUT.** F = bge-small vs MiniLM-L6 at 20M, 12 contrasts, 14 trained arms (§Amendment 2026-09-05 C2) |
| **paired row** | **DRAFT REGISTRATION READY for ratification: `research/m10-paired-row-registration-draft.md`** (8 numbered items, drafted from both reviewer passes, nothing new added). **Item 7 expires first** — both six-set transactions must emit aligned PER-QUERY scores and qids, and that must be in force BEFORE M9's close-out runs or the row cannot be built at all, ever. Original: **Register it — as a whole-protocol RECIPE delta, not a coverage test.** M9→M10 changes dose 4.5× (3.69B → ≈16.8B), head width 384→1152, schedule, objective, mix, batch, and possibly the student. **Normalise on the TEACHER**, not M9 (`Δr_d = (S10−S9)/T_d`); never call `S10/S9` "retention". Register now: the confound list, a fixed non-causal claim sentence, the exact datasets/statistic/B/seed, a conditional "same backbone family" label (F can pick a MiniLM), and **that both six-set transactions emit aligned PER-QUERY scores and qids** — otherwise the paired row cannot be built at all |
| **W14** | **No decision-bearing surface sees the headline forms.** COV selects on forum/medical/legal/finance; C1b is clean-4 (scientific, biomedical). The screen can optimise the build away from the release bar and nothing would notice until M10.4. Codex frames the root question: **is family A a causal experiment, a catastrophe veto, or diagnostics? It cannot be all three** |

## astra pass 2 — the three broken decision rules: **ALL FIXED 2026-09-09**

`research/m10-astra-v2-dispositions-2026-09-08.md`. The fixes landed as
`m10/AMENDMENT_STAGED_2026-09-08.md`, applied verbatim post-chain / pre-contrast; the corrected
rules are executable in `m10src/decision_rules.py` with tests, and the registry carries them.

| # | was | now |
|---|---|---|
| 1 | `confirmation.stands_iff` compared a paired margin to the range of **absolute** scores — rejecting a perfect replication, accepting a reversal on both fresh seeds | corrected to the paired rule, then **moot: the confirmation block is CUT** and replaced by SYNTH-20M |
| 2 | `E_cost`'s cost branch was unreachable — a cost-motivated bs128 pick has a negative quality margin and always reverted | E is **exempt from quality confirmation** |
| 3 | `D_tie` reverted to squared L2 when both alternatives resolved | **keeps a winner**, deterministic tie-break to `leaf_norm_e2`, tie reported |

Also from that pass, still live: the Bonferroni denominator is **12** (F1 used α/24, so no number
was ever wrong — the stale `/13` strings are annotated, not corrected numbers). The FORMS-12
plateau top-up is **removed**. `WARMUP_STEPS` is fixed in steps, so **E1 measures an optimizer
policy, not batch size** — which is a reason to read E's eventual result narrowly.

**The structural objection stands and is now partly answered:** the screen's isolated 5M effects
were assumed to transfer to a 200M combined build. **SYNTH-20M** measures the assembled recipe
against the anchor at 20M — descriptive, selecting nothing. It does not close W14.

## gpt-6-astra whole-plan review, 2026-09-08 — 3 verified gaps, all on the COST axis

Dispositions and verification: `research/m10-astra-review-dispositions-2026-09-08.md`; verbatim
`research/m10-astra-whole-plan-2026-09-08.log`. It did NOT rediscover W14 or the frozen-tower
debate — the brief withheld them, and it went elsewhere.

| # | finding | status | owner |
|---|---|---|---|
| 1 | **No small-model + BM25 row exists.** `zero`+BM25 = 0.4911 is compared against OpenSearch, LR-hybrid and dense-only baselines, never against **bge-small + BM25** — the obvious hybrid of the release bar. Not recoverable from `perquery.json` (it holds per-query nDCG *values*, not score lists) | **VERIFIED** in `FINAL_MATRIX.md` | **Dylan** — adding a comparator post-hoc, but it can only WEAKEN our claim, so conservative |
| 2 | **nano buys no compute saving.** 34,540,672 params vs bge-small's ~33.4M, same backbone forward pass. nano's claim is *quality at equal edge cost sharing stella's index*, NOT "near-zero query compute" — that is `zero`'s claim alone | **VERIFIED** from our own counts | session — a CLAIM fix owed to `CLAUDE.md`'s north star, M14 and nano's card |
| 3 | **The cost axis is unmeasured for the two points that matter.** `m9_edge_cost_Apple_M5_Pro.json` profiles cold start, peak RSS and p50/p95 for nano variants + mdbr-leaf-ir only; `zero` itself and conventional bge-small have no such profile under one harness | **VERIFIED** | session — **do it**, Mac-only, no labels, no GPU, no protocol impact |
| 4 | **Compile queries into the INDEX, not a student** — our 0.83M generated queries carry source doc IDs; index them as lexical aliases (doc2query-shaped). No student at all. Trades index expansion for query-model complexity | new avenue | **Dylan** to scope; needs the contamination rule re-applied to index content |
| 5 | **Spearman ≈ 0 may mean compressibility is TRAINABLE**, not that teachers are interchangeable — no independently-trained tower was trained to be compressible. Reframes M14's headline; connects to M8's unrun `E14-LORA` | framing | **Dylan** / M14 |

**Its #1 claim is wrong** — it assumed the index needs a query-time server. It does not: the edge
client holds the index. The salvageable half is to state whether the requirement is near-zero
**edge** or **total** compute, which finding 2 makes load-bearing.

**The through-line:** *"gate further training spend on finding an operational region where
asymmetry wins."* We have measured quality carefully and cost loosely.

## NEXT, in order — rewritten 2026-09-09

**DONE, removed:** the W8 band-1 chain (13/13, `exit 0`) · the staged amendment · the F verdict
re-issue · the ten computable contrasts · the selection.

| # | open item | blocks | who |
|---|---|---|---|
| 1 | ~~reviews~~ **BOTH LANDED.** Fable on the amendment + verdicts (11 findings, all applied); Codex on the FIXES, which returned **NO-GO** — 2 did not work, 4 partial, all now fixed, and it overturned three of my claims. `research/m10-{fable-verdicts,codex-fixes}-review-2026-09-09.md` | — | done |
| 1a | **DECISION: re-point or cut SYNTH-20M.** As registered it compares the anchor recipe **to itself** — the screen selected no non-default component, so its premise is gone. Candidates that measure something: A4 vs A3 at 20M, or the anchor at seeds 1/2. `confirmation.synth_20M._VACUOUS_AS_REGISTERED` | 20M examples of budget | **Dylan** |
| 1b | **DECISION: buy the seed number?** No interval in this screen contains training-seed variance; confirmation was the only thing that would have measured it and it is cut. ANCHOR at seed 1, 5M, ~1.5 h on the idle box, zero cloud cost, selects nothing. Six contrasts sit within ±0.004 of their bar. Bound we have: 0.0013869 from the calibration (n=1). `confirmation.seed_variance_gap` | nothing; it is a reporting quality buy | **Dylan** |
| 2 | **Push the M10.2 recipe lock** once the reviews land. E stays **PENDING** in it | the build | session |
| 3 | **Family E** — `E-bs128` is CLOUD_ONLY and both E arms run on the A100 together. Until then `rules.E_cost` has no reading: an unread contrast trivially fails to resolve, and taking bs128 from that would decide E on a measurement nobody made | the build's batch size | session, on the A100 |
| 4 | **CUREv1 admission** (decision 12, adopted 2026-09-04, **never executed**). Precondition: harvest/PAQ/seed draws were screened against an index lacking it — **re-screen or disclose before reading it** | nothing on the critical path; a reported diagnostic, never selection-bearing | session |
| 5 | **`results/m10_data_manifest.json`** — still MISSING (§1 owes it) | the recipe lock's provenance | session |
| 6 | **Own-source word-5-gram screen** on the harvest corpus (W11's second half): run it if a pass over the sources is cheap, else **disclose it as not run** in §1 | a disclosure in the report | session |
| 7 | **Register the paired M9-vs-M10 row** — draft ready, `research/m10-paired-row-registration-draft.md`. Its conditional item (§5, backbone family) self-resolved when F picked bge-small. **§7 expires first:** both six-set transactions must emit aligned per-query scores AND qids before M9's close-out runs, or the row is unbuildable forever | M9's close-out | **Dylan** |
| 8 | **W14** — is family A a causal experiment, a catastrophe veto, or diagnostics? *"It cannot be all three."* Sharpened by the result: **A4−A3 RESOLVED**, and the registry now says every family-A contrast is an exposure re-allocation, so what resolved is *form breadth at the cost of per-form exposure* | the recipe lock's interpretation, not its content | **Dylan** |
| 9 | **`torch.compile` for registered arms** (~1.7×). Currently SMOKE-ONLY | build wall-clock | **Dylan** |

## Screen design, settled

**W8 band 1** (P2 = 0.00288 ≤ 0.0056): run F · ANCHOR · A · G · B · E · D, **cut C** — `C-M9init`
at 5M starts from 3.69B tokens and wins its own contrast by construction. **L12 cut, A's STOP
removed** (2026-09-05). Both E arms run on the **A100**, together.

## Hazards a cold session will hit

- **`E-bs128` fails on this box** at max_len ≥ 256 (`CUDA driver error: device not ready`)
  on three morning runs, then PASSED at 512 on the evening re-run — intermittent, not deterministic. Passes at 128 (2,188 ex/s). **Accepted, not worked around** — do
  NOT add gradient accumulation. The 1,517 ex/s in the rate table is a random-token microbenchmark.
- **`arm_smoke` reaches ~8 GB RSS**; torch's CPU allocator does not release across 12 model loads.
  Run it alone. **Kill by PID** — `pkill -f arm_smoke` matches your own waiting shell.
- **Smokes overwrite real artifacts.** `calib.run_arm` writes the real `P0.json`; `harvest.draw`
  wrote the real `harvest_draw.json` until `out_dir` was added. Check `total_steps`/`_partial`.
- **The system python has no numpy.** Use `.venv/bin/python`.
- **`m10src/forms.RUBRIC` is the frozen gate standard; `forms.FORMS` is the revisable prompt.**
- Never overwrite `results/perquery.json`. No six/reserved/LoTTE read outside a registered
  transaction. Every review brief carries the reserved read-exclusion; **audit the log afterwards**.

## Do NOT redo

- Do not move the seed-precision bar (W6: there is no bar). Do not re-weight or drop BRIGHT
  (LEDGER §5 — **legal** is the uninformative family, 32.5% of variance for 12% of signal).
  Do not run T2-8's rungs 2/3 (demoted to diagnostics). Do not treat the on-form diagnostic as an
  admission instrument. There is **no post-generation admission test** — only A8's manifest gates
  and the FORMS-12 hold-out.

