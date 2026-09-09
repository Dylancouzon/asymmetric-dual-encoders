# M10 runs, in order

Every row points at a committed artifact. Numbers live in the JSON; this file is the index.
Nothing has trained. Dev reads below total **86 raw score reads** (43 per CQADupStack component) and enter the dev-reuse count at M10.0 (`m8src/dev_reuse_m8.py`); because of them the two components are DEV, not COV.

## M10.0 diagnostics (Mac, 2026-09-01; all `-diag`, read by no rule)

| run | artifact | one-line reading | dev reads |
|---|---|---|---|
| rank-bottleneck probe | `results/m10_rank_probe_mac.json` | the reconstruction-optimal 384-d subspace of stella's query space keeps 99.5% of one distribution and 90–93% of three; 98–100% at 640. The Mac reproduces the box ceiling on both CQA components to four decimals | cqadup-programmers, cqadup-physics: **40 raw score reads each** (1024-d head: 10 k-values + full + oracle; 768-d: 9 + full + oracle; 256-d: 3 + full + oracle; mixture bases 4 × 3) |
| head-width probe | `results/m10_head_width_probe_mac.json` | frozen bge-small + ridge head: 384 → 768 → 1152 features retain 27 → 33 → 37% (programmers), 36 → 41 → 44% (physics) of stella | same two components, **3 raw score reads each** |
| three-layer head serving parity | `results/m10_head_width_parity_mac.json` | fastembed 0.8.0 reproduces the pool-then-head output to 2e-7; zero custom ops; 34.5M parameters | none |

## M10.0-c family-F serving parity (box, CPU, 2026-09-05) — the gate that disqualifies an arm

| run | artifact | one-line reading | dev reads |
|---|---|---|---|
| family-F head parity | `results/m10_student_parity_box.json`, script `m10src/student_parity.py` | **all six PASS** — bge-small, MiniLM-L6, MiniLM-L12 at 3 and 4 feature layers — min-cos ≥ **0.99999988** through fastembed, zero custom ONNX ops, params 23.89M / 34.54M / 34.54M all under the 35M cap (the mandate's 33.4M for L12 was an estimate; L12 and bge-small are identical in size, which is why the serve-cost tie-break had to be registered). **A first run read 0.93–0.95 for both MiniLM students and would have disqualified them.** Diagnosed, not accepted: `fastembed.common.preprocessor_utils.load_tokenizer` serves `min(model_max_length, max_length)` from `tokenizer_config.json`, and `all-MiniLM-*-v2` ships max_length 128 beside model_max_length 512, so fastembed ran at 128 while the torch reference ran at 512. Every text under the limit was bit-exact (median cos 0.99999998). The export now writes the tokenizer we intend to ship | none |

## M10.0 rubric word-range filter (2026-09-05) — the largest on-form win available, and it was already registered

| run | artifact | one-line reading | dev reads |
|---|---|---|---|
| enforce the rubric's own word range | `results/m10_qfilter_effect.json`, `m10src/qfilter.py` | `wikipedia-body` health **0.780 → 0.857** on-form (drops 9.0% of output), finance **0.790 → 0.806** (drops 2.0%); incumbent 0.735 → 0.839 and 0.635 → 0.672. **Both candidate forms land above the 0.80 the forms were approved at, with no bar moved.** Out-of-range queries score **0.0–0.04** on-form, so the mechanical test and the judge agree almost perfectly and the filter removes waste, not borderline cases. Being under the word floor was the single largest on-form failure in the T2-7 diagnostic — larger than every topical failure combined. **Caveat first: a conditional re-read of verdicts that already existed, not a fresh measurement**; the confirming number is a fresh on-form read after the filter runs at build time | none |

## M10.0 teacher ceiling on COV (box, 2026-09-05) — the denominator every retention figure needs

| run | artifact | one-line reading | dev reads |
|---|---|---|---|
| stella's own COV macro | `results/m10_cov_teacher_ceiling.json`, script `m10src/cov_eval10.py` | **0.5567** family-weighted. Per family: legal **0.8845**, consumer-health **0.7507**, finance **0.3726**, **BRIGHT 0.2191** (slices 0.136–0.318). **The teacher is near the floor on BRIGHT**, which carries **50.0% of the macro's variance** for 25% of its weight — so a student's BRIGHT retention is a ratio of two small noisy numbers, and that, not a shortage of data, is where the screen's power goes. Raised for Dylan before family F; no surface change taken | none — COV, logged as COV read #3 |

## M10.0-d COV resolution number (box, 2026-09-05) — the surface's power disclosure

| run | artifact | one-line reading | dev reads |
|---|---|---|---|
| COV resolution number | `results/m10_cov_resolution.json`, scripts `m10src/cov_probe.py` (encode + score), `m10src/cov_macro.py` (the family-macro contrast rule), `m10src/cov_resolution.py` | **distance 0.008619**, paired SD 0.00302, implied z **2.8516**, on 13,416 queries across four families. The registered **MDE 0.0056 is BELOW it**, so a contrast landing at the MDE cannot resolve — §Surfaces' 0.009–0.0135 expectation, at the bottom of its band. Variance share: **BRIGHT 50.0%, legal 32.5%**, consumer-health 14.7%, **finance 2.7%** for 10,000 of the queries — family-equal weighting is why LEDGER bought dilution, not power. Direction discarded by construction (the contrast is oriented on the sign of the point estimate, so every recorded quantity is identical under either ordering) | none — COV, logged as COV read #1 in `m10/LEDGER.md` §4 |

## M10.0 rate benchmark (box, 2026-09-04; `-diag`, and one rule reads it)

| run | artifact | one-line reading | dev reads |
|---|---|---|---|
| trainer-shape throughput | `results/m10_rate_bench_box.json`, script `m10src/rate_bench.py` | the M10 recipe shape runs at 400 ex/s in M9's two-chunk collate, 718 / 596 examples/s in single query / document buckets (**683** blended at 75/25) and 2,311 / 748 at batch 128 (1,517 blended) — the mandate's earlier 745 / 1,331 were not in this file (Opus 2026-09-04), against the plan's imported 560 for a rented A100: the step is launch-bound at batch 32. Random token ids, fixed shapes, no data loading, no evaluation — it bounds the hardware, not the pipeline | none |
| registered bars, both partitions | `results/m10_bars.json`, script `scripts/clean4_bars.py` | amendment A3's four bars, recomputed from the frozen comparator rows: release 0.5042 avg-6 / **0.5046** clean-4; aim 0.5155 / **0.5233** | none — comparator rows only, no dev surface |
| conjunct arithmetic (2026-09-04b) | `results/m10_conjunct_arithmetic.json`, script `scripts/m10_conjunct_arithmetic.py` | uniform retention reaching each planning proxy (0.025 quantile, after the Codex pass): C1a **89.3%**, C2a 91.3%, C1b **91.6%**, C2b **94.9%**; to equal bge-small per dataset: scifact 91.4, nfcorpus 83.0, fiqa 72.9, arguana 94.7, scidocs 85.7, trec-covid 92.0%; at uniform 92% fiqa supplies 73% of the avg-6 margin; stress: one set at 65% clears nothing if it is trec-covid or scifact | none — comparator rows and the M7 ceiling only |
| per-token nonlinear head serving parity (2026-09-04b, box CPU) | `results/m10_head_mlp_parity_box.json`, script `m10src/head_mlp_parity.py` | fastembed 0.8.0 reproduces the per-token residual head `W_lin·x + W₂·GELU(W₁·x+b₁)` (W₁ 1152→192) to min-cos **0.99999989**, max-abs 1.1e-07; zero custom ops; **34.96M** parameters. The bottleneck form `1152→512→1024` also passed (34.48M) but caps output rank at 512 (Codex M3), so the residual form is arm G-MLP | none |

## M10.2 SCREEN VERDICTS (box, 2026-09-09) — the ten computable contrasts

Driver `m10src/contrasts.py` (it reproduces the pre-existing F1 record bit-for-bit, draws digest
included). Records `results/m10_contrast_<id>.json`; selection `results/m10_screen_verdicts.json`.
MDE 0.0056; one-sided lower bound at α/12, F1 two-sided at α/24 per tail. Family A resolves on
`lower > MDE` and is exempt from sign stability.

| id | a − b | point | lower | label |
|---|---|---:|---:|---|
| F1 | F-bge-small − F-MiniLM-L6 @20M | +0.011595 | +0.007221 | RESOLVED |
| **A4-A3** | **ANCHOR(A4) − A3** | **+0.012080** | **+0.006909** | **RESOLVED** → the 834,463 generated queries enter the build |
| A3-A2 | A3 − A2 | +0.010068 | +0.005363 | POSITIVE, NOT RESOLVED (0.00024 short of the bar) |
| G1 | ANCHOR(1152) − G-384 | +0.021651 | +0.015273 | RESOLVED — selects nothing (`G-384.selectable: false`); it is evidence on M9's 384-wide diagnosis |
| G2 | G-1536 − ANCHOR | +0.000335 | −0.002787 | NOT RESOLVED |
| G3 | G-MLP − ANCHOR | +0.002415 | −0.000307 | NOT RESOLVED |
| B1 | B-100/0 − ANCHOR | −0.001057 | −0.004704 | NOT RESOLVED |
| B2 | B-50/50 − ANCHOR | +0.002690 | −0.000537 | NOT RESOLVED |
| D1 | D-NORM − ANCHOR | −0.000202 | −0.001859 | NOT RESOLVED |
| D2 | D-COV − ANCHOR | −0.018391 | −0.023786 | NOT RESOLVED, **and confounded** — see below |
| E1 | E-bs32 − E-bs128 | — | — | NOT COMPUTED: `E-bs128` is CLOUD_ONLY and has not run |
| C1 | C-M9init − C-fresh | — | — | NOT COMPUTED: the arm is CUT (pre-observation, W8 band 1) |

**Selected recipe:** student **bge-small** · corpus **A4** (harvest + generated) · head **1152-wide
linear** · mix **75/25** · objective **squared L2** · batch **PENDING**.

**The A4−A3 gain is concentrated in one family, and that is the finding.** Decomposing the +0.012080:
MedicalQARetrieval alone contributes **+0.009885 (81.8%)**; LEDGER +0.001915; legal +0.001615;
**BRIGHT is net NEGATIVE at −0.001335**. Drop consumer-health and the three-family point is
**+0.0029, half the MDE**. A4 adds seven generated forms to A3's five, and two of them are
**`health`** and **`finance`** — exactly the two COV families that gained, in that order, while
BRIGHT has no matching added form and lost.

**What may be claimed, and no more.** A4−A3 is ONE BUNDLED INTERVENTION — seven forms added while
every existing form's exposure is diluted — and there is no per-form ablation anywhere in the
screen. It cannot separate form matching from targeted semantic coverage, from generated-data
quality, or from another correlated property of the added text. The defensible sentence: **the gain
is concentrated in consumer-health and is consistent with domain/form matching, while broad
cross-domain benefit is not established.** The build action is unchanged (the same 12-form
allocation at ~40× dose); the CLAIM is much narrower, and the risk cuts both ways for clean-4,
which is scientific/biomedical.

Three readings that are not optional:

- **A4−A3 is an exposure re-allocation, not a synthetic-data test.** The sampler is form-balanced,
  so A4's 12 forms take 312,500 presentations each against A3's 5 at 750,000, and 58.3% of A4's
  exposure is generated (`results/m10_exposure_table.json`). What resolved is *form breadth at the
  cost of per-form exposure*. The registered action follows either way.
- **D2's mechanism is UNRESOLVED.** D-COV's gradients are ~500× smaller than squared L2's on
  identical batches (0.0017 vs 0.868) at the same peak LR, and ε is not the cause. **An earlier
  version of this file said clipping was — that was asserted, not measured, and is retracted:** the
  clip instrumentation landed 14:20, after the last arm finished at 14:13:48, so no registered arm
  recorded a clip rate, and the audited 0.868 is at an init-adjacent loss implying ~0.43 at the
  first logged training loss, under the 1.0 threshold. Two candidate mechanisms stand, neither
  established: an optimizer-scale artifact by some other route, or a real failure of
  document-aware regression (Σ down-weights low-pooled-variance directions — exactly what each
  domain needs; D2 is negative in all four families). Report it as unresolved, not as a clean test
  and not as a proven artifact.
- **E is PENDING, not unresolved.** `rules.E_cost` selects bs128 "in every other case", and an
  unread contrast trivially fails to resolve — reading it that way would take E's decision from a
  measurement nobody made. The lock carries E open until the A100 runs both E arms, and
  **`rules.E_warmup_parity` was registered before that arm runs**: warmup was fixed in STEPS, so
  bs128 would have had 4× the warmup exposure and ¼ the updates, handicapping the arm whose
  selection saves 2.2× build cost.

**Every interval here is a query-resampling interval.** No interval in this screen contains
training-seed variance — confirmation was the only thing that would have measured it, and it is
cut. The one figure we have: the calibration recorded a seed effect of **0.0013869**
(`results/m10_calib_report.json`) — a single P0-vs-P1 difference, a sensitivity datum, **not a
bound and not a distribution**. A shift that size moves A4−A3's lower bound to ~+0.0055,
the MDE, and moves no other verdict. `confirmation.seed_variance_gap` carries the disclosure and
the cheap way to close it.

## Known pre-existing test failure (not caused by any M10 work)

`./run_tests.sh` reports `test_encoders` FAIL — **7 of 158 replayed caches**, all of them M8 `T1`
teacher-probe caches written for `Alibaba-NLP/gte-modernbert-base` and
`ibm-granite/granite-embedding-english-r2` with meta pooling `cls-l2`, which matches 0 current
`Spec` entries, so the replay refuses rather than guessing. Every other suite passes. Confirmed
pre-existing at 2026-09-04: `git diff --name-only main HEAD -- m7src m8src m9src` is empty, so no
M10 change touches the path. It blocks nothing in M10 (both encoders are closed avenues,
`m8/EXPLORED.md`) but it means `run_tests.sh` is not green, and a future session must not read that
red as its own breakage. Fixing it means either adding the two `cls-l2` Specs back or dropping the
stale caches.

## Deliberately not run

- The capacity probe (`m9src/capacity_probe.py`): optional and report-only since the 35M cap is hard (Dylan, 2026-09-01).
- Any synthetic generation yet: under decision 14 the smoke and the generation run on the box with Qwen's official AWQ release; the smoke is gated by the contract rate, an independent-model on-form read and Dylan's veto window (decision 15).

## M10 weekend window, steps 0a and 0b (box, 2026-09-04 evening)

| run | artifact | one-line reading | dev reads |
|---|---|---|---|
| **0b — pipeline rate on REAL corpora** | `results/m10_rate_bench_real_box.json`, script `m10src/rate_bench_real.py` | the M10 shape on the real M9 tokenized corpora with real memmapped teacher targets: length-bucketed + prefetched + `torch.compile` on fixed buckets reaches **914–960 ex/s** on the query corpora and **792 ex/s** on the 94.5-token document bucket, zero allocator retries, peak ≤ 3.9 GB. Blended 75/25 ≈ **890 ex/s → ~62 GPU-hours for the 200M build on the box**, against PLANNING §11's 683 ex/s random-token bound and M9's realized 226. Without `torch.compile`: 616–698 ex/s, ≈ 84 h. **M9's 226 is CONSISTENT WITH the `m9 path` document arm here (249 ex/s, 79% padding waste) but that does not establish it** — M9's steps mixed every scheduled source and used a different model (Codex 2026-09-04) | none |
| **0a — generation health assertion** | `results/m10_gen_health_box.json` | `Qwen/Qwen3-8B-AWQ` rev `4da05a8e…` on vLLM 0.28.0, 64 prompts (8 seeds × 8 forms, n=5) all in flight: contract **93.75%** (gate 90%) and **1,027–1,173 aggregate output tok/s** (gate 700). **PASS on both.** ≈1.0M queries ≈ 10 box-hours | none |
| **step 1 — generation smoke** | `m10/SMOKE.md`, `work/m10gen/smoke/*.json` | all seven generated forms at **100% contract**, 200 queries each, zero exact duplicates, after one registered `howto` prompt revision (LEDGER §1). On-form verdicts pending the independent judge | none |

**Serving fixes this box needed** (all in `work/m10gen/serve.sh`, pinned in `results/m10_gen_health_box.json`): `VLLM_WSL2_ENABLE_PIN_MEMORY=1` — WSL disables pinned memory and vLLM's buffer path then hard-fails on UVA; `VLLM_USE_FLASHINFER_SAMPLER=0` — flashinfer JIT-compiles sampling kernels and CUDA 12.6's nvcc rejects this box's gcc 15 (no older gcc installed); `--gpu-memory-utilization 0.88` — only 8.86 of 10 GiB is actually free; port 8001. **`--enforce-eager` costs 1.69×** (441 vs 745 cold tok/s) and must stay off: it disables CUDA graphs and all compilation.

**Seed-supply projection (step 8 input, not a result).** At `min_score ≥ 4` over `hotpotqa-corpus`, the full store projects to **howto 36.3K, finance 21.0K, health 8.8K** topical seeds against the ~28.6K a 143K-query form needs at 5 queries per seed. `health` and `finance` cannot be seeded at build scale from this corpus alone — a Tier-2 question before generation, not a footnote.
| **2 — COV admission screen** | `work/m10cov/screen.json`, `work/m10cov/bright_len_filtered.json`, scripts `m10src/cov_admit.py`, `m10src/cov_screen.py` | four components verified and fingerprint-screened against the six's documents and the protected query index: MedicalQA, CorporateLobbying and ConsumerContractsQA all **0 exact / 0 near** on both sides; BRIGHT queries 0/0 and its documents **0 exact / 23 near (0.008%)** once the 91,626 sub-8-word boilerplate documents are excluded — the raw 6,123 "exact" hits are that boilerplate (4,606 are the literal `".\n"`), not contamination. **Three family IDs admitted, the registered floor; no STOP.** `finance`/LEDGER still pending | none |
| **headroom H3 — seed supply** | `results/m10_seed_supply.json`, `m10src/seeds.supply()` | full-store scan, 5.35M documents, cross-store dedup, identical stores and `min_score = 4` in every row. **v1 of the widened routing is WITHDRAWN** (it wildcarded every keyword: `pain`→"paints", `capital`→"capital city"). **v2:** health 10,399 → **36,284** (3.5×, clears), howto 37,473 (clears), **finance 22,375 → 23,504 — still short**, since ~93% of v1's finance gain was precision loss. Router recall was the real constraint for `health`; `finance` still needs the relax-the-floor rung with its judged gate | none |
| **headroom H1 — LEDGER** | `m10/LEDGER.md` §2, `work/m10cov/ledger_structure.json` | **ADMITTED as the fourth family**, after a refusal that was **withdrawn on fact** — I read a stale 8-column builder listing and missed the `qrels` column. Verified: 10,000 queries, 494 reports → 47,820 pages, 116,912 graded qrels, and **all 116,912 qrel page ids resolve in the dataset's own page split**; under the 100K cap. Screen clean. The COV surface goes **3,416 → 13,416 queries**, recovering the one remedy §Surfaces names for the screen's power | none |

