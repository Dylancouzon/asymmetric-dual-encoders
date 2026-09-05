# M10 runs, in order

Every row points at a committed artifact. Numbers live in the JSON; this file is the index.
Nothing has trained. Dev reads below total **86 raw score reads** (43 per CQADupStack component) and enter the dev-reuse count at M10.0 (`m8src/dev_reuse_m8.py`); because of them the two components are DEV, not COV.

## M10.0 diagnostics (Mac, 2026-09-01; all `-diag`, read by no rule)

| run | artifact | one-line reading | dev reads |
|---|---|---|---|
| rank-bottleneck probe | `results/m10_rank_probe_mac.json` | the reconstruction-optimal 384-d subspace of stella's query space keeps 99.5% of one distribution and 90–93% of three; 98–100% at 640. The Mac reproduces the box ceiling on both CQA components to four decimals | cqadup-programmers, cqadup-physics: **40 raw score reads each** (1024-d head: 10 k-values + full + oracle; 768-d: 9 + full + oracle; 256-d: 3 + full + oracle; mixture bases 4 × 3) |
| head-width probe | `results/m10_head_width_probe_mac.json` | frozen bge-small + ridge head: 384 → 768 → 1152 features retain 27 → 33 → 37% (programmers), 36 → 41 → 44% (physics) of stella | same two components, **3 raw score reads each** |
| three-layer head serving parity | `results/m10_head_width_parity_mac.json` | fastembed 0.8.0 reproduces the pool-then-head output to 2e-7; zero custom ops; 34.5M parameters | none |

## M10.0 rate benchmark (box, 2026-09-04; `-diag`, and one rule reads it)

| run | artifact | one-line reading | dev reads |
|---|---|---|---|
| trainer-shape throughput | `results/m10_rate_bench_box.json`, script `m10src/rate_bench.py` | the M10 recipe shape runs at 400 ex/s in M9's two-chunk collate, 718 / 596 examples/s in single query / document buckets (**683** blended at 75/25) and 2,311 / 748 at batch 128 (1,517 blended) — the mandate's earlier 745 / 1,331 were not in this file (Opus 2026-09-04), against the plan's imported 560 for a rented A100: the step is launch-bound at batch 32. Random token ids, fixed shapes, no data loading, no evaluation — it bounds the hardware, not the pipeline | none |
| registered bars, both partitions | `results/m10_bars.json`, script `scripts/clean4_bars.py` | amendment A3's four bars, recomputed from the frozen comparator rows: release 0.5042 avg-6 / **0.5046** clean-4; aim 0.5155 / **0.5233** | none — comparator rows only, no dev surface |
| conjunct arithmetic (2026-09-04b) | `results/m10_conjunct_arithmetic.json`, script `scripts/m10_conjunct_arithmetic.py` | uniform retention reaching each planning proxy (0.025 quantile, after the Codex pass): C1a **89.3%**, C2a 91.3%, C1b **91.6%**, C2b **94.9%**; to equal bge-small per dataset: scifact 91.4, nfcorpus 83.0, fiqa 72.9, arguana 94.7, scidocs 85.7, trec-covid 92.0%; at uniform 92% fiqa supplies 73% of the avg-6 margin; stress: one set at 65% clears nothing if it is trec-covid or scifact | none — comparator rows and the M7 ceiling only |
| per-token nonlinear head serving parity (2026-09-04b, box CPU) | `results/m10_head_mlp_parity_box.json`, script `m10src/head_mlp_parity.py` | fastembed 0.8.0 reproduces the per-token residual head `W_lin·x + W₂·GELU(W₁·x+b₁)` (W₁ 1152→192) to min-cos **0.99999989**, max-abs 1.1e-07; zero custom ops; **34.96M** parameters. The bottleneck form `1152→512→1024` also passed (34.48M) but caps output rank at 512 (Codex M3), so the residual form is arm G-MLP | none |

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

