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
| trainer-shape throughput | `results/m10_rate_bench_box.json`, script `m10src/rate_bench.py` | the M10 recipe shape runs at 400 ex/s in M9's two-chunk collate, **745** in one padded chunk and 1,331 at batch 128, against the plan's imported 560 for a rented A100: the step is launch-bound at batch 32. Random token ids, fixed shapes, no data loading, no evaluation — it bounds the hardware, not the pipeline | none |
| registered bars, both partitions | `results/m10_bars.json`, script `scripts/clean4_bars.py` | amendment A3's four bars, recomputed from the frozen comparator rows: release 0.5042 avg-6 / **0.5046** clean-4; aim 0.5155 / **0.5233** | none — comparator rows only, no dev surface |
| conjunct arithmetic (2026-09-04b) | `results/m10_conjunct_arithmetic.json`, script `scripts/m10_conjunct_arithmetic.py` | uniform retention reaching each pass point: C1a **89.5%**, C2a 91.5%, C1b **91.9%**, C2b **95.3%**; to equal bge-small per dataset: scifact 91.4, nfcorpus 83.0, fiqa 72.9, arguana 94.7, scidocs 85.7, trec-covid 92.0%; at uniform 92% fiqa supplies 73% of the avg-6 margin | none — comparator rows and the M7 ceiling only |
| per-token MLP head serving parity (2026-09-04b, box CPU) | `results/m10_head_mlp_parity_box.json`, script `m10src/head_mlp_parity.py` | fastembed 0.8.0 reproduces `1152→512→GELU→1024` applied per token to min-cos **0.99999988**, max-abs 1.0e-07; zero custom ops; **34.48M** parameters. The "nonlinear head has no serving path" closure held only for post-pooling heads; this is arm G-MLP | none |

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
- Any synthetic generation: the approving per-form smoke runs on the GPU instance with the pinned bf16 Qwen3-8B, read by a person, before anything scales; a Mac mlx-lm 4-bit pass only develops the prompts.
