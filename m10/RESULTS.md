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

## Deliberately not run

- The capacity probe (`m9src/capacity_probe.py`): optional and report-only since the 35M cap is hard (Dylan, 2026-09-01).
- Any synthetic generation: the approving per-form smoke runs on the GPU instance with the pinned bf16 Qwen3-8B, read by a person, before anything scales; a Mac mlx-lm 4-bit pass only develops the prompts.
