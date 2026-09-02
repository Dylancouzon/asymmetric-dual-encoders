# M10 runs, in order

Every row points at a committed artifact. Numbers live in the JSON; this file is the index.
Nothing has trained. Dev reads below must enter the dev-reuse count at M10.0 (`m8src/dev_reuse_m8.py`).

## M10.0 diagnostics (Mac, 2026-09-01; all `-diag`, read by no rule)

| run | artifact | one-line reading | dev reads |
|---|---|---|---|
| rank-bottleneck probe | `results/m10_rank_probe_mac.json` | the reconstruction-optimal 384-d subspace of stella's query space keeps 99.5% of one distribution and 90–93% of three; 98–100% at 640. The Mac reproduces the box ceiling on both CQA components to four decimals | cqadup-programmers, cqadup-physics: ~40 scoring passes each (three heads × k-grid + oracle + mixture bases) |
| head-width probe | `results/m10_head_width_probe_mac.json` | frozen bge-small + ridge head: 384 → 768 → 1152 features retain 27 → 33 → 37% (programmers), 36 → 41 → 44% (physics) of stella | same two components, 3 passes each |
| three-layer head serving parity | `results/m10_head_width_parity_mac.json` | fastembed 0.8.0 reproduces the pool-then-head output to 2e-7; zero custom ops; 34.5M parameters | none |

## Deliberately not run

- The capacity probe (`m9src/capacity_probe.py`): optional and report-only since the 35M cap is hard (Dylan, 2026-09-01).
- Any synthetic generation: the per-form smoke is the first generation step and it runs on a GPU (or the Mac with mlx-lm, Qwen3-8B 4-bit), read by a person, before anything scales.
