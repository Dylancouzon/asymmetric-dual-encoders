# Nano recipe handoff — M10 to M13

Scope updated 2026-09-10; original half-A lock preserved in
`research/archive/m10-cleanup-2026-09-10/m10/M102_LOCK.md`. This concise handoff changes no
selected component or registered decision. **Batch and execution readiness are still pending.**

| Axis | Selected | Authority |
|---|---|---|
| Student | bge-small, 34,540,672 parameters with head | F1 / screen verdicts |
| Corpus | A4: M9 pool, PAQ-build, harvest, 834,463 generated queries | A4−A3 / data manifest |
| Head | 1152-wide linear, layers 12/8/4 | G2/G3 defaults; G1 descriptive for selection |
| Mix / objective | 75/25 query/document; squared L2 | B/D registered defaults |
| Warm start / seed | Ridge head, 60,000 fit rows, fit seed 21; training seed 0 | screen registry |
| Batch | Pending E1; both E-bs32 and E-bs128 on the same cloud GPU | `rules.E_cost` |

Screen authority: `m10/screen_registry.json`. Outputs: `results/m10_screen_verdicts.json`.
Data authority: `results/m10_data_manifest.json`. The screen chose no non-default component
among the computed selection contrasts; E remains uncomputed.

## Inherited build contract

200M examples, three cycles, linear LR 1e-4 → 1e-5, 64,000 warmup examples in cycle 1;
AdamW β=(0.9,0.999), eps=1e-8, wd=0.01 on dim>1, clip=1.0, length-bucketed batching.
The old “three × 66.7M” is rounded: pin integer cycle boundaries against the 200M total in the
build configuration. Do not silently train 200.1M. Registered `torch.compile` remains smoke-only.

Reuse the registered kill/plateau rules. Extensions are whole cycles, require improvement ≥0.003
over the best previous annealed cycle, and stop at the budget-derived cap. M13 must implement
and test the build controller; the screen runner alone does not provide it.

## Before training

1. Implement/smoke the build path and restart behavior before renting idle compute.
2. Run both E arms on the cloud GPU and apply the existing E1 rule. Do not interpret missing E1
   as failure to resolve. Warmup is matched in examples; update counts still differ.
3. Resolve M13's **pre-build LoTTE** handling. A bs128 selection can be vetoed in favor of bs32;
   “half B deferred” does not authorize skipping this step.
4. Fix the billed-price allocation and extension cap under $1,000. Include cloud E, build,
   encodes, LoTTE, conditional reserved batch, disk and egress. The original table omitted the
   reserved allowance and misstated the extension range; use the formula, not that range.
5. Commit the executable build configuration and required clean reviews, then start the build.

Final-run constants remain in `m10/final_run_registry.json`, **not yet locked for execution**.
No six/reserved/LoTTE read or spent tag is authorized by this handoff. `m13/EXECUTION.md` owns the gate.
