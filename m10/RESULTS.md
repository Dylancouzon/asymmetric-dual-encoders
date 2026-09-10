# M10 result index

Box screen and descriptive reads complete, 2026-09-10. No cloud E, 200M build or final nano
quality result is implied. `m10/FINDINGS.md` carries interpretations and limitations.

## Screen

Source: `results/m10_screen_verdicts.json` and `results/m10_contrast_*.json`.
MDE 0.0056; the registered twelve-contrast adjustment remains unchanged.

| Contrast | Point delta | Lower bound | Result |
|---|---:|---:|---|
| F1 | +0.011595 | +0.007221 | RESOLVED |
| A3-A2 | +0.010068 | +0.005363 | POSITIVE, NOT RESOLVED |
| A4-A3 | +0.012080 | +0.006909 | RESOLVED |
| G1 | +0.021651 | +0.015273 | RESOLVED |
| G2 | +0.000335 | -0.002787 | NOT RESOLVED |
| G3 | +0.002415 | -0.000307 | NOT RESOLVED |
| B1 | -0.001057 | -0.004704 | NOT RESOLVED |
| B2 | +0.002690 | -0.000537 | NOT RESOLVED |
| E1 | — | — | Cloud E pending |
| C1 | — | — | Cut before computation |
| D1 | -0.000202 | -0.001859 | NOT RESOLVED |
| D2 | -0.018391 | -0.023786 | NOT RESOLVED |

Selected components and pending batch: `m10/M102_LOCK.md`. G1 is not a selection comparison;
unresolved results retain defaults without establishing equivalence. All intervals condition on
the trained checkpoints and exclude training-seed variation.

## Descriptive and preparation evidence

| Record | Artifact |
|---|---|
| Anchor seed sensitivity (+0.000712) | `results/m10_descriptive_seed_sensitivity.json` |
| A4−A3 at 20M (+0.016453; file stores A3−A4 orientation) | `results/m10_descriptive_corpus_at_20M.json` |
| Licensed corpus, counts, hashes, exclusions | `results/m10_data_manifest.json` |
| COV resolution and calibration | `results/m10_cov_resolution.json`, `results/m10_calib_report.json` |
| Teacher COV ceiling | `results/m10_cov_teacher_ceiling.json` |
| PCA/head-width probes | `results/m10_rank_probe_mac.json`, `results/m10_head_width_probe_mac.json` |
| Serving parity | `results/m10_head_width_parity_mac.json`, `results/m10_student_parity_box.json` |
| Real-data box rates (compile is not the registered build path) | `results/m10_rate_bench_real_box.json` |
| Generation, assembly and re-screening | `results/m10_gen_health_box.json`, `results/m10_assemble10.json`, `results/m10_rescreen10.json` |

The full diagnostic/run chronology through `46ea18d` is preserved at
`research/archive/m10-cleanup-2026-09-10/m10/RESULTS.md`. Its old “not generated yet” and test
failure notes describe their dates, not current status. Current verification: `PROJECT_STATUS.md`.
