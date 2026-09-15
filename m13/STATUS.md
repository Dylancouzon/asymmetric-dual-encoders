# M13 status — closed

M13 completed its build, frozen six-dataset evaluation, M9 closeout, descriptive paired
comparison, serving measurements, and the tested execution package for M14. No Hugging Face
publication occurred. Resolve the immutable terminal integration with
`git rev-parse refs/tags/m13-closed^{commit}` after fetching tags.

Owner ruling R19 moved the triggered A100 reserved-four execution to M14. M13 therefore closes
honestly with `results/m10_final_run.json.end_status = INCOMPLETE_RESERVED`; this denotes deferred
registered work, not a failed or missing six-set result. M14 must execute that unchanged batch and
the separate broad descriptive BEIR-18 validation before public release.

## Terminal benchmark results

All values are exact-search nDCG@10 from persisted per-query rows. ArguAna and FiQA have disclosed
Stella training/evaluation contact; the clean-four partition excludes them.

| dataset | Nano | M9 | Nano − M9 |
|---|---:|---:|---:|
| SciFact | 0.721097 | 0.634021 | +0.087076 |
| NFCorpus | 0.363080 | 0.313345 | +0.049735 |
| FiQA† | 0.477765 | 0.214259 | +0.263507 |
| ArguAna† | 0.623296 | 0.542813 | +0.080484 |
| SCIDOCS | 0.217710 | 0.142694 | +0.075017 |
| TREC-COVID | 0.787116 | 0.382908 | +0.404208 |

Nano's registered results:

- versus BGE-small: clean four +0.017648 (one-sided lower 2.5% bound +0.003674,
  sign-flip p=0.006410) and all six +0.027449 (lower +0.017271, p=0.000010);
- versus LEAF asym: all six +0.016181 (lower +0.006504, p=0.000540); clean four did not
  establish superiority (-0.001063, lower -0.014456, p=0.559474);
- TREC-COVID is the main per-dataset limitation against LEAF: Nano − LEAF = -0.042982.

M9 ended `COMPLETE_SIX_ONLY`; neither registered superiority contrast passed and its decision is
`measurement; no claim`. The separately registered descriptive Nano-minus-M9 comparison is
+0.154009 on clean four (95% interval [0.132215, 0.175887]) and +0.160004 on all six
([0.144713, 0.175498]). Different recipes, doses, and training histories prevent causal
attribution.

## Build, serving, and recommendation

The frozen Nano checkpoint saw 199,999,721 examples, 279 (0.0001395%) below the nominal 200M.
The discrepancy is exactly reconciled; the original failed supervisor receipt remains preserved.
The final checkpoint, ONNX export, reference/ORT/FastEmbed parity, backup inventory, and serving
measurements are durable. On the same four-thread CPU protocol, median warm latency was 0.1119 ms
for constella-zero, 6.8400 ms for BGE-small, and 7.2511 ms for Nano; these are synthetic latency
measurements, not workload-distribution estimates.

M13 recommends proceeding to M14 release preparation: the evidence shows neither gross
overfitting nor catastrophic failure, and missing one ambitious clean-four LEAF target is not a
release veto under owner policy. Publication remains conditional on M14 completing the triggered
reserved four, the broad descriptive BEIR-18 run, final packaging/parity checks, private upload,
downloaded-byte verification, and the public transition.

## Durable evidence and verification

- Nano final: `results/m10_final_run.json`, SHA-256
  `f5b5ad8a63060fbe1184aa3e5319259855d3ca4a9e3c3de1e91eeb76ac685b23`.
- M9 final: `results/m9_final_run.json`, SHA-256
  `770344d7a7c18d0933e0f922af385fb0b4e514c98822e955e959cf139a9c936b`.
- Paired comparison: `results/m13_paired_m9_nano.json`, SHA-256
  `4eaf20ebef5548338314fd547abd0f76c55eae8b5334e3be2f11260d2f8071b4`.
- Frozen checkpoint: `work/m13-final/cycle3.pt`, SHA-256
  `3e49e0bfaa633abe276da7847d0ac44cbfb3d7d5215b9343f255706eeb5789a1`.
- ONNX model: the verified backup path in `m14/HANDOFF.md`, SHA-256
  `9ba0acf57b71dc31bc5512c5445078a797fa51cf3e85587d6b8a506bfc55dbc2`.
- Serving costs: `results/m13_serving_costs.json`, SHA-256
  `a1e8eabe412460c9e0cc95f34846179a4d0c885787b6e4da6a49353f852b578f`.
- Final verification: 267 M13 tests passed; all 36 M8 path-guard checks passed; compilation and
  focused reserved/controller tests passed.

All retained pods were `EXITED` at the last live provider check. They and their volumes remain
STOP-only and must not be terminated without explicit owner authorization. Exact next steps,
artifact restoration, cost bounds, and publication checks are in `m14/HANDOFF.md`.
