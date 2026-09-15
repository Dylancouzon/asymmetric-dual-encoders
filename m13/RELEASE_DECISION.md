# Release decision — 2026-09-14

Updated user instruction: release unless evidence shows gross overfitting or a
catastrophic failure. Missing a BGE-small/LEAF superiority target alone does not
block release. Statistical goals govern supported claims, not publication.
Do not declare absence of overfitting from COV alone; assess per-dataset failures,
DEV-6, final-suite transfer, serving correctness and any triggered reserved rows.
No new numerical threshold is retroactively presented as preregistered.
Do not change the in-flight final registry, fixed-sequence tests, thresholds,
checkpoint, or historical results in response to this instruction.

After final scoring: publish all results and caveats; complete the registered
conditional reserved evaluation if triggered; finish M9 close-out and the paired
descriptive row; measure same-machine zero/BGE-small/nano serving costs; prepare
and verify the nano checkpoint/ONNX/tokenizer bundle, usage example and model
card inputs for M14. M13 closes with a documented release recommendation and
verified artifacts; actual Hub publication and the FastEmbed PR are M14
deliverables, for the subsequent agent per the latest user instruction.
Keep the 199,999,721 actual dose and retained exact-dose failure disclosed.
Retire unnecessary paid storage after verifying unique artifact backups and
reconciling the existing STOP-only constraint. No new training is required.
