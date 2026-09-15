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
card. Latest user instruction (2026-09-15 UTC) explicitly includes Hugging Face
upload in the next session's M13 closure: publish nano under the existing owner
namespace after these checks, verify the Hub commit/hashes and downloaded inference,
and record the exact public revision. No additional publication approval is needed.
This supersedes the earlier M14-only publication boundary. The clean upstream
FastEmbed PR remains M14 work.
Keep the 199,999,721 actual dose and retained exact-dose failure disclosed.
Retire unnecessary paid storage after verifying unique artifact backups and
reconciling the existing STOP-only constraint. No new training is required.
