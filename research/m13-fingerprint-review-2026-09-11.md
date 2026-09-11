# Restart fingerprint repair review — 2026-09-11

Two independent agents returned GO for the narrow repair and a fresh bounded stage-0 retry.
`fingerprint_review` checked the runner diff, loader manifest construction, build delegation and
seven synthetic fingerprint tests. `retry_review` checked the diff, manifest construction,
document pool metadata and both cloud supervisors; seven fingerprint tests passed independently.
Neither found additional volatile manifest fields or a blocker. Access audits reported no
credentials, cloud calls, protected content, recursive searches or edits. The second reviewer
also inspected the archived initial receipt and run guidance; the first reviewed archive/refusal
references in the cloud scripts. Parent validation: 151 runner/build tests passed.

The repair excludes only query.generated_at and query.sources[*].seconds without mutating the
original manifest. Every semantic field and code identity remains checked. The build delegates
to the corrected runner fingerprint. Old checkpoints remain invalid under the changed code.

Retry conditions: preserve local receipt/logs/evidence and remote result JSONs, audit directory
and both E smoke directories; commit/push, deploy exact HEAD; rerun the existing stage-0 controller.
It rechecks uploaded bytes and CPU tests, runs both 512-token shapes and real interrupted/resumed
600-step smokes, verifies local backups and stops the Pod. No registered training or protected
evaluation is authorized by this diagnostic.
