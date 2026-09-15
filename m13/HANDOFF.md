# M13 live handoff — START HERE

Worktree: /home/dylan/asymetric-dual-encoders/work/m13cloud
Branch: m13-stage1-execution-prep. Parent main checkout/GPU belongs to M17.
Read this file, m13/RESUME.md, m13/RELEASE_DECISION.md, instructions-m13.md,
and instructions-m14.md. Do not assume M13 is closed until a closure record says so.

## Current state — see results/m13_live_status.json for the latest timestamp

AGENT: active, pursuing M13 close-out. NEXT AGENT: monitor or resume M13 if not
closed; only start M14 publication after the M13 closure/handoff exists.
TRAINING: STOPPED/FINISHED. All three Runpod pods EXITED at last live check.
FINAL NANO EVALUATION: RUNNING locally on CPU. Supervisor PID330084; child330139.
Authoritative live receipt: work/m13-final/execution.json; log: logs/m13-final-six.log.
Monitor: m13-monitor.service; health: work/m13-monitor/health.json.
The supervisor heartbeat is liveness, not proof that document encoding advances.
Inspect child CPU/log/dataset output times if progress appears stalled.

One-shot access m10-six-spent is already pushed. NEVER launch another nano scoring
run after that tag. If a process fails, preserve its state; only decisions-only
recovery from complete saved rows is available. The current process must finish.
Preflight clean,49tests and production synthetic rehearsal passed. SciFact and NFCorpus
rows persisted with bridge passed. No full verdict exists yet.

## Evidence and scope

Final checkpoint: m10/FREEZE.json. Actual dose199,999,721; all6,250,000steps complete.
279shortfall exactly reconciled from nine1-row document batches. Original supervisor
FAILED receipt is intentionally preserved. See results/m13_dose_reconciliation.json.
Freeze/ONNX/ORT/FastEmbed parity and final full backup verified.
Final COV0.528708958 /94.9669% teacher; DEV6 macro0.613048 /91.18% teacher.
COV is selection-informed. Final claims require the in-flight six-set decision.
User wants release unless gross overfitting or catastrophic failure; missing an
ambitious target alone does not block release. Do not alter statistical gates.

## Work remaining for M13

1. Finish and publish nano final six scores/decisions; report each family/dataset.
2. Implement and run the conditional reserved batch if triggered (production
   reserved encoder is currently a refusing stub; this is explicitly unfinished).
3. Finish M9 six-only close-out with its already-dated R3 bridge amendment,
   independent executor review, then calculate the registered descriptive paired row.
4. Measure zero/BGE-small/nano serving costs on one idle machine/protocol; do not
   contaminate latency measurements with the active CPU benchmark.
5. Evaluate gross-overfitting/catastrophic-failure concerns against actual evidence;
   document release recommendation, limitations, budget and artifact identities.
6. Reconcile all retained storage and unique backups before retiring idle disks.
7. Commit M13 closure and M14 handoff with exact paths and outstanding M14 tasks.

## M14 boundary

M14 owns Hub nano publication, model card, release packaging validation, and one
clean FastEmbed PR. Zero/stella already shipped; do not republish them. M13 must
provide the fixed artifact, quality decisions, costs and qualified release decision.

## Monitoring / resumption

Read execution.json and ps for BOTH supervisor and child before acting. Check
systemctl --user status m13-monitor.service. Final rows are results/m10_final_scores;
the executor commits final results at completion. Live git dirt in m10/LEDGER.md
and that scores directory is expected. Do not stash/reset/delete it. Do not
modify score13/access13 or their bound code while the run is active. Docs and new
independent modules can be prepared without changing the in-flight code.
No Runpod restart is currently needed. Credential paths are in RESUME, never
copy credential values into git/logs. User available until approximately03:03UTC,
then traveling; continue autonomously and leave explicit running/stopped status.

## Prepared next commands (not started)

M9: m13src/m9_closeout13.py, eight synthetic tests passed and two independent
reviews completed. Real frozen M9 adapter passed a synthetic one-query smoke.
M9 registry R3 bridge is dated and ratified_by_owner is now true.
The reviewed execution pin is m13/M9_CLOSEOUT.md; its metadata-only preflight
passed clean. Recheck preflight immediately before execution; do not run it
concurrently with nano. Do not modify bound shared code to accelerate nano.

Costs: `.venv/bin/python scripts/m13_serving_costs.py --measure` AFTER the CPU
benchmark ends and while no other heavy workload runs. BGE export preparation
passed stock-FastEmbed parity (min cosine0.99999994 on5synthetic inputs including
long truncation). Uses actual final nano ONNX, published zero NumPy path and BGE
ONNX. Three fresh processes/model, batch1,4threads; synthetic latency lengths.
No cost measurement started yet. No reserved implementation added before trigger.
User steering: do not over-engineer; reuse existing components and keep scope to
M13 closure and a useful M14 handoff.

Additional serving correctness check PASSED: results/m13_serving_validation.json.
Existing build13.fastembed_parity reused on a separate copy of the frozen export,
with the exact eight synthetic texts recorded in the result. No original artifact
modified. Max absolute difference1.112e-7, min cosine0.999999956.

Last agent checkpoint: 2026-09-15T00:15:24.946604+00:00.
M13 goal remains ACTIVE. Benchmark RUNNING on FiQA; no workload stopped.
Preparation complete for now; next dependent action waits for the existing final
benchmark. The agent may yield between checks; the detached supervisor and monitor
continue independently. Check live receipt/child PID rather than assuming chat activity.
