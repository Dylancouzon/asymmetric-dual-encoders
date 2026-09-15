# M9 six-only close-out — execution pin

The user authorized completing M13; its required M9 close-out follows the fixed
nano recipe. M9's R3 bridge amendment is already dated September10. Ratification
is recorded now with the reviewed concrete executor, before any M9 six-set access.
No statistical constant or original M9 freeze has changed.

Executor: m13src/m9_closeout13.py. It uses M9's own registry and two-contrast
statistics, plus the existing M13 dataset/scoring helpers. Eight synthetic tests
passed, including decisions-only recovery, post-tag failure, remote tag/BEGIN
binding and drift refusal. Two independent reviewers checked it; final findings
addressed. The real frozen adapter also encoded a synthetic query successfully.

M9 SHA256: 9d631b2c64244aff72db605e94cd973eeb4b80527f14d91fbeab5bc45fec6b52.
The existing parent checkpoint is symlinked at the original freeze-relative path.
Two-build-lock provenance remains disclosed. No reserved stage is reachable.
The shared tag helper's annotation wording says M10, but this entrypoint passes
and checks the distinct m9-six-spent tag and M9 ledger; the annotation wording
has no effect on identity or access. Shared sources remain unchanged during nano.

Execution must wait for the active nano transaction AND serving-cost measurement
to finish. Then use the same CPU/offline/four-thread environment as the nano run:
`.venv/bin/python m13src/m9_closeout13.py --preflight-only`, followed by the same
command without that flag only after a clean report and exact pushed HEAD.
Do not run two Git-writing scoring transactions concurrently. Do not rerun after
the M9 tag is pushed; only complete saved rows admit decisions-only recovery.
No M9 scoring has started at this pin. Results will be results/m9_final_run.json
and results/m9_final_scores/, committed by the executor on success.
