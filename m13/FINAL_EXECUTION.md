# Final benchmark continuation — 2026-09-14

Candidate: m10/FREEZE.json; final-cycle SHA256 3e49e0bfaa633abe276da7847d0ac44cbfb3d7d5215b9343f255706eeb5789a1.
Actual dose 199,999,721 examples across all 6,250,000 scheduled updates. Nine
one-row document batches account exactly for the 279-example deficit. The
original exact-dose lock and FAILED supervisor receipt remain unchanged.
results/m13_dose_reconciliation.json records the disclosed evaluation deviation.
No retraining, checkpoint selection, or extension is authorized by this continuation.

R6 authorizes the sole registry change: ratified_by_owner true at this freeze/executor
pin. The actual reviewed executor is m13src/score13.py + access13.py, whose
code_identity binds its dependencies. This document supersedes stale implementation
status prose, not the statistical rules, in m10/final_run_registry.json.

Production synthetic rehearsal passed all six and the conditional reserved fixture.
Final execution is CPU-only on this isolated worktree, CUDA hidden, four CPU threads,
M7_ENCODER=stella-400M-v5. Frozen document caches are read through symlinks to the
shared parent caches. Anchor document/query re-encoding is required by the bridge.
No paid GPU restart is needed. Preflight must pass after this commit is pushed.
The durable m10-six-spent tag precedes protected reads; no post-tag scoring retry.

The conditional reserved encoder remains unimplemented in production. If triggered,
the executor persists and pushes six-set decisions and reports INCOMPLETE_RESERVED;
that status must remain explicit until the separately prepared reserved suite runs.
M9 close-out and the descriptive paired row follow the fixed nano comparison and
require their own bridge amendment and implementation; they do not gate this nano run.
