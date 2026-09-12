# Codex (gpt-6-astra, xhigh) final P1 re-check of the LoTTE gate after the second fix pass — 2026-09-10, read-only

Brief: `research/m13-codex-gate-final-brief-2026-09-10.md`. Subject: `5adcc13`. Access audit: 39 commands on the named files, zero protected-path touches, no recursive search. Verbatim output follows.

1. **P1**, `m13src/lotte_gate13.py:871` — After a crash, accidentally deleting or incompletely restoring a completed slice file leaves its journal entry intact. Recovery returns `None` before consulting that entry, then reopens and scores the completed slice at line 935. Minimal fix: refuse a missing file when its key is already journaled.

2. **P1**, `m13src/lotte_gate13.py:796`, `:821` — First-run setup creates `gate13`, but fsyncs only that directory, not its entry in the containing directory. A power failure can therefore lose the directory and receipt, admitting a fresh read. Minimal fix: fsync `gate_work_dir.parent` after directory creation, while holding the lock and before payload access.

3. **P2**, `m13src/lotte_gate13.py:670`, `:680` — Dependency checking hashes separately loaded tokenizer/config objects; `Nano10` subsequently loads both again by unpinned repository name (`m10src/nano10.py:87`). An upstream update between those calls can change the actual scoring dependencies without refusal. Minimal fix: authenticate the objects actually used by the student. Downgraded from the original P1 because persistent dependency drift is now caught; the remaining exposure is the check/load window.

Second review's findings

1. **CLOSED** — Both paths claim the guard before acquiring the lock. Receipt creation/loading and payload access follow acquisition; successful acquisitions close in `finally`. The lock-file open itself necessarily precedes `flock` and the receipt.
2. **CLOSED** — Exact `inverted_cdf` equality; recognized alternatives are tested for refusal.
3. **CLOSED** — Required recipe fields are compared with the registered configuration, dose/batch and seed; the manifest carries them.
4. **NOT CLOSED** — Remaining P2 dependency check/load gap described above.
5. **CLOSED** — Manifest and pin receive the shared provenance checks. Any remote-tracking branch containing the commit qualifies; that is acceptable artifact provenance, regardless of the working branch. It is not live remote verification.
6. **CLOSED** — Production build preflight requires committed/pushed artifacts; the decision and flag are checked against the recomputed rule. Skipped records reject bootstrap content.
7. **NOT CLOSED** — Receipt and atomic-write directory fsyncs exist, but newly created directory ancestry remains unprotected.
8. **CLOSED** for the original edited-file scenario — Journal digests cover scores, surface metadata and cache provenance. Missing completed files expose the separate P1 above.
9. **CLOSED** — Both interpreter prefixes are fixed. DEV-6 precedes manifest creation; subsequent pin and gate-record commits do not invalidate the manifest binding.
10. **CLOSED** — Reports `gate["manifest_commit"]` and compares it with the live manifest last commit under production enforcement.

Claims I could not verify and why

- Tests and scripts were not run. Execution artifacts, protected data, installed dependency versions, filesystem crash behavior and actual remote state were not inspected.
- The fixture uses real Git commits/pushes to a bare origin, real journal operations, real `flock` contention and real `check_gate` recomputation. Its contention test uses separately opened descriptors in one process, rather than two running gate processes.
- Uncovered claims include default Hugging Face dependency hashing/loading (#4), fsync durability (#7), runbook commands (#9), and live manifest-commit equality (#10). Production build provenance enforcement (#5/#6) is bypassed in the inspected tests. Neither P1 scenario above is tested.
- Claim #8 overstates explicit validation: recovery checks `n_queries`, but not `n_docs` or `n_qrels_pairs`; qids are compared across score maps, not hashed against the pin’s query-id digest. The journal authenticates those originally validated bytes.
- The manifest dependency helper requests the student tokenizer/config, not model weights or LoTTE content. Benign library-version or serialization differences can cause refusal; refusing before scoring is appropriate. Actual cross-machine compatibility remains untested.

Files opened

- `CLAUDE.md`
- `research/m13-codex-gate-rereview-2026-09-10.md`
- `m13/REVIEW_TRIAGE.md`
- `m13/LOTTE_GATE_REGISTRATION.json`
- `m10/LOTTE_LOCK.md`
- `m13/RULINGS.md`
- `m13src/lotte_gate13.py`
- `m13src/test_lotte_gate13.py`
- `m13src/dev6_from_checkpoint.py`
- `m13src/test_dev6_from_checkpoint.py`
- `m13src/build13.py` — specified excerpts and diff
- `m13src/test_build13.py` — targeted excerpts
- `m10src/run_arm.py` — specified diff only
- `m13/EXECUTION.md`
- `m13/build_config.json`
- `m8src/paths_guard.py`
- `m8src/freeze_lotte.py`
- `m13src/build_lock.py`
- `m7src/evalkit.py`
- `m7src/teacher.py`
- `m10src/nano10.py`
- `m10src/trainer10.py`

VERDICT: NO-GO — Recovery can reread a completed slice after accidental file loss, and receipt durability does not cover the newly created gate directory’s parent entry.
