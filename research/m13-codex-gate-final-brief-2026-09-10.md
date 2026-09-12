# Final review: the LoTTE gate script after two fix passes — GO or NO-GO

Repository `/home/dylan/asymetric-dual-encoders`, branch `m13-stage1-execution-prep`, read-only.
Two independent reviews preceded you: the first (nine findings) and the second (ten findings, verdict
NO-GO). Every finding from both was accepted and fixed; the fixes are in the latest commits. You are
the P1 re-check. Confirm that each of the second review's ten findings is closed, look for defects the
fixes introduced, and end with exactly one line: `VERDICT: GO` or `VERDICT: NO-GO — <one sentence>`.
GO means no P1 remains and you would let this script perform the one-shot read as written.

The owner has said the code base must not grow further without cause. Do not propose additional
machinery for hypothetical adversaries; the standard (`m13/REVIEW_TRIAGE.md`, "Dropped, and why") is
a plausible accident by one researcher on one box, plus the protocol's one-read guarantee. A finding
that requires a deliberate attacker with write access to the repository and both machines is P3 at
most. Say so explicitly when you downgrade for that reason.

## Hard rules for this review

- **Never open, list, cat, grep, hash or otherwise read**: anything under `work/lotte/`,
  anything under `work/m9reserve/`, `results/frozen_eval/untouched-*`, `work/dev/cqadup-android.json`,
  `work/dev/cqadup-english.json`, or any reserved qrels cache. Reason about their FORMAT from the code.
- **No recursive searches**: no `grep -r`, `rg`, `find`, `ls -R`, `tree`, or globbing. Open only the
  files named below, by path, plus `git diff aa2bc47..HEAD -- <path>` / `git show` on those paths.
- Do not run the test suites or any script. In-memory Python one-liners for arithmetic are fine.
- Read `CLAUDE.md` first.

## Files in scope (open only these)

- `research/m13-codex-gate-rereview-2026-09-10.md` — the ten findings you are re-checking
- `m13/REVIEW_TRIAGE.md` — §Stage 2, both fix tables
- `m13/LOTTE_GATE_REGISTRATION.json`, `m10/LOTTE_LOCK.md`, `m13/RULINGS.md` (R16–R18)
- `m13src/lotte_gate13.py`, `m13src/test_lotte_gate13.py`
- `m13src/dev6_from_checkpoint.py`, `m13src/test_dev6_from_checkpoint.py`
- `m13src/build13.py` lines 117–320 (`check_gate`, `check_gate_manifest`, `REQUIRE_COMMITTED`) and lines 480–520 (`_preflight`)
- `m13src/test_build13.py` — `_gate`, `_manifest_path`, `_run_refusing`, `test_the_gate_must_be_bound_to_the_committed_manifest`, `test_the_decision_is_recomputed_from_the_recorded_bootstrap`
- `m10src/run_arm.py` — `git diff bc117c1..HEAD -- m10src/run_arm.py`
- `m13/EXECUTION.md` — the readiness table and the day-one runbook
- `m13/build_config.json` — `data_knobs_equal_to_anchor` (the registered recipe the gate compares against)
- Dependencies (behaviour only): `m8src/paths_guard.py`, `m8src/freeze_lotte.py`, `m13src/build_lock.py`,
  `m7src/evalkit.py`, `m7src/teacher.py`, `m10src/nano10.py`, `m10src/trainer10.py`

## What changed since the second review (verify each)

1. `acquire_lock`: a non-blocking exclusive `flock` on `work/lotte/gate13/lock`, taken after the guard
   claim and before the receipt, held (fd open) until `run` returns; `--recover` takes the same lock.
2. `registration()` requires `veto.bootstrap.quantile_method == "inverted_cdf"` exactly.
3. `registered_recipe()` reads `data_knobs_equal_to_anchor` from `m13/build_config.json`; `arm_record`
   compares student, n_layers (= feature_layers), head, objective and pattern (= mix); `RECIPE_FIELDS`
   are all required; the manifest carries the full recipe.
4. `dependency_identity()` hashes the repository name, the tokenizer JSON (`backend_tokenizer.to_str()`)
   and the backbone config JSON; recorded per role in the manifest at `--write-manifest`, re-derived in
   `load_student` and required equal. `nano10` itself is unchanged and still pins no revision (disclosed).
5. `committed_and_pushed(path, what, repo, require_pushed)`: tracked, `status --porcelain` clean,
   `log -1` commit, `branch -r --contains` non-empty. Applied to the manifest and the pin in the gate,
   and (shared function, imported) to the gate record and the manifest in `build13._preflight` under
   `REQUIRE_COMMITTED` (tests switch it off; production leaves it on).
6. `check_gate` recomputes `fires = delta_macro_raw < -margin and upper_q975_raw < -margin` with the
   margin from the registration and requires `decision` and `veto_fired` to agree; a `skipped` record
   may carry no bootstrap.
7. `_create_receipt` fsyncs the directory; `write_atomic` fsyncs the directory after `os.replace`.
8. Slice digests are journaled (`slices.jsonl`, fsynced) when a slice file is written; `_persisted_slice`
   requires a journaled digest, the pin's hashes and counts, the identity/key/roles, and finite rows in
   [0, 1] over exactly the slice's qids.
9. Runbook interpreter prefixes.
10. `check_gate_manifest` reports `gate["manifest_commit"]` and, under `REQUIRE_COMMITTED`, requires it
    to equal the manifest's live last commit.

## Questions

- Is the lock acquired before every LoTTE open on both the fresh and the recover path, and is the
  receipt still created before the first open? Is there any path where `run` returns without closing
  the lock fd, or where the lock is taken before the guard claim (the lock file is inside the tree)?
- `--write-manifest` calls `dependency_identity` for both arms; does the default path load anything
  it should not, and can it differ between the manifest machine and the read machine for a benign
  reason (e.g. a transformers version) such that the read refuses when it should not? Is that refusal
  the right outcome?
- Can `committed_and_pushed` be satisfied by a commit on a remote branch other than the one being
  worked on? Is that acceptable for provenance?
- Ordering in the runbook: DEV-6 fill (rewrites the E records) → `--write-manifest` → commit/push →
  pin → commit/push → `--preflight-only` → read → commit/push the record → build. Any step that
  invalidates an earlier binding?
- Does the fixture in `test_lotte_gate13.py` exercise the real `committed_and_pushed` (bare origin,
  push), the real lock, the real journal, and the real recompute in `check_gate`? Name any claim in
  the list above that no test covers.

## Output format

Numbered findings ordered by severity, each: **P1/P2/P3**, `file:line`, concrete failure scenario,
minimal fix. Then "Second review's findings" — one line per finding 1–10: CLOSED or NOT CLOSED with
the reason. Then "Claims I could not verify and why". Then every file you opened. Then the single
VERDICT line. Terse; no praise.
