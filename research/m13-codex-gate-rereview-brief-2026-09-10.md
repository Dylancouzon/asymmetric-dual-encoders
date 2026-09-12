# Second independent review: the LoTTE gate script after the fix pass — GO or NO-GO

You are the SECOND independent reviewer of `m13src/lotte_gate13.py` and its companions in the
repository at `/home/dylan/asymetric-dual-encoders`, branch `m13-stage1-execution-prep`, read-only.
The first reviewer (a different model) returned nine findings on 2026-09-10; every one was accepted
and fixed in the latest commit. Your job is twofold: (a) verify that each fix actually closes its
finding, and (b) break the fixed code — the receipt/recovery path, the manifest and pin bindings,
the bytes-once loading, the git checks and the refusal ordering are all NEW and unreviewed.

This code performs a ONE-SHOT protected read (LoTTE read #1) and its record decides which batch
size a $1,000 build trains. End your report with exactly one line: `VERDICT: GO` or
`VERDICT: NO-GO — <one sentence>`. GO means: no P1 remains and you would let this script perform
the read as written. Do not soften a NO-GO; do not manufacture a P1 to avoid saying GO.

## Hard rules for this review

- **Never open, list, cat, grep, hash or otherwise read**: anything under `work/lotte/`,
  anything under `work/m9reserve/`, `results/frozen_eval/untouched-*`, `work/dev/cqadup-android.json`,
  `work/dev/cqadup-english.json`, or any reserved qrels cache. These are protected evaluation
  surfaces. Reason about their FORMAT from the code that reads them.
- **No recursive searches**: no `grep -r`, `rg`, `find`, `ls -R`, `tree`, or globbing over the
  repository. Open only the files named below, by path, plus `git diff bc117c1..HEAD -- <path>` /
  `git show` restricted to those paths.
- Do not run the test suites or any script (read-only sandbox). In-memory Python one-liners for
  arithmetic are fine.
- Read `CLAUDE.md` first for the project's evidence and protocol rules.

## Files in scope (open only these)

The first review and its triage:
- `research/m13-codex-gate-review-2026-09-10.md` — the nine findings and the unverified claims
- `m13/REVIEW_TRIAGE.md` — §Stage 2, the fix table (top of the file)

Authority:
- `m13/LOTTE_GATE_REGISTRATION.json` (amended 2026-09-10: `veto.bootstrap.quantile_method`, `_amended_2026_09_10`)
- `m10/LOTTE_LOCK.md`; `m13/RULINGS.md` (R16, R17, R18); `m8/LEDGER.md` lines 1472–1495

The subject (fixed code):
- `m13src/lotte_gate13.py`, `m13src/test_lotte_gate13.py`
- `m13src/dev6_from_checkpoint.py`, `m13src/test_dev6_from_checkpoint.py`
- `m13src/build13.py` lines 117–265 (`_registered_e_checkpoints`, `check_gate`, `check_gate_manifest`) and lines 420–455 (`_preflight`)
- `m13src/test_build13.py` — `_gate`, `_manifest_path`, `test_the_gate_must_be_bound_to_the_committed_manifest`
- `m10src/run_arm.py` — `git diff bc117c1..HEAD -- m10src/run_arm.py`, plus `dev6()`, `dev6_deferred()`, the `--dev6` handling in `_run()`
- `m10src/test_run_arm.py` — the three `test_dev6_*` tests
- `m13/EXECUTION.md` — the readiness table and day-one runbook

Dependencies (read for behaviour, do not review):
- `m8src/paths_guard.py` (`ALLOWLIST`, `claim`, `check`, `install`); `m8src/freeze_lotte.py` (the pin's format and `_hash_slice`)
- `m13src/build_lock.py` (`verdicts`, `check_verdict_binding`, `PENDING`)
- `m7src/evalkit.py`; `m7src/teacher.py` (`encode_cached`, `_combined`, `PROVENANCE`, `ENC`); `m10src/nano10.py` (`Nano10`); `m10src/trainer10.py` (`save`)
- `m10src/contrasts.py` lines 90–112 and 270–285 only
- `m9src/final_stats.py` (`bootstrap`)

## The fixes, as the author describes them (verify each; break what you can)

1. Receipt + recovery: `_create_receipt` uses `O_CREAT|O_EXCL` under `work/lotte/gate13/` AFTER
   `claim_lotte()` and BEFORE the first slice is opened; `preflight` refuses when the receipt exists
   without `--recover` and when `--recover` is given without a receipt (existence is a stat, not an
   open); `_load_receipt` compares every `IDENTITY_FIELDS` entry; `_persisted_slice` accepts a slice
   file only under the identical identity and role set; recovered slices are not re-read.
   Questions: can a crash between `_create_receipt` and `_attempt`, or between a slice's scoring and
   its `write_atomic`, leave a state that `--recover` misreads? Can the receipt be created before the
   guard is installed on the real path? Is there any window where two processes both read?
2. Manifest: `write_manifest` refuses if the file exists; `manifest_of` requires the file tracked
   (`git ls-files --error-unmatch`), unmodified (`git status --porcelain -- path`), with a commit,
   and equal on branch, verdict sha, registry sha, candidate/comparator arm, sha256, record_sha256
   and checkpoint path. `arm_record` now checks `arm`, `seed`, `registry_sha256`.
   Questions: does "tracked and unmodified" prove "pushed"? Does a staged-but-uncommitted edit pass
   `status --porcelain`? Can the manifest be regenerated to match swapped records (the
   `--write-manifest` refusal is only on existence)? Is `record_sha256` of the arm record a stable
   binding given `dev6_from_checkpoint.py` REWRITES the E arm records (filling `dev6`) — in which
   ORDER does the runbook run the DEV-6 fill and `--write-manifest`, and does that order break the
   manifest check or the `check_gate` arm-record check?
3. Bytes-once: `checkpoint_bytes` reads and hashes; `load_student` deserialises from
   `io.BytesIO(data)`; preflight hashes the file once more (a second read) — is anything scored from
   a path rather than from the hashed bytes?
4. Pin: `pin_of` requires `results/m8_lotte_pin.json` to cover every registered slice with five
   shas; `read_slice` compares all five hashes and the three counts; duplicate qrel rows and
   duplicate positives refuse. Is `slice_hashes` byte-identical to `freeze_lotte._hash_slice`
   (list order, sorted qids, `{qid: sorted pids}` qrels object, `json.dumps(sort_keys=True)`)?
5. `encode_cached(..., verify=True)`: does `verify=True` on a FIRST encode behave (no shards yet)?
   On a resume with shards written by the crashed attempt?
6. The missing-query refusal prints a count. Is there any other print/refusal that can carry a
   qid, pid, text or per-query number outside `work/lotte`?
7. `dev6_from_checkpoint.note_attempt` writes an attempts line before `R.dev6`.
8. `run_arm._run` refuses `--dev6 defer` unless the arm's registry family is `E`.
9. Runbook: interpreter prefixes, `git pull` steps, manifest and pin steps.
R17: `registration()` refuses without `veto.bootstrap.quantile_method`; `paired_bootstrap` uses it.
Code identity: taken at preflight (into the receipt), re-taken before the record; a change refuses.

## Output format

Numbered findings ordered by severity, each: **P1/P2/P3**, `file:line`, the concrete failure
scenario, the minimal fix. Then "Fixes verified" — one line per finding 1–9 stating CLOSED or NOT
CLOSED with the reason. Then "Claims I could not verify and why". Then every file you opened. Then
the single VERDICT line. Be terse and specific; no praise.
