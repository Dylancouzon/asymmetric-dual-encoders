# Codex (gpt-5.6-sol, high) second independent review of the LoTTE gate after the first fix pass — 2026-09-10, read-only

Brief: `research/m13-codex-gate-rereview-brief-2026-09-10.md`. Subject: `aa2bc47` on `m13-stage1-execution-prep`. Access audit of the Codex session log (`~/.codex/sessions/2026/09/10/rollout-2026-09-10T21-19-25-*.jsonl`): 47 commands, all on the named files; zero touches of protected paths, no recursive search. Verbatim output follows. Triage: `m13/REVIEW_TRIAGE.md` §Stage 2, second table.

1. **P1**, `m13src/lotte_gate13.py:780` — While the original process is running, a second process using `--recover` can load its receipt and score every unfinished slice concurrently. `O_EXCL` protects only initial creation. Minimal fix: hold a nonblocking exclusive `flock` for the entire attempt; recovery may proceed only after the original holder has exited.

2. **P1**, `m13src/lotte_gate13.py:215` — `registration()` accepts every listed NumPy quantile method, including `higher`; line 833 then uses it. An accidental `inverted_cdf`→`higher` edit passes and can reverse the veto at the registered boundary. Minimal fix: require equality to `"inverted_cdf"` and test another recognized method’s refusal.

3. **P1**, `m13src/lotte_gate13.py:264` — `student`, `n_layers`, and `head` need only exist; only dose and batch are compared with the registry. A wrong-recipe checkpoint labeled with the correct arm, seed, and registry SHA is frozen into a fresh manifest and accepted. Minimal fix: derive each E arm’s complete registered plan and compare every recipe field or its registered fingerprint.

4. **P1**, `m13src/lotte_gate13.py:612`; `m10src/nano10.py:87` — Checkpoint tensors come from authenticated bytes, but `Nano10` loads an unpinned Hugging Face tokenizer and model configuration by repository name. A changed tokenizer or shape-compatible configuration changes scoring without changing the recorded checkpoint SHA. Minimal fix: pin revisions and tokenizer/config hashes in the arm record and manifest, then load those exact local assets.

5. **P1**, `m13src/lotte_gate13.py:343`, `m13src/lotte_gate13.py:359` — The manifest check proves only a local commit, not that it was pushed; the pin receives no tracked, clean, committed, or pushed check at all. An uncommitted pin or locally committed-only manifest licenses the irreversible read. Minimal fix: apply common Git provenance checks to both artifacts and require the containing commit to be present at the branch’s upstream/tag before access.

6. **P1**, `m13src/build13.py:172` — The build trusts the gate record’s `decision` enum without recomputing it or requiring the record to be committed and clean. Changing `no_veto` to `veto`, while leaving all hashes intact, changes the $1,000 build and passes. Minimal fix: recompute the decision from the recorded bootstrap fields and registered constants, and require the gate record’s committed/pushed identity.

7. **P1**, `m13src/lotte_gate13.py:733` — The receipt file is fsynced, but its parent directory is not. A host/power crash after protected access can lose the directory entry and permit a fresh run. Minimal fix: fsync `gate13/` after creating and closing the receipt.

8. **P2**, `m13src/lotte_gate13.py:763` — Recovery authenticates only the slice’s outer identity, key, and role names. A valid-JSON edit to per-query scores, surface metadata, or cache provenance is silently aggregated and can change the decision. Minimal fix: validate qid sets, counts, finite score ranges and pinned surface fields, and durably bind each completed slice digest into the receipt journal.

9. **P2**, `m13/EXECUTION.md:57` — The runbook still invokes non-executable, shebang-free files directly: `m10src/run_arm.py --plan` and, at line 59, `m13src/build13.py ...`. Minimal fix: prefix both with `.venv/bin/python`.

10. **P3**, `m13src/build13.py:253` — `check_gate_manifest` reports the manifest’s embedded `git_head`, which is the HEAD before the manifest commit, instead of the gate’s actual `manifest_commit`. Minimal fix: retain, validate, and report `gate["manifest_commit"]`.

## Fixes verified

1. **NOT CLOSED** — Creation precedes protected opens and ordinary crash recovery is coherent, but concurrent `--recover` permits two readers; directory durability is also missing.
2. **NOT CLOSED** — Staged edits are detected and DEV-6 precedes manifest creation, but full recipe provenance and pushed state are not established.
3. **CLOSED** — Each scoring load hashes and deserializes the same checkpoint buffer; the earlier preflight read is not scored. Finding 4 above is a separate dependency-identity failure.
4. **NOT CLOSED** — The five hash calculations match `_hash_slice`, and duplicates refuse, but the pin need not be committed or clean and its three count fields are optional.
5. **CLOSED** — `verify=True` works on first encode and verifies properly manifested resumed shards; an orphan shard is safely refused as trust-on-first-use.
6. **CLOSED** — The missing-query refusal emits counts only; no other explicit refusal/print inspected emits a qid, pid, text, or per-query value.
7. **CLOSED** — `note_attempt` is called before `R.dev6`; under the accepted policy, retries are disclosed rather than prohibited.
8. **CLOSED** — `--dev6 defer` refuses every registry family except E.
9. **NOT CLOSED** — Pull, DEV-6-before-manifest, manifest, and pin ordering are present, but two direct Python invocations remain invalid.

## Claims I could not verify and why

- I did not inspect execution-time E records, checkpoints, manifest, pin, or protected slices; they were outside the allowed files or explicitly protected.
- Actual remote/push state cannot be inferred from the local Git checks, and no remote verification was authorized.
- Tests and scripts were not run, per the review rules.
- Stable ordering for score ties crossing the GPU `topk` cutoff remains unproven without exercising production-sized rankings.

## Files opened

- `CLAUDE.md`
- `research/m13-codex-gate-review-2026-09-10.md`
- `m13/REVIEW_TRIAGE.md`
- `m13/LOTTE_GATE_REGISTRATION.json`
- `m10/LOTTE_LOCK.md`
- `m13/RULINGS.md`
- `m8/LEDGER.md`
- `m13src/lotte_gate13.py`
- `m13src/test_lotte_gate13.py`
- `m13src/dev6_from_checkpoint.py`
- `m13src/test_dev6_from_checkpoint.py`
- `m13src/build13.py`
- `m13src/test_build13.py`
- `m10src/run_arm.py`
- `m10src/test_run_arm.py`
- `m13/EXECUTION.md`
- `m8src/paths_guard.py`
- `m8src/freeze_lotte.py`
- `m13src/build_lock.py`
- `m7src/evalkit.py`
- `m7src/teacher.py`
- `m10src/nano10.py`
- `m10src/trainer10.py`
- `m10src/contrasts.py`
- `m9src/final_stats.py`

VERDICT: NO-GO — Concurrent recovery can duplicate the protected read, and the decision remains vulnerable to unpinned method, recipe, dependency, and artifact identities.
