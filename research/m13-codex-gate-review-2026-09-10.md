# Codex (gpt-6-astra, xhigh) review of the LoTTE gate script and the deferred DEV-6 read — 2026-09-10, read-only

Brief: `research/m13-codex-gate-brief-2026-09-10.md` (names files, forbids recursive searches, carries the reserved read-exclusion). Subject: commits `f957bd4` and `bc117c1` on `m13-stage1-execution-prep`. Access audit of the Codex session log (`~/.codex/sessions/2026/09/10/rollout-2026-09-10T20-46-25-*.jsonl`): 33 commands, all `cat`/`nl`/`sed`/`git show` on the named files; zero touches of `work/lotte`, `work/m9reserve`, `untouched-*`, the reserved cqadup files, and no recursive search. Verbatim output follows. Triage: `m13/REVIEW_TRIAGE.md` §Stage 2.

1. **P1 — [m13src/lotte_gate13.py:525](/home/dylan/asymetric-dual-encoders/m13src/lotte_gate13.py:525): read #1 is repeatable.** Crash after a slice’s scores print but before `LOTTE_GATE.json` exists → rerun reads and scores every slice again. `attempts.jsonl` is counted, never checked for permission or matching identities; changing checkpoints or branch between attempts also passes. Concurrent processes can both pass preflight. **Fix:** create an exclusive, durable spent receipt before protected access, binding the committed inputs; refuse subsequent reads, including incomplete attempts. Keep recovery limited to persisted outputs.

2. **P1 — [m13src/lotte_gate13.py:201](/home/dylan/asymetric-dual-encoders/m13src/lotte_gate13.py:201): mutable arm records substitute for the committed checkpoint manifest.** Replace an E record and checkpoint with a previous run having the same dose/batch → preflight passes while E1 still describes the original run. Neither `arm`, seed, registry binding nor the full registered recipe is checked. A complete same-dose bs32 record from another arm can also masquerade as `E-bs32`. `check_gate` checks those same mutable records, so accepts the result. **Fix:** validate arm/recipe/seed provenance and require the pushed checkpoint manifest mandated by `LOTTE_LOCK.md:99`. The runbook currently omits filling that manifest.

3. **P1 — [m13src/lotte_gate13.py:404](/home/dylan/asymetric-dual-encoders/m13src/lotte_gate13.py:404), [m13src/dev6_from_checkpoint.py:117](/home/dylan/asymetric-dual-encoders/m13src/dev6_from_checkpoint.py:117): checkpoint verification and loading consume different opens.** Replace `cycle3.pt` after verification but before `torch.load`—model initialization occurs between them—with another shape-compatible checkpoint → it scores the replacement while recording the old SHA. The gate record still passes `check_gate`. **Fix:** hash and deserialize the same immutable byte snapshot, completing reconstruction before protected access.

4. **P1 — [m13src/lotte_gate13.py:357](/home/dylan/asymetric-dual-encoders/m13src/lotte_gate13.py:357): slice hashes authenticate nothing.** Change query/document text or exchange positive pids while preserving counts and set consistency → the altered slice is scored and labelled registered. The five hashes are computed but never compared with `freeze_lotte`’s existing pins; `remedy_record_sha256` is unused. Duplicate qrel rows are silently overwritten, and duplicate positives inflate counts before scoring deduplicates them. **Fix:** bind the existing pin manifest before access; compare all five hashes before scoring and reject duplicate qrel rows/pairs.

5. **P1 — [m13src/lotte_gate13.py:380](/home/dylan/asymetric-dual-encoders/m13src/lotte_gate13.py:380): resumed document caches are not verified.** A previous attempt leaves a cache; its `combined.f16` changes without changing size → `encode_cached(..., verify=False)` accepts it and reports the manifest’s old `combined_sha256`. Rankings use different bytes from those recorded, and `check_gate` accepts them. Existing shards have the equivalent problem. **Fix:** pass `verify=True` and reject unauthenticated caches; retain hashes verified against the consumed bytes.

6. **P1 — [m13src/lotte_gate13.py:437](/home/dylan/asymetric-dual-encoders/m13src/lotte_gate13.py:437): the missing-query refusal leaks qids.** If the evaluator omits expected queries, the refusal includes up to five actual `q:<qid>` identifiers in stderr, outside `work/lotte`. **Fix:** emit only the missing count publicly; keep identifier diagnostics inside the protected tree.

7. **P2 — [m13src/dev6_from_checkpoint.py:144](/home/dylan/asymetric-dual-encoders/m13src/dev6_from_checkpoint.py:144): deferred DEV-6 is not once-only across failures.** Crash during evaluation or after it returns but before records are rewritten → both copies remain deferred and the next invocation repeats the read. Concurrent invocations also pass. **Fix:** persist an exclusive checkpoint-bound read receipt before `R.dev6`; distinguish failed/spent from unread.

8. **P2 — [m10src/run_arm.py:842](/home/dylan/asymetric-dual-encoders/m10src/run_arm.py:842): deferral can invalidate existing bindings.** `defer` accepts family-F arms too. Compute their F verdict while deferred, then fill DEV-6 → the whole-record hashes used at `contrasts.py:278` change. The blanket “invalidates no binding” claim is false. **Fix:** restrict this workflow to the two E arms, or explicitly reissue affected bindings after filling.

9. **P2 — [m13/EXECUTION.md:63](/home/dylan/asymetric-dual-encoders/m13/EXECUTION.md:63): the machine handoff and commands are incomplete.** Copying only `work/m10arms/...` leaves the box without the updated published records required by `load_records`; committing E1 only on the box leaves the instance’s verdict stale. Also, the new scripts are mode `100644`, without shebangs: direct commands in steps 5–6 cannot execute as written. **Fix:** add explicit box pull and instance push/pull handoffs, and prefix Python commands with `.venv/bin/python`.

**Claims I could not verify and why**

- **1–2: partial.** Branch dispatch, explicit smoke refusal and basic SHA consistency hold; findings 2–4 defeat identity guarantees. The dose fallback does **not** admit another dose: the separate `SCREEN_DOSE` comparison still requires 5M.
- **3: partial.** Default prefixes, namespace, search parameters and Success@5 over the returned run are correct. Global pid tie ordering is not guaranteed when a tie crosses the top-100 cutoff. Tests use fewer than 100 documents, injected encoders/students and one search chunk; they do not exercise production loading, caching or that cutoff.
- **4: arithmetic holds; upper-bound convention is unspecified.** Pairing, macro weighting, sorted RNG traversal, B/seed and strict conjunction match. `inverted_cdf` selects order statistic 9750; reflecting the lower-tail convention selects 9751. An in-memory boundary example produced `−0.004000001` versus `−0.003`, reversing the veto. This establishes sensitivity, not a proven registration violation: explicitly pin the upper method before access. Constant-delta, equal-size tests cannot validate this or distinguish pooled weighting.
- **5: false overall.** On the explicit default path, `claim_lotte()` installs the guard before `_attempt` opens LoTTE. A crash **after the gate record is written**, before the end line, correctly blocks rerun. Neither fact repairs findings 1 and 6. `read_relpath`, cache provenance and `_environment` expose metadata, not payload text; the aggregate row is an authorized output.
- **6: conditional.** It hashes the four specified files, but at execution’s end. A checkout/edit during encoding makes it hash new disk bytes rather than the already-loaded implementation. Tests only repeat the disk-hash calculation.
- **7: partial.** E-only DEV-6 updates preserve the bindings shown in the permitted `contrasts.py` excerpts. Deferrals remain visibly marked but no reviewed consumer enforces eventual filling. Tests mock DEV-6 and use a toy model; crashes, concurrency and two-copy failure recovery are uncovered. Joint atomicity cannot be certified from the permitted `run_arm.py` excerpts.
- **Dtype/tower:** fp16 encoding is disclosed; no reviewed consumer numerically compares LoTTE with the frozen six-set caches. Ordinary conflicting teacher IDs/revisions are refused despite `setdefault`. Full encoder-registry behavior and wider consumers were outside scope.

**Files opened**

- `CLAUDE.md`
- `m13/LOTTE_GATE_REGISTRATION.json`, `m13/RULINGS.md`, `m13/CODEMAP.md`, `m13/EXECUTION.md`
- `m10/LOTTE_LOCK.md`
- `m8/LEDGER.md`
- `m13src/lotte_gate13.py`, `m13src/test_lotte_gate13.py`
- `m13src/dev6_from_checkpoint.py`, `m13src/test_dev6_from_checkpoint.py`
- `m13src/build13.py`, `m13src/build_lock.py`
- `m10src/run_arm.py`, `m10src/test_run_arm.py`, `m10src/contrasts.py`
- `m10src/nano10.py`, `m10src/trainer10.py`
- `m8src/paths_guard.py`, `m8src/freeze_lotte.py`
- `m7src/evalkit.py`, `m7src/teacher.py`
- `m9src/final_stats.py`

No protected surfaces or tests were accessed/executed. Only the permitted in-memory arithmetic check ran.
