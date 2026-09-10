# Codex (gpt-6-astra, high) implementation review of branch m13-stage1-execution-prep at 4d7a2f1 — 2026-09-10, read-only

Brief: every change in `git diff main..4d7a2f1`, plus a status check of the 13 design-review findings. Verbatim output follows. Triage: `m13/REVIEW_TRIAGE.md`.

I would not merge this as execution-ready. Several safeguards remain incomplete, and recovery can produce incorrectly attributed results or freeze the wrong checkpoint.

Part A below refers to the prior review’s numbering. Unqualified filenames are under `m13src/`.

| Finding | Status | Implementation evidence | What remains |
|---|---|---|---|
| 1. Query/qrel authentication | **OPEN** | `score13.py:83,216–238` | Loader verifies corpus hashes; payload checks verify qids only. Authenticate query text and qrels on the exact loaded objects. |
| 2. Frozen-cache-only scoring | **PARTIALLY** | `score13.py:109–110`; `access13.py:255–279` | fp32 is explicit, but Stella identity is not. Missing shards still invoke encoding; consumed cache provenance is not persisted. |
| 3. Durable identity/attempts/bridge | **PARTIALLY** | `score13.py:297–301,483–486,518–597` | State arrives **after** tagging; zero-score continuation is refused despite R11. Successful bridges persist; failures and attempt counts do not. Recovery authentication is incomplete. |
| 4. Phase-specific guard/M9 policy | **OPEN** | `access13.py:165–175`; `score13.py:469–474,553–597` | Six-set files remain unsealed; an option enables pre-tag reads. M9 inherits continuation without its own amendment. |
| 5. Extension trajectory | **PARTIALLY** | `build13.py:603–622,548`; `trainer10.py:160–189` | Warmup is zero and wrapper history survives, but AdamW/RNG restart. Trainer kill history resets; global kill runs only after the extension returns. |
| 6. Global position/atomic stop | **OPEN** | `build13.py:119–142,593–634` | Stream offsets coexist with local mix phase. Extension progress, evaluation and completion are separate publications. |
| 7. E1/LoTTE selection | **PARTIALLY** | `build_lock.py:75–84`; `build13.py:350–361`; `m10src/arm_smoke.py:77` | E-bs32 now has a shape and PENDING is refused. R9 is not applied; gate contents/veto and verdict provenance are not enforced. |
| 8. Corpus/cut/forms policy | **PARTIALLY** | `m10src/corpus_loader.py:879–881,1506`; `build_lock.py:186–249` | Shipped configuration fixes the original mapping/cut/forms failures. Validator still accepts disabling `require_all_forms` and ignores document-policy subfields. |
| 9. R5 sampler specification | **PARTIALLY** | `m13/RULINGS.md:55`; `m10src/corpus_loader.py:1100–1115` | Full A4 and reshuffling are ratified. Inherited ragged-tail dropping violates “every eligible document once per epoch.” |
| 10. Budget accounting | **PARTIALLY** | `build13.py:184–190,560,632`; `build_lock.py:138–149` | Mandatory estimates are reserved upfront—this is addressed. Actual overruns, outstanding reserve reconciliation and unaffordable mandatory plans remain unchecked. |
| 11. Rehearsal isolation | **PARTIALLY** | `score13.py:64,85–98`; `rehearse13.py:69,170–188` | Corpus injection and canonical-registry rebinding work. No enforced network/production-data isolation; fixture git does not contain the executed source. |
| 12. Independent evidence failures | **OPEN** | `score13.py:332` | Dictionary comprehension aborts all decisions on any evidence exception. |
| 13. Narrow scope/remaining packages | **PARTIALLY** | `score13.py:460,352–384`; `m13/EXECUTION.md:8–16` | Explicit all-six iteration fixes the five-set issue. Reserved execution is a stub; M9 support is incomplete; remaining packages lack named owners. |

Part B, ordered by severity:

1. **P1 — `--preflight-only` can resume scoring and publish results.**  
   [score13.py:469](/tmp/claude-1000/-home-dylan-asymetric-dual-encoders/c7e885ae-a989-47ee-b115-24557a212405/scratchpad/review-snapshot/m13src/score13.py:469) dispatches to continuation before checking `preflight_only` at line 479. A routine readiness check after a crash can therefore reopen protected datasets. An extracted-function check reproduced this dispatch. Separately, line 474 permits payload reads before spending access. **Fix:** make preflight-only observational in every state, remove the obsolete R1 switch, and enforce dataset/phase permissions at the read boundary.

2. **P1 — Persisted rows are not authenticated as belonging to this run.**  
   [score13.py:580](/tmp/claude-1000/-home-dylan-asymetric-dual-encoders/c7e885ae-a989-47ee-b115-24557a212405/scratchpad/review-snapshot/m13src/score13.py:580) checks registry hashes but accepts missing run state. Recovery never validates each row’s system, dataset, checkpoint or BEGIN identity, never rehashes the comparator, and does not pin executed code or the remote tag target. Same-registry rows copied from another checkpoint can be relabelled as the current candidate. **Fix:** require a BEGIN-bound manifest, authenticate every row and comparator, validate the peeled remote tag, and permit only declared output changes.

3. **P1 — A failed bridge remains retryable.**  
   `score13.py:291` raises before persisting anything for that dataset. If earlier datasets succeeded, continuation treats the failed dataset as missing and rescoring remains unlimited (`:538–548`). Recovery also trusts stored bridge summaries without checking success. **Fix:** persist terminal bridge failure and attempt state before further dispatch; require authenticated bridge success before decisions. Create the empty manifest before spend and implement the accepted R11 zero-score continuation.

4. **P1 — Extension completion is not crash-consistent.**  
   [build13.py:258](/tmp/claude-1000/-home-dylan-asymetric-dual-encoders/c7e885ae-a989-47ee-b115-24557a212405/scratchpad/review-snapshot/m13src/build13.py:258) publishes the extension macro through `_sync()` before its checkpoint and completion entry (`:623`). A crash after that publication leaves the new macro but the old extension count. Restart evaluates plateau first and can select `cycle3.pt` while reporting extension-1’s macro. **Fix:** persist an explicit in-flight phase; reconcile its authoritative checkpoint, stop decision, counts and macro before deciding whether another cycle starts.

5. **P1 — Extensions change optimizer history and global data order.**  
   `build13.py:607–622` restores only weights for a fresh extension. Local trainer history can miss a kill at the first extension midpoint; the controller notices only after more training. Local Q/D phase also diverges from global phase: extension 2 starts at phase 3 for bs32 and phase 1 for bs128. Extracted arithmetic showed bs32 extension 3’s document offset skips one batch relative to actual consumption. **Fix:** preserve optimizer/RNG/kill state and pass separate global data and local LR positions.

6. **P1 — LoTTE veto and screen provenance are bypassable.**  
   [build13.py:354](/tmp/claude-1000/-home-dylan-asymetric-dual-encoders/c7e885ae-a989-47ee-b115-24557a212405/scratchpad/review-snapshot/m13src/build13.py:354) accepts `{}` as a gate; a recorded veto never changes bs128. `build_lock.py:75,203` trusts selected values without checking the verdict’s registry hash, E evidence or authenticated F records. **Fix:** validate a completed gate’s checkpoint identities and outcome, apply its batch veto, and bind authenticated selection artifacts into the receipt and benchmark.

7. **P1 — The ceiling limits estimates, not billed expenditure.**  
   `build13.py:632` adds the original projected cycle price regardless of actual duration. `--benchmark --resume` can replace the cap/spend during a running build (`:410–418`). An extracted cap check accepted a **$11,724.27 mandatory plan**, returning cap zero. **Fix:** reject unaffordable mandatory allocations; freeze benchmark identity before training; reconcile billed elapsed/setup/evaluation costs plus remaining reserves before each extension.

8. **P1 — Failed export/parity still produces a successful terminal build.**  
   [build13.py:740](/tmp/claude-1000/-home-dylan-asymetric-dual-encoders/c7e885ae-a989-47ee-b115-24557a212405/scratchpad/review-snapshot/m13src/build13.py:740) swallows export exceptions; neither parity result changes `ok`, so lines 659–685 publish `complete` and a final checkpoint. Scoring preflight merely checks existence of an older parity artifact (`access13.py:403`). **Fix:** require checkpoint-bound export/parity success for readiness, with resumable post-training finalization.

9. **P2 — “Every document” silently excludes a fixed tail.**  
   `EpochShuffledStream` inherits `data10.length_buckets`, which permanently drops the longest ragged batch (`m10src/data10.py:98`). It shuffles existing batches, not their membership. With 65 synthetic documents and batch eight, the extracted implementation presented only 64. **Fix:** specify and implement deterministic cross-epoch carry or another owner-consistent tail policy preserving the dose; test non-divisible pools.

10. **P2 — Validation and fingerprints omit consequential inputs.**  
    `build_lock.py:228–249` does not pin `extension_examples` or require `require_all_forms=True`. Changing extension length remains accepted. `build13.py:172–175` inherits a code fingerprint that omits `corpus_loader.py` and `data10.py`, allowing sampler changes with unchanged manifests to resume. **Fix:** validate all operative policy fields and fingerprint the actual loader/sampler code, handoff checkpoint hash and benchmark/gate identities.

11. **P2 — Default artifacts are not byte-identical, and tests miss the critical cases.**  
    `corpus_loader.py:1157,1549` unconditionally adds `epoch_shuffle:false` and `document_policy:null`; `trainer10.py:199,337` adds fields with `loss_log=None`. Thus manifests, checkpoint schemas and fingerprints change despite unchanged default training math. **Fix:** conditionally emit new fields or explicitly register the compatibility break. Tests use divisible 64-document pools, phase-aligned 40-step extensions, and only one extension; the sidecar-cap test runs 120 steps against a 200-loss limit (`test_build13.py:175,348,538`). Add non-divisible, multiple-extension, publication-window and >200-step cases.

12. **P2 — Rehearsal does not establish production isolation.**  
    [rehearse13.py:69](/tmp/claude-1000/-home-dylan-asymetric-dual-encoders/c7e885ae-a989-47ee-b115-24557a212405/scratchpad/review-snapshot/m13src/rehearse13.py:69) calls the Hub without `local_files_only=True`; the standalone CLI does not establish offline mode. Fixture commits contain data/configuration while executing source from the parent checkout, so source drift is invisible to fixture HEAD checks. **Fix:** run copied source in an isolated checkout, deny network/production paths, and use a fully local model configuration.

13. **P2 — Successful END can hide unfinished required work.**  
    `score13.py:418–438` converts reserved failure into `"NOT RUN"` and returns success. M9’s registry has `datasets`, whereas `run()` requires `partitions.all6`; its amended bridge and distinct crash policy are unwired. **Fix:** keep validated six-set decisions durable, but report mandatory reserved work as incomplete/nonzero; provide thin, separately tested M9/nano entrypoints. Assign owners for reserved adapters/statistics/recovery, LoTTE execution, paired-row registration, open-data bridge rehearsal and cost frontier. Drop the premature generic M9 mode and obsolete policy switches until implemented.

Checked and found sound:

- Exact 200M base arithmetic, bs128 extension flooring, example-matched warmup, compile/PENDING refusals and best-previous-cycle gain comparison.
- E-bs32’s shape matches ANCHOR; only E-bs128 receives the CUDA smoke length cap.
- Explicit all-six iteration, exact-search/qid checks, successful bridge summaries and atomic score writes.
- `git diff --check` passed. No training/evaluation or pytest ran. Torch import failed because the read-only environment had no writable temporary directory; subsequent probes used extracted functions and synthetic inputs. Reported test counts were not independently revalidated.

Access log — repository content files opened, deduplicated; paths relative to the snapshot. Git also inspected stat/diff/check/status/HEAD and tracked-path metadata. Python loaded installed dependencies. No excluded data was opened and no file was modified.

- `.gitignore`, `CLAUDE.md`, `run_checks.sh`
- `m13/STAGE1_DESIGN.md`, `m13/RULINGS.md`, `m13/EXECUTION.md`, `m13/CODEMAP.md`, `m13/STATUS.md`, `m13/build_config.json`, `m13/_codex_design_review_2026-09-10.md`
- `m13src/access13.py`, `m13src/score13.py`, `m13src/rehearse13.py`, `m13src/build13.py`, `m13src/build_lock.py`, `m13src/conftest.py`, `m13src/test_access13.py`, `m13src/test_score13.py`, `m13src/test_build13.py`
- `m10/CODEMAP.md`, `m10/M102_LOCK.md`, `m10/final_run_registry.json`, `m10/screen_registry.json`
- `research/archive/m10-cleanup-2026-09-10/m10/M102_LOCK.md` — lines 60–145 only
- `m10src/arm_smoke.py`, `m10src/corpus_loader.py`, `m10src/trainer10.py`, `m10src/nano10.py`, `m10src/run_arm.py`, `m10src/data10.py`, `m10src/final10.py`, `m10src/contrasts.py`, `m10src/test_nano10.py`, `m10src/test_run_arm.py`
- `m9/final_run_registry.json`, `m9src/final9.py`, `m9src/data.py`
- `m7src/final_run.py`, `m7src/evalkit.py`, `m7src/teacher.py`, `m7src/encoders.py`, `m7src/_paths.py`
- `m8src/paths_guard.py`, `bench/core.py`
