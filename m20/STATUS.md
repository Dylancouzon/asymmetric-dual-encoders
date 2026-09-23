# M20 status — stages A, B, C COMPLETE; stage D archive BUILT; two exit conditions outstanding

## Closure state

`instructions-m20.md` sets five exit conditions. Four are met; one is not, and it needs an owner
decision rather than more work.

| # | Exit condition | State |
|---|---|---|
| 1 | Reserved four complete with receipt | **MET** — `results/m13_reserved_run.json`, tag `m8-reserved-spent` at `0350d060` |
| 2 | BEIR-15 complete | **MET** — `results/m20_beir15_run.json` `COMPLETE`, `missing: []`, 176 rows |
| 3 | Archive verified at **both** targets | **NOT MET** — archive `BUILT`; `D:` copy and verification in progress; object storage unavailable |
| 4 | Pod `EXITED` | **MET BY RECORD** — three retained pods `EXITED` and STOP-only at the last live provider check; not re-verified against the provider in this session |
| 5 | The two items M22 needs: result files and archive manifest | **MET** — result files above, `results/m20_archive_manifest.json` `BUILT`, 2,223 files, 142.6 GiB, 26 of 26 corpora |

**M20 therefore cannot be declared closed by an executor.** Condition 3 requires an S3-compatible
bucket, credentials and `rclone`, none of which exist on this box. The owner's choices are to
supply them, or to rule M20 closed with the remote half of stage D carried into M22 — which is a
release-policy decision and is explicitly not inferable from any handoff.

Condition 3 has two parts and **both** are outstanding: the `D:` re-hash must finish and be
recorded, and the object-storage half needs a bucket *and* the `--verify-remote` fix that
2026-09-23's review exposed. Nothing else in M20 is outstanding.

## RESUME HERE

**Stage C is done.** `results/m20_beir15_run.json` ends `COMPLETE` with `missing: []` — eight
systems x fifteen BEIR datasets, 22 corpora scored, 176 per-system rows, finished 40.5 h inside
its deadline. Do not re-run it; the launcher will refuse anyway.

**Stage D's archive is BUILT** (2026-09-23): `results/m20_archive_manifest.json`, 2,223 files,
142.6 GiB, 26 of 26 corpora, every staged file hashed against its recorded digest during the build.
The copy to `/mnt/d/constella-archive/beir15/` and its re-hash verification are the remaining local
step. The object-storage half needs credentials the box does not have — an owner decision.

Stage C took three attempts, none of them wasted work: a projection gate that refused a viable run
on the wrong unit, a host crash, and a latent split bug. All three are written up below and in
`m20/FINDINGS.md`. Every attempt receipt is preserved under
`results/m20_stage_c_run_attempt*.json`, and **the original deadline was inherited through all of
them, never reminted**.

If stage C ever needs re-running, the invocation is below — and it must **append (`>>`), never
`>`**, and must preserve `results/m20_stage_c_run.json` under the `attempt*` pattern first.

Then watch it: first shard rate per tower, dataset completions, and
`Traceback|Error|FAILED|OOM|Killed|assert`. `setsid nohup` matters — a harness interrupt kills a
plain background job. Expect 130–215 docs/s for stella-class encoding on this box; a rate far
below that is the allocator symptom in `m20/FINDINGS.md`, not normal variation.

Three things that are easy to get wrong here, all covered below: do not invoke `m20src/beir15.py`
directly, the reserved four must never be re-decided, and the projection gate is not resume-aware
so a refusal on relaunch may be counting finished work as outstanding.


**Opened 2026-09-16 under R22. Host moved to the local box 2026-09-17 under R24.**

**The reserved access has been spent, once, as registered.** `m8-reserved-spent` is on origin at
`0350d060`, pushed before the first protected read. `results/m10_final_run.json` now ends
`COMPLETE`. The reserved four — FEVER, DBpedia-entity, cqadup-android, cqadup-english — are from
here on **known-test**: unusable for re-deciding anything in this project, and a disclosed
known-test set for future ones. The three retained Runpod pods remain `EXITED` and STOP-only;
retirement is M22's to request.

## The result

Eight systems x four datasets, nDCG@10, exact retrieval, descriptive, `alpha = 0`.
`results/m13_reserved_run.json`; per-query rows under `results/m10_final_scores/reserved/`.

| system | fever* | dbpedia | cqad-android | cqad-english |
|---|---|---|---|---|
| nano-dense | 0.6231 | 0.4190 | 0.4774 | 0.4298 |
| zero-dense | 0.6978 | 0.3900 | 0.4431 | 0.3690 |
| stella-query | 0.8207 | 0.4603 | 0.5275 | 0.5183 |
| bge-small-en-v1.5 | 0.8662 | 0.4003 | 0.4761 | 0.4558 |
| leaf-ir-asym | 0.8671 | 0.4520 | 0.5260 | 0.4707 |
| bm25 | 0.5036 | 0.3045 | 0.3952 | 0.3481 |
| zero+bm25 dbsf@100 | 0.7559 | 0.4030 | 0.4629 | 0.4016 |
| nano+bm25 dbsf@100 | 0.6879 | 0.4296 | 0.4750 | 0.4323 |

\* FEVER is the disclosed double-contamination sensitivity row: outside NDO-3 and outside both
registered contrasts.

- **R1, nano vs bge-small:** NDO-3 `+0.0032`, CI `[-0.0069, +0.0134]` — straddles zero. Not
  uniform: `+0.0187` on DBpedia, `+0.0013` on android, `-0.0260` on english, and the query-pooled
  weighting turns it `-0.0121` with a CI excluding zero. The registered NDO-3 weighting
  (0.50/0.25/0.25) is the headline precisely because it was registered in advance.
- **R2, nano vs leaf-ir-asym:** NDO-3 `-0.0389`, CI `[-0.0488, -0.0290]` — behind, consistently
  across all three clean datasets and every reweighting.

**Astra verified these numbers independently** (2026-09-19): all 32 means reproduced from the
per-query rows, the paired within-dataset bootstrap regenerated (B=10000, seed 902, `inverted_cdf`)
with a matching draw-plan hash and max point-estimate difference below 7e-18, query counts and sets
confirmed, and every fusion row shown to name the correct dense producer plus BM25 with matching
run hashes and truncation to 100 before DBSF. The FEVER reversals — nano ~0.24 below both
comparators there, bge-small above stella-query there only — were examined and found **not** to
establish a defect; they were also **not** attributed to contamination on this evidence.
Record: `research/m20-numbers-review-astra-2026-09-19.md`.

Intervals quantify query resampling only, not training-seed variation.

## The stage C result

`results/m20_beir15_run.json`, per-system rows under `results/m20_beir15_scores/<corpus>/`.
Eight systems, fifteen datasets, nDCG@10, exact retrieval, **descriptive, `alpha = 0`**.

| dataset | nano | zero | stella | bge-s | leaf | bm25 | z+bm25 | n+bm25 |
|---|---|---|---|---|---|---|---|---|
| scifact | 0.7211 | 0.6101 | 0.7796 | 0.7127 | 0.6990 | 0.6791 | 0.7173 | 0.7433 |
| nfcorpus | 0.3631 | 0.3124 | 0.4134 | 0.3430 | 0.3608 | 0.3180 | 0.3442 | 0.3683 |
| scidocs | 0.2177 | 0.1677 | 0.2395 | 0.2052 | 0.2034 | 0.1565 | 0.1850 | 0.2031 |
| trec-covid | 0.7871 | 0.5490 | 0.8234 | 0.7575 | 0.8301 | 0.6099 | 0.7184 | 0.8072 |
| fiqa\* | 0.4778 | 0.3728 | 0.5536 | 0.4035 | 0.4165 | 0.2532 | 0.3872 | 0.4353 |
| arguana\* | 0.6233 | 0.5916 | 0.6369 | 0.6034 | 0.5833 | 0.4878 | 0.5800 | 0.5759 |
| fever\*† | 0.6231 | 0.6978 | 0.8207 | 0.8662 | 0.8671 | 0.5036 | 0.7559 | 0.6879 |
| dbpedia-entity† | 0.4190 | 0.3900 | 0.4603 | 0.4003 | 0.4520 | 0.3045 | 0.4030 | 0.4296 |
| cqadupstack† | 0.3859 | 0.3416 | 0.4448 | 0.3905 | 0.4269 | 0.3184 | 0.3756 | 0.3933 |
| msmarco | 0.4063 | 0.3371 | 0.4373 | 0.4082 | 0.4136 | 0.2256 | 0.3410 | 0.3699 |
| nq | 0.5838 | 0.5173 | 0.6371 | 0.5018 | 0.6164 | 0.2916 | 0.4594 | 0.4999 |
| hotpotqa | 0.6102 | 0.6127 | 0.6950 | 0.6993 | 0.6931 | 0.5851 | 0.7015 | 0.6968 |
| webis-touche2020 | 0.2755 | 0.2288 | 0.2980 | 0.2603 | 0.3024 | 0.3324 | 0.3102 | 0.3398 |
| quora | 0.8805 | 0.8504 | 0.8912 | 0.8863 | 0.8615 | 0.8055 | 0.8688 | 0.8838 |
| climate-fever\* | 0.2473 | 0.2785 | 0.2907 | 0.3183 | 0.3777 | 0.1372 | 0.2521 | 0.2315 |

\* disclosed teacher exposure. † **copied** from the tagged reserved transaction, never rescored —
fever and dbpedia-entity match `results/m13_reserved_run.json` to 1e-12; the cqadupstack row
aggregates all twelve forums, two copied and ten scored here.

**This is descriptive.** No gate, threshold, release rule, superiority or equivalence claim, and
it does not enter or reweight M13's registered six-set inference.

Two patterns worth carrying into the writeup:

- **DBSF@100's value tracks BM25's strength on the dataset.** It lifts zero-dense +0.089 to the
  top of hotpotqa (BM25 0.585) and costs nano-dense 0.036 on msmarco (BM25 0.226). M12's
  recommendation is a disclosed product-policy override with no equivalence CI, so report this
  dataset-dependence beside it.
- **The bge-small > stella-query reversal is not unique to FEVER.** It also appears on hotpotqa
  and climate-fever. That weakens, without refuting, reading FEVER's reversal as a contamination
  signature — consistent with the adversarial review that found it compatible with the disclosed
  exposure but declined to attribute it.

External checks passed on the way: SciFact reproduces `m20/SMOKE.md` for all eight systems,
NFCorpus reproduces M12's registered DBSF@100 value, and msmarco's bm25 (0.2256 vs a published
0.228) and bge-small (0.4082 vs a published 0.408) match published BEIR figures.

## Next step

**Stage D, the archive** — blocked on object storage; see the open items. Nothing else in M20 is
outstanding.

The encoded row count is 20 short of the registered 23,744,806. Fully attributed: every corpus
with a published count matches exactly, and the whole delta is in the one rounded budget line,
`cqadupstack_ten_forums: 394,000` against an actual 393,980 (0.005%). Nothing is missing.

`results/m20_climate_fever_comparison.json` records the climate-fever/FEVER corpus comparison the
registry requires — different on document count and both hashes, so encoding it separately was
correct. **No code performs that comparison**; it was recorded by hand from the existing pin and
the frozen manifest. If the registration expects it automatically, `m20src/beir15.py` is where it
belongs.

## Start here to execute

Stages A and B are **done** and must not be re-run: their launcher refuses anyway, because the
access is spent and the six-set result no longer reads `INCOMPLETE_RESERVED`. Stage A took 25 h
against its 52 h cap and stage B 0.93 h against 55.2 h; receipts `results/m20_local_run.json` and
`results/m20_local_run_attempt1_projection_stop.json`.

Stage C, BEIR-15, is **complete** and must not be re-run. This is the form to use only if some
future change genuinely requires re-running it:

```bash
.venv/bin/python -u scripts/m20_local_run.py --stage-c             # preflight only
setsid nohup .venv/bin/python -u scripts/m20_local_run.py --stage-c --execute \
  >> work/m20/logs/stage_c.log 2>&1 < /dev/null &
```

**Append (`>>`), never `>`** — see the RESUME HERE note. A truncating redirect here would destroy
the controller log of every earlier attempt (Sol, 2026-09-22, P2). Before relaunching, preserve
`results/m20_stage_c_run.json` under the `m20_stage_c_run_attempt*.json` pattern or the launcher
refuses; that pattern is also what makes the original deadline bind instead of a fresh one.

**Do not invoke `m20src/beir15.py` directly.** It is a scorer: its document-vector path loads
existing shards and fails on a missing one, it never schedules an encode, and it carries no
deadline, timeout or projection gate, so alone it would neither do the ~23.7M documents of missing
corpus work nor enforce the registered 120 h cap (Astra, 2026-09-19). `--stage-c` pre-encodes
BEIR-15's own corpora on all three towers and then scores, under one durable deadline that starts
before the preflight, with 17.06 h of the cap held back for download and scoring and the
projection gate armed against the registered volume.

Stage D, the archive, is `m20src/archive.py` and is blocked on object storage; see the open items
below.

## A latent split bug stopped scoring on 2026-09-22, and how it was fixed

Scoring completed six corpora and died entering msmarco:
`ValueError: Unknown split "dev". Should be one of ['train', 'validation', 'test']`.

The registry registers msmarco's split as `dev` — BEIR's standard MS MARCO evaluation split — and
the scorer passed that name to Hugging Face, which publishes it as `validation`. The SciFact-only
smoke never reached msmarco, so the path had never executed.

Established, not assumed: `validation` holds 7,437 rows over 6,980 unique queries, the published
BEIR figures to the row, while `test` is TREC-DL's 43 queries and is not BEIR's row. Astra added
the mechanism — the upstream repo at the pinned revision carries `dev.tsv`, and Hugging Face
normalizes `dev` filenames to `validation`. All 22 public corpora were checked: msmarco is the
only mismatch.

Fixed in `registered_split`/`hf_split`: the scored split is read from the registry rather than
restated in code, and only the physical name is remapped at the loader, so every score row records
the **registered** name. Astra reviewed: **GO**, no essential defect, 48 already-written rows
verified unaffected. Record: `research/m20-split-fix-review-astra-2026-09-23.md` — which also
records a **reviewer access incident** (a web search returned a reserved-forum qrels preview into
its tool output; contained, nothing reached the repository, no finding depended on it). **Future
review briefs must forbid web searches naming a reserved dataset, not only local loads.**

## The host crashed on 2026-09-22, and what it cost

The WSL box went down at ~09:10 EDT mid-encode of leaf-ir-asym and returned at 18:02 — **~8.9 h
off a durable wall clock**, which runs whether or not the machine does. Cause is **undetermined**:
the reboot cleared the kernel ring buffer and journald's last write preceded the crash, so there
is no OOM, thermal or driver evidence either way. Boot did log
`EXT4-fs (sdd): 8 orphan inodes deleted; recovery complete`.

**One shard was damaged:** `leaf-ir-asym/climate-fever/shard_00030.npy`, recorded at 76,800,128
bytes and present on disk as 0. Every written shard on every tower was then re-hashed against its
manifest record — **1,978 shards, 131.1 GiB, 19.2 min, exactly that one damaged**. A size check
would not have been enough: a shard that kept its length and lost its contents would pass one and
enter a published number invisibly. The reserved four's stage A shards are included and intact.

The 0-byte file was deleted so it re-encoded, and **it reproduced byte for byte** — sha256
`9c46302ee7f5c3fe655c1a039e468ac4fb0dd32c8b48e421e477dc7f01519f0a`, the value recorded before the
crash. That confirms the crash destroyed the file rather than the encoder producing anything
different, and re-demonstrates the deterministic encoding stage A established. No number was
affected: climate-fever was mid-corpus on the third tower with no score produced.

Receipt `results/m20_crash_shard_verification.json`. Crashed receipt preserved as
`results/m20_stage_c_run_attempt2_host_crash.json`, so the original deadline still binds.

Sol reviewed the recovery: **GO WITH CONDITIONS**, no P0/P1, run continues. It independently
re-derived the repaired shard's hash and confirmed the deadline is retained in all three receipts.
Its P2 (stale runbook, truncating `>`) and P3 (over-attributed crash mechanism, wrong
"unrepeatable" wording) are fixed here and in the receipt.
Record: `research/m20-crash-recovery-review-sol-2026-09-22.md`.

## The projection gate refused attempt 1 on 2026-09-19, and why attempt 2 fits

The gate stopped stage C after 0.6 h at a shard boundary: 59 docs/s measured, projecting 59.9 h
past the deadline. Nothing protected was touched, the 134,473 documents written are hash-recorded
and resumable, and the registration had reserved exactly this decision for the owner ("raising the
ceiling, narrowing BEIR-15, or accepting a partial descriptive table; no executor may decide that").

None of those three was needed, because the run does fit. The gate arms at `min_rows` on
cumulative **documents per second**, and BEIR-15's order runs its six longest corpora first --
272,117 documents at ~300 tokens, 1.1% of the volume, against the batch's measured 92.2-token
mean. Token throughput was flat the whole time (13,400-15,200 tok/s, matching stage A); only
docs/s moved, tracking length. The 2,190M tokens cost **66.5 h across all three towers** at the
conservative rate, against **97.9 h** of encode budget remaining.

Fixed by `batch_min_projection_rows`: BEIR-15 arms at 1,000,000 rows, ~728,000 documents into
msmarco; the uniformly short reserved batch keeps the 100,000 stage A ran under. No existing
function changed -- Astra verified this by AST diff -- so no already-written vector and no stage
A/B number can move. Tests: `m8src/test_projection.py`, wired into `run_m8_tests.sh`.

Astra reviewed the diagnosis and the fix: **GO WITH CONDITIONS**, no P0/P1 on the patch, all three
conditions discharged before relaunch (attempt-1 receipt and log preserved, deadline recalculated
and inherited, change committed and pushed first).
Record: `research/m20-stagec-gate-review-astra-2026-09-20.md`.

**The limit of the claim** (Astra): front-loading the long corpora makes the *initial* sample
conservative; it does not make every later prefix conservative, since climate-fever and touché
follow millions of short msmarco documents. The gate is an early-stop estimate, not a guarantee.

## Stage A stopped once on 2026-09-17, and why it now fits

The projection gate stopped the first launch at a shard boundary after 100,000 documents:
37 docs/s measured, 58.3 h past the stage-A deadline. It worked exactly as registered — nothing
protected was touched, access stayed unspent, both shards are hash-recorded and resumable.

The cause was not the data (FEVER token length is flat across the two shards, 85.2 vs 86.3 mean),
not thermal (40k documents in 2,000-document calls hold 141 docs/s at 68–74 °C), and not the
launcher's thread pinning (the rates reproduce in a clean 8-thread process). A length-sorted
50,000-document call fragments the caching allocator — 23.06 GiB reserved on a 10.24 GiB card for
1.76 GiB of live tensors — and WSL's driver then falls back to host memory over PCIe instead of
raising OOM, so `num_alloc_retries` stayed 0 and `results/m20_vram_probe.json` passed honestly
without seeing it.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` bounds the reservation to 4.40 GiB and gives
179 docs/s, and re-encoding FEVER 0:50000 under it reproduces the already-written shard **byte for
byte**. No registered parameter, estimand or encoding contract changed. Stage A now projects 23 h
(29 h at the conservative 141 docs/s) against the 52 h cap, and the gate stays armed.
Receipt: `results/m20_allocator_probe.json`.

Sol reviewed the stop, the fix and the relaunch: **GO WITH CONDITIONS**, one P1 — the launcher
minted a fresh `now + 52 h` on every execution, so repeated relaunches could have accumulated past
a cap the registration defines as **one** wall clock. Fixed: the earliest preserved attempt
receipt's deadline now binds. Effective slip to date is 1.83 h. The binding stage-A deadline is
**epoch 1789812044.886**; stage A must finish, or the owner must rule, before it.
Record: `research/m20-allocator-review-sol-2026-09-17.md`.

## Done

- **Registration** (`m20/REGISTRATION.md`, `m20/beir15_registry.json`, the dated
  `_amended_2026_09_16` block in `m10/final_run_registry.json`). Reversing the registry amendment
  reproduces the pre-amendment bytes exactly, which is the hash the frozen six-set result pinned.
- **Executor** extended to the eight-system roster across `m20src/roster.py`,
  `m13src/reserved_support.py`, `m13src/score13.py`, `m13src/reserved_transaction.py` and
  `m8src/pre_encode.py`. 279 `m13src` tests pass.
- **BEIR-15 runner and archive builder**, `m20src/beir15.py` and `m20src/archive.py`.
- **Two review rounds.** Astra NO-GO with seven P1 blockers, all fixed; Fable GO with no blockers.
  `m20/REVIEW_TRIAGE.md`.
- **Host validation.** Astra ENDORSE WITH CONDITIONS, all conditions discharged. `m13/RULINGS.md`
  R24, `m20/REGISTRATION.md` §8.
- **Evidence the numbers are right.** `m20/SMOKE.md`: the real BEIR-15 path on SciFact reproduces
  every independently published value. `results/m20_dbsf_reproduction.json`: three zero deltas.
- **Feasibility measured, not assumed.** `results/m20_vram_probe.json` and
  `results/m20_bm25_memory_probe_*.json`. No registered parameter needed changing.
- **Stage A** (2026-09-17/18): 41 GiB of fp16 vectors, three towers x four datasets, ~25 h against
  the 52 h cap. nano-dense 19.3 GiB, bge-small 7.2 GiB, leaf-ir-asym 14.5 GiB.
- **Stage B** (2026-09-19): the tagged transaction, 0.93 h against 55.2 h. Access spent once,
  eight systems x four datasets, per-query rows persisted, reserved queries and qrels exported to
  the archive from the payloads already open — no additional protected read.
- **Four review rounds on the execution path.** Astra NO-GO on stage B (2 P1 + 1 P2), fixed; Sol
  NO-GO on those fixes (3 P1) — the owner directed simplification, so the recovery paths were
  deleted rather than hardened; Astra SOUND on the numbers and NO-GO on stage C readiness, fixed;
  Sol NO-GO on the stage-C invocation (5 P1), four fixed and one accepted. Records in `research/`
  dated 2026-09-17 to 2026-09-19.

## Open items for the owner

1. **Object storage, plus an implementation gap behind it.** No account, bucket, credentials or
   `rclone` on the box. This blocks exit condition 3.

   **A bucket alone would not have been enough.** `verify_remote` called
   `rclone hashsum sha256 <remote>` with no `--download`, which asks the backend for a checksum it
   already holds; S3 advertises MD5, not SHA-256, so the call returns nothing to compare and the
   check could never pass whatever was stored (Astra, 2026-09-23, P2). `--download` is now passed,
   so rclone fetches each object and computes the digest from the bytes. **That path has still
   never run against a real remote** — it fails closed, but treat its first real execution as a
   smoke, not a verification.

   The registration also requires the bucket name and provider to be recorded in
   `m20/REGISTRATION.md` §4 by the executing session at creation time, before upload, with no
   credentials in git, logs or the pod.

   Either supply a bucket, or rule M20 closed with the remote half carried into M22. The second is
   a release-policy decision; an executor may not infer it.

   **What the local half does and does not give you.** `work/m20-archive` is a *staging* root: its
   shards are hard links to `work/m13-reserved-enc`, deliberately, because copying ~147 GB beside
   itself needs 294 GB. It shares inodes with the encode tree and is not an independent copy.
   `/mnt/d/constella-archive/beir15/` **is** an independent copy on separate hardware. So today
   there is exactly one durable copy of the 142.6 GiB, and it is on a drive in this machine.
2. **Retained storage bills about $0.41/hour** across the three stopped pods, roughly double the
   $0.2174/hour the M13 allocation assumed, whether or not anything runs. Measured over a 7-hour
   window with everything stopped. Retirement is M22's to request; they stay STOP-only until then.
3. **Stage C is done** — closed 2026-09-23, 40.5 h inside its 120 h cap, so it is no longer an open
   item. Recorded because the cap's derivation was wrong in an instructive direction: it assumed
   100.76 docs/s from an A100 benchmark on a 153-token surrogate, and the real corpora averaged
   92.2 tokens, so the box ran far faster per document than the cap's own arithmetic expected. The
   margin came from document length, not from the hardware the cap was priced on.

## The tightest point in the plan

BM25's peak host memory. FEVER needs 20.1 GB and MS MARCO 20.9 GB on a 26.7 GB box, and two large
corpora in one process reached 23.7 GB. Document count does not predict it: FEVER needs 60% more
than HotpotQA at almost the same count. The datasets run largest-first and each corpus's text is
freed and collected before the next loads. If a future change touches that ordering or that
freeing, re-measure before running.

## Noted, not fixed

- Running the `m10src` suite rewrites `results/m10_contrast_E1.json`, a real registered result
  file, which `CLAUDE.md` forbids. Pre-existing on `main`. Check `git status` after `run_checks.sh`.
- Three review debt items are recorded in `m20/REVIEW_TRIAGE.md` and deliberately not fixed.

## Pointers

- Mandate `instructions-m20.md`. Rulings `m13/RULINGS.md` R19, R20, R22, R23, R24.
- Registration `m20/REGISTRATION.md`, `m20/beir15_registry.json`. Smoke `m20/SMOKE.md`.
- Reviews `m20/REVIEW_TRIAGE.md`, `research/m20-codex-impl-review-2026-09-16.md`,
  `research/m20-host-decision-brief-astra-2026-09-17.md`.
- Failed cloud attempts `results/m13_reserved_cloud_attempt{1,2}_nocapacity_*.json`.
- Artifact paths, hashes, restore commands `m14/HANDOFF.md`. Checks `HARNESS.md`.
- Follow-on `instructions-m22.md` (release), `instructions-m23.md` (upstream PRs).
