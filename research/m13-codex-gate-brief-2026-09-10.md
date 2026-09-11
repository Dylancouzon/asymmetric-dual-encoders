# Adversarial review: M13 stage 2 code — the LoTTE gate script and the deferred DEV-6 read

You are reviewing, read-only, the two commits `f957bd4` and `bc117c1` on branch
`m13-stage1-execution-prep` of the repository at `/home/dylan/asymetric-dual-encoders`. Your job is
to BREAK this code before it spends a one-shot protected read. A review that confirms returns
nothing useful; find the defect that makes the read wrong, repeatable, leaky or unverifiable.

## Hard rules for this review

- **Never open, list, cat, grep, hash or otherwise read**: anything under `work/lotte/`,
  anything under `work/m9reserve/`, `results/frozen_eval/untouched-*`, `work/dev/cqadup-android.json`,
  `work/dev/cqadup-english.json`, or any reserved qrels cache (FEVER, DBpedia-entity,
  cqadup-android, cqadup-english). These are protected evaluation surfaces; touching them is a
  protocol violation regardless of intent. Reason about their FORMAT from the code that reads them.
- **No recursive searches**: no `grep -r`, `rg`, `find`, `ls -R`, `tree`, or globbing over the
  repository. Open only the files named below, by path, plus `git diff` / `git show` on the named
  commits restricted to those paths.
- Do not run the test suites or any script (the sandbox is read-only and the tests write to tmp).
  You may run pure in-memory Python one-liners to check arithmetic.
- Read `CLAUDE.md` first for the project's evidence and protocol rules.

## Files in scope (open only these)

Authority the code must implement:
- `m13/LOTTE_GATE_REGISTRATION.json` — the registered slices, counts, metric, veto constants, identities, record contract
- `m10/LOTTE_LOCK.md` — WHEN the read happens, the two branches, the veto rule, the firewall
- `m13/RULINGS.md` — R8 and R16 (and the provider section)
- `m8/LEDGER.md` lines 1472–1495 — the two allowlist amendments

New code (the subject):
- `m13src/lotte_gate13.py`
- `m13src/test_lotte_gate13.py`
- `m13src/dev6_from_checkpoint.py`
- `m13src/test_dev6_from_checkpoint.py`
- `m10src/run_arm.py` — only the diff in `f957bd4` (`git show f957bd4 -- m10src/run_arm.py`), plus `dev6()` and the record-writing tail of `_run()` for context
- `m10src/test_run_arm.py` — only the diff in `f957bd4`
- `m8src/paths_guard.py` — `ALLOWLIST`, `claim()`, `check()`, `install()`

Code the subject depends on (read for how it behaves, do not review it):
- `m13src/build13.py` lines 117–225 (`_registered_e_checkpoints`, `check_gate`) — the consumer of the gate record
- `m13src/build_lock.py` — `verdicts`, `check_verdict_binding`, `resolve_batch`, `PENDING`
- `m7src/evalkit.py` — `topk_arrays`, `run_from_arrays`, `topk_ids_scores`, `per_query_ndcg`
- `m7src/teacher.py` — `encode_cached`, `cache_key`, `_combined`, `PROVENANCE`, `ENC`
- `m10src/nano10.py` — `Nano10.__init__`, `encode_queries`
- `m10src/trainer10.py` — `save()` (checkpoint blob shape)
- `m9src/final_stats.py` — `draw_plan`, `bootstrap` (the repo's registered bootstrap convention)
- `m8src/freeze_lotte.py` — the remediated slice FORMAT and the five-hash scheme (this module was the only prior reader of the remediated files)
- `m13/CODEMAP.md` pitfalls 4, 16–19; `m13/EXECUTION.md` (the readiness table and day-one runbook)

## What the author believes (state whether each holds; break it if you can)

1. **Branch and identities.** `branch_of` reads `selected.batch` from `results/m10_screen_verdicts.json`
   after `check_verdict_binding` ties it to the live `m10/screen_registry.json`. bs32 → candidate
   `E-bs32`, no comparator, decision `skipped`, observational row read. bs128 → candidate `E-bs128`,
   comparator `E-bs32`, veto runs. Both arm records must be `complete`, non-smoke, with
   `checkpoints.cycle3.sha256 == final_checkpoint_sha256`, dose 5,000,000 and the registry's batch;
   the loaded checkpoint file is re-hashed. The record `build13.check_gate` reads is then bound to
   exactly these shas and the live verdict sha.
2. **Slices.** `read_slice` opens exactly `collection.tsv`, `questions.forum.tsv`, `qas.forum.jsonl`
   under `work/lotte/remediated/<topic>/<split>/`, checks the three registered counts, uniqueness,
   qrel/query/doc set consistency, and hashes with `freeze_lotte`'s scheme. Never the search split.
3. **Scoring.** Documents: stella (`teacher.encode_cached`, prefix "", fp16 store, `teacher.ENC`
   rebound to `work/lotte/enc`), one cache per slice, shard-resumable. Queries: `Nano10.encode_queries`
   with prefix "" (prompt policy (b)). Exact search `topk_ids_scores(k=100, chunk=250_000)`.
   LoTTE qids and pids share one integer space and `run_from_arrays` drops a doc whose id equals the
   qid, so queries are scored under a `q:` namespace; `per_query_ndcg`'s result set is asserted equal
   to the slice's qid set. Success@5 is computed from the same run dict (top-5 by score, ties by pid).
4. **Veto.** `paired_bootstrap`: per-query candidate−comparator deltas on identical qids, slices in
   sorted order, `rng = default_rng(903)`, `(B, n_s)` index draws per slice, draw = mean over slices
   of the resampled slice mean, upper bound = `np.quantile(draws, 0.975, method="inverted_cdf")`.
   Fires iff point macro delta < −0.004 AND upper < −0.004. The author believes this matches
   `m10/LOTTE_LOCK.md` and the registration's `fires_iff`. Question whether `inverted_cdf` is the
   right direction for an UPPER bound (the repo uses it for the lower 0.0125 quantile).
5. **Access.** `claim_lotte()` runs inside `run()` only (one claim per process; import stays clean),
   claims `m13src.lotte_gate13` whose kind is `{lotte}`. Preflight opens no LoTTE path. An existing
   `m13/LOTTE_GATE.json` refuses. `attempts.jsonl` under `work/lotte/gate13/` counts starts/ends.
   Per-query scores are written under `work/lotte/gate13/`, never in the record. The record carries
   no query text, label or qid.
6. **Code identity** hashes `m13src/lotte_gate13.py`, `m10src/nano10.py`, `m7src/evalkit.py`,
   `m7src/teacher.py` from the source tree that runs.
7. **DEV-6 deferral.** `run_arm.py --dev6 defer` keeps the record `complete` (E1 needs the COV files,
   not DEV-6), writes `dev6: {deferred: true, checkpoint_sha256, fill_with}`, is refused under a
   smoke. `dev6_from_checkpoint.py` requires both published copies to agree, `complete`, non-smoke,
   deferred `dev6`, re-hashes `final_checkpoint`, rebuilds `Nano10` from the record's recipe with a
   parameter-count check, runs `run_arm.dev6`, rewrites both copies atomically with only `dev6`
   changed. The author believes no binding hashes the E arm records (`contrasts.py` hashes family F's
   records only) so the rewrite invalidates nothing — verify that claim against
   `m10src/contrasts.py` lines 90–112 and 270–285 ONLY.

## Questions to answer, in priority order

- Can the gate accept, score or record the WRONG checkpoint (stale record, smoke record, re-trained
  arm, swapped files), or score a slice that is not the registered one?
- Can read #1 execute twice, or execute for the wrong branch, or write a record `check_gate` accepts
  but that misdescribes what was read? Consider crashes between slices, a crash after the record is
  written but before `attempts.jsonl`'s end line, and re-runs after a partial encode.
- Does anything derived from LoTTE leave `work/lotte` (record fields, logs, prints, cache paths,
  `teacher.PROVENANCE`, pytrec_eval, the `_environment` block)? Is the record's `read_relpath` or
  `doc_caches` content a leak?
- Is the bootstrap the registered statistic? Check pairing, the macro-of-slice-means estimand, the
  RNG order, B and seed, the quantile method and direction, and the two-condition fire rule.
- Is Success@5 correct given the run dict holds top-100 after the self-hit rule under the namespace?
- fp16 document store here vs fp32 for the six-set frozen caches: does any consumer compare these
  numbers across dtypes? Is the choice disclosed where it must be?
- `teacher.ENC` is rebound process-wide; `M7_ENCODER` is `setdefault`-ed; the teacher identity check
  compares against `Config` pins. Any way the wrong tower encodes?
- Is the guard actually installed before the first LoTTE open on the REAL path? Trace `run()`.
- `arm_record`'s dose/batch checks: `int(entry.get("dose_examples", SCREEN_DOSE))` — is the fallback a hole?
- `dev6_from_checkpoint`: can it fill a record for a checkpoint that is not the arm's final one? Can
  `--dev6 defer` be used to skip DEV-6 permanently without anyone noticing?
- The day-one runbook in `m13/EXECUTION.md`: any ordering error, missing precondition, or command
  that cannot work as written?
- Tests: which of the author's claims above are NOT actually covered by `test_lotte_gate13.py` /
  `test_dev6_from_checkpoint.py` / the `test_run_arm.py` additions?

## Output format

Numbered findings ordered by severity, each: **P1/P2/P3**, `file:line`, the concrete failure
scenario (inputs/state → wrong outcome), and the minimal fix. P1 = wrong or repeatable protected read,
wrong checkpoint, leak, or a `check_gate`-acceptable record that misdescribes the read. Then a
short list titled "Claims I could not verify and why". Then the list of every file you opened.
Be terse and specific; no praise.
