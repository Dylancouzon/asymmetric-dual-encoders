# Closing re-check: your three findings on the LoTTE gate — GO or NO-GO

Repository `/home/dylan/asymetric-dual-encoders`, branch `m13-stage1-execution-prep`, read-only.
Your previous report (`research/m13-codex-gate-final-2026-09-10.md`) returned NO-GO on two P1s and one
P2; each is fixed in the latest commit (`git diff 5adcc13..HEAD -- m13src/lotte_gate13.py
m13src/test_lotte_gate13.py`). Re-check those three and the two validation gaps you listed under
"claims I could not verify" (recovery's `n_docs`/`n_qrels_pairs`/query-id digest). Look for a
regression the fixes could have introduced. Then end with exactly one line: `VERDICT: GO` or
`VERDICT: NO-GO — <one sentence>`.

The owner has ruled that the code base must not grow further without cause; the standard is a
plausible accident by one researcher on one box plus the one-read guarantee, not a deliberate
attacker with write access to the repository. Downgrade accordingly and say when you do.

## Hard rules

- Never open, list, cat, grep, hash or otherwise read anything under `work/lotte/`, `work/m9reserve/`,
  `results/frozen_eval/untouched-*`, `work/dev/cqadup-android.json`, `work/dev/cqadup-english.json`,
  or any reserved qrels cache.
- No recursive searches (`grep -r`, `rg`, `find`, `ls -R`, `tree`, globbing). Open only:
  `CLAUDE.md`, `research/m13-codex-gate-final-2026-09-10.md`, `m13/REVIEW_TRIAGE.md`,
  `m13src/lotte_gate13.py`, `m13src/test_lotte_gate13.py`, `m13src/build13.py` lines 117–320,
  `m10src/nano10.py` (`Nano10.__init__`), plus `git diff 5adcc13..HEAD` on the two m13src files.
- Do not run tests or scripts.

## What changed (verify)

1. `_persisted_slice`: a missing slice file whose key appears in `slices.jsonl` REFUSES ("a completed
   slice is never re-read"); only a key with no journal entry returns None and is read.
   Test: `test_recover_refuses_when_a_journaled_slice_file_is_missing`.
2. `acquire_lock`: `_fsync_dir(d.parent)` after `mkdir`, before the lock file is opened; the receipt
   and every `write_atomic` already fsync their directory.
3. `dependency_identity(cfg, student_key, model=None)`: at load time it is computed from the
   constructed `Nano10`'s own `model.tok` and `model.backbone.config` (the objects that score), after
   construction and before `torch.load`; at manifest time from a by-name load. `load_student`
   compares before deserialising the checkpoint.
4. `_persisted_slice` also requires `n_docs`, `n_queries`, `n_qrels_pairs` equal to the pin's and
   `sha_obj(sorted(qids)) == pin.hashes.query_ids_sha256`.

## Output

Numbered findings (P1/P2/P3, `file:line`, scenario, minimal fix), then one line per item 1–4:
CLOSED or NOT CLOSED, then files opened, then the VERDICT line. Terse; no praise.
