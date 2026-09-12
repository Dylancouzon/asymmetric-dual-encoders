# M17 Astra P1 re-check brief — dev-suite reader after two fix rounds (2026-09-12)

You are an adversarial, READ-ONLY reviewer. Do not edit any file. Do not run anything that
writes under `results/`, `work/`, or `m17/`. Findings only; do not confirm. This is the single
permitted re-check before the one irreversible V0 read; after you, only owner rulings remain.

## Context

Repository `/home/dylan/asymetric-dual-encoders`, branch `m17-zero-v1.1-planning`, HEAD `ef718a1`.
M17 is ON THE CLOCK, registry `LOCKED_EXECUTABLE`. Sequence so far:

- `2c5321c` wired the dev-suite reader in `m17src/evaluate.py`.
- Your review (final block of `work/m17/logs/astra_devreader_review.log`): six P1 / three P2.
- `4824f84` fixed those nine (dispositions in `m17/REVIEW.md`).
- Codex Sol reviewed the fix (final block of `work/m17/logs/sol_devreader_fix_review.log`):
  seven P1 / three P2; one P1 dropped as spurious (HEAD had moved because a brief was committed).
- `ef718a1` fixed the rest (dispositions in `m17/REVIEW.md`, latest table).

Current state as claimed by the executor (unverified by me):

- Production entry point `dev_suite_read(bundle, out, *, allow_dev_suite)` loads the on-disk
  registry itself; no manifest/subset/loader/depth/fixture parameters. Test-only
  `_dev_suite_read_fixture` holds the overrides; the CLI cannot reach it.
- Writes only to `results/m17_v0_read.json`; receipt claimed with `O_CREAT|O_EXCL`; refuses
  unless `lock.executed.v0_export.read is False`; the flag is never flipped in code.
- Per-component per-query metrics persisted into the receipt via tmp + `os.replace`; a receipt in
  `started`/`failed` resumes remaining components only if every `IDENTITY_FIELDS` entry matches
  the fresh preflight; `complete` or unparseable receipts are refused; an empty claim left by a
  preflight refusal is released.
- Dirty tracked tree refused; sha + porcelain recorded.
- nDCG@10 and Recall@10 from one pytrec_eval evaluator over one run.
- `_pool_identity` reproduces `heldout._verify_pool` plus active-encoder and memmap-filename checks.
- `_teacher_doc_vecs` uses `encode_cached(verify=True)`: the four real stella caches are all
  trust-on-first-use, so the read is currently REFUSED at preflight. Owner ruling pending.
- Ordered `(qid, text)` digests are verified for the two held-out components and recorded-only
  for the four text-backed ones (no pinned identity exists). Owner ruling pending.
- `pytest m17src/test_evaluate.py -q` → 32 passed. Full `pytest -q m17src` → 4 failed / 320
  passed (four pre-existing registry-status assertions in test_lock / test_train).

## Files you may open (no others; no `grep -r`, `rg`, or `find`)

- `m17src/evaluate.py`, `m17src/test_evaluate.py`, `m17src/common.py`, `m17src/lock.py`,
  `m17src/export.py`, `m17src/loader_np.py`
- `m17/registry.json`, `m17/REVIEW.md`, `m17/CODEMAP.md`, `m17/STATUS.md`
- `results/m7_dev_manifest.json`
- `m7src/devsuite.py`, `m7src/heldout.py`, `m7src/evalkit.py`, `m7src/teacher.py`,
  `m7src/hashing.py`, `m7src/pool.py`, `m7src/encoders.py`
- `git diff 2c5321c ef718a1`
- You may run `.venv/bin/python -m pytest m17src/test_evaluate.py -q` with
  `TMPDIR=/home/dylan/asymetric-dual-encoders/work/m17/scratch/pytest_tmp` (create nothing else).

## Read-exclusions (absolute)

- Never open `results/frozen_eval/untouched-*`, any reserved qrels cache, `work/m9reserve`, or
  anything concerning FEVER, DBpedia-entity, cqadup-android, cqadup-english.
- Never read anything under `work/m17/prepared/full` or `work/m17/bundles/V0` as a quality
  surface; never execute `dev_suite_read`, `--surface dev-suite`, or anything that scores an M17
  artifact on the development suite.

## Questions

1. Of your original six P1 and three P2, which are now closed, and which remain open? Cite lines.
2. Of Sol's findings, is any fix wrong or incomplete in a way that could spend the read on a
   wrong or unrecoverable number? In particular: the resume rule (can a resumed run mix metrics
   from two different bundles or manifests?), the O_EXCL claim and its release on preflight
   refusal (can a refused preflight leave the read permanently blocked, or the release delete a
   legitimate receipt?), and the single-evaluator metric path.
3. New regressions: anything the two fix rounds broke or newly exposed.
4. If the owner rules to accept the trust-on-first-use caches with a dated disclosure (perhaps
   after a sample re-encode check), what is the smallest code change that does so WITHOUT weakening
   the refusal for any cache lacking that disclosure? If the owner rules recorded-only for the
   text-component query digests, is the current recording sufficient to make the read auditable?

## Output

Ranked P1/P2/P3 with file:line, a one-sentence failure scenario and the smallest fix each. State
explicitly: apart from the two owner rulings, does anything block the V0 read? Under 50 lines.
