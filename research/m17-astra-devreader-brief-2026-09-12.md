# M17 Astra review brief — the dev-suite reader before the one irreversible V0 read (2026-09-12)

You are an adversarial, READ-ONLY reviewer. Do not edit any file. Do not run anything that
writes under `results/`, `work/`, or `m17/`. Report findings only; do not confirm.

## Context

Repository: `/home/dylan/asymetric-dual-encoders`, branch `m17-zero-v1.1-planning`, HEAD `2c5321c`.
M17 is ON THE CLOCK in status `LOCKED_EXECUTABLE`. The next step is the single declared,
irreversible read of the untrained vocabulary export V0 on the pinned M7/M8 development suite
(registry entry `untrained_vocab_export_v0`, `reads: 1`). Commit `2c5321c` wired the reader that
this read will use. If the reader is wrong, the one read is spent on a wrong number. Break it.

What the executor claims (from its report, unverified by me):

- `m17src/evaluate.py` gained: `dev_manifest`, `dev_components`, `_require_dev_suite`,
  `load_dev_component`, `_load_text_component`, `_load_heldout`, `_dev_doc_vecs`,
  `dev_suite_read`; CLI `--surface dev-suite --bundle`.
- Gate: `load_dev_component` / `dev_suite_read` need `allow_dev_suite=True` (CLI
  `--allow-dev-suite`) AND `require_executable(..., rehearsal=False)`, no rehearsal bypass.
  Panel branch still refuses with "reader is not wired up". `evaluate(query_ids=...)` key-set
  refusal untouched.
- Reuse: text components via `m7src/devsuite.load` and the M7 teacher encode cache for document
  vectors; held-out slices from their pinned JSON plus `heldout.pool_doc_ids`, corpus via
  `prepare_data.PoolReader`; hashes via `m7src/hashing`; scoring via `m7src/evalkit.score` /
  `macro`; queries via `loader_np.M17QueryEncoder` (int8, resident).
- Load-only smoke (no scoring) matched every pinned hash and count: nq-250k 250,000 docs /
  3,452 q; hotpotqa 5,233,329 / 7,405; cqadup-programmers 32,176 / 876; cqadup-physics
  38,316 / 1,039; heldout-train 6,169,142 / 7,325; heldout-longq 6,169,142 / 55.
  Peak VmHWM 4.60 GiB, 22.4 s. `pytest m17src/test_evaluate.py -q` → 20 passed.
- `dev_suite_read` has never been executed.

## Files you may open (no others; no `grep -r`, `rg`, or `find` over the repository)

- `m17src/evaluate.py`, `m17src/test_evaluate.py`, `m17src/loader_np.py`, `m17src/export.py`,
  `m17src/common.py`, `m17src/lock.py` (for `require_executable`)
- `m17/registry.json` (entries `screen_routing_surface`, `untrained_vocab_export_v0`,
  `training.decision_protocol`, the lock block), `m17/CODEMAP.md`, `m17/STATUS.md`
- `results/m7_dev_manifest.json`
- `m7src/devsuite.py`, `m7src/heldout.py`, `m7src/evalkit.py`, `m7src/hashing.py`
- `git show 2c5321c` and `git diff 3ff9dbf 2c5321c`
- You may run `.venv/bin/python -m pytest m17src/test_evaluate.py -q` (writes only to pytest tmp).

## Read-exclusions (absolute)

- Never open `results/frozen_eval/untouched-*`, any reserved qrels cache, `work/m9reserve`, or
  anything concerning FEVER, DBpedia-entity, cqadup-android, cqadup-english.
- Never read anything under `work/m17/prepared/full` or `work/m17/bundles/V0` as a quality
  surface, and never execute `dev_suite_read`, `--surface dev-suite`, or any command that scores
  an M17 artifact on the development suite. V0 has exactly one declared read and it is not yours.

## Questions to break

1. Does the surface the reader produces equal the registered `screen_routing_surface` exactly:
   the six pinned components, int8 folded artifacts through the released QueryTable path, exact
   dense retrieval, per-domain nDCG@10 / Recall@10 with the equal-weight domain macro? Any
   silent change of metric, cutoff, document set, query set, dedupe, or tie handling versus
   what M7/M8 registered for this suite?
2. Can the gate be bypassed (rehearsal flag, env var, default argument, CLI path that skips
   `require_executable`), or does it fail open on a missing/altered registry?
3. Does the reader consume the V0 bundle through the same code path the released loader uses
   (`loader_np.M17QueryEncoder`, int8 resident rows), or could it score a float or differently
   folded table and mis-state V0?
4. Does the manifest hash verification actually bind what is scored (documents, queries, qrels),
   or only some of them? Could a stale `work/dev` cache or teacher encode cache pass the check
   with different vectors?
5. Is the held-out corpus (6.17M shared pool through `PoolReader`) the same corpus M7/M8 scored
   held-out against, with the same document ids and order-independence? Anything in
   `evaluate.search`'s tie/dedupe logic that changes results on this corpus?
6. Will the one read's result JSON (`results/m17_v0_read.json`, `reads: 1`) capture enough
   provenance (bundle hash, manifest hashes, registry status, git sha, per-domain numbers) that it
   can never be confused with a later read?
7. Do the tests prove any of the above, or only the synthetic fixture's own consistency?

## Output

Findings only, ranked P1/P2/P3 with file:line, a one-sentence failure scenario each and the
smallest fix. State explicitly whether anything blocks the V0 read. Under 60 lines.
