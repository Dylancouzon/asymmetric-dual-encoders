# Experimental harness

Reusable components remain in their original directories so historical artifacts keep their
meaning. Start with `CLAUDE.md`, `ROADMAP.md` and the active milestone's `CODEMAP.md`.

| Component | Reuse |
|---|---|
| `m7src/` | Frozen teacher/index, encode caches, lookup tables, retrieval, fusion and statistics |
| `m8src/`, `m9src/` | Access guards, student baseline, registered decision/statistics helpers |
| `m10src/` | Corpus admission/packing, target cache, student, resumable training and screen arms |
| `m12src/qfusion.py` | Qdrant RRF/DBSF operators; vendored reference is test-only |
| `m11/release/` | Standalone zero encoder bundle and release verification |
| `m17src/` | Zero v1.1 candidate cache, vocabulary extension, the one training driver, bundle export/gates, numpy loader comparison and evaluation |

## Checks

```bash
./run_checks.sh       # M10 pytest, M9 statistics, M12 fusion parity, M13 and M17 pytest
./run_tests.sh        # legacy M7, including machine-local cache/GPU checks
./run_m8_tests.sh     # historical M8 checks; prints its remaining coverage gaps
```

M17 also has two direct commands:

```bash
.venv/bin/python -m pytest -q -ra m17src            # cache, vocabulary, driver, gates, loader, eval
.venv/bin/python m17src/rehearse17.py --device cuda # tiny synthetic end-to-end rehearsal
```

The rehearsal writes disposable outputs to gitignored `work/m17/rehearsal` and a small record to
`results/m17_rehearsal.json`; it uses synthetic fixtures only and never reads a development
component, the M17 panel or any protected surface. `m17src/train.py` refuses a real arm while
`m17/registry.json` is `DRAFT_NOT_EXECUTABLE`; `--rehearsal` is the only bypass and announces
itself. The M17 gates in `m17src/export.py` are separate from M11's, which remain bound to
`m7/FREEZE.json` and are not weakened by them.

The active runner uses `.venv/bin/python`, two CPU threads and offline Hugging Face settings.
It reports every suite and returns nonzero if any fails. M10 needs locally cached
`BAAI/bge-small-en-v1.5` weights/tokenizer and the `hf-internal-testing/tiny-random-BertModel`
configuration; missing packages/models fail visibly. Optional local cache checks may report skips.
The active suite uses synthetic fixtures for guarded evaluation and does not execute a final run.
Legacy M7 calibration checks rewrite their own result JSONs; review the diff before committing.

## Environment and artifacts

Use the working `.venv` for this checkout. `m7/requirements.lock.txt` records the historical M7
environment; it is not a validated fresh-install recipe for today's complete harness. The current
Linux checks run with Python 3.12, torch 2.8, transformers 4.57 and pytest 9; Stella's remote code
requires the documented transformers compatibility (`m10/CODEMAP.md`). A portable environment
bootstrap remains repository maintenance, not a reason to delay the closed preparation milestone.

Git tracks code, registrations, summaries and provenance. Gitignored `work/` holds model weights,
tokens, target/encode caches and checkpoints: a clone alone cannot replay the trained experiments.
Restore or rebuild the specific admitted artifacts named by the relevant registry/manifests,
verify their identities, and check the actual hardware before running an arm. Do not rebuild
`results/perquery.json`, the irreplaceable frozen comparator vectors.

`m10src/run_arm.py` is the registered screen runner, not a general build/final-evaluation command.
M13 still owes the cloud/build driver and guarded final-evaluation executor with rehearsal.
Passing unit checks does not certify either unwritten execution path. Keep registrations and
guards when reusing components; never read reserved qrels, `work/m9reserve` or
`results/frozen_eval/untouched-*` outside the registered evaluation transaction.
