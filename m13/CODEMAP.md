# M13 code map

New code lives in `m13src/` and imports `m7src`, `m9src` and `m10src` without changing registered
behavior. `m10/CODEMAP.md` and its pitfalls still apply. Design: `m13/STAGE1_DESIGN.md`.

| module | what | artifact |
|---|---|---|
| `build13.py` (planned) | the 200M build controller: 200M pin, batch from E1, extension cycles and cap, kill/plateau, provenance, ONNX + parity at freeze | `work/m13build/<run>/`, `results/m13_build_record.json` |
| `m13/build_config.json` (planned) | the build arm: every data knob equal to ANCHOR's except dose, batch, document policy and cut; validated before any run | — |
| `access13.py` | `m9src/final9.py`'s access machinery retargeted: one `Config` object, flock, fail-closed spent tag with a pinned origin, ledger, atomic writes, **fatal** `seal_protected_paths`, manifest-only preflight | `results/m10_final_run.json`, tag `m10-six-spent` |
| `score13.py` | the six-set scoring transaction for a nano or M9 checkpoint: frozen document caches, exact search, per-query nDCG@10 on the frozen qids, dataset-mean bridge, `final10` decision layer, post-tag continuation, `--recover`, the conditional reserved stage | `results/m10_final_scores/<ds>.json` beside the run record |
| `rehearse13.py` | the synthetic six-set, fake comparator, real encode cache and bare origin that let the real executor run end to end with zero protected access | `work/m13rehearse/` |
| `test_access13.py` · `test_score13.py` · `conftest.py` | the executor's checks, all on the fixture; in `./run_checks.sh` | — |

## Pitfalls carried into M13

1. `corpus_loader.arm_doc_count` scales unique documents with dose; at 200M it asks for 50M documents from a ~6.15M pool. The build needs the registered document-repetition policy, not a bigger draw.
2. `nano10.lr_at`'s default `warmup` is 2,000 STEPS; every caller passes `warmup=` explicitly (the trainer does), an extension cycle passes 0.
3. `bench/core.py:DATASETS` defaults to five datasets; `m7src/_paths` sets six only if imported first. Iterate `partitions.all6`.
4. `pytrec_eval` silently omits a run qid without qrels; assert the scored qid set against the frozen list every time.
5. `m9base` installs `paths_guard`; import the scorer lazily from any training path (pitfall 14).
6. `paths_guard.claim()` verifies the caller IS the module the allowlist entry names, so `score13`
   cannot borrow `m8src.final_run`'s entry. The conditional reserved batch therefore needs its own
   ALLOWLIST entry — a LEDGER 15 amendment — on top of the unpriced encodes (R10).
7. `access13.sh` strips its output; `git status --porcelain` puts a SPACE in column 1, so the drift
   parser uses `sh_raw`. Stripping shifted every parsed path by one character.
8. A frozen payload read from JSON can never carry int qids (keys are strings by construction).
   The int-vs-string hazard is real in `perquery.json`'s qid LIST and in a scored dict, and is
   checked there.
9. `final10.REGISTRY` and `teacher.ENC` are module-level paths; the rehearsal rebinds them from
   `Config` rather than copying either module.
