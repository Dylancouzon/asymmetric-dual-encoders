# M13 code map

New code lives in `m13src/` and imports `m7src`, `m9src` and `m10src` without changing registered
behavior. `m10/CODEMAP.md` and its pitfalls still apply. Design: `m13/STAGE1_DESIGN.md`.

| module | what | artifact |
|---|---|---|
| `build13.py` | the 200M build controller: the 200M pin, the batch from E1, cycles 1-3 as ONE `trainer10.train_arm` schedule, the extension/plateau/kill loop under the day-one cap and the $1,000 ceiling, a wall-clock rolling checkpoint and per-phase `losses*.jsonl`, resume with `run_arm`'s semantics, and the freeze (DEV-6, ONNX, ORT + fastembed parity, provenance). `--plan` / `--benchmark --rate --price` / `--resume` / `--smoke-steps` | `work/m13build/<arm>/`, `results/m13_build_record.json` |
| `m13/build_config.json` · `build_lock.py` | the build arm (every data knob equal to ANCHOR's except dose, batch, document policy and cut) and its validator against the LIVE screen registry: the anchor comparison field by field, the E1 batch, the cycle arithmetic, the extension floor and the cap formula | — |
| `m13/LOTTE_GATE_REGISTRATION.json` | ruling R8: LoTTE-clean's seven remediated slices, nDCG@10 veto metric, Success@5 descriptive, veto constants, identities and the gate-record contract; written while LoTTE is unread. NOT the gate record `build13` requires | — |
| `m13/PAIRED_ROW.md` | ruling R4: the descriptive paired M9/nano per-query row, every recipe difference listed, no decision attached | — |
| `m13/RULINGS.md` | Dylan's rulings R1–R12 with the recommendations they accepted (2026-09-10) | — |
| `test_build13.py` | the controller's checks: cycle arithmetic at bs32/bs128, the refusals, extension accept/reject/cap/budget, a kill at an extension end, SIGKILL-and-resume equivalence, and the document policy | — |
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
10. A SMOKE must scale the EXTENSION too. The first box smoke of `build13.py` cleared the gain bar at 20 steps and started a 2,084,375-step (30 GPU-hour) extension cycle, because only the dose was smoke-scaled.
11. Export runs on the CPU. `nano10.export_onnx` builds its trace inputs on the CPU, so freezing a CUDA-resident model raises "Expected all tensors to be on the same device" — after the build has finished.
12. `trainer10.train_arm(loss_log=...)` TRUNCATES the sidecar it is given on a fresh run (so a resume cannot double-log). One shared file across phases therefore erases cycles 1-3's losses when extension 1 starts; each phase gets its own.
