# M7 protocol ledger

Append-only. Records the partition ledger, the freeze record, every six-set and
untouched-final access, and crash re-runs.

## Bring-up

- 2026-08-25 — Machine confirmed: RTX 3080 10 GB VRAM, 25 GB RAM, 16 cores, 946 GB free ext4, nvcc 12.6.

## Bring-up (continued)

- 2026-08-25 — Env: Python 3.12.14 venv, torch 2.8.0+cu126 (CUDA available, RTX 3080),
  transformers 4.57.6, datasets 5.0.1, pytrec-eval-terrier 0.5.10. Lock: `m7/requirements.lock.txt`.
- 2026-08-25 — `scripts/validate_perquery.py`: OK, 54 cells (4 allowlisted per FINAL_MATRIX.md).
- 2026-08-25 — `scripts/verify_manifest.py` (new): all six datasets re-downloaded from HF and
  matched to `results/eval_manifest.json` on n_docs/n_queries/corpus_ids/corpus_text/qids/qrels,
  and `results/frozen_eval/` matched to the fresh download. Frozen comparator pairing is valid.
- 2026-08-25 — **SIX-SET ACCESS, class (a) harness validation** (`m7src/validate_harness.py`,
  `results/m7_harness_validation.json`): bge-small ArguAna 0.6038 (want 0.6034, +0.0004);
  bge-small SciFact 0.7127 (0.0000); bm25 FiQA 0.2532 (-0.0000). All within the 0.003 standard.
  No new-model number was scored against six-set qrels in this access.
