# M7 status

**Stage:** bring-up complete → Stage 0 (representation compatibility) in flight
**Updated:** 2026-08-25

## Machine

RTX 3080, **10 GB VRAM** (not 12) · 25 GB RAM · 16 cores · 946 GB free ext4 · nvcc 12.6 →
torch 2.8.0+cu126. Teacher **BAAI/bge-base-en-v1.5 @ a5beb1e3e68b9ab74eb54cfd186867f64f240e1a**.
Encode throughput: 891 texts/s fp16 (2.4x fp32, cosine-identical) → `results/m7_throughput.json`.

## Done

- Bring-up steps 1–4 all green. Harness re-validated on this box: bge-small ArguAna 0.6038,
  SciFact 0.7127, bm25 FiQA 0.2532 — all inside 0.003 (`results/m7_harness_validation.json`).
- All six datasets re-downloaded and hash-matched to `results/eval_manifest.json`
  (`scripts/verify_manifest.py`); `scripts/validate_perquery.py` OK, 54 cells.
- Partition ledger written (`m7/LEDGER.md`), with source-level license evidence per dev and
  untouched-final set. **Climate-FEVER dropped** — no affirmative license at any primary source.
- Dev suite pinned and hashed: nq-250k (250,000/3,452), hotpotqa (5,233,329/7,405),
  cqadup-programmers (32,176/876), cqadup-physics (38,316/1,039) → `results/m7_dev_manifest.json`.
- Training mix built from approved sources only: hotpotqa-train 85,000q/170,000pos ·
  fever-train 109,810/140,085 · squad-train 87,599 · esci-us 74,888/988,062 (+122,273 hard negs) ·
  mrtydi-en 3,547 (+104,854 hard negs) · query-text-only nqopen 87,925 + triviaqa 138,384.
- **Conformance suite passes 24/24** (`m7src/test_conformance.py`) — the mandated gate before
  the first training run: special tokens, padding, multiplicity, truncation, empty/degenerate
  queries, byte-for-byte prefix, double-application refusal, batch invariance, int8 round-trip.

## Running

- Teacher encode of the dev HotpotQA corpus (5.23M docs), shard ~50/105.
- Fingerprint decontamination (TRAIN ↔ six / dev / untouched-final), indexing the six's 272,117 docs.
- Dev reference rows. So far: bm25 nq-250k 0.5804 · potion-32M nq-250k 0.5479 (a 250K-doc
  subsample is an easier corpus than full BEIR NQ — dev numbers are not comparable to BEIR rows).

## Next step

1. Finish decontamination; log removal counts.
2. Build the frozen doc-vector pool (~6.2M vectors, 9.5 GB fp16); hotpotqa reuses the dev encode.
3. **Stage 0.1** — closed-form ridge table (`m7src/stage0_ridge.py`): the MSE-optimal
   flat-weight bag-of-tokens approximation of the teacher's query encoder. One linear solve,
   and an exact upper bound on what flat distillation can reach. This is the cheapest possible
   answer to the central structural question.
4. **Stage 0.2** — gradient-trained distilled table (objective B), then the capacity probe.
5. Go/no-go gate on dev.

## Open blockers

None. Nothing needed from Dylan yet; the HF release go is the only pending item and is far off.

## Notes for the next session

- Never `git add -A` without checking `.gitignore` first: an early one committed the multi-GB
  encode cache and made `git push` hang. `work/` and `logs/` are now ignored; history cleaned.
- Subagents must be told to write scratch files to the scratchpad, not the repo.
