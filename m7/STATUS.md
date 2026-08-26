# M7 status

**Stage:** data pipeline complete → Stage 0 (representation compatibility) next
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

## Decontamination results (`results/m7_decontam.json`)

353,519 TRAIN pairs → **352,145 kept**. R1 (query overlap) removed 1,329; R2 (positive document
matches one of the six) removed 45, from just 23 of 855,324 unique positives — a 3e-05 rate, so
the source-level contamination map was already doing the work. Query-text-only sources: nq-open
−213, TriviaQA −155.

**R3 finding that changes the report's framing.** DBpedia-entity was the intended clean
generalization probe. It is not clean: **9.32%** of TRAIN positives near-duplicate one of its
documents (15,523 exact + 79,595 near). The cause is structural — DBpedia abstracts are Wikipedia
lead paragraphs, and so are HotpotQA's documents and SQuAD's contexts. Combined with
Climate-FEVER being dropped for licensing and BEIR FEVER sharing its corpus with fever-train,
**the untouched-final partition now has no clean member.** Both rows will be reported with their
overlap rate attached and neither presented as an uncontaminated generalization number.

## Running

`./run_stage0.sh` — **strictly sequential**, 7 steps, log in `logs/stage0.log`:
1. dev corpus encodes (resuming; HotpotQA shard ~97/105) 2. asset freeze 3. decontamination
4. query-text decontamination 5. frozen doc-vector pool (~6.2M vectors, 9.5 GB fp16)
6. dev reference rows on all four components 7. Stage-0 ridge probe.

Dev reference rows so far (three fast components; a 250K-doc NQ subsample is an easier corpus
than full BEIR NQ, so dev numbers are not comparable to BEIR rows):

| system | nq-250k | cqa-prog | cqa-phys | macro-3 |
|---|---|---|---|---|
| bge-base symmetric, prefixed (teacher ceiling) | 0.8198 | 0.4240 | 0.4727 | **0.5722** |
| bge-base symmetric, bare | 0.7982 | 0.4020 | 0.4608 | 0.5537 |
| bm25 | 0.5804 | 0.2975 | 0.3471 | **0.4083** |
| potion-retrieval-32M | 0.5479 | 0.2261 | 0.2835 | **0.3525** |

So the table has to hold ~71% of the teacher to clear BM25 on dev, and ~62% to clear potion.

## Next step

1. **Stage 0.1** — closed-form ridge table (`m7src/stage0_ridge.py`): the MSE-optimal
   flat-weight bag-of-tokens approximation of the teacher's query encoder, from one linear
   solve. It is an exact upper bound on what flat distillation can reach, so it is the cheapest
   possible answer to the central structural question.
2. **Stage 0.2** — gradient-trained distilled table (objective B), then the capacity probe.
3. Go/no-go gate on dev, then the phased program in `m7src/program.py`.

Everything downstream is already written and imports clean: training (objectives A/B/C),
bootstrap + Holm, fusion, gate, sweep driver, costs, ANN sweep on real HNSW, the
two-collection edge demo, and the guarded final-run script — the last written *before* any
candidate result exists, so its freeze rules cannot have been shaped by the numbers.

## Open blockers

None blocking. Two things for Dylan to know, neither urgent:

1. **`qdrant-edge` is not installable from PyPI on this box** (`uv pip install qdrant-edge` →
   not found). M5's Edge prototype ran on the Mac, so it came from an internal index or a local
   wheel. Needed only for the last deliverable (the Edge demo running our table).
   **Ask: is there a wheel or index URL for qdrant-edge?** Not blocking: the standalone Qdrant
   v1.19.0 server binary is installed at `~/qdrant-bin/qdrant` (no Docker needed — Docker
   Desktop's WSL integration is off for this distro, and it turned out not to matter), which
   gives real HNSW for the ANN sweep and can host the two-collection architecture. Only the Edge
   *shard format* would be missing, and the report would say so.
2. **WSL is configured with ~25 GB of the host's 32 GB**, leaving Windows ~6 GB.
   `setup-windows.md` suggested starting at 20 GB. Higher is useful here, but it is why the
   concurrent-job OOM took the whole distro down rather than one process. Jobs are now strictly
   sequential and peak is budgeted under ~18 GB, so no change is needed.

The HF release go remains Dylan's and is still far off.

## Notes for the next session

- Never `git add -A` without checking `.gitignore` first: an early one committed the multi-GB
  encode cache and made `git push` hang. `work/` and `logs/` are now ignored; history cleaned.
- Subagents must be told to write scratch files to the scratchpad, not the repo.
