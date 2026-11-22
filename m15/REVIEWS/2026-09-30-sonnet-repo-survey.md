# Sonnet research agent (build recipe, harness reuse, local artifacts)

Date 2026-09-30. Access: named-file allowlist, no work/, no recursive results/ search. Returned text below, verbatim. Dispositions are in `m15/LOG.md` (2026-09-30) and `m15/PLAN.md`.

---

I read the docs and code and listed non-protected directories only. I did not open `work/`, `results/frozen_eval/untouched-*`, or any reserved queries or qrels. Dollar and hour figures marked "(computed)" are my arithmetic from cited numbers.

## A. Build recipe and cost

**Zero (lookup table).** It is trained, not closed-form.
- **Table and init:** the table is 30,522 x 1024, stored as int8 with one fp32 scale per row (`m7/RECIPE.md:14-25`). Rows are initialised by forwarding each vocab token through the tower inside a query context, 30,522 forward passes (`m7/RECIPE.md:52-54`, `m7src/init_table.py:1-9`).
- **Loss:** phase B (16,000 steps, batch 512) uses cosine to the tower's query vector plus a KL ranking loss over the top-32 similarities against frozen document vectors. Phase A (2,500 steps) adds InfoNCE against a 2M-vector negative bank (`m7/RECIPE.md:46-91`, `m7src/train.py:4-11`).
- **Serving rule:** query pooling weight is sqrt(count) (`m7/RECIPE.md:23-24`).
- **Data:** 340,850 query-document pairs plus 220,632 query-only rows from ESCI, FEVER-train, HotpotQA, Mr. TyDi, SQuAD, NQ-open and TriviaQA. MS MARCO is excluded (`m7/RECIPE.md:35-39`). A pool of 924,704 pseudo-queries is also used (`m7/RECIPE.md:56-58`).
- **Extra input the recipe needs:** a 6.17M-document pool encoded by the tower. `m7src/train.py` refuses to run without a decontamination mask (`m7/RECIPE.md:43-44`).
- **Compute:** RTX 3080, 10 GB, so $0 cloud (`m7/RECIPE.md:143`). Retraining is "~20 minutes" (`m7/LEDGER.md:228`). Re-encoding the pool, dev corpora and query targets for a new tower is 8-12 h (`m7/LEDGER.md:226-228,253`). Total wall-clock: not found.
- **Card mismatch:** the Zero card says "L2 regression" (`m11/release/MODEL_CARD.md:184`), but the recipe is cosine + KL + InfoNCE.
- **Tower portability:** the tower registry (`M7_ENCODER`) already swaps towers (`m7src/encoders.py:1-15`). The 7-tower study shows Zero retention varies from 43% to 72% by tower (`research/constella-in-plain-english.md:139-144`).

**Nano (34,540,672 parameters).**
- **Architecture:**
  - It uses the pinned bge-small backbone. Hidden states from layers 12, 8 and 4 are concatenated (1152-d), passed through a per-token linear head to 1024-d, mean-pooled and normalised (`m10src/nano10.py:1-30`, `m14/MODEL_CARD.md:119-123`).
  - The head is warm-started by ridge regression (60,000 samples, lambda 0.001), taking 74.2 s.
  - The loss is squared L2 to the tower's vector (`results/m13_build_record.json`: `warm_start`, `recipe.objective`).
- **Schedule:** batch 32, 6,250,000 steps, 3 cycles, peak LR 1e-4, final LR 1e-5 (same file, `schedule`). The dose is 199,999,721 examples in a 75% query / 25% document mix (`training.mix`).
- **Query sources:**
  - 3,486,034 query rows in total (same file, `assemble_manifest.query`).
  - M9 pool: ESCI, HotpotQA, SQuAD, Mr. TyDi, NQ-open and TriviaQA, 462,605 rows.
  - PAQ: 1.0M rows, CC BY-SA 3.0.
  - Harvested real text (claims, titles, keywords): 1,248,386 rows.
  - Generated: 834,463 rows across 7 forms, made locally with Qwen3-8B-AWQ (`m10src/gen.py:3`).
  - MS MARCO is used for validation only (`m14/MODEL_CARD.md:171-176`).
- **Documents in the mix:** 6,070,049 passages, given to the student with a `passage: ` prefix.
- **How targets were made:**
  - Query targets are the tower's prompted query encodings, cached by content hash in an fp16 memmap (`m10src/targets10.py:1-30`).
  - Document rows in M9 target the tower's unprompted document path (`m9src/longrun.py:286-289`). That the M13 build did the same is my inference from its `m9-pool` source.
- **Compute:**
  - The build ran on one A100-SXM4-80GB for 206,171.92 s training (57.27 h) at 970.1 examples/s (`results/m13_build_record.json`: `training`, `environment.gpu`).
  - The rate was $1.5936/h GPU, $1.6636/h with storage (`m13/CLOUD_READINESS.md:20,43`). Training alone is about $95 (computed).
  - The projected cost was 52.71 h and about $87.69 (`m13/BUDGET_EXPLAINED.md:19-20`).
  - The prose says "about $150 after substantial local precomputation" (`research/constella-in-plain-english.md:40-41`).
  - The admission allowance was $721.36 (`m13/BUDGET_EXPLAINED.md:17`), under the $1,000 ceiling.
  - An exact total spend figure: not found.

**Teacher target cost.**
- Measured Stella query-encode rates on the RTX 3080 are 345 texts/s on harvested text and 484 on PAQ (`m10/CODEMAP.md:57`). The 3.49M query rows are therefore roughly 2-3 h (computed).
- The A100 document-encode rate was 100.76-110.81 passages/s on a SQuAD surrogate (`m13/EXECUTION.md:83-86`).
- The real RTX 3080 corpora ran at 130-215 docs/s (`m20/STATUS.md:49-51`).
- The document pool was encoded once and reused (`m7src/pool.py:6-8`).

## B. Harness reusability

| Step | Path (LOC) | Coupling | Tests |
|---|---|---|---|
| 1 Table build | `m7src/train.py` (787), `table.py` (406), `init_table.py` (205), `teacher.py` (392), `encoders.py` (248), `pool.py` (152) | The tower is a registry `Spec`. All registered towers share one BERT WordPiece vocab, and `table.py:27` hardcodes `CLS_ID=101` (`encoders.py:63,68`). `train.py` imports `dev_eval`, `mix` and `decontam`, all pinned to `results/m7_*` and `work/`. Export is `m11/release/export_onnx.py` (377) plus `zero_encoder.py` (93). | Conformance, encoders, encode-cache and init-rows suites (`run_tests.sh`). No `train.py` test. |
| 2 Nano distillation | `m10src/nano10.py` (664), `trainer10.py` (341), `targets10.py` (354), `data10.py` (182) | The model, trainer and target cache are generic. `corpus_loader.py` (1615) and `run_arm.py` (1184) are bound to `m10/screen_registry.json`. The trainer takes a caller-supplied `batch_fn` (`trainer10.py:3`). | `test_nano10`, `test_trainer10` (resume equivalence), `test_targets10` |
| 3 Exact-search eval and stats | `bench/core.py` (137), `m7src/evalkit.py` (78), `m7src/boot.py` (368: `signflip_dep`, `paired_dep`, `holm`), `m7src/retention.py` (79) | `core.py` loads BEIR from HF and keeps a protected-dataset audit trail. `retention.py:27` hardcodes `stella-400M-v5-symmetric`. `m9src/final_stats.py`, `m10src/contrasts.py` and `m20src/roster.py` + `beir15.py` are registry-bound and protected-access-bound. | `test_dep_stats`, `test_signflip_*` |
| 4 Serving protocol | `scripts/m13_serving_costs.py` (185) | Hardcodes `/home/dylan/...` (`:23`), fixed `work/` asset paths, `/proc/cpuinfo` (`:162`, Linux only) and three fixed models. The worker logic is generic. | none |
| 5 Qdrant/Edge latency | `bench/edge_prototype_pair.py` (609), `m9src/edge_cost.py` (232), `bench/edge_ann_sweep.py` (71), `m7src/ann_sweep.py` (169), `m12src/qfusion.py` (89) | The prototype hardcodes `work/m9onnx/nano-minilm-l6` and `edge_data/m9_pair` (`:37-42`). | Only `qfusion` |

**Smallest kit for steps 1-4 on a new tower.** This is my estimate, not a repo claim.
- **Lift (about 4,500 LOC):** `encoders`, `teacher`, `table`, `init_table`, `pool`, `train`, `nano10`, `trainer10`, `targets10`, `data10`, `evalkit`, `boot`, and the worker from `m13_serving_costs.py`.
- **Write new (about 600-1,000 LOC):**
  - A data loader replacing `dev_eval`, `mix` and `decontam`, with a small licensed query and document set.
  - A `batch_fn` for Nano.
  - A BEIR runner with paired bootstrap, sign-flip, and retention against the tower's own query path.
  - Path and CPU de-hardcoding for the serving script.
  - Tokenizer, vocab-size and CLS-id handling if the tower's tokenizer differs from BERT WordPiece 30,522.
- **Not needed for a kit:** `corpus_loader`, `run_arm`, the `m13src` access machinery, and the `m20src` roster and `beir15` runner.

## C. Local artifacts on this Mac

- **Committed Stella document vectors:** none. `artifacts/` and `edge_data/` are gitignored (`.gitignore:3-4`).
- **Local Stella document vectors (fp16, 1024-d, normalised):**
  - `artifacts/demo-stella/fiqa/doc_vecs.npy`: 57,638 vectors, 118 MB.
  - `artifacts/demo-stella/scifact/doc_vecs.npy`: 5,183 vectors, 10.6 MB.
  - Both are far short of 1M.
  - `edge_data/demo/fiqa/stella` is a 374 MB Qdrant Edge shard of the same FiQA vectors.
- **Where 1M+ vectors live:**
  - The M20 archive holds 142.6 GiB across 26 corpora, verified on `D:` of the RTX box, not on this Mac (`m20/STATUS.md:12-14,31-36`).
  - Public corpora in it: Quora 522,931 docs, Touche 382,545, NQ 2.68M (`m20/FINDINGS.md:88-93`).
  - The Mac cannot reach it without a transfer.
  - Encoding 1M documents locally on the M5 Pro at 20-100 docs/s takes about 3-14 h (computed from the `mac-stella-venv` memory note).
- **Model assets in `~/.cache/huggingface/hub`:**
  - Stella `NovaSearch/stella_en_400M_v5` at `ffeb2b7e`: 1.7 GB.
  - `BAAI/bge-small-en-v1.5` at `5c38ec7c`: 128 MB.
  - `constella-zero`: full bundle (`model.npz`, `model.onnx`, `model_tokens.onnx`, tokenizer), 90 MB.
  - `constella-nano` and `stella-en-400M-v5-doc-onnx`: snapshot dirs contain only `README.md`, so the weights are not cached. They are public on the Hub.
- **Venv:** `.venv-mac` has `qdrant_client` 1.19.0, `qdrant_edge_py` 0.8.0, `onnxruntime`, `pytrec_eval`, `bm25s`, `transformers` 4.57.6. It has no `fastembed`.
- **M5 Pro scripts:** `bench/edge_prototype_pair.py` and `m9src/edge_cost.py`, run as described in `m9/EDGE_PROTOTYPE_MAC.md` and `m9/EDGE_COST_MAC.md`.
- **Mac constraints:** one MPS job at a time, both MPS watermark env vars set, encode batch 64 (memory note `m5-pro-benchmarking-constraints`).

## D. Serving-cost evidence

- **Source:** `scripts/m13_serving_costs.py` wrote `results/m13_serving_costs.json` (script sha `84b2448a`, verified against the file).
- **Protocol:** 3 fresh processes per model, batch one, 4 threads, 5 warmups, 20 synthetic samples per length. The README table is the median of the 3 trials.
- **Hardware:** Intel i7-10700KF under WSL2 Linux, not the M5 Pro.
- **Runtime difference:** Zero ran on its NumPy encoder. Nano and bge-small ran on FastEmbed/ONNX (the `caveats` field in the JSON).
- **Stella query path:** never timed under this protocol. The script only accepts `zero`, `nano` and `bge-small` (`:169`).
  - Other timing evidence for it is throughput only: 345 and 484 texts/s on the RTX 3080 (`m10/CODEMAP.md:57`) and about 170 short queries/s on the Mac's MPS.
  - The document ONNX graph does reproduce the prompted query path (min-cos 1.0), so it could be timed (`m15/HOTSWAP.md`, on the `m15-whitepaper` branch).
- **M9 edge numbers** (`m9/RESULTS.md:121-260`):
  - Hardware is an Apple M5 Pro, 4 threads.
  - The index is **synthetic**: 1M random unit vectors, 1024-d (`bench/edge_prototype_pair.py:4-11`, `m9/RESULTS.md:137-138`).
  - Zero's table is synthetic too, and recall was not measured.
  - "Nano" in that prototype is the untrained MiniLM-L6 fp16 ONNX (`bench/edge_prototype_pair.py:41-42`), not the shipped bge-small-based Nano.
  - Rounds 1-3 used in-process Qdrant Edge shards. Round 4 used Docker Qdrant with memory limits: 3.387 ms Zero and 4.469 ms Nano at 256 MB, binary quantization, `rescore=false`.
- **Real-vector ANN evidence:** `results/ann_sweep.json` covers bge-small and LightRetriever only, not Zero or Nano (`m15/HOTSWAP.md`).

## E. Failures by query type, coverage, length

- **Dataset-level only** (`m20/STATUS.md:109-125`, nDCG@10). Zero beats Nano on all three claim-style or multi-hop sets:

| Dataset | Nano | Zero | Stella | bge-small |
|---|---|---|---|---|
| FEVER | 0.6231 | 0.6978 | 0.8207 | 0.8662 |
| climate-fever | 0.2473 | 0.2785 | 0.2907 | 0.3183 |
| HotpotQA | 0.6102 | 0.6127 | 0.6950 | 0.6993 |

- **Attribution:** the FEVER gap is left unattributed; the bge-over-Stella reversal also appears on HotpotQA and climate-fever (`m20/FINDINGS.md:68-72`, `m20/STATUS.md:140-143`).
- **Per-query-type breakdown:** not found. No analysis splits Nano or Zero failures by claim versus multi-hop, and none stratifies quality by query length.
- **Training coverage** (`results/m13_build_record.json`):
  - The `claim` form is 415,454 of 3,486,034 query rows (11.9%, computed).
  - HotpotQA-train contributes 81,743 queries.
  - The `fever-pos` document store is excluded from Nano's documents.
  - Family retention on the COV surface at build midpoint: BRIGHT 84.1%, consumer-health 95.0%, finance 87.3%, legal 96.9% (`m13/QUALITY_PROGRESS.md:7-13`).
  - DEV-6 retention: 0.9118 (`results/m13_build_record.json`, `freeze.dev6`).
- **First Nano (M9) coverage failure:** 93.8%, 71.0% and 50.1% retention on three slices. The docs say coverage and capacity are not separated (`m9/FINDINGS.md:15-24`, `CLAUDE.md`).
- **Zero retention:** 0.915 on dev, 0.764 out-of-domain, 0.755 on the six-set (`m7/STATUS.md:18-21`).
- **Query length:**
  - Only for Zero: tower agreement is flat from 16 to 256 words, measured as rank agreement, not quality (`m7/FINDINGS.md:44-47`).
  - Length effects on latency are measured, not quality (`m9/RESULTS.md:141-163`).
- **Zero on WordPiece-fragmented terms:** qualitative only (`research/constella-in-plain-english.md:176-178`). The Zero card says it is weak on word order and negation (`m11/release/MODEL_CARD.md:193-194`), but I found no measurement backing that.
