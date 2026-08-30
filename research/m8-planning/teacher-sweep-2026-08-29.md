# M8 teacher-candidate sweep (Sonnet, 2026-08-29) + orchestrator verification

*Delta over `research/m7-teacher-shortlist-2026-08-26.md`. Sweep filters: derived-weight-releasable
licence, relaxed vendor rule, quality plausibly ≥ stella-400M, table byte arithmetic under the E7
cap, ONNX status, 3080-feasible. The sweep's headline contamination finding was verified by the
orchestrator the same day (see the addendum at the end, which corrects its epistemic status).*

## Probe-next (ranked)

1. **NovaSearch/stella_en_1.5B_v5** — MIT, clean vendor, same lineage as the incumbent, GTE-Qwen2-
   1.5B-instruct backbone, hidden 1536. Vocab **151,646 Qwen2 BBPE** — breaks the shared-WordPiece
   property of every previously probed candidate: row indexing and decontamination fingerprints
   must be rebuilt if it wins. MRL floor 512 → table floor ≈ 77.6 MB int8. ONNX tag present,
   unverified. Card shows FEVER 94.8 nDCG@10 (consistent with in-domain training — see addendum).
2. **microsoft/harrier-oss-v1-0.6b** — the one genuinely new family clearing every hard filter.
   MIT (base Qwen3 = Apache-2.0, legitimate relicense), Microsoft = OK-with-justification, vocab
   151,936 × 1024 → 155.6 MB int8, official onnx-community export exists, decoder-only last-token
   pooling = a different inductive bias worth one closed-form probe. Training data undisclosed —
   contamination black box; NEEDS-RULING before any adoption.
3. *(completeness only)* Qwen/Qwen3-Embedding-0.6B — unchanged OUT: dominated on the v1 anchor
   scale; still no published MRL dimension-vs-quality curve anywhere (re-checked).

## Confirmed OUT (new information only)

- **microsoft/harrier-oss-v1-270m / -27b — licence laundering flag**: shipped as "MIT" but
  `config.json` says `gemma3_text` with the stock Gemma3 262,144 vocab, i.e. Gemma-3 derivatives;
  Gemma Terms flow down to derivatives and cannot be relicensed permissively. OUT under our
  "no Gemma terms" rule regardless of Microsoft's label. (270m also fails byte arithmetic:
  262,144 × 640 int8 = 168 MB with no MRL.)
- BAAI/bge-en-icl (7B Mistral, needs few-shot query prompting — wrong shape and too big);
  bge-multilingual-gemma2 (Gemma terms + 9B); NV-Embed-v2 (CC-BY-NC confirmed); Qwen3-Embedding-4B
  (388 MB table arithmetic); arctic-embed-v2 family (250K vocab, MRL truncates dim not vocab);
  no new BAAI English dense retriever since bge-en-icl (2024); no new intfloat release.

## Unblocked incumbent-family rows (solver fix, no new info needed)

granite-embedding-english-r2 (50,368 vocab, 768d, official ONNX incl. int8) and gte-modernbert-base
(50,368, 768d, official onnx/ with fp16/int8/q4) — both confirmed unchanged and below the incumbent
on the v1 anchor, BUT they were never probed on the table criterion (float64-Gram memory limit);
the M8 CG solver makes them probeable. Expectation per the base-out-approximates-large rule is they
still lose; run them as cheap controls of the CG frame.

## Orchestrator verification addendum (same day) — the FEVER finding, corrected

The sweep reported "stella's MTEB registry entry lists NQ/HotpotQA/FEVER/MSMARCO/ArguAna/FiQA".
Verified against `mteb/models/model_implementations/stella_models.py` @ main: stella's entries set
`training_datasets = nvidia_training_datasets` with the comment *"also distilled from gte-qwen (but
training data is unknown)"*, and `nvidia_models.py` defines that list (source: NV-Retriever, arXiv
2405.17428) including **ArguAna, HotpotQA, MSMARCO, NQ, FEVER, FiQA2018** (+ hard-negative and Nano
variants).

Epistemic status, precisely: this is a **community-assigned PROXY list** (NVIDIA's), applied to
stella because stella's own data was never disclosed and its distillation source (gte-Qwen) is also
undisclosed. It is NOT an author disclosure. However: (a) M7's teacher decision already treated
this same registry entry as stella's disclosure for ArguAna/FiQA — consistency requires treating
FEVER/HotpotQA/NQ identically; (b) stella-1.5B's card-reported FEVER 94.8 nDCG@10 independently
suggests in-domain FEVER training in the lineage.

**Protocol consequence for M8 (registered as decision/disclosure item E9 in the plan):** FEVER is
one of the four reserved confirmatory sets. The paired M8-vs-M7 legs (C1, C2) share the teacher, so
teacher contamination largely cancels in the delta; the BM25 floor (C3) and any absolute FEVER
claim are teacher-flattered. Pre-register, before any M8 number: the proxy-disclosure caveat at the
FEVER row, and a FEVER-excluded sensitivity read of all three legs (the reserved-4 analogue of
M7's clean-4). Also binds HotpotQA/NQ dev components (already train-adjacent — unchanged) and any
future teacher probe's contamination column (use the registry proxy list consistently).
