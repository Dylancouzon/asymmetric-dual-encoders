# M9 — LEAF-style distilled query tower (tiny mandate)

*Renumbered from M8 on 2026-08-28 when the M7 learnings follow-up became M8 (v2). Brought up to
date at the same time: the original draft named bge-base-en-v1.5 as teacher and justified the
default student by tokenizer identity with it — both stale since the 2026-08-26 teacher swap.*

Run after M8, same box, same session rules. M9 builds the other point on the spectrum: a small
transformer query encoder distilled to search a frozen document index — MongoDB's LEAF technique,
with a clean-vendor teacher.

Everything binds from `instructions-m7.md` unchanged: decision authority, licensing and
decontamination rules, dev-only selection, frozen comparator pairing, the freeze/ledger protocol
for one final run, Sonnet subagents for research, and the headless git contract — working files
under `m9/`, same four-file split.

What changes:

- **Teacher:** the frozen teacher the shipping table line uses — **`NovaSearch/stella_en_400M_v5`**
  (M7's teacher; M8 inherits it unless its own ledger records a swap, in which case M9 follows M8).
  Docs are indexed with the teacher; the student is distilled into its query space: embedding
  alignment plus ranking preservation (M7's objective B), then optional contrastive finetune (C).
  No new data, doc vectors, dev suite, or conformance tests — reuse M7's.
- **Student:** a ≤35M transformer from a permissive clean-vendor backbone. The old default
  (bge-small-en-v1.5, MIT) was justified by sharing bge-base's tokenizer; **stella's tokenizer is
  not bge-small's, so that rationale is void** — the student must re-tokenize and learn the
  alignment through its own vocabulary, which LEAF itself demonstrates across mismatched
  tokenizers. Re-derive the student shortlist at M9 start (licence-clean, ≤35M, quality on MTEB);
  bge-small remains a candidate, not the default.
- **Bars, paired on the frozen vectors** (`results/perquery.json` carries leaf-ir-asym 0.5155,
  mdbr-leaf-ir 0.5123, arctic-embed-m-v1.5 0.5264, bge-small-en-v1.5 0.5042, frozen 2026-08-25):
  **release = CI-resolved above bge-small symmetric (0.5042)** — an asymmetric student is only
  worth shipping if it beats simply running the small model on both sides. Reference rows, not
  gates: leaf-ir-asym (MongoDB's model, on the Snowflake teacher) and the stella symmetric ceiling
  from M7's final run; report retention vs teacher next to LEAF's published 97.1–98.6%.
- **Costs:** expect ~3–5 ms per query plus a model load. Report the same three cost numbers so
  M7, M8 and M9 land in one frontier table.

Deliverables: weights on HF under the Qdrant org (Dylan's explicit go), a section added to the M7
report artifact (not a separate report), the frontier table updated, decisions logged in CLAUDE.md.
