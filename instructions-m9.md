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

- **Teacher: M9 PICKS ITS OWN, on measurement — Dylan's ruling 2026-08-29.** *"M9 should go with
  whatever performs best, not as a strict continuation of M8."* The default remains
  **`NovaSearch/stella_en_400M_v5`** and the incumbent must be re-probed in the identical frame,
  but M9 is **not bound to inherit M8's document tower**, including any E14 head or LoRA M8 ships.
  Two consequences, both load-bearing. (1) **T1's NO SWAP does not transfer.** T1 measured that a
  teacher's own retrieval quality does not predict its distilled TABLE (Spearman 0.000 over eight
  candidates) — that is a fact about tables, and M9's artifact is a distilled TOWER, where LEAF
  reports 97.1–98.6% retention of its teacher. M9 must run its own screen **on M9's own artifact**;
  inheriting T1's answer would repeat, in reverse, the exact error M7 made by selecting a teacher
  on the tower instead of on the table. (2) **If M9's pick differs from the table line's, that is
  two document indexes**, not one index with two query paths — the cost Dylan's ruling accepts in
  exchange for M9 not being taxed by a document space re-shaped for a bag of token vectors.
  Docs are indexed with the teacher; the student is distilled into its query space: embedding
  alignment plus ranking preservation (M7's objective B), then optional contrastive finetune (C).
  No new data, doc vectors, dev suite, or conformance tests — reuse M7's.
- **TIE-BREAK — PREFER THE PAIR (Dylan, 2026-08-29, refining the above):** *"it would be great if
  we released that as a pair with the same model document side."* The shared document tower is the
  **preferred outcome and the registered default**; M9 diverges from the table line's document side
  **only on CI-resolved evidence** — raw two-sided 95% CI excluding 0 **and** `signflip_dep`
  p < 0.05, the definition §10 already uses. An unresolved difference is NOT a reason to break the
  pair. This does not weaken "whatever performs best": a tie goes to the pair, and only a measured
  loss breaks it.
- **What the pair costs to build, stated rather than assumed.** M9's student is distilled into the
  teacher's QUERY space, which a document-side head does not touch — so the **same student and the
  same distillation run serve both lines**, and only the document index differs. With an E14 head
  that index is a matmul over already-cached document vectors; with an E14 LoRA it is a full pool
  re-encode — hours, once. The student is not retrained either way.
- **Whether the pair is FREE is already being measured.** If documents carry the head while the
  student imitates the teacher's original query vectors, the quantity that decides it is
  teacher-style queries against HEADED documents — one of the four cells in `E14-HEAD`'s mechanism
  control. A loss there means the shared document side costs M9 quality, known before M9 starts.
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
