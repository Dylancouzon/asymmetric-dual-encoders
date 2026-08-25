# M8 — LEAF-style distilled query tower (tiny mandate)

Run after M7's final run, same box, same session rules. M8 builds the other point on the spectrum: a small transformer query encoder distilled to search a frozen bge-base-en-v1.5 document index — MongoDB's LEAF technique, with a clean-vendor teacher.

Everything binds from `instructions-m7.md` unchanged: decision authority, licensing and decontamination rules, dev-only selection, frozen comparator pairing, the freeze/ledger protocol for one final run, Sonnet subagents for research, and the headless git contract — working files under `m8/`, same four-file split.

What changes:

- **Student:** a ≤35M transformer initialized from a permissive clean-vendor backbone — default bge-small-en-v1.5 (MIT, same tokenizer as the teacher) — distilled into bge-base's query space: embedding alignment plus ranking preservation (M7's objective B), then optional contrastive finetune (C). No new data, doc vectors, dev suite, or conformance tests — reuse M7's.
- **Bars, paired on the frozen vectors** (`results/perquery.json` carries leaf-ir-asym 0.5155, mdbr-leaf-ir 0.5123, arctic-embed-m-v1.5 0.5264, bge-small-en-v1.5 0.5042): **release = CI-resolved above bge-small symmetric (0.5042)** — an asymmetric student is only worth shipping if it beats simply running the small model on both sides. Reference rows, not gates: leaf-ir-asym (MongoDB's model, on the stronger Snowflake teacher) and the bge-base symmetric ceiling from M7's final run; report retention vs teacher next to LEAF's published 97.1–98.6%.
- **Costs:** expect ~3–5 ms per query plus a model load. Report the same three cost numbers so M7 and M8 land in one frontier table.

Deliverables: weights on HF under the Qdrant org (Dylan's explicit go), a section added to the M7 report artifact (not a separate report), the frontier table updated, decisions logged in CLAUDE.md.
