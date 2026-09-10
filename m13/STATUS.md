# M13 status — stage 1 code written, rulings applied, review in progress (2026-09-10)

**Next:** close the Codex implementation review's findings, then the stage-2 checklist: ship list,
allocation table with the measured price, provider (Dylan). Branch `m13-stage1-execution-prep`.
Rulings R1–R12 are recorded and applied (`m13/RULINGS.md`); R6 flips at the pinning commit.
M10's prepared data, completed screen and selected components are the input, not work to repeat.

| Stage | State / exit |
|---|---|
| Execution preparation | `build13` (38 tests, box smoke with ONNX/fastembed parity 1.0) and `score13`/`access13` (33 tests, synthetic rehearsal) written; E-bs32 runner fix; Codex review pending |
| Cloud E comparison | `pending` cleared under R9, F verdict and twelve contrasts re-bound byte-identically; both arms unrun; E1 applies after both finish |
| Pre-build gate | Registered (`m13/LOTTE_GATE_REGISTRATION.json`: seven slices, nDCG@10 veto, Success@5 descriptive); executor and gate record still owed |
| Build | Actual rate/price and complete allocation under $1,000; then 200M examples plus permitted extensions, freeze/provenance |
| Evaluation | Locked/reviewed six-set executor, M9 close-out, nano decisions and conditional reserved access |
| Cost frontier | Comparable zero/bge-small/nano serving and index costs on the reference hardware |

Detailed execution defects have one home: `m13/EXECUTION.md`. Recipe: `m10/M102_LOCK.md`.
A recipe push alone triggers neither M9 close-out nor final access. M9's rows must not inform
an open recipe decision. LoTTE handling belongs before the expensive build when its veto applies.

Done means a frozen candidate or documented stop, durable measurements/decisions and an evidence
handoff. A miss is publishable. Release is M14; the paper is M15. `ROADMAP.md` maps the old numbers.
