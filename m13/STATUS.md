# M13 status — stage 1 in progress, 2026-09-10

**Next:** implement `m13src/build13.py` and `m13src/score13.py` per `m13/STAGE1_DESIGN.md`, rehearse both
on synthetic inputs, then Codex review. Branch `m13-stage1-execution-prep`. Owner rulings R1–R8 are
listed in the design; none blocks the code, all block protected access.
M10's prepared data, completed screen and selected components are the input, not work to repeat.

| Stage | State / exit |
|---|---|
| Execution preparation | Surveys done, design written (`m13/STAGE1_DESIGN.md`, `m13/CODEMAP.md`); controller and scorer not yet written |
| Cloud E comparison | Both E-bs32 and E-bs128 pending; apply existing E1 rule after both finish |
| Pre-build gate | Lock/review LoTTE handling and run or skip under its registered branch; batch may still be vetoed |
| Build | Actual rate/price and complete allocation under $1,000; then 200M examples plus permitted extensions, freeze/provenance |
| Evaluation | Locked/reviewed six-set executor, M9 close-out, nano decisions and conditional reserved access |
| Cost frontier | Comparable zero/bge-small/nano serving and index costs on the reference hardware |

Detailed execution defects have one home: `m13/EXECUTION.md`. Recipe: `m10/M102_LOCK.md`.
A recipe push alone triggers neither M9 close-out nor final access. M9's rows must not inform
an open recipe decision. LoTTE handling belongs before the expensive build when its veto applies.

Done means a frozen candidate or documented stop, durable measurements/decisions and an evidence
handoff. A miss is publishable. Release is M14; the paper is M15. `ROADMAP.md` maps the old numbers.
