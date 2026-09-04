# M14 — the paper

Created 2026-09-04 (Dylan): *"paper writing can be its own milestone after building nano. We'll
decide then if we want to do 2 papers or 1 but probably only 1."* Split out of M13, which held it
as deliverable 4 and would have kept it hostage to a milestone that may never run.

Binds from `instructions-m7.md` unchanged. Working files under `m14/`.

## Sequencing

**Runs after M13 delivers nano.** If M10/M13 never run, M14 still runs — on the zero-only frontier
plus whatever M12 produced. **One paper is the default**; the one-vs-two decision is Dylan's, taken
at the start of M14 when the evidence is known, not now.

## What the paper argues

The frontier is the *deliverable*, but it is not the *finding*. The finding is:

**Teacher quality does not predict distilled-table quality.** Spearman ≈ 0 (ceiling→table 0.000 over
eight; MTEB→table −0.11 over seven verified rows) across **eleven** measured teachers. The
best-ceiling candidate ranked 7th. `mxbai-embed-large-v1` has the same measured tower quality as
`bge-base-en-v1.5` (0.4433 vs 0.4484) while its table is **0.057 worse** — three times the
parameters, higher MTEB, worse product. Selection must run on the distilled artifact, never the
tower. Survives a contamination control on data the winner never trained on (SQuAD +0.1652, ESCI
+0.0384, `results/m7_offfamily_report.json`).

This is measured today and needs no nano. Everything else is evidence for it or around it.

## Contents

1. The finding above, with the eleven-candidate table and the exposure-free control.
2. The quality-vs-query-cost frontier: `zero`, `zero-hybrid` (M12), `nano` (M13, if it lands),
   against the 21-system × 6-dataset matrix.
3. Edge cost rows — query asset, document index, hydration, CPU latency — plus the offline BM25
   index cost M12 measures.
4. The Qdrant Edge prototype: architecture and latency, with the exact-search caveat.
5. **The comparator table deliberately kept OFF the model cards** (`instructions-m11.md`
   Amendment B): `LR-dense-pertask 0.4583`, the OpenSearch tie, the missed bar. This is its home.
6. Negative results, stated as results: M8's twelve levers, M9's coverage failure, and every kill.

## Rules that bind the writing

- **Every headline number is paired on frozen comparator vectors with pre-registered statistics.**
  No number enters the paper that is not already in a committed result JSON.
- **stella's ArguAna/FiQA/FEVER exposure is disclosed wherever a stella-derived number appears.**
- **Deltas inside the ~0.005 lever band are labelled as ties**, not wins.
- Do not quote the off-family macro (+0.1018) as a product delta — it is a contamination control.
- Do not claim `zero` confirmatorily beat BM25: C2 was +0.0165 [0.0017, 0.0311], Holm p=0.0149 >
  0.0083, **not rejected**.
- No new measurement. If the paper wants a number that does not exist, that is a milestone, not a
  paragraph.
