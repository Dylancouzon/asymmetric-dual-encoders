# M7 status

**Stage: stella candidate selected and gated — GO (2026-08-26 21:03).** In one evening: teacher
swap landed, all Codex review #2 findings fixed/evidenced/ledgered, decontamination strengthened
twice (4-grams + containment; TRAIN 340,850 pairs; 7,190 pool rows banned as negatives), the
phase-2 band confirmed under stella with a labeled edge extension, and the winner
**`s2w-1e3-s1000`** (A @ lr 1e-3, 1,000 steps from `s1-objB`) gated on the RELEASE-shape
artifact: **all four PASS**, G3 vs BM25 +0.0711 [0.0629, 0.0792] with a broad win (hotpotqa
near-tie, not the bge candidate's resolved loss). **Retention 0.8245 text-backed / 0.8903
all-six** (bge candidate: 0.785/0.807). Details: `m7/RESULTS.md`, `m7/LEDGER.md`,
`results/m7_gate_s2w-1e3-s1000.json`, `results/m7_stella_winner.json`.

## Run next, in order (all pre-freeze; review #2 B7/M24 bind this ordering)

1. **Capacity levers** — the only known bridge from here to the bars: (a) n-gram rows (top-K
   frequent bigrams as extra table rows, closed-form-distilled first — genuinely new capacity per
   `m7_absorb_check`); (b) the pseudo-query coverage phase (`program.py`, unused); (c) the cheap
   doc2query test (EXPLORED.md reopened row). Kill each on evidence; log every arm.
2. **Mandatory ablations** (`instructions-m7.md`): three inits, two prefix variants, flat vs
   learned weights, int8 re-report — all on the release shape.
3. **Fusion re-selection** on the final checkpoint with the fixed builder (grid now includes
   convex0), then released-table ANN sweep + costs.
4. **Freeze** (`freeze.py` now pins the release artifact + training-checkpoint sha) → the single
   final run (tier rule: Holm(sign-flip) AND CI>0; clean-4 robustness computed in-scorer).

## Honesty rails for the projection question

No six-set projection is quoted for the candidate: the single-anchor projection was withdrawn,
and the composition rule (per-dataset products, never ratio x mean) is in LEDGER. What is known:
retention improved 0.807 → 0.890 all-six and the teacher ceiling rose; whether that clears
0.4583 (release) or 0.4868 (Tier 1) is decided only by the final run, after the capacity levers.

## Session ops notes (2026-08-26 evening)

Codex review #2: 7 BLOCKER / 15 MAJOR / 2 MINOR — every code-level finding fixed same-day
(`research/m7-codex-review-2026-08-26b.md`, dispositions in LEDGER); one crash bug (bank-mask
OOB) would have killed tonight's training. A-phase re-runs are nondeterministic by ~0.0007 proxy
(CUDA atomics) — the saved artifact is what is judged and ships. One grant violation
(amend+force-push) self-reported in LEDGER incidents.

## Open for Dylan

1. Nothing blocking. The GO is real and broad-based this time; capacity levers queue next.
2. Review #2 items needing your eye eventually: the third six-set access deviation (my `--bm25`
   pairing check — conceded in LEDGER, report will enumerate all three); android/english are
   labeled within-family transfer, not untouched generalization.
3. Host: Windows Update reboots remain the top operational risk.
