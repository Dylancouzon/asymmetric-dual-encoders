# M7 status

**Stage: capacity levers being executed and killed on evidence (2026-08-27 evening).** The GO
winner is unchanged: `s2w-1e3-s1000`, full-suite dev macro 0.5987, retention 0.890 all-six.
Yesterday's "WSL crash" is root-caused and fixed: the kernel OOM-killed the K=10,000 bigram
probe — `solve_ridge` made two hidden 13 GB copies of the Gram (an `astype` no-op copy, plus
scipy ≥1.15's batched `solve` ignoring `overwrite_a`). Now one copy, raw LAPACK `posv`, smoke
reproduces the committed numbers exactly.

## Capacity levers (the pre-registered bridge to the bars) — two down, one running

1. **Bigram rows: CLOSED for closed-form integration.** Probe ladder was real (K=10000 +0.0143
   resolved, proxy, ridge frame) but the pre-registered adoption run failed decisively:
   **−0.0301 [−0.0357, −0.0247]** vs the winner on the full suite, worse everywhere. Diagnosed,
   not just observed: λ-sweep is monotone toward zero from below — structural. A closed-form
   fit's only supervision is the teacher target, and the winner's A-phase deviations from the
   teacher ARE its gains. Escalation (joint retrain with bigram features) stays open, needs its
   own pre-registration. `results/m7_bigram_residual_k10000.json`.
2. **doc2query: CLOSED per the pre-registered rule.** +0.0054 [−0.0007, +0.0114] p=0.085 —
   positive-leaning on both components but unresolved at the cheap-test price (N=5/doc,
   T5-base). Parked, not disproved; revival needs a clean-licensed generator (your ruling — all
   available ones are MS MARCO-trained) plus a larger sample budget.
3. **Pseudo-query coverage: arms RUNNING** (`run_lever2.sh`, protocol pre-registered in LEDGER
   before any arm). Pools: nominal 500k/2m cap at 324,704 / 924,704 spans; R1 pass kept
   324,156 / 923,590. 500k arm first (B 8k + mix, then the A phase exactly mirroring the winner
   arm); 2m runs only if 500k is not resolved below the winner.

## Run next, in order

1. Lever #2 best-step selection + `compare_release.py` vs the winner (adoption bar: signflip
   p<0.05 AND CI>0, int8 independently).
2. **Mandatory ablations** (`program.phase4_mandatory`) on the surviving release shape.
3. Fusion re-selection on the final checkpoint (grid incl. convex0) → released-table ANN sweep
   + costs.
4. Freeze → the single final run (tier rule: Holm(sign-flip) AND CI>0).

## Open for Dylan

1. Nothing blocking tonight's work.
2. doc2query revival is yours if ever wanted (licensing ruling on a generator).
3. Ops note: background-task kills today were memory-pressure kills, not kernel OOM — long
   memory-heavy jobs now run foreground-or-watched with per-stage RSS prints.
4. Windows Update reboots remain the top operational risk.
