# Opus adversarial review of PLAN-DRAFT v4 (2026-08-29) — the scientific-judgment pass

*Verbatim-in-substance record of the Opus 5 review commissioned to complement the three Codex
protocol gates. Verdict: "procedurally excellent, unsound in its aim" — the protocol machinery is
right, but after Dylan's rulings removed two high-EV directions the plan never re-derived its
expected value. Dispositions applied in PLAN-DRAFT v5 are marked [→ v5: ...]. Findings #5 and #2
were put to Dylan the same evening: he ruled STRICT C2 (E11) and comparators-inside-access YES
(E12).*

## A. Scientific aim

1. **BLOCKER — all four confirmatory sets are training-adjacent and nothing converts the
   disclosures into a guard.** FEVER-train is a TRAIN source (and in the teacher's proxy
   provenance), DBpedia is Wikipedia, the CQA pair is same-family with the dev pair. M7's
   FINDINGS #13 signature (gains concentrated near training) would read as a WIN on FEVER/DBpedia.
   Fix: a descriptive **six-set no-regression guard** as ship-blocker (frozen per-query vectors
   already exist; zero new access), a pre-registered per-dataset mechanism decomposition, and a
   plain LEDGER sentence that C1/C2/C3 measure improvement on training-adjacent domains while the
   six remain the honest generalization read. [→ v5: adopted, §2e]
2. **BLOCKER — no external anchor.** Scoring frozen off-the-shelf comparators INSIDE the single
   access, after the manifest is cut, leaks nothing (nothing can be decided afterwards). Fix:
   pre-register bge-small-en-v1.5 (+ LR-dense-websearch) scored on the reserved four as
   descriptive context outside the Holm family. [→ v5: adopted as ruling E12 (Dylan: yes, both)]
3. **BLOCKER — every probe bar (+0.005 OOD) sits at or below the measured noise floor
   (0.0027–0.0078 band; ~0.005 per-arm OOD resolution), across ~9 adopt/reject decisions.**
   Fix: measure M8's own noise floor first (two matched null replicates: seed change, ±10% step
   perturbation, on each endpoint); set every bar ≥2x the floor OR require sign-consistency across
   both OOD components plus a seed replicate (B3's design as the template); size the R1-vs-R0 gate
   from the confirmatory arithmetic and write the number before Stage R. [→ v5: adopted, §2b]
4. **MAJOR — no P(ship) estimate for the conjunctive C1∧C2∧C3 rule given the surviving EV
   (D2 +0.010–0.020 @ P≈0.45; D1 +0.003–0.010 @ P≈0.35; D4' fused-only; R1 unknown).**
   Fix: compute P(ship) from the joint power simulation BEFORE Phase 0 spends a week; put it in
   front of Dylan; cap the R programme at wave 1 + hard calendar budget; a knowing report-only
   choice beats a discovered one. [→ v5: adopted as a Phase-0 deliverable + wake-up-note item]
5. **MAJOR — the fused-objective lever (recipe P3; fusion is worth +0.057 and C1 IS fused) is
   absent because strict C2 disincentivizes it — a trade-off resolved silently.**
   [→ RULED by Dylan (E11): STRICT C2 stands — "we want something that looks good on benchmarks
   too; hybrid should be the default but isn't to everyone." The fused-objective lever is
   therefore consciously excluded from M8, recorded with this reasoning; the two-labelled-outcomes
   alternative was offered and declined.]
6. **MAJOR — the decisive routing diagnostic (in-domain oracle-generalization: 50/50 query split
   on dev CQA, oracle table on one half, score the other, against the 0.481 ceiling) is missing,
   while B4 — which E1's ruling made decision-irrelevant for M8 — is budgeted at up to a day.**
   Fix: swap them. [→ v5: adopted — new probe B17 in wave 1 with a routing rule; B4/B1' removed
   from the M8 calendar, recorded as M9 planning diagnostics]
7. **MAJOR — D2 has no registered coverage criterion** (924,704-span pool vs 3–5M needed at 128K
   vocab; Zipf-tail rows are the exact fragmenting types; 749 reachable never-trained rows already
   ship today). A D2 failure would be misread as "capacity doesn't help."
   Fix: register min-updates-per-reachable-row, targeted rare-row sampling + pool expansion, and a
   coverage-vs-capacity diagnosis rule; plus the "bag mass on cold rows vs per-query retention"
   diagnostic. [→ v5: adopted into D2's spec]
8. **MAJOR — the teacher-swap bar ignores the swap's statistical price**: near-sibling half-width
   ≈0.005 doubles to ≈0.0096 for dissimilar systems, the reserved encode runs twice, the FEVER
   cancellation dies, WordPiece compatibility breaks (stella-1.5B).
   Fix: swap must beat the incumbent by MORE than the CI-widening penalty (stated numerically);
   same-teacher is the registered default. [→ v5: adopted, §2f-T]
9. **MAJOR — the milestone's dominant compute item (reserved-4 doc encode, 10.12M docs ≈ 31x the
   six) is unplanned and would run INSIDE the guarded one-shot access** (`final_run.py` encodes
   docs in `score_set`; M7's first access attempt was already killed by a harness interrupt).
   Fix: pre-encode and hash-pin reserved-4 document vectors BEFORE the access for every scored
   system, via a named reviewed script physically unable to open the untouched query/qrel
   payloads; doc encoding reads no queries/qrels and is the same contact class as the mandated
   decontamination. The access becomes a minutes-long gather/rank step. [→ v5: adopted, §2a/§2f-T]
10. **MAJOR — LoTTE's independence from reserved android/english and dev physics/programmers is
    asserted, not measured; and the shadow covers only the CQA half of the estimand** (no
    Wikipedia/entity-shaped component).
    Fix: measure community-list + doc-hash overlap before final adoption (reopen E10 if overlap);
    add or explicitly register the missing Wikipedia-shaped shadow coverage. [→ v5: adopted —
    overnight measurement + registered coverage statement; wake-up-note if overlap found]
11. **MAJOR — no "don't spend the access" rule.** Fix: shadow GO threshold = the minimum
    detectable effect from the power simulation, with an explicit "defer to M9, panel preserved"
    branch. [→ v5: adopted, §2a shadow gate]

## B. Ruling consistency

12. E2 (synthetic queries YES) was unused by the plan. [→ v5: registered as a dosed component of
    R1's frozen pool spec with its own bar; EmbedDistill +3.2-on-11.3M-student prior cited]
13. B5 still in Phase 0 despite E5. [→ v5: deleted; post-access research slot]
14. B1'/B4 gate nothing under E1. [→ v5: removed from calendar, recorded for M9]
15. D2's row still had the 4-bit gate circularity. [→ v5: int8-only, gate B7]
16. E3's ONNX-fusibility must be a PREcondition of B6. [→ v5: adopted]
17. §4's "kept for context" block contains superseded proposals a future session could mine as
    live. [→ v5: the block is deleted; rulings table is the only §4 content]
18. E8's PMID measurement is a cost with no decision attached. [→ v5: deferred to the biomedical-
    gap revisit condition]

## C. Overnight-autonomy guardrails (adopted nearly verbatim into m8/NEXT-SESSION.md)

19. "No bar, no run" must be a code guard (`probe_guard.py`), not prose — this project found a
    pre-registered rule unapplied for four arms, by accident (FINDINGS #16).
20. M9-reserve construction is the most irreversible overnight item: freeze ONLY the corpus and
    query-text inventories overnight (all the contamination filter needs); qrel construction and
    the final hash-pin wait for a reviewed step; publish a PROVISIONAL sanity report.
Plus: path guard + grep test banning `results/frozen_eval/untouched-*`, LoTTE payloads, M9
payloads from `m8src/`; nothing irreversible overnight (no freeze/tag/access/upload; no writes to
results/perquery.json, eval_manifest.json, frozen_eval/, m7/); noise floor before any bar; smoke
+ setsid nohup + failure-signature monitors + rate-vs-estimate with a kill-at-2x rule; serial
GPU/RAM/disk schedule published first (incl. the 20.6 GB/teacher reserved encode); ordering
interlock (no teacher probe until the protected-query filter covering six+reserved+shadow+M9
inventories is committed); dev-reuse counter from evaluation #1; reviews read-only; blocking
questions go to the top of STATUS as a wake-up note, never decided alone at 4am.

## Verdict

Scientific core sound in protocol, previously unsound in aim; with #1–#6 fixed the aim matches
the protocol. Most likely failure mode if shipped as-was: a well-pre-registered report-only
milestone unable to say where the artifact stands against a small transformer — now addressed by
E12 (comparators in access), the no-regression guard, the noise-floor bars, B17 routing, and the
P(ship) gate.
