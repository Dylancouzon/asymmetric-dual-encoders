# Fable adversarial review of the amendment and the screen verdicts — 2026-09-09

Brief: `research/m10-review-brief-fable-2026-09-09.md`. Read-exclusion honoured (it reported
touching only git diffs of files under review and an `ls -l` of `work/m10arms/*/cov_cycle*.json`
for mtimes — metadata only; no reserved path was listed, globbed or opened). Eleven findings, all
applied; dispositions in `m10/LEDGER.md` §Fable pass and in the registry's own amended text.

Summary of its findings, ranked as it ranked them. Every number below was re-derived here before
being acted on, and two of the reviewer's own framings were corrected in the process.

## 1. E1 is rigged toward the expensive answer — and was still fixable

`WARMUP_STEPS` fixed in STEPS gives bs128 256,000 warmup examples (5.12% of a 5M arm) against
bs32's 64,000 (1.28%) and ~¼ the AdamW updates, so a handicapped bs128 makes a resolved bs32 win
likely — and bs32 costs 2.2× the build. Also: `E-bs128` carried `trained: true` with its real
disposition inferable only from a missing file; and E1 alone carries a hardware difference
(box-trained ANCHOR vs A100 bs128).

**Applied:** `rules.E_warmup_parity` (warmup registered in EXAMPLES, implemented as
`nano10.warmup_steps_for(batch)`), `arms.E-bs128.pending: CLOUD_ONLY`, hardware confound disclosed.

## 2. The amendment removed the only seed-variance measurement, and its budget arithmetic was wrong

(a) Every interval is a **query-resampling** interval; confirmation was the only design element
that would have re-trained at a fresh seed. ANCHOR is the comparator in ten of twelve contrasts.
(b) `budget_released` booked the amendment's **worst case** as realized: no eligible non-default
won, so confirmation would have confirmed **zero** decisions. (c) SYNTH-20M compares the anchor
recipe to itself. (d) `confirmation.plus` was orphaned.

**Applied:** all four. **Corrected in the reviewer's own framing:** its illustrative ±0.004 anchor
shift is ~3× the only seed figure the project has (0.0013869, `results/m10_calib_report.json`),
which it had not read; and (c) is conditional, not absolute, because `batch` is PENDING.

## 3. A4−A3 is 82% one unit

MedicalQARetrieval contributes +0.009885 of the +0.012080 (81.8%); BRIGHT is net negative
(−0.001335); the three-family point without consumer-health is +0.0029, half the MDE. A4 adds
`health` and `finance` forms — the two families that gained. **Verified exactly**, and independently
reproduced by the Codex pass. See `_interpretation.A4_A3_is_a_FORM_MATCH_effect` for what may and
may not be claimed from it — the "form-match effect" reading was subsequently judged over-claimed.

## 4. The D2 clipping story was asserted, not measured

**Correct, and retracted.** The clip instrumentation landed 14:20, after the last arm finished at
14:13:48, so no registered arm recorded a clip rate; and the audited 0.868 norm is at an
init-adjacent loss implying ~0.43 at the first logged training loss, under the 1.0 threshold. Its
alternative mechanism — Σ down-weights low-pooled-variance directions, i.e. what each domain needs
— fits the per-family deltas. D2's mechanism is **unresolved**.

## 5. The registry carried two contradictory timing statements

`_exposure_timing` said "no A number existed"; `_amended_2026_09_09` said macros were on disk. The
first is false (A1/A2/ANCHOR finals and A3's cycle-1/2 were all present). **Corrected.** The
reviewer also noted the relabel runs *against* our interest — it withholds the favourable reading
from a contrast that went on to resolve — which is why it stands.

## 6. Five latent code defects, none with a live effect

Sign stability accepted a missing previous cycle end; no cross-check of the COV file against the
committed arm record; `serve_cost_order` never consulted; `multi_arm_winner`'s tie-to-default not
implemented; the family-A exemption hardcoded in one place and rule-driven in another. **All
fixed** — and the Codex pass then found that two of those fixes did not work
(`research/m10-codex-fixes-review-2026-09-09.md`).

## 7. Claims it could not break

Claims 3 (no threshold set after a number was visible), 5 (E PENDING is the registry's own
disposition, not an invention), and 7 (no bar, partition, comparator, statistic, contrast spec, arm
spec or frozen artifact changed) — all verified correct against the diffs.
