# Brief addendum for Codex Sol: second independent review of the M17 step-5 builder, AFTER the Astra fixes

Read `research/m17-step5-review-brief-2026-09-11.md` first: its rules (read-only, forbidden reads,
named files only, no recursive search, the one permitted pytest command), its file list, and its
"What I believe" claims all apply unchanged. This addendum tells you what changed since.

## What changed

Codex Astra reviewed the same code at commit `6fc3b6a` and reported 21 findings (10 P1, 8 P2,
3 P3) plus four owner decisions. The owner ruled on the four (ledger `A4`; registry
`accepted_plan_revision_a4`, `training.labeled_positive_budget_share`,
`data.documentless_sources_vote`, `candidate_construction.bank_sampling_amendment_a4`). All 21
findings were then fixed in commits `c76de64` (code and tests), `22e296d` (rebuilds, timing,
REVIEW/CODEMAP) and `9d8e3e4` (smoke evidence), plus `f8a8294`/`b6a8719` (rulings) and the
CODEMAP commit after them. Dispositions: `m17/REVIEW.md`, section "Codex Astra step-5 review".

Additional files you may read: `m17/REVIEW.md` (that section), `m17/LEDGER.md` section "A4",
`m17/FINDINGS.md`, `research/m17-astra-step5-review-2026-09-11.log` lines 7490–7630 ONLY (the
report; the rest is an exec trace), and the two smoke run records
`work/m17/runs/smoke-resume-VL-A/run_record.json` and `work/m17/runs/smoke-C/run_record.json`
(or whatever `ls work/m17/runs` shows; `ls` on that one path is permitted). Git: also
`git diff 6fc3b6a HEAD -- m17src m17/CODEMAP.md m17/REVIEW.md`.

## Your angle: break the fixes and find what Astra missed

1. For each Astra P1 marked Fixed, confirm it is really closed: the input that used to pass
   wrongly, the line that now refuses it, whether a test covers it, and whether the fix opened a
   new fail-open path or broke the happy path (the rebuilt s2000/s10000 directories and the
   400-step resume smoke are the evidence that the happy path still runs).
2. What Astra did not look at: the pool builder's memory. The fixed builder's RSS high-water is
   **12.4 GiB at both sizes** (was 3.5 GiB before the fixes) — a size-independent jump. Find the
   stage and structure responsible (the lowest-hash selection over the whole eligible population,
   the exclusion predicate's document-group tables, or the screen receipt are the suspects) and
   say whether the full-pool build stays inside the 25 GB host, with the 8–9 GiB of other
   residents the timing JSON records.
3. Anything wrong in the new `grad_shares` diagnostic (per-term `autograd.grad` on the rows) and
   the `--checkpoint-minutes` override — the resume binding must be unchanged, and the shares must
   be of the ROW gradient only.
4. The stage identity/`--force` invalidation (P2-11) and the artifact digest binding (P2-17): can a
   careless researcher on one box still end up training on arrays from two different builds?
5. The v1 int8 path (P1-9): is the dequantization exactly the released loader's, and is the
   residual used for vocabulary discovery computed from the same vectors as the cache's v1 lists?

Report only findings, ranked P1/P2/P3 as the original brief defines them, each with file:line,
the concrete input, observed vs expected, the smallest fix and test coverage. If an Astra fix is
incomplete, cite the Astra number. Confirmations belong in one closing paragraph. End with the
list of files you read.
