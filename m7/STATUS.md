# M7 status

**Stage: freeze prep DONE, gate is GO. Everything on the assistant's queue is closed. Three
decisions are yours, and the first one blocks the freeze.**

Candidate **`p35w-2m-s2500`**, served at `pool_mode=sqrt`. Full pinned dev macro **0.6153**,
out-of-domain subset **0.3672** — both dev SELECTION numbers, not evidence about the six.

## The gate: GO, 2026-08-28 (`results/m7_gate_p35w-2m-s2500.json`)

| | |
|---|---|
| G1 Stage-0 (`s1-objB`) > potion | **+0.1159** [0.1074, 0.1244] |
| G2 capacity probe | pass — gate-ineligible as evidence of anything but expressibility |
| G3 candidate > BM25 | **+0.0845** [0.0764, 0.0926] |
| G4 int8 equivalence | upper **0.00013** against a 0.005 bar |

Retention **0.846** text-backed / **0.915** all six (teacher ceiling 0.635 / 0.6724).
**G3 is broad, which is the check GO #1 failed**: nq-250k +0.2206, cqadup-physics +0.0487,
cqadup-programmers +0.0412, hotpotqa +0.0276 — every component resolved above zero, including both
out-of-domain ones.

**Fusion** (`results/m7_fusion_p35w-2m-s2500.json`): `convex0` **w=0.8**, dev macro (text-backed)
**0.5727** against the int8 table alone at 0.5370 and BM25 alone at 0.4525. So `released_system`
derives mechanically to **fusion**, by +0.036 over the dense-only endpoint, with no tie.
**But read the breakdown before expecting +0.036 on the six** (`m7_fusion_report_*.json`): the gain
is hotpotqa +0.0865, cqadup-programmers +0.0312, cqadup-physics +0.0238, and nq-250k **+0.0008,
unresolved**. The two components carrying the dev macro have no analogue among the six; the two that
do — the CQADupStack pair — gain +0.024 to +0.031. The smaller figure is the defensible expectation.

## Open for Dylan, in this order

1. **The familywise question — it BLOCKS the freeze**, because it changes a registered rule and a
   rule may only move before its numbers exist. `research/m7-fwer-decision-2026-08-28.md` has the
   arithmetic and four options. Measured: the three-leg tier rule's weak-null familywise rate is
   **0.0198 and 0.0283** across two stand-ins against a nominal 0.025 — so the review's 0.039 union
   bound is loose, but the rule is mildly anti-conservative on the closer stand-in. A weak-null-valid
   studentized leg measures 0.0203/0.0278, i.e. buys nothing. I lean weakly to option (d), tightening
   the simultaneous leg until the measurement lands ≤0.025 — it costs power on a bar the projections
   already straddle, which is why it is your call and not mine.
2. **The freeze.** `freeze.write('p35w-2m-s2500')` — the fusion spec and `released_system` are NOT
   arguments; it loads the selection and the gate result itself and refuses on any mismatch. Then
   commit, and **push the tag**: `git tag -a m7-freeze -m "..." <commit> && git push origin m7-freeze`.
3. **The final run.** One shot. `final_run.py --freeze-hash <commit>`.

**Budget you should know before scheduling 3.** The six are ~40–60 min. The non-confirmatory
`untouched-final` tail is **10,115,709 documents, 37x the six** — tens of hours and ~21 GB. The
confirmatory result and all three tier decisions are written to disk *before* that stage starts, and
`--untouched-only` resumes it independently, so it can be deferred or skipped without touching the
tier claims.

Also still yours: the doc2query licensing ruling (a revival needs a commercially clean generator),
and the HF release go.

## What closed on 2026-08-28

Two adversarial reviews on the one-shot path (3/5/2 then **6/11**), all findings actioned. The one
that mattered: **the freeze tag was never peeled**, so `git tag -a` — the procedure this file
documents — produced a tag object the guard compared against a commit hash, and the final run could
not have started at all. Also: the access counted as spent from an editable ledger line; `--infra-retry`
allowed two retries where its docstring promised one; query TEXT was never hash-pinned; the result
was written after the 10M-document tail rather than before it.

Freeze prep then exposed three more, all in the gate path: it was handed the **BGE-era** Stage-0
table, `ensure_release` had stamped the ambient teacher onto it, and the gate never checked a
checkpoint's teacher at all. Fixed; the rerun is the GO above.

Detail: `LEDGER.md` (protocol and every bar) · `RECIPE.md` (what ships) · `FINDINGS.md`
(transferable) · `EXPLORED.md` (closed avenues) · `CODEMAP.md` (code) · `RESULTS.md` (every run).
