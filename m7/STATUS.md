# M7 status

**Stage: FINAL RUN DONE, 2026-08-28 (freeze `d24c704`, one `--infra-retry` after a harness
interrupt killed the first attempt — access spent once, receipt tag pushed). ZERO TIER CLAIMS:
the release bar is missed CI-resolved, the BM25 comparison does not survive the registered
familywise rule, and the fused system statistically ties OpenSearch. This is the pre-registered
publishable outcome — a measurement of how much quality a zero-compute query side retains — and
nothing about the system may change now.** `results/m7_final_run.json`.

## The six (confirmatory, one shot)

| comparison | Δ | raw 95% CI | sign-flip p | verdict |
|---|---|---|---|---|
| C1 int8 table > LR-dense-pertask 0.4583 | **−0.0243** | [−0.0405, −0.0086] | 0.997 | **miss, resolved BELOW the bar** |
| C2 int8 table > BM25 0.4174 | +0.0165 | [+0.0017, +0.0311] | 0.0149 vs Holm 0.0083 | not resolved (CI leg passes, multiplicity legs fail) |
| C3 fused system > OpenSearch 0.4868 | +0.0043 | [−0.0063, +0.0151] | 0.219 | not resolved — a statistical tie |

Macros (avg-6): **int8 table 0.4339** · fp16 0.4337 (int8 quality-free, as G4 said) · BM25 0.4174
· **fused 0.4911** · teacher-symmetric 0.5744. Six-set retention vs the teacher **0.755** — almost
exactly the dev out-of-domain retention (0.764) and far below the all-six dev retention (0.915):
the dev macro's in-distribution bias, measured and disclosed in advance, was real.

**Clean-4 robustness** (pre-registered, exposure-restricted, descriptive): C1 −0.0443
[−0.0675, −0.0212] · C2 **−0.0311** [−0.0517, −0.0109] — on the four datasets with no disclosed
teacher overlap the table is BELOW BM25 — · C3 −0.0107 [−0.0262, +0.0043]. The system's two
strongest datasets (ArguAna 0.5916, FiQA 0.3728) are exactly stella's two disclosed training sets;
the report must carry that at the dataset rows.

**Fusion vs dense on the six, descriptive as registered**: +0.057 (0.4911 vs 0.4339), carried by
trec-covid +0.153 and scifact +0.097 — much larger than the dev CQADupStack hint, and the one
clearly bright spot: the fused zero-query-compute system descriptively tops every Group-B system
in FINAL_MATRIX (opensearch 0.4868, LR-hybrid-pertask 0.4720) while its query side remains a
lookup table plus token counts.

The untouched-final tail was **skipped — reserved for M8** as registered.

**The clean-stack tax, measured (2026-08-28/29, `results/m7_cleanstack_tax.json`):** the frozen
recipe + decontaminated MS MARCO (490,241 pairs) gains **+0.0058 [−0.0015, +0.0131]** on the six
(int8; fused +0.0027) — and **still misses the release bar CI-resolved** (−0.0185
[−0.0344, −0.0030]). The licensing exclusion is NOT what the miss is made of; the gap is
architectural, which is where M8 should spend. Non-confirmatory by construction; the arm is
refused by `freeze.assert_releasable` (asserted in-run) and is never released.

Dev-selection numbers, for contrast with the six: full pinned dev macro 0.6153, out-of-domain
subset 0.3672. The gap between 0.6153 and 0.4339 is the measured cost of reading dev as a forecast.

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
**But no number from that decomposition is a forecast for the six** (`m7_fusion_report_*.json`,
re-run 2026-08-28 on the release artifact): the gain is hotpotqa +0.0865, cqadup-programmers
+0.0312, cqadup-physics +0.0239, and nq-250k **+0.0008, unresolved**. The two components carrying
the dev macro have no analogue among the six; the CQADupStack pair says only, qualitatively, that
the gain is not exclusively Wikipedia-shaped. Per the fifth review (MAJOR 4) the report may not
quote +0.036 — or any number — as an expected six-set transfer.

## Open for Dylan

*(Closed 2026-08-28: the familywise question — option (c); the freeze — written, committed
`d24c704`, tag pushed; the final run — done, on your explicit go, with the one permitted retry
after the interrupt.)*

1. **Release decision, now trivially framed**: the registered release bar (CI-resolved over
   LR-dense-pertask) was missed CI-resolved, so under the mandate the v1 dense table does not
   ship — consistent with your "v1 will probably not be released publicly". The fused system's
   descriptive edge over OpenSearch is a tie statistically and carries no release claim. Say the
   word if you want anything shipped anyway; otherwise this closes itself.
2. **The M7 report** (updating the M6 artifact with the M7 section) — I can draft it next; the
   binding framing is already registered in the LEDGER.
3. **M8 kickoff** — the v2 mandate is live (`instructions-m8.md`), the reserved sets are intact,
   and the clean-stack-tax variant is now legal to run (the freeze is immutable and the final
   number exists). The doc2query licensing ruling remains open if M8 wants that lever.

## The fifth review (2026-08-28, post-gate pre-freeze): STOP, then all findings actioned

`research/m7-codex-prefreeze-2026-08-28.md`, dispositions in LEDGER § Reviews and audits. The
short version: the one-shot path had eight remaining BLOCKER paths — a BM25 package upgrade would
have silently changed the fused system C3 judges; deleting the untracked result file plus a ledger
trim resurrected the one shot; two concurrent launches could both score; a `--untouched-only`
resume read all six frozen payloads; a diagnostic gate subset overwrote the official GO file and
would have been accepted at freeze; a hand-authored FREEZE.json bypassed the gate and licence
guards entirely. All fixed, all covered by new tests (guard suite + freeze-binding suite). The
fusion decomposition was re-run on the actual RELEASE artifact (it had read the training npz):
every number reproduces within ±0.0001. Registered before any six-set number: fusion-vs-dense on
the six is DESCRIPTIVE, no numerical fusion-transfer forecast may be quoted, and the only
subgroups reported are clean-4 + per-dataset rows + the fusion split. The review also
independently picked option (c) on the familywise question, matching Dylan's ruling.

## What closed earlier on 2026-08-28

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
