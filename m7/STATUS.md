# M7 status

**Stage: freeze prep DONE, gate is GO, the familywise question is CLOSED (option c, 2026-08-28),
and the fifth adversarial review (8 BLOCKER / 9 MAJOR — verdict STOP) is fully actioned. The gate
is being re-run from a clean commit (its GO had `m7src_dirty: true`); once it reproduces, nothing
blocks the freeze. Two decisions are yours.**

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
**But no number from that decomposition is a forecast for the six** (`m7_fusion_report_*.json`,
re-run 2026-08-28 on the release artifact): the gain is hotpotqa +0.0865, cqadup-programmers
+0.0312, cqadup-physics +0.0239, and nq-250k **+0.0008, unresolved**. The two components carrying
the dev macro have no analogue among the six; the CQADupStack pair says only, qualitatively, that
the gain is not exclusively Wikipedia-shaped. Per the fifth review (MAJOR 4) the report may not
quote +0.036 — or any number — as an expected six-set transfer.

## Open for Dylan, in this order

*(Closed 2026-08-28: the familywise question — Dylan ruled option (c), keep the rule as
registered and report the measured rates 0.0198/0.0283 vs nominal 0.025. `final_run.py`
untouched. See LEDGER § "The familywise question".)*

1. **The freeze.** `freeze.write('p35w-2m-s2500')` — the fusion spec and `released_system` are NOT
   arguments; it loads the selection and the gate result itself and refuses on any mismatch. Then
   commit, and **push the tag**: `git tag -a m7-freeze -m "..." <commit> && git push origin m7-freeze`.
2. **The final run.** One shot. `final_run.py --freeze-hash <commit>`.

**Budget you should know before scheduling 2.** The six are ~40–60 min. The non-confirmatory
`untouched-final` tail (10.1M docs, tens of hours, ~21 GB) is now **RESERVED FOR M8 by default**
(registered 2026-08-28, before any six-set number): the final run skips it, keeping FEVER /
DBpedia / cqadup-android / english un-scored as the v2's confirmatory sets. Override is yours,
before it would run; scoring them burns them for M8.

Also still yours: the doc2query licensing ruling (a revival needs a commercially clean generator),
and the HF release go. Milestones renumbered per your call: **M8 = the learnings v2**
(`instructions-m8.md`, new), **M9 = the LEAF-style distilled tower** (`instructions-m9.md`,
updated for the stella inheritance and the dead tokenizer rationale).

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
