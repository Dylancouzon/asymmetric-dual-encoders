# M7 status

**Stage: pre-freeze. Candidate `p35w-2m-s2500` served at `pool_mode=sqrt`, dev macro 0.6153.**
Not the 0.6258 quoted before 2026-08-28. The candidate moved *backwards* twice that day, both
times because a pre-registered rule was honoured against our own interest.

1. **Lever #4 (count-saturation pooling) failed re-adjudication** on the negatives candidate
   (+0.0033, CI [−0.0010,+0.0073], p=0.063/0.067, nothing clears Holm) — a consequence the
   negatives pre-registration had named in advance.
2. **The negatives avenue then closed entirely.** The step-selection rule had never been applied
   to those arms; the corrected arms score +0.0023 (p=0.107), −0.0007, and −0.0056 (a resolved
   loss) — **zero survivors**. The candidate reverted to the pre-negatives artifact, for which
   lever #4's `sqrt` *does* stand.
3. **The recipe simplification failed** non-inferiority (−0.0048, raw CI lower −0.0102 against a
   −0.0040 margin), so the recipe ships unchanged and no back-off ladder was run.

**The revert cost almost nothing where it matters:** macro 0.6225 → 0.6153, out-of-domain subset
0.3674 → 0.3672.

## The finding that outlives the arms

**A nuisance A-phase step count moves the dev macro by 0.0027–0.0078** across three arms at
matched pooling. **Every effect this project has adopted or adjudicated is inside that band** —
lever #4's +0.0040, lever #2's +0.0065/+0.0038/+0.0023, the simplification's −0.0048. Replay noise
is only ~5e-6, so the band is a property of the *recipe choice*, not of re-running. Every interval
in the repo is a **query-sampling** interval with no recipe-replication term: the bars answer
"would another sample of queries agree", not "would another equally-defensible recipe agree".

This bears on **dev selection claims only**. The final run fixes the recipe first and scores frozen
comparator vectors on datasets never used for selection, so the tier comparisons are unaffected.

Read `LEDGER.md` § "THE DEV MACRO IS A BIASED ESTIMATOR" before believing any dev gain transfers.
`m7_dev_reuse_count.json`: **58 trained arms, 322 in-training dev evaluations, 90 eval-only
variants**, Holm inside named families only. `m7_retention_p35w-2m-s2500.json`: retention **0.915**
all-six, **0.846** text-backed, **0.764** out-of-domain, where BM25 scores 0.3223. All six
confirmatory datasets are out-of-domain, so the last figure is the honest one.

## Settled

- Levers **#4** (passed on this artifact, failed on the next — kept because the bar was
  pre-registered and the rule is free), **#5**, **#6**, and **mined negatives**: all closed under
  their own bars. The mandated ablations are finally scored on the full suite and **no arm clears
  Holm over the family of eight**; learned token weights are the one component that clearly earns
  its place (−0.0062 to remove).
- **Novelty re-swept**: claim stands, phrasing weakened to "we found none". Nearest miss **KAHM**
  (arXiv 2605.02950) clears the frozen-doc axis, misses the dense-rows one.
- **Corrections made rather than papered over**: the doc-side-linear-map absorbability claim was
  half wrong; the pseudo-query doses are 924,704 / 324,704, not 2m / 500k (2.85x, not 4x); a
  lever-4 evidence pointer resolved to a different artifact's failure; "deterministic to the last
  digit" read a *rounded* CI, which this ledger's own statistics rule forbids — the real figure is
  4.47e-06; and the headline retention 0.926 was the reverted candidate's, not the shipping one's.

## Closed today, all of them before the freeze

- **Lever #7, long-span distillation — CLOSED without training its arm.** Its gating probe drew a
  fresh document sample per length bucket, so its "length effect" was a document-population
  effect. Re-measured as nested prefixes of the same documents, agreement is **flat from 16 to 256
  words**. The lever's own pre-registration makes that probe the pre-condition for spending a
  training chain, so the chain was not bought — the arm was already running, 7 of 23 encode shards
  done, when the review exposed the confound.
- **Teacher re-examination — NO SWAP.** `arctic-embed-m-v1.5` 0.3002 and `gte-base-en-v1.5` 0.2741
  against stella's 0.3439, both CI-resolved below, both below even bge-base. Run on the M5 Mac,
  which also reproduced the incumbent row to **7e-4** — the criterion is hardware-robust.
- **Both adversarial reviews returned and are actioned.** Codex (2 BLOCKER / 6 MAJOR / 2 MINOR)
  and an over-fitting/over-engineering pass (2 BLOCKER / 6 MAJOR / 5 MINOR). Between them they
  caught: a rounded CI read as exact in `final_run.py` — the one irreversible decision; `gate.py`
  passing the wrong Stage-0 checkpoint and returning 0 on NO-GO; `freeze.assert_releasable`
  failing open; the tier rule's CI leg never being simultaneous; a retention figure belonging to
  the reverted candidate; unpinned training-data revisions; and a test suite that had been failing
  since the teacher swap with nothing running it.

## Queued, in order — and the first two BLOCK freeze prep

A third adversarial review (`research/m7-codex-onepath-2026-08-28.md`, targeted at the one-shot
path only) returned **3 BLOCKER / 5 MAJOR / 2 MINOR**. All three BLOCKERs and both MINORs are
fixed; four MAJORs remain and their order is forced by what each one touches.

1. **MAJOR 1 — bind fusion to the artifact it was selected on.** `select_fusion` writes no run id,
   table hash or preproc fingerprint into the spec, and `freeze.write` accepts whatever spec the
   caller passes without consulting the selected file or the gate result. So a spec fitted on
   artifact A can be frozen with artifact B undetected. `released_system` is also not an enum:
   anything other than exactly `"fusion"` is silently treated as dense, and an unknown fusion
   family is silently treated as convex. **Must land BEFORE `run_freeze_prep.sh`, because step 1
   of that script IS the fusion selection.**
2. **MAJOR 2 — the BM25 cache is keyed by pathname only.** The cached files hold integer positions
   and scores, with no corpus/query hash, BM25 config or version, so a compatible-shaped but
   different corpus selects a parameter on one lexical run and applies it to another. Also blocks
   freeze prep, same reason.
3. **MAJOR 5 — the weak-null familywise claim.** My Bonferroni fix this morning does not deliver
   what it claimed: three marginal procedures each running at 0.013 bound the family to **0.039**,
   not 0.025. Two honest options — define the inferential claim as the sharp/exchangeability null
   actually tested, or replace the decisive leg with a weak-null-valid studentized or simultaneous
   procedure and calibrate it. **Must be settled BEFORE the freeze**, since it changes a registered
   rule, and a registered rule may only move before its numbers exist.
4. **MAJOR 4 — final document vectors come from mutable, unverified gitignored cache bytes.**
   `encode_cached` trusts an existing shard, and `combined.f16` is trusted on byte size alone.
   Needs hashes anchored in the freeze, or a verified rebuild. **Before the final run.**
5. `RECIPE.md` owes `fn_margin`, `use_provided_hardneg`, a fusion section and the env/dataset pins.
6. Vendor or pin the teacher's `trust_remote_code`.
7. **`LEDGER.md` is ~21K tokens against its own 4K budget** — compact before the freeze commit.
8. `./run_freeze_prep.sh p35w-2m-s2500`, then stop.
9. Post-freeze, pre-registered: the clean-stack tax.

**The freeze commit now needs a pushed tag.** `final_run.py` resolves the freeze commit from an
immutable `m7-freeze` tag rather than from `--freeze-hash`, so the freeze procedure is:
`git tag -a m7-freeze -m "..." <commit> && git push origin m7-freeze`.

**The final run is NOT scheduled.** It is the one-shot confirmatory access to the six, and the
freeze and that run are Dylan's calls.

## Open for Dylan

1. **Nothing blocking *you*** — but the recipe is NOT frozen yet and `p35w-2m-s2500` is still a
   provisional candidate. The queue above has to close first; the freeze and the final run are
   yours to call after it does.
2. doc2query revival (licensing ruling on a clean generator) remains yours.
3. Windows Update reboots remain the top operational risk.
