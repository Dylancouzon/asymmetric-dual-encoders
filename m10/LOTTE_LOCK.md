# LoTTE read #1 — pre-LoTTE amendment for M10

**Current owner: M13. Not ready for access.** Open metric/slice and execution findings:
`m13/EXECUTION.md`. Historical clauses below remain subject to that gate.

**Written 2026-09-09, while LoTTE is ENTIRELY UNREAD for M10.** M9's read #1 was withdrawn
unexecuted at its close-out (2026-09-01) precisely so M10 would inherit a fresh surface, so this is
a **first** access, not a repeat.

**Classification: a dated PRE-LoTTE AMENDMENT, and the vacuity branch below is POST-SCREEN.**

This was my own error and the review caught it (Codex, 2026-09-09). I first classified the whole
file a preregistration. It is not: the original rule (`instructions-m10.md`:748) says the veto
RUNS, and the skip branch below **relies on an observed screen outcome** — that no non-default
component was selected. So it is pre-LoTTE and pre-build, which is what protects the *veto*, but it
is a post-screen amendment and must never be described as the original registration executed
unchanged. M9's equivalent was likewise an amendment, for a different reason.

## The veto rule — unchanged from M9 §7, deliberately

The registered two-model non-inferiority veto runs and is reported **exactly** as locked:

> the selection is vetoed iff the selected recipe's 7-slice macro is worse than the anchor's by
> more than **0.004** AND the one-sided 97.5% paired-bootstrap upper bound (B = 10,000, seed 903,
> paired within slice) on (selected − anchor) is below **−0.004**.

A veto means **the anchor recipe builds**. Margin, bootstrap rule and consequence: untouched. **No
other action may follow from read #1.** Adoption may not trigger any retraining choice, any recipe
edit, or any change to a screen verdict.

## WHEN, and on WHICH artifact — corrected 2026-09-10 to the mandate

**This was a real deviation and it is withdrawn.** My first version said read #1 happens *"after
the recipe lock **and after the build**"*, with the candidate being the 200M final checkpoint. The
mandate (`instructions-m10.md`:748-753) says something different and better:

> **read #1 after the recipe lock.** Before LoTTE opens, the **selected recipe is trained once as a
> single synthesized arm at screen dose** (its checkpoint hash committed); read #1 scores that
> checkpoint and the anchor's in one atomic batch.

So read #1 is **after the lock and BEFORE the build**, on a **screen-dose (5M) synthesized arm**,
against the **anchor's screen-dose checkpoint**. That is what makes the veto's consequence — *"the
anchor recipe builds"* — coherent: it chooses **which recipe gets the expensive build**, before the
money is spent.

My post-build version broke it in two ways Fable identified (2026-09-10): the veto would have
compared a 200M build against a 5M arm, so it could essentially never fire and would "run"
vacuously; and had it fired, "the anchor recipe builds" would have required a **second 200M build
that the budget table does not carry**. Both problems disappear once the read sits where the
mandate put it.

**M10's specific case.** The screen selected no non-default component, so *the selected recipe and
the anchor recipe are the same in every axis except `batch`* (`M102_LOCK.md`). At screen dose:

- if E selects **bs32** → the selected recipe IS the anchor recipe, so the synthesized screen-dose
  arm is `E-bs32` (the A100 arm of that same recipe; `ANCHOR` is the box-trained twin and the
  build runs on the A100). The veto would compare a recipe to itself and its action is invariant,
  so the
  **VETO is SKIPPED** and reported skipped. **The criterion is identical RECIPE AND ACTION, not
  identical checkpoint hashes** — my second version made the skip conditional on the two candidates
  hashing identically, and that was wrong: two stochastic builds essentially never hash alike, so
  it would have forced a vacuous veto to run while leaving the differing-hash branch with no
  mandated action. One criterion, one action.
- if E selects **bs128** → the selected recipe at screen dose is `E-bs128` (5M) and the comparator
  is **`E-bs32`, the A100-trained bs32 arm — NOT the box-trained `ANCHOR`.** Corrected 2026-09-10:
  both E arms run on the A100 together precisely so E1 carries no hardware difference
  (`M102_LOCK.md` §The one field still open), and naming `ANCHOR` here would have re-imported that
  same confound into the **veto** after removing it from the contrast. A veto means the bs32 recipe
  is what gets built — no extra training and no second build, because read #1 sits before the
  build and only chooses which recipe it is.
- **The candidate's OBSERVATIONAL LoTTE row is NOT skipped with the veto.** Fable's point, and it is
  right: the veto being vacuous does not make the out-of-domain *observation* vacuous, and this file
  argues LoTTE-clean is the only fresh OOD surface we have before the final run. So in the bs32
  branch the veto is skipped and **the observational row is still read and reported**, descriptively,
  on the synthesized arm. It selects nothing.
- **A skipped VETO is FORFEITED, not banked.** It does not become a spare access, and read #2
  remains the pre-freeze audit and nothing else. Without this sentence "we never used read #1"
  could be argued into a second decision-bearing look.

## Why a read happens at all

LoTTE-clean is the only genuinely fresh out-of-domain surface available before the final run, and
the six are development-informed. The M7 precedent is the reason: an in-domain dev read
overestimated six-set retention badly (0.915 dev against 0.755 actual) while the out-of-domain read
(0.764) was accurate. M10's dev surfaces have the same shape of problem — COV shares no dataset
with the release bar, and DEV-6 contains NQ-adjacent data that is in the training mix.

So the candidate's LoTTE number is **prospective out-of-domain observational evidence**. It
supplements; it does not convert an unactionable veto into an actionable one, and it **cannot
retroactively validate the screen selection**.

## Final-checkpoint identity — REGISTERED NOW, before any LoTTE access

Deterministic, and **may not be changed using LoTTE or any LoTTE-derived output**:

The candidate for read #1 is the **synthesized selected-recipe arm at screen dose (5,000,000
examples)**, on the recipe of `m10/M102_LOCK.md` with `seed_rule`'s seed 0 — its final annealed
cycle-end checkpoint, no dev selection. The comparator is the **same-hardware** screen-dose arm of
the recipe a veto would fall back to: `E-bs32` (A100) in the bs128 branch. Both hashes are committed and pushed in a second manifest commit **after both 5M E arms finish and
before the build** — never "after training ends", which is the post-build wording of the ordering
error this file already withdrew (Codex round 8). Neither arm exists yet: family E is CLOUD_ONLY
and unrun, so read #1 cannot happen before the A100 runs it.

**That second commit is a BOUNDED, PRE-CLASSIFIED amendment**, and it is the only edit this file
permits after the lock: it fills the four `*pending*` cells of the manifest table below with
measured checkpoint hashes, and changes nothing else. Naming it here is what stops it being an
unclassified later edit against `M102_LOCK.md`'s no-post-lock-edits rule.

*Materialisation (2026-09-10, rulings R17/R18, pre-observation):* the second commit's content is
the machine-readable file `m13/LOTTE_GATE_MANIFEST.json`, written by
`m13src/lotte_gate13.py --write-manifest` from the two published E arm records and committed; the
table cells below are filled by copying its two hashes. The executor refuses to read without that
file tracked and unmodified, and `build13.check_gate_manifest` binds the gate record to it.

(Read #2, the pre-freeze audit, is the one that sees the BUILD's final checkpoint, and it is the
only place any post-build wording belongs. Its identity
rule: the final annealed cycle-end checkpoint if the build reaches its registered end; otherwise
the annealed cycle end carrying the best COV macro `m_k`, disclosed as **dev-selected**.)

## Firewall

- **Read #1 happens AFTER the recipe lock is pushed and BEFORE the build.** Its only outputs are
  the veto verdict and the observational row — and the veto's only consequence is which recipe gets
  built.
- **Read #2** is the pre-freeze audit, and is audit only. **No third read.**
- LoTTE is **not** a selection surface, a bar, a tie-break, or an input to any C-conjunct. It
  appears in the final-run registry nowhere, and that is not an oversight.
- The brief for any adversarial review of this file carries the standing read-exclusion; LoTTE
  joins `results/frozen_eval/untouched-*`, the reserved qrels caches and `work/m9reserve` on it
  until read #1 executes.
- **M9's LoTTE state is not inherited.** M9's read #1 was withdrawn unexecuted; nothing about
  M10's access is conditioned on anything M9 did or did not see.

## Manifest — filled in the second commit, after training ends

| field | value |
|---|---|
| candidate checkpoint | *pending: the synthesized selected-recipe arm at 5M, A100. If E selects bs32 this is `E-bs32` and the VETO is skipped (its observational row is still read)* |
| candidate sha256 | *pending* |
| comparator checkpoint | the A100 screen-dose arm of the fallback recipe (`E-bs32` in the bs128 branch) |
| comparator sha256 | *pending the second manifest commit* |
| slices | LoTTE-clean, 7 slices, macro at equal weight |
| bootstrap | B = 10,000, seed 903, paired within slice, one-sided 97.5% upper bound |
| veto margin | 0.004 |
| observational row | read in BOTH branches, on the candidate, labelled a **5M-dose** number |
| executed | **NO — LoTTE is entirely unread for M10 as of 2026-09-10** |
| encode cost | LoTTE-clean's ~2.8M passages, stella once, ≈1.3 GPU-hours (mandate :748) — a named line in `M102_LOCK.md`'s budget table |
