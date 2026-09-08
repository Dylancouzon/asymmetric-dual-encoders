# DRAFT registration — the paired M9-candidate vs M10-candidate row

**STATUS: NOT REGISTERED. Needs Dylan's ratification.** Drafted 2026-09-07 while family F trains,
from the two reviewer passes archived in `m10/LEDGER.md` (§SECOND OPINION on W12·W13·the paired
row, §VALIDATION 2026-09-05). Every element below was required by a reviewer; nothing here is new
judgement. Ratify, amend, or reject as a whole.

**Why it is time-sensitive.** Two windows are closing. (i) A pre-registration is only valid before
the numbers it affects exist, and this row reads BOTH six-set transactions. (ii) Item 7 below must
be in force **before M9's close-out transaction runs** — that transaction executes once M10's
recipe lock is pushed, and if it emits only aggregate rows the paired row **cannot be built at
all, ever**. Item 7 is the part that expires first.

## 1. What it is

A **whole-protocol RECIPE delta** between the M9 candidate and the M10 candidate, on the six named
datasets and on the pre-registered clean-4, against the frozen comparator vectors.

**It is NOT a coverage test.** Codex's framing, adopted: data, sampling, head width and form,
optimizer, batch, loss, schedule, init, selection, dose and stopping all changed. It is a
historical delta between two recipes, reported as such.

## 2. Confounds, listed on the record

| changed | M9 | M10 |
|---|---|---|
| dose | 3.69B tokens | ≈16.8B tokens (4.5×) |
| head | 384 wide, linear | 1152 wide, three pooled layers |
| schedule | M9's | LEAF small-batch cyclic |
| objective | L2 regression | squared L2 (+ the phase-2 ‖e‖₂ arm) |
| data mix | M9's | ≈1.25M harvested forms + ≈1.0M generated |
| batch | M9's | 32 (registry-owned) |
| init | off-the-shelf | off-the-shelf (C-M9init CUT) |
| selection | M9's | the COV screen |
| stopping | plateau rule at 3.74B | registered dose |

## 3. Statistic — teacher-normalised, and never called "retention"

Per dataset `d`, with `T_d` the teacher (stella_en_400M_v5) score:

- `r_9d  = S_9d / T_d`
- `r_10d = S_10d / T_d`
- **`Δr_d = (S_10d − S_9d) / T_d`** — the registered headline quantity
- the **raw paired difference** `S_10d − S_9d`, reported beside it

**`S_10/S_9` must never be called "retention"** (Codex): M9 is a failed candidate, not a ceiling,
so a ratio against it names no meaningful fraction. Retention language is reserved for `r_*d`,
which is against the teacher.

## 4. Fixed claim sentence — the only sentence this row licenses

> On the six named datasets and the pre-registered clean-4, the M10 candidate scores `Δr_d` of the
> teacher above the M9 candidate per dataset, measured on frozen comparator vectors. The two
> candidates differ in dose (3.69B vs ≈16.8B tokens), head width and form, schedule, objective,
> data mix, batch, selection and stopping, so this is a delta between two whole recipes and
> attributes nothing to any one of them.

**No causal "coverage caused" language**, in the report, the cards, or the paper. The causal version
is the matched near-full-dose control, which the budget can only buy if the screen is cut.

## 5. Conditional label on the backbone

"Same student family" is **not guaranteed** — family F may select MiniLM-L6 over bge-small. So the
row carries one of two labels, decided by F's outcome, not by preference:

- **within one backbone family** (F selects bge-small, which M9 distilled into), or
- **across backbone families** (F selects MiniLM), in which case the backbone change joins §2.

## 6. Reporting status — descriptive, and it costs no α

The row is **descriptive**. It is reported with a paired CI at the registered machinery (paired
stratified bootstrap over queries within dataset, **B = 200,000, seed 0, `inverted_cdf`**), but:

- it is **not** gated by `resolve_rule`,
- it does **not** consume a confirmation slot,
- it does **not** enter the Bonferroni denominator (which stays **13**),
- and no build or release decision reads it.

Fewer tails at the same per-tail level is strictly conservative, so adding a descriptive row needs
no α re-registration — and gating it would turn a confounded contrast into a decision instrument,
which is precisely what §1 refuses.

## 7. THE PRECONDITION THAT EXPIRES FIRST

**Both six-set transactions must emit aligned PER-QUERY scores and qids, not merely aggregate
rows.** Codex: *"zero extra access is true only if both transactions preserve aligned per-query
contributions and qids."* A paired bootstrap needs the per-query pairing; aggregate rows cannot be
paired after the fact, and neither transaction can be re-run.

Required, in the M10.2 lock, before M9's close-out executes:

- M9's close-out (`m9/FINAL_LOCK.md`) writes per-query scores **and qids** for all six.
- M10's six-set transaction does the same.
- Both are byte-verified as aligned on qid before any paired number is computed.
- If either emits aggregates only, **the paired row is abandoned and reported as not built** —
  it is never reconstructed from aggregates.

## 8. What is NOT decided here

Whether family A is a causal experiment, a catastrophe veto, or diagnostics (**W14** — Codex:
*"it cannot be all three"*). That is a separate open ruling and this row does not resolve it.
