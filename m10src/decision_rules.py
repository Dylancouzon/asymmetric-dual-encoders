"""The screen's decision rules, as pure functions, so they can be tested on examples.

Why this module exists. Four adversarial reviews passed over `m10/screen_registry.json` and none
caught that three of its rules did not implement their stated intent, because the validator checks
that a decision is NAMED, not that its behaviour is coherent (`screen_lock.py`). A rule written as
registry prose is only as good as the examples nobody ran against it. gpt-6-astra, 2026-09-08,
supplied the counterexamples; `research/m10-astra-v2-dispositions-2026-09-08.md` has the audit.

**The amendment LANDED 2026-09-09** (`m10/AMENDMENT_STAGED_2026-09-08.md`), after the W8 band-1
chain read COMPLETE and before any contrast was computed. The `*_as_registered` functions and their
defect tests were deleted with it, as this module said they would be; the defects themselves are on
the record in the registry's own amended rule text and in the disposition file. What remains is what
the registry now says.

`confirmation_stands_corrected` is kept as the recorded rule although the confirmation block itself
was CUT by the same amendment (§2, replaced by SYNTH-20M): nothing calls it in M10.
"""
from __future__ import annotations


# --------------------------------------------------------------- 1. confirmation ----------------

def confirmation_stands_corrected(winner_seeds, default_seeds):
    """The paired reading: the effect must SURVIVE the fresh seeds, not merely have been large once.

    Requires (a) every paired difference the same positive sign -- so a reversal on any fresh seed
    fails -- and (b) the fresh seeds' mean effect retains at least half the original, so a real but
    collapsing effect is not confirmed. Shared seed shifts cancel in the pairing and are therefore
    invisible to it, which is the point.
    """
    diffs = [w - d for w, d in zip(winner_seeds, default_seeds)]
    if not all(x > 0 for x in diffs):
        return False
    fresh = diffs[1:]
    if not fresh:
        return False
    return (sum(fresh) / len(fresh)) >= 0.5 * diffs[0]


# ------------------------------------------------------------------- 2. family E ----------------

def e_selection(e1_resolves, bs32_minus_bs128):
    """`rules.E_cost`: select bs32 iff E1 resolves; in every other case select bs128."""
    return "bs32" if e1_resolves else "bs128"


def e_after_confirmation_corrected(e1_resolves, bs32_minus_bs128, seed_range=0.001):
    """A cost-motivated selection is not a quality claim, so it is not quality-confirmable.
    E is exempt from quality confirmation; the cost rule stands on its own.
    """
    return e_selection(e1_resolves, bs32_minus_bs128)


# ------------------------------------------------------------------- 3. family D ----------------

def d_selection_corrected(dnorm_resolves, dnorm_margin, dcov_resolves, dcov_margin, tol=1e-4):
    """`rules.multi_arm_winner` already gives the coherent rule: among alternatives whose own
    default comparison resolves, take the highest point estimate, with NO alternative-vs-alternative
    claim. A near-exact tie between two resolving winners is broken deterministically (D-NORM, the
    cheaper implementation change) and reported as a tie -- never by reverting to the loser.
    """
    if dnorm_resolves and dcov_resolves and abs(dnorm_margin - dcov_margin) <= tol:
        return "leaf_norm_e2"              # deterministic tie-break, both beat the default
    return _highest_resolving(dnorm_resolves, dnorm_margin, dcov_resolves, dcov_margin)


def _highest_resolving(dnorm_resolves, dnorm_margin, dcov_resolves, dcov_margin):
    cands = []
    if dnorm_resolves:
        cands.append((dnorm_margin, "leaf_norm_e2"))
    if dcov_resolves:
        cands.append((dcov_margin, "document_covariance_weighted"))
    if not cands:
        return "squared_l2"
    return max(cands)[1]
