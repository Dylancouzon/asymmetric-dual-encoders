"""The screen's decision rules, as pure functions, so they can be tested on examples.

Why this module exists. Four adversarial reviews passed over `m10/screen_registry.json` and none
caught that three of its rules do not implement their stated intent, because the validator checks
that a decision is NAMED, not that its behaviour is coherent (`screen_lock.py`). A rule written as
registry prose is only as good as the examples nobody ran against it. gpt-6-astra, 2026-09-08,
supplied the counterexamples; `research/m10-astra-v2-dispositions-2026-09-08.md` has the audit.

Each rule appears TWICE: `*_as_registered` reproduces the registry verbatim, defect included, and
`*_corrected` is the proposed replacement. The tests assert what each one does on the same inputs,
so the defect is a fact in CI rather than a paragraph in a review. When the amendment lands, the
`*_as_registered` pair and its defect test are deleted together.

**The registry is NOT edited here.** `results/m10_F_verdict.json` pins `registry_sha256` and
`run_arm.f_verdict` refuses every post-F arm if it changes, so amending the registry mid-chain
would halt the screen. Sequence: chain finishes -> amend -> re-issue the verdict -> then confirm.
"""
from __future__ import annotations


# --------------------------------------------------------------- 1. confirmation ----------------

def confirmation_stands_as_registered(winner_seeds, default_seeds):
    """`confirmation.stands_iff`: "the winner's margin exceeds the largest seed range observed in
    either arm", with `margin` = the ORIGINAL-screen (seed-0) difference and `seed_range` =
    "max minus min of an arm's COV macro over its three seeds".

    THE DEFECT: `seed_range` is spread in ABSOLUTE scores, while the decision depends on spread in
    the PAIRED difference. A seed that moves both arms together inflates both ranges without
    touching the effect, and a reversal that keeps both arms tightly clustered passes.
    """
    margin = winner_seeds[0] - default_seeds[0]
    seed_range = max(max(winner_seeds) - min(winner_seeds),
                     max(default_seeds) - min(default_seeds))
    return margin > seed_range


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


def e_after_confirmation_as_registered(e1_resolves, bs32_minus_bs128, seed_range=0.001):
    """E is in `confirmation_eligibility`, confirmation computes winner-minus-default, and
    `unconfirmed_non_defaults` reverts. THE DEFECT: when E picks bs128 on COST, its quality margin
    against the default is <= 0, so no non-negative seed range can be exceeded -- the cost branch
    can never survive its own confirmation and always reverts.
    """
    selected = e_selection(e1_resolves, bs32_minus_bs128)
    if selected == "bs32":
        return "bs32"                      # the default; nothing to confirm
    margin = -bs32_minus_bs128             # bs128 - bs32, negative whenever bs32 leads
    return "bs128" if margin > seed_range else "bs32"


def e_after_confirmation_corrected(e1_resolves, bs32_minus_bs128, seed_range=0.001):
    """A cost-motivated selection is not a quality claim, so it is not quality-confirmable.
    E is exempt from quality confirmation; the cost rule stands on its own.
    """
    return e_selection(e1_resolves, bs32_minus_bs128)


# ------------------------------------------------------------------- 3. family D ----------------

def d_selection_as_registered(dnorm_resolves, dnorm_margin, dcov_resolves, dcov_margin, tol=1e-4):
    """`rules.D_tie`: "if D-NORM and D-COV both resolve with margins within 1e-4, the default
    (squared L2) stands".

    THE DEFECT: it fires exactly when the evidence is STRONGEST -- two independent alternatives
    both resolving against the default -- and returns to the objective both of them beat.
    """
    if dnorm_resolves and dcov_resolves and abs(dnorm_margin - dcov_margin) <= tol:
        return "squared_l2"
    return _highest_resolving(dnorm_resolves, dnorm_margin, dcov_resolves, dcov_margin)


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
