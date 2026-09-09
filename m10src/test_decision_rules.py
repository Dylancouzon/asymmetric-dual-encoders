"""Executable decision examples for the screen's three defective rules.

astra's recommendation #1, and the cheapest item in its whole review: four adversarial passes
missed these because `screen_lock.py` validates that a decision is NAMED, not that it BEHAVES.
These tests put the defect in CI. Each `*_as_registered` test asserts the WRONG answer the current
registry gives -- it passes today, and it is deleted when the amendment lands, together with the
function it guards.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decision_rules as DR


# ------------------------------------------------------- 1. confirmation, both directions --------

def test_registered_confirmation_REJECTS_perfect_replication():
    """A seed that shifts BOTH arms equally leaves the effect identical on every seed -- +0.020
    three times, replication as perfect as it gets -- and the registered rule throws it away,
    because the absolute-score range (0.040) exceeds the margin (0.020)."""
    winner, default = [.520, .480, .480], [.500, .460, .460]
    diffs = [round(w - d, 6) for w, d in zip(winner, default)]
    assert diffs == [.020, .020, .020], "the effect is identical on all three seeds"
    assert DR.confirmation_stands_as_registered(winner, default) is False, \
        "the registered rule rejects a perfectly replicated effect"
    assert DR.confirmation_stands_corrected(winner, default) is True, \
        "the paired rule confirms it, because shared seed shifts cancel in the pairing"


def test_registered_confirmation_ACCEPTS_a_reversal_on_both_fresh_seeds():
    """The effect REVERSES on both confirmation seeds (-0.008 twice) and the registered rule
    confirms it anyway, because each arm stayed inside a 0.014 band while the retained seed-0
    margin was 0.020. This is the one that can change the build."""
    winner, default = [.520, .506, .506], [.500, .514, .514]
    diffs = [round(w - d, 6) for w, d in zip(winner, default)]
    assert diffs == [.020, -.008, -.008]
    assert DR.confirmation_stands_as_registered(winner, default) is True, \
        "the registered rule confirms a winner that lost on both fresh seeds"
    assert DR.confirmation_stands_corrected(winner, default) is False, \
        "the paired rule refuses any sign reversal"


def test_corrected_confirmation_still_refuses_a_collapsing_effect():
    """The fix must not merely flip every answer: an effect that keeps its sign but collapses to a
    tenth of its original size is not a confirmation either."""
    winner, default = [.520, .5012, .5012], [.500, .500, .500]
    assert DR.confirmation_stands_corrected(winner, default) is False
    # and a genuine, sign-stable, undiminished effect passes
    assert DR.confirmation_stands_corrected([.520, .519, .521], [.500, .500, .500]) is True


# ------------------------------------------------------------------ 2. family E ------------------

def test_registered_E_cost_branch_is_unreachable_through_confirmation():
    """E deliberately accepts a small UNRESOLVED quality disadvantage to save build cost, selecting
    bs128. Confirmation then computes bs128 - bs32 = -0.003, which cannot exceed a non-negative
    seed range, so `unconfirmed_non_defaults` reverts to bs32 and the saving evaporates. A
    quality-superiority rule cannot implement a cost decision."""
    assert DR.e_selection(False, 0.003) == "bs128", "E's own table picks bs128 when unresolved"
    assert DR.e_after_confirmation_as_registered(False, 0.003) == "bs32", \
        "confirmation silently undoes E's cost decision"
    assert DR.e_after_confirmation_corrected(False, 0.003) == "bs128", \
        "exempting a cost decision from quality confirmation preserves it"


def test_E_resolved_bs32_win_is_unaffected_either_way():
    """The fix must not touch the case E was designed for: a RESOLVED bs32 win selects bs32."""
    for fn in (DR.e_after_confirmation_as_registered, DR.e_after_confirmation_corrected):
        assert fn(True, 0.012) == "bs32"


# ------------------------------------------------------------------ 3. family D ------------------

def test_registered_D_tie_discards_two_resolved_improvements():
    """Both alternatives RESOLVE against squared L2 at +0.015, agreeing to within 1e-4 -- the
    strongest evidence the design can produce -- and the registered rule returns to the objective
    both of them beat."""
    assert DR.d_selection_as_registered(True, 0.0150, True, 0.01505) == "squared_l2", \
        "the registered tie rule reverts to the loser"
    assert DR.d_selection_corrected(True, 0.0150, True, 0.01505) == "leaf_norm_e2", \
        "the corrected rule keeps a winner and breaks the tie deterministically"


def test_D_selection_unchanged_where_the_rule_was_already_right():
    """Only the near-exact-tie branch changes."""
    cases = [(True, 0.020, True, 0.010, "leaf_norm_e2"),
             (True, 0.010, True, 0.020, "document_covariance_weighted"),
             (False, 0.0, True, 0.012, "document_covariance_weighted"),
             (False, 0.0, False, 0.0, "squared_l2")]
    for a, am, b, bm, want in cases:
        assert DR.d_selection_as_registered(a, am, b, bm) == want
        assert DR.d_selection_corrected(a, am, b, bm) == want


def test_corrected_confirmation_refuses_a_SIGN_FLIP_that_averages_out_positive():
    """The case the sign-stability clause exists for, and the one my first mutation test missed.

    Diffs +0.020 / -0.030 / +0.070: the fresh seeds AVERAGE +0.020, which clears the
    retain-half-the-effect bar outright, so the magnitude check alone would confirm it. But the
    effect reversed on a fresh seed, which is precisely the evidence that it is not real. Deleting
    the sign guard leaves every other test in this module green -- verified by mutation -- so this
    test is the only thing holding it."""
    winner, default = [.520, .470, .570], [.500, .500, .500]
    diffs = [round(w - d, 6) for w, d in zip(winner, default)]
    assert diffs == [.020, -.030, .070]
    fresh_mean = sum(diffs[1:]) / 2
    assert fresh_mean >= 0.5 * diffs[0], "the magnitude bar is cleared, so only the sign guard binds"
    assert DR.confirmation_stands_corrected(winner, default) is False
