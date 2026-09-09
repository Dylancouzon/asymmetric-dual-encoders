"""Executable decision examples for the screen's three amended rules.

astra's recommendation #1, and the cheapest item in its whole review: four adversarial passes
missed the defects because `screen_lock.py` validates that a decision is NAMED, not that it
BEHAVES. **The amendment landed 2026-09-09**, so the `*_as_registered` tests that asserted the
wrong answers were deleted with the functions they guarded; what is left pins the behaviour the
registry now registers, including the counterexamples that motivated each fix.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decision_rules as DR


# ------------------------------------------------------- 1. confirmation, both directions --------

def test_confirmation_confirms_a_perfect_replication():
    """A seed that shifts BOTH arms equally leaves the effect identical on every seed -- +0.020
    three times, replication as perfect as it gets. The superseded rule threw it away, because the
    absolute-score range (0.040) exceeded the margin (0.020); the paired rule confirms it, because
    shared seed shifts cancel in the pairing."""
    winner, default = [.520, .480, .480], [.500, .460, .460]
    diffs = [round(w - d, 6) for w, d in zip(winner, default)]
    assert diffs == [.020, .020, .020], "the effect is identical on all three seeds"
    assert DR.confirmation_stands_corrected(winner, default) is True


def test_confirmation_refuses_a_reversal_on_both_fresh_seeds():
    """The effect REVERSES on both confirmation seeds (-0.008 twice). The superseded rule confirmed
    it anyway -- each arm stayed inside a 0.014 band while the retained seed-0 margin was 0.020 --
    and that was the defect that could change the build."""
    winner, default = [.520, .506, .506], [.500, .514, .514]
    diffs = [round(w - d, 6) for w, d in zip(winner, default)]
    assert diffs == [.020, -.008, -.008]
    assert DR.confirmation_stands_corrected(winner, default) is False


def test_corrected_confirmation_still_refuses_a_collapsing_effect():
    """The fix must not merely flip every answer: an effect that keeps its sign but collapses to a
    tenth of its original size is not a confirmation either."""
    winner, default = [.520, .5012, .5012], [.500, .500, .500]
    assert DR.confirmation_stands_corrected(winner, default) is False
    # and a genuine, sign-stable, undiminished effect passes
    assert DR.confirmation_stands_corrected([.520, .519, .521], [.500, .500, .500]) is True


# ------------------------------------------------------------------ 2. family E ------------------

def test_E_cost_branch_survives():
    """E deliberately accepts a small UNRESOLVED quality disadvantage to save build cost, selecting
    bs128. Under the superseded rule confirmation computed bs128 - bs32 = -0.003, which cannot
    exceed a non-negative seed range, so the selection reverted to bs32 and the saving evaporated:
    a quality-superiority rule cannot implement a cost decision. E is now exempt."""
    assert DR.e_selection(False, 0.003) == "bs128", "E's own table picks bs128 when unresolved"
    assert DR.e_after_confirmation_corrected(False, 0.003) == "bs128"


def test_E_resolved_bs32_win_is_unaffected_either_way():
    """The fix must not touch the case E was designed for: a RESOLVED bs32 win selects bs32."""
    assert DR.e_after_confirmation_corrected(True, 0.012) == "bs32"


# ------------------------------------------------------------------ 3. family D ------------------

def test_D_tie_keeps_a_winner():
    """Both alternatives RESOLVE against squared L2 at +0.015, agreeing to within 1e-4 -- the
    strongest evidence the design can produce. The superseded tie rule returned to the objective
    both of them beat; the amended one keeps a winner and breaks the tie deterministically."""
    assert DR.d_selection_corrected(True, 0.0150, True, 0.01505) == "leaf_norm_e2"


def test_D_selection_unchanged_where_the_rule_was_already_right():
    """Only the near-exact-tie branch changes."""
    cases = [(True, 0.020, True, 0.010, "leaf_norm_e2"),
             (True, 0.010, True, 0.020, "document_covariance_weighted"),
             (False, 0.0, True, 0.012, "document_covariance_weighted"),
             (False, 0.0, False, 0.0, "squared_l2")]
    for a, am, b, bm, want in cases:
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
