"""The experiment harness.

Nothing tested `exp/` until finding 27, which is an experiment that compared a
model with itself for a day while a decision record cited it. These tests cover
the part that decides what an experiment *concluded*, because that is the part
whose mistakes end up in a decision file rather than in a traceback.

See issue #15 for the rest: a test per experiment that its arms actually differ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "exp"
if not EXPERIMENTS.is_dir():
    pytest.skip("exp/ is part of the development record, not of a release", allow_module_level=True)
sys.path.insert(0, str(EXPERIMENTS))

from common import _verdict, compare_paired  # noqa: E402

# --------------------------------------------------------------------------
# The arms are paired, and the rule this replaced threw that away.
#
# run_comparison seeds before building each model and gives both arms of a seed
# the same data, so they start from identical weights, see the same batch order
# and draw the same noise. Everything that varies between seeds cancels inside
# a pair. The old rule compared the difference of means against the across-seed
# spread of levels, which is exactly the variation the pairing removes.
# --------------------------------------------------------------------------


def test_a_small_consistent_difference_is_not_noise():
    """The case the old rule got wrong: tiny gap, same sign every time."""
    a = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    b = [a_ + 1.0 for a_ in a]  # dwarfed by the spread of the levels

    paired = compare_paired(a, b)
    assert paired["consistent_signs"] == 8
    assert paired["sign_test_p"] == pytest.approx(0.0078, abs=1e-4)
    assert _verdict(paired) == "8/8 p=0.008"


def test_a_large_inconsistent_difference_is_noise():
    a = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    b = [150.0, 150.0, 350.0, 350.0, 550.0, 550.0]  # big gaps, no agreed direction
    assert _verdict(compare_paired(a, b)) == "within noise"


def test_one_wild_seed_does_not_decide_the_verdict():
    """A paired t would be dragged about by this; a sign test is not.

    ``exp/03``'s seed 5 gave one arm a reconstruction half again as large as any
    other seed produced. That moves a mean and a standard deviation a long way
    and moves a count of signs not at all.
    """
    a = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    b = [10.1, 10.1, 10.1, 10.1, 10.1, 900.0]
    paired = compare_paired(a, b)
    assert paired["consistent_signs"] == 6
    assert _verdict(paired) == "6/6 p=0.031"


# --------------------------------------------------------------------------
# Ties are dropped, not counted as agreement.
#
# exp/01's fraction_variance_below_one has five exact ties in eight seeds and
# three differences that agree. Counting ties as agreement reports "8 of 8,
# p = 0.008", which is a finding. Dropping them reports "3 of 3, p = 0.25",
# which is not. An audit reported the first of those.
# --------------------------------------------------------------------------


def test_ties_are_dropped_before_the_sign_test():
    differences_with_five_ties = ([1.0] * 8, [0.9998, 1.0, 1.0, 1.0, 0.9998, 0.9998, 1.0, 1.0])
    paired = compare_paired(*differences_with_five_ties)

    assert paired["ties"] == 5
    assert paired["n_decided"] == 3
    assert paired["consistent_signs"] == 3
    assert paired["sign_test_p"] == pytest.approx(0.25)

    # These are exp/01's real values. Eight seeds were run, five landed on the
    # same number to the digit, and three cannot settle anything. "Too few" is
    # a more honest reading than either "within noise" or "8 of 8, p = 0.008".
    assert _verdict(paired) == "5/8 tied"


def test_counting_ties_as_agreement_would_have_called_it_a_finding():
    """Guards the specific arithmetic, so the mistake cannot come back quietly."""
    from common import _sign_test_p

    assert _sign_test_p(8, 8) == pytest.approx(0.0078, abs=1e-4)  # what was reported
    assert _sign_test_p(3, 3) == pytest.approx(0.25)  # what is true


# --------------------------------------------------------------------------
# Finding 27: two arms that are secretly the same model say so.
#
# They used to report "within noise" in every measure, which reads like an
# ordinary negative result. It is not; it is a broken experiment.
# --------------------------------------------------------------------------


def test_identical_arms_are_called_identical_not_noise():
    values = [4807.32, 4953.05, 4880.18]
    paired = compare_paired(values, list(values))

    assert paired["all_identical"]
    assert _verdict(paired) == "IDENTICAL"
    assert _verdict(paired) != "within noise"


def test_almost_identical_arms_are_not_called_identical():
    """One differing bit is a real comparison, however small."""
    paired = compare_paired([1.0, 2.0, 3.0], [1.0, 2.0, 3.0000001])
    assert not paired["all_identical"]
    assert _verdict(paired) != "IDENTICAL"


# --------------------------------------------------------------------------
# Arithmetic.
# --------------------------------------------------------------------------


def test_the_sign_test_is_two_sided():
    from common import _sign_test_p

    assert _sign_test_p(5, 5) == pytest.approx(2 / 32)
    assert _sign_test_p(4, 5) == pytest.approx(2 * 6 / 32)


def test_the_paired_t_is_the_mean_difference_over_its_standard_error():
    import math
    import statistics

    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 4.0, 5.0, 9.0]
    differences = [y - x for x, y in zip(a, b, strict=True)]
    expected = statistics.fmean(differences) / (
        statistics.stdev(differences) / math.sqrt(len(differences))
    )
    assert compare_paired(a, b)["paired_t"] == pytest.approx(expected)


def test_arms_of_different_length_are_refused():
    with pytest.raises(ValueError):
        compare_paired([1.0, 2.0], [1.0])


def test_a_flat_difference_has_no_t_but_still_has_a_verdict():
    """Every seed differs by exactly the same amount, so there is no spread."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    paired = compare_paired(a, [x + 1.0 for x in a])
    assert paired["paired_t"] is None
    assert _verdict(paired) == "6/6 p=0.031"  # the sign test still works


# --------------------------------------------------------------------------
# A short run must not read as a negative result.
#
# The smallest two-sided sign-test probability is 2 / 2**n, which first drops
# below 0.05 at six seeds. Reporting "within noise" at four seeds would be
# reporting a conclusion the test is incapable of reaching.
# --------------------------------------------------------------------------


def test_a_four_seed_run_says_too_few_rather_than_within_noise():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 3.5, 4.2, 5.9]  # separates cleanly every time
    paired = compare_paired(a, b)

    assert paired["consistent_signs"] == 4
    assert paired["sign_test_p"] > 0.05  # the best four seeds can do is 0.125
    assert _verdict(paired) == "too few (4)"


def test_six_seeds_is_enough_to_decide():
    a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    b = [x + 0.5 for x in a]
    assert _verdict(compare_paired(a, b)) == "6/6 p=0.031"


def test_the_threshold_matches_the_arithmetic_it_claims():
    from common import MIN_SEEDS_TO_DECIDE, _sign_test_p

    assert _sign_test_p(MIN_SEEDS_TO_DECIDE, MIN_SEEDS_TO_DECIDE) <= 0.05
    assert _sign_test_p(MIN_SEEDS_TO_DECIDE - 1, MIN_SEEDS_TO_DECIDE - 1) > 0.05


def test_a_measure_that_is_mostly_ties_says_so_rather_than_too_few():
    """Ties and a short run both block a verdict, for different reasons.

    Ties mean the experiment has no resolution on that measure. A short run
    means there were not enough seeds. Reporting both as "too few" hides which
    one a longer run would fix.
    """
    mostly_tied = compare_paired([1.0] * 8, [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0])
    assert _verdict(mostly_tied) == "7/8 tied"

    short_but_clean = compare_paired([1.0, 2.0, 3.0], [1.5, 2.5, 3.5])
    assert short_but_clean["ties"] == 0
    assert _verdict(short_but_clean) == "too few (3)"
