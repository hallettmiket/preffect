"""Fitting a count distribution to one vector of counts.

Four kinds of check. First, that version 0.2 finds the same fit as version 0.1
wherever version 0.1 found one, so the port did not change the likelihood.
Second, that it finds a fit where version 0.1 silently returned nothing or
stopped short, which is findings 57 and 58. Third, that the fit recovers
parameters it was given, which neither version ever checked. Fourth, that the
comparison between the two distributions points the way the data does.

Where a test exists because of a specific defect, its name says which, and
`how_this_project_breaks.md` has the full account.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preffect.errors import DataError, FitError
from preffect.zinb_fit import (
    DEFAULT_STARTING_ZERO_PROBABILITIES,
    DISPERSION_BOUNDS,
    compare_fits,
    compare_fits_over_columns,
    fit_negative_binomial,
    fit_zero_inflated_negative_binomial,
)

GOLDEN = Path(__file__).parent / "data" / "version_0_1_zinb_fits.json"

#: How close the two versions' criteria have to be. Both maximise the same
#: likelihood, so the criterion is the quantity to compare: it is what the
#: search optimises and what the paper's conclusion rests on. The fitted
#: parameters are compared far more loosely, and on sparse counts not at all,
#: because the likelihood has a long flat ridge there along which the mean
#: trades against the zero-inflation probability. See the agreement tests
#: below for what that looks like in numbers.
AIC_TOLERANCE = 0.5


def load_cases():
    with open(GOLDEN) as fh:
        return json.load(fh)["cases"]


CASES = load_cases()
CASE_NAMES = [case["name"] for case in CASES]
FITTED_BY_VERSION_0_1 = [case for case in CASES if case["version_0_1_zinb"] is not None]
REFUSED_BY_VERSION_0_1 = [case for case in CASES if case["version_0_1_zinb"] is None]


def counts_of(case) -> np.ndarray:
    return np.asarray(case["counts"], dtype=np.float64)


def negative_binomial_counts(rng, size, mean, dispersion):
    """Counts with the given mean and inverse dispersion, in PREFFECT's naming."""
    return rng.negative_binomial(
        dispersion, dispersion / (dispersion + mean), size=size
    ).astype(np.float64)


# --------------------------------------------------------------------------
# Agreement with version 0.1.
#
# The reference values were produced by version 0.1 under its own environment
# (Python 3.9, numpy 1.26, scipy 1.13) and committed, so this runs without
# version 0.1 present. The count vectors are stored in the same file rather
# than regenerated, so the comparison does not depend on two numpy versions
# drawing the same numbers. Regenerate with tools/make_golden_zinb_fits.py.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", FITTED_BY_VERSION_0_1, ids=[c["name"] for c in FITTED_BY_VERSION_0_1]
)
def test_criterion_matches_version_0_1(case):
    fitted = fit_zero_inflated_negative_binomial(counts_of(case))
    assert fitted.aic == pytest.approx(
        case["version_0_1_zinb"]["aic"], abs=AIC_TOLERANCE
    )


@pytest.mark.parametrize(
    "case", FITTED_BY_VERSION_0_1, ids=[c["name"] for c in FITTED_BY_VERSION_0_1]
)
def test_the_port_never_settles_for_a_worse_fit_than_version_0_1(case):
    """The strong form of the agreement above, and the one worth keeping.

    Both versions maximise the same likelihood, so a lower criterion means a
    better fit and never a different answer to a different question. Version
    0.2 is ahead on every reference case, by between nothing at all and 0.34.
    """
    fitted = fit_zero_inflated_negative_binomial(counts_of(case))
    assert fitted.aic <= case["version_0_1_zinb"]["aic"] + 1e-6


@pytest.mark.parametrize(
    "case", FITTED_BY_VERSION_0_1, ids=[c["name"] for c in FITTED_BY_VERSION_0_1]
)
def test_the_mean_matches_version_0_1_where_the_fit_is_identified(case):
    """Away from sparse counts, the two versions agree on the parameters too.

    The sparse case is left out on purpose. Its likelihood has a ridge along
    which a low mean with few extra zeros fits as well as a higher mean with
    many, and the two versions sit at different points on it: version 0.1
    reports a mean of 0.384 with a zero probability of 0.010, version 0.2 a
    mean of 0.557 with 0.317, and their criteria differ by 0.34. Comparing
    the parameters there would be testing which end of a flat ridge the search
    stopped at.
    """
    if case["name"] == "sparse":
        pytest.skip("the parameters are not identified on sparse counts")
    fitted = fit_zero_inflated_negative_binomial(counts_of(case))
    assert fitted.mean == pytest.approx(case["version_0_1_zinb"]["mean"], rel=0.05)
    assert fitted.zero_probability == pytest.approx(
        case["version_0_1_zinb"]["zero_probability"], abs=0.01
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_NAMES)
def test_the_negative_binomial_matches_the_fit_the_paper_compared_against(case):
    """The paper's best-AIC notebooks fitted the negative binomial with statsmodels.

    Version 0.2 fits it with the same machinery as the zero-inflated side, so
    the comparison is like for like and statsmodels is not a dependency. The
    two agree to well under a tenth of a criterion unit on every case either
    can fit.
    """
    expected = case["paper_statsmodels_nb"]["aic"]
    if expected is None:
        pytest.skip("statsmodels could not fit this case either")
    if np.all(counts_of(case) == 0):
        pytest.skip("no mean is identified when every count is zero")
    fitted = fit_negative_binomial(counts_of(case))
    assert fitted.aic == pytest.approx(expected, abs=0.1)


def test_the_reference_covers_both_what_version_0_1_could_fit_and_what_it_could_not():
    """Otherwise half of these tests would be parameterised over nothing."""
    assert len(FITTED_BY_VERSION_0_1) >= 5
    assert len(REFUSED_BY_VERSION_0_1) >= 3


# --------------------------------------------------------------------------
# Finding 57. Version 0.1 returned nothing for counts whose variance falls
# below their mean, because every one of its eight searches started at the
# sample mean and the sample variance, and that point is not a negative
# binomial. On the DCIS cohort that is 26 of 19,231 genes.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", REFUSED_BY_VERSION_0_1, ids=[c["name"] for c in REFUSED_BY_VERSION_0_1]
)
def test_finding_57_counts_version_0_1_returned_nothing_for_are_fitted(case):
    values = counts_of(case)
    if np.all(values == 0):
        pytest.skip("checked separately: no distribution exists to fit")
    fitted = fit_zero_inflated_negative_binomial(values)
    assert np.isfinite(fitted.aic)
    assert fitted.dispersion > 0
    assert np.isfinite(fitted.dispersion)


def test_finding_57_an_under_dispersed_vector_is_fitted_as_a_poisson():
    """Counts tighter than a Poisson push the dispersion to its upper bound.

    That is the right answer rather than a failure: the negative binomial
    cannot have variance below its mean, so the closest it comes is the
    Poisson, and the fit says so through `dispersion_at_bound` instead of
    dividing by a negative number as version 0.1 did.
    """
    rng = np.random.default_rng(3)
    counts = rng.binomial(10, 0.3, size=400).astype(np.float64)
    assert counts.var() < counts.mean()

    fitted = fit_zero_inflated_negative_binomial(counts)
    assert fitted.dispersion_at_bound
    assert fitted.dispersion == pytest.approx(DISPERSION_BOUNDS[1])
    assert fitted.mean == pytest.approx(counts.mean(), rel=0.01)
    assert fitted.zero_probability < 1e-5


def test_finding_57_the_dispersion_is_never_negative_or_infinite():
    """Version 0.1 divided by the fitted variance minus the fitted mean.

    Nothing in its search kept that difference positive, and its own objective
    stays finite for a difference as low as minus 1e-10, at which point the
    dispersion it reports is about minus 4e12. Here the dispersion is a
    parameter with a positivity bound, so there is no division to go wrong.
    """
    rng = np.random.default_rng(4)
    awkward = [
        rng.poisson(1.5, size=300).astype(np.float64),
        rng.poisson(12.0, size=200).astype(np.float64),
        rng.binomial(20, 0.5, size=150).astype(np.float64),
        np.full(60, 3.0),
    ]
    for counts in awkward:
        for fitted in (
            fit_zero_inflated_negative_binomial(counts),
            fit_negative_binomial(counts),
        ):
            assert np.isfinite(fitted.dispersion)
            assert DISPERSION_BOUNDS[0] <= fitted.dispersion <= DISPERSION_BOUNDS[1]


# --------------------------------------------------------------------------
# Finding 58. The likelihood of heavily zero-inflated counts has a second
# peak, where the extra zeros are explained as overdispersion instead, and a
# search started near a zero-inflation probability of nothing walks into it.
# Version 0.1 reached the better peak from three of its eight starts; making
# the start consistent with the probability being tried raises that to six.
# The grid is what rescues both, so these tests exist to stop anyone deciding
# it is redundant.
# --------------------------------------------------------------------------


def per_start_criteria(counts):
    return [
        fit_zero_inflated_negative_binomial(counts, starting_zero_probabilities=(p,)).aic
        for p in DEFAULT_STARTING_ZERO_PROBABILITIES
    ]


def test_finding_58_the_grid_of_starts_is_doing_real_work():
    """At least one start walks into the wrong peak, so the grid is not optional.

    On these counts, the two lowest starting probabilities settle 194
    criterion units short of the best fit. Version 0.1 lost five of its eight
    that way, by 209 units.
    """
    case = next(c for c in CASES if c["name"] == "zero_inflated_heavy")
    per_start = per_start_criteria(counts_of(case))
    assert max(per_start) - min(per_start) > 100


def test_finding_58_the_grid_finds_the_best_fit_any_single_start_can():
    case = next(c for c in CASES if c["name"] == "zero_inflated_heavy")
    counts = counts_of(case)
    assert fit_zero_inflated_negative_binomial(counts).aic == pytest.approx(
        min(per_start_criteria(counts)), abs=1e-6
    )


def test_finding_58_a_start_consistent_with_its_own_assumption_loses_fewer_searches():
    """Version 0.1 started every search at the sample mean and variance.

    Those two numbers describe counts including their extra zeros, so on
    heavily zero-inflated counts they point the search at a component whose
    mean is far too low, whatever zero-inflation probability that search was
    meant to explore. Here the start is derived from the probability being
    tried. Three of version 0.1's eight searches reached the best fit; six of
    these eight do.
    """
    case = next(c for c in CASES if c["name"] == "zero_inflated_heavy")
    per_start = per_start_criteria(counts_of(case))
    best = min(per_start)
    reached = sum(1 for value in per_start if value < best + AIC_TOLERANCE)
    assert reached >= 6


def test_all_eight_starts_converge_on_ordinary_counts():
    """Version 0.1 lost three of its eight to an objective that became nan."""
    for case in FITTED_BY_VERSION_0_1:
        fitted = fit_zero_inflated_negative_binomial(counts_of(case))
        assert fitted.starts_converged == len(DEFAULT_STARTING_ZERO_PROBABILITIES), (
            f"{case['name']} lost "
            f"{len(DEFAULT_STARTING_ZERO_PROBABILITIES) - fitted.starts_converged} starts"
        )


# --------------------------------------------------------------------------
# Finding 57, the second half of it. Version 0.1 returned None rather than
# raising, so a caller fitting a distribution per gene saw a gene it had no
# answer for only if it thought to check.
# --------------------------------------------------------------------------


def test_finding_57_counts_that_are_all_zero_raise_rather_than_return_nothing():
    for fit in (fit_zero_inflated_negative_binomial, fit_negative_binomial):
        with pytest.raises(FitError, match="all 100 counts are zero"):
            fit(np.zeros(100))


def test_finding_57_the_error_says_how_many_counts_and_what_they_looked_like():
    """An error a reader cannot act on is barely better than returning nothing."""
    with pytest.raises(FitError) as raised:
        fit_zero_inflated_negative_binomial(np.zeros(7))
    assert "7" in str(raised.value)
    assert "identified" in str(raised.value)


# --------------------------------------------------------------------------
# Finding 60. Both versions asked the optimiser's success flag whether the
# search had converged. On a gene with counts in the thousands that flag is
# false at the top of the likelihood, because the line search can find no
# better step, and both versions threw the fit away.
# --------------------------------------------------------------------------


def highly_expressed_counts(rng, n=220):
    """Counts at the scale the flag goes wrong on: a mean in the thousands."""
    return negative_binomial_counts(rng, n, 3000.0, 1.0)


def test_finding_60_a_gene_with_counts_in_the_thousands_is_fitted():
    rng = np.random.default_rng(51)
    for _ in range(8):
        counts = highly_expressed_counts(rng)
        assert counts.mean() > 1000
        comparison = compare_fits(counts)
        assert np.isfinite(comparison.zero_inflated.aic)
        assert np.isfinite(comparison.negative_binomial.aic)


def test_finding_60_the_plain_fit_survives_what_the_flag_calls_a_failure():
    """The plain fit has one start, so it has nothing to fall back on.

    This is where the defect showed: eleven of 2,000 DCIS genes lost their
    plain fit, and so their whole comparison, while sitting at gradients
    between 1e-5 and 4e-4.
    """
    rng = np.random.default_rng(52)
    for _ in range(8):
        fitted = fit_negative_binomial(highly_expressed_counts(rng))
        assert np.isfinite(fitted.aic)
        assert fitted.mean > 1000


def test_finding_60_a_search_that_has_not_arrived_is_still_refused():
    """The flag was replaced by a check, not by nothing.

    A gradient that has not flattened means the search stopped somewhere that
    is not the top of the likelihood, and that is still a failure.
    """
    from preffect.zinb_fit import _has_arrived

    class Stopped:
        success = False
        fun = 1000.0
        jac = np.array([3.5, -0.2])

    class Diverged:
        success = False
        fun = float("inf")
        jac = np.array([0.0, 0.0])

    class Flat:
        success = False
        fun = 1000.0
        jac = np.array([1e-9, -2e-9])

    assert not _has_arrived(Stopped())
    assert not _has_arrived(Diverged())
    assert _has_arrived(Flat())


# --------------------------------------------------------------------------
# The fit recovers what it was given. Neither version ever checked this.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mean,dispersion,zero_probability",
    [(25.0, 3.0, 0.0), (40.0, 2.0, 0.35), (8.0, 5.0, 0.6), (200.0, 10.0, 0.2)],
)
def test_known_parameters_are_recovered_from_a_long_vector(
    mean, dispersion, zero_probability
):
    rng = np.random.default_rng(11)
    counts = negative_binomial_counts(rng, 20_000, mean, dispersion)
    if zero_probability:
        counts[rng.random(20_000) < zero_probability] = 0.0

    fitted = fit_zero_inflated_negative_binomial(counts)
    assert fitted.mean == pytest.approx(mean, rel=0.05)
    assert fitted.dispersion == pytest.approx(dispersion, rel=0.15)
    assert fitted.zero_probability == pytest.approx(zero_probability, abs=0.02)


def test_the_plain_fit_recovers_its_parameters_too():
    rng = np.random.default_rng(12)
    counts = negative_binomial_counts(rng, 20_000, 30.0, 4.0)
    fitted = fit_negative_binomial(counts)
    assert fitted.mean == pytest.approx(30.0, rel=0.03)
    assert fitted.dispersion == pytest.approx(4.0, rel=0.1)
    assert fitted.variance == pytest.approx(30.0 + 900.0 / 4.0, rel=0.15)


def test_the_reported_mean_belongs_to_the_component_not_the_whole_distribution():
    """Version 0.1 reported the component's mean, and so does this."""
    rng = np.random.default_rng(13)
    counts = negative_binomial_counts(rng, 20_000, 50.0, 6.0)
    counts[rng.random(20_000) < 0.5] = 0.0

    fitted = fit_zero_inflated_negative_binomial(counts)
    assert fitted.mean == pytest.approx(50.0, rel=0.05)
    assert fitted.observed_mean == pytest.approx(counts.mean(), rel=0.05)
    assert fitted.observed_mean < fitted.mean


# --------------------------------------------------------------------------
# The comparison between the two distributions.
# --------------------------------------------------------------------------


def test_zero_inflated_counts_prefer_the_zero_inflated_fit():
    rng = np.random.default_rng(21)
    counts = negative_binomial_counts(rng, 4_000, 30.0, 3.0)
    counts[rng.random(4_000) < 0.4] = 0.0

    comparison = compare_fits(counts)
    assert comparison.preferred == "ZINB"
    assert comparison.aic_difference > 100


def test_plain_counts_prefer_the_plain_fit():
    rng = np.random.default_rng(22)
    counts = negative_binomial_counts(rng, 4_000, 30.0, 3.0)

    comparison = compare_fits(counts)
    assert comparison.preferred == "NB"
    assert comparison.aic_difference < 0


def test_with_no_extra_zeros_the_two_criteria_differ_by_exactly_the_spare_parameter():
    """A zero-inflated fit with the probability at zero *is* a negative binomial.

    Its log-likelihood is then the same number, so the only thing left between
    the two criteria is the parameter it spent and did not use. Getting 2.0
    here means the two fits agree about everything they should.
    """
    rng = np.random.default_rng(23)
    counts = negative_binomial_counts(rng, 4_000, 15.0, 2.0)

    comparison = compare_fits(counts)
    assert comparison.zero_inflated.zero_probability < 1e-6
    assert comparison.aic_difference == pytest.approx(-2.0, abs=0.05)


@pytest.mark.parametrize("case", CASES, ids=CASE_NAMES)
def test_the_criterion_difference_is_the_plain_fit_minus_the_zero_inflated_one(case):
    """Signed the way the name says, so a reader cannot get the direction backwards."""
    if np.all(counts_of(case) == 0):
        pytest.skip("no distribution to fit")
    comparison = compare_fits(counts_of(case))
    assert comparison.aic_difference == pytest.approx(
        comparison.negative_binomial.aic - comparison.zero_inflated.aic
    )
    assert comparison.preferred == ("ZINB" if comparison.aic_difference > 0 else "NB")


def test_a_tie_goes_to_the_distribution_with_fewer_parameters():
    rng = np.random.default_rng(24)
    counts = negative_binomial_counts(rng, 500, 10.0, 2.0)
    comparison = compare_fits(counts)
    # No extra zeros to find, so the zero-inflated fit cannot win.
    assert comparison.preferred == "NB"


# --------------------------------------------------------------------------
# What is refused, and why. Version 0.1 passed whatever it was given to
# scipy's nbinom, which returns nan for anything that is not a whole number.
# --------------------------------------------------------------------------


def test_counts_that_are_not_whole_numbers_are_refused():
    with pytest.raises(DataError, match="not whole numbers"):
        fit_zero_inflated_negative_binomial(np.array([1.0, 2.5, 3.0]))


def test_negative_values_are_refused():
    with pytest.raises(DataError, match="negative"):
        fit_zero_inflated_negative_binomial(np.array([1.0, -2.0, 3.0]))


def test_missing_values_are_refused():
    with pytest.raises(DataError, match="missing or infinite"):
        fit_zero_inflated_negative_binomial(np.array([1.0, np.nan, 3.0]))


def test_an_empty_vector_is_refused():
    with pytest.raises(DataError, match="no counts"):
        fit_zero_inflated_negative_binomial(np.array([]))


def test_a_matrix_is_refused_with_the_name_of_the_call_that_takes_one():
    with pytest.raises(DataError, match="compare_fits_over_columns"):
        fit_zero_inflated_negative_binomial(np.ones((4, 5)))


def test_a_starting_probability_outside_the_unit_interval_is_refused():
    counts = np.array([0.0, 1.0, 5.0, 2.0, 0.0, 7.0])
    with pytest.raises(DataError, match="strictly"):
        fit_zero_inflated_negative_binomial(counts, starting_zero_probabilities=(0.0,))
    with pytest.raises(DataError, match="strictly"):
        fit_zero_inflated_negative_binomial(counts, starting_zero_probabilities=(1.0,))


def test_counts_from_an_integer_array_are_accepted():
    """Nobody should have to convert a count matrix to floats first."""
    rng = np.random.default_rng(31)
    counts = rng.integers(0, 40, size=200)
    assert counts.dtype.kind == "i"
    assert np.isfinite(fit_zero_inflated_negative_binomial(counts).aic)


# --------------------------------------------------------------------------
# Running over the columns of a matrix.
# --------------------------------------------------------------------------


def build_matrix(rng, n_samples=200):
    """Three genes: plain, zero-inflated, and one that cannot be fitted."""
    plain = negative_binomial_counts(rng, n_samples, 20.0, 2.0)
    inflated = negative_binomial_counts(rng, n_samples, 40.0, 3.0)
    inflated[rng.random(n_samples) < 0.5] = 0.0
    empty = np.zeros(n_samples)
    return np.column_stack([plain, inflated, empty])


def test_every_column_becomes_one_row_in_the_same_order():
    rng = np.random.default_rng(41)
    matrix = build_matrix(rng)[:, :2]

    table = compare_fits_over_columns(matrix, names=["plain", "inflated"])
    assert list(table["name"]) == ["plain", "inflated"]
    assert list(table["preferred"]) == ["NB", "ZINB"]
    assert table.shape[0] == 2


def test_a_dataframe_brings_its_own_column_names():
    rng = np.random.default_rng(42)
    matrix = build_matrix(rng)[:, :2]
    frame = pd.DataFrame(matrix, columns=["GENE_A", "GENE_B"])

    table = compare_fits_over_columns(frame)
    assert list(table["name"]) == ["GENE_A", "GENE_B"]


def test_a_column_that_cannot_be_fitted_stops_the_sweep_by_default():
    """A sweep that quietly drops genes turns a claim about a cohort into a
    claim about whichever subset happened to fit."""
    rng = np.random.default_rng(43)
    with pytest.raises(FitError, match="column '2'"):
        compare_fits_over_columns(build_matrix(rng))


def test_a_column_that_cannot_be_fitted_can_be_recorded_instead():
    rng = np.random.default_rng(44)
    table = compare_fits_over_columns(build_matrix(rng), on_failure="record")

    assert len(table) == 3
    assert list(table["problem"])[:2] == ["", ""]
    assert "all 200 counts are zero" in table["problem"].iloc[2]
    assert pd.isna(table["zinb_aic"].iloc[2])
    assert list(table["preferred"])[:2] == ["NB", "ZINB"]


def test_the_table_records_how_many_counts_and_how_many_were_zero():
    rng = np.random.default_rng(45)
    table = compare_fits_over_columns(build_matrix(rng), on_failure="record")
    assert list(table["n_counts"]) == [200, 200, 200]
    assert table["zero_fraction"].iloc[2] == 1.0
    assert table["zero_fraction"].iloc[1] > table["zero_fraction"].iloc[0]


def test_a_wrong_number_of_names_is_refused():
    rng = np.random.default_rng(46)
    with pytest.raises(DataError, match="they have to match"):
        compare_fits_over_columns(build_matrix(rng), names=["only_one"])


def test_a_vector_is_refused_with_the_name_of_the_call_that_takes_one():
    with pytest.raises(DataError, match="use compare_fits"):
        compare_fits_over_columns(np.ones(10))


def test_an_unknown_failure_mode_is_refused_rather_than_guessed_at():
    rng = np.random.default_rng(47)
    with pytest.raises(DataError, match="'raise' or 'record'"):
        compare_fits_over_columns(build_matrix(rng), on_failure="ignore")
