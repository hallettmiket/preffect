"""Fitting a count distribution to one vector of counts, and saying which fits better.

This answers a question that comes before any model is trained: **do these
counts need a zero-inflated distribution at all?** It fits a zero-inflated
negative binomial and a plain negative binomial to the same counts and reports
both, with the Akaike information criterion for each so they can be compared.
The criterion is the fitted log-likelihood penalised by the number of
parameters, and the smaller value wins, so adding the zero-inflation parameter
has to buy enough extra fit to pay for itself.

Nothing here trains PREFFECT. This is the standalone estimator the paper that
accompanies version 0.1 used to decide that the negative binomial is enough for
fRNA-seq, and that conclusion is why
:class:`~preffect.config.Likelihood` defaults to the negative binomial. It is
here so the evidence can be reproduced and extended to new cohorts.

## What it reports

For the zero-inflated fit: the mean of the negative binomial component, the
inverse dispersion, the probability of an extra zero, the log-likelihood and
the criterion. For the plain fit: the same without the zero probability. Note
that the reported mean belongs to the **negative binomial component**, as it
did in version 0.1; the mean of the zero-inflated distribution as a whole is
``(1 - zero_probability) * mean``, which is lower.

``dispersion`` here is PREFFECT's ``theta``, the **inverse** dispersion: large
theta means counts close to Poisson, small theta means heavily overdispersed.
The variance is ``mu + mu**2 / theta``. See
:mod:`preffect.distributions`, which this module shares its arithmetic with.

## How it differs from version 0.1

Version 0.1 shipped this as ``wrappers/_ZINBEstimator.py``. The likelihood is
the same and the answers agree on counts version 0.1 could fit, but three
things changed. They are recorded as findings 57, 58 and 60 in
``how_this_project_breaks.md`` with the measurements, and in the decision of
2026-11-02 on this estimator.

1. **The search runs over the mean and the dispersion**, not over the mean and
   the variance with the dispersion divided out afterwards. Version 0.1's
   division could return a negative or an infinite dispersion, because nothing
   kept the fitted variance above the fitted mean. Here the dispersion is a
   parameter with a positivity bound, so there is nothing to divide.
2. **Counts whose variance falls below their mean are fitted**, rather than
   silently returning nothing. Version 0.1 started every one of its eight
   searches at the sample mean and variance, which for such counts is outside
   the model, so no search moved and the caller got ``None``.
3. **A fit that cannot be made raises**, naming the counts and the reason,
   instead of returning ``None`` for a caller to notice or not.

The gradient comes from :mod:`preffect.distributions` by automatic
differentiation rather than being estimated by finite differences, which is why
the search needs about a fifth as many evaluations of the likelihood.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from preffect.distributions import (
    log_prob_negative_binomial,
    log_prob_zero_inflated_negative_binomial,
)
from preffect.errors import DataError, FitError

__all__ = [
    "DEFAULT_STARTING_ZERO_PROBABILITIES",
    "DISPERSION_BOUNDS",
    "MEAN_BOUNDS",
    "FitComparison",
    "NegativeBinomialFit",
    "ZeroInflatedFit",
    "compare_fits",
    "compare_fits_over_columns",
    "fit_negative_binomial",
    "fit_zero_inflated_negative_binomial",
]

#: The zero-inflation probabilities the search starts from, one search each.
#: These are version 0.1's eight values, kept because the likelihood of a
#: heavily zero-inflated vector has a second peak where the extra zeros are
#: explained as overdispersion instead, and a search started near a probability
#: of nothing walks into it. On the reference case that peak is 194 criterion
#: units short of the best fit, so the grid is the method and not a precaution.
#: Finding 58 has the numbers.
DEFAULT_STARTING_ZERO_PROBABILITIES: tuple[float, ...] = (
    1e-5,
    1e-4,
    1e-3,
    5e-3,
    1e-2,
    0.1,
    0.5,
    0.9,
)

#: Smallest and largest mean the search will consider. The lower end keeps the
#: mean away from zero, where the log-likelihood has no gradient to follow.
MEAN_BOUNDS: tuple[float, float] = (1e-8, 1e12)

#: Smallest and largest inverse dispersion the search will consider. The upper
#: end is where the negative binomial has become a Poisson for any practical
#: purpose: at a mean of 1000 and a theta of 1e6, the extra spread is one part
#: in a thousand of the mean. Counts tighter than a Poisson push the search
#: against this bound, which is reported rather than hidden.
DISPERSION_BOUNDS: tuple[float, float] = (1e-6, 1e6)


@dataclass(frozen=True)
class NegativeBinomialFit:
    """The best negative binomial for one vector of counts.

    :param mean: fitted mean.
    :param dispersion: fitted inverse dispersion, PREFFECT's ``theta``. The
        variance is ``mean + mean**2 / dispersion``.
    :param log_likelihood: the fitted log-likelihood, summed over the counts.
    :param aic: the Akaike information criterion, ``2 * 2 - 2 *
        log_likelihood``, counting the mean and the dispersion as the two
        parameters. Smaller is a better fit for the number of parameters spent.
    :param dispersion_at_bound: the counts are at least as tight as a Poisson,
        so the dispersion is not identified and the value reported is the
        largest :data:`DISPERSION_BOUNDS` allows.
    """

    mean: float
    dispersion: float
    log_likelihood: float
    aic: float
    dispersion_at_bound: bool

    @property
    def variance(self) -> float:
        """The variance the fitted distribution has, ``mean + mean**2 / dispersion``."""
        return self.mean + self.mean**2 / self.dispersion


@dataclass(frozen=True)
class ZeroInflatedFit:
    """The best zero-inflated negative binomial for one vector of counts.

    :param mean: fitted mean **of the negative binomial component**, as version
        0.1 reported it. The mean of the whole distribution is
        ``(1 - zero_probability) * mean``.
    :param dispersion: fitted inverse dispersion of the negative binomial
        component, PREFFECT's ``theta``.
    :param zero_probability: the chance that a count is zero for a reason
        unrelated to expression level. Zero means the fit found no evidence of
        extra zeros and reduces to a plain negative binomial.
    :param log_likelihood: the fitted log-likelihood, summed over the counts.
    :param aic: the Akaike information criterion, ``2 * 3 - 2 *
        log_likelihood``, counting the mean, the dispersion and the zero
        probability as the three parameters.
    :param dispersion_at_bound: as for :class:`NegativeBinomialFit`.
    :param starts_converged: how many of the starting values in
        :data:`DEFAULT_STARTING_ZERO_PROBABILITIES` produced a search that
        converged. Fewer than all of them is not itself a problem, since only
        the best is kept, but one is worth looking at.
    """

    mean: float
    dispersion: float
    zero_probability: float
    log_likelihood: float
    aic: float
    dispersion_at_bound: bool
    starts_converged: int

    @property
    def observed_mean(self) -> float:
        """The mean of the whole distribution, extra zeros included."""
        return (1.0 - self.zero_probability) * self.mean


@dataclass(frozen=True)
class FitComparison:
    """Both fits for one vector of counts, and which of them the criterion prefers.

    :param zero_inflated: the zero-inflated negative binomial fit.
    :param negative_binomial: the plain negative binomial fit.
    :param preferred: ``"ZINB"`` or ``"NB"``, whichever has the smaller
        criterion. A tie goes to the plain fit, which spends one parameter
        fewer.
    :param aic_difference: the plain fit's criterion minus the zero-inflated
        one's. Positive means the extra zeros earned their parameter, and the
        size says by how much: a difference under 2 is usually read as the two
        fitting equally well.
    """

    zero_inflated: ZeroInflatedFit
    negative_binomial: NegativeBinomialFit
    preferred: str
    aic_difference: float


# --------------------------------------------------------------------------
# Checking the counts before fitting anything.
# --------------------------------------------------------------------------


def _as_counts(counts) -> np.ndarray:
    """Return ``counts`` as a one-dimensional array of whole numbers, or raise.

    Version 0.1 passed whatever it was given to ``scipy.stats.nbinom``, which
    returns ``nan`` for a count that is not a whole number, and ``nan``
    propagates through the sum into an objective the search cannot follow. The
    log-probabilities in :mod:`preffect.distributions` do not return ``nan``
    there; they return the continuous extension of the distribution, which is a
    finite number that is not a probability of anything. Refusing is the only
    honest option.
    """
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 1:
        raise DataError(
            f"expected one vector of counts, got an array of shape "
            f"{tuple(values.shape)}; for a matrix of samples by genes use "
            f"compare_fits_over_columns"
        )
    if values.size == 0:
        raise DataError("there are no counts to fit")
    if not np.all(np.isfinite(values)):
        raise DataError(
            "the counts contain missing or infinite values; remove or replace "
            "them before fitting, since the fit cannot say what they mean"
        )
    if np.any(values < 0):
        raise DataError(
            f"the counts contain negative values, the smallest being "
            f"{values.min():g}; a count distribution has no probability to give them"
        )
    if np.any(values != np.rint(values)):
        raise DataError(
            "the counts are not whole numbers, so they are not counts; round "
            "them yourself if that is what you intend, as the paper's "
            "notebooks did, rather than having it done behind your back"
        )
    return values


def _refuse_if_every_count_is_zero(values: np.ndarray) -> None:
    """Raise when there is nothing in the counts to fit a distribution to.

    With every count zero, any mean fits equally well as long as the extra-zero
    probability is one, so neither the mean nor the dispersion is identified.
    Version 0.1 returned ``None`` here, which a caller summing over genes reads
    as a gene it has no answer for only if it thought to check.
    """
    if np.all(values == 0):
        raise FitError(
            f"all {values.size} counts are zero, so neither the mean nor the "
            f"dispersion is identified; there is no distribution to fit, and a "
            f"gene like this is usually dropped before fitting"
        )


# --------------------------------------------------------------------------
# The objective, and its gradient, from version 0.2's own log-probabilities.
# --------------------------------------------------------------------------
#
# The search runs over the logarithms of the mean and the dispersion, and over
# the log-odds of the extra zero, so that every point it can reach is a
# distribution that exists. Version 0.1 searched over the mean and the variance
# with box bounds, where the whole region with variance below the mean is
# reachable and is not a negative binomial at all.


def _negative_log_likelihood_and_gradient(
    params: np.ndarray,
    counts: torch.Tensor,
    zero_inflated: bool,
) -> tuple[float, np.ndarray]:
    """Cost and gradient at one point, for :func:`scipy.optimize.minimize`.

    The log-probabilities keep their package default ``eps`` of 1e-8, so the
    likelihood maximised here is exactly the one the model is trained on. That
    is not only tidiness: the zero-inflated log-probability decides which
    counts are zero by testing ``x < eps``, so an ``eps`` of zero would classify
    every count as non-zero and the search would drive the zero-inflation
    probability to nothing on any data at all. The perturbation the ``eps``
    costs is a part in 1e8 of the mean, far below the criterion's resolution.
    """
    point = torch.tensor(params, dtype=torch.float64, requires_grad=True)
    mean = torch.exp(point[0])
    dispersion = torch.exp(point[1])
    if zero_inflated:
        per_count = log_prob_zero_inflated_negative_binomial(
            counts, mean, dispersion, point[2]
        )
    else:
        per_count = log_prob_negative_binomial(counts, mean, dispersion)
    cost = -per_count.sum()

    if not torch.isfinite(cost):
        # The search has stepped somewhere the likelihood cannot be evaluated.
        # Reporting an infinite cost with a flat gradient makes the line search
        # shorten its step, which is what should happen.
        return float("inf"), np.zeros_like(params)

    cost.backward()
    return float(cost.detach()), point.grad.detach().numpy().astype(np.float64)


def _starting_point(
    mean: float,
    variance: float,
    zero_probability: float,
) -> tuple[float, float]:
    """Mean and dispersion of the component, implied by the counts and an assumed pi.

    Version 0.1 started every search at the sample mean and the sample
    variance, whatever zero-inflation probability that search was exploring. On
    counts where most zeros really are extra, those two numbers describe the
    counts including the extra zeros, and the component they belong to has a
    much higher mean. Five of version 0.1's eight searches on the heavily
    zero-inflated reference case settled 209 criterion units short of the best
    fit for that reason, and a sixth 196 short; only three found it. With the
    start made consistent, six of the eight find it, and the fit they find is
    0.18 better than version 0.1's (finding 58).

    So the start is made consistent with the probability being tried: if a
    fraction ``pi`` of the zeros are extra, the component's mean is
    ``mean / (1 - pi)``, and matching the variance gives the dispersion. Where
    that leaves no spread for the negative binomial to account for, the start is
    a Poisson instead.
    """
    retained = max(1.0 - zero_probability, 1e-12)
    component_mean = max(mean / retained, MEAN_BOUNDS[0])
    spread = variance - mean - zero_probability * mean**2 / retained
    # Where there is no spread left for the negative binomial to account for,
    # the counts are at least as tight as a Poisson, so the start is the tight
    # end of the range and the search loosens it if the likelihood wants that.
    tight = DISPERSION_BOUNDS[1]
    dispersion = component_mean**2 * retained / spread if spread > 0 else tight
    dispersion = min(max(dispersion, DISPERSION_BOUNDS[0]), DISPERSION_BOUNDS[1])
    return component_mean, dispersion


def _log_bounds(zero_inflated: bool) -> list[tuple[float | None, float | None]]:
    bounds: list[tuple[float | None, float | None]] = [
        (math.log(MEAN_BOUNDS[0]), math.log(MEAN_BOUNDS[1])),
        (math.log(DISPERSION_BOUNDS[0]), math.log(DISPERSION_BOUNDS[1])),
    ]
    if zero_inflated:
        bounds.append((-40.0, 40.0))  # log-odds, so about 4e-18 to 1 - 4e-18
    return bounds


def _at_dispersion_bound(dispersion: float) -> bool:
    return dispersion >= DISPERSION_BOUNDS[1] * (1 - 1e-9)


#: How small the gradient has to be, relative to the cost itself, for a search
#: to count as having arrived. A search that stops with a gradient this flat
#: cannot move the log-likelihood by anything the criterion would notice.
GRADIENT_TOLERANCE = 1e-6


def _has_arrived(result) -> bool:
    """Whether a finished search is sitting at the top of the likelihood.

    ``result.success`` alone is not the question to ask. L-BFGS-B reports an
    abnormal line search whenever it cannot find a step that improves on where
    it already is, which at the top of the likelihood is exactly what should
    happen, and which happens more readily on a gene with counts in the
    thousands. On 2,000 genes of the DCIS cohort, eleven searches were reported
    as failures while sitting at gradients between 1e-5 and 4e-4, flatter than
    several of the searches reported as successes. Taking the flag at its word
    discarded eleven perfectly good fits.

    So the flag is accepted when it is given, and otherwise the gradient is
    looked at: a search that stops where the likelihood has no slope left has
    arrived whatever the line search made of its last step. Finding 60.
    """
    if not np.isfinite(result.fun):
        return False
    if result.success:
        return True
    gradient = np.asarray(result.jac, dtype=np.float64)
    if not np.all(np.isfinite(gradient)):
        return False
    scale = max(1.0, abs(float(result.fun)))
    return float(np.max(np.abs(gradient))) <= GRADIENT_TOLERANCE * scale


# --------------------------------------------------------------------------
# The two fits.
# --------------------------------------------------------------------------


def fit_negative_binomial(counts) -> NegativeBinomialFit:
    """Fit a negative binomial to one vector of counts by maximum likelihood.

    This is the companion the zero-inflated fit is compared against. The
    comparison only means something if both sides are fitted the same way, so
    this maximises the same likelihood with the same search rather than reading
    the mean and variance off the data, which is what the simplest of the
    paper's notebooks did. The difference that makes is measured in the
    decision of 2026-11-02 on this estimator: under a tenth of a criterion unit
    on counts without extra zeros, and 230 units on counts with them.

    :param counts: one vector of whole, non-negative numbers, one per sample.
    :raises DataError: the counts are not counts.
    :raises FitError: every count is zero, or no search converged.
    """
    values = _as_counts(counts)
    _refuse_if_every_count_is_zero(values)
    tensor = torch.tensor(values, dtype=torch.float64)

    start_mean, start_dispersion = _starting_point(
        float(values.mean()), float(values.var()), 0.0
    )
    result = minimize(
        _negative_log_likelihood_and_gradient,
        np.array([math.log(start_mean), math.log(start_dispersion)]),
        args=(tensor, False),
        jac=True,
        method="L-BFGS-B",
        bounds=_log_bounds(zero_inflated=False),
    )
    if not _has_arrived(result):
        raise FitError(
            f"the negative binomial fit did not converge on {values.size} counts "
            f"with mean {values.mean():g} and variance {values.var():g}; the "
            f"search reported: {result.message}"
        )

    mean = float(np.exp(result.x[0]))
    dispersion = float(np.exp(result.x[1]))
    log_likelihood = -float(result.fun)
    return NegativeBinomialFit(
        mean=mean,
        dispersion=dispersion,
        log_likelihood=log_likelihood,
        aic=2 * 2 - 2 * log_likelihood,
        dispersion_at_bound=_at_dispersion_bound(dispersion),
    )


def fit_zero_inflated_negative_binomial(
    counts,
    starting_zero_probabilities: tuple[float, ...] = DEFAULT_STARTING_ZERO_PROBABILITIES,
) -> ZeroInflatedFit:
    """Fit a zero-inflated negative binomial to one vector of counts.

    One search is run from each starting zero-inflation probability and the best
    fit is kept, because the likelihood of heavily zero-inflated counts has a
    second peak where the extra zeros are explained as overdispersion instead.

    :param counts: one vector of whole, non-negative numbers, one per sample.
    :param starting_zero_probabilities: the probabilities to start from, one
        search each. Defaults to version 0.1's eight values.
    :raises DataError: the counts are not counts, or a starting probability is
        not strictly between zero and one.
    :raises FitError: every count is zero, or no search converged.
    """
    values = _as_counts(counts)
    _refuse_if_every_count_is_zero(values)
    if not starting_zero_probabilities:
        raise DataError("no starting zero-inflation probabilities were given")
    for probability in starting_zero_probabilities:
        if not 0.0 < probability < 1.0:
            raise DataError(
                f"a starting zero-inflation probability has to lie strictly "
                f"between 0 and 1; got {probability!r}"
            )

    tensor = torch.tensor(values, dtype=torch.float64)
    sample_mean = float(values.mean())
    sample_variance = float(values.var())
    bounds = _log_bounds(zero_inflated=True)

    best = None
    converged = 0
    last_message = "no search was run"
    for probability in starting_zero_probabilities:
        start_mean, start_dispersion = _starting_point(
            sample_mean, sample_variance, probability
        )
        start = np.array(
            [
                math.log(start_mean),
                math.log(start_dispersion),
                math.log(probability / (1.0 - probability)),
            ]
        )
        result = minimize(
            _negative_log_likelihood_and_gradient,
            start,
            args=(tensor, True),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
        )
        last_message = str(result.message)
        if not _has_arrived(result):
            continue
        converged += 1
        if best is None or result.fun < best.fun:
            best = result

    if best is None:
        raise FitError(
            f"none of the {len(starting_zero_probabilities)} searches converged "
            f"on {values.size} counts with mean {sample_mean:g} and variance "
            f"{sample_variance:g}; the last one reported: {last_message}"
        )

    mean = float(np.exp(best.x[0]))
    dispersion = float(np.exp(best.x[1]))
    zero_probability = float(1.0 / (1.0 + np.exp(-best.x[2])))
    log_likelihood = -float(best.fun)
    return ZeroInflatedFit(
        mean=mean,
        dispersion=dispersion,
        zero_probability=zero_probability,
        log_likelihood=log_likelihood,
        aic=2 * 3 - 2 * log_likelihood,
        dispersion_at_bound=_at_dispersion_bound(dispersion),
        starts_converged=converged,
    )


def compare_fits(
    counts,
    starting_zero_probabilities: tuple[float, ...] = DEFAULT_STARTING_ZERO_PROBABILITIES,
) -> FitComparison:
    """Fit both distributions to one vector of counts and say which the criterion prefers.

    This is the call the paper's comparison is one of: both fits, both criteria,
    and the verdict, for one gene's counts across samples.

    :param counts: one vector of whole, non-negative numbers, one per sample.
    :param starting_zero_probabilities: passed to
        :func:`fit_zero_inflated_negative_binomial`.
    :raises DataError: the counts are not counts.
    :raises FitError: every count is zero, or either fit failed to converge.
    """
    zero_inflated = fit_zero_inflated_negative_binomial(
        counts, starting_zero_probabilities=starting_zero_probabilities
    )
    negative_binomial = fit_negative_binomial(counts)
    difference = negative_binomial.aic - zero_inflated.aic
    return FitComparison(
        zero_inflated=zero_inflated,
        negative_binomial=negative_binomial,
        preferred="ZINB" if difference > 0 else "NB",
        aic_difference=difference,
    )


#: The columns :func:`compare_fits_over_columns` returns, in order.
_TABLE_COLUMNS = [
    "name",
    "n_counts",
    "zero_fraction",
    "preferred",
    "aic_difference",
    "zinb_mean",
    "zinb_dispersion",
    "zinb_zero_probability",
    "zinb_log_likelihood",
    "zinb_aic",
    "nb_mean",
    "nb_dispersion",
    "nb_log_likelihood",
    "nb_aic",
    "problem",
]


def compare_fits_over_columns(
    matrix,
    names: list[str] | None = None,
    on_failure: str = "raise",
    starting_zero_probabilities: tuple[float, ...] = DEFAULT_STARTING_ZERO_PROBABILITIES,
) -> pd.DataFrame:
    """Run :func:`compare_fits` down every column of a counts matrix.

    A counts matrix here is samples by genes, the way PREFFECT and AnnData hold
    one, so a column is one gene across the cohort. The result has one row per
    column of the input, in the same order.

    :param matrix: a two-dimensional array or a ``DataFrame`` of samples by
        genes.
    :param names: a name per column. Taken from a ``DataFrame``'s columns when
        it is one and nothing is given, and otherwise from the column's
        position.
    :param on_failure: ``"raise"`` stops at the first column that cannot be
        fitted, which is the default because a sweep that quietly drops genes
        is how a conclusion about a cohort ends up being about a subset of it.
        ``"record"`` carries on and writes the reason into the row's
        ``problem`` column, leaving its numbers missing.
    :param starting_zero_probabilities: passed to
        :func:`fit_zero_inflated_negative_binomial`.
    :raises DataError: the input is not a matrix, the names do not match its
        columns, or ``on_failure`` is not one of the two words.
    :raises FitError: a column could not be fitted and ``on_failure`` is
        ``"raise"``.
    """
    if on_failure not in ("raise", "record"):
        raise DataError(
            f"on_failure has to be 'raise' or 'record'; got {on_failure!r}"
        )

    if isinstance(matrix, pd.DataFrame):
        if names is None:
            names = [str(column) for column in matrix.columns]
        values = matrix.to_numpy(dtype=np.float64)
    else:
        values = np.asarray(matrix, dtype=np.float64)

    if values.ndim != 2:
        raise DataError(
            f"expected a matrix of samples by genes, got an array of shape "
            f"{tuple(values.shape)}; for one vector of counts use compare_fits"
        )
    if names is None:
        names = [str(index) for index in range(values.shape[1])]
    if len(names) != values.shape[1]:
        raise DataError(
            f"got {len(names)} names for {values.shape[1]} columns; they have to match"
        )

    rows = []
    for index, name in enumerate(names):
        column = values[:, index]
        row: dict[str, object] = {
            "name": name,
            "n_counts": int(column.size),
            "zero_fraction": float(np.mean(column == 0)),
            "problem": "",
        }
        try:
            comparison = compare_fits(
                column, starting_zero_probabilities=starting_zero_probabilities
            )
        except (DataError, FitError) as problem:
            if on_failure == "raise":
                raise type(problem)(f"column {name!r}: {problem}") from problem
            row["problem"] = str(problem)
            rows.append(row)
            continue

        zinb, nb = comparison.zero_inflated, comparison.negative_binomial
        row.update(
            {
                "preferred": comparison.preferred,
                "aic_difference": comparison.aic_difference,
                "zinb_mean": zinb.mean,
                "zinb_dispersion": zinb.dispersion,
                "zinb_zero_probability": zinb.zero_probability,
                "zinb_log_likelihood": zinb.log_likelihood,
                "zinb_aic": zinb.aic,
                "nb_mean": nb.mean,
                "nb_dispersion": nb.dispersion,
                "nb_log_likelihood": nb.log_likelihood,
                "nb_aic": nb.aic,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=_TABLE_COLUMNS)
