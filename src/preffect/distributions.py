"""The count distributions PREFFECT fits.

Two distributions, both for counts.

**Negative binomial.** A Poisson distribution says that if a gene is expressed
at some average level, the counts you see vary only as much as pure counting
noise allows. Real sequencing counts vary more than that, so the negative
binomial adds a second parameter for the extra spread. It arises by letting the
Poisson rate itself vary from sample to sample according to a gamma
distribution, which is why it is sometimes called the gamma-Poisson.

**Zero-inflated negative binomial.** The same thing, plus a separate
probability that a count is zero for a reason unrelated to expression level,
such as the transcript never being captured. The paper that accompanies version
0.1 found little evidence that fRNA-seq needs the zero-inflated form, and the
negative binomial is the default here for that reason, but both are available.

## How the parameters are named

``mu`` is the mean. ``theta`` is the **inverse** dispersion: large theta means
counts close to Poisson, small theta means heavily overdispersed. The variance
is ``mu + mu**2 / theta``, so theta appears in the denominator, and this is the
usual trap when reading the code against a textbook that calls the dispersion
``alpha = 1 / theta``.

``zero_logits`` is the log-odds of an extra zero, not a probability. Working in
log-odds keeps the arithmetic stable when the probability is near 0 or 1.

## Lineage

The parameterisation and the arrangement of the log-probability follow
scvi-tools (Gayoso, Lopez, Kim, Boyeau, Yosef and others; BSD 3-Clause), which
is what version 0.1 vendored. These implementations are written from the
gamma-Poisson definition rather than copied, and
``tests/test_distributions.py`` checks them against version 0.1's numbers to
within floating point tolerance, so the lineage is verified rather than
asserted.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "log_prob_negative_binomial",
    "log_prob_zero_inflated_negative_binomial",
    "NegativeBinomial",
    "ZeroInflatedNegativeBinomial",
]

#: Added inside logarithms so that a parameter of exactly zero does not produce
#: a negative infinity. Version 0.1 used the same value.
DEFAULT_EPS = 1e-8


def log_prob_negative_binomial(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Log probability of observing counts ``x``, element by element.

    :param x: observed counts.
    :param mu: mean, positive.
    :param theta: inverse dispersion, positive.
    """
    log_theta_mu = torch.log(theta + mu + eps)
    return (
        theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )


def log_prob_zero_inflated_negative_binomial(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    zero_logits: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Log probability under the zero-inflated negative binomial.

    A count of zero can arise two ways, through the extra-zero process or
    through the negative binomial itself, so the zero case sums both. Any other
    count can only come from the negative binomial.

    :param zero_logits: log-odds of the extra zero process, any sign.
    """
    softplus_pi = F.softplus(-zero_logits)
    log_theta_mu = torch.log(theta + mu + eps)
    pi_theta_log = -zero_logits + theta * (torch.log(theta + eps) - log_theta_mu)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    # Selected rather than branched, so both sides stay differentiable.
    is_zero = (x < eps).to(case_zero.dtype)
    return is_zero * case_zero + (1.0 - is_zero) * case_non_zero


class NegativeBinomial:
    """A negative binomial over counts, parameterised by mean and inverse dispersion.

    Deliberately not a subclass of :class:`torch.distributions.Distribution`.
    The only things PREFFECT asks of it are a log probability, a mean, a
    variance and a sample, and the constraint machinery that comes with the
    base class costs more than it gives here.
    """

    def __init__(self, mu: torch.Tensor, theta: torch.Tensor, eps: float = DEFAULT_EPS):
        if mu.shape != theta.shape:
            raise ValueError(
                f"mu is shaped {tuple(mu.shape)} and theta {tuple(theta.shape)}; "
                f"they have to match"
            )
        self.mu = mu
        self.theta = theta
        self.eps = eps

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    @property
    def variance(self) -> torch.Tensor:
        """``mu + mu**2 / theta``: Poisson variance plus the extra spread."""
        return self.mu + (self.mu**2) / self.theta

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log probability of each count in ``x`` under this distribution.

        One value per element, not a sum, so the caller decides what to reduce
        over. :func:`log_prob_negative_binomial` does the arithmetic; this is
        the method form, for code that holds a distribution object.
        """
        return log_prob_negative_binomial(x, self.mu, self.theta, self.eps)

    def sample(self, sample_shape: torch.Size | tuple[int, ...] = ()) -> torch.Tensor:
        """Draw counts, through the gamma-Poisson construction it is defined by."""
        shape = torch.Size(sample_shape)
        concentration = self.theta.expand(shape + self.theta.shape)
        rate = (self.theta / self.mu).expand(shape + self.mu.shape)
        rates = torch.distributions.Gamma(concentration, rate).sample()
        return torch.poisson(rates)

    def __repr__(self) -> str:
        return f"NegativeBinomial(shape={tuple(self.mu.shape)})"


class ZeroInflatedNegativeBinomial(NegativeBinomial):
    """A negative binomial with an extra source of zeros."""

    def __init__(
        self,
        mu: torch.Tensor,
        theta: torch.Tensor,
        zero_logits: torch.Tensor,
        eps: float = DEFAULT_EPS,
    ):
        super().__init__(mu=mu, theta=theta, eps=eps)
        if zero_logits.shape != mu.shape:
            raise ValueError(
                f"zero_logits is shaped {tuple(zero_logits.shape)} and mu "
                f"{tuple(mu.shape)}; they have to match"
            )
        self.zero_logits = zero_logits

    @property
    def zero_probs(self) -> torch.Tensor:
        """The extra-zero probability, converted from log-odds."""
        return torch.sigmoid(self.zero_logits)

    @property
    def mean(self) -> torch.Tensor:
        """Lower than ``mu``, because some counts are forced to zero."""
        return (1 - self.zero_probs) * self.mu

    @property
    def variance(self) -> torch.Tensor:
        pi = self.zero_probs
        nb_variance = self.mu + (self.mu**2) / self.theta
        return (1 - pi) * (nb_variance + pi * self.mu**2)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log probability of each count in ``x``, allowing for extra zeros.

        One value per element, not a sum. A zero can arise two ways here,
        either from the negative binomial or from the extra-zero component,
        and both routes are counted;
        :func:`log_prob_zero_inflated_negative_binomial` does the arithmetic.
        """
        return log_prob_zero_inflated_negative_binomial(
            x, self.mu, self.theta, self.zero_logits, self.eps
        )

    def sample(self, sample_shape: torch.Size | tuple[int, ...] = ()) -> torch.Tensor:
        counts = super().sample(sample_shape)
        forced_zero = torch.rand_like(counts) < self.zero_probs.expand_as(counts)
        return torch.where(forced_zero, torch.zeros_like(counts), counts)

    def __repr__(self) -> str:
        return f"ZeroInflatedNegativeBinomial(shape={tuple(self.mu.shape)})"
