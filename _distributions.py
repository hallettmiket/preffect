# BSD 3-Clause License
#
# Copyright (c) 2023, Adam Gayoso, Romain Lopez,
# Martin Kim, Pierre Boyeau, Nir Yosef

import warnings
from preffect._utils import check_for_nans
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F

from torch.distributions import Distribution, Gamma, constraints
from torch.distributions import Poisson as PoissonTorch
from torch.distributions.utils import (
    broadcast_all,
    logits_to_probs,
    probs_to_logits
)

import logging
input_log = logging.getLogger('input')
forward_log = logging.getLogger('forward')


def _gamma(theta, mu):
    """
    Generate a Gamma distribution object given shape and rate parameters.

    This function creates a Gamma distribution which is parameterized by the
    concentration (shape parameter, `theta`) and the rate (inverse of the scale,
    `theta/mu`).

    Args:
        theta (torch.Tensor): The shape parameter of the Gamma distribution, often denoted
                              as `k` or `alpha`, representing the concentration.
        mu (torch.Tensor): The mean of the distribution, used here to calculate the rate
                           as `theta/mu`.

    Returns:
        torch.distributions.Gamma: A Gamma distribution object parameterized by
                                   `concentration` and `rate`.
    """
    concentration = theta
    rate = theta / mu
    # Important remark: Gamma is parametrized by the rate = 1/scale!
    gamma_d = Gamma(concentration=concentration, rate=rate)
    return gamma_d


def _convert_counts_logits_to_mean_disp(total_count, logits):
    """
    NB parameterizations conversion.

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    logits
        success logits.

    Returns
    -------
    type
        the mean and inverse overdispersion of the NB distribution.

    """
    theta = total_count
    mu = logits.exp() * theta
    return mu, theta


class Poisson(PoissonTorch):
    r"""
    Poisson distribution.

    The Poisson distribution is a discrete probability distribution that expresses the probability 
    of a given number of events occurring in a fixed interval of time or space, if these events 
    occur with a known constant mean rate and independently of the time since the last event.

    The Poisson distribution is characterized by a single parameter :math:`\lambda` that represents
    the average rate of occurrence. It is commonly used to model the number of times a random 
    event occurs in a given amount of time, distance, area, etc.

    :param rate: Rate parameter :math:`\lambda` of the Poisson distribution.
    :type rate: torch.Tensor
    :param validate_args: Whether to validate input arguments.
    :type validate_args: bool, optional
    :param scale: Normalized mean expression of the distribution.
    :type scale: torch.Tensor, optional
    """

    def __init__(
        self,
        rate: torch.Tensor,
        validate_args: Optional[bool] = None,
        scale: Optional[torch.Tensor] = None,
    ):
        super().__init__(rate=rate, validate_args=validate_args)
        self.scale = scale


class NegativeBinomial(Distribution):
    r"""
    Negative binomial distribution.

    One of the following parameterizations must be provided:

    1. (`total_count`, `probs`) where:
       - `total_count` is the number of failures until the experiment is stopped
       - `probs` is the success probability
    2. (`mu`, `theta`) parameterization, which is the one used by scvi-tools, where:
       - `mu` controls the mean of the distribution
       - `theta` controls the inverse dispersion of the distribution

    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as follows:
    
    
    .. math::

        w \sim \textrm{Gamma}\left(\theta, \frac{\theta}{\mu}\right)

        x \sim \textrm{Poisson}(w)

    :param total_count: Number of failures until the experiment is stopped.
    :type total_count: torch.Tensor, optional
    :param probs: The success probability.
    :type probs: torch.Tensor, optional
    :param mu: Mean of the distribution.
    :type mu: torch.Tensor, optional
    :param theta: Inverse dispersion.
    :type theta: torch.Tensor, optional
    :param scale: Normalized mean expression of the distribution.
    :type scale: torch.Tensor, optional
    :param validate_args: Raise ValueError if arguments do not match constraints.
    :type validate_args: bool, optional
    
    .. attribute:: arg_constraints

       Dictionary of argument constraints for the distribution parameters:
       
       - `mu` must be greater than or equal to 0
       - `theta` must be greater than or equal to 0  
       - `scale` must be greater than or equal to 0

    .. attribute:: support

       Constraints on the support of the distribution (non-negative integers).
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "scale": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. \
                    Refer to the documentation for more information."
            )

        using_param_1 = total_count is not None and (
            logits is not None or probs is not None
        )
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs)
            total_count = total_count.type_as(logits)
            total_count, logits = broadcast_all(total_count, logits)
            mu, theta = _convert_counts_logits_to_mean_disp(
                total_count, logits)
        else:
            mu, theta = broadcast_all(mu, theta)
        self.mu = mu
        self.theta = theta
        self.scale = scale
        super().__init__(validate_args=validate_args)

    def mean(self):
        """
        Calculate the mean of the Negative Binomial distribution.

        This method computes the expected value (mean) of the distribution,
        which is defined as `mu` for the Negative Binomial distribution.

        Returns:
            torch.Tensor: The mean of the distribution.
        """
        return self.mu

    def variance(self):
        """
        Calculate the variance of the Negative Binomial distribution.

        This method computes the variance of the distribution, which is derived from
        the formula: `mean + (mean^2) / theta`.

        Returns:
            torch.Tensor: The variance of the distribution.
        """
        return self.mean + (self.mean**2) / self.theta

    def sample(
        self,
        sample_shape: Optional[Union[torch.Size, Tuple]] = None,
    ) -> torch.Tensor:
        """
        Generate samples from the Negative Binomial distribution.

        This method samples from the distribution using the gamma-Poisson mixture:
        a gamma distribution to generate rate parameters, and a Poisson distribution
        to generate the final count data.

        Args:
            sample_shape (Union[torch.Size, Tuple], optional): The shape of the tensor to generate.
                Defaults to torch.Size(), which generates a single sample.

        Returns:
            torch.Tensor: Samples from the Negative Binomial distribution.
        """
        sample_shape = sample_shape or torch.Size()
        gamma_d = self._gamma()
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = torch.clamp(p_means, max=1e8)
        counts = PoissonTorch(
            l_train
        ).sample()  # Shape : (n_samples, n_cells_batch, n_vars)
        return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log probability of a given value under the Negative Binomial distribution.

        This method computes the log likelihood of observing a given count value under the
        specified Negative Binomial distribution parameters.

        Args:
            value (torch.Tensor): The observed count values for which the log probability is calculated.

        Returns:
            torch.Tensor: The log probability of the observed counts.

        Raises:
            UserWarning: If the provided values are outside the support of the distribution.
        """
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within \
                        the support of the distribution",
                    UserWarning,
                    stacklevel=2
                )
        
        return log_nb_positive(
            value, mu=self.mu, theta=self.theta, eps=self._eps
        )

    def _gamma(self):
        """
        Creates a Gamma distribution object based on the distribution's parameters.

        Returns:
            torch.distributions.Gamma: A Gamma distribution object parameterized by the
            `theta` as the concentration and `theta/mu` as the rate.
        """
        return _gamma(self.theta, self.mu)


class ZeroInflatedNegativeBinomial(NegativeBinomial):
    r"""Zero-inflated negative binomial distribution.

    One of the following parameterizations must be provided:

    (1), (`total_count`, `probs`) where `total_count` is the number of
    failures until the experiment is stopped and `probs` the success
    probability. (2), (`mu`, `theta`) parameterization, which is the one
    used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.

    In the (`mu`, `theta`) parameterization, samples from the negative
    binomial are generated as follows:

    .. math::

        w \sim \textrm{Gamma}\left(\theta, \frac{\theta}{\mu}\right)

        x \sim \textrm{Poisson}(w)
        
    where :math:`\theta` is the shape parameter and :math:`\frac{\theta}{\mu}` is the rate parameter of the Gamma distribution.


    :param total_count: Number of failures until the experiment is stopped.
    :type total_count: torch.Tensor, optional
    :param probs: The success probability.
    :type probs: torch.Tensor, optional
    :param mu: Mean of the distribution.
    :type mu: torch.Tensor, optional
    :param theta: Inverse dispersion.
    :type theta: torch.Tensor, optional
    :param zi_logits: Logits scale of zero inflation probability.
    :type zi_logits: torch.Tensor, optional
    :param scale: Normalized mean expression of the distribution.
    :type scale: torch.Tensor, optional
    :param validate_args: Raise ValueError if arguments do not match constraints.
    :type validate_args: bool, optional
        
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "zi_logits": constraints.real,
        "scale": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        zi_logits: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        validate_args: bool = False,
    ):
        super(ZeroInflatedNegativeBinomial, self).__init__(
            total_count=total_count,
            probs=probs,
            logits=logits,
            mu=mu,
            theta=theta,
            scale=scale,
            validate_args=validate_args,
        )
        self.zi_logits, self.mu, self.theta = broadcast_all(
            zi_logits, self.mu, self.theta
        )

    def mean(self):
        """
        Calculate the mean of the Zero-Inflated Negative Binomial distribution.

        Returns:
            torch.Tensor: The expected mean of the distribution.
        """
        pi = self.zi_probs
        return (1 - pi) * self.mu

    def variance(self):
        """
        Calculate the variance of the Zero-Inflated Negative Binomial distribution.

        Raises:
            NotImplementedError: Indicates that the method is not implemented.
        """
        raise NotImplementedError

    # zi_logits is defined as an attribute and a function;
    # renamed to "zi_logits_fun"
    # due to name overlap, it is unclear when function is to be used
    def zi_logits(self) -> torch.Tensor:
        """ZI logits."""
        return probs_to_logits(self.zi_probs, is_binary=True)

    def zi_probs(self) -> torch.Tensor:
        
        # changed to zi_logits_fun function
        return logits_to_probs(self.zi_logits, is_binary=True)

    def sample(
        self,
        sample_shape: Optional[Union[torch.Size, Tuple]] = None,
    ) -> torch.Tensor:
        """
        Generate samples from the Zero-Inflated Negative Binomial distribution.

        Args:
            sample_shape (Union[torch.Size, Tuple], optional): The shape of the tensor to generate.
                Defaults to an empty tuple, which generates a single sample.

        Returns:
            torch.Tensor: Samples from the Zero-Inflated Negative Binomial distribution.
        """
        sample_shape = sample_shape or torch.Size()
        samp = super().sample(sample_shape=sample_shape)
        is_zero = torch.rand_like(samp) <= self.zi_probs
        samp_ = torch.where(is_zero, torch.zeros_like(samp), samp)
        return samp_

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log probability of a given value under the Zero-Inflated Negative Binomial distribution.

        Args:
            value (torch.Tensor): The observed count values for which the log probability is calculated.

        Returns:
            torch.Tensor: The log probability of the observed counts.

        Raises:
            UserWarning: If the provided values are outside the support of the distribution.
        """
        try:
            self._validate_sample(value)
        except ValueError:
            warnings.warn(
                "The value argument must be within \
                    the support of the distribution",
                UserWarning,
                stacklevel=2,
            )
        return log_zinb_positive(
            value, self.mu, self.theta, self.zi_logits, eps=1e-16
        )

    # added to test functions, to see if they'd work
    # def __repr__(self):
    #    return f'ZINB mean:{self.mean()}'


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
):
    """Log likelihood (scalar) of a minibatch according to a nb model.

    :param x: The input data tensor.
    :type x: torch.Tensor
    :param mu: The mean parameter of the negative binomial distribution (must be positive).
               Shape: (minibatch, vars)
    :type mu: torch.Tensor  
    :param theta: The inverse dispersion parameter of the negative binomial distribution (must be positive).
                  Shape: (minibatch, vars)
    :type theta: torch.Tensor
    :param eps: A small constant for numerical stability (default: 1e-8).
    :type eps: float, optional
    :param log_fn: The logarithm function to use (default: torch.log).
    :type log_fn: callable, optional
    :param lgamma_fn: The log-gamma function to use (default: torch.lgamma).
    :type lgamma_fn: callable, optional

    :return: The log likelihood of the input data under the specified negative binomial distribution.
    :rtype: torch.Tensor
    
    Returns
    -------
    torch.Tensor
        The log likelihood of the input data under the specified negative binomial distribution.
        
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta + eps)
        - lgamma(theta + eps)
        - lgamma(x + 1)
    )


    return res

# BSD 3-Clause License
# Copyright (c) 2023, Adam Gayoso, Romain Lopez
# Martin Kim, Pierre Boyeau, Nir Yosef


def log_zinb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    eps=1e-8
):
    """Log likelihood (scalar) of a minibatch according to a zinb model.

    :param x: The observed count data.
    :type x: torch.Tensor
    :param mu: The mean parameter of the negative binomial component (must be positive).
               Shape: (minibatch, vars)
    :type mu: torch.Tensor
    :param theta: The inverse dispersion parameter of the negative binomial component (must be positive).
                  Shape: (minibatch, vars)
    :type theta: torch.Tensor
    :param pi: The logit of the zero-inflation probability parameter (has real support).
               Shape: (minibatch, vars)  
    :type pi: torch.Tensor
    :param eps: A small constant for numerical stability (default: 1e-8).
    :type eps: float

    :return: The log-likelihood of the input data under the ZINB model.
    :rtype: torch.Tensor


    .. note::
        - We parametrize the Bernoulli component using logits, hence the appearance of the softplus function.
    """

    # theta is the dispersion rate.
    # If .ndimension() == 1, it is shared for all cells
    # (regardless of batch or labels)
    if theta.ndimension() == 1:
        print("THIS HAPPENS") # this never happens
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    # Uses log(sigmoid(x)) = -softplus(-x)
    softplus_pi = F.softplus(-pi)

    #print("pi", torch.min(pi).item(), torch.max(pi).item())
    #print("softplus_pi", torch.min(softplus_pi).item(), torch.max(softplus_pi).item())


    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    # formula from scVI: to confirm formula is properly working
    # og_case_zero = F.softplus(- pi + theta * torch.log(theta + eps) \
    #                            - theta * torch.log(theta + mu + eps)) \
    #                            - F.softplus( - pi)

    check_for_nans(mu, "nans in mu")
    check_for_nans(theta, "nans in theta")
    check_for_nans(log_theta_mu_eps, "nans in log_theta_mu_eps)")
    check_for_nans(pi_theta_log, "nans in pi_theta_log)")

    case_zero = F.softplus(pi_theta_log) - softplus_pi

    #print("softplus_pi_theta_log", torch.min(F.softplus(pi_theta_log)).item(), torch.max(F.softplus(pi_theta_log)).item())
    

    check_for_nans(case_zero, "nans in case zero")

    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)
    #print("case_zero", torch.min(mul_case_zero).item(), torch.max(mul_case_zero).item(), torch.min(x).item(), torch.max(x).item())

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )

    gamma_contribution = torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1)
    #print("non-zero_breakdown", softplus_pi*-1, pi_theta_log, x * (torch.log(mu + eps) - log_theta_mu_eps), gamma_contribution)



    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    #print("mul_case_non_zero", torch.min(mul_case_non_zero).item(), torch.max(mul_case_non_zero).item())

    check_for_nans(mul_case_zero, "nans in case zero 2")
    check_for_nans(mul_case_non_zero, "nans in non-zero")
    return res
