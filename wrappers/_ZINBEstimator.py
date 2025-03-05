import numpy as np

from scipy.stats import nbinom
from scipy.optimize import minimize
from scipy.stats import nbinom
from scipy.special import expit, logit

# Compute ZINB parameters from a numpy array of count data

class ZINBEstimator:
    r"""
    A class for estimating the parameters of a Zero-Inflated Negative Binomial (ZINB) distribution from count data.

    :param initial_pi_values: Initial values of pi to perform a grid search to find the optimal 'pi' parameter.
    :type initial_pi_values: list, optional
    """
    def __init__(self, initial_pi_values=[0.00001, 0.0001, 0.001, 0.005, 0.01, 0.1, 0.5, 0.9]):
        """
        Initialize the ZINBEstimator with optional initial values for pi.
        
        :param initial_pi_values: Initial values of pi to perform a grid search to find the optimal 'pi' parameter.
        :type initial_pi_values: list, optional
        """

        self.initial_pi_values = initial_pi_values

    def _zinb_loglike(self, params, counts):
        """
        Computes the negative log-likelihood for the ZINB distribution given parameters and count data.

        :param params: A list containing the parameters [mu, variance, logit_pi] where:
            - mu: The mean of the Negative Binomial component.
            - variance: The variance of the count data.
            - logit_pi: The logit-transformed zero-inflation parameter.
        :type params: list
        :param counts: A set of counts to estimate ZINB parameters from.
        :type counts: numpy.ndarray

        :return: The negative log-likelihood value.
        :rtype: float
        """
        mu, variance, logit_pi = params
        pi = expit(logit_pi)
        epsilon = 1e-10 # used to avoid division by zero
            
        # Compute n and p from mu/variance
        n = mu**2 / (variance - mu + epsilon)
        p = n / (n + mu + epsilon)
            
        # Check for invalid values
        if np.any(np.isnan([n, p])) or np.any(np.isinf([n, p])):
            return np.inf  # Return a high cost for invalid values
            
        loglik_nb = nbinom.logpmf(counts, n, p)
        loglik_zero = np.log(pi + (1 - pi) * np.exp(nbinom.logpmf(0, n, p)))
        loglik = np.where(counts == 0, loglik_zero, np.log(1 - pi) + loglik_nb)
            
        return -np.sum(loglik)


    def compute_zinb_params(self, counts):
        """
        Estimates the Zero-Inflated Negative Binomial (ZINB) parameters and AIC from a given set of count data.

        :param counts: A set of counts to estimate ZINB parameters from.
        :type counts: numpy.ndarray

        :return: A tuple containing the estimated parameters (mu, theta, pi, aic) where:
            
            - mu: The mean of the Negative Binomial component.
            
            - theta: The dispersion parameter of the Negative Binomial component.
            
            - pi: The zero-inflation parameter.
            
            - aic: The Akaike Information Criterion value for the fit.
        :rtype: tuple

        Usage:
            zinb_estimator = ZINBEstimator()
            mu, theta, pi, aic = zinb_estimator.compute_zinb_params(counts)
        """
        optimal_params = None
        best_aic = np.inf

        for initial_pi in self.initial_pi_values:
            initial_params = np.array([np.mean(counts), np.var(counts), logit(initial_pi)])
            bounds = [(0, None), (0, None), (None, None)]
            result = minimize(self._zinb_loglike, initial_params, args=(counts), bounds=bounds, tol=1e-6)
            
            if result.success:
                mu, variance, logit_pi = result.x
                pi = expit(logit_pi)
                theta = (mu**2) / (variance - mu)
                loglik = -result.fun
                k = 3  # Number of parameters
                aic = 2 * k - 2 * loglik

                if aic < best_aic:
                    best_aic = aic
                    optimal_params = (mu, theta, pi, aic)
        
        return optimal_params

        
        
