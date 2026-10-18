"""
Monte Carlo with importance sampling for variance reduction
"""
from a import indicator, L
from parameters import *
N = 10000
np.random.seed(1)
sampling_dist = np.random.normal(loc = 0, scale = 1, size = (1, N))

def likilihood_ratio(X, mu_p, sigma_p, mu_q, sigma_q):
    """
    We use another normal distribution with greater mean as importance sampling.
    Compute dP(X) / dQ(X), where:
    Under P: X_i ~ N(m_P, variance)
    Under Q: X_i ~ N(m_P + drift, variance)
    """
    # Compute the likelihood ratio
    p = (1 / np.sqrt(2 * np.pi * sigma_p**2)) * np.exp(-0.5 * ((X - mu_p) / sigma_p)**2)
    q = (1 / np.sqrt(2 * np.pi * sigma_q**2)) * np.exp(-0.5 * ((X - mu_q) / sigma_q)**2)
    return p / q


import numpy as np

def importance_monte_carlo(
    N,
    importance_sample,
    likelihood_ratio_func,
    likelihood_ratio_args=None,
    use_sigma_s = True,
):
    """
    Estimate probability of exceedance using importance sampling.

    Parameters
    ----------
    N : int
        Number of simulations.
    likelihood_ratio_func : callable
        Function that computes likelihood ratios.
    likelihood_ratio_args : dict, optional
        Keyword arguments passed to likelihood_ratio_func.
    importance_sample : ndarray, optional
        Samples drawn from the importance distribution.

    Returns
    estimate of exceedance probability.
    variance of estimator.
    """

    if likelihood_ratio_args is None:
        likelihood_ratio_args = {}

    # Compute likelihood ratios
    weights = likelihood_ratio_func(**likelihood_ratio_args)
    # Diagnosis for LR should be around 1: print("Mean of weights", np.mean(weights))
    # Simulate N outcomes of L
    loss = L(importance_sample, use_sigma_s = use_sigma_s)

    # Mean of indicator converges a.s to P(L > x), weighted by the likihood ratio
    exeedance = indicator(x, loss) * weights
    # Estimate
    exeedance_prop = np.mean(exeedance)
    # Variance of estimate
    variance = np.var(exeedance) / N
    return exeedance_prop, variance

if __name__ == '__main__':

    from a import crude_monte_carlo
    l_crude, s2_crude = crude_monte_carlo(N, sampling_dist)
    print("Importance sampling Monte Carlo simulation: \n")
    importance_sample = np.random.multivariate_normal(
        mu_q, sigma_s, size=N
    ).T
    lr_args = {"X" : importance_sample,
               "mu_p" : mu_p,
               "sigma_p" : sigma_p,
               "mu_q" : mu_q,
               "sigma_q" : sigma_q,}

    l_importance, s2_importance = importance_monte_carlo(N,
                                                         importance_sample,
                                                         likelihood_ratio_func =  likilihood_ratio,
                                                         likelihood_ratio_args=lr_args)

    print("Estimated probability of exceedance of x using importance sampling: ", l_importance)
    print("Variance of MC estimator using importance sampling: ", s2_importance)
    # Variance reduction factor
    print("The variance reduction factor of importance sampling: ", s2_crude / s2_importance)
