"""
Code for task (a) - Crude monte carlo

"""

from parameters import *

def indicator(x, L):
    # Exceedance of x
    return L > x

def L(Delta_s, delta, gamma):
    """L for N simulations

    input:
    N (int): Number of simulations
    Delta_S: N(0, Sigma_S) of shape (m, N) samples
    gamma: vector of size m
    delta: vector of size m
    """
    # Returns (N,) losses
    L = delta.T @ Delta_s + np.sum(Delta_s * (gamma @ Delta_s), axis=0)
    return L

def crude_monte_carlo(N, Delta_s= None):
    """
    Crude Monte Carlo simulation to estimate the probability of exceedance of x.

    Parameters:
    N (int): Number of simulations

    Returns:
    float: Estimated probability of exceedance
    """

    if Delta_s is None:
        Delta_s = np.random.multivariate_normal(np.zeros(m), sigma_s, size = N).T

    # Simulate N outcomes of L
    loss = L(Delta_s, delta, gamma)

    # Mean of indicator converges a.s to P(L > x)
    exeedance = indicator(x, loss)
    # Estimate
    exeedance_prop = np.mean(exeedance)
    # Variance of estimate
    variance = np.var(exeedance) / N
    return exeedance_prop, variance

if __name__ == "__main__":
    print("Crude Monte Carlo simulation: \n")
    np.random.seed(0)
    N = 100000
    # Estimate and variance of estimator
    l, s2 = crude_monte_carlo(N)
    print("Estimated probability of exceedance of x: ", l)
    print("Variance of MC estimator: ", s2)
