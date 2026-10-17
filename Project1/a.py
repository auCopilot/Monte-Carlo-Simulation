"""
Code for task (a) - Crude monte carlo

"""
from parameters import *
np.random.seed(0)
N = 100000
if m == 1:
    sampling_dist = np.random.normal(loc = 0, scale = 1, size = (1,N))
if m == 2:
    sampling_dist = np.random.multivariate_normal(np.zeros(2), sigma_s, size = N).T


def crude_monte_carlo(N, sampling_dist):
    """
    Crude Monte Carlo simulation to estimate the probability of exceedance of x.

    Parameters:
    N (int): Number of simulations

    Returns:
    float: Estimated probability of exceedance
    """


    # Simulate N outcomes of L
    loss = L(sampling_dist)

    # Mean of indicator converges a.s to P(L > x)
    exeedance = indicator(x, loss)
    # Estimate
    exeedance_prop = np.mean(exeedance)
    # Variance of estimate
    variance = np.var(exeedance) / N
    return exeedance_prop, variance

if __name__ == "__main__":
    print("Crude Monte Carlo simulation: \n")
    # Estimate and variance of estimator
    l, s2 = crude_monte_carlo(N, sampling_dist)
    print("Estimated probability of exceedance of x: ", l)
    print("Variance of MC estimator: ", s2)
