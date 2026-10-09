"""
Monte Carlo with importance sampling for variance reduction
"""
from a import indicator, L
from parameters import *

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


def importance_monte_carlo(N):
    """
    Importance Monte Carlo simulation to estimate the probability of exceedance of x.

    Key features are that we sample from another distribution that makes the probablily of
    exceedance more likely, and then we normalize the samples by the likelihood ratio.

    Parameters:
    N (int): Number of simulations

    Returns:
    float: Estimated probability of exceedance
    """
    # Draw samples - this time from the importance distribution (expandable for m > 1)
    importance_sample = np.random.multivariate_normal(mu_q, sigma_s, size = N).T

    # Compute likihood ratio for each sample


    weights = likilihood_ratio(importance_sample, mu_p, sigma_p, mu_q, sigma_q)

    # Simulate N outcomes of L
    loss = L(importance_sample, delta, gamma)

    # Mean of indicator converges a.s to P(L > x), weighted by the likihood ratio
    exeedance = indicator(x, loss) * weights
    # Estimate
    exeedance_prop = np.mean(exeedance)
    # Variance of estimate
    variance = np.var(exeedance) / N
    return exeedance_prop, variance

if __name__ == '__main__':
    N = 10000

    from a import crude_monte_carlo
    l_crude, s2_crude = crude_monte_carlo(N)
    print("Importance sampling Monte Carlo simulation: \n")
    l_importance, s2_importance = importance_monte_carlo(N)
    print("Estimated probability of exceedance of x using importance sampling: ", l_importance)
    print("Variance of MC estimator using importance sampling: ", s2_importance)
    # Variance reduction factor
    print("The variance reduction factor of importance sampling: ", s2_crude / s2_importance)
