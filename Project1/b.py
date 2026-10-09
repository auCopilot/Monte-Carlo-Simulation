"""
We introduce a control variable to reduce the varinace.
"""
# Same as in (a)
from a import L, indicator, crude_monte_carlo
from parameters import *

# Use Q as control variable
def Q(Delta_s,  delta):
    return delta.T @ Delta_s

def compute_optimal_coefficients(Y, Z):
    """
    Y: target, shape (1, N)
    Z: control variates, shape (k, N)
       where k is the number of control variates.

    Returns:
        beta_opt: shape (k,)
    """
    Y = np.asarray(Y).reshape(-1)
    Z = np.atleast_2d(Z)

    if Z.shape[1] != Y.size:
        raise ValueError("Y and Z must have the same number of simulations.")

    cov = np.cov(np.vstack([Y, Z]), ddof=1)

    cov_YZ = cov[0, 1:]
    cov_ZZ = cov[1:, 1:]

    beta_opt = np.linalg.solve(cov_ZZ, cov_YZ)
    return beta_opt

def control_monte_carlo(N, Delta_s = None):
    """
    Control variate Monte Carlo simulation to estimate the probability of exceedance of x.

    Parameters:
    N (int): Number of simulations

    Returns:
    float: Estimated probability of exceedance
    """

    if Delta_s is None:
        Delta_s = np.random.multivariate_normal(np.zeros(m), sigma_s, size = N).T

    # Compute control variable
    Z = Q(Delta_s, delta)
    # Expectation of control variable
    E_Z = 0

    # Simulate N outcomes of L
    loss = L(Delta_s, delta, gamma)

    # Mean of indicator converges a.s to P(L > x)
    exeedence = indicator(x, loss)


    # optimal coeffcient
    beta_opt = compute_optimal_coefficients(exeedence, Z)

    # Control estimator
    exeedence_tilde = exeedence -  beta_opt*(Z - E_Z)

    # Estimate
    exeedance_prop = np.mean(exeedence_tilde)

    # Variance of estimate
    variance = np.var(exeedence_tilde) / N

    return exeedance_prop, variance

if __name__ == '__main__':
    print("Control variate Monte Carlo simulation: \n")
    N = 10000
    Delta_s = np.random.multivariate_normal(np.zeros(m), sigma_s, size=N).T
    # Estimate and variance of estimator
    l_crude, s2_crude = crude_monte_carlo(N, Delta_s)
    l_control, s2_control = control_monte_carlo(N, Delta_s)
    print("Estimated probability of exceedance of x using control variable: ", l_control)
    print("Variance of MC estimator using control: ", s2_control)
    print()
    print("Variance reduction factor: ", s2_crude / s2_control)