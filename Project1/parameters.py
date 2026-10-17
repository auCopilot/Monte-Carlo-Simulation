import numpy as np

### (a), (b) ###
# Set m = 1 for (a) and (b) and m = 2 for repetition of (a) and (b) with m = 2
m = 2

# delta, gamma is a column vector of size m
delta = np.ones((m, 1)) * 2
gamma = np.eye(m)

def indicator(x, L):
    # Exceedance of x
    return L > x




if m == 1:

    sigma_s = np.ones((m, m))

    x = 10

    def L(X, kw = None):
        """L for N simulations, keyword added for interchangeability of m = 1 or 2
        input:
        N (int): Number of simulations
        X: of shape (m, N) samples
        gamma: vector of size m
        delta: vector of size m
        """
        # Returns (N,) losses

        L = delta.T @ X + np.sum(X * (gamma @ X), axis=0)
        return L


if m == 2:

    sigma_s = np.array([[1, 0.5], [0.5, 1]])



    x = 15

    # Define the matrix
    C_T_C = np.array([[0.75, 0.5 * np.sqrt(0.75)],
                      [0.5 * np.sqrt(0.75), 1.25]])

    # Compute the eigenvalues and eigenvectors
    eigenvalues, U = np.linalg.eig(C_T_C)
    C_tilde = np.array([[1, 1 / 2],
                        [0, np.sqrt(3 / 4)]])
    C = C_tilde @ U
    b = np.real(C.T @ delta)


    def L(X, use_sigma_s = True):
        """L for N simulations
        input:
        N (int): Number of simulations
        X: of shape (m, N) samples
        gamma: vector of size m
        delta: vector of size m
        """
        if use_sigma_s:
            L = delta.T @ X + np.sum(X * (gamma @ X), axis=0)
            return L
        else:
            # Loss computed in accordance with (f)
            lam1 = np.real(eigenvalues[0])
            lam2 = np.real(eigenvalues[1])
            L = b[0] * X[0, :] + b[1] * X[1, :] + lam1 * np.pow(X[0, :], 2) + lam2 * np.pow(X[1 , :], 2)
            return L

### (c) ###

# Q measure paramters
mu_q = np.ones(m) * np.sqrt(10) - 1
sigma_q = sigma_s

# P measure parameters
mu_p = np.zeros(m)
sigma_p = sigma_s

######## (d) ########
import seaborn as sns
from scipy.stats import norm

Nd = 10000

SIMULATION_RUNS = 30
plot_n = np.arange(1, SIMULATION_RUNS + 1)

methods = ["Crude Monte Carlo", "Control Variable", "Importance Sampling"]
colors = sns.color_palette("colorblind", 3)

def confidence_interval(s, l , N, level = 0.05):
    """Compute confidence interval, equation (1.2)"""
    z_alpha = norm.ppf(1 - level)
    return (l - z_alpha*s / np.sqrt(N), l + z_alpha*s / np.sqrt(N) )
