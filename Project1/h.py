from scipy.optimize import fsolve
from c import importance_monte_carlo

from parameters import *
if m != 2:
    raise ValueError("m must be 2 for (h).")
N = 10**4
np.random.seed(1)

# Step 1. Solve for theta
def psi_prime_L(theta):
    """Solve for theta in the first derivative of the cumulant generating function of L
    such that the shifted distribution will have mean x"""
    lam = eigenvalues
    return (np.sum(theta*b**2*(1-theta*lam) /
            (1-2*theta*lam)**2 +
            lam /
            (1-2*theta*lam))) - 15

theta_star = fsolve(psi_prime_L, 0.1)[0]

# Step 2. Compute sampling distribution
def sampling_distribution(theta, N):
    """Sampling distribution for importance sampling"""

    A_inv = np.diag(1 / (1 - 2 * theta * eigenvalues))

    mu = (theta*A_inv @ b).flatten()

    return np.random.multivariate_normal(mean=mu, cov=A_inv, size=N).T


# Step 3. Define functions to compute maximum likilihood ratio

def psi_L(theta):
    """Cumulant Generating Function of L - use to compute likelihood ratio for importance sampling"""
    lam = eigenvalues
    return 0.5 * np.sum(theta ** 2 * b ** 2 /
                        (1 - 2 * theta * lam) -
                        np.log(1 - 2 * theta * lam)
                        )

def likelihood_ratio(theta, X):
    """Likelihood ratio for importance sampling"""

    return np.exp(-theta*L(X, use_sigma_s= False) + psi_L(theta))

# Step 4. compute us
if __name__ == '__main__':
    print("Theta_star for importance sampling: ", theta_star)
    print("Theta_star < 1/3: ", theta_star < 1/3)
    print("Importance sampling Monte Carlo simulation for (h): \n")
    importance_sample = sampling_distribution(theta_star, N)
    lr_args = {"X" : importance_sample,
               "theta" : theta_star
               }

    l_importance, s2_importance = importance_monte_carlo(N,
                                                         importance_sample,
                                                         likelihood_ratio_func =  likelihood_ratio,
                                                         likelihood_ratio_args=lr_args,
                                                         use_sigma_s= False)

    print("Estimated probability of exceedance of x using importance sampling: ", l_importance)
    print("Variance of MC estimator using importance sampling: ", s2_importance)
#
#
# # Range of theta values
# theta_vals = np.linspace(-2, 0.25, 1000)
#
# # Evaluate the function
# psi_vals = np.array([psi_L(theta) for theta in theta_vals])
#
# # Plot
# plt.plot(theta_vals, psi_vals, label=r'$\psi_L(\theta)$')
# plt.axhline(0, color='black', linestyle='--')
# plt.xlabel(r'$\theta$')
# plt.ylabel(r'$\psi_L(\theta)$')
# plt.title(r'Plot of $\psi_L(\theta)$')
# plt.grid(True)
# plt.legend()
# plt.show()
