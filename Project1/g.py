import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from parameters import *
from c import importance_monte_carlo

if m == 1:
    raise ValueError("m must be 2 for (g).")

N = 10000
np.random.seed(0)

def importance_sample(N, theta):
    return np.random.multivariate_normal(
        mean=theta,
        cov=np.eye(m),
        size=N
    ).T


def likelihood_ratio(importance_sample, theta):
    """
    We use another normal distribution with shifted mean as importance sampling.
    Compute dP(X) / dQ(X), where:
    Under P: X_i ~ N(0, I2)
    Under Q: X_i ~ N(theta, I2)
    """
    x = importance_sample
    theta = theta[:, None]  # shape (2, 1)
    # Follows from (f):
    lr = np.exp(-theta * x + 0.5 * theta**2)
    lr = np.prod(lr, axis=0)
    return lr


np.random.seed(0)
N = 10**5
results = {}
theta_values = np.linspace(-1, 1/3, 50)

# Generate grid of theta pairs
thetas = np.array([
    [t1, t2]
    for t1 in theta_values
    for t2 in theta_values
])

# Compute estimates
for theta in thetas:
    imp_sample = importance_sample(N, theta)
    estimate, s2 = importance_monte_carlo(
        N,
        importance_sample=imp_sample,
        likelihood_ratio_func=likelihood_ratio,
        likelihood_ratio_args={
            "importance_sample": imp_sample,
            "theta": theta
        },
        use_sigma_s=False
    )

    results[tuple(theta)] = (estimate, s2)


# --- 3D surface plot of estimated variance ---
from plots import *
fig, ax = three_d_estimate(theta_values, results)
fig.savefig("figures/2Dest.pdf", bbox_inches="tight")
plt.close(fig)

fig, ax = three_d_variance(theta_values, results)
fig.savefig("figures/2Dvar.pdf", bbox_inches="tight")
plt.close(fig)