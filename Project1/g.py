
from scipy.stats import multivariate_normal
from parameters import *
from c import importance_monte_carlo


def importance_sample(N, theta):
    return np.random.multivariate_normal(
        mean=theta,
        cov=sigma_s,
        size=N
    ).T

def likelihood_ratio(importance_sample, theta):
    x = importance_sample.T

    p = multivariate_normal.pdf(
        x, mean=np.zeros(m), cov=sigma_s
    )

    q = multivariate_normal.pdf(
        x, mean=theta, cov=sigma_s
    )

    return p / q


np.random.seed(0)
N = 10**4
results = {}
theta_values = np.linspace(0, 2, 20)

thetas = np.array([
    [t1, t2]
    for t1 in theta_values
    for t2 in theta_values
])
for i, theta in enumerate(thetas):
    imp_sample = importance_sample(N, theta)
    estimate, s2 = importance_monte_carlo(N,
                                        importance_sample = imp_sample,
                                        likelihood_ratio_func=likelihood_ratio,
                                        likelihood_ratio_args={"importance_sample":imp_sample,
                                                               "theta": theta})
    results[i] = s2
    print(i, estimate)
theta_1 , theta_2 = thetas[:, 0], thetas[:, 1]

import matplotlib.pyplot as plt
# Plot of estimated variance as a function of theta
plt.figure(figsize=(10, 5))
variances = results.values()
idx = np.arange(len(variances))
plt.plot(idx, variances, label = "Variance", color = "blue")
plt.xlabel("index of theta")
plt.ylabel("Variance of estimator")
plt.title("Variance of estimator as a function of theta")
plt.grid()
plt.show()
interesting_idx = [25, 26, 27, 28, 29, 30]
for idx in interesting_idx:
    print(thetas[idx,:])
