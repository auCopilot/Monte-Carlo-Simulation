import numpy as np
import matplotlib.pyplot as plt
from a import crude_monte_carlo
from b import control_monte_carlo
from c import importance_monte_carlo, likelihood_ratio
from parameters import *
sns.set_theme(style="whitegrid", context="notebook")
from matplotlib.ticker import FuncFormatter

np.random.seed(0)
if m == 2:
    raise ValueError("m must be 1 for (d).")

def confidence_interval(s, l, N, level=0.05):
    """Compute confidence interval, equation (1.2)"""
    z_alpha = norm.ppf(1 - level)
    return (l - z_alpha * s / np.sqrt(N), l + z_alpha * s / np.sqrt(N))

# Column is method, row is simulation run
estimates = np.zeros((SIMULATION_RUNS, len(methods)))
ci_lower = np.zeros_like(estimates)
ci_upper = np.zeros_like(estimates)

# Define seeds for reproducibility
random_seeds = range(0, SIMULATION_RUNS*len(methods))

for j, m in enumerate(methods):
    for i in range(SIMULATION_RUNS):
        np.random.seed(random_seeds[i])
        """
        Simulate the three methods and store the estimates and confidence intervals.
        """
        if m == "Crude Monte Carlo":
            # Simulate
            sampling_dist = np.random.normal(loc=0, scale=1, size=(1, Nd))
            l, s2 = crude_monte_carlo(Nd, sampling_dist)
        if m == "Control Variable":

            sampling_dist = np.random.normal(loc=0, scale=1, size=(1, Nd))
            l, s2 = control_monte_carlo(Nd, sampling_dist)
        if m == "Importance Sampling":

            importance_sample = np.random.multivariate_normal(
                mu_q, sigma_s, size=Nd
            ).T
            lr_args = {"X": importance_sample,
                       "mu_p": mu_p,
                       "sigma_p": sigma_p,
                       "mu_q": mu_q,
                       "sigma_q": sigma_q, }
            l, s2 = importance_monte_carlo(Nd,
                                            importance_sample,
                                            likelihood_ratio_func =  likelihood_ratio,
                                            likelihood_ratio_args=lr_args)

        s = np.sqrt(s2)
        # Compute confidence interval
        ci = confidence_interval(s, l, Nd)
        ci_lower[i, j] = ci[0]
        ci_upper[i, j] = ci[1]
        estimates[i, j] = l

fig, ax = plt.subplots(figsize=(10, 5.5))

# Horizontal offsets to separate the three methods
offsets = [0,0,0]

for j, method in enumerate(methods):
    x = plot_n + offsets[j]

    # Asymmetric error lengths: estimate - lower, upper - estimate
    yerr = np.vstack([
        estimates[:, j] - ci_lower[:, j],
        ci_upper[:, j] - estimates[:, j],
    ])*100

    ax.errorbar(
        x,
        estimates[:, j],
        yerr=yerr,
        fmt="o",
        color=colors[j],
        label=method,
        capsize=4,
        elinewidth=1.8,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )

ax.set_xticks(plot_n)
ax.set_xlabel("Simulation Run")
ax.set_ylabel("Estimated Probability of Exceedance")
ax.set_title(f"Estimates and 95% confidence intervals, N = {Nd}\n", pad=14)
ax.legend(title="Estimation method", frameon=False)
ax.grid(axis="y", alpha=0.3)
ax.grid(axis="x", visible=False)

ax.yaxis.set_major_formatter(
    FuncFormatter(lambda value, _: f"{value / 100:g}")
)

sns.despine()
fig.tight_layout()
# Save as SVG
plt.savefig("figures/ConfInts.pdf", format="pdf", bbox_inches="tight")
plt.show()