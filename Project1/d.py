import numpy as np
import matplotlib.pyplot as plt
from a import crude_monte_carlo
from b import control_monte_carlo
from c import importance_monte_carlo
from parameters import *

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
            l, s2 = crude_monte_carlo(Nd)
        if m == "Control Variable":
            l, s2 = control_monte_carlo(Nd)
        if m == "Importance Sampling":
            l, s2 = importance_monte_carlo(Nd)

        s = np.sqrt(s2)
        # Compute confidence interval
        ci = confidence_interval(s, l, Nd)
        ci_lower[i, j] = ci[0]
        ci_upper[i, j] = ci[1]
        estimates[i, j] = l

# Mean interval lengths

for j,m in enumerate(methods):
    ci_lengths = ci_upper[:,j] - ci_lower[:, j]
    mean_ci_length = np.mean(ci_lengths)
    print("The mean CI length for method " + m + " is: ", mean_ci_length)

sns.set_theme(style="whitegrid", context="notebook")
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
from matplotlib.ticker import FuncFormatter
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda value, _: f"{value / 100:g}")
)


sns.despine()
fig.tight_layout()
# Save as SVG
plt.savefig("figures/ConfInts.pdf", format="pdf", bbox_inches="tight")
plt.show()