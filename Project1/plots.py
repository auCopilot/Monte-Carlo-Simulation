"""3D plotting helpers for importance-sampling results.
DISCLAIMER: VISUALIZATION CODE MADE BY AI.
Expected results format:
    results[(theta1, theta2)] = (estimate, variance)

Example from g.py:
    from plots import three_d_estimate, three_d_variance

    three_d_estimate(theta_values, results)
    three_d_variance(theta_values, results)

The aliases `3destimate` and `3dvariance` are not valid Python identifiers
because an identifier cannot start with a digit. Use `three_d_estimate`
and `three_d_variance` instead.
"""

import numpy as np
import matplotlib.pyplot as plt


def _make_grid(theta_values, results, value_index):
    """Build a grid from results, preserving the order in theta_values."""
    theta_values = list(theta_values)
    n = len(theta_values)

    grid = np.empty((n, n), dtype=float)
    for i, theta1 in enumerate(theta_values):
        for j, theta2 in enumerate(theta_values):
            grid[i, j] = results[(theta1, theta2)][value_index]

    return theta_values, grid


def three_d_estimate(theta_values, results, *, ax=None, show=True):
    """Plot the estimated probability (tuple element 0) over the theta grid."""
    theta_values, estimate_grid = _make_grid(theta_values, results, 0)
    theta1_grid, theta2_grid = np.meshgrid(theta_values, theta_values, indexing="ij")

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    surface = ax.plot_surface(
        theta1_grid, theta2_grid, estimate_grid,
        cmap="viridis", edgecolor="none"
    )
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_zlabel(r"Estimated probability $\hat{p}$")
    ax.set_title("Importance Sampling: Estimated Probability")
    fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.1, label=r"$\hat{p}$")
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


def three_d_variance(theta_values, results, *, ax=None, show=True):
    """Plot the estimated variance (tuple element 1) over the theta grid."""
    theta_values, variance_grid = _make_grid(theta_values, results, 1)
    theta1_grid, theta2_grid = np.meshgrid(theta_values, theta_values, indexing="ij")

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    surface = ax.plot_surface(
        theta1_grid, theta2_grid, variance_grid,
        cmap="viridis", edgecolor="none"
    )
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_zlabel("Estimated variance")
    ax.set_title("Importance Sampling: Estimated Variance")
    fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.1, label="Variance")
    fig.tight_layout()

    if show:
        plt.show()
    return fig, ax


# Short aliases that are valid Python identifiers.
plot_3d_estimate = three_d_estimate
plot_3d_variance = three_d_variance
