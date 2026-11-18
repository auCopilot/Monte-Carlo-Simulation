# Week 9 - Monte Carlo Simulation
import numpy as np
def pertubation_analysis_derivative(X, dH):
    """
    input: Iterations N ⩾ 1; function H(x; θ) satisfying ∇ℓ(θ) = E[∇θH(X; θ)]
    output: Estimate ∇ℓ(θ)
    """
    delta_H = dH(X)

    S = np.std(delta_H, axis = 0 if X.ndim == 1 else 1)

    estimate = np.mean(delta_H, axis = 0 if X.ndim == 1 else 1)

    return estimate, S


def score_function_derivative(X, h_func, score_func):
    """
    Algorithm 8.3.1: Score function
    input: Number of simulations N ⩾ 1; functions H(x), f(x; θ) satisfying that
    we can interchange differentiation and integration.

    output: Estimate ∇ℓ(θ) for E[H(X)

    """


    eval = h_func(X) * score_func(X)

    S = np.std(eval, axis = 0 if X.ndim == 1 else 1)
    estimate = np.mean(eval, axis = 0 if X.ndim == 1 else 1)

    return estimate, S
