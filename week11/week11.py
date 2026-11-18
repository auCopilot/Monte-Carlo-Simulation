# Week 11 - Monte Carlo Simulation
import numpy as np


def metropolis_hastings_sampler(proposal_sampler, proposal_density, target_density, x0, n_samples):
    """
    Metropolis-Hastings sampler for generating samples from a target distribution.

    :param proposal_sampler: Function that generates proposals given the current state
    :param target_density: Function that computes the target density at a given state
    :param x0: Initial state
    :param n_samples: Number of samples to generate
    :return: Array of generated samples
    """
    # Initialise
    x0 = np.asarray(x0, dtype=float)
    x_current = x0.item() if x0.ndim == 0 else x0.copy()

    samples = np.empty((n_samples,) + np.shape(x_current))

    current = target_density(x_current)

    accept_count = 0
    # For each iteration we:
    for i in range (n_samples):
        # Sample a new propsal given the current sample
        x_proposal = proposal_sampler(x_current)

        proposed = target_density(x_proposal)
        # Compute the alpha
        alpha = min(1, proposed / current
                    * proposal_density(x_current,x_proposal) / proposal_density(x_proposal, x_current))

        # Reject or accept based on the alpha and a uniform sample and update the current value
        if np.random.uniform(0,1) <= alpha:

            x_current = x_proposal
            # Track acceptance rate
            accept_count += 1

            # Update current density
            current = proposed

        # Store current state
        samples[i] = x_current


    acceptance_rate = accept_count / n_samples

    return samples, acceptance_rate


def markov_sample(current_state, p_mat):
    # p is an array of transitioning from state i to j
   return np.random.choice(np.arange(1, len(p_mat) + 1), p = p_mat[current_state - 1])