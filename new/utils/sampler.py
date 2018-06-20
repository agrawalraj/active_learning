from __future__ import division  # in case python2 is used

from graph_utils import *
import numpy as np


def sample_dags(g0, siginv, data, burn_in=100, thin_factor=20, iterations=1000):
    g_curr = g0.copy()
    cov_edges_curr = get_covered_edges(g_curr)
    sample_dags = []
    for t in range(iterations):
        # Randomly select a covered edge to flip
        xi, xj = np.random.shuffle(list(cov_edges_curr))[0]
        g_next = g_curr.copy()
        reverse_edge(g_next, xi, xj)
        cov_edges_next = update_covered_edges(g_next, xi, xj, cov_edges_curr)

        # Compute the acceptance probability
        p_curr_unnormalized = compute_log_posterior_unnormalized(g_curr, siginv, data)
        p_next_unnormalized = compute_log_posterior_unnormalized(g_next, siginv, data)
        accept_prob = len(cov_edges_curr) / len(cov_edges_next) \
                      * np.exp(p_next_unnormalized - p_curr_unnormalized)
        accept_prob = np.min([1, accept_prob])

        # Make an MCMC step
        accepted = np.random.binomial(1, accept_prob, 1)[0]
        if accepted == 1:
            g_curr = g_next.copy()  # should we copy this graph?
            cov_edges_curr = cov_edges_next.copy()
        if t > burn_in and t % thin_factor == 0:
            sample_dags.append(g_curr)

    return sample_dags


def sample_dags_uniform(g0):
    return sample_dags(g0, [])

